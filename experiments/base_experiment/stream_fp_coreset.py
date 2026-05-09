"""
StreamFP: Learnable Fingerprint-guided Data Selection for streaming coresets.

Reference: Algorithm 1 (Coreset Selection) and Algorithm 2 (Buffer Update).

Adaptation note
---------------
The original StreamFP paper assumes a model that emits (b, L, D) token sequences
(e.g. a ViT). Here the input is already a ResNet-18 embedding of shape (b, d).
``IdentityEmbedder`` wraps the flat embedding as a 1-token sequence (b, 1, d)
so the cosine-similarity machinery in ``_compute_similarities`` is unchanged.
"""

from __future__ import annotations

import os
import sys
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from streamers.abstract_streamer import AbstractStreamingCoreset


# ---------------------------------------------------------------------------
# Minimal feature extractor for pre-embedded data
# ---------------------------------------------------------------------------
class IdentityEmbedder(nn.Module):
    """Wraps a flat embedding vector (b, d) into a 1-token sequence (b, 1, d)."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(1)


class StreamFPCoreset(AbstractStreamingCoreset):
    """
    StreamFP: Learnable Fingerprint-guided Data Selection.
    Includes dynamic fingerprint updating (EMA) to simulate attunement.
    """
    def __init__(
        self,
        buffer_size: int,
        coreset_ratio: float,
        feature_extractor: nn.Module,
        fingerprints: torch.Tensor,
        device: str = "cpu",
    ):
        self.M = buffer_size
        self.sigma = coreset_ratio
        self.feature_extractor = feature_extractor.to(device)
        self.fingerprints = fingerprints.to(device)
        self.device = device

        self.buffer_X: List[np.ndarray] = []
        self.buffer_y: List[int] = []
        self.buffer_prov: List[Tuple[int, int]] = []

        self.n_seen: int = 0
        self.mmd_history: List[float] = []
        
        self.current_coreset_X: np.ndarray | None = None
        self.current_coreset_y: np.ndarray | None = None

    @property
    def buffer_weights(self) -> np.ndarray:
        n = len(self.buffer_X)
        if n == 0:
            return np.empty(0, dtype=np.float64)
        return np.ones(n, dtype=np.float64) / float(n)

    def _compute_similarities(self, X_np: np.ndarray) -> np.ndarray:
        self.feature_extractor.eval()
        with torch.no_grad():
            X_t = torch.tensor(X_np, dtype=torch.float32, device=self.device)
            emd = self.feature_extractor(X_t)                     # (b, 1, D)

            P_sum = torch.sum(self.fingerprints, dim=1)           # (N, D)
            P_norm = F.normalize(P_sum, p=2, dim=1)               # (N, D)
            emd_norm = F.normalize(emd, p=2, dim=2)               # (b, 1, D)

            S_prime = torch.matmul(emd_norm, P_norm.transpose(0, 1)) # (b, 1, N)
            S = torch.mean(S_prime, dim=(1, 2))                   # (b,)
        return S.cpu().numpy()

    @staticmethod
    def _rank_probabilities(S_arr: np.ndarray, invert: bool = False) -> np.ndarray:
        n = len(S_arr)
        if n == 0:
            return np.empty(0, dtype=np.float64)
        if n == 1:
            return np.ones(1, dtype=np.float64)

        ranks = np.empty(n, dtype=np.float64)
        ranks[np.argsort(S_arr)[::-1]] = np.arange(1, n + 1)
        rho = 1.0 / ranks
        rho_sum = rho.sum()
        
        if invert:
            pi = rho / rho_sum
        else:
            pi = 1.0 - rho / rho_sum
            s = pi.sum()
            pi = pi / s if s > 1e-12 else np.ones(n, dtype=np.float64) / n
        return pi

    def process_batch(self, X_batch_np: np.ndarray, y_batch_np: np.ndarray, batch_idx: int) -> None:
        b = X_batch_np.shape[0]
        if b == 0:
            return

        # ---- Algorithm 1: Coreset Selection --------------------------------
        S_batch = self._compute_similarities(X_batch_np)
        I_sorted = np.argsort(S_batch)[::-1]

        c = max(1, int(self.sigma * b))
        mid = b // 2
        start = max(0, mid - c // 2)
        end = min(b, start + c)
        C_indices = I_sorted[start:end]

        self.current_coreset_X = X_batch_np[C_indices]
        self.current_coreset_y = y_batch_np[C_indices]

        # ---- Algorithm 2: Buffer Update ------------------------------------
        prov_batch = [(batch_idx, i) for i in range(b)]

        if len(self.buffer_X) < self.M:
            slots = self.M - len(self.buffer_X)
            fill_idx = I_sorted[:slots]
            for i in fill_idx:
                self.buffer_X.append(X_batch_np[i].copy())
                self.buffer_y.append(int(y_batch_np[i]))
                self.buffer_prov.append(prov_batch[i])
            self.n_seen += b
            self.mmd_history.append(0.0)
            
            # Attunement for initial fill
            self._update_fingerprints(X_batch_np[fill_idx], y_batch_np[fill_idx])
            return

        # ---- FIX 1: Allow updates even when b=1 ---------------------------
        n_left = b - max(0, self.M - self.n_seen)
        v_t = 0
        if n_left > 0:
            rand_indices = np.random.uniform(0, self.n_seen, int(n_left))
            v_t = int(np.sum(rand_indices < self.M))
            
        max_updates = max(1, b // 2) # Ensures v_t can be 1 when b=1
        v_t = int(min(max_updates, max(1, v_t))) # Floor at 1 if n_left triggered it

        if v_t == 0:
            self.n_seen += b
            self.mmd_history.append(0.0)
            return

        X_buf_np = np.vstack(self.buffer_X)
        S_buffer = self._compute_similarities(X_buf_np)

        pi_batch = self._rank_probabilities(S_batch, invert=False)
        pi_drop = self._rank_probabilities(S_buffer, invert=True)

        I_retain = np.random.choice(b, size=v_t, replace=False, p=pi_batch)
        I_drop = set(np.random.choice(len(self.buffer_X), size=v_t, replace=False, p=pi_drop))

        new_X, new_y, new_prov = [], [], []
        for i in range(len(self.buffer_X)):
            if i not in I_drop:
                new_X.append(self.buffer_X[i])
                new_y.append(self.buffer_y[i])
                new_prov.append(self.buffer_prov[i])
                
        for idx in I_retain:
            new_X.append(X_batch_np[idx].copy())
            new_y.append(int(y_batch_np[idx]))
            new_prov.append(prov_batch[idx])

        self.buffer_X = new_X
        self.buffer_y = new_y
        self.buffer_prov = new_prov
        self.n_seen += b
        self.mmd_history.append(0.0)

        # ---- FIX 2: Simulate Fingerprint Attunement -----------------------
        self._update_fingerprints(X_batch_np[I_retain], y_batch_np[I_retain])

    def _update_fingerprints(self, embeddings: np.ndarray, labels: np.ndarray) -> None:
        """Dynamically update fingerprints using EMA to simulate attunement."""
        if len(embeddings) == 0:
            return
            
        learning_rate = 0.05
        with torch.no_grad():
            for idx, label in enumerate(labels):
                target_fp = self.fingerprints[label, 0, :]
                new_val = torch.tensor(embeddings[idx], dtype=torch.float32, device=self.device)
                updated_fp = (1 - learning_rate) * target_fp + learning_rate * new_val
                self.fingerprints[label, 0, :] = updated_fp

    def get_final_coreset(self) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
        if not self.buffer_X:
            return np.array([]), np.array([]), []
        n = len(self.buffer_X)
        return np.arange(n), np.ones(n) / n, list(self.buffer_prov)

    def print_coreset_provenance(self) -> None:
        if not self.buffer_prov:
            print("Coreset buffer is empty.")
            return
        w = 1.0 / len(self.buffer_prov)
        print(f"Coreset size: {len(self.buffer_prov)}")
        for b_idx, l_idx in self.buffer_prov:
            print(f"  weight={w:.4f} | batch={b_idx} | local={l_idx}")