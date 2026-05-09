"""
OCS: Online Coreset Selection via gradient matching.

Reference: "Online Coreset Selection for Rehearsal-based Continual Learning"
           (Yoon et al., 2022).
"""

from __future__ import annotations

import os   
import sys
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from streamers.abstract_streamer import AbstractStreamingCoreset


class OCSStreamingCoreset(AbstractStreamingCoreset):
    """
    Online Coreset Selection (OCS) via gradient similarity + diversity scoring.

    Includes mini-batch accumulation (to support b=1 stream loops) and
    surrogate model training on the retained points to generate meaningful gradients.
    """

    def __init__(
        self,
        capacity: int,
        model: nn.Module,
        criterion: nn.Module,
        num_classes: int,
        tau: float = 1000.0,
        r2c_iter: int = 100,
        device: str = "cpu",
        selection_ratio: float = 0.5,
        mini_batch_size: int = 20,  # <-- Accumulates incoming points to score them together
    ):
        self.capacity = capacity
        self.model = model.to(device)
        self.criterion = criterion
        self.num_classes = num_classes
        self.tau = tau
        self.r2c_iter = r2c_iter
        self.device = device
        self.selection_ratio = selection_ratio
        self.mini_batch_size = mini_batch_size

        self.global_batch_counter: int = 0

        # Physical buffer
        self.buffer_X: List[np.ndarray] = []
        self.buffer_y: List[int] = []
        self.buffer_provenance: List[Tuple[int, int]] = []
        self.mmd_history: List[float] = []

        # Local accumulation buffers (to handle b=1 loop)
        self.local_X_accum: List[np.ndarray] = []
        self.local_y_accum: List[int] = []
        self.local_prov_accum: List[Tuple[int, int]] = []

        # Setup the optimizer to actually train the surrogate model!
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)

    # ------------------------------------------------------------------
    # Interface compatibility
    # ------------------------------------------------------------------

    @property
    def M(self) -> int:
        return self.capacity

    @property
    def buffer_weights(self) -> np.ndarray:
        n = len(self.buffer_X)
        if n == 0:
            return np.empty(0, dtype=np.float64)
        return np.ones(n, dtype=np.float64) / float(n)

    # ------------------------------------------------------------------
    # Gradient utilities
    # ------------------------------------------------------------------

    def _flat_grads(self) -> torch.Tensor:
        parts = [
            p.grad.detach().flatten()
            for p in self.model.parameters()
            if p.grad is not None
        ]
        return torch.cat(parts) if parts else torch.zeros(1, device=self.device)

    def _per_example_grads(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        grads = []
        for i in range(len(X)):
            self.model.zero_grad()
            loss = self.criterion(self.model(X[i : i + 1]), y[i : i + 1])
            loss.backward()
            grads.append(self._flat_grads())
        return torch.stack(grads)  # (b, P)

    def _score(self, g: torch.Tensor, eg: torch.Tensor, ref_grads: torch.Tensor | None) -> torch.Tensor:
        eps = torch.ones(eg.shape[0], device=self.device) * 1e-6

        ng = torch.norm(g)
        neg = torch.norm(eg, dim=1)

        mean_sim = torch.matmul(g, eg.t()) / torch.maximum(ng * neg, eps)

        negd = neg.unsqueeze(1)
        cross_div = torch.matmul(eg, eg.t()) / torch.maximum(
            torch.matmul(negd, negd.t()),
            torch.ones(eg.shape[0], eg.shape[0], device=self.device) * 1e-6,
        )
        mean_div = cross_div.mean(dim=0)

        coreset_aff = torch.zeros(eg.shape[0], device=self.device)
        if ref_grads is not None:
            ref_ng = torch.norm(ref_grads)
            coreset_aff = torch.matmul(ref_grads, eg.t()) / torch.maximum(ref_ng * neg, eps)

        return mean_sim - mean_div + self.tau * coreset_aff

    # ------------------------------------------------------------------
    # AbstractStreamingCoreset interface
    # ------------------------------------------------------------------

    def process_batch(self, X_batch_np: np.ndarray, y_batch_np: np.ndarray, batch_idx: int) -> None:
        # 1. Accumulate points until we hit the mini-batch threshold
        for i in range(X_batch_np.shape[0]):
            self.local_X_accum.append(X_batch_np[i])
            self.local_y_accum.append(y_batch_np[i])
            self.local_prov_accum.append((batch_idx, i))

        if len(self.local_X_accum) < self.mini_batch_size:
            return  # Wait for more data before processing

        # 2. Convert accumulated batch to tensors
        X_t = torch.tensor(np.array(self.local_X_accum), dtype=torch.float32, device=self.device)
        y_t = torch.tensor(np.array(self.local_y_accum), dtype=torch.long, device=self.device)

        b = len(self.local_X_accum)
        k = max(1, int(b * self.selection_ratio))
        self.global_batch_counter += 1

        # 3. OCS Selection (Random warmup vs Gradient Scoring)
        if self.global_batch_counter <= self.r2c_iter:
            pick = torch.randperm(b)[:k].cpu().numpy()
        else:
            eg = self._per_example_grads(X_t, y_t)
            g = eg.mean(dim=0)

            ref_grads = None
            if self.buffer_X:
                buf_X = torch.tensor(np.vstack(self.buffer_X), dtype=torch.float32, device=self.device)
                buf_y = torch.tensor(self.buffer_y, dtype=torch.long, device=self.device)
                self.model.zero_grad()
                ref_loss = self.criterion(self.model(buf_X), buf_y)
                ref_loss.backward()
                ref_grads = self._flat_grads()

            scores = self._score(g, eg, ref_grads)
            pick = torch.argsort(scores, descending=True)[:k].cpu().numpy()

        # 4. Train the surrogate model on the selected informative points!
        self.model.train()
        self.optimizer.zero_grad()
        loss = self.criterion(self.model(X_t[pick]), y_t[pick])
        loss.backward()
        self.optimizer.step()

        # 5. Add to buffer and handle eviction (Label-Agnostic FIFO)
        for idx in pick:
            self.buffer_X.append(self.local_X_accum[idx])
            self.buffer_y.append(self.local_y_accum[idx])
            self.buffer_provenance.append(self.local_prov_accum[idx])

        while len(self.buffer_X) > self.capacity:
            # Simple FIFO eviction 
            self.buffer_X.pop(0)
            self.buffer_y.pop(0)
            self.buffer_provenance.pop(0)

        # 6. Clear local accumulation and record history
        self.local_X_accum = []
        self.local_y_accum = []
        self.local_prov_accum = []
        self.mmd_history.append(0.0)

    def get_final_coreset(self) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
        """
        Returns the current buffer as the final coreset representation.
        """
        if not self.buffer_X:
            return np.array([]), np.array([]), []

        n = len(self.buffer_X)
        X_out = np.vstack(self.buffer_X)
        weights = np.ones(n) / n
        return X_out, weights, list(self.buffer_provenance)

    def print_coreset_provenance(self) -> None:
        if not self.buffer_provenance:
            print("Coreset buffer is empty.")
            return
        w = 1.0 / len(self.buffer_provenance)
        print(f"Coreset size: {len(self.buffer_provenance)}")
        for b_idx, l_idx in self.buffer_provenance:
            print(f"  weight={w:.4f} | batch={b_idx} | local={l_idx}")