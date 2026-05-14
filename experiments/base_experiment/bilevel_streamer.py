from abc import ABC, abstractmethod
from typing import Optional, Tuple, List, Any
import numpy as np
import torch
import torch.nn as nn

class AbstractStreamingCoreset(ABC):
    @abstractmethod
    def process_batch(self, X_batch_np: np.ndarray, y_batch_np: np.ndarray, batch_idx: int) -> None:
        pass
    @abstractmethod
    def get_final_coreset(self) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
        pass
    @abstractmethod
    def print_coreset_provenance(self) -> None:
        pass


class BilevelStreamingCoreset(AbstractStreamingCoreset):
    """
    Implements the Streaming Coreset via Bilevel Optimization as described in:
    "Coresets via Bilevel Optimization for Continual Learning and Streaming"

    Uses the Merge-Reduce framework for streaming buffer management and
    Greedy Forward Selection via Matching Pursuit (Algorithm 1) for
    summarization, computing implicit gradients through a surrogate model.

    The surrogate model determines the gradient geometry used for selection.
    By default a linear head (nn.Linear) is built once the first batch
    arrives and the embedding dimension is known; pass `surrogate_model`
    explicitly to use a fixed architecture (e.g. ResNet-18 linear head).
    """
    def __init__(self,
                 buffer_capacity: int,
                 surrogate_model: Optional[nn.Module] = None,
                 criterion: Optional[nn.Module] = None,
                 device: Optional[torch.device] = None,
                 nr_slots: int = 10,
                 n_classes: int = 10):

        self.buffer_capacity = buffer_capacity
        self._surrogate_model = surrogate_model   # may be None until first batch
        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()
        self.device = device if device is not None else torch.device("cpu")
        self.n_classes = n_classes

        # Buffer Management (Section 5: Streaming Coresets via Merge-Reduce)
        # "we divide the buffer into s equally-sized slots"
        self.nr_slots = nr_slots
        self.slot_size = max(1, buffer_capacity // nr_slots)

        # Buffer holds tuples of: (X_tensor, y_tensor, slot_weight, provenance_list)
        self.buffer = []

    # ------------------------------------------------------------------
    # Lazy surrogate initialisation
    # ------------------------------------------------------------------

    @property
    def surrogate_model(self) -> nn.Module:
        return self._surrogate_model

    def _ensure_surrogate(self, embed_dim: int) -> None:
        """Build a linear head if no surrogate was supplied at construction."""
        if self._surrogate_model is None:
            self._surrogate_model = nn.Linear(embed_dim, self.n_classes)
        self._surrogate_model = self._surrogate_model.to(self.device)

    # ------------------------------------------------------------------
    # Public buffer views  (expected by _extract_buffer in eval scripts)
    # ------------------------------------------------------------------

    @property
    def buffer_X(self) -> List[np.ndarray]:
        """One numpy array per coreset point (shape: [embed_dim])."""
        result: List[np.ndarray] = []
        for X_tensor, _, _, _ in self.buffer:
            for i in range(len(X_tensor)):
                result.append(X_tensor[i].detach().cpu().numpy())
        return result

    @property
    def buffer_y(self) -> List[int]:
        """Class label for each coreset point."""
        result: List[int] = []
        for _, y_tensor, _, _ in self.buffer:
            for yi in y_tensor.detach().cpu().numpy():
                result.append(int(yi))
        return result

    @property
    def buffer_weights(self) -> np.ndarray:
        """Per-point importance weights normalised to sum to 1.0.

        Within each slot, weight is proportional to how many original
        stream points that slot represents (slot_weight), divided equally
        among the slot's coreset points.
        """
        if not self.buffer:
            return np.array([], dtype=np.float64)
        total_weight = sum(slot[2] for slot in self.buffer)
        if total_weight <= 0.0:
            total_weight = 1.0
        weights: List[float] = []
        for X_tensor, _, w, _ in self.buffer:
            n = len(X_tensor)
            if n > 0:
                per_point = w / (n * total_weight)
                weights.extend([per_point] * n)
        arr = np.array(weights, dtype=np.float64)
        s = arr.sum()
        if s > 1e-12:
            arr /= s
        return arr

    def process_batch(self, X_batch_np: np.ndarray, y_batch_np: np.ndarray, batch_idx: int) -> None:
        """
        Receives a new batch of data, summarizes it down to slot_size, and appends it to the buffer.
        If the buffer exceeds the maximum number of slots, we execute the Merge-Reduce step.
        """
        # Lazy-build the surrogate the first time we see data
        self._ensure_surrogate(X_batch_np.shape[1])

        # The Numpy/Torch Bridge
        X_tensor = torch.from_numpy(X_batch_np).float().to(self.device)
        y_tensor = torch.from_numpy(y_batch_np).long().to(self.device)
        
        # Track data provenance (batch_idx, local_idx)
        prov = [(batch_idx, i) for i in range(len(X_batch_np))]
        
        # Compress the new batch into a single slot size using Bilevel Optimization
        X_reduced, y_reduced, prov_reduced = self._bilevel_greedy_selection(X_tensor, y_tensor, prov, self.slot_size)
        
        # "A new batch is compressed into a new slot with associated \beta... and is appended to the buffer"
        batch_weight = len(X_batch_np) / self.slot_size  # \beta proportional to the number of points it represents
        self.buffer.append((X_reduced, y_reduced, batch_weight, prov_reduced))
        
        # "The reduction to size m happens as follows: select consecutive slots i and i+1..., 
        # join the contents (merge) and create the coreset of the merged data (reduce)."
        if len(self.buffer) > self.nr_slots:
            self._merge_reduce()

    def _merge_reduce(self) -> None:
        """
        Maintains the buffer capacity M by merging consecutive slots with the lowest 
        combined weight and then reducing them back down to `slot_size`.
        """
        # Find the two consecutive slots with the minimum combined weight
        min_weight = float('inf')
        merge_idx = 0
        
        for i in range(len(self.buffer) - 1):
            combined_weight = self.buffer[i][2] + self.buffer[i+1][2]
            if combined_weight < min_weight:
                min_weight = combined_weight
                merge_idx = i
                
        # Merge contents
        X1, y1, w1, prov1 = self.buffer[merge_idx]
        X2, y2, w2, prov2 = self.buffer[merge_idx + 1]
        
        X_merged = torch.cat([X1, X2], dim=0)
        y_merged = torch.cat([y1, y2], dim=0)
        prov_merged = prov1 + prov2
        new_weight = w1 + w2 
        
        # Reduce back to slot_size using Algorithm 1
        X_reduced, y_reduced, prov_reduced = self._bilevel_greedy_selection(X_merged, y_merged, prov_merged, self.slot_size)
        
        # Update buffer by removing the two old slots and inserting the new merged-reduced slot
        new_slot = (X_reduced, y_reduced, new_weight, prov_reduced)
        self.buffer.pop(merge_idx + 1)
        self.buffer[merge_idx] = new_slot

    def _bilevel_greedy_selection(self, X: torch.Tensor, y: torch.Tensor, prov: List, m: int):
        """
        Implements Algorithm 1: Coresets via Bilevel Optimization using matching pursuit.
        Selects a subset of size `m` by greedily maximizing the bilinear similarity (Eq 5).
        """
        n = len(X)
        if n <= m:
            return X, y, prov
            
        self.surrogate_model.eval()  # Keep deterministic for scoring
        params = list(self.surrogate_model.parameters())
        
        # Step 1 & 2: Initialize selected set (S_t) with 1 random index
        selected_indices = [np.random.randint(n)]
        
        # Step 3: Incremental Subset Selection
        for t in range(2, m + 1):
            
            # Outer objective gradient: \nabla_\theta g(\theta) over ALL candidate data
            self.surrogate_model.zero_grad()
            out_all = self.surrogate_model(X)
            loss_outer = self.criterion(out_all, y)
            grad_outer = self._flat_grad(loss_outer, params, create_graph=False, retain_graph=True)
            
            # Inner objective loss evaluation: f(\theta) over CURRENT selected data S_{t-1}
            X_S = X[selected_indices]
            y_S = y[selected_indices]
            
            self.surrogate_model.zero_grad()
            out_inner = self.surrogate_model(X_S)
            loss_inner = self.criterion(out_inner, y_S)
            
            # Inverse Hessian-Vector Product: (\nabla^2 f)^{-1} \nabla g 
            # Note: Approximates Eq. (3)
            ihvp_v = self._ihvp(loss_inner, params, grad_outer)
            
            # Find point k* that maximizes the negative implicit gradient (Eq. 5)
            best_score = -float('inf')
            best_k = -1
            
            candidates = [i for i in range(n) if i not in selected_indices]
            for k in candidates:
                self.surrogate_model.zero_grad()
                out_k = self.surrogate_model(X[k:k+1])
                loss_k = self.criterion(out_k, y[k:k+1])
                
                # J_k = \nabla_\theta l_k(\theta)
                J_k = self._flat_grad(loss_k, params, create_graph=False, retain_graph=False)
                
                # Bilinear similarity: J_k^T * (H^{-1} * \nabla g)
                score = torch.dot(J_k, ihvp_v).item()
                
                if score > best_score:
                    best_score = score
                    best_k = k
                    
            # Add selected atom to support
            selected_indices.append(best_k)

        # Gather the final subset
        X_core = X[selected_indices]
        y_core = y[selected_indices]
        prov_core = [prov[i] for i in selected_indices]
        
        return X_core, y_core, prov_core

    def get_final_coreset(self) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
        """
        Concatenates all slots in the buffer to yield the final task-agnostic coreset.
        """
        if not self.buffer:
            return np.array([]), np.array([]), []
            
        X_final = torch.cat([slot[0] for slot in self.buffer], dim=0).cpu().numpy()
        y_final = torch.cat([slot[1] for slot in self.buffer], dim=0).cpu().numpy()
        prov_final = sum([slot[3] for slot in self.buffer], [])
        
        return X_final, y_final, prov_final

    def print_coreset_provenance(self) -> None:
        if not self.buffer:
            print("Buffer is currently empty.")
            return
            
        print(f"--- Coreset Provenance (Buffer Capacity: {self.buffer_capacity}, Slots: {len(self.buffer)}) ---")
        for idx, (_, _, w, prov) in enumerate(self.buffer):
            print(f"Slot {idx} [Weight: {w:.2f}]: {len(prov)} items -> {prov[:3]} ...")

    # =========================================================================
    # Mathematical / PyTorch utilities for Implicit Gradients
    # =========================================================================

    def _flat_grad(self, loss: torch.Tensor, params: List[torch.Tensor], retain_graph: bool, create_graph: bool) -> torch.Tensor:
        """Flattens the gradients of the model parameters into a single 1D vector."""
        grads = torch.autograd.grad(loss, params, retain_graph=retain_graph, create_graph=create_graph)
        return torch.cat([g.contiguous().view(-1) for g in grads])

    def _hvp(self, loss: torch.Tensor, params: List[torch.Tensor], v: torch.Tensor) -> torch.Tensor:
        """
        Computes the Hessian-Vector Product exactly without fully materializing the Hessian matrix.
        H * v = \nabla_\theta ( \nabla_\theta L \cdot v )
        """
        dl_dtheta = self._flat_grad(loss, params, create_graph=True, retain_graph=True)
        dl_dtheta_dot_v = torch.dot(dl_dtheta, v)
        hvp_val = self._flat_grad(dl_dtheta_dot_v, params, retain_graph=True, create_graph=False)
        return hvp_val

    def _cg_solve(self, f_Ax, b: torch.Tensor, cg_iters: int = 10) -> torch.Tensor:
        """
        Conjugate Gradient method to solve Ax = b for x entirely in PyTorch tensors.
        Used to approximate the inverse Hessian.
        """
        x = torch.zeros_like(b)
        r = b.clone()
        p = r.clone()
        rsold = torch.dot(r, r)
        
        for i in range(cg_iters):
            Ap = f_Ax(p)
            alpha = rsold / (torch.dot(p, Ap) + 1e-8)
            x = x + alpha * p
            r = r - alpha * Ap
            rsnew = torch.dot(r, r)
            if torch.sqrt(rsnew) < 1e-6:
                break
            p = r + (rsnew / rsold) * p
            rsold = rsnew
            
        return x

    def _ihvp(self, loss: torch.Tensor, params: List[torch.Tensor], v: torch.Tensor, cg_iters: int = 15) -> torch.Tensor:
        """
        Solves for the Inverse Hessian Vector Product: x = H^{-1} v
        Where H is the Hessian matrix of the inner loss.
        """
        def f_Ax(x):
            # Added slight damping/Tikhonov regularization (1e-4) to ensure positive-definiteness
            return self._hvp(loss, params, x) + 1e-4 * x 
            
        return self._cg_solve(f_Ax, v, cg_iters)