from abc import ABC, abstractmethod
from typing import Tuple, List, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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

class OCSStreamingCoreset(AbstractStreamingCoreset):
    """
    Online Coreset Selection (OCS) for Rehearsal-based Continual Learning.
    Paper: https://openreview.net/forum?id=f9D-5WNG4Nv (ICLR 2022)
    
    Corrected Implementation: 
    Applies the S, V, A heuristics strictly WITHIN class-wise budgets to prevent 
    gradient-magnitude domination by the current task, mirroring the authors' 
    `classwise_fair_selection` logic.
    """
    
    def __init__(
        self, 
        buffer_capacity: int, 
        surrogate_model: nn.Module, 
        criterion: nn.Module, 
        optimizer: optim.Optimizer,
        device: torch.device,
        tau: float = 1000.0  
    ):
        self.buffer_capacity = buffer_capacity
        self.surrogate_model = surrogate_model.to(device)
        self.criterion = criterion.to(device)
        self.optimizer = optimizer
        self.device = device
        self.tau = tau
        
        self.buffer_X: torch.Tensor = None
        self.buffer_y: torch.Tensor = None
        self.buffer_provenance: List[Tuple[int, int]] = []

    def _train_surrogate(self, X: torch.Tensor, y: torch.Tensor) -> None:
        """
        Takes a single online gradient step on the incoming stream chunk
        so the surrogate model tracks the true task boundaries.
        """
        self.surrogate_model.train()
        self.optimizer.zero_grad()
        outputs = self.surrogate_model(X)
        loss = self.criterion(outputs, y)
        loss.backward()
        self.optimizer.step()
        
    def _compute_per_example_gradients(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        self.surrogate_model.train() # Must be in train mode to build graph properly
        example_grads = []
        
        for i in range(len(X)):
            self.surrogate_model.zero_grad()
            output = self.surrogate_model(X[i:i+1])
            loss = self.criterion(output, y[i:i+1])
            
            grads = torch.autograd.grad(loss, self.surrogate_model.parameters())
            flat_grad = torch.cat([g.detach().view(-1) for g in grads])
            example_grads.append(flat_grad)
            
        return torch.stack(example_grads)

    def process_batch(self, X_batch_np: np.ndarray, y_batch_np: np.ndarray, batch_idx: int) -> None:
        X_batch = torch.from_numpy(X_batch_np).float().to(self.device)
        y_batch = torch.from_numpy(y_batch_np).long().to(self.device)
        batch_prov = [(batch_idx, i) for i in range(len(X_batch))]

        self._train_surrogate(X_batch, y_batch)
        
        # 1. Pool current buffer and new batch
        if self.buffer_X is None or len(self.buffer_X) == 0:
            pool_X = X_batch
            pool_y = y_batch
            pool_prov = batch_prov
            ref_grads = None
        else:
            # OCS Affinity (A) relies on the gradients of the PREVIOUS coreset
            # to measure interference before the new batch corrupts it.
            with torch.enable_grad():
                buffer_eg = self._compute_per_example_gradients(self.buffer_X, self.buffer_y)
                ref_grads = torch.mean(buffer_eg, dim=0).detach()
            
            pool_X = torch.cat([self.buffer_X, X_batch], dim=0)
            pool_y = torch.cat([self.buffer_y, y_batch], dim=0)
            pool_prov = self.buffer_provenance + batch_prov

        if len(pool_X) <= self.buffer_capacity:
            self.buffer_X, self.buffer_y, self.buffer_provenance = pool_X, pool_y, pool_prov
            return

        # 2. Determine exact budgets per class to prevent catastrophic eviction
        unique_classes = torch.unique(pool_y)
        budget_per_class = self.buffer_capacity // len(unique_classes)
        residual_slots = self.buffer_capacity % len(unique_classes)

        new_X, new_y, new_prov = [], [], []

        # 3. Apply OCS intra-class
        for c in unique_classes:
            mask = (pool_y == c)
            cX = pool_X[mask]
            cy = pool_y[mask]
            cProv = [pool_prov[i] for i in torch.where(mask)[0].tolist()]
            
            # Distribute residuals fairly
            budget = budget_per_class + (1 if residual_slots > 0 else 0)
            if residual_slots > 0: residual_slots -= 1

            if len(cX) <= budget:
                new_X.append(cX)
                new_y.append(cy)
                new_prov.extend(cProv)
                continue
                
            # If a class exceeds its budget, use OCS to compress it
            with torch.enable_grad():
                eg = self._compute_per_example_gradients(cX, cy)
                
            g = torch.mean(eg, dim=0)
            ng = torch.norm(g)
            neg = torch.norm(eg, dim=1)
            eps = torch.ones_like(neg) * 1e-6
            neg_clamped = torch.maximum(neg, eps)
            
            # S: Minibatch similarity (Relative to the class mean gradient)
            mean_sim = torch.matmul(eg, g) / torch.maximum(ng * neg, eps)
            
            # V: Sample Diversity (Cross-divergence within the class)
            negd = torch.unsqueeze(neg_clamped, 1)
            cross_sim = torch.matmul(eg, eg.t()) / torch.maximum(torch.matmul(negd, negd.t()), eps.unsqueeze(1) * eps)
            mean_div = torch.mean(cross_sim, dim=0) 
            
            # A: Coreset Affinity (Interference with past buffer knowledge)
            coreset_aff = torch.zeros_like(mean_sim)
            if ref_grads is not None:
                ref_ng = torch.norm(ref_grads)
                coreset_aff = torch.matmul(eg, ref_grads) / torch.maximum(ref_ng * neg, eps)
                
            # OCS Score
            measure = mean_sim - mean_div + (self.tau * coreset_aff)
            
            # Evict the lowest scoring elements within this class
            _, top_indices = torch.sort(measure, descending=True)
            keep_indices = top_indices[:budget]
            
            new_X.append(cX[keep_indices])
            new_y.append(cy[keep_indices])
            new_prov.extend([cProv[i.item()] for i in keep_indices])

        # 4. Finalize the updated state
        self.buffer_X = torch.cat(new_X, dim=0)
        self.buffer_y = torch.cat(new_y, dim=0)
        self.buffer_provenance = new_prov

    @property
    def buffer_weights(self) -> np.ndarray:
        n = len(self.buffer_X) if self.buffer_X is not None else 0
        if n == 0: return np.array([], dtype=np.float64)
        return np.full(n, 1.0 / n, dtype=np.float64)

    def get_final_coreset(self) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
        if self.buffer_X is None:
            return np.array([]), np.array([]), []
        return self.buffer_X.cpu().numpy(), self.buffer_y.cpu().numpy(), self.buffer_provenance

    def print_coreset_provenance(self) -> None:
        if not self.buffer_provenance:
            print("Buffer is empty.")
            return
        print(f"--- Coreset Provenance (Capacity {self.buffer_capacity}) ---")
        for i, (batch_idx, local_idx) in enumerate(self.buffer_provenance):
            print(f"Slot {i:03d} -> Source Batch {batch_idx:03d}, Local Index: {local_idx:03d}")