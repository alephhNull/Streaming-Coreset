from abc import ABC, abstractmethod
from typing import Tuple, List, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    
    This algorithm manages a streaming memory buffer by selecting samples that maximize:
    1. Minibatch Similarity (S): Representativeness for the current task.
    2. Sample Diversity (V): Minimal redundancy among selected samples.
    3. Coreset Affinity (A): Minimal interference with previously stored knowledge.
    """
    
    def __init__(
        self, 
        buffer_capacity: int, 
        surrogate_model: nn.Module, 
        criterion: nn.Module, 
        device: torch.device,
        tau: float = 1000.0  # Hyperparameter balancing current adaptation vs. past affinity (from author's config)
    ):
        self.buffer_capacity = buffer_capacity
        self.surrogate_model = surrogate_model.to(device)
        self.criterion = criterion.to(device)
        self.device = device
        self.tau = tau
        
        # Internal state for the streaming buffer
        self.buffer_X: torch.Tensor = None
        self.buffer_y: torch.Tensor = None
        self.buffer_provenance: List[Tuple[int, int]] = []
        
    def _compute_per_example_gradients(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Dynamically computes flattened per-example gradients for arbitrary surrogate models.
        This ensures task and architecture agnosticism.
        """
        self.surrogate_model.eval()
        example_grads = []
        
        # We loop over the batch to extract per-example gradients.
        # (In a production environment, libraries like functorch/vmap or Backpack could optimize this)
        for i in range(len(X)):
            self.surrogate_model.zero_grad()
            output = self.surrogate_model(X[i:i+1])
            loss = self.criterion(output, y[i:i+1])
            
            # Extract and flatten gradients for all parameters
            grads = torch.autograd.grad(loss, self.surrogate_model.parameters())
            flat_grad = torch.cat([g.detach().view(-1) for g in grads])
            example_grads.append(flat_grad)
            
        return torch.stack(example_grads)

    def process_batch(self, X_batch_np: np.ndarray, y_batch_np: np.ndarray, batch_idx: int) -> None:
        """
        Processes an incoming batch, scoring elements based on OCS equations, 
        and evicts the lowest-scoring points if the buffer exceeds capacity M.
        """
        # 1. Bridge numpy arrays to PyTorch Tensors
        X_batch = torch.from_numpy(X_batch_np).float().to(self.device)
        y_batch = torch.from_numpy(y_batch_np).long().to(self.device)
        
        # Create provenance tracking for the incoming batch
        batch_provenance = [(batch_idx, i) for i in range(len(X_batch))]
        
        if self.buffer_X is None or len(self.buffer_X) == 0:
            # First initialization: No previous coreset to compute Affinity (A) against.
            pool_X = X_batch
            pool_y = y_batch
            pool_prov = batch_provenance
            ref_grads = None
        else:
            # 2. Compute the reference gradients (mean gradients of the EXISTING buffer)
            # This represents the "knowledge of previous tasks" for Coreset Affinity
            with torch.no_grad():
                buffer_eg = self._compute_per_example_gradients(self.buffer_X, self.buffer_y)
                ref_grads = torch.mean(buffer_eg, dim=0)
            
            # Combine existing buffer and new batch into a single evaluation pool
            pool_X = torch.cat([self.buffer_X, X_batch], dim=0)
            pool_y = torch.cat([self.buffer_y, y_batch], dim=0)
            pool_prov = self.buffer_provenance + batch_provenance

        # If total pool is within capacity, just store and return
        if len(pool_X) <= self.buffer_capacity:
            self.buffer_X = pool_X
            self.buffer_y = pool_y
            self.buffer_provenance = pool_prov
            return

        # 3. Compute per-example gradients for the ENTIRE pool
        eg = self._compute_per_example_gradients(pool_X, pool_y)
        
        # 4. Compute Mean Gradient of the current pool (used for Minibatch Similarity)
        g = torch.mean(eg, dim=0)
        
        # Precompute norms for stability and cosine similarity computations
        ng = torch.norm(g)
        neg = torch.norm(eg, dim=1)
        eps = torch.ones_like(neg) * 1e-6
        neg_clamped = torch.maximum(neg, eps)
        
        # --- Eq 3: Minibatch Similarity (S) ---
        # Selects samples representative of the current combined target dataset
        # S(b | B) = dot(grad(b), mean_grad(B)) / (norm(grad(b)) * norm(mean_grad(B)))
        mean_sim = torch.matmul(eg, g) / torch.maximum(ng * neg, eps)
        
        # --- Eq 4: Sample Diversity (V) ---
        # Discourages redundancy by computing negative cross-divergence
        # V(b | B \ b) represented as negative average similarity to all other samples
        negd = torch.unsqueeze(neg_clamped, 1)
        cross_sim_matrix = torch.matmul(eg, eg.t()) / torch.maximum(torch.matmul(negd, negd.t()), eps.unsqueeze(1) * eps)
        mean_div = torch.mean(cross_sim_matrix, dim=0) 
        
        # --- Eq 7: Coreset Affinity (A) ---
        # Promotes minimum interference with the previous tasks
        coreset_aff = torch.zeros_like(mean_sim)
        if ref_grads is not None:
            ref_ng = torch.norm(ref_grads)
            coreset_aff = torch.matmul(eg, ref_grads) / torch.maximum(ref_ng * neg, eps)
            
        # --- Eq 8: Final OCS Selection Criterion ---
        # argmax (S + V + tau * A) -> V is represented by subtracting mean_div
        measure = mean_sim - mean_div + (self.tau * coreset_aff)
        
        # 5. Eviction Logic: Sort descending and truncate to buffer_capacity
        _, top_indices = torch.sort(measure, descending=True)
        keep_indices = top_indices[:self.buffer_capacity]
        
        # 6. Update the internal buffer state
        self.buffer_X = pool_X[keep_indices]
        self.buffer_y = pool_y[keep_indices]
        self.buffer_provenance = [pool_prov[i.item()] for i in keep_indices]

    def get_final_coreset(self) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
        """
        Returns the final coreset points, their labels, and their exact provenance.
        """
        if self.buffer_X is None:
            return np.array([]), np.array([]), []
            
        final_X = self.buffer_X.cpu().numpy()
        final_y = self.buffer_y.cpu().numpy()
        
        return final_X, final_y, self.buffer_provenance

    def print_coreset_provenance(self) -> None:
        """
        Utility print out demonstrating which incoming batches specific points survived from.
        """
        if not self.buffer_provenance:
            print("Buffer is currently empty.")
            return
            
        print(f"--- Coreset Provenance (Capacity {self.buffer_capacity}) ---")
        for i, (batch_idx, local_idx) in enumerate(self.buffer_provenance):
            print(f"Slot {i:03d} -> Source Batch {batch_idx:03d}, Local Index: {local_idx:03d}")