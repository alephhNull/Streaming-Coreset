import numpy as np
from typing import List, Tuple
from sklearn.kernel_approximation import RBFSampler
from abc import ABC, abstractmethod

# --- Abstract Base Class ---
class AbstractStreamingCoreset(ABC):
    @abstractmethod
    def process_batch(self, X_batch_np: np.ndarray, y_batch_np: np.ndarray, batch_idx: int) -> None:
        pass
    @abstractmethod
    def get_final_coreset(self) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
        pass

# --- New Adaptive Algorithm ---
class AdaptiveWeightedHerdingStreamer(AbstractStreamingCoreset):
    """
    Online Coreset Selection via Sparse Mirror Descent.
    
    Mechanism:
    1. Tracks global mean mu using RFF.
    2. Maintains a sparse 'active set' of size M.
    3. Uses Frank-Wolfe style 'swaps' to update the support (geometry).
    4. Uses Mirror Descent (Exponentiated Gradient) to optimize weights on the simplex.
    
    This ensures sublinear regret by continuously aligning the weighted hull 
    with the drifting stream mean.
    """
    
    def __init__(self, 
                 coreset_size: int, 
                 buffer_capacity: int, 
                 sampler: RBFSampler, 
                 batch_size: int,
                 learning_rate: float = 1.0, 
                 md_steps: int = 5):
        
        assert coreset_size <= buffer_capacity
        self.coreset_size = coreset_size
        self.buffer_capacity = buffer_capacity
        self.sampler = sampler
        self.batch_size = batch_size
        
        # Hyperparameters for Optimization
        self.base_lr = learning_rate
        self.md_steps = md_steps # Steps of Mirror Descent per batch
        
        # Dimensions
        self.rff_dim = sampler.n_components
        self.feature_dim = sampler.random_weights_.shape[0]
        
        # Stream State
        self.mean_rff_stream = np.zeros(self.rff_dim)
        self.num_points_seen = 0
        
        # Buffer (Candidate Pool)
        # buffer_X stores the raw features
        self.buffer_X = np.empty((0, self.feature_dim))
        self.buffer_y = np.empty(0)
        self.buffer_rff = np.empty((0, self.rff_dim)) # Cached embeddings
        self.buffer_prov: List[Tuple[int, int]] = []
        
        # Coreset State (Sparse Support)
        # indices point to locations in self.buffer_X
        self.active_indices: np.ndarray = np.array([], dtype=int)
        self.active_weights: np.ndarray = np.array([], dtype=float)

    def _update_global_mean(self, X_batch_rff: np.ndarray) -> np.ndarray:
        """Standard moving average update for the target mean."""
        batch_len = X_batch_rff.shape[0]
        batch_mean = np.mean(X_batch_rff, axis=0)
        
        old_mean = self.mean_rff_stream.copy()
        
        if self.num_points_seen == 0:
            self.mean_rff_stream = batch_mean
        else:
            alpha = batch_len / (self.num_points_seen + batch_len)
            self.mean_rff_stream = (1 - alpha) * self.mean_rff_stream + alpha * batch_mean
            
        self.num_points_seen += batch_len
        return self.mean_rff_stream - old_mean

    def _get_current_residual(self) -> np.ndarray:
        """
        Computes r = mu - sum(w_i * phi(x_i))
        This residual incorporates the weights directly.
        """
        if len(self.active_indices) == 0:
            return self.mean_rff_stream
        
        # Weighted sum of active set
        active_rffs = self.buffer_rff[self.active_indices]
        # (M, D) -> (D,)
        weighted_estimate = np.dot(self.active_weights, active_rffs)
        
        return self.mean_rff_stream - weighted_estimate

    def _mirror_descent_step(self, residual: np.ndarray, eta: float):
        """
        Performs one step of Exponentiated Gradient Descent on the Simplex.
        
        Gradient of 1/2 ||mu - w*Phi||^2 wrt w_i is -<phi_i, residual>.
        Update: w_i *= exp(eta * <phi_i, residual>)
        Then Normalize.
        """
        if len(self.active_indices) == 0:
            return

        active_rffs = self.buffer_rff[self.active_indices]
        
        # Calculate gradients (correlations with residual)
        # Higher correlation = should have higher weight
        grads = active_rffs @ residual  # Shape (M,)
        
        # Exponentiated update (LogSumExp trick for numerical stability not strictly needed if eta small, 
        # but good practice. Here we use direct exp for speed as active set is small)
        
        # We limit the exponent to avoid overflow
        scaled_grads = eta * grads
        scaled_grads = np.clip(scaled_grads, -50, 50) 
        
        multipliers = np.exp(scaled_grads)
        
        # Update weights
        new_weights = self.active_weights * multipliers
        
        # Project to Simplex (Normalize)
        w_sum = np.sum(new_weights)
        if w_sum > 0:
            self.active_weights = new_weights / w_sum
        else:
            # Fallback to uniform if numerics fail
            self.active_weights = np.ones_like(self.active_weights) / len(self.active_weights)

    def _manage_support(self, residual: np.ndarray):
        """
        Structure Learning:
        Uses Frank-Wolfe duality gap logic to decide if we should
        Swap a point in the coreset with a point in the buffer.
        """
        # 1. Identify Buffer Candidate (Best alignment with residual)
        # We look at ALL buffer points (candidates)
        scores = self.buffer_rff @ residual
        
        # Mask out points already in active set to avoid self-swapping
        mask = np.ones(len(self.buffer_X), dtype=bool)
        mask[self.active_indices] = False
        
        if not np.any(mask):
            return

        scores[~mask] = -np.inf
        
        best_candidate_idx = np.argmax(scores)
        best_candidate_score = scores[best_candidate_idx]
        
        # 2. If Coreset not full, just add
        if len(self.active_indices) < self.coreset_size:
            self.active_indices = np.append(self.active_indices, best_candidate_idx)
            # Initialize with small weight
            new_w = 1.0 / len(self.active_indices)
            self.active_weights = np.append(self.active_weights * (1-new_w), new_w)
            return

        # 3. If Coreset full, check for Swap
        # We look for the "Worst" point in active set.
        # Definition of worst: Lowest weight (mirror descent kills bad points) 
        # OR lowest correlation with residual. 
        # Using weight is more stable for OMD.
        
        worst_active_local_idx = np.argmin(self.active_weights)
        worst_global_idx = self.active_indices[worst_active_local_idx]
        
        # Get gradient of the point we might remove
        # gradient ~ <phi, r>
        worst_point_score = np.dot(self.buffer_rff[worst_global_idx], residual)
        
        # FRANK-WOLFE GAP Condition:
        # If (Best_Candidate_Score - Worst_Current_Score) > Threshold
        # It implies the convex hull can be significantly expanded/improved in direction of residual.
        
        # Heuristic threshold can be 0 or small epsilon
        if best_candidate_score > worst_point_score + 1e-6:
            # Perform Swap
            self.active_indices[worst_active_local_idx] = best_candidate_idx
            # Reset weight for the new point (give it average weight to start)
            # self.active_weights[worst_active_local_idx] = 1.0 / self.coreset_size
            # Alternatively, keep the small weight and let MD grow it (safer)
            pass 

    def _prune_buffer(self):
        """
        Keeps: 
        1. The Active Coreset
        2. The points with highest correlation to the current residual (Orthogonal Candidates)
        """
        if len(self.buffer_X) <= self.buffer_capacity:
            return
            
        keep_mask = np.zeros(len(self.buffer_X), dtype=bool)
        keep_mask[self.active_indices] = True
        
        # Fill remaining spots with points maximizing <phi, r>
        slots_open = self.buffer_capacity - np.sum(keep_mask)
        if slots_open > 0:
            residual = self._get_current_residual()
            scores = self.buffer_rff @ residual
            # Don't pick points already kept
            scores[keep_mask] = -np.inf
            
            top_indices = np.argsort(scores)[::-1][:slots_open]
            keep_mask[top_indices] = True
            
        # Compact arrays
        indices_to_keep = np.where(keep_mask)[0]
        
        # Map old indices to new indices for active set tracking
        old_to_new = {old: new for new, old in enumerate(indices_to_keep)}
        
        # Update Active Indices map
        new_active = []
        new_weights = []
        for i, idx in enumerate(self.active_indices):
            if idx in old_to_new:
                new_active.append(old_to_new[idx])
                new_weights.append(self.active_weights[i])
        
        self.active_indices = np.array(new_active, dtype=int)
        self.active_weights = np.array(new_weights, dtype=float)
        
        # Re-normalize if we accidentally dropped something (shouldn't happen due to keep_mask)
        if np.sum(self.active_weights) > 0:
            self.active_weights /= np.sum(self.active_weights)
            
        # Slice Data
        self.buffer_X = self.buffer_X[indices_to_keep]
        self.buffer_y = self.buffer_y[indices_to_keep]
        self.buffer_rff = self.buffer_rff[indices_to_keep]
        self.buffer_prov = [self.buffer_prov[i] for i in indices_to_keep]

    def process_batch(self, X_batch_np: np.ndarray, y_batch_np: np.ndarray, batch_idx: int) -> None:
        if X_batch_np.shape[0] == 0:
            return

        # 1. Transform & Ingest
        X_rff = self.sampler.transform(X_batch_np)
        
        if len(self.buffer_X) == 0:
            self.buffer_X = X_batch_np
            self.buffer_y = y_batch_np
            self.buffer_rff = X_rff
        else:
            self.buffer_X = np.vstack([self.buffer_X, X_batch_np])
            self.buffer_y = np.concatenate([self.buffer_y, y_batch_np])
            self.buffer_rff = np.vstack([self.buffer_rff, X_rff])
            
        self.buffer_prov.extend([(batch_idx, i) for i in range(len(X_batch_np))])
        
        # 2. Update Global Target (Distribution Shift)
        # This implicitly invalidates the old residual
        self._update_global_mean(X_rff)
        
        # 3. Main Optimization Loop (Structure + Weights)
        # We perform a few interleaved steps of Structure Update and Weight Update
        
        # Adaptive Learning Rate: decays with time to ensure convergence, 
        # but has a floor to handle non-stationarity
        eta = self.base_lr / np.sqrt(1 + self.num_points_seen / 1000.0)
        eta = max(eta, 0.1) # Floor for tracking
        
        # A. Update Structure (Swap/Add)
        # We calculate residual based on OLD weights first
        current_res = self._get_current_residual()
        self._manage_support(current_res)
        
        # B. Optimize Weights (Mirror Descent)
        # We run k steps to realign weights to the new mean and new support
        for _ in range(self.md_steps):
            current_res = self._get_current_residual()
            self._mirror_descent_step(current_res, eta)
            
        # 4. Prune Buffer (Keep coreset + orthogonal candidates)
        self._prune_buffer()

    def get_final_coreset(self) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
        flat_indices = []
        final_prov = []
        
        for idx in self.active_indices:
            b_idx, l_idx = self.buffer_prov[idx]
            flat_idx = b_idx * self.batch_size + l_idx
            flat_indices.append(flat_idx)
            final_prov.append((b_idx, l_idx))
            
        return np.array(flat_indices, dtype=int), self.active_weights, final_prov

    def print_coreset_provenance(self) -> None:
        flat_indices, weights, provenance = self.get_final_coreset()
        print(f"\n--- Final Weighted Coreset (Mirror Descent) ---")
        print(f"Points seen: {self.num_points_seen}")
        print(f"{'Global Index':<15} {'Provenance':<20} {'Weight':<10} {'Label':<10}")
        print("-" * 60)
        
        # Get labels using local buffer indices
        active_labels = self.buffer_y[self.active_indices]
        
        # Sort by weight desc for visualization
        sort_order = np.argsort(weights)[::-1]
        
        for i in sort_order:
            prov_str = f"(B{provenance[i][0]}, I{provenance[i][1]})"
            lbl = active_labels[i]
            print(f"{flat_indices[i]:<15} {prov_str:<20} {weights[i]:<10.6f} {lbl:<10}")