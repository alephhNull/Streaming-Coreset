import numpy as np
from typing import List, Tuple
from sklearn.kernel_approximation import RBFSampler
from abc import ABC, abstractmethod
import cvxpy as cv

# --- Abstract Base Class (Unchanged) ---
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

# --- Updated Adaptive Streamer ---

class AdaptiveKernelHerdingStreamer(AbstractStreamingCoreset):
    """
    Adaptive Online Kernel Herding with Margin-Based Updates.
    Now maintains buffer_y to support supervised learning tasks.
    """

    def __init__(self, coreset_size: int, buffer_capacity: int, sampler: RBFSampler, batch_size: int):
        assert coreset_size <= buffer_capacity, "Coreset size must be <= buffer capacity."
        
        self.coreset_size = coreset_size
        self.buffer_capacity = buffer_capacity
        self.sampler = sampler
        self.batch_size = batch_size
        
        # Feature dimensions
        self.rff_dim = sampler.n_components
        self.feature_dim = sampler.random_weights_.shape[0]

        # Stream State
        self.mean_rff_stream = np.zeros(self.rff_dim)
        self.num_points_seen = 0
        
        # Buffer State
        self.buffer_X = np.empty((0, self.feature_dim))
        self.buffer_y = np.empty(0) # Initialize empty labels array
        self.buffer_provenance: List[Tuple[int, int]] = []
        
        # Cache RFF of buffer to avoid recomputing repeatedly
        self.buffer_rff_cache = np.empty((0, self.rff_dim))
        
        # Herding History State (for warm starts)
        self.coreset_indices: List[int] = [] 
        self.margins: List[float] = []
        self.runner_up_indices: List[int] = []

    def _update_mean(self, X_batch_rff: np.ndarray) -> np.ndarray:
        """
        Updates global mean and returns the Delta vector (change in mean).
        """
        batch_len = X_batch_rff.shape[0]
        batch_mean = np.mean(X_batch_rff, axis=0)
        
        old_mean = self.mean_rff_stream.copy()
        
        if self.num_points_seen == 0:
            self.mean_rff_stream = batch_mean
            delta = np.zeros_like(batch_mean)
        else:
            alpha = batch_len / (self.num_points_seen + batch_len)
            self.mean_rff_stream = (1 - alpha) * self.mean_rff_stream + alpha * batch_mean
            delta = self.mean_rff_stream - old_mean
            
        self.num_points_seen += batch_len
        return delta

    def _rebuild_herding(self, start_step: int, current_embedding: np.ndarray):
        """
        Continues the herding process from `start_step` to `coreset_size`.
        """
        # Truncate history to the valid prefix
        self.coreset_indices = self.coreset_indices[:start_step]
        self.margins = self.margins[:start_step]
        self.runner_up_indices = self.runner_up_indices[:start_step]
        
        n_candidates = self.buffer_rff_cache.shape[0]
        available = np.ones(n_candidates, dtype=bool)
        
        # Mark already selected in the valid prefix as unavailable
        for idx in self.coreset_indices:
            available[idx] = False

        # Continue Herding
        for k in range(start_step, self.coreset_size):
            if not np.any(available):
                break
                
            residual = self.mean_rff_stream - current_embedding
            scores = self.buffer_rff_cache @ residual
            
            scores_masked = scores.copy()
            scores_masked[~available] = -np.inf
            
            best_idx = int(np.argmax(scores_masked))
            best_score = scores_masked[best_idx]
            
            # Select Second Best (for Margin)
            scores_masked[best_idx] = -np.inf
            second_best_idx = int(np.argmax(scores_masked))
            second_best_score = scores_masked[second_best_idx]
            
            if np.isneginf(second_best_score):
                margin = 9999.0 
                second_best_idx = -1
            else:
                margin = best_score - second_best_score
            
            self.coreset_indices.append(best_idx)
            self.margins.append(margin)
            self.runner_up_indices.append(second_best_idx)
            available[best_idx] = False
            
            if k == 0:
                current_embedding = self.buffer_rff_cache[best_idx]
            else:
                current_embedding = (current_embedding * k + self.buffer_rff_cache[best_idx]) / (k + 1)

    def _manage_buffer_capacity(self, batch_idx: int, batch_len: int):
        """
        Compacts the buffer while preserving the coreset, runner-ups, and orthogonal outliers.
        Syncs buffer_X, buffer_y, and provenance.
        """
        if len(self.buffer_X) <= self.buffer_capacity:
            return

        # 1. Protection Set A: Coreset
        keep_indices = set(self.coreset_indices)
        
        # 2. Protection Set B: Runner Ups
        for runner_up in self.runner_up_indices:
            if runner_up != -1 and len(keep_indices) < self.buffer_capacity:
                keep_indices.add(runner_up)
        
        # 3. Protection Set C: Orthogonal/Residual Max
        if len(keep_indices) < self.buffer_capacity:
            selected_rffs = self.buffer_rff_cache[self.coreset_indices]
            curr_embed = np.mean(selected_rffs, axis=0) if len(selected_rffs) > 0 else np.zeros(self.rff_dim)
            final_residual = self.mean_rff_stream - curr_embed
            
            scores = self.buffer_rff_cache @ final_residual
            
            remaining_candidates = [i for i in range(len(self.buffer_X)) if i not in keep_indices]
            remaining_candidates = np.array(remaining_candidates)
            
            if len(remaining_candidates) > 0:
                rem_scores = scores[remaining_candidates]
                sorted_args = np.argsort(rem_scores)[::-1]
                slots_left = self.buffer_capacity - len(keep_indices)
                top_k_rem = remaining_candidates[sorted_args[:slots_left]]
                keep_indices.update(top_k_rem)

        # Apply Selection
        indices_to_keep = np.array(list(keep_indices), dtype=int)
        indices_to_keep.sort()

        # Update X, Y, Cache, and Provenance
        self.buffer_X = self.buffer_X[indices_to_keep]
        self.buffer_y = self.buffer_y[indices_to_keep]  # <--- SYNC Y HERE
        self.buffer_rff_cache = self.buffer_rff_cache[indices_to_keep]
        self.buffer_provenance = [self.buffer_provenance[i] for i in indices_to_keep]
        
        # Remap History Indices
        old_to_new = {old: new for new, old in enumerate(indices_to_keep)}
        self.coreset_indices = [old_to_new[i] for i in self.coreset_indices if i in old_to_new]
        self.runner_up_indices = [old_to_new[i] if i in old_to_new else -1 for i in self.runner_up_indices]
        
        valid_len = len(self.coreset_indices)
        self.margins = self.margins[:valid_len]
        self.runner_up_indices = self.runner_up_indices[:valid_len]

    def process_batch(self, X_batch_np: np.ndarray, y_batch_np: np.ndarray, batch_idx: int) -> None:
        batch_len = X_batch_np.shape[0]
        if batch_len == 0:
            return

        # 1. RFF Transform
        X_batch_rff = self.sampler.transform(X_batch_np)
        
        # 2. Update Global Mean
        delta = self._update_mean(X_batch_rff)
        
        # 3. Add to Buffer (X and Y)
        if len(self.buffer_X) == 0:
            self.buffer_X = X_batch_np
            self.buffer_y = y_batch_np  # <--- INIT Y
            self.buffer_rff_cache = X_batch_rff
        else:
            self.buffer_X = np.vstack([self.buffer_X, X_batch_np])
            self.buffer_y = np.concatenate([self.buffer_y, y_batch_np]) # <--- APPEND Y
            self.buffer_rff_cache = np.vstack([self.buffer_rff_cache, X_batch_rff])
            
        new_prov = [(batch_idx, i) for i in range(batch_len)]
        self.buffer_provenance.extend(new_prov)

        # 4. Check Margins & Update Coreset
        valid_steps = 0
        if len(self.coreset_indices) > 0:
            current_embedding = np.zeros(self.rff_dim)
            for s in range(len(self.coreset_indices)):
                idx_y = self.coreset_indices[s]
                idx_next = self.runner_up_indices[s]
                
                # Check bounds (safety)
                if idx_y >= len(self.buffer_X) or (idx_next != -1 and idx_next >= len(self.buffer_X)):
                    break
                    
                is_valid = True
                if idx_next != -1:
                    vec_y = self.buffer_rff_cache[idx_y]
                    vec_next = self.buffer_rff_cache[idx_next]
                    # Margin Condition: <x_next - y_s, Delta> <= G_s
                    lhs = np.dot(vec_next - vec_y, delta)
                    if lhs > self.margins[s]:
                        is_valid = False
                
                if is_valid:
                    if s == 0:
                        current_embedding = self.buffer_rff_cache[idx_y]
                    else:
                        current_embedding = (current_embedding * s + self.buffer_rff_cache[idx_y]) / (s + 1)
                    valid_steps += 1
                else:
                    break
        
        # 5. Rebuild if broken
        if valid_steps < self.coreset_size:
            if valid_steps == 0:
                current_embedding = np.zeros(self.rff_dim)
            self._rebuild_herding(start_step=valid_steps, current_embedding=current_embedding)

        # 6. Manage Capacity
        if len(self.buffer_X) > self.buffer_capacity:
            self._manage_buffer_capacity(batch_idx, batch_len)

    def get_final_coreset(self) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
        flat_indices = []
        final_prov = []
        
        for idx in self.coreset_indices:
            b_idx, l_idx = self.buffer_provenance[idx]
            flat_idx = b_idx * self.batch_size + l_idx
            flat_indices.append(flat_idx)
            final_prov.append((b_idx, l_idx))
            
        weights = np.full(len(flat_indices), 1.0 / len(flat_indices))
        return np.array(flat_indices, dtype=int), weights, final_prov
    
    def get_coreset_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Helper to retrieve the actual X and y data of the coreset.
        """
        if not self.coreset_indices:
            return np.empty((0, self.feature_dim)), np.empty(0)
            
        return self.buffer_X[self.coreset_indices], self.buffer_y[self.coreset_indices]

    def print_coreset_provenance(self) -> None:
        flat_indices, weights, provenance = self.get_final_coreset()
        print(f"--- Final Coreset Provenance (Adaptive Herding) ---")
        print(f"Points seen: {self.num_points_seen}")
        print(f"{'Global Index':<15} {'Provenance':<20} {'Weight':<10} {'Label':<10}")
        print("-" * 55)
        
        # Access labels for printing
        labels = self.buffer_y[self.coreset_indices] if len(self.coreset_indices) > 0 else []
        
        for i in range(len(provenance)):
            prov_str = f"(Batch {provenance[i][0]}, Idx {provenance[i][1]})"
            lbl = labels[i] if i < len(labels) else "N/A"
            print(f"{flat_indices[i]:<15} {prov_str:<20} {weights[i]:<10.6f} {lbl:<10}")