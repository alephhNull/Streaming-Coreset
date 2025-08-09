from typing import List, Tuple

import numpy as np
from sklearn.kernel_approximation import RBFSampler
from streamers.abstract_streamer import AbstractStreamingCoreset

def weighted_kernel_herding_rff(mu_pi, X_candidate, sampler, m):
    """
    Fully-corrective weighted kernel herding in RFF space using a provided RBFSampler.
    
    Args:
      mu_pi (np.ndarray): The (D,) Euclidean mean embedding of the dataset in RFF space.
      X_candidate (np.ndarray): The (n_candidates, d) candidate points to pick from (original input space).
      sampler (RBFSampler): A fitted sklearn RBFSampler used to compute mu_pi (ensures consistent mapping).
      m (int): The desired size of the coreset.
    
    Returns:
      tuple: A tuple containing:
        - np.ndarray: An array of shape (m,) with the indices into X_candidate.
        - np.ndarray: An array of shape (m,) with the final weights for the coreset points.
    """
    n_candidates, d_orig = X_candidate.shape
    D = mu_pi.shape[0] # Dimension of the RFF space

    X_candidate_rff = sampler.transform(X_candidate)
    selected_indices = []
    current_embedding = np.zeros(D)

    for k in range(m):
        residual = mu_pi - current_embedding
        search_values = X_candidate_rff @ residual
        search_values[selected_indices] = -np.inf
        
        best_x_idx = np.argmax(search_values)
        selected_indices.append(best_x_idx)

        coreset_rff = X_candidate_rff[selected_indices]
        K_rff = coreset_rff @ coreset_rff.T
        z_rff = coreset_rff @ mu_pi
        weights = np.linalg.pinv(K_rff) @ z_rff
        
        current_embedding = weights @ coreset_rff

    final_weights = weights
    return np.array(selected_indices), final_weights


class WKHStreamingCoreset(AbstractStreamingCoreset):
    """
    Implements a streaming coreset selection algorithm using Weighted Kernel Herding (WKH)
    in Random Fourier Feature (RFF) space.

    This class maintains a buffer of points. When a new batch of data causes the
    buffer to exceed its capacity, it uses WKH to select a smaller, representative
    subset of points from the combined buffer and new batch. This selection is
    guided by a running mean of the entire stream's RFF embedding.
    """

    def __init__(self, coreset_size: int, buffer_capacity: int, sampler: RBFSampler, batch_size: int):
        """
        Args:
            coreset_size (int): The target size of the coreset to be selected.
            buffer_capacity (int): The maximum number of points to store in memory.
            sampler (RBFSampler): A pre-fitted RBFSampler for consistent feature mapping.
            batch_size (int): The size of incoming data batches, used for index calculation.
        """
        assert coreset_size <= buffer_capacity, "Coreset size must be less than or equal to buffer capacity."
        
        self.coreset_size = coreset_size
        self.buffer_capacity = buffer_capacity
        self.sampler = sampler
        self.batch_size = batch_size
        
        # Initialize state variables
        self.rff_dim = sampler.n_components
        self.feature_dim = sampler.random_weights_.shape[0]

        # Buffers to hold current data and its origin
        self.buffer_X = np.empty((0, self.feature_dim))
        self.buffer_y = np.empty((0,))
        self.buffer_provenance: List[Tuple[int, int]] = []

        # State for tracking the global stream properties
        self.mean_rff_full_stream = np.zeros(self.rff_dim)
        self.num_points_seen = 0
        self._finalized = False # Flag to indicate if the final coreset has been computed

    def process_batch(self, X_batch_np: np.ndarray, y_batch_np: np.ndarray, batch_idx: int) -> None:
        """
        Processes a new batch of data from the stream.
        """
        if self._finalized:
            print("Warning: Coreset has been finalized. Ignoring new batch.")
            return

        batch_len = X_batch_np.shape[0]

        # 1. Update the running mean embedding of the entire stream
        X_batch_rff = self.sampler.transform(X_batch_np)
        current_batch_mean = np.mean(X_batch_rff, axis=0)

        if self.num_points_seen == 0:
            self.mean_rff_full_stream = current_batch_mean
        else:
            # Weighted average to update the running mean
            alpha = batch_len / (self.num_points_seen + batch_len)
            self.mean_rff_full_stream = (1 - alpha) * self.mean_rff_full_stream + alpha * current_batch_mean
        
        self.num_points_seen += batch_len

        # 2. Check if the buffer will overflow and run reduction if necessary
        if len(self.buffer_X) + batch_len > self.buffer_capacity:
            # Create a candidate pool from the buffer and the new batch
            X_candidate = np.vstack([self.buffer_X, X_batch_np])
            y_candidate = np.concatenate([self.buffer_y, y_batch_np])
            
            new_provenance = [(batch_idx, i) for i in range(batch_len)]
            provenance_candidate = self.buffer_provenance + new_provenance

            # Run WKH to select a new, smaller buffer of size `coreset_size`
            selected_indices, _ = weighted_kernel_herding_rff(
                mu_pi=self.mean_rff_full_stream,
                X_candidate=X_candidate,
                sampler=self.sampler,
                m=self.coreset_size
            )
            
            # Update the buffer with the selected coreset points
            self.buffer_X = X_candidate[selected_indices]
            self.buffer_y = y_candidate[selected_indices]
            self.buffer_provenance = [provenance_candidate[i] for i in selected_indices]
        else:
            # Buffer has space, just append the new batch
            self.buffer_X = np.vstack([self.buffer_X, X_batch_np])
            self.buffer_y = np.concatenate([self.buffer_y, y_batch_np])
            
            new_provenance = [(batch_idx, i) for i in range(batch_len)]
            self.buffer_provenance.extend(new_provenance)

    def _finalize_coreset(self):
        """Internal method to run the final reduction on the buffer."""
        if self._finalized:
            return

        # If the final buffer is larger than the target coreset size, run one last reduction
        if len(self.buffer_X) > self.coreset_size:
            selected_indices, _ = weighted_kernel_herding_rff(
                mu_pi=self.mean_rff_full_stream,
                X_candidate=self.buffer_X,
                sampler=self.sampler,
                m=self.coreset_size
            )
            self.buffer_X = self.buffer_X[selected_indices]
            self.buffer_y = self.buffer_y[selected_indices]
            self.buffer_provenance = [self.buffer_provenance[i] for i in selected_indices]
        
        self._finalized = True

    def get_final_coreset(self) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
        """
        Finalizes and returns the coreset.
        """
        self._finalize_coreset()
        
        # Calculate global indices from provenance
        flat_indices = np.array(
            [p[0] * self.batch_size + p[1] for p in self.buffer_provenance],
            dtype=int
        )
        
        # Generate uniform weights as per the abstract class requirements
        num_coreset_points = len(self.buffer_X)
        if num_coreset_points == 0:
            weights = np.array([])
        else:
            weights = np.ones(num_coreset_points) / num_coreset_points
        
        return flat_indices, weights, self.buffer_provenance

    def print_coreset_provenance(self) -> None:
        """
        Prints the details of each point in the final coreset.
        """
        flat_indices, weights, provenance = self.get_final_coreset()
        
        print("--- Final Coreset Provenance ---")
        if not provenance:
            print("Coreset is empty.")
            return
            
        print(f"{'Global Index':<15} {'Provenance':<20} {'Weight':<10}")
        print("-" * 45)
        for i in range(len(provenance)):
            prov_str = f"(Batch {provenance[i][0]}, Idx {provenance[i][1]})"
            print(f"{flat_indices[i]:<15} {prov_str:<20} {weights[i]:<10.6f}")
        print(f"\nTotal points in coreset: {len(provenance)}")
        print(f"Total points seen in stream: {self.num_points_seen}")