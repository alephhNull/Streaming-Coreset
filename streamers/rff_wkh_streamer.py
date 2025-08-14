from typing import List, Tuple
from qpsolvers import solve_qp
import numpy as np
from sklearn.kernel_approximation import RBFSampler
from streamers.abstract_streamer import AbstractStreamingCoreset

def weighted_kernel_herding_rff_qp(mu_pi, X_candidate, sampler, m):
    """
    Fully-corrective weighted kernel herding in RFF space using a QP solver 
    (non-negative weights, sum to 1).
    """
    n_candidates = X_candidate.shape[0]
    D = mu_pi.shape[0]  # RFF feature dimension

    X_candidate_rff = sampler.transform(X_candidate)
    selected_indices = []
    current_embedding = np.zeros(D)

    for k in range(m):
        residual = mu_pi - current_embedding
        search_values = X_candidate_rff @ residual
        search_values[selected_indices] = -np.inf
        best_x_idx = np.argmax(search_values)
        selected_indices.append(best_x_idx)

        # RFF features of selected points
        coreset_rff = X_candidate_rff[selected_indices]

        # QP matrices
        K_rff = coreset_rff @ coreset_rff.T  # m x m Gram
        z_rff = coreset_rff @ mu_pi          # m vector

        # QP form: min 0.5 x^T P x + q^T x
        eps = 1e-10
        P = (K_rff + K_rff.T) / 2.0 + eps * np.eye(K_rff.shape[0])
        q = -z_rff

        # Equality constraint: sum w = 1
        A = np.ones((1, len(selected_indices)))
        b = np.array([1.0])

        # Inequality: w >= 0
        G = -np.eye(len(selected_indices))
        h = np.zeros(len(selected_indices))

        weights = solve_qp(P, q, G, h, A, b, solver="quadprog")

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
        self.buffer_weights = np.empty(0, dtype=float)
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
            
            new_provenance = [(batch_idx, i) for i in range(batch_len)]
            provenance_candidate = self.buffer_provenance + new_provenance

            # Run WKH to select a new, smaller buffer of size `coreset_size`
            selected_indices, final_weights = weighted_kernel_herding_rff_qp(
                mu_pi=self.mean_rff_full_stream,
                X_candidate=X_candidate,
                sampler=self.sampler,
                m=self.coreset_size
            )
            
            # Update the buffer with the selected coreset points
            self.buffer_X = X_candidate[selected_indices]
            self.buffer_provenance = [provenance_candidate[i] for i in selected_indices]
            self.buffer_weights = final_weights
        else:
            # Buffer has space, just append the new batch
            alpha = batch_len / self.num_points_seen
            self.buffer_weights *= (1 - alpha)
            new_weights = np.full(batch_len, alpha / batch_len)
            self.buffer_weights = np.concatenate((self.buffer_weights, new_weights))
            
            self.buffer_X = np.vstack([self.buffer_X, X_batch_np])
            self.buffer_provenance.extend([(batch_idx, i) for i in range(batch_len)])

    def _finalize_coreset(self):
        """Internal method to run the final reduction on the buffer."""
        if self._finalized:
            return

        # If the final buffer is larger than the target coreset size, run one last reduction
        if len(self.buffer_X) > self.coreset_size:
            selected_indices, final_weights = weighted_kernel_herding_rff_qp(
                mu_pi=self.mean_rff_full_stream,
                X_candidate=self.buffer_X,
                sampler=self.sampler,
                m=self.coreset_size
            )
            self.buffer_X = self.buffer_X[selected_indices]
            self.buffer_provenance = [self.buffer_provenance[i] for i in selected_indices]
            self.buffer_weights = final_weights
        
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
        
        weights = self.buffer_weights
        
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