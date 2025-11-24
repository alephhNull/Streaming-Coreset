from typing import List, Tuple
import numpy as np
from sklearn.kernel_approximation import RBFSampler
from streamers.abstract_streamer import AbstractStreamingCoreset

def kernel_herding_rff(mu_pi: np.ndarray, X_candidate: np.ndarray, sampler: RBFSampler, m: int):
    """
    Simple (unweighted) kernel herding in RFF space.
    Greedy selection: at step k pick x maximizing <x_rff, mu_pi - current_embedding>.
    Current embedding is the simple average of selected RFF features.
    Returns selected indices and uniform weights (1/len(selected)).
    """
    n_candidates = X_candidate.shape[0]
    if n_candidates == 0:
        return np.array([], dtype=int), np.array([])

    m = min(m, n_candidates)
    X_candidate_rff = sampler.transform(X_candidate)
    D = mu_pi.shape[0]

    selected_indices = []
    current_embedding = np.zeros(D)

    # Keep a mask of available indices
    available = np.ones(n_candidates, dtype=bool)

    for k in range(m):
        residual = mu_pi - current_embedding
        # Compute scores = <x_rff, residual>
        scores = X_candidate_rff @ residual
        # invalidate already selected
        scores[~available] = -np.inf
        best_idx = int(np.argmax(scores))
        selected_indices.append(best_idx)
        available[best_idx] = False

        # update current embedding to be average of selected features
        if k == 0:
            current_embedding = X_candidate_rff[best_idx].copy()
        else:
            current_embedding = (current_embedding * k + X_candidate_rff[best_idx]) / (k + 1)

    final_weights = np.full(len(selected_indices), 1.0 / len(selected_indices))
    return np.array(selected_indices, dtype=int), final_weights


class KernelHerdingStreamer(AbstractStreamingCoreset):
    """
    Streaming coreset using unweighted RFF-based kernel herding.
    - selects `coreset_size` points when buffer overflows
    - returned weights are uniform (1 / coreset_size)
    """

    def __init__(self, coreset_size: int, buffer_capacity: int, sampler: RBFSampler, batch_size: int):
        assert coreset_size <= buffer_capacity, "Coreset size must be <= buffer capacity."

        self.coreset_size = coreset_size
        self.buffer_capacity = buffer_capacity
        self.sampler = sampler
        self.batch_size = batch_size

        # dims
        self.rff_dim = sampler.n_components
        # feature dimension of raw X
        # RBFSampler stores random_weights_ of shape (n_features, n_components)
        self.feature_dim = sampler.random_weights_.shape[0]

        # buffer and provenance
        self.buffer_X = np.empty((0, self.feature_dim))
        self.buffer_provenance: List[Tuple[int, int]] = []
        # weights over buffer points (kept uniform across the buffer)
        self.buffer_weights = np.empty(0, dtype=float)

        # stream-level running mean in RFF space
        self.mean_rff_full_stream = np.zeros(self.rff_dim)
        self.num_points_seen = 0
        self._finalized = False

    def process_batch(self, X_batch_np: np.ndarray, y_batch_np: np.ndarray, batch_idx: int) -> None:
        if self._finalized:
            print("Warning: Coreset has been finalized. Ignoring new batch.")
            return

        batch_len = X_batch_np.shape[0]
        if batch_len == 0:
            return

        # 1) update running mean embedding in RFF space
        X_batch_rff = self.sampler.transform(X_batch_np)
        batch_mean = np.mean(X_batch_rff, axis=0)

        if self.num_points_seen == 0:
            self.mean_rff_full_stream = batch_mean
        else:
            alpha = batch_len / (self.num_points_seen + batch_len)
            self.mean_rff_full_stream = (1 - alpha) * self.mean_rff_full_stream + alpha * batch_mean

        self.num_points_seen += batch_len

        # 2) buffer management
        if len(self.buffer_X) + batch_len > self.buffer_capacity:
            # overflow -> build candidate pool and reduce to coreset_size
            X_candidate = np.vstack([self.buffer_X, X_batch_np]) if len(self.buffer_X) > 0 else X_batch_np.copy()

            # provenance: existing buffer provenance first, then new batch provenance
            new_prov = [(batch_idx, i) for i in range(batch_len)]
            prov_candidate = self.buffer_provenance + new_prov

            selected_indices, final_weights = kernel_herding_rff(
                mu_pi=self.mean_rff_full_stream,
                X_candidate=X_candidate,
                sampler=self.sampler,
                m=self.coreset_size
            )

            # update buffer to selected coreset
            self.buffer_X = X_candidate[selected_indices]
            self.buffer_provenance = [prov_candidate[i] for i in selected_indices]
            self.buffer_weights = final_weights
        else:
            # no overflow -> just append batch and keep uniform weights across buffer
            self.buffer_X = np.vstack([self.buffer_X, X_batch_np])
            self.buffer_provenance.extend([(batch_idx, i) for i in range(batch_len)])
            n = len(self.buffer_X)
            # reset to uniform weights summing to 1
            self.buffer_weights = np.full(n, 1.0 / n)

    def _finalize_coreset(self):
        if self._finalized:
            return

        if len(self.buffer_X) > self.coreset_size:
            selected_indices, final_weights = kernel_herding_rff(
                mu_pi=self.mean_rff_full_stream,
                X_candidate=self.buffer_X,
                sampler=self.sampler,
                m=self.coreset_size
            )
            self.buffer_X = self.buffer_X[selected_indices]
            self.buffer_provenance = [self.buffer_provenance[i] for i in selected_indices]
            self.buffer_weights = final_weights
        else:
            # if buffer is already <= coreset_size, make weights uniform
            if len(self.buffer_X) > 0:
                self.buffer_weights = np.full(len(self.buffer_X), 1.0 / len(self.buffer_X))

        self._finalized = True

    def get_final_coreset(self) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
        self._finalize_coreset()

        flat_indices = np.array(
            [p[0] * self.batch_size + p[1] for p in self.buffer_provenance],
            dtype=int
        )
        weights = self.buffer_weights
        return flat_indices, weights, self.buffer_provenance

    def print_coreset_provenance(self) -> None:
        flat_indices, weights, provenance = self.get_final_coreset()
        print("--- Final Coreset Provenance (KernelHerdingStreamer) ---")
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
