from typing import List, Tuple, Optional
import numpy as np
from sklearn.kernel_approximation import RBFSampler
from streamers.abstract_streamer import AbstractStreamingCoreset


def weighted_kernel_herding_rff_sampler(
    mu_pi: np.ndarray,
    X_candidate: np.ndarray,
    sampler: RBFSampler,
    m: int,
    random_state: Optional[int] = None,
    max_iters: int = 20000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sampling-based replacement for the QP-based fully-corrective WKH.

    Instead of solving a quadratic program to compute non-negative weights that
    sum to one, this routine repeatedly *samples with replacement* from the
    candidate pool using probabilities proportional to the (non-negative)
    alignment of each candidate with the current residual (mu_pi - current_embedding).

    Sampling increments a discrete count for the chosen candidate. We continue
    until exactly `m` distinct candidates have been selected at least once. The
    final weights are the normalized counts among the selected candidates
    (i.e. weight = count / total_counts_over_selected).

    This procedure merges repeated samples of the same candidate by increasing
    its integer count, and therefore increases its eventual weight.

    Args:
        mu_pi: (D,) target mean in RFF space.
        X_candidate: (n_candidates, feature_dim) original inputs (not RFF).
        sampler: a fitted RBFSampler instance (used to map X_candidate -> RFF).
        m: desired number of distinct selected points in the coreset.
        random_state: optional integer RNG seed.
        max_iters: maximum number of sampling steps before forced termination.

    Returns:
        selected_indices: (m,) indices of distinct chosen candidates (into X_candidate).
        final_weights: (m,) weights corresponding to the chosen indices (sum to 1).
    """
    rng = np.random.default_rng(random_state)

    X_candidate_rff = sampler.transform(X_candidate)
    n_candidates = X_candidate_rff.shape[0]

    assert m <= n_candidates, "m must be <= number of candidates"

    counts = np.zeros(n_candidates, dtype=int)
    total_counts = 0

    # Current embedding (weighted average of chosen candidates in RFF space)
    current_embedding = np.zeros_like(mu_pi)

    it = 0
    # Keep sampling until exactly m distinct indices have been chosen
    while np.count_nonzero(counts) < m and it < max_iters:
        it += 1

        # Residual and alignment scores
        residual = mu_pi - current_embedding
        scores = X_candidate_rff @ residual  # n_candidates

        # We only allow non-negative alignment to contribute to probabilities
        pos_scores = np.maximum(scores, 0.0)

        # If all scores are non-positive, fall back to selecting from the
        # candidates not yet selected using uniform probability
        if pos_scores.sum() <= 0:
            not_selected = np.where(counts == 0)[0]
            if not_selected.size == 0:
                # Everything already selected but not reached m distinct (shouldn't happen)
                idx = rng.integers(0, n_candidates)
            else:
                idx = rng.choice(not_selected)
        else:
            probs = pos_scores / pos_scores.sum()
            # sample a single index with replacement
            idx = rng.choice(n_candidates, p=probs)

        counts[idx] += 1
        total_counts += 1

        # Update embedding as weighted average of selected candidates' RFF features
        if total_counts > 0:
            weights_over_all = counts / counts.sum()
            current_embedding = weights_over_all @ X_candidate_rff

    if it >= max_iters:
        # If we hit the max iters, try to salvage: pick the top-m distinct by
        # either count (prefer) or alignment
        distinct_idx = np.where(counts > 0)[0]
        if distinct_idx.size < m:
            # fill remaining slots by best-aligned unseen candidates
            residual = mu_pi - current_embedding
            scores = X_candidate_rff @ residual
            unseen = np.where(counts == 0)[0]
            fill = unseen[np.argsort(-scores[unseen])[: (m - distinct_idx.size)]]
            for f in fill:
                counts[f] = 1
            total_counts = counts.sum()

    # Now extract exactly m distinct indices
    selected_indices = np.where(counts > 0)[0]

    if selected_indices.size > m:
        # Trim to top-m by counts, breaking ties by alignment score
        counts_sel = counts[selected_indices]
        # alignment score for tie-breaking
        residual = mu_pi - current_embedding
        align = (X_candidate_rff[selected_indices] @ residual)
        order = np.lexsort(( -align, -counts_sel ))  # prefer larger count, then larger align
        selected_indices = selected_indices[order[:m]]
    elif selected_indices.size < m:
        # Rare: still fewer than m, fill by highest alignment among unseen
        unseen = np.where(counts == 0)[0]
        residual = mu_pi - current_embedding
        scores = X_candidate_rff @ residual
        fill = unseen[np.argsort(-scores[unseen])[: (m - selected_indices.size)]]
        for f in fill:
            counts[f] = 1
        selected_indices = np.where(counts > 0)[0]

    # Final weights: normalized counts among the selected indices
    sel_counts = counts[selected_indices].astype(float)
    final_weights = sel_counts / sel_counts.sum()

    return np.array(selected_indices, dtype=int), final_weights


class MultiSamplerWKHStreamingCoreset(AbstractStreamingCoreset):
    """Streaming coreset that uses the sampling-based WKH replacement.

    Behavior matches the structure of the original WKHStreamingCoreset you
    provided but replaces the QP fully-correction step with the cheaper
    `weighted_kernel_herding_rff_sampler` routine.
    """

    def __init__(self, coreset_size: int, buffer_capacity: int, sampler: RBFSampler, batch_size: int, random_state: Optional[int] = None):
        assert coreset_size <= buffer_capacity, "coreset_size must be <= buffer_capacity"

        self.coreset_size = coreset_size
        self.buffer_capacity = buffer_capacity
        self.sampler = sampler
        self.batch_size = batch_size
        self.random_state = random_state

        self.rff_dim = sampler.n_components
        self.feature_dim = sampler.random_weights_.shape[0]

        self.buffer_X = np.empty((0, self.feature_dim))
        self.buffer_weights = np.empty(0, dtype=float)
        self.buffer_provenance: List[Tuple[int, int]] = []

        self.mean_rff_full_stream = np.zeros(self.rff_dim)
        self.num_points_seen = 0
        self._finalized = False

    def process_batch(self, X_batch_np: np.ndarray, y_batch_np: np.ndarray, batch_idx: int) -> None:
        if self._finalized:
            print("Warning: Coreset has been finalized. Ignoring new batch.")
            return

        batch_len = X_batch_np.shape[0]

        # Update running mean
        X_batch_rff = self.sampler.transform(X_batch_np)
        current_batch_mean = np.mean(X_batch_rff, axis=0)

        if self.num_points_seen == 0:
            self.mean_rff_full_stream = current_batch_mean
        else:
            alpha = batch_len / (self.num_points_seen + batch_len)
            self.mean_rff_full_stream = (1 - alpha) * self.mean_rff_full_stream + alpha * current_batch_mean

        self.num_points_seen += batch_len

        # Buffer overflow check
        if len(self.buffer_X) + batch_len > self.buffer_capacity:
            X_candidate = np.vstack([self.buffer_X, X_batch_np])
            provenance_candidate = self.buffer_provenance + [(batch_idx, i) for i in range(batch_len)]

            selected_indices, final_weights = weighted_kernel_herding_rff_sampler(
                mu_pi=self.mean_rff_full_stream,
                X_candidate=X_candidate,
                sampler=self.sampler,
                m=self.coreset_size,
                random_state=self.random_state,
            )

            # Update buffer with selected points
            self.buffer_X = X_candidate[selected_indices]
            self.buffer_provenance = [provenance_candidate[i] for i in selected_indices]
            self.buffer_weights = final_weights
        else:
            # Append to buffer and appropriately reweight
            if self.num_points_seen - batch_len == 0:
                # buffer was empty and this is the first batch
                self.buffer_X = X_batch_np.copy()
                self.buffer_weights = np.full(batch_len, 1.0 / batch_len)
                self.buffer_provenance = [(batch_idx, i) for i in range(batch_len)]
            else:
                # decay existing weights proportionally and append new uniform weights
                alpha = batch_len / self.num_points_seen
                if self.buffer_weights.size > 0:
                    self.buffer_weights *= (1 - alpha)
                new_weights = np.full(batch_len, alpha / batch_len)
                self.buffer_weights = np.concatenate((self.buffer_weights, new_weights))

                self.buffer_X = np.vstack([self.buffer_X, X_batch_np])
                self.buffer_provenance.extend([(batch_idx, i) for i in range(batch_len)])

    def _finalize_coreset(self):
        if self._finalized:
            return

        if len(self.buffer_X) > self.coreset_size:
            selected_indices, final_weights = weighted_kernel_herding_rff_sampler(
                mu_pi=self.mean_rff_full_stream,
                X_candidate=self.buffer_X,
                sampler=self.sampler,
                m=self.coreset_size,
                random_state=self.random_state,
            )
            self.buffer_X = self.buffer_X[selected_indices]
            self.buffer_provenance = [self.buffer_provenance[i] for i in selected_indices]
            self.buffer_weights = final_weights

        self._finalized = True

    def get_final_coreset(self) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
        self._finalize_coreset()

        flat_indices = np.array([p[0] * self.batch_size + p[1] for p in self.buffer_provenance], dtype=int)
        weights = self.buffer_weights
        return flat_indices, weights, self.buffer_provenance

    def print_coreset_provenance(self) -> None:
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
