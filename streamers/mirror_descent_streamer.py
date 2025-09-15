# """
# MirrorDescentHerdingStreamer

# A simplified streaming coreset selector that mirrors the behaviour of the
# provided FairKernelHerdingStreamer but removes all fairness-related logic.

# Key behaviour differences from the original fair implementation:
# - No sensitive attributes anywhere (no `sensitive_batch` parameter).
# - Selection is driven purely by MMD minimization (greedy weighted kernel
#   herding step that evaluates the MMD objective when considering additions).
# - Weight optimization on the selected support uses Mirror Descent (Exponentiated
#   Gradient) constrained to the probability simplex (no fairness constraints).

# Usage (short):
# - Instantiate with an already-fitted `RBFSampler` and appropriate sizes.
# - Call `process_batch(X_batch_np, y_batch_np, batch_idx)` for each arriving batch.
# - Call `get_final_coreset()` to obtain final indices, weights and provenance.

# This file intentionally contains only the functions/classes required for the
# pure-MMD mirror-descent herding workflow.
# """

# from typing import List, Tuple, Dict, Any, Optional
# import numpy as np
# import torch
# from sklearn.kernel_approximation import RBFSampler
# from scipy.optimize import minimize

# # NOTE: AbstractStreamingCoreset must be available in the import path
# from streamers.abstract_streamer import AbstractStreamingCoreset


# def rbf_kernel_np(X, Y=None, sigma=1.0):
#     X = np.asarray(X)
#     Y = X if Y is None else np.asarray(Y)
#     XX = np.sum(X**2, axis=1)[:, None]
#     YY = np.sum(Y**2, axis=1)[None, :]
#     D2 = np.maximum(XX + YY - 2.0 * (X @ Y.T), 0.0)
#     return np.exp(-D2 / (2.0 * sigma**2))



# def weighted_kernel_herding_greedy(X_np, m, sigma, ridge=1e-8, verbose=False):
#     """
#     Greedy weighted kernel-herding style selection of m indices from X_np.
#     At each step we consider adding each remaining candidate j and compute the
#     MMD objective of the set S U {j} where the weights on S U {j} are the solution
#     of the linear system (K_SS + ridge I) w = K_XS.mean(axis=0).

#     Returns: selected_indices (np.array), K_full (kernel on X_np), selection_history
#     """
#     N = X_np.shape[0]
#     K = rbf_kernel_np(X_np, X_np, sigma=sigma)
#     mu_pi = K.mean(axis=0)
#     selected = []
#     remaining = set(range(N))

#     selection_history = {'step': [], 'mmd': []}

#     for t in range(m):
#         best_idx, best_score = -1, np.inf

#         for j in remaining:
#             S_candidate = selected + [j]
#             S_arr = np.array(S_candidate, dtype=int)
#             K_SS = K[np.ix_(S_arr, S_arr)]
#             K_XS = K[:, S_arr]
#             z = None
#             try:
#                 z = np.linalg.solve(K_SS + ridge * np.eye(len(S_arr)), K_XS.mean(axis=0))
#             except np.linalg.LinAlgError:
#                 z = np.linalg.pinv(K_SS + ridge * np.eye(len(S_arr))) @ K_XS.mean(axis=0)

#             residual = (K_XS @ z) - mu_pi
#             score = float(residual @ residual)

#             if score < best_score:
#                 best_score = score
#                 best_idx = j

#         selected.append(best_idx)
#         remaining.remove(best_idx)

#         # compute current MMD on the selected set using QP or analytical approx
#         S_arr = np.array(selected, dtype=int)
#         K_SS = K[np.ix_(S_arr, S_arr)]
#         k_S = K[:, S_arr].mean(axis=0)

#         # compute weights on S via simple linear solve then normalize
#         try:
#             w_sub = np.linalg.solve(K_SS + ridge * np.eye(len(S_arr)), k_S)
#         except np.linalg.LinAlgError:
#             w_sub = np.linalg.pinv(K_SS + ridge * np.eye(len(S_arr))) @ k_S

#         if w_sub.sum() <= 0:
#             w_sub = np.ones_like(w_sub) / float(len(w_sub))
#         else:
#             w_sub = np.maximum(w_sub, 0.0)
#             w_sub = w_sub / (w_sub.sum() + 1e-12)

#         mmd_val = float(w_sub @ (K_SS @ w_sub) - 2.0 * (w_sub @ k_S) + float(K.mean()))

#         selection_history['step'].append(len(selected))
#         selection_history['mmd'].append(mmd_val)

#         if verbose:
#             print(f"[sel {t+1}/{m}] picked {best_idx} | mmd={mmd_val:.6g}")

#     return np.array(selected, dtype=int), K, selection_history


# def baseline_mirror_descent(P_tensor, sigma, m, md_iterations=200, eta=0.1, ridge=1e-8, verbose=False):
#     """
#     1) Select coreset S with weighted_kernel_herding_greedy (pure MMD-driven greedy).
#     2) Optimize weights on the selected support using Mirror Descent (EG) on the
#        simplex minimizing the MMD objective: z^T K_SS z - 2 z^T k_S + const.

#     Returns: (w_full, history)
#     """
#     P_np = P_tensor.cpu().numpy()

#     S, K_full, selection_history = weighted_kernel_herding_greedy(P_np, m, sigma, ridge=ridge, verbose=verbose)

#     K_SS = K_full[np.ix_(S, S)].astype(float)
#     k_S = K_full[:, S].mean(axis=0).astype(float)

#     mu_const = float(K_full.mean())

#     m_sub = len(S)
#     eps = 1e-12

#     # initialize uniform on subset simplex
#     z = np.ones(m_sub, dtype=float) / float(m_sub)

#     history = {'mmd': [], 'selection_history': selection_history}

#     for it in range(md_iterations):
#         # gradient of MMD wrt z: 2*K_SS*z - 2*k_S
#         grad = 2.0 * (K_SS @ z) - 2.0 * k_S

#         # EG update (log-space stable)
#         log_z = np.log(z + eps) - eta * grad
#         log_z = log_z - log_z.max()
#         tilde = np.exp(log_z)
#         tilde = tilde / (tilde.sum() + eps)

#         z = tilde

#         mmd_val = float(z @ K_SS @ z - 2.0 * (z @ k_S) + mu_const)
#         history['mmd'].append(mmd_val)

#         if verbose and (it % max(1, md_iterations // 5) == 0):
#             print(f"[MD it {it+1}/{md_iterations}] MMD={mmd_val:.6g}")

#     # After MD: return full vector with subset weights (z)
#     w_full = np.zeros(P_np.shape[0], dtype=float)
#     w_full[S] = np.maximum(z, 0.0)

#     # ensure subset weights sum to 1 (numerical)
#     ssum = w_full[S].sum()
#     if ssum > 0:
#         w_full[S] = w_full[S] / (ssum + 1e-12)
#     else:
#         # fallback uniform
#         if len(S) > 0:
#             w_full[S] = np.ones(len(S), dtype=float) / float(len(S))

#     return w_full, history


# class MirrorDescentHerdingStreamer(AbstractStreamingCoreset):
#     """
#     Streaming coreset selector that uses a pure-MMD greedy herding selection
#     followed by mirror-descent weight optimization on the selected support.

#     Differences from FairKernelHerdingStreamer:
#     - No fairness attributes or penalties.
#     - process_batch(...) does not accept sensitive attributes.
#     - Weight optimization is plain MMD minimization subject only to the simplex.
#     """

#     def __init__(
#         self,
#         coreset_size: int,
#         buffer_capacity: int,
#         sampler: RBFSampler,
#         batch_size: int,
#         sigma: float,
#         md_iterations: int = 200,
#         eta: float = 0.1,
#         ridge: float = 1e-8,
#         verbose: bool = False,
#     ) -> None:
#         assert coreset_size <= buffer_capacity, "coreset_size must be <= buffer_capacity"

#         self.coreset_size = coreset_size
#         self.buffer_capacity = buffer_capacity
#         self.sampler = sampler
#         self.batch_size = batch_size
#         self.sigma = sigma
#         self.md_iterations = md_iterations
#         self.eta = eta
#         self.ridge = ridge
#         self.verbose = verbose

#         # Derived dims
#         self.rff_dim = sampler.n_components
#         try:
#             self.feature_dim = sampler.random_weights_.shape[1]
#         except Exception:
#             self.feature_dim = None

#         # Buffer storage
#         self.buffer_X = np.empty((0, self.feature_dim)) if self.feature_dim is not None else np.empty((0, 0))
#         self.buffer_y = np.empty(0, dtype=int)
#         self.buffer_weights = np.empty(0, dtype=float)

#         # provenance: list[(batch_idx, local_idx)] for each buffered point
#         self.buffer_provenance: List[Tuple[int, int]] = []

#         # Running mean RFF and counters
#         self.mean_rff_full_stream = np.zeros(self.rff_dim)
#         self.num_points_seen = 0
#         self._finalized = False

#         # Last selection diagnostics
#         self.last_history: Optional[Dict[str, Any]] = None

#     def process_batch(
#         self,
#         X_batch_np: np.ndarray,
#         y_batch_np: np.ndarray,
#         batch_idx: int,
#     ) -> None:
#         """
#         Process an incoming batch. No sensitive attributes are used.
#         """
#         if self._finalized:
#             if self.verbose:
#                 print("Warning: streamer finalized, ignoring incoming batch")
#             return

#         batch_len = X_batch_np.shape[0]

#         # Update running mean in RFF space
#         X_batch_rff = self.sampler.transform(X_batch_np)
#         current_batch_mean = np.mean(X_batch_rff, axis=0)

#         if self.num_points_seen == 0:
#             self.mean_rff_full_stream = current_batch_mean.copy()
#         else:
#             alpha = batch_len / float(self.num_points_seen + batch_len)
#             self.mean_rff_full_stream = (1 - alpha) * self.mean_rff_full_stream + alpha * current_batch_mean

#         self.num_points_seen += batch_len

#         # Append new batch to buffer (if space)
#         if self.buffer_X.size == 0:
#             # initialize shapes correctly if empty
#             self.buffer_X = np.asarray(X_batch_np).copy()
#         else:
#             self.buffer_X = np.vstack([self.buffer_X, X_batch_np])

#         self.buffer_y = np.concatenate([self.buffer_y, np.asarray(y_batch_np, dtype=int)])

#         # update buffer weights with an exponential/ageing style similar to WKH class
#         if self.buffer_weights.size == 0:
#             new_weights = np.full(batch_len, 1.0 / float(self.num_points_seen))
#             self.buffer_weights = new_weights
#         else:
#             alpha = float(batch_len) / float(self.num_points_seen)
#             self.buffer_weights *= (1 - alpha)
#             new_weights = np.full(batch_len, alpha / float(batch_len))
#             self.buffer_weights = np.concatenate([self.buffer_weights, new_weights])

#         # provenance
#         self.buffer_provenance.extend([(batch_idx, i) for i in range(batch_len)])

#         # If buffer exceeded capacity, run MMD-driven selection on candidate pool
#         if len(self.buffer_X) > self.buffer_capacity:
#             if self.verbose:
#                 print(f"Buffer overflow (size={len(self.buffer_X)} > cap={self.buffer_capacity}) - running MD herding selection")

#             X_candidate = self.buffer_X
#             y_candidate = self.buffer_y
#             provenance_candidate = list(self.buffer_provenance)

#             P_tensor = torch.tensor(X_candidate, dtype=torch.float32)

#             w_full, history = baseline_mirror_descent(
#                 P_tensor,
#                 self.sigma,
#                 self.coreset_size,
#                 md_iterations=self.md_iterations,
#                 eta=self.eta,
#                 ridge=self.ridge,
#                 verbose=self.verbose,
#             )

#             self.last_history = history

#             # Pick selected indices and weights from w_full
#             eps = 1e-12
#             selected_mask = w_full > eps
#             selected_idx_relative = np.where(selected_mask)[0]

#             # If baseline returned no selections due to numerical issues, fall back to top-k by weight
#             if selected_idx_relative.size == 0:
#                 order = np.argsort(-w_full)
#                 selected_idx_relative = order[: self.coreset_size]

#             # If it selected more than needed, take top-k by weight
#             if selected_idx_relative.size > self.coreset_size:
#                 rel_weights = w_full[selected_idx_relative]
#                 top_order = np.argsort(-rel_weights)[: self.coreset_size]
#                 selected_idx_relative = selected_idx_relative[top_order]

#             # Normalize weights of selected
#             sel_weights = w_full[selected_idx_relative]
#             ssum = sel_weights.sum()
#             if ssum > 0:
#                 sel_weights = sel_weights / float(ssum)
#             else:
#                 sel_weights = np.ones(len(selected_idx_relative), dtype=float) / float(len(selected_idx_relative))

#             # Update buffer to the new coreset
#             self.buffer_X = X_candidate[selected_idx_relative]
#             self.buffer_y = y_candidate[selected_idx_relative]
#             self.buffer_weights = sel_weights
#             self.buffer_provenance = [provenance_candidate[i] for i in selected_idx_relative]

#             if self.verbose:
#                 print(f"Selected {len(self.buffer_X)} points into buffer after MD herding selection")

#     def _finalize_coreset(self) -> None:
#         if self._finalized:
#             return

#         # If buffer > target, run final selection on the buffer itself
#         if len(self.buffer_X) > self.coreset_size:
#             X_candidate = self.buffer_X
#             y_candidate = self.buffer_y
#             provenance_candidate = list(self.buffer_provenance)

#             P_tensor = torch.tensor(X_candidate, dtype=torch.float32)

#             w_full, history = baseline_mirror_descent(
#                 P_tensor,
#                 self.sigma,
#                 self.coreset_size,
#                 md_iterations=self.md_iterations,
#                 eta=self.eta,
#                 ridge=self.ridge,
#                 verbose=self.verbose,
#             )

#             self.last_history = history

#             eps = 1e-12
#             sel_mask = w_full > eps
#             sel_idx_rel = np.where(sel_mask)[0]

#             if sel_idx_rel.size == 0:
#                 order = np.argsort(-w_full)
#                 sel_idx_rel = order[: self.coreset_size]

#             if sel_idx_rel.size > self.coreset_size:
#                 rel_weights = w_full[sel_idx_rel]
#                 top_order = np.argsort(-rel_weights)[: self.coreset_size]
#                 sel_idx_rel = sel_idx_rel[top_order]

#             sel_weights = w_full[sel_idx_rel]
#             ssum = sel_weights.sum()
#             if ssum > 0:
#                 sel_weights = sel_weights / float(ssum)
#             else:
#                 sel_weights = np.ones(len(sel_idx_rel), dtype=float) / float(len(sel_idx_rel))

#             self.buffer_X = X_candidate[sel_idx_rel]
#             self.buffer_y = y_candidate[sel_idx_rel]
#             self.buffer_weights = sel_weights
#             self.buffer_provenance = [provenance_candidate[i] for i in sel_idx_rel]

#         self._finalized = True

#     def get_final_coreset(self) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
#         """Finalize (if needed) and return global flat indices, weights and provenance."""
#         self._finalize_coreset()

#         flat_indices = np.array([
#             p[0] * self.batch_size + p[1] for p in self.buffer_provenance
#         ], dtype=int)

#         return flat_indices, self.buffer_weights, list(self.buffer_provenance)

#     def print_coreset_provenance(self) -> None:
#         flat_indices, weights, provenance = self.get_final_coreset()

#         print("--- Final Coreset Provenance (Mirror Descent Herding) ---")
#         if not provenance:
#             print("Coreset is empty.")
#             return

#         print(f"{'Global Index':<15} {'Provenance':<20} {'Weight':<10}")
#         print("-" * 55)
#         for i in range(len(provenance)):
#             prov_str = f"(Batch {provenance[i][0]}, Idx {provenance[i][1]})"
#             print(f"{flat_indices[i]:<15} {prov_str:<20} {weights[i]:<10.6f}")
#         print(f"\nTotal points in coreset: {len(provenance)}")
#         print(f"Total points seen in stream: {self.num_points_seen}")



"""
MirrorDescentHerdingStreamer (RFF version)

- Selection and weight optimization done in RFF space using a provided RBFSampler.
- No `sigma` parameter required anymore.
- We replace the QP solve with exponentiated-gradient (mirror descent) for non-negative simplex weights.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import torch
from sklearn.kernel_approximation import RBFSampler

# NOTE: AbstractStreamingCoreset must be available in the import path
from streamers.abstract_streamer import AbstractStreamingCoreset


from typing import Tuple, List
import numpy as np
# from sklearn.kernel_approximation import RBFSampler  # type: ignore

def weighted_kernel_herding_rff_md_after_selection(
    mu_pi: np.ndarray,
    X_candidate: np.ndarray,
    sampler,
    m: int,
    md_iterations: int = 200,
    eta: float = 0.1,
    ridge: float = 1e-8,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Greedy selection (m points) in RFF space, then a single Mirror Descent (EG)
    reweighting on the final support.

    Selection: at each greedy step we score candidates by <x, residual> where
    residual = mu_pi - current_embedding and current_embedding is the
    equal-weight average of already selected points.

    After selecting m points we run Exponentiated Gradient (Mirror Descent)
    on the simplex restricted to the selected set to obtain final non-negative
    weights summing to 1.

    Returns:
        selected_indices (np.array length <= m), final_weights (np.array length = s)
    """
    # transform to RFF space
    X_candidate_rff = sampler.transform(X_candidate)  # (N, D)
    N, D = X_candidate_rff.shape

    if m <= 0:
        return np.array([], dtype=int), np.array([], dtype=float)

    selected_indices: List[int] = []
    selected_mask = np.zeros(N, dtype=bool)

    # we'll maintain sum of selected features to compute equal-weight average cheaply
    sum_selected_rff = np.zeros(D, dtype=float)

    # initial current embedding is zero
    current_embedding = np.zeros(D, dtype=float)

    for k in range(m):
        residual = mu_pi - current_embedding  # (D,)

        # score each candidate by inner-product with residual
        scores = X_candidate_rff.dot(residual)  # (N,)
        scores[selected_mask] = -np.inf

        best_idx = int(np.argmax(scores))
        # handle case where all remaining candidates have -inf (shouldn't happen unless m>N)
        if scores[best_idx] == -np.inf:
            if verbose:
                print(f"Selection stopped early at k={k}: no remaining candidates.")
            break

        # add to support
        selected_indices.append(best_idx)
        selected_mask[best_idx] = True

        # update running sum and equal-weight average
        sum_selected_rff += X_candidate_rff[best_idx]
        s = len(selected_indices)
        current_embedding = sum_selected_rff / float(s)

        if verbose:
            # quick RFF-MMD squared for the equal-weight embedding used during selection
            mmd_rff = float(np.sum((mu_pi - current_embedding) ** 2))
            print(f"[selection {k+1}/{m}] picked {best_idx} | s={s} | rff-mmd(equally-weighted)={mmd_rff:.6g}")

    # Build coreset RFF matrix
    if len(selected_indices) == 0:
        return np.array([], dtype=int), np.array([], dtype=float)

    coreset_rff = X_candidate_rff[selected_indices]  # (s, D)
    s = coreset_rff.shape[0]

    # If only one element, weight=1
    if s == 1:
        final_weights = np.array([1.0], dtype=float)
        if verbose:
            mmd_final = float(np.sum((mu_pi - coreset_rff[0]) ** 2))
            print(f"Single-element support. final mmd={mmd_final:.6g}")
        return np.array(selected_indices, dtype=int), final_weights

    # Prepare Gram and z vectors in RFF space
    K_rff = coreset_rff.dot(coreset_rff.T)  # (s, s)
    z_rff = coreset_rff.dot(mu_pi)          # (s,)

    # Mirror Descent (Exponentiated Gradient) on simplex for this support
    # initialize uniform z
    z = np.ones(s, dtype=float) / float(s)
    eps = 1e-12
    for it in range(md_iterations):
        # gradient of ||mu - sum w_i x_i||^2 = 2*K_rff z - 2*z_rff
        grad = 2.0 * (K_rff.dot(z)) - 2.0 * z_rff

        # EG update in log-space (stable)
        log_z = np.log(z + eps) - eta * grad
        # numerical stabilization
        log_z = log_z - log_z.max()
        z = np.exp(log_z)
        z = z / (z.sum() + eps)

    final_weights = z

    if verbose:
        # compute final embedding and mmd
        final_embedding = final_weights.dot(coreset_rff)  # (D,)
        mmd_final = float(np.sum((mu_pi - final_embedding) ** 2))
        # Also compute equal-weight mmd for comparison
        equal_weights = np.ones(s, dtype=float) / float(s)
        equal_embedding = equal_weights.dot(coreset_rff)
        mmd_equal = float(np.sum((mu_pi - equal_embedding) ** 2))
        print(f"[final reweight] s={s} | mmd_equal={mmd_equal:.6g} | mmd_final={mmd_final:.6g}")

    return np.array(selected_indices, dtype=int), final_weights


def weighted_kernel_herding_rff_md(
    mu_pi: np.ndarray,
    X_candidate: np.ndarray,
    sampler: RBFSampler,
    m: int,
    md_iterations: int = 200,
    eta: float = 0.1,
    ridge: float = 1e-8,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fully-corrective weighted kernel herding in RFF space with Mirror Descent (EG)
    to obtain non-negative simplex weights (no QP).

    Args:
        mu_pi: mean embedding in RFF space (D,)
        X_candidate: raw candidates in input space (N, input_dim)
        sampler: fitted RBFSampler instance
        m: number of points to select
        md_iterations: iterations of EG used to compute weights for the current support
        eta: EG step-size
        ridge: small regularizer (kept for compatibility, not used in RFF-based GD)
        verbose: printing

    Returns:
        selected_indices (np.array of length <= m), final_weights (weights on selected set summing to 1)
    """
    X_candidate_rff = sampler.transform(X_candidate)  # (N, D)
    N, D = X_candidate_rff.shape

    selected_indices: List[int] = []
    current_embedding = np.zeros(D, dtype=float)
    weights = np.zeros(0, dtype=float)

    # mask for candidates already selected
    selected_mask = np.zeros(N, dtype=bool)

    for k in range(m):
        residual = mu_pi - current_embedding  # (D,)

        # score each candidate by inner-product with residual
        search_values = X_candidate_rff.dot(residual)  # (N,)
        search_values[selected_mask] = -np.inf

        best_x_idx = int(np.argmax(search_values))
        selected_indices.append(best_x_idx)
        selected_mask[best_x_idx] = True

        # features of current support
        coreset_rff = X_candidate_rff[selected_indices]  # (s, D)
        s = coreset_rff.shape[0]

        # Gram and z in RFF-space
        K_rff = coreset_rff.dot(coreset_rff.T)  # (s, s)
        z_rff = coreset_rff.dot(mu_pi)          # (s,)

        # Mirror Descent (Exponentiated Gradient) on simplex for this support
        if s == 1:
            weights = np.array([1.0], dtype=float)
        else:
            z = np.ones(s, dtype=float) / float(s)
            eps = 1e-12
            for it in range(md_iterations):
                # gradient of ||mu - sum_i w_i x_i||^2 = 2*K_rff z - 2*z_rff
                grad = 2.0 * (K_rff.dot(z)) - 2.0 * z_rff

                # EG update in log-space (stable)
                log_z = np.log(z + eps) - eta * grad
                log_z = log_z - log_z.max()
                z = np.exp(log_z)
                z = z / (z.sum() + eps)

            weights = z

        # update current_embedding for next greedy step
        current_embedding = weights.dot(coreset_rff)  # (D,)

        if verbose:
            # compute current RFF-MMD squared: ||mu - sum w x||^2
            mmd_rff = float(np.sum((mu_pi - current_embedding) ** 2))
            print(f"[RFF sel {k+1}/{m}] picked {best_x_idx} | rff-mmd={mmd_rff:.6g}")

    final_weights = weights
    return np.array(selected_indices, dtype=int), final_weights


def baseline_mirror_descent_rff(
    P_tensor: torch.Tensor,
    sampler: RBFSampler,
    m: int,
    mu_pi: np.ndarray,
    md_iterations: int = 200,
    eta: float = 0.1,
    ridge: float = 1e-8,
    verbose: bool = False,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Top-level entry: compute coreset indices and weights using RFF herding + EG weights.
    Returns a full-length weight vector (length = number of candidates) and a history dict.

    Note: P_tensor is expected to be the candidate *raw* inputs (not RFFs) so we can transform with sampler.
    """
    P_np = P_tensor.cpu().numpy()
    X_rff = sampler.transform(P_np)  # (N, D)

    S_indices, sel_weights = weighted_kernel_herding_rff_md(
        mu_pi,
        P_np,
        sampler,
        m,
        md_iterations=md_iterations,
        eta=eta,
        ridge=ridge,
        verbose=verbose,
    )

    # S_indices, sel_weights = weighted_kernel_herding_rff_qp(
    #     mu_pi,
    #     P_np,
    #     sampler,
    #     m
    # )

    # Build full vector of length N with zeros except on S_indices
    N = P_np.shape[0]
    w_full = np.zeros(N, dtype=float)
    if len(S_indices) > 0:
        w_full[S_indices] = np.maximum(sel_weights, 0.0)
        ssum = w_full[S_indices].sum()
        if ssum > 0:
            w_full[S_indices] = w_full[S_indices] / (ssum + 1e-12)
        else:
            w_full[S_indices] = np.ones(len(S_indices), dtype=float) / float(len(S_indices))

    history = {
        "selected_indices": S_indices,
        "selection_size": len(S_indices),
        # For backward-compatibility we can add an approximate RFF-MMD history if desired later
    }

    return w_full, history


class MirrorDescentHerdingStreamer(AbstractStreamingCoreset):
    """
    Streaming coreset selector that uses an RFF-based greedy herding selection
    and Mirror Descent (EG) weight optimization on the simplex.
    """

    def __init__(
        self,
        coreset_size: int,
        buffer_capacity: int,
        sampler: RBFSampler,
        batch_size: int,
        md_iterations: int = 200,
        eta: float = 0.1,
        ridge: float = 1e-8,
        verbose: bool = False,
    ) -> None:
        assert coreset_size <= buffer_capacity, "coreset_size must be <= buffer_capacity"

        self.coreset_size = coreset_size
        self.buffer_capacity = buffer_capacity
        self.sampler = sampler
        self.batch_size = batch_size
        self.md_iterations = md_iterations
        self.eta = eta
        self.ridge = ridge
        self.verbose = verbose

        # Derived dims
        self.rff_dim = sampler.n_components
        try:
            self.feature_dim = sampler.random_weights_.shape[1]
        except Exception:
            self.feature_dim = None

        # Buffer storage
        self.buffer_X = np.empty((0, self.feature_dim)) if self.feature_dim is not None else np.empty((0, 0))
        self.buffer_y = np.empty(0, dtype=int)
        self.buffer_weights = np.empty(0, dtype=float)

        # provenance: list[(batch_idx, local_idx)] for each buffered point
        self.buffer_provenance: List[Tuple[int, int]] = []

        # Running mean RFF and counters
        self.mean_rff_full_stream = np.zeros(self.rff_dim)
        self.num_points_seen = 0
        self._finalized = False

        # Last selection diagnostics
        self.last_history: Optional[Dict[str, Any]] = None

    def process_batch(
        self,
        X_batch_np: np.ndarray,
        y_batch_np: np.ndarray,
        batch_idx: int,
    ) -> None:
        """
        Process an incoming batch. No sensitive attributes are used.
        """
        if self._finalized:
            if self.verbose:
                print("Warning: streamer finalized, ignoring incoming batch")
            return

        batch_len = X_batch_np.shape[0]

        # Update running mean in RFF space
        X_batch_rff = self.sampler.transform(X_batch_np)
        current_batch_mean = np.mean(X_batch_rff, axis=0)

        if self.num_points_seen == 0:
            self.mean_rff_full_stream = current_batch_mean.copy()
        else:
            alpha = batch_len / float(self.num_points_seen + batch_len)
            self.mean_rff_full_stream = (1 - alpha) * self.mean_rff_full_stream + alpha * current_batch_mean

        self.num_points_seen += batch_len

        # Append new batch to buffer (if space)
        if self.buffer_X.size == 0:
            # initialize shapes correctly if empty
            self.buffer_X = np.asarray(X_batch_np).copy()
        else:
            self.buffer_X = np.vstack([self.buffer_X, X_batch_np])

        self.buffer_y = np.concatenate([self.buffer_y, np.asarray(y_batch_np, dtype=int)])

        # update buffer weights with an exponential/ageing style similar to WKH class
        if self.buffer_weights.size == 0:
            new_weights = np.full(batch_len, 1.0 / float(self.num_points_seen))
            self.buffer_weights = new_weights
        else:
            alpha = float(batch_len) / float(self.num_points_seen)
            self.buffer_weights *= (1 - alpha)
            new_weights = np.full(batch_len, alpha / float(batch_len))
            self.buffer_weights = np.concatenate([self.buffer_weights, new_weights])

        # provenance
        self.buffer_provenance.extend([(batch_idx, i) for i in range(batch_len)])

        # If buffer exceeded capacity, run MMD-driven selection on candidate pool
        if len(self.buffer_X) > self.buffer_capacity:
            if self.verbose:
                print(f"Buffer overflow (size={len(self.buffer_X)} > cap={self.buffer_capacity}) - running MD herding selection")

            X_candidate = self.buffer_X
            y_candidate = self.buffer_y
            provenance_candidate = list(self.buffer_provenance)

            P_tensor = torch.tensor(X_candidate, dtype=torch.float32)

            w_full, history = baseline_mirror_descent_rff(
                P_tensor,
                self.sampler,
                self.coreset_size,
                self.mean_rff_full_stream,
                md_iterations=self.md_iterations,
                eta=self.eta,
                ridge=self.ridge,
                verbose=self.verbose,
            )

            self.last_history = history

            # Pick selected indices and weights from w_full
            eps = 1e-12
            selected_mask = w_full > eps
            selected_idx_relative = np.where(selected_mask)[0]

            # If baseline returned no selections due to numerical issues, fall back to top-k by weight
            if selected_idx_relative.size == 0:
                order = np.argsort(-w_full)
                selected_idx_relative = order[: self.coreset_size]

            # If it selected more than needed, take top-k by weight
            if selected_idx_relative.size > self.coreset_size:
                rel_weights = w_full[selected_idx_relative]
                top_order = np.argsort(-rel_weights)[: self.coreset_size]
                selected_idx_relative = selected_idx_relative[top_order]

            # Normalize weights of selected
            sel_weights = w_full[selected_idx_relative]
            ssum = sel_weights.sum()
            if ssum > 0:
                sel_weights = sel_weights / float(ssum)
            else:
                sel_weights = np.ones(len(selected_idx_relative), dtype=float) / float(len(selected_idx_relative))

            # Update buffer to the new coreset
            self.buffer_X = X_candidate[selected_idx_relative]
            self.buffer_y = y_candidate[selected_idx_relative]
            self.buffer_weights = sel_weights
            self.buffer_provenance = [provenance_candidate[i] for i in selected_idx_relative]

            if self.verbose:
                print(f"Selected {len(self.buffer_X)} points into buffer after MD herding selection")

    def _finalize_coreset(self) -> None:
        if self._finalized:
            return

        # If buffer > target, run final selection on the buffer itself
        if len(self.buffer_X) > self.coreset_size:
            X_candidate = self.buffer_X
            y_candidate = self.buffer_y
            provenance_candidate = list(self.buffer_provenance)

            P_tensor = torch.tensor(X_candidate, dtype=torch.float32)

            w_full, history = baseline_mirror_descent_rff(
                P_tensor,
                self.sampler,
                self.coreset_size,
                self.mean_rff_full_stream,
                md_iterations=self.md_iterations,
                eta=self.eta,
                ridge=self.ridge,
                verbose=self.verbose,
            )

            self.last_history = history

            eps = 1e-12
            sel_mask = w_full > eps
            sel_idx_rel = np.where(sel_mask)[0]

            if sel_idx_rel.size == 0:
                order = np.argsort(-w_full)
                sel_idx_rel = order[: self.coreset_size]

            if sel_idx_rel.size > self.coreset_size:
                rel_weights = w_full[sel_idx_rel]
                top_order = np.argsort(-rel_weights)[: self.coreset_size]
                sel_idx_rel = sel_idx_rel[top_order]

            sel_weights = w_full[sel_idx_rel]
            ssum = sel_weights.sum()
            if ssum > 0:
                sel_weights = sel_weights / float(ssum)
            else:
                sel_weights = np.ones(len(sel_idx_rel), dtype=float) / float(len(sel_idx_rel))

            self.buffer_X = X_candidate[sel_idx_rel]
            self.buffer_y = y_candidate[sel_idx_rel]
            self.buffer_weights = sel_weights
            self.buffer_provenance = [provenance_candidate[i] for i in sel_idx_rel]

        self._finalized = True

    def get_final_coreset(self) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
        """Finalize (if needed) and return global flat indices, weights and provenance."""
        self._finalize_coreset()

        flat_indices = np.array([
            p[0] * self.batch_size + p[1] for p in self.buffer_provenance
        ], dtype=int)

        return flat_indices, self.buffer_weights, list(self.buffer_provenance)

    def print_coreset_provenance(self) -> None:
        flat_indices, weights, provenance = self.get_final_coreset()

        print("--- Final Coreset Provenance (Mirror Descent Herding - RFF) ---")
        if not provenance:
            print("Coreset is empty.")
            return

        print(f"{'Global Index':<15} {'Provenance':<20} {'Weight':<10}")
        print("-" * 55)
        for i in range(len(provenance)):
            prov_str = f"(Batch {provenance[i][0]}, Idx {provenance[i][1]})"
            print(f"{flat_indices[i]:<15} {prov_str:<20} {weights[i]:<10.6f}")
        print(f"\nTotal points in coreset: {len(provenance)}")
        print(f"Total points seen in stream: {self.num_points_seen}")
