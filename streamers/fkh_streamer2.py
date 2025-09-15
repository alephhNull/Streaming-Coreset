# """
# FairKernelHerdingStreamer

# This module provides a streaming coreset selector that replaces the plain
# WKH selection step with a fair mirror-descent reweighting + alternating
# selection routine provided by the user (`baseline_fair_mirror` +
# `weighted_kernel_herding_frank_wolfe_alternate`).

# The class extends AbstractStreamingCoreset and follows the same buffer and
# RFF running-mean bookkeeping used by the WKH implementation. When the
# buffer overflows it calls the fair baseline to select a subset of the
# candidate pool and returns coreset indices, weights and provenance.

# Assumptions:
# - The helper functions `baseline_fair_mirror(...)` and
#   `weighted_kernel_herding_frank_wolfe_alternate(...)` are available in the
#   same module or imported into the namespace where this file is used.
# - Incoming labels `y_batch_np` are binary {0,1} and sensitive attributes
#   are provided per-batch via `sensitive_batch` (dict of arrays or lists).

# Usage (short):
# - Instantiate with an already-fitted `RBFSampler` and appropriate sizes.
# - Call `process_batch(X_batch_np, y_batch_np, batch_idx, sensitive_batch)`
#   for each arriving batch. `sensitive_batch` is optional but required if
#   you want fairness-aware selection.
# - Call `get_final_coreset()` to obtain final indices, weights and provenance.

# """

# from typing import List, Tuple, Dict, Any, Optional
# import numpy as np
# import pandas as pd
# import torch
# from sklearn.kernel_approximation import RBFSampler
# from scipy.optimize import minimize


# # NOTE: AbstractStreamingCoreset must be available in the import path
# # as well as baseline_fair_mirror and weighted_kernel_herding_frank_wolfe_alternate
# from streamers.abstract_streamer import AbstractStreamingCoreset


# class FairKernelHerdingStreamer(AbstractStreamingCoreset):
#     """
#     Streaming coreset selector that uses the provided fair mirror-descent
#     baseline for coreset selection/weighting when the buffer overflows.

#     Key behaviour:
#     - Maintains a buffer of raw points, their labels and sensitive attributes.
#     - Tracks the running mean embedding in RFF space (via a pre-fitted RBFSampler).
#     - When the buffer would overflow, forms a candidate pool from old buffer +
#       the incoming batch and calls `baseline_fair_mirror` to select/weight a
#     coreset of size ``coreset_size`` from the pool.

#     The implementation intentionally mirrors the earlier WKH streaming
#     class so you can swap implementations easily.
#     """

#     def __init__(
#         self,
#         coreset_size: int,
#         buffer_capacity: int,
#         sampler: RBFSampler,
#         batch_size: int,
#         sigma: float,
#         select_alternate_freq: int = 2,
#         md_iterations: int = 200,
#         eta: float = 0.1,
#         verbose: bool = False,
#     ) -> None:
#         assert coreset_size <= buffer_capacity, "coreset_size must be <= buffer_capacity"

#         self.coreset_size = coreset_size
#         self.buffer_capacity = buffer_capacity
#         self.sampler = sampler
#         self.batch_size = batch_size
#         self.sigma = sigma
#         self.select_alternate_freq = select_alternate_freq
#         self.md_iterations = md_iterations
#         self.eta = eta
#         self.verbose = verbose

#         # Derived dims
#         self.rff_dim = sampler.n_components
#         # feature_dim - dimension of original input space
#         # Some RBFSampler implementations store the original dim in random_weights_.shape[1]
#         try:
#             self.feature_dim = sampler.random_weights_.shape[1]
#         except Exception:
#             # fallback - will still work but user must pass consistent X
#             self.feature_dim = None

#         # Buffer storage
#         self.buffer_X = np.empty((0, self.feature_dim)) if self.feature_dim is not None else np.empty((0, 0))
#         self.buffer_y = np.empty(0, dtype=int)
#         self.buffer_weights = np.empty(0, dtype=float)
#         # sensitive attributes: dict[name] -> numpy array stacked for buffer
#         self.buffer_sensitive: Dict[str, np.ndarray] = {}

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
#         sensitive_batch: Optional[Dict[str, Any]] = None,
#     ) -> None:
#         """
#         Process an incoming batch. `sensitive_batch` should be a dict mapping
#         attribute name -> array-like (length batch) with group labels for that
#         attribute.
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
#             # assign initial uniform weights proportional to this batch
#             new_weights = np.full(batch_len, 1.0 / float(self.num_points_seen))
#             self.buffer_weights = new_weights
#         else:
#             # scale old weights and append new
#             alpha = float(batch_len) / float(self.num_points_seen)
#             self.buffer_weights *= (1 - alpha)
#             new_weights = np.full(batch_len, alpha / float(batch_len))
#             self.buffer_weights = np.concatenate([self.buffer_weights, new_weights])

#         # sensitive attributes
#         if sensitive_batch is not None:
#             for k, v in sensitive_batch.items():
#                 arr = np.asarray(v)
#                 if k in self.buffer_sensitive and self.buffer_sensitive[k].size > 0:
#                     self.buffer_sensitive[k] = np.concatenate([self.buffer_sensitive[k], arr])
#                 else:
#                     self.buffer_sensitive[k] = arr.copy()
#         else:
#             # if no sensitive attrs provided, maintain empty dict
#             pass

#         # provenance
#         self.buffer_provenance.extend([(batch_idx, i) for i in range(batch_len)])

#         # If buffer exceeded capacity, run fair selection on candidate pool
#         if len(self.buffer_X) > self.buffer_capacity:
#             if self.verbose:
#                 print(f"Buffer overflow (size={len(self.buffer_X)} > cap={self.buffer_capacity}) - running fair selection")

#             # Candidate pool is entire buffer
#             X_candidate = self.buffer_X
#             y_candidate = self.buffer_y
#             provenance_candidate = list(self.buffer_provenance)

#             # Build sensitive_cols as dict of pandas.Series (required by baseline function)
#             sensitive_cols_pd: Dict[str, pd.Series] = {}
#             for k, arr in self.buffer_sensitive.items():
#                 sensitive_cols_pd[k] = pd.Series(arr)

#             # Convert inputs to types expected by baseline_fair_mirror
#             P_tensor = torch.tensor(X_candidate, dtype=torch.float32)
#             outcome_idx = torch.tensor(y_candidate, dtype=torch.long)

#             # Run baseline fair mirror selection
#             w_full, history = baseline_fair_mirror(
#                 P_tensor,
#                 sensitive_cols_pd,
#                 outcome_idx,
#                 self.sigma,
#                 self.coreset_size,
#                 select_alternate_freq=self.select_alternate_freq,
#                 md_iterations=self.md_iterations,
#                 eta=self.eta,
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

#             # Update buffer sensitive attributes
#             new_buf_sensitive: Dict[str, np.ndarray] = {}
#             for k, arr in self.buffer_sensitive.items():
#                 new_buf_sensitive[k] = arr[selected_idx_relative]
#             self.buffer_sensitive = new_buf_sensitive

#             if self.verbose:
#                 print(f"Selected {len(self.buffer_X)} points into buffer after fair selection")

#     def _finalize_coreset(self) -> None:
#         if self._finalized:
#             return

#         # If buffer > target, run final selection on the buffer itself
#         if len(self.buffer_X) > self.coreset_size:
#             X_candidate = self.buffer_X
#             y_candidate = self.buffer_y
#             provenance_candidate = list(self.buffer_provenance)

#             sensitive_cols_pd: Dict[str, pd.Series] = {}
#             for k, arr in self.buffer_sensitive.items():
#                 sensitive_cols_pd[k] = pd.Series(arr)

#             P_tensor = torch.tensor(X_candidate, dtype=torch.float32)
#             outcome_idx = torch.tensor(y_candidate, dtype=torch.long)

#             w_full, history = baseline_fair_mirror(
#                 P_tensor,
#                 sensitive_cols_pd,
#                 outcome_idx,
#                 self.sigma,
#                 self.coreset_size,
#                 select_alternate_freq=self.select_alternate_freq,
#                 md_iterations=self.md_iterations,
#                 eta=self.eta,
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

#             new_buf_sensitive: Dict[str, np.ndarray] = {}
#             for k, arr in self.buffer_sensitive.items():
#                 new_buf_sensitive[k] = arr[sel_idx_rel]
#             self.buffer_sensitive = new_buf_sensitive

#         self._finalized = True

#     def get_final_coreset(self) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
#         """Finalize (if needed) and return global flat indices, weights and provenance."""
#         self._finalize_coreset()

#         # Convert provenance -> flat global indices (batch_idx * batch_size + local_idx)
#         flat_indices = np.array([
#             p[0] * self.batch_size + p[1] for p in self.buffer_provenance
#         ], dtype=int)

#         return flat_indices, self.buffer_weights, list(self.buffer_provenance)

#     def print_coreset_provenance(self) -> None:
#         flat_indices, weights, provenance = self.get_final_coreset()

#         print("--- Final Coreset Provenance (Fair Kernel Herding) ---")
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


# # End of file
# def rbf_kernel_np(X, Y=None, sigma=1.0):
#     X = np.asarray(X)
#     Y = X if Y is None else np.asarray(Y)
#     XX = np.sum(X**2, axis=1)[:, None]
#     YY = np.sum(Y**2, axis=1)[None, :]
#     D2 = np.maximum(XX + YY - 2.0 * (X @ Y.T), 0.0)
#     return np.exp(-D2 / (2.0 * sigma**2))


# # ---------------------- Fairness Penalty Calculation Logic (modified) ----------------------
# def get_fairness_penalty(w_full, sensitive_cols, outcome_mask_np):
#     """Calculates total fairness penalty using marginal positive p (global).
#     Penalty uses (T_g - p * S_g)^2 where T_g = sum_{i in g} y_i w_i, S_g = sum_{i in g} w_i,
#     and p = overall marginal positive rate (constant).
#     This yields linear constraints T_g - p*S_g = 0 in z.
#     """
#     total_penalty = 0.0
#     eps = 1e-9
#     # marginal positive from full dataset (constant)
#     p = float(outcome_mask_np.sum() / (len(outcome_mask_np) + eps))
#     for key, s_col in sensitive_cols.items():
#         groups = s_col.unique()
#         if len(groups) <= 1: continue

#         for group in groups:
#             group_mask = (s_col == group).to_numpy(dtype=bool)
#             S_g = w_full[group_mask].sum()
#             T_g = w_full[group_mask & outcome_mask_np].sum()
#             # residual linear in w: T_g - p*S_g
#             residual = (T_g - p * S_g)
#             total_penalty += float(residual ** 2)
#     return total_penalty



# def qp_weights_slsqp(K_SS, k_S, ridge=1e-8):
#     m = K_SS.shape[0]
#     P = K_SS + ridge * np.eye(m)
#     def obj(w): return float(w @ (P @ w) - 2.0 * (k_S @ w))
#     def jac(w): return 2.0 * (P @ w) - 2.0 * k_S
#     cons = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
#     bounds = [(0.0, None) for _ in range(m)]
#     res = minimize(fun=obj, x0=np.ones(m)/m, jac=jac, bounds=bounds, constraints=cons, method='SLSQP', options={'maxiter': 200, 'ftol': 1e-9})
#     w = np.maximum(res.x, 0.0)
#     return w / (w.sum() + 1e-9)


# # ---------------------- Helper fairness utilities ----------------------
# def compute_weighted_dp_gaps(weights, sensitive_cols, outcome_mask_np, eps=1e-9):
#     """Compute weighted demographic parity gaps and per-group rates.
#     weights: 1D numpy array aligned with sensitive_cols index (full dataset).
#     sensitive_cols: dict of pandas Series (or single Series) keyed by attribute.
#     outcome_mask_np: boolean numpy array of positives.

#     Returns: dp_gaps (dict attr -> gap), rates_dict (dict attr -> dict group->rate)
#     """
#     dp_gaps = {}
#     rates_dict = {}
#     for key, s_col in sensitive_cols.items():
#         groups = s_col.unique()
#         if len(groups) <= 1:
#             continue
#         rates = {}
#         group_rates = []
#         for g in groups:
#             mask = (s_col == g).to_numpy(dtype=bool)
#             group_weight_sum = weights[mask].sum()
#             pos_weight = weights[mask & outcome_mask_np].sum()
#             rate = pos_weight / (group_weight_sum + eps)
#             rates[g] = float(rate)
#             group_rates.append(rate)
#         if group_rates:
#             dp_gaps[key] = float(max(group_rates) - min(group_rates))
#             rates_dict[key] = rates
#     return dp_gaps, rates_dict


# def compute_unweighted_dp_gaps(selected_indices, sensitive_cols, outcome_mask_np):
#     """Compute unweighted demographic parity gaps for a selected subset.
#     selected_indices: iterable of integer indices (selected items)
#     sensitive_cols: dict of pandas Series aligned with full dataset
#     outcome_mask_np: boolean numpy array of positives (full dataset)

#     Returns: dp_gaps (dict), rates_dict (dict)
#     """
#     dp_gaps = {}
#     rates_dict = {}
#     sel_idx = np.array(list(selected_indices), dtype=int)
#     if len(sel_idx) == 0:
#         return dp_gaps, rates_dict
#     for key, s_col in sensitive_cols.items():
#         groups = s_col.unique()
#         if len(groups) <= 1:
#             continue
#         rates = {}
#         group_rates = []
#         for g in groups:
#             mask = (s_col == g).to_numpy(dtype=bool)
#             sel_mask = np.isin(sel_idx, np.where(mask)[0])
#             group_sel_indices = sel_idx[sel_mask]
#             group_count = len(group_sel_indices)
#             if group_count == 0:
#                 # treat as rate 0 for gap calc (alternatively skip); here we skip this group
#                 continue
#             pos_count = outcome_mask_np[group_sel_indices].sum()
#             rate = float(pos_count) / (group_count + 1e-9)
#             rates[g] = rate
#             group_rates.append(rate)
#         if group_rates:
#             dp_gaps[key] = float(max(group_rates) - min(group_rates))
#             rates_dict[key] = rates
#     return dp_gaps, rates_dict


# def weighted_kernel_herding_frank_wolfe_alternate(
#     X_np, m, sigma, sensitive_cols, outcome_mask,
#     alternate_freq=2, fairness_beta=1.0, ridge=1e-8, verbose=False
# ):
#     N = X_np.shape[0]
#     K = rbf_kernel_np(X_np, X_np, sigma=sigma)
#     mu_pi = K.mean(axis=0)
#     selected = []
#     remaining = set(range(N))
#     outcome_mask_np = outcome_mask.cpu().numpy()

#     # diagnostics
#     selection_history = {'step': [], 'unweighted_max_dp': [], 'weighted_max_dp': [], 'weighted_rates': [], 'unweighted_rates': []}

#     for t in range(m):
#         best_idx, best_score = -1, np.inf

#         # Alternate between MMD objective and Fairness objective
#         is_fairness_step = (alternate_freq is not None and (t + 1) % alternate_freq == 0)

#         for j in remaining:
#             S_candidate = selected + [j]
#             S_arr = np.array(S_candidate, dtype=int)
#             K_SS = K[np.ix_(S_arr, S_arr)]
#             K_XS = K[:, S_arr]
#             z = K_XS.mean(axis=0)

#             try:
#                 w_candidate = np.linalg.solve(K_SS + ridge * np.eye(len(S_arr)), z)
#             except np.linalg.LinAlgError:
#                 w_candidate = np.linalg.pinv(K_SS + ridge * np.eye(len(S_arr))) @ z

#             if is_fairness_step:
#                 w_full_cand = np.zeros(N)
#                 w_full_cand[S_arr] = w_candidate
#                 score = get_fairness_penalty(w_full_cand, sensitive_cols, outcome_mask_np)
#             else: # MMD step
#                 residual = (K_XS @ w_candidate) - mu_pi
#                 score = float(residual @ residual)

#             if score < best_score:
#                 best_score = score
#                 best_idx = j

#         selected.append(best_idx)
#         remaining.remove(best_idx)

#         # --- compute diagnostics after this selection ---
#         S_arr = np.array(selected, dtype=int)
#         # unweighted: uniform over selected indices
#         un_dp, un_rates = compute_unweighted_dp_gaps(S_arr, sensitive_cols, outcome_mask_np)

#         # # weighted: solve QP on current S to obtain proper weights
#         # w_curr = np.zeros(N)
#         # try:
#         #     w_sub = qp_weights_slsqp(K[np.ix_(S_arr, S_arr)], K[:, S_arr].mean(axis=0), ridge=ridge)
#         #     w_curr[S_arr] = w_sub
#         #     w_dp, w_rates = compute_weighted_dp_gaps(w_curr, sensitive_cols, outcome_mask_np)
#         # except Exception as e:
#         #     # fallback to uniform weighted dp if QP fails
#         #     w_dp, w_rates = un_dp, un_rates

#         selection_history['step'].append(len(selected))
#         selection_history['unweighted_max_dp'].append(max(un_dp.values()) if un_dp else 0.0)
#         # selection_history['weighted_max_dp'].append(max(w_dp.values()) if w_dp else 0.0)
#         # selection_history['weighted_rates'].append(w_rates)
#         selection_history['unweighted_rates'].append(un_rates)

#         if verbose:
#             obj_type = "FAIR" if is_fairness_step else "MMD"
#             print(f"[sel {t+1}/{m} - {obj_type}] picked {best_idx} | score {best_score:.6g} | un_max_dp={selection_history['unweighted_max_dp'][-1]:.6g} | w_max_dp={selection_history['weighted_max_dp'][-1]:.6g}")

#     return np.array(selected, dtype=int), K, selection_history

# # ---------------------- New: Fair Mirror baseline (mirror descent with closed-form group rescaling) ----------------------
# # Replace the previous baseline_fair_mirror with this function in your script.

# def baseline_fair_mirror(P_tensor, sensitive_cols, outcome_idx, sigma, m,
#                          select_alternate_freq=2, md_iterations=200, eta=0.1, verbose=False):
#     """
#     1) Select coreset S with weighted_kernel_herding_frank_wolfe_alternate (alternating MMD/fairness).
#     2) Optimize weights on the selected support using Mirror Descent (entropy mirror / EG).
#        - Closed-form marginal-positive correction available when target_mode='marginal_p'
#          (binary outcomes, disjoint groups per attribute).
#        - Otherwise supports 'equal' or 'proportional' group-mass targets (per-attribute).
#     Returns: (w_full, history) where w_full is full-N vector (zeros off S) and history contains diagnostics.
#     """
#     # --- 1) Select coreset using your alternating selection (same as ADMM baseline) ---
#     P_np = P_tensor.cpu().numpy()
#     S, K_full, selection_history = weighted_kernel_herding_frank_wolfe_alternate(
#         P_np, m, sigma, sensitive_cols, outcome_idx,
#         alternate_freq=select_alternate_freq, verbose=verbose
#     )

#     # Subset kernels & target vector for subset optimization
#     K_SS = K_full[np.ix_(S, S)].astype(float)
#     k_S = K_full[:, S].mean(axis=0).astype(float)
#     mu_const = float(K_full.mean())

#     # Subset-sensitive columns and outcomes
#     sensitive_cols_sub = {key: val.iloc[S].reset_index(drop=True) for key, val in sensitive_cols.items()}
#     outcome_sub = outcome_idx[S].cpu().numpy().astype(int)  # binary 0/1 expected

#     # Precompute attr_info for the subset (m-sized)
#     attr_info = []
#     for key, s_col in sensitive_cols_sub.items():
#         groups = list(s_col.unique())
#         masks = [(s_col == g).to_numpy(dtype=bool) for g in groups]   # length m boolean masks (subset)
#         attr_info.append({'name': key, 'groups': groups, 'masks': masks})

#     # global marginal positive rate (from full dataset) - used in marginal_p correction
#     p_global = float(outcome_idx.cpu().numpy().mean())
#     print("Global marginal positive rate:", p_global)

#     # --- 2) Mirror Descent (optimize weights on S) ---
#     eps = 1e-12
#     m_sub = len(S)
#     # initialize uniform on subset simplex
#     z = np.ones(m_sub, dtype=float) / float(m_sub)

#     history = {'mmd': [], 'max_dp': [], 'penalty': [], 'subset_weighted_dp': [], 'subset_unweighted_dp': [], 'selection_history': selection_history}

#     for it in range(md_iterations):
#         # gradient of MMD: grad = 2*K_SS*z - 2*k_S_subset
#         grad = 2.0 * (K_SS @ z) - 2.0 * k_S[S] if False else 2.0 * (K_SS @ z) - 2.0 * k_S  # k_S already sliced to full->S mean in caller
#         # note: k_S variable is already K_full[:, S].mean so it's shape (m_sub,); just use it directly
#         grad = 2.0 * (K_SS @ z) - 2.0 * k_S

#         # EG update (log-space stable)
#         log_z = np.log(z + eps) - eta * grad
#         log_z = log_z - log_z.max()
#         tilde = np.exp(log_z)
#         tilde = tilde / (tilde.sum() + eps)

#         # NOTE: exact per-attribute closed-form projection requires attribute groups to partition the subset.
#         for attr in attr_info:
#                 masks = attr['masks']   # list of boolean masks (length m_sub)
#                 for mask in masks:
#                     group_inds = np.where(mask)[0]
#                     if len(group_inds) == 0:
#                         continue

#                     # indices of positives and negatives inside the group (subset indexing)
#                     pos_idx = group_inds[np.array(outcome_sub[group_inds]) == 1]
#                     neg_idx = group_inds[np.array(outcome_sub[group_inds]) == 0]

#                     # compute current un-normalized mass inside pos/neg parts
#                     A = float(tilde[pos_idx].sum()) if pos_idx.size > 0 else 0.0
#                     B = float(tilde[neg_idx].sum()) if neg_idx.size > 0 else 0.0

#                     # feasibility / numerical guards:
#                     # if either side has zero mass, inject tiny mass uniformly inside that sub-block
#                     tiny = 1e-12
#                     if pos_idx.size > 0 and A <= eps:
#                         tilde[pos_idx] = tiny / float(pos_idx.size)
#                         A = float(tilde[pos_idx].sum())
#                     if neg_idx.size > 0 and B <= eps:
#                         tilde[neg_idx] = tiny / float(neg_idx.size)
#                         B = float(tilde[neg_idx].sum())

#                     # if one of pos/neg is completely absent (no members in group), skip (nothing to do)
#                     if pos_idx.size == 0 or neg_idx.size == 0:
#                         # If e.g. no positives in this group (pos_idx.size==0) but rho>0, the constraint is infeasible.
#                         # We skip enforcement here (user should handle infeasible target or add epsilon mass prior).
#                         continue

#                     # Now compute closed-form dual lambda_g and scaling factors (stable in log-domain)
#                     # lambda_g = log( (1-rho) * A / (rho * B) )
#                     if p_global <= eps:
#                         # target p=0 -> require zero positive mass; set positives to 0 inside group
#                         tilde[pos_idx] = 0.0
#                     elif (1.0 - p_global) <= eps:
#                         # target p=1 -> require zero negative mass; set negatives to 0 inside group
#                         tilde[neg_idx] = 0.0
#                     else:
#                         lambda_g = np.log(((1.0 - p_global) * A) / (p_global * B))
#                         # scale factors:
#                         scale_pos = np.exp(-(1.0 - p_global) * lambda_g)   # exp(-(1-rho)*lambda)
#                         scale_neg = np.exp(p_global * lambda_g)           # exp(rho*lambda)
#                         if pos_idx.size > 0:
#                             tilde[pos_idx] = tilde[pos_idx] * scale_pos
#                         if neg_idx.size > 0:
#                             tilde[neg_idx] = tilde[neg_idx] * scale_neg

#                 # After processing all groups for this attribute, re-normalize tilde
#                 tilde = tilde / (tilde.sum() + eps)


#         # accept iterate
#         z = tilde

#         # diagnostics
#         mmd_val = float(z @ K_SS @ z - 2.0 * (z @ k_S) + mu_const)
#         # compute penalty = sum_g (T_g - p_global*S_g)^2 (over attributes)
#         penalty_val = 0.0
#         dp_vals = []
#         for attr in attr_info:
#             for mask in attr['masks']:
#                 Sg = float(z[mask].sum())
#                 Tg = float((z * outcome_sub)[mask].sum())
#                 residual = Tg - p_global * Sg
#                 penalty_val += float(residual ** 2)
#                 # group positive rate (weighted)
#             # dp gap for this attr (if groups exist)
#             rates_attr = []
#             for mask in attr['masks']:
#                 group_sum = float(z[mask].sum())
#                 if group_sum <= eps:
#                     rates_attr.append(0.0)
#                 else:
#                     pos_rate = float((z * outcome_sub)[mask].sum()) / (group_sum + eps)
#                     rates_attr.append(pos_rate)
#             if rates_attr:
#                 dp_vals.append(max(rates_attr) - min(rates_attr))

#         max_dp = max(dp_vals) if dp_vals else 0.0
#         # unweighted dp on subset (uniform)
#         z_unif = np.ones_like(z) / float(len(z))
#         dp_unw_vals = []
#         for attr in attr_info:
#             rates = []
#             for mask in attr['masks']:
#                 group_count = mask.sum()
#                 if group_count == 0:
#                     continue
#                 pos_count = (outcome_sub[mask]).sum()
#                 rate = float(pos_count) / (float(group_count) + eps)
#                 rates.append(rate)
#             if rates:
#                 dp_unw_vals.append(max(rates) - min(rates))
#         subset_unw_dp = max(dp_unw_vals) if dp_unw_vals else 0.0

#         history['mmd'].append(mmd_val)
#         history['penalty'].append(penalty_val)
#         history['max_dp'].append(max_dp)
#         history['subset_weighted_dp'].append(max_dp)
#         history['subset_unweighted_dp'].append(subset_unw_dp)

#         if verbose and (it % max(1, md_iterations // 5) == 0):
#                 # compute worst residual
#                 worst_resid = 0.0
#                 for attr in attr_info:
#                     for mask in attr['masks']:
#                         Sg = float(z[mask].sum())
#                         Tg = float((z * outcome_sub)[mask].sum())
#                         resid = abs(Tg - p_global * Sg)
#                         worst_resid = max(worst_resid, resid)
#                 print(f"[FairMirror-MD it {it+1}/{md_iterations}] MMD={mmd_val:.6g} | pen={penalty_val:.6g} | worst_resid={worst_resid:.6g}")

#     # After MD: return full vector with subset weights (z)
#     w_full = np.zeros(P_np.shape[0], dtype=float)
#     w_full[S] = np.maximum(z, 0.0)
#     # ensure subset weights sum to 1 (numerical)
#     ssum = w_full[S].sum()
#     if ssum > 0:
#         w_full[S] = w_full[S] / (ssum + 1e-12)

#     return w_full, history


# fair_mirror_rff_streamer.py
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
import torch
from sklearn.kernel_approximation import RBFSampler

# NOTE: AbstractStreamingCoreset must be available in the import path
from streamers.abstract_streamer import AbstractStreamingCoreset

# ---------------------- Utility kernels / fairness helpers ----------------------
def rbf_kernel_np(X, Y=None, sigma=1.0):
    X = np.asarray(X)
    Y = X if Y is None else np.asarray(Y)
    XX = np.sum(X**2, axis=1)[:, None]
    YY = np.sum(Y**2, axis=1)[None, :]
    D2 = np.maximum(XX + YY - 2.0 * (X @ Y.T), 0.0)
    return np.exp(-D2 / (2.0 * sigma**2))


def get_fairness_penalty(w_full, sensitive_cols, outcome_mask_np):
    """Calculates total fairness penalty using marginal positive p (global).
    Penalty uses (T_g - p * S_g)^2 where T_g = sum_{i in g} y_i w_i, S_g = sum_{i in g} w_i,
    and p = overall marginal positive rate (constant).
    sensitive_cols: dict[str -> pd.Series]
    outcome_mask_np: boolean 1D numpy (len = N)
    """
    total_penalty = 0.0
    eps = 1e-9
    # marginal positive from full dataset (constant)
    p = float(outcome_mask_np.sum() / (len(outcome_mask_np) + eps))
    for key, s_col in sensitive_cols.items():
        groups = s_col.unique()
        if len(groups) <= 1:
            continue
        for group in groups:
            group_mask = (s_col == group).to_numpy(dtype=bool)
            S_g = float(w_full[group_mask].sum())
            T_g = float(w_full[group_mask & outcome_mask_np].sum())
            residual = (T_g - p * S_g)
            total_penalty += float(residual ** 2)
    return total_penalty


def compute_weighted_dp_gaps(weights, sensitive_cols, outcome_mask_np, eps=1e-9):
    dp_gaps = {}
    rates_dict = {}
    for key, s_col in sensitive_cols.items():
        groups = s_col.unique()
        if len(groups) <= 1:
            continue
        rates = {}
        group_rates = []
        for g in groups:
            mask = (s_col == g).to_numpy(dtype=bool)
            group_weight_sum = weights[mask].sum()
            pos_weight = weights[mask & outcome_mask_np].sum()
            rate = pos_weight / (group_weight_sum + eps)
            rates[g] = float(rate)
            group_rates.append(rate)
        if group_rates:
            dp_gaps[key] = float(max(group_rates) - min(group_rates))
            rates_dict[key] = rates
    return dp_gaps, rates_dict


def compute_unweighted_dp_gaps(selected_indices, sensitive_cols, outcome_mask_np):
    dp_gaps = {}
    rates_dict = {}
    sel_idx = np.array(list(selected_indices), dtype=int)
    if len(sel_idx) == 0:
        return dp_gaps, rates_dict
    for key, s_col in sensitive_cols.items():
        groups = s_col.unique()
        if len(groups) <= 1:
            continue
        rates = {}
        group_rates = []
        for g in groups:
            mask = (s_col == g).to_numpy(dtype=bool)
            sel_mask = np.isin(sel_idx, np.where(mask)[0])
            group_sel_indices = sel_idx[sel_mask]
            group_count = len(group_sel_indices)
            if group_count == 0:
                continue
            pos_count = int(outcome_mask_np[group_sel_indices].sum())
            rate = float(pos_count) / (group_count + 1e-9)
            rates[g] = rate
            group_rates.append(rate)
        if group_rates:
            dp_gaps[key] = float(max(group_rates) - min(group_rates))
            rates_dict[key] = rates
    return dp_gaps, rates_dict

# ---------------------- RFF-based alternating greedy (MD weights inside loop) ----------------------
def weighted_kernel_herding_rff_alternate(
    X_np: np.ndarray,
    sampler: RBFSampler,
    mu_pi: np.ndarray,
    m: int,
    sensitive_cols: Dict[str, pd.Series],
    outcome_mask: torch.Tensor,
    alternate_freq: int = 2,
    md_iterations: int = 200,
    eta: float = 0.1,
    fair_md_iter: int = 50,
    ridge: float = 1e-8,
    verbose: bool = False,
):
    """
    RFF-based alternating greedy selection. At each greedy step:
      - If it's an MMD step: pick by inner-product with residual (fast).
      - If it's a fairness step: for each candidate, form support S + [j],
        run short MD (with closed-form per-group rescaling) on that support to get weights,
        then compute fairness penalty and pick candidate minimizing it.
    After pick, run full MD on the chosen support (md_iterations) to update weights.
    Returns: selected indices (np.array), final_weights_on_support (np.array), selection_history
    """
    N = X_np.shape[0]
    X_rff = sampler.transform(X_np)  # (N, D)
    D = X_rff.shape[1]
    selected: List[int] = []
    remaining = set(range(N))
    selected_mask = np.zeros(N, dtype=bool)

    # running embedding and weights on current support
    current_embedding = np.zeros(D, dtype=float)
    weights = np.zeros(0, dtype=float)

    outcome_mask_np = outcome_mask.cpu().numpy().astype(bool)

    selection_history = {
        'step': [],
        'unweighted_max_dp': [],
        'subset_unweighted_rates': [],
        'subset_weighted_dp': [],
        'subset_weighted_rates': [],
        'mmd_vals': []
    }

    # attribute info building helper for closed-form rescaling (like in your baseline)
    def _build_attr_info_for_subset(S_idx):
        attr_info = []
        for key, s_col in sensitive_cols.items():
            s_sub = s_col.iloc[S_idx].reset_index(drop=True)
            groups = list(s_sub.unique())
            masks = [(s_sub == g).to_numpy(dtype=bool) for g in groups]  # masks length |S|
            attr_info.append({'name': key, 'groups': groups, 'masks': masks})
        return attr_info

    # helper: run mirror-descent EG on coreset_rff with optional per-attribute rescaling (closed-form)
    def _run_md_with_rescaling(coreset_rff, mu_pi_local, outcome_sub, attr_info,
                               md_iters=100, eta_local=0.1):
        s_local = coreset_rff.shape[0]
        if s_local == 0:
            return np.array([])

        # initialize
        z = np.ones(s_local, dtype=float) / float(s_local)
        eps = 1e-12

        K_SS = coreset_rff.dot(coreset_rff.T)
        k_S = coreset_rff.dot(mu_pi_local)

        for it in range(md_iters):
            grad = 2.0 * (K_SS @ z) - 2.0 * k_S
            log_z = np.log(z + eps) - eta_local * grad
            log_z = log_z - log_z.max()
            tilde = np.exp(log_z)
            tilde = tilde / (tilde.sum() + eps)

            # closed-form marginal-positive rescaling per attribute groups (uses global p)
            for attr in attr_info:
                masks = attr['masks']  # boolean masks of length s_local
                for mask in masks:
                    group_inds = np.where(mask)[0]
                    if len(group_inds) == 0:
                        continue

                    pos_idx = group_inds[np.array(outcome_sub[group_inds]) == 1]
                    neg_idx = group_inds[np.array(outcome_sub[group_inds]) == 0]

                    A = float(tilde[pos_idx].sum()) if pos_idx.size > 0 else 0.0
                    B = float(tilde[neg_idx].sum()) if neg_idx.size > 0 else 0.0

                    tiny = 1e-12
                    if pos_idx.size > 0 and A <= eps:
                        tilde[pos_idx] = tiny / float(pos_idx.size)
                        A = float(tilde[pos_idx].sum())
                    if neg_idx.size > 0 and B <= eps:
                        tilde[neg_idx] = tiny / float(neg_idx.size)
                        B = float(tilde[neg_idx].sum())

                    if pos_idx.size == 0 or neg_idx.size == 0:
                        continue

                    # global marginal p computed from outcome_sub's global origin is not known here;
                    # But baseline uses p_global (from full dataset). We pass p_global later via attr_info closure.
                    # We'll assume attr_info contains 'p_global' key if needed. For now we compute p_local from outcome.
                    p_local = float(outcome_sub.mean()) if outcome_sub.size > 0 else 0.0

                    if p_local <= eps:
                        tilde[pos_idx] = 0.0
                    elif (1.0 - p_local) <= eps:
                        tilde[neg_idx] = 0.0
                    else:
                        # same form as earlier
                        lambda_g = np.log(((1.0 - p_local) * A) / (p_local * B))
                        scale_pos = np.exp(-(1.0 - p_local) * lambda_g)
                        scale_neg = np.exp(p_local * lambda_g)
                        if pos_idx.size > 0:
                            tilde[pos_idx] = tilde[pos_idx] * scale_pos
                        if neg_idx.size > 0:
                            tilde[neg_idx] = tilde[neg_idx] * scale_neg

                tilde = tilde / (tilde.sum() + eps)

            z = tilde

        return z

    # MAIN greedy loop
    for t in range(m):
        is_fairness_step = (alternate_freq is not None and (t + 1) % alternate_freq == 0)

        if not is_fairness_step:
            # MMD step (fast): pick candidate by inner-product with residual
            residual = mu_pi - current_embedding  # (D,)
            search_values = X_rff.dot(residual)
            search_values[selected_mask] = -np.inf
            best_idx = int(np.argmax(search_values))
            best_score = float(-search_values[best_idx])  # not used much, but for logging
        else:
            # fairness step: test each remaining candidate by doing a short MD on support S + [j]
            best_idx = -1
            best_score = np.inf
            # Precompute p_global from full outcomes (so closed-form uses global marginal)
            p_global = float(outcome_mask_np.mean()) if outcome_mask_np.size > 0 else 0.0

            for j in list(remaining):
                S_candidate = selected + [j]
                S_arr = np.array(S_candidate, dtype=int)
                coreset_rff = X_rff[S_arr]  # (s, D)
                outcome_sub = outcome_mask_np[S_arr].astype(int)

                # attr_info for this subset but include p_global for consistent rescaling
                attr_info = []
                for key, s_col in sensitive_cols.items():
                    s_sub = s_col.iloc[S_arr].reset_index(drop=True)
                    groups = list(s_sub.unique())
                    masks = [(s_sub == g).to_numpy(dtype=bool) for g in groups]
                    attr_info.append({'name': key, 'groups': groups, 'masks': masks, 'p_global': p_global})

                # run short MD with rescaling (fairness MD)
                try:
                    z_cand = _run_md_with_rescaling(coreset_rff, mu_pi, outcome_sub, attr_info,
                                                    md_iters=fair_md_iter, eta_local=eta)
                except Exception:
                    # fallback uniform if MD fails
                    z_cand = np.ones(len(S_candidate)) / float(len(S_candidate))

                # build full w vector and compute penalty
                w_full_cand = np.zeros(N, dtype=float)
                w_full_cand[S_arr] = np.maximum(z_cand, 0.0)
                ssum = w_full_cand[S_arr].sum()
                if ssum > 0:
                    w_full_cand[S_arr] = w_full_cand[S_arr] / (ssum + 1e-12)
                else:
                    w_full_cand[S_arr] = np.ones(len(S_arr)) / float(len(S_arr))

                penalty = get_fairness_penalty(w_full_cand, sensitive_cols, outcome_mask_np)
                if penalty < best_score:
                    best_score = penalty
                    best_idx = int(j)

            if best_idx == -1:
                # fallback: pick any
                remaining_list = list(remaining)
                best_idx = int(remaining_list[0])
                best_score = 1e9

        # Add selected
        selected.append(best_idx)
        selected_mask[best_idx] = True
        remaining.remove(best_idx)

        # AFTER adding, run full MD on support to obtain updated weights and current embedding
        S_arr = np.array(selected, dtype=int)
        coreset_rff = X_rff[S_arr]
        outcome_sub = outcome_mask_np[S_arr].astype(int)
        attr_info = _build_attr_info_for_subset(S_arr)

        # Run MD with full md_iterations and closed-form rescaling using p_global of full data
        # Insert p_global into attr_info entries to ensure consistent rescaling
        p_global_full = float(outcome_mask_np.mean()) if outcome_mask_np.size > 0 else 0.0
        for attr in attr_info:
            attr['p_global'] = p_global_full

        z_full = _run_md_with_rescaling(coreset_rff, mu_pi, outcome_sub, attr_info,
                                        md_iters=md_iterations, eta_local=eta)
        # normalize
        z_full = np.maximum(z_full, 0.0)
        ssum = z_full.sum()
        if ssum > 0:
            z_full = z_full / (ssum + 1e-12)
        else:
            z_full = np.ones_like(z_full) / float(len(z_full))

        weights = z_full
        current_embedding = weights.dot(coreset_rff)  # (D,)

        # diagnostics
        # mmd value in RFF: ||mu - sum w x||^2
        mmd_rff = float(np.sum((mu_pi - current_embedding) ** 2))
        # unweighted dp for subset
        un_dp, un_rates = compute_unweighted_dp_gaps(S_arr, sensitive_cols, outcome_mask_np)
        # weighted dp for subset (by z_full)
        w_full_now = np.zeros(N, dtype=float)
        w_full_now[S_arr] = weights
        w_dp, w_rates = compute_weighted_dp_gaps(w_full_now, sensitive_cols, outcome_mask_np)

        selection_history['step'].append(len(selected))
        selection_history['mmd_vals'].append(mmd_rff)
        selection_history['unweighted_max_dp'].append(max(un_dp.values()) if un_dp else 0.0)
        selection_history['subset_unweighted_rates'].append(un_rates)
        selection_history['subset_weighted_dp'].append(max(w_dp.values()) if w_dp else 0.0)
        selection_history['subset_weighted_rates'].append(w_rates)

        if verbose:
            obj_type = "FAIR" if is_fairness_step else "MMD"
            print(f"[sel {t+1}/{m} - {obj_type}] picked {best_idx} | mmd_rff={mmd_rff:.6g} | un_max_dp={selection_history['unweighted_max_dp'][-1]:.6g} | w_max_dp={selection_history['subset_weighted_dp'][-1]:.6g}")

    # final full weights vector
    w_full = np.zeros(N, dtype=float)
    S_arr = np.array(selected, dtype=int)
    w_full[S_arr] = weights
    ssum = w_full[S_arr].sum()
    if ssum > 0:
        w_full[S_arr] = w_full[S_arr] / (ssum + 1e-12)
    else:
        if len(S_arr) > 0:
            w_full[S_arr] = np.ones(len(S_arr), dtype=float) / float(len(S_arr))

    return np.array(selected, dtype=int), w_full, selection_history

# ---------------------- Baseline: Fair Mirror (RFF-backed) ----------------------
def baseline_fair_mirror_rff(
    P_tensor: torch.Tensor,
    sampler: RBFSampler,
    mu_pi: np.ndarray,
    sensitive_cols: Dict[str, pd.Series],
    outcome_idx: torch.Tensor,
    m: int,
    select_alternate_freq: int = 2,
    md_iterations: int = 200,
    eta: float = 0.1,
    fair_md_iter: int = 50,
    verbose: bool = False,
):
    """
    1) Select coreset S with weighted_kernel_herding_rff_alternate (alternating MMD/fairness) in RFF space.
    2) Optimize weights on S using the same MD-with-rescaling loop (already used inside selection,
       but we run a final MD to be safe).
    Returns: (w_full, history)
    """
    P_np = P_tensor.cpu().numpy()
    outcome_np = outcome_idx.cpu().numpy().astype(int)
    outcome_mask = torch.tensor(outcome_np > 0, dtype=torch.bool)

    S, w_full_sel, selection_history = weighted_kernel_herding_rff_alternate(
        P_np,
        sampler,
        mu_pi,
        m,
        sensitive_cols,
        outcome_mask,
        alternate_freq=select_alternate_freq,
        md_iterations=md_iterations,
        eta=eta,
        fair_md_iter=fair_md_iter,
        ridge=1e-8,
        verbose=verbose,
    )

    # ensure normalized and build final history
    ssum = w_full_sel[S].sum() if len(S) > 0 else 0.0
    if ssum > 0:
        w_full_sel[S] = w_full_sel[S] / (ssum + 1e-12)
    else:
        if len(S) > 0:
            w_full_sel[S] = np.ones(len(S), dtype=float) / float(len(S))

    history = {
        'selected_indices': S,
        'selection_size': len(S),
        'selection_history': selection_history,
    }

    return w_full_sel, history

# ---------------------- FairKernelHerdingStreamer (uses RFF baseline above) ----------------------
class FairKernelHerdingStreamer(AbstractStreamingCoreset):
    """
    Streaming coreset selector that uses RFF-based fair mirror herding (MD weights)
    with alternating MMD/fairness greedy steps.
    """

    def __init__(
        self,
        coreset_size: int,
        buffer_capacity: int,
        sampler: RBFSampler,
        batch_size: int,
        select_alternate_freq: int = 2,
        md_iterations: int = 200,
        eta: float = 0.1,
        fair_md_iter: int = 50,
        verbose: bool = False,
    ) -> None:
        assert coreset_size <= buffer_capacity, "coreset_size must be <= buffer_capacity"

        self.coreset_size = coreset_size
        self.buffer_capacity = buffer_capacity
        self.sampler = sampler
        self.batch_size = batch_size
        self.select_alternate_freq = select_alternate_freq
        self.md_iterations = md_iterations
        self.eta = eta
        self.fair_md_iter = fair_md_iter
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
        # sensitive attributes: dict[name] -> numpy array stacked for buffer
        self.buffer_sensitive: Dict[str, np.ndarray] = {}

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
        sensitive_batch: Optional[Dict[str, Any]] = None,
    ) -> None:
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
            self.buffer_X = np.asarray(X_batch_np).copy()
        else:
            self.buffer_X = np.vstack([self.buffer_X, X_batch_np])

        self.buffer_y = np.concatenate([self.buffer_y, np.asarray(y_batch_np, dtype=int)])

        # weights aging similar to other streamers
        if self.buffer_weights.size == 0:
            new_weights = np.full(batch_len, 1.0 / float(self.num_points_seen))
            self.buffer_weights = new_weights
        else:
            alpha = float(batch_len) / float(self.num_points_seen)
            self.buffer_weights *= (1 - alpha)
            new_weights = np.full(batch_len, alpha / float(batch_len))
            self.buffer_weights = np.concatenate([self.buffer_weights, new_weights])

        # sensitive attributes
        if sensitive_batch is not None:
            for k, v in sensitive_batch.items():
                arr = np.asarray(v)
                if k in self.buffer_sensitive and self.buffer_sensitive[k].size > 0:
                    self.buffer_sensitive[k] = np.concatenate([self.buffer_sensitive[k], arr])
                else:
                    self.buffer_sensitive[k] = arr.copy()

        # provenance
        self.buffer_provenance.extend([(batch_idx, i) for i in range(batch_len)])

        # If buffer exceeded capacity, run fair selection on candidate pool
        if len(self.buffer_X) > self.buffer_capacity:
            if self.verbose:
                print(f"Buffer overflow ({len(self.buffer_X)} > {self.buffer_capacity}) - running RFF fair mirror selection")

            X_candidate = self.buffer_X
            y_candidate = self.buffer_y
            provenance_candidate = list(self.buffer_provenance)

            # Build sensitive_cols as dict of pandas.Series
            sensitive_cols_pd: Dict[str, pd.Series] = {}
            for k, arr in self.buffer_sensitive.items():
                sensitive_cols_pd[k] = pd.Series(arr)

            P_tensor = torch.tensor(X_candidate, dtype=torch.float32)
            outcome_idx = torch.tensor(y_candidate, dtype=torch.long)

            w_full, history = baseline_fair_mirror_rff(
                P_tensor,
                self.sampler,
                self.mean_rff_full_stream,
                sensitive_cols_pd,
                outcome_idx,
                self.coreset_size,
                select_alternate_freq=self.select_alternate_freq,
                md_iterations=self.md_iterations,
                eta=self.eta,
                fair_md_iter=self.fair_md_iter,
                verbose=self.verbose,
            )

            self.last_history = history

            # select indices from w_full
            eps = 1e-12
            selected_mask = w_full > eps
            selected_idx_relative = np.where(selected_mask)[0]

            if selected_idx_relative.size == 0:
                order = np.argsort(-w_full)
                selected_idx_relative = order[: self.coreset_size]

            if selected_idx_relative.size > self.coreset_size:
                rel_weights = w_full[selected_idx_relative]
                top_order = np.argsort(-rel_weights)[: self.coreset_size]
                selected_idx_relative = selected_idx_relative[top_order]

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

            # Update sensitive attributes
            new_buf_sensitive: Dict[str, np.ndarray] = {}
            for k, arr in self.buffer_sensitive.items():
                new_buf_sensitive[k] = arr[selected_idx_relative]
            self.buffer_sensitive = new_buf_sensitive

            if self.verbose:
                print(f"Selected {len(self.buffer_X)} points into buffer after fair RFF selection")

    def _finalize_coreset(self) -> None:
        if self._finalized:
            return

        if len(self.buffer_X) > self.coreset_size:
            X_candidate = self.buffer_X
            y_candidate = self.buffer_y
            provenance_candidate = list(self.buffer_provenance)

            sensitive_cols_pd: Dict[str, pd.Series] = {}
            for k, arr in self.buffer_sensitive.items():
                sensitive_cols_pd[k] = pd.Series(arr)

            P_tensor = torch.tensor(X_candidate, dtype=torch.float32)
            outcome_idx = torch.tensor(y_candidate, dtype=torch.long)

            w_full, history = baseline_fair_mirror_rff(
                P_tensor,
                self.sampler,
                self.mean_rff_full_stream,
                sensitive_cols_pd,
                outcome_idx,
                self.coreset_size,
                select_alternate_freq=self.select_alternate_freq,
                md_iterations=self.md_iterations,
                eta=self.eta,
                fair_md_iter=self.fair_md_iter,
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

            new_buf_sensitive: Dict[str, np.ndarray] = {}
            for k, arr in self.buffer_sensitive.items():
                new_buf_sensitive[k] = arr[sel_idx_rel]
            self.buffer_sensitive = new_buf_sensitive

        self._finalized = True

    def get_final_coreset(self) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
        self._finalize_coreset()
        flat_indices = np.array([
            p[0] * self.batch_size + p[1] for p in self.buffer_provenance
        ], dtype=int)
        return flat_indices, self.buffer_weights, list(self.buffer_provenance)

    def print_coreset_provenance(self) -> None:
        flat_indices, weights, provenance = self.get_final_coreset()
        print("--- Final Coreset Provenance (Fair Kernel Herding - RFF MD) ---")
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
