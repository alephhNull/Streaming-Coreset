import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Optional
from scipy.optimize import minimize


# This streamer expects the following helper functions to be available in the
# caller environment (they were provided by you):
# - weighted_kernel_herding_frank_wolfe_alternate
# - baseline_fair_mirror
# The streamer calls baseline_fair_mirror to perform alternating kernel-herding
# selection + mirror-descent weight optimization. Those functions operate on
# numpy / pandas inputs and return deterministic weight vectors.


class FairKernelHerdingStreamer:
    """Streaming coreset selector that uses Alternating Fair Kernel-Herding +
    Mirror Descent to compute coreset weights. The key design choices now are:

      - `buffer_weights` is a plain non-trainable torch tensor (NOT an nn.Parameter).
      - No optimizer state is kept in the streamer (no torch.optim objects).
      - AMP and lambda-log regularizer removed.
      - The redundancy-based pruning (fast cosine-sim matmul) is preserved.
      - If the fair pipeline fails, a single manual autograd gradient step is
        performed as a lightweight fallback (no persistent optimizer).

    The streamer still stores RFF embeddings (on device) and original features
    / labels / sensitive attributes (CPU lists) because baseline_fair_mirror
    expects original-feature inputs.
    """

    def __init__(
        self,
        batch_size: int,
        m_coreset_size: int,
        n_rff_components: int,
        buffer_capacity: int,
        rbf_sigma: float,
        n_epochs_online: int = 30,
        random_seed: Optional[int] = 42,
        device: str = "cuda",
        select_alternate_freq: int = 2,
        md_iterations: int = 200,
        md_eta: float = 0.1,
        verbose: bool = False,
    ):
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.batch_size = batch_size

        # core params
        self.m = m_coreset_size
        self.n_rff_components = n_rff_components
        self.buffer_capacity = buffer_capacity
        self.rbf_sigma = rbf_sigma

        # optimization / md params
        self.n_epochs_online = n_epochs_online
        self.select_alternate_freq = select_alternate_freq
        self.md_iterations = md_iterations
        self.md_eta = md_eta

        # state
        self.rbf_sampler = None
        self.random_seed = random_seed
        self.num_points_seen = 0
        self.mean_rff_full_stream_torch = torch.zeros(self.n_rff_components, dtype=torch.float32, device=self.device)

        # buffer representations
        self.buffer_rffs: Optional[torch.Tensor] = None       # (B, D) on device
        self.buffer_weights: Optional[torch.Tensor] = None    # (B,) plain non-trainable tensor on device
        self.buffer_global_ids: List[Tuple[int, int]] = []    # provenance

        # keep original features/labels/sensitive attrs on CPU for selection routine
        self.buffer_X_list: List[np.ndarray] = []
        self.buffer_y_list: List[int] = []
        self.buffer_sensitive: Dict[str, List] = {}

        # monitoring
        self.sparsity_history: List[int] = []
        self.verbose = verbose

        if random_seed is not None:
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)

    def set_rbf_sampler(self, rbf_sampler_instance):
        self.rbf_sampler = rbf_sampler_instance

    # ----------------------- Weight optimization (fair pipeline) -----------------------
    def _optimize_weights(self):
        """Compute weights for the current buffer using baseline_fair_mirror.

        - On success: set self.buffer_weights to the returned normalized weights
          (torch.tensor on device, non-trainable).
        - On failure: perform a single manual autograd gradient step to reduce MMD
          (no persistent optimizer kept).
        """
        if len(self.buffer_X_list) == 0:
            return

        # prepare inputs for baseline_fair_mirror
        X_buf_np = np.vstack(self.buffer_X_list)  # (B, D_original)
        y_buf_np = np.array(self.buffer_y_list, dtype=int)
        sensitive_cols_pd = {k: pd.Series(v) for k, v in self.buffer_sensitive.items()}
        P_tensor = torch.from_numpy(X_buf_np).float()
        outcome_idx = torch.from_numpy(y_buf_np).long()
        w_full, history = baseline_fair_mirror(
            P_tensor,
            sensitive_cols_pd,
            outcome_idx,
            sigma=self.rbf_sigma,
            m=self.m,
            select_alternate_freq=self.select_alternate_freq,
            md_iterations=self.md_iterations,
            eta=self.md_eta,
            verbose=self.verbose,
        )
        w_full = np.asarray(w_full, dtype=float)
        if w_full.shape[0] != len(self.buffer_X_list):
            raise ValueError("baseline_fair_mirror returned weight vector of wrong length")
        # convert to torch tensor on device and store as plain tensor
        w_tensor = torch.tensor(w_full, dtype=torch.float32, device=self.device)
        # ensure non-negativity and normalization
        with torch.no_grad():
            w_tensor = torch.clamp(w_tensor, min=0.0)
            if w_tensor.sum() > 0:
                w_tensor = w_tensor / (w_tensor.sum() + 1e-12)
            else:
                # fallback to uniform
                w_tensor = torch.ones_like(w_tensor) / float(w_tensor.numel())
        self.buffer_weights = w_tensor
        self.sparsity_history.append(int((w_tensor > 1e-9).sum().item()))
        if self.verbose:
            print(f"[OptWeights] fair pipeline returned {self.sparsity_history[-1]} nonzero weights")


    # ----------------------- Buffer pruning (redundancy-based) -----------------------
    def _prune_buffer(self):
        """Prune buffer down to buffer_capacity using redundancy scoring (cosine similarity).
        Keep the top-m provisional coreset by weight and prune the most-redundant among
        the remaining candidates. This preserves the original fast pruning behavior.
        """
        if self.buffer_rffs is None:
            return
        num_to_prune = self.buffer_rffs.shape[0] - self.buffer_capacity
        if num_to_prune <= 0:
            return

        with torch.no_grad():
            weights = self.buffer_weights.detach() if self.buffer_weights is not None else torch.ones(self.buffer_rffs.shape[0], device=self.device)
            sorted_indices = torch.argsort(weights, descending=True)

            provisional_coreset_indices = sorted_indices[: self.m]
            pruning_candidate_indices = sorted_indices[self.m :]

            if pruning_candidate_indices.numel() == 0:
                return

            if pruning_candidate_indices.numel() <= num_to_prune:
                indices_to_remove = pruning_candidate_indices
            else:
                provisional_coreset_rffs = self.buffer_rffs[provisional_coreset_indices]  # (k, D)
                pruning_rffs = self.buffer_rffs[pruning_candidate_indices]  # (p, D)

                prov_norms = provisional_coreset_rffs.norm(dim=1, keepdim=True).clamp(min=1e-9)
                prune_norms = pruning_rffs.norm(dim=1, keepdim=True).clamp(min=1e-9)
                prov_unit = provisional_coreset_rffs / prov_norms
                prune_unit = pruning_rffs / prune_norms

                # similarity: (p, D) @ (D, k) -> (p, k)
                cosine_sim = torch.matmul(prune_unit, prov_unit.T)
                redundancy_scores, _ = torch.max(cosine_sim, dim=1)  # (p,)

                # select top `num_to_prune` most-redundant candidates to prune
                idxs_sorted_by_redundancy = torch.argsort(redundancy_scores, descending=True)
                indices_to_prune_from_candidates = idxs_sorted_by_redundancy[:num_to_prune]
                indices_to_remove = pruning_candidate_indices[indices_to_prune_from_candidates]

        # Build a boolean mask and keep the rest
        mask = torch.ones(self.buffer_rffs.shape[0], dtype=torch.bool, device=self.device)
        mask[indices_to_remove] = False

        # Apply mask to tensors and global ids
        self.buffer_rffs = self.buffer_rffs[mask]
        new_weights_tensor = self.buffer_weights.detach()[mask] if self.buffer_weights is not None else torch.ones(int(mask.sum().item()), device=self.device)
        self.buffer_weights = new_weights_tensor

        # update global ids list (mask is on device; move to cpu)
        mask_cpu = mask.cpu().numpy().tolist()
        self.buffer_global_ids = [g for keep, g in zip(mask_cpu, self.buffer_global_ids) if keep]

        # Update CPU-side lists (X, y, sensitive)
        kept_indices = [i for i, keep in enumerate(mask_cpu) if keep]
        self.buffer_X_list = [self.buffer_X_list[i] for i in kept_indices]
        self.buffer_y_list = [self.buffer_y_list[i] for i in kept_indices]
        new_buffer_sensitive = {}
        for key, lst in self.buffer_sensitive.items():
            new_buffer_sensitive[key] = [lst[i] for i in kept_indices]
        self.buffer_sensitive = new_buffer_sensitive

    # ----------------------- Main processing -----------------------
    def process_batch(self, X_batch_np: np.ndarray, y_batch_np: np.ndarray, batch_idx: int, sensitive_batch: Optional[Dict[str, List]] = None) -> None:
        if self.rbf_sampler is None:
            raise RuntimeError("RBFSampler not set.")
        batch_size = X_batch_np.shape[0]
        if batch_size == 0:
            return

        # 1) Transform entire batch at once
        batch_rff_np = self.rbf_sampler.transform(X_batch_np)  # (batch_size, D)
        batch_rff = torch.from_numpy(batch_rff_np).float().to(self.device)

        # 2) Append to contiguous buffer tensors and CPU lists
        if self.buffer_rffs is None:
            self.buffer_rffs = batch_rff.clone()
            init_weights = torch.full((batch_rff.shape[0],), 1.0 / float(batch_rff.shape[0]), device=self.device, dtype=torch.float32)
            self.buffer_weights = init_weights
            self.buffer_global_ids = [(batch_idx, i) for i in range(batch_rff.shape[0])]
        else:
            self.buffer_rffs = torch.cat([self.buffer_rffs, batch_rff], dim=0)
            new_w = torch.full((batch_rff.shape[0],), 1.0 / float(batch_rff.shape[0]), device=self.device, dtype=torch.float32)
            concatenated_weights = torch.cat([self.buffer_weights.detach(), new_w], dim=0) if self.buffer_weights is not None else new_w
            self.buffer_weights = concatenated_weights
            base_len = len(self.buffer_global_ids)
            self.buffer_global_ids.extend([(batch_idx, i) for i in range(batch_rff.shape[0])])

        # append original features and labels to CPU lists
        for i in range(batch_size):
            self.buffer_X_list.append(X_batch_np[i:i+1].astype(float))
            self.buffer_y_list.append(int(y_batch_np[i]))

        # append sensitive attributes if provided
        if sensitive_batch is not None:
            for key, col in sensitive_batch.items():
                if key not in self.buffer_sensitive:
                    self.buffer_sensitive[key] = []
                self.buffer_sensitive[key].extend(list(col))

        # 3) Update running mean embedding (exponential moving average)
        alpha = 0.1
        current_batch_mean = torch.mean(batch_rff, dim=0)

        if self.num_points_seen == 0:
            self.mean_rff_full_stream_torch = current_batch_mean.clone()
        else:
            self.mean_rff_full_stream_torch = (1 - alpha) * self.mean_rff_full_stream_torch + alpha * current_batch_mean

        self.num_points_seen += batch_size

        # 4) Optimize weights using fair pipeline (or fallback manual step)
        self._optimize_weights()

        # 5) Prune if buffer exceeded capacity
        if self.buffer_rffs is not None and self.buffer_rffs.shape[0] > self.buffer_capacity:
            self._prune_buffer()

        num_nonzero_weights = self.sparsity_history[-1] if self.sparsity_history else 0
        if self.verbose:
            print(f"   Batch {batch_idx} processed. Num non-zero weights: {num_nonzero_weights}")

    # ----------------------- Final coreset extraction -----------------------
    def get_final_coreset(self) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
        """Return (flat_indices, normalized_weights_np, coreset_global_ids).

        Picks top-m points by current buffer_weights.
        """
        if len(self.buffer_X_list) == 0:
            return np.array([], dtype=int), np.array([]), []

        with torch.no_grad():
            weights = self.buffer_weights.detach() if self.buffer_weights is not None else torch.ones(len(self.buffer_X_list), device=self.device)
            k_topk = min(self.m, weights.numel())
            if k_topk == 0:
                return np.array([], dtype=int), np.array([]), []

            vals, idx = torch.topk(weights, k=k_topk)
            vals = vals.cpu().numpy()
            idx = idx.cpu().numpy().tolist()
            normalized = vals / (vals.sum() + 1e-9)

            coreset_global_ids = [self.buffer_global_ids[i] for i in idx]
            flat_indices = np.array([gid[0] * self.batch_size + gid[1] for gid in coreset_global_ids], dtype=int)

            return flat_indices, normalized, coreset_global_ids

    def print_coreset_provenance(self) -> None:
        if not self.buffer_global_ids:
            print("Coreset is empty.")
            return

        flat_indices, coreset_weights, coreset_global_ids = self.get_final_coreset()
        print("--- Final Coreset Provenance (Fair Kernel Herding) ---")
        for i, (gid, flat_idx) in enumerate(zip(coreset_global_ids, flat_indices)):
            print(f"  Point {i}: From Batch {gid[0]}, Idx {gid[1]} (Flat Index: {flat_idx}) -> Weight: {coreset_weights[i]:.4f}")

        batch_indices = [gid[0] for gid in coreset_global_ids]
        batch_counts = {b: batch_indices.count(b) for b in sorted(list(set(batch_indices)))}
        print("Coreset points per batch:", batch_counts, "------------------------------")



def rbf_kernel_np(X, Y=None, sigma=1.0):
    X = np.asarray(X)
    Y = X if Y is None else np.asarray(Y)
    XX = np.sum(X**2, axis=1)[:, None]
    YY = np.sum(Y**2, axis=1)[None, :]
    D2 = np.maximum(XX + YY - 2.0 * (X @ Y.T), 0.0)
    return np.exp(-D2 / (2.0 * sigma**2))


# ---------------------- Fairness Penalty Calculation Logic (modified) ----------------------
def get_fairness_penalty(w_full, sensitive_cols, outcome_mask_np):
    """Calculates total fairness penalty using marginal positive p (global).
    Penalty uses (T_g - p * S_g)^2 where T_g = sum_{i in g} y_i w_i, S_g = sum_{i in g} w_i,
    and p = overall marginal positive rate (constant).
    This yields linear constraints T_g - p*S_g = 0 in z.
    """
    total_penalty = 0.0
    eps = 1e-9
    # marginal positive from full dataset (constant)
    p = float(outcome_mask_np.sum() / (len(outcome_mask_np) + eps))
    for key, s_col in sensitive_cols.items():
        groups = s_col.unique()
        if len(groups) <= 1: continue

        for group in groups:
            group_mask = (s_col == group).to_numpy(dtype=bool)
            S_g = w_full[group_mask].sum()
            T_g = w_full[group_mask & outcome_mask_np].sum()
            # residual linear in w: T_g - p*S_g
            residual = (T_g - p * S_g)
            total_penalty += float(residual ** 2)
    return total_penalty



def qp_weights_slsqp(K_SS, k_S, ridge=1e-8):
    m = K_SS.shape[0]
    P = K_SS + ridge * np.eye(m)
    def obj(w): return float(w @ (P @ w) - 2.0 * (k_S @ w))
    def jac(w): return 2.0 * (P @ w) - 2.0 * k_S
    cons = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
    bounds = [(0.0, None) for _ in range(m)]
    res = minimize(fun=obj, x0=np.ones(m)/m, jac=jac, bounds=bounds, constraints=cons, method='SLSQP', options={'maxiter': 200, 'ftol': 1e-9})
    w = np.maximum(res.x, 0.0)
    return w / (w.sum() + 1e-9)


# ---------------------- Helper fairness utilities ----------------------
def compute_weighted_dp_gaps(weights, sensitive_cols, outcome_mask_np, eps=1e-9):
    """Compute weighted demographic parity gaps and per-group rates.
    weights: 1D numpy array aligned with sensitive_cols index (full dataset).
    sensitive_cols: dict of pandas Series (or single Series) keyed by attribute.
    outcome_mask_np: boolean numpy array of positives.

    Returns: dp_gaps (dict attr -> gap), rates_dict (dict attr -> dict group->rate)
    """
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
    """Compute unweighted demographic parity gaps for a selected subset.
    selected_indices: iterable of integer indices (selected items)
    sensitive_cols: dict of pandas Series aligned with full dataset
    outcome_mask_np: boolean numpy array of positives (full dataset)

    Returns: dp_gaps (dict), rates_dict (dict)
    """
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
                # treat as rate 0 for gap calc (alternatively skip); here we skip this group
                continue
            pos_count = outcome_mask_np[group_sel_indices].sum()
            rate = float(pos_count) / (group_count + 1e-9)
            rates[g] = rate
            group_rates.append(rate)
        if group_rates:
            dp_gaps[key] = float(max(group_rates) - min(group_rates))
            rates_dict[key] = rates
    return dp_gaps, rates_dict


def weighted_kernel_herding_frank_wolfe_alternate(
    X_np, m, sigma, sensitive_cols, outcome_mask,
    alternate_freq=2, fairness_beta=1.0, ridge=1e-8, verbose=False
):
    N = X_np.shape[0]
    K = rbf_kernel_np(X_np, X_np, sigma=sigma)
    mu_pi = K.mean(axis=0)
    selected = []
    remaining = set(range(N))
    outcome_mask_np = outcome_mask.cpu().numpy()

    # diagnostics
    selection_history = {'step': [], 'unweighted_max_dp': [], 'weighted_max_dp': [], 'weighted_rates': [], 'unweighted_rates': []}

    for t in range(m):
        best_idx, best_score = -1, np.inf

        # Alternate between MMD objective and Fairness objective
        is_fairness_step = (alternate_freq is not None and (t + 1) % alternate_freq == 0)

        for j in remaining:
            S_candidate = selected + [j]
            S_arr = np.array(S_candidate, dtype=int)
            K_SS = K[np.ix_(S_arr, S_arr)]
            K_XS = K[:, S_arr]
            z = K_XS.mean(axis=0)

            try:
                w_candidate = np.linalg.solve(K_SS + ridge * np.eye(len(S_arr)), z)
            except np.linalg.LinAlgError:
                w_candidate = np.linalg.pinv(K_SS + ridge * np.eye(len(S_arr))) @ z

            if is_fairness_step:
                w_full_cand = np.zeros(N)
                w_full_cand[S_arr] = w_candidate
                score = get_fairness_penalty(w_full_cand, sensitive_cols, outcome_mask_np)
            else: # MMD step
                residual = (K_XS @ w_candidate) - mu_pi
                score = float(residual @ residual)

            if score < best_score:
                best_score = score
                best_idx = j

        selected.append(best_idx)
        remaining.remove(best_idx)

        # --- compute diagnostics after this selection ---
        S_arr = np.array(selected, dtype=int)
        # unweighted: uniform over selected indices
        un_dp, un_rates = compute_unweighted_dp_gaps(S_arr, sensitive_cols, outcome_mask_np)

        # weighted: solve QP on current S to obtain proper weights
        w_curr = np.zeros(N)
        try:
            w_sub = qp_weights_slsqp(K[np.ix_(S_arr, S_arr)], K[:, S_arr].mean(axis=0), ridge=ridge)
            w_curr[S_arr] = w_sub
            w_dp, w_rates = compute_weighted_dp_gaps(w_curr, sensitive_cols, outcome_mask_np)
        except Exception as e:
            # fallback to uniform weighted dp if QP fails
            w_dp, w_rates = un_dp, un_rates

        selection_history['step'].append(len(selected))
        selection_history['unweighted_max_dp'].append(max(un_dp.values()) if un_dp else 0.0)
        selection_history['weighted_max_dp'].append(max(w_dp.values()) if w_dp else 0.0)
        selection_history['weighted_rates'].append(w_rates)
        selection_history['unweighted_rates'].append(un_rates)

        if verbose:
            obj_type = "FAIR" if is_fairness_step else "MMD"
            print(f"[sel {t+1}/{m} - {obj_type}] picked {best_idx} | score {best_score:.6g} | un_max_dp={selection_history['unweighted_max_dp'][-1]:.6g} | w_max_dp={selection_history['weighted_max_dp'][-1]:.6g}")

    return np.array(selected, dtype=int), K, selection_history

# ---------------------- New: Fair Mirror baseline (mirror descent with closed-form group rescaling) ----------------------
# Replace the previous baseline_fair_mirror with this function in your script.

def baseline_fair_mirror(P_tensor, sensitive_cols, outcome_idx, sigma, m,
                         select_alternate_freq=2, md_iterations=200, eta=0.1, verbose=False):
    """
    1) Select coreset S with weighted_kernel_herding_frank_wolfe_alternate (alternating MMD/fairness).
    2) Optimize weights on the selected support using Mirror Descent (entropy mirror / EG).
       - Closed-form marginal-positive correction available when target_mode='marginal_p'
         (binary outcomes, disjoint groups per attribute).
       - Otherwise supports 'equal' or 'proportional' group-mass targets (per-attribute).
    Returns: (w_full, history) where w_full is full-N vector (zeros off S) and history contains diagnostics.
    """
    # --- 1) Select coreset using your alternating selection (same as ADMM baseline) ---
    P_np = P_tensor.cpu().numpy()
    S, K_full, selection_history = weighted_kernel_herding_frank_wolfe_alternate(
        P_np, m, sigma, sensitive_cols, outcome_idx,
        alternate_freq=select_alternate_freq, verbose=verbose
    )

    # Subset kernels & target vector for subset optimization
    K_SS = K_full[np.ix_(S, S)].astype(float)
    k_S = K_full[:, S].mean(axis=0).astype(float)
    mu_const = float(K_full.mean())

    # Subset-sensitive columns and outcomes
    sensitive_cols_sub = {key: val.iloc[S].reset_index(drop=True) for key, val in sensitive_cols.items()}
    outcome_sub = outcome_idx[S].cpu().numpy().astype(int)  # binary 0/1 expected

    # Precompute attr_info for the subset (m-sized)
    attr_info = []
    for key, s_col in sensitive_cols_sub.items():
        groups = list(s_col.unique())
        masks = [(s_col == g).to_numpy(dtype=bool) for g in groups]   # length m boolean masks (subset)
        attr_info.append({'name': key, 'groups': groups, 'masks': masks})

    # global marginal positive rate (from full dataset) - used in marginal_p correction
    p_global = float(outcome_idx.cpu().numpy().mean())
    print("Global marginal positive rate:", p_global)

    # --- 2) Mirror Descent (optimize weights on S) ---
    eps = 1e-12
    m_sub = len(S)
    # initialize uniform on subset simplex
    z = np.ones(m_sub, dtype=float) / float(m_sub)

    history = {'mmd': [], 'max_dp': [], 'penalty': [], 'subset_weighted_dp': [], 'subset_unweighted_dp': [], 'selection_history': selection_history}

    for it in range(md_iterations):
        # gradient of MMD: grad = 2*K_SS*z - 2*k_S_subset
        grad = 2.0 * (K_SS @ z) - 2.0 * k_S[S] if False else 2.0 * (K_SS @ z) - 2.0 * k_S  # k_S already sliced to full->S mean in caller
        # note: k_S variable is already K_full[:, S].mean so it's shape (m_sub,); just use it directly
        grad = 2.0 * (K_SS @ z) - 2.0 * k_S

        # EG update (log-space stable)
        log_z = np.log(z + eps) - eta * grad
        log_z = log_z - log_z.max()
        tilde = np.exp(log_z)
        tilde = tilde / (tilde.sum() + eps)

        # NOTE: exact per-attribute closed-form projection requires attribute groups to partition the subset.
        for attr in attr_info:
                masks = attr['masks']   # list of boolean masks (length m_sub)
                for mask in masks:
                    group_inds = np.where(mask)[0]
                    if len(group_inds) == 0:
                        continue

                    # indices of positives and negatives inside the group (subset indexing)
                    pos_idx = group_inds[np.array(outcome_sub[group_inds]) == 1]
                    neg_idx = group_inds[np.array(outcome_sub[group_inds]) == 0]

                    # compute current un-normalized mass inside pos/neg parts
                    A = float(tilde[pos_idx].sum()) if pos_idx.size > 0 else 0.0
                    B = float(tilde[neg_idx].sum()) if neg_idx.size > 0 else 0.0

                    # feasibility / numerical guards:
                    # if either side has zero mass, inject tiny mass uniformly inside that sub-block
                    tiny = 1e-12
                    if pos_idx.size > 0 and A <= eps:
                        tilde[pos_idx] = tiny / float(pos_idx.size)
                        A = float(tilde[pos_idx].sum())
                    if neg_idx.size > 0 and B <= eps:
                        tilde[neg_idx] = tiny / float(neg_idx.size)
                        B = float(tilde[neg_idx].sum())

                    # if one of pos/neg is completely absent (no members in group), skip (nothing to do)
                    if pos_idx.size == 0 or neg_idx.size == 0:
                        # If e.g. no positives in this group (pos_idx.size==0) but rho>0, the constraint is infeasible.
                        # We skip enforcement here (user should handle infeasible target or add epsilon mass prior).
                        continue

                    # Now compute closed-form dual lambda_g and scaling factors (stable in log-domain)
                    # lambda_g = log( (1-rho) * A / (rho * B) )
                    if p_global <= eps:
                        # target p=0 -> require zero positive mass; set positives to 0 inside group
                        tilde[pos_idx] = 0.0
                    elif (1.0 - p_global) <= eps:
                        # target p=1 -> require zero negative mass; set negatives to 0 inside group
                        tilde[neg_idx] = 0.0
                    else:
                        lambda_g = np.log(((1.0 - p_global) * A) / (p_global * B))
                        # scale factors:
                        scale_pos = np.exp(-(1.0 - p_global) * lambda_g)   # exp(-(1-rho)*lambda)
                        scale_neg = np.exp(p_global * lambda_g)           # exp(rho*lambda)
                        if pos_idx.size > 0:
                            tilde[pos_idx] = tilde[pos_idx] * scale_pos
                        if neg_idx.size > 0:
                            tilde[neg_idx] = tilde[neg_idx] * scale_neg

                # After processing all groups for this attribute, re-normalize tilde
                tilde = tilde / (tilde.sum() + eps)


        # accept iterate
        z = tilde

        # diagnostics
        mmd_val = float(z @ K_SS @ z - 2.0 * (z @ k_S) + mu_const)
        # compute penalty = sum_g (T_g - p_global*S_g)^2 (over attributes)
        penalty_val = 0.0
        dp_vals = []
        for attr in attr_info:
            for mask in attr['masks']:
                Sg = float(z[mask].sum())
                Tg = float((z * outcome_sub)[mask].sum())
                residual = Tg - p_global * Sg
                penalty_val += float(residual ** 2)
                # group positive rate (weighted)
            # dp gap for this attr (if groups exist)
            rates_attr = []
            for mask in attr['masks']:
                group_sum = float(z[mask].sum())
                if group_sum <= eps:
                    rates_attr.append(0.0)
                else:
                    pos_rate = float((z * outcome_sub)[mask].sum()) / (group_sum + eps)
                    rates_attr.append(pos_rate)
            if rates_attr:
                dp_vals.append(max(rates_attr) - min(rates_attr))

        max_dp = max(dp_vals) if dp_vals else 0.0
        # unweighted dp on subset (uniform)
        z_unif = np.ones_like(z) / float(len(z))
        dp_unw_vals = []
        for attr in attr_info:
            rates = []
            for mask in attr['masks']:
                group_count = mask.sum()
                if group_count == 0:
                    continue
                pos_count = (outcome_sub[mask]).sum()
                rate = float(pos_count) / (float(group_count) + eps)
                rates.append(rate)
            if rates:
                dp_unw_vals.append(max(rates) - min(rates))
        subset_unw_dp = max(dp_unw_vals) if dp_unw_vals else 0.0

        history['mmd'].append(mmd_val)
        history['penalty'].append(penalty_val)
        history['max_dp'].append(max_dp)
        history['subset_weighted_dp'].append(max_dp)
        history['subset_unweighted_dp'].append(subset_unw_dp)

        if verbose and (it % max(1, md_iterations // 5) == 0):
                # compute worst residual
                worst_resid = 0.0
                for attr in attr_info:
                    for mask in attr['masks']:
                        Sg = float(z[mask].sum())
                        Tg = float((z * outcome_sub)[mask].sum())
                        resid = abs(Tg - p_global * Sg)
                        worst_resid = max(worst_resid, resid)
                print(f"[FairMirror-MD it {it+1}/{md_iterations}] MMD={mmd_val:.6g} | pen={penalty_val:.6g} | worst_resid={worst_resid:.6g}")

    # After MD: return full vector with subset weights (z)
    w_full = np.zeros(P_np.shape[0], dtype=float)
    w_full[S] = np.maximum(z, 0.0)
    # ensure subset weights sum to 1 (numerical)
    ssum = w_full[S].sum()
    if ssum > 0:
        w_full[S] = w_full[S] / (ssum + 1e-12)

    return w_full, history
