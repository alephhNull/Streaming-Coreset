import numpy as np
import pandas as pd
import torch
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_openml

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)


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


# ---------------------- Dataset Loader (CORRECTED) ----------------------
def get_adult_data(num_samples=1000):
    """
    Robust loader for Adult dataset.
    Returns (P_tensor, sensitive_cols_dict, outcome_idx).
    sensitive_cols_dict contains raw pandas Series for each sensitive attribute.
    """
    print("Fetching Adult dataset...")
    df_original = None
    
    oml = fetch_openml(name="adult", version=2, as_frame=True)
    df_original = oml.frame
    df_original.rename(columns={'class': 'income'}, inplace=True)

    # Clean the entire DataFrame first
    df = df_original.replace([' ?', '?'], np.nan).dropna().reset_index(drop=True)
    df['income'] = df['income'].str.contains('>50K').astype(bool)

    # Separate target, sensitive, and feature columns from the aligned DataFrame
    y = df['income']
    sensitive_cols = {
        'sex': df['sex'].str.strip(),
        # 'race': df['race'].str.strip(),
        # 'marital-status': df['marital-status'].str.strip()
    }
    # Create feature set and dummify it BEFORE splitting to ensure consistent columns
    X = df.drop(columns=['income', 'sex', 'race', 'fnlwgt', 'education-num'], errors='ignore')
    X_dummified = pd.get_dummies(X, drop_first=True)

    # Now split based on indices
    if num_samples is not None and num_samples < len(X_dummified):
        indices = np.arange(len(X_dummified))
        train_indices, _ = train_test_split(indices, train_size=num_samples, stratify=y, random_state=42)

        X_train = X_dummified.iloc[train_indices]
        y_train = y.iloc[train_indices]
        sensitive_cols_train = {key: val.iloc[train_indices] for key, val in sensitive_cols.items()}
    else:
        X_train, y_train, sensitive_cols_train = X_dummified, y, sensitive_cols

    # Scale and convert to tensor
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    P_tensor = torch.from_numpy(X_train_scaled.astype(np.float32)).to(DEVICE)
    outcome_idx = torch.from_numpy(y_train.to_numpy(dtype=bool)).to(DEVICE)

    print(f"Loaded dataset: {P_tensor.shape[0]} samples, {P_tensor.shape[1]} features.")
    print("Sensitive attributes being tracked: 'sex', 'race' (multi-category)")
    return P_tensor, sensitive_cols_train, outcome_idx


# ---------------------- Kernels & QP Solver (Unchanged) ----------------------
def rbf_kernel_np(X, Y=None, sigma=1.0):
    X = np.asarray(X)
    Y = X if Y is None else np.asarray(Y)
    XX = np.sum(X**2, axis=1)[:, None]
    YY = np.sum(Y**2, axis=1)[None, :]
    D2 = np.maximum(XX + YY - 2.0 * (X @ Y.T), 0.0)
    return np.exp(-D2 / (2.0 * sigma**2))

def rbf_kernel_torch(X, Y=None, sigma=1.0):
    Y = X if Y is None else Y
    dist_sq = torch.cdist(X, Y, p=2) ** 2
    return torch.exp(-dist_sq / (2.0 * sigma ** 2))


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


# ---------------------- Alternating Fair Selection (unchanged except penalty call) ----------------------
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


# ---------------------- ADMM Refinement for Multi-Attribute Fairness (modified) ----------------------
class MultiFairADMMSubset:
    def __init__(self, K_SS_torch, k_S_torch, sensitive_cols_sub, outcome_mask_sub,
                 mu_pi_const=0.0, marginal_p=0.0, rho=1.0, lambda_fair=0.1, lr_fair=0.02, projection_steps=40, device=DEVICE):
        self.K = K_SS_torch.to(device)
        self.k = k_S_torch.to(device)
        self.device = device
        self.N = self.K.shape[0]
        self.rho, self.lambda_fair, self.lr_fair, self.proj_steps = rho, lambda_fair, lr_fair, projection_steps
        self.eps = 1e-9
        self.mu_const = float(mu_pi_const)
        self.inv = torch.pinverse(self.K + (self.rho / 2.0) * torch.eye(self.N, device=device))

        # Pre-process sensitive attributes into masks/labels for the subset
        self.fairness_attrs = []
        for key, s_col in sensitive_cols_sub.items():
            groups = s_col.unique()
            group_masks = [(s_col == g).to_numpy(dtype=bool) for g in groups]
            self.fairness_attrs.append({
                'name': key,
                'groups': groups,
                'masks': [torch.from_numpy(m).to(device) for m in group_masks]
            })
        self.outcome_mask = outcome_mask_sub.to(device)

        # marginal positive rate (constant) - used to form linear residuals T_g - p*S_g
        self.marginal_p = float(marginal_p)

    def _total_fairness_penalty(self, z):
        total_penalty = 0.0
        p = self.marginal_p
        for attr in self.fairness_attrs:
            for mask in attr['masks']:
                S_g = z[mask].sum()
                T_g = z[(mask & self.outcome_mask)].sum()
                residual = T_g - p * S_g
                total_penalty += float(residual ** 2)
        return total_penalty

    def _dp_gaps(self, z):
        # returns dict of dp gaps per attribute for current z (weighted)
        dp_gaps = {}
        for attr in self.fairness_attrs:
            rates = []
            for mask in attr['masks']:
                group_sum = z[mask].sum()
                pos_rate = z[(mask & self.outcome_mask)].sum() / (group_sum + self.eps)
                rates.append(float(pos_rate))
            dp_gaps[attr['name']] = (max(rates) - min(rates)) if rates else 0.0
        return dp_gaps

    def _unweighted_dp_gaps(self):
        # uniform weights across the subset (unweighted parity among chosen subset)
        z_uniform = torch.ones(self.N, device=self.device) / float(self.N)
        dp_gaps = {}
        for attr in self.fairness_attrs:
            rates = []
            for mask in attr['masks']:
                group_inds = mask.nonzero(as_tuple=False).squeeze()
                # selected uniform weight on subset -> compute fraction positive among group's members in subset
                group_count = (mask).sum()
                if group_count == 0:
                    continue
                pos_count = (self.outcome_mask & mask).sum()
                rate = float(pos_count) / (float(group_count) + self.eps)
                rates.append(rate)
            if rates:
                dp_gaps[attr['name']] = max(rates) - min(rates)
        return dp_gaps

    def solve(self, iterations=200, verbose=False):
        w = torch.ones(self.N, device=self.device) / self.N
        z = w.clone()
        u = torch.zeros(self.N, device=self.device)

        history = {'mmd': [], 'max_dp': [], 'penalty': [], 'subset_weighted_dp': [], 'subset_unweighted_dp': []}

        for it in range(iterations):
            # w-update (closed form)
            w = self.inv @ (self.k + (self.rho / 2.0) * (z - u))

            # v and z updates (explicit gradient on z)
            v = w + u
            z_var = v.clone().detach()

            for _ in range(self.proj_steps):
                # Compute per-attribute group statistics (vectorized)
                grad_penalty = torch.zeros_like(z_var)
                p = self.marginal_p
                for attr in self.fairness_attrs:
                    masks = attr['masks']
                    for mask in masks:
                        S_g = z_var[mask].sum()
                        T_g = z_var[(mask & self.outcome_mask)].sum()
                        # residual linear form
                        residual = T_g - p * S_g
                        # gradient contribution for members: 2 * residual * (y_i - p)
                        if residual.abs().item() == 0:
                            continue
                        coeff = 2.0 * residual
                        y_mask = self.outcome_mask[mask].to(z_var.dtype)
                        grad_penalty[mask] += coeff * (y_mask - p)

                # explicit gradient of full loss: rho*(z-v) + lambda_fair * grad_penalty
                grad_loss = self.rho * (z_var - v) + self.lambda_fair * grad_penalty

                # gradient descent step
                z_var.data.sub_(self.lr_fair * grad_loss.data)

                # Simplex projection (same as before)
                z_var_flat = z_var.flatten()
                if torch.any(z_var_flat < 0) or not torch.isclose(z_var_flat.sum(), torch.tensor(1.0, device=self.device)):
                    u_sorted, _ = torch.sort(z_var_flat, descending=True)
                    cssv = torch.cumsum(u_sorted, dim=0)
                    idx = torch.nonzero(u_sorted * torch.arange(1, self.N + 1, device=self.device) > (cssv - 1))
                    if len(idx) > 0:
                        rho_idx = idx.max()
                        theta = (cssv[rho_idx] - 1) / (rho_idx + 1)
                        z_var.data = torch.clamp(z_var.data - theta, min=0)
                    else: # Fallback if all points are below the line
                        z_var.data = torch.clamp(z_var.data, min=0)
                        z_var.data /= (z_var.data.sum() + self.eps)

            z = z_var.detach()
            u = u + (w - z)

            # --- Logging / diagnostics ---
            # MMD for the full-weight vector that places z on subset indices and 0 elsewhere
            mmd_val = float(z @ self.K @ z - 2.0 * (z @ self.k) + self.mu_const)
            penalty_val = float(self._total_fairness_penalty(z))
            dp_gaps = self._dp_gaps(z)
            max_dp = max(dp_gaps.values()) if dp_gaps else 0.0

            # Weighted dp on subset (z) and unweighted dp (uniform on subset)
            subset_weighted_dp = max(dp_gaps.values()) if dp_gaps else 0.0
            subset_unweighted_dict = self._unweighted_dp_gaps()
            subset_unweighted_dp = max(subset_unweighted_dict.values()) if subset_unweighted_dict else 0.0

            history['mmd'].append(mmd_val)
            history['max_dp'].append(max_dp)
            history['penalty'].append(penalty_val)
            history['subset_weighted_dp'].append(subset_weighted_dp)
            history['subset_unweighted_dp'].append(subset_unweighted_dp)

            if verbose and (it % max(1, iterations // 10) == 0 or it == iterations-1):
                print(f"[ADMM it {it+1}/{iterations}] MMD={mmd_val:.6g} | max_DP={max_dp:.6g} | pen={penalty_val:.6g} | subset_unw_dp={subset_unweighted_dp:.6g} | subset_w_dp={subset_weighted_dp:.6g}")

        return z, history


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




# ---------------------- Evaluation and Baselines ----------------------
def calculate_metrics(weights, K_full_np, mu_pi_np, sensitive_cols, outcome_idx, selected_indices=None):
    # MMD
    mmd_val = weights @ K_full_np @ weights - 2 * (weights @ mu_pi_np) + K_full_np.mean()
    # Weighted Fairness
    eps = 1e-9
    outcome_np = outcome_idx.cpu().numpy()

    dp_gaps_weighted, rates_weighted = compute_weighted_dp_gaps(weights, sensitive_cols, outcome_np, eps)

    # Unweighted fairness on selected indices
    if selected_indices is None:
        # infer selected indices as indices with weight > 0 (or top-k)
        selected_indices = np.where(weights > 0)[0]
    dp_gaps_unweighted, rates_unweighted = compute_unweighted_dp_gaps(selected_indices, sensitive_cols, outcome_np)

    return mmd_val, dp_gaps_weighted, dp_gaps_unweighted, rates_weighted, rates_unweighted


def baseline_random(P, m):
    N = P.shape[0]
    S = np.random.choice(N, m, replace=False)
    w = np.zeros(N)
    w[S] = 1.0 / m
    return w


def baseline_standard_wkh(P_np, K, m):
    # Simple greedy selection for standard WKH
    mu_pi = K.mean(axis=0)
    S = []
    for _ in range(m):
        best_j = -1; best_val = -np.inf
        for j in range(P_np.shape[0]):
            if j in S: continue
            val = mu_pi[j] - K[j, S].mean() if S else mu_pi[j]
            if val > best_val: best_val, best_j = val, j
        S.append(best_j)
    S = np.array(S)
    w = np.zeros(P_np.shape[0])
    w[S] = qp_weights_slsqp(K[np.ix_(S,S)], K[:,S].mean(axis=0))
    return w


def baseline_multi_fair_admm(P_tensor, sensitive_cols, outcome_idx, sigma, m):
    P_np = P_tensor.cpu().numpy()
    S, K_full, selection_history = weighted_kernel_herding_frank_wolfe_alternate(
        P_np, m, sigma, sensitive_cols, outcome_idx,
        alternate_freq=2, verbose=False
    )

    K_SS_torch = torch.from_numpy(K_full[np.ix_(S, S)]).float()
    k_S_torch = torch.from_numpy(K_full[:, S].mean(axis=0)).float()
    sensitive_cols_sub = {key: val.iloc[S] for key, val in sensitive_cols.items()}
    outcome_sub = outcome_idx[S]

    mu_const = float(K_full.mean())

    # compute global marginal positive p from full dataset and pass to ADMM
    marginal_p = float(outcome_idx.cpu().numpy().mean())

    admm = MultiFairADMMSubset(
        K_SS_torch, k_S_torch, sensitive_cols_sub, outcome_sub,
        mu_pi_const=mu_const, marginal_p=marginal_p, lambda_fair=0.1, rho=10.0, lr_fair=0.001
    )
    w_sub, history = admm.solve(iterations=200, verbose=True)
    w_full = np.zeros(P_np.shape[0])
    w_full[S] = w_sub.cpu().numpy()

    # attach selection history to returned history so caller can inspect selection diagnostics
    history['selection_history'] = selection_history
    return w_full, history


# ---------------------- Plotting weighted rates across baselines (updated orientation) ----------------------
def plot_weighted_rates_across_baselines(baseline_rates, sensitive_cols):
    """Creates grouped bar plots (one figure per sensitive attribute) showing
    weighted positive rates for each subgroup with baselines on the x-axis and subgroup bars.

    baseline_rates: dict baseline_name -> {attr: {group: rate}}
    sensitive_cols: dict attr -> pandas.Series (for canonical group ordering)
    """
    # Build dataframe rows with explicit baseline order and group order
    baseline_order = list(baseline_rates.keys())
    rows = []
    for baseline in baseline_order:
        rates = baseline_rates.get(baseline, {}) or {}
        for attr, s_col in sensitive_cols.items():
            # preserve canonical ordering of groups as they appear in the full dataset
            groups = list(s_col.unique())
            for g in groups:
                rate = rates.get(attr, {}).get(g, 0.0)
                rows.append({"Baseline": baseline, "Attribute": attr, "Group": str(g), "Rate": rate})

    df_plot = pd.DataFrame(rows)
    if df_plot.empty:
        print("No baseline rates available to plot.")
        return

    # Plot one figure per attribute with baselines on x-axis and subgroup bars (hue=Group)
    for attr in df_plot['Attribute'].unique():
        df_attr = df_plot[df_plot['Attribute'] == attr]
        plt.figure(figsize=(8, 5))
        sns.barplot(data=df_attr, x='Baseline', y='Rate', hue='Group', order=baseline_order)
        plt.title(f'Weighted positive rates — {attr}')
        plt.ylabel('Weighted positive rate')
        plt.ylim(0, 1)
        plt.xlabel('Baseline')
        plt.xticks(rotation=10)
        plt.legend(title='Subgroup', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()


# ---------------------- Main Experiment Loop ----------------------
def main():
    M_CORESET = 50
    SIGMA_RBF = 10.0
    NUM_SAMPLES = 1000

    P, sensitive_cols, y = get_adult_data(num_samples=NUM_SAMPLES)
    P_np = P.cpu().numpy()
    K_full_np = rbf_kernel_np(P_np, sigma=SIGMA_RBF)
    mu_pi_np = K_full_np.mean(axis=0)

    baselines = {
        "Random": lambda: baseline_random(P, M_CORESET),
        "Standard WKH": lambda: baseline_standard_wkh(P_np, K_full_np, M_CORESET),
        # "Alternate + ADMM": lambda: baseline_multi_fair_admm(P, sensitive_cols, y, SIGMA_RBF, M_CORESET),
"Fair Mirror (MD)": lambda: baseline_fair_mirror(P, sensitive_cols, y, SIGMA_RBF, M_CORESET,
                                               select_alternate_freq=2, md_iterations=200, eta=1, verbose=True),
}

    results = []
    md_history = None
    baseline_rates = {}

    print("--- Evaluating baselines ---")
    for name, func in baselines.items():
        start = time.time()
        out = func()
        duration = time.time() - start

        # support functions that return (weights, history) or just weights
        if isinstance(out, tuple):
            weights, history = out
            if name == "Fair Mirror (MD)":
                md_history = history
                print("MD weights sum:", weights.sum())
        else:
            weights = out
            history = None

        # compute both weighted and unweighted dp gaps
        mmd, dp_w, dp_unw, rates_w, rates_unw = calculate_metrics(weights, K_full_np, mu_pi_np, sensitive_cols, y)
        max_dp = max(dp_w.values()) if dp_w else 0
        max_dp_unw = max(dp_unw.values()) if dp_unw else 0
        results.append({"Method": name, "MMD": mmd, "Max DP Gap (weighted)": max_dp, "Max DP Gap (unweighted)": max_dp_unw, "DP Gaps (weighted)": dp_w, "DP Gaps (unweighted)": dp_unw, "Time (s)": duration})

        # store weighted rates for plotting (may be attr->dict)
        baseline_rates[name] = rates_w

        print(f"{name:<20} | MMD={mmd:12.8f} | Max DP (w)={max_dp:12.8f} | Max DP (u)={max_dp_unw:12.8f} | time={duration:.2f}s")
        for key, val in dp_w.items():
            print(f"  └─ Weighted DP Gap ({key}): {val:.8f}")
        for key, val in dp_unw.items():
            print(f"  └─ Unweighted DP Gap ({key}): {val:.8f}")

    # --- Plotting Results (existing scatter) ---
    print("--- Generating plot ---")
    df_results = pd.DataFrame(results)
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(8, 6))

    # extend markers to include the new method
    sns.scatterplot(data=df_results, x='Max DP Gap (weighted)', y='MMD', hue='Method', s=200, ax=ax, palette='viridis', style='Method', markers=['o', 's', 'P', 'X'])

    ax.set_title('Fairness-MMD Trade-off (weighted DP gap)', fontsize=16)
    ax.set_xlabel('Max Demographic Parity Gap (Weighted) (Lower is Fairer)', fontsize=12)
    ax.set_ylabel('Maximum Mean Discrepancy (Lower is Better)', fontsize=12)
    ax.set_yscale('log')
    ax.legend(title='Method', fontsize=10)
    plt.tight_layout()
    plt.show()

    # --- New: Plot weighted positive rates per subgroup for each baseline ---
    print("--- Plotting weighted positive rates per subgroup for each baseline ---")
    plot_weighted_rates_across_baselines(baseline_rates, sensitive_cols)

    # --- If MD history was collected, show iterative diagnostics ---
    if md_history is not None:
        iters = np.arange(1, len(md_history['mmd']) + 1)
        fig, ax1 = plt.subplots(figsize=(9, 5))
        ax1.plot(iters, md_history['mmd'], linestyle='-', marker='o')
        ax1.set_ylabel('MMD (subset objective)', fontsize=12)
        ax1.set_yscale('log')
        ax1.set_xlabel('MD iteration')

        ax2 = ax1.twinx()
        ax2.plot(iters, md_history['subset_weighted_dp'], linestyle='--', marker='x', label='subset weighted dp')
        ax2.plot(iters, md_history['subset_unweighted_dp'], linestyle=':', marker='.', label='subset unweighted dp')
        ax2.set_ylabel('Max DP gap (subset)', fontsize=12)

        plt.title('MD iteration diagnostics: MMD vs Max DP gap (subset weighted & unweighted)')
        ax1.grid(True)
        ax2.legend()
        fig.tight_layout()
        plt.show()

        # quick check
        first_mmd, last_mmd = md_history['mmd'][0], md_history['mmd'][-1]
        first_dp_w, last_dp_w = md_history['subset_weighted_dp'][0], md_history['subset_weighted_dp'][-1]
        first_dp_u, last_dp_u = md_history['subset_unweighted_dp'][0], md_history['subset_unweighted_dp'][-1]
        print(f"MD diagnostics: MMD {first_mmd:.6g} -> {last_mmd:.6g} | Subset Weighted DP {first_dp_w:.6g} -> {last_dp_w:.6g} | Subset Unweighted DP {first_dp_u:.6g} -> {last_dp_u:.6g}")
        if last_mmd < first_mmd and last_dp_w <= first_dp_w:
            print("Both MMD and Max DP decreased during MD (good).")
        elif last_mmd < first_mmd:
            print("MMD improved during MD, but Max DP did not improve (trade-off).")
        elif last_dp_w < first_dp_w:
            print("Max DP decreased during MD, but MMD did not improve (trade-off).")
        else:
            print("No clear improvement on both fronts — consider tuning lambda_fair / rho / lr_fair or running more iterations.")


if __name__ == '__main__':
    main()
