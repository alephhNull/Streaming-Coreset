"""
Two-stage baselines for weighted kernel herding:
 - Standard WKH: Franke-Wolfe greedy selection -> QP weights (nonneg sum-to-one)
 - Random Sampling: random selection -> QP weights
 - Fair ADMM Herding (baseline): fairness-aware greedy selection -> ADMM weight refinement on subset

Author: adapted for your streaming-coreset experiment
"""

import numpy as np
import pandas as pd
import torch
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# --- Dataset loader (robust: ucimlrepo fallback to openml) ------------------
try:
    from ucimlrepo import fetch_ucirepo
except Exception:
    fetch_ucirepo = None
from sklearn.datasets import fetch_openml

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

def get_adult_data(num_samples=2000):
    """Robust loader for Adult dataset. Returns (P_tensor, sensitive_idx, outcome_idx)."""
    print("Fetching Adult dataset...")
    df_original = None
    features_df = None
    targets_df = None

    # try ucimlrepo first
    if fetch_ucirepo is not None:
        try:
            ds = fetch_ucirepo(id=2)
            data_block = ds.data
            features_df = getattr(data_block, 'features', None)
            df_original = getattr(data_block, 'original', None)
            targets_df = getattr(data_block, 'targets', None)
        except Exception as e:
            print("ucimlrepo failed:", repr(e))

    # fallback to openml
    if df_original is None:
        oml = fetch_openml(name="adult", version=2, as_frame=True)
        df_original = oml.frame
        if 'class' in df_original.columns:
            targets_df = df_original[['class']]
            features_df = df_original.drop(columns=['class'])

    # pick features_df if available else use original
    X = features_df.copy() if features_df is not None else df_original.drop(columns=[c for c in df_original.columns if df_original[c].nunique() == 2][-1])
    # Try to detect target
    if targets_df is not None:
        if isinstance(targets_df, pd.DataFrame):
            y = targets_df.iloc[:, 0].copy()
        else:
            y = targets_df.copy()
    else:
        # find a binary column likely to be income
        cand = [c for c in df_original.columns if df_original[c].nunique() == 2]
        if len(cand) == 0:
            raise RuntimeError("Couldn't find binary target in dataset.")
        y = df_original[cand[-1]].copy()

    # Clean missing tokens
    X = X.replace([' ?', '?', 'None', 'nan'], np.nan)
    y = y.replace([' ?', '?', 'None', 'nan'], np.nan)
    valid_mask = ~y.isnull()
    X = X.loc[valid_mask].reset_index(drop=True)
    y = y.loc[valid_mask].reset_index(drop=True)

    # boolean outcome (>=50k)
    if y.dtype == object or y.dtype.name == 'category':
        y_bool = y.astype(str).str.contains('>50K').fillna(False).astype(bool)
    else:
        # numeric or already bool
        if set(y.unique()) <= {0, 1}:
            y_bool = y.astype(bool)
        else:
            # fallback: string check
            y_bool = y.astype(str).str.contains('>50K').fillna(False).astype(bool)

    # drop some columns often not useful
    for col in ['fnlwgt', 'education']:
        if col in X.columns:
            X = X.drop(columns=[col])

    # one-hot categoricals
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    # coerce numeric & impute mean
    X = X.apply(pd.to_numeric, errors='coerce')
    allnan = X.columns[X.isna().all()].tolist()
    if allnan:
        X = X.drop(columns=allnan)
    if X.isna().any().any():
        X = X.fillna(X.mean())

    # align
    X, y_bool = X.align(y_bool, axis=0, join='inner')

    # sample
    if num_samples is not None and num_samples < len(X):
        X_train, _, y_train, _ = train_test_split(X, y_bool, train_size=num_samples, stratify=y_bool, random_state=42)
    else:
        X_train = X
        y_train = y_bool

    # scale numeric
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        scaler = StandardScaler()
        X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])

    # sensitive attribute: try sex_Male or similar
    sens_candidates = [c for c in X_train.columns if 'sex' in c.lower() or 'gender' in c.lower()]
    if len(sens_candidates) == 0:
        # try from original before get_dummies
        if 'sex' in df_original.columns:
            sens_raw = df_original.loc[X_train.index, 'sex'].replace([' ?', '?'], np.nan).fillna('Male')
            sensitive = sens_raw.astype(str).str.contains('Male').to_numpy(dtype=bool)
        else:
            raise RuntimeError("Couldn't find 'sex' column after preprocessing.")
    else:
        if 'sex_Male' in X_train.columns:
            sensitive = X_train['sex_Male'].to_numpy(dtype=bool)
        else:
            sensitive = (X_train[sens_candidates[0]] == 1).to_numpy(dtype=bool)

    outcome = y_train.to_numpy(dtype=bool)
    P_np = X_train.to_numpy(dtype=np.float32)
    P_tensor = torch.from_numpy(P_np).to(DEVICE)
    sensitive_idx = torch.from_numpy(sensitive).to(DEVICE)
    outcome_idx = torch.from_numpy(outcome).to(DEVICE)

    print(f"Loaded dataset: {P_np.shape[0]} samples, {P_np.shape[1]} features.")
    return P_tensor, sensitive_idx, outcome_idx

# --- kernels ----------------------------------------------------------------
def rbf_kernel_np(X, Y=None, sigma=1.0):
    """RBF kernel: K(x,y)=exp(-||x-y||^2 / (2 sigma^2))."""
    X = np.asarray(X)
    if Y is None:
        Y = X
    else:
        Y = np.asarray(Y)
    XX = np.sum(X**2, axis=1)[:, None]
    YY = np.sum(Y**2, axis=1)[None, :]
    D2 = XX + YY - 2.0 * (X @ Y.T)
    D2 = np.maximum(D2, 0.0)
    gamma = 1.0 / (2.0 * (sigma ** 2))
    return np.exp(-gamma * D2)

def rbf_kernel_torch(X, Y=None, sigma=1.0):
    if Y is None:
        Y = X
    dist_sq = torch.cdist(X, Y, p=2) ** 2
    return torch.exp(-dist_sq / (2.0 * sigma ** 2))

# --- selection: Franke-Wolfe / greedy weighted kernel herding -----------------
def weighted_kernel_herding_frank_wolfe(
    X_np, m, sigma=1.0, ridge=1e-8, verbose=False, fairness_beta=0.0,
    sensitive_mask=None, outcome_mask=None
):
    """
    Greedy Franke-Wolfe style selection:
      - fairness_beta > 0 includes a small fairness penalty when evaluating candidates (penalizes DP violation)
      - returns selected indices (length m) and last intermediate weights on S
    """
    X = np.asarray(X_np)
    N = X.shape[0]
    if m < 1 or m > N:
        raise ValueError("m must be between 1 and N")

    K = rbf_kernel_np(X, X, sigma=sigma)  # full kernel (N,N)
    mu_pi = K.mean(axis=0)                # length N
    selected = []
    current_embedding = np.zeros(N, dtype=float)
    remaining = set(range(N))

    # small helper to compute candidate fairness violation given candidate set S and candidate weights (w)
    def dp_violation_for_weights(w_sub, S_idx):
        if fairness_beta == 0.0 or sensitive_mask is None or outcome_mask is None:
            return 0.0
        # build full-length weights for DP calculation (zeros outside S)
        w_full = np.zeros(N, dtype=float)
        w_full[np.asarray(S_idx, dtype=int)] = w_sub
        # compute male/female masks (numpy)
        male_mask = np.asarray(sensitive_mask.cpu().numpy(), dtype=bool)
        female_mask = ~male_mask
        outcome_mask_np = np.asarray(outcome_mask.cpu().numpy(), dtype=bool)
        # avoid zero division
        eps = 1e-8
        p_male = w_full[male_mask & outcome_mask_np].sum() / (w_full[male_mask].sum() + eps)
        p_female = w_full[female_mask & outcome_mask_np].sum() / (w_full[female_mask].sum() + eps)
        return abs(p_male - p_female)

    for t in range(m):
        best_idx = None
        best_score = np.inf
        # Evaluate each candidate (costly: O(N^2) inner worst-case)
        for j in remaining:
            S_candidate = selected + [j]
            S_arr = np.array(S_candidate, dtype=int)
            K_SS = K[np.ix_(S_arr, S_arr)]
            K_XS = K[:, S_arr]
            z = K_XS.mean(axis=0)
            # solve K_SS w = z (closed-form intermediate weights used for selection)
            try:
                A = K_SS + ridge * np.eye(K_SS.shape[0])
                w_candidate = np.linalg.solve(A, z)
            except np.linalg.LinAlgError:
                w_candidate = np.linalg.pinv(K_SS + ridge * np.eye(K_SS.shape[0])) @ z

            # compute candidate embedding residual (scalar) as current measure
            # current_embedding_for_candidate = K_XS @ w_candidate
            emb = K_XS @ w_candidate
            # score = mean_{i} [ (embedding_i - mu_pi_i) ]? We use the norm (squared) of difference as score
            residual = emb - mu_pi
            score_mmd = float(residual @ residual)  # squared norm
            # fairness penalty (if requested)
            fair_pen = dp_violation_for_weights(w_candidate, S_arr)
            score = score_mmd + fairness_beta * (fair_pen ** 2)
            if score < best_score:
                best_score = score
                best_idx = j

        selected.append(best_idx)
        remaining.discard(best_idx)
        # update current_embedding with current selected set's weights
        S = np.array(selected, dtype=int)
        K_SS = K[np.ix_(S, S)]
        K_XS = K[:, S]
        z = K_XS.mean(axis=0)
        try:
            A = K_SS + ridge * np.eye(K_SS.shape[0])
            w_current = np.linalg.solve(A, z)
        except np.linalg.LinAlgError:
            w_current = np.linalg.pinv(K_SS + ridge * np.eye(K_SS.shape[0])) @ z
        current_embedding = (K_XS @ w_current)

        if verbose:
            print(f"[sel {t+1}/{m}] picked {best_idx} | score {best_score:.6g} | fair_beta {fairness_beta}")

    # final weights on selected set (we'll re-optimize with QP or ADMM later)
    S_final = np.array(selected, dtype=int)
    K_SS = K[np.ix_(S_final, S_final)]
    K_XS = K[:, S_final]
    z = K_XS.mean(axis=0)
    try:
        final_w = np.linalg.solve(K_SS + ridge * np.eye(K_SS.shape[0]), z)
    except np.linalg.LinAlgError:
        final_w = np.linalg.pinv(K_SS + ridge * np.eye(K_SS.shape[0])) @ z

    return S_final, final_w, K  # return full K for reuse

# --- QP solver for weights (SLSQP) ------------------------------------------
def qp_weights_slsqp(K_SS, k_S, ridge=1e-8, init=None, maxiter=200):
    """
    Solve: min_w w^T K_SS w - 2 k_S^T w
           s.t. sum(w) = 1, w_i >= 0

    We use scipy.optimize.minimize (SLSQP) with analytic gradient.
    Inputs: K_SS (m,m) numpy, k_S (m,) numpy
    Returns: w (m,) numpy
    """
    m = K_SS.shape[0]
    P = K_SS + ridge * np.eye(m)
    q = -2.0 * k_S  # careful: objective = w^T K w - 2 k^T w = 0.5 * (2K) w^2 - 2 k^T w

    if init is None:
        # try closed-form then project to simplex as warm start
        try:
            Kinv = np.linalg.inv(P)
            one = np.ones(m)
            Kinv_k = Kinv @ k_S
            Kinv_1 = Kinv @ one
            denom = one @ Kinv_1
            if abs(denom) < 1e-12:
                alpha = 0.0
            else:
                alpha = (one @ Kinv_k - 1.0) / denom
            w0 = Kinv_k - alpha * Kinv_1
            # if negative entries appear, project to simplex for feasible init
            w0 = np.maximum(w0, 0.0)
            s = w0.sum()
            if s == 0:
                w0 = np.ones(m) / m
            else:
                w0 = w0 / s
        except np.linalg.LinAlgError:
            w0 = np.ones(m) / m
    else:
        w0 = np.asarray(init, dtype=float)

    # objective and jac
    def obj(w):
        return float(w @ (P @ w) - 2.0 * (k_S @ w))

    def jac(w):
        # gradient: 2 P w - 2 k_S
        return 2.0 * (P @ w) - 2.0 * k_S

    # constraints: sum(w) = 1
    cons = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0, 'jac': lambda w: np.ones_like(w)}
    bounds = [(0.0, None) for _ in range(m)]
    res = minimize(fun=obj, x0=w0, jac=jac, bounds=bounds, constraints=cons,
                   method='SLSQP', options={'maxiter': maxiter, 'ftol': 1e-9, 'disp': False})
    if not res.success:
        # fallback: project solution candidate
        w = res.x if res.x is not None else w0
        # ensure feasibility
        w = np.maximum(w, 0.0)
        s = w.sum()
        if s == 0:
            w = np.ones(m) / m
        else:
            w = w / s
        return w
    return res.x


# --- SAFER ADMM weight refinement (fairness-aware) for subset -------------
def project_onto_simplex_torch(y: torch.Tensor) -> torch.Tensor:
    y_flat = y.view(-1)
    n = y_flat.numel()
    if n == 0:
        return y
    u, _ = torch.sort(y_flat, descending=True)
    cssv = torch.cumsum(u, dim=0)
    rho_idx = torch.nonzero(u * torch.arange(1, n + 1, device=y.device) > (cssv - 1.0), as_tuple=False)
    if rho_idx.numel() == 0:
        theta = 0.0
    else:
        rho = rho_idx[-1].item() + 1
        theta = (cssv[rho - 1] - 1.0) / rho
    x = torch.clamp(y_flat - theta, min=0.0)
    return x.view_as(y)

class FairADMMSubset:
    """
    Safer ADMM for subset weights:
      - z-step minimizes (rho/2)||z-v||^2 + lambda_fair * dp(z)^2
    """
    def __init__(self, K_SS_torch, k_S_torch, sensitive_mask_sub, outcome_mask_sub,
                 rho=1.0, lambda_fair=0.05, lr_fair=0.02, projection_steps=40, device=DEVICE):
        self.K = K_SS_torch.to(device)
        self.k = k_S_torch.to(device)
        self.N = self.K.shape[0]
        self.rho = float(rho)
        self.lambda_fair = float(lambda_fair)
        self.lr_fair = float(lr_fair)
        self.projection_steps = int(projection_steps)
        self.device = device

        I = torch.eye(self.N, device=self.device)
        try:
            self.inv = torch.inverse(self.K + (self.rho / 2.0) * I)
        except RuntimeError:
            self.inv = torch.pinverse(self.K + (self.rho / 2.0) * I)

        # masks on the subset (bool tensors)
        self.male_mask = sensitive_mask_sub.to(self.device).bool()
        self.female_mask = (~self.male_mask)
        self.outcome_mask = outcome_mask_sub.to(self.device).bool()
        self.male_outcome_mask = self.male_mask & self.outcome_mask
        self.female_outcome_mask = self.female_mask & self.outcome_mask
        self.eps = 1e-9

        # target group mass priors inside subset S (to discourage collapse)
        total = float(self.male_mask.sum().item() + self.female_mask.sum().item())
        if total <= 0:
            self.pi_male = 0.5
            self.pi_female = 0.5
        else:
            self.pi_male = float(self.male_mask.sum().item()) / total
            self.pi_female = float(self.female_mask.sum().item()) / total

    def _dp_violation(self, z: torch.Tensor) -> torch.Tensor:
        sum_male = z[self.male_mask].sum()
        sum_female = z[self.female_mask].sum()
        p_male = z[self.male_outcome_mask].sum() / (sum_male + self.eps)
        p_female = z[self.female_outcome_mask].sum() / (sum_female + self.eps)
        return p_male - p_female

    def _mmd_obj(self, w: torch.Tensor) -> torch.Tensor:
        return (w @ (self.K @ w)) - 2.0 * (w @ self.k)

    def solve(self, iterations=200, return_history=False, verbose=False):
        w = torch.ones(self.N, device=self.device) / float(self.N)
        z = w.clone()
        u = torch.zeros(self.N, device=self.device)

        history = {k: [] for k in ('mmd_w', 'dp_w', 'mmd_z', 'dp_z', 'primal_resid', 'dual_resid')}

        for t in range(iterations):
            # w-update
            rhs = self.k + (self.rho / 2.0) * (z - u)
            w = self.inv @ rhs

            # z-update: minimize (rho/2)||z-v||^2 + lambda*(dp)^2 + mass_reg*((sum_male - pi_m)^2 + ...)
            v = w + u
            z_var = v.clone().detach().requires_grad_(True)
            for _ in range(self.projection_steps):
                dp = self._dp_violation(z_var)
                loss = (self.rho / 2.0) * torch.sum((z_var - v) ** 2) + self.lambda_fair * (dp ** 2)
                grad, = torch.autograd.grad(loss, z_var, create_graph=False, retain_graph=False)
                z_var = z_var - self.lr_fair * grad
                # project back to simplex
                z_var = project_onto_simplex_torch(z_var).detach().requires_grad_(True)

            z_prev = z.clone()
            z = z_var.detach()

            # dual update
            u = u + w - z

            if return_history:
                mmdw = float(self._mmd_obj(w).item())
                dpw = float(self._dp_violation(w).item())
                mmdz = float(self._mmd_obj(z).item())
                dpz = float(self._dp_violation(z).item())
                primal = float(torch.norm(w - z).item())
                dual = float(self.rho * torch.norm(z - z_prev).item())

                history['mmd_w'].append(mmdw)
                history['dp_w'].append(dpw)
                history['mmd_z'].append(mmdz)
                history['dp_z'].append(dpz)
                history['primal_resid'].append(primal)
                history['dual_resid'].append(dual)

            if verbose and (t % max(1, iterations // 10) == 0 or t == iterations - 1):
                if return_history:
                    print(f"ADMM iter {t:4d}: mmd_w={history['mmd_w'][-1]:.6f}, |dp_w|={abs(history['dp_w'][-1]):.6f}, primal={history['primal_resid'][-1]:.6f}")
                else:
                    print(f"ADMM iter {t:4d}")

        if return_history:
            return w.detach(), z.detach(), history
        return w.detach()



# --- evaluation metrics -----------------------------------------------------
def calculate_metrics(weights, K_pp, k_pq, sensitive_idx, outcome_idx):
    """Return (mmd, fairness_violation) given:
       - weights: torch tensor shape (N,)
       - K_pp: torch tensor (N,N)
       - k_pq: torch tensor (N,)
       - sensitive_idx, outcome_idx: bool torch tensors (N,)
    """
    w = weights.view(-1).float()
    wKw = float(w @ (K_pp @ w))
    wkp = float(w @ k_pq)
    mean_K = float(K_pp.mean())
    mmd_val = wKw - 2.0 * wkp + mean_K

    male_idx = sensitive_idx
    female_idx = ~sensitive_idx
    male_outcome = male_idx & outcome_idx
    female_outcome = female_idx & outcome_idx
    eps = 1e-8
    p_male = float(w[male_outcome].sum() / (w[male_idx].sum() + eps))
    p_female = float(w[female_outcome].sum() / (w[female_idx].sum() + eps))
    fairness_violation = abs(p_male - p_female)
    return mmd_val, fairness_violation

# --- Baseline wrappers ------------------------------------------------------
def baseline_standard_wkh(P_tensor, sigma, m, ridge=1e-8, return_indices=False, verbose=False):
    """Standard WKH: greedy FW selection -> QP weights (nonneg, sum=1)"""
    P_np = P_tensor.cpu().numpy()
    S, w_intermediate, K = weighted_kernel_herding_frank_wolfe(
        P_np, m, sigma=sigma, ridge=ridge, verbose=verbose)
    # compute K_SS and k_S for QP
    K_SS = K[np.ix_(S, S)]
    K_XS = K[:, S]
    k_S = K_XS.mean(axis=0)
    w_sub = qp_weights_slsqp(K_SS, k_S, ridge=ridge)
    # build full weight vector
    w_full = np.zeros(P_np.shape[0], dtype=float)
    w_full[S] = w_sub
    w_full_torch = torch.from_numpy(w_full.astype(np.float32)).to(DEVICE)
    if return_indices:
        return w_full_torch, S
    return w_full_torch

def baseline_random(P_tensor, m, seed=42, return_indices=False):
    """
    Random sampling baseline -> UNIFORM weights on the selected subset.

    Returns a full-length weight vector `w_full_torch` (shape N,) where
    w_full_torch[i] = 1/m for i in the selected subset S, and 0 otherwise.
    If return_indices is True, also returns S (numpy array of indices).
    """
    N = P_tensor.shape[0]
    rng = np.random.default_rng(seed)
    S = rng.choice(N, size=m, replace=False)

    # uniform weights on subset S
    w_sub = np.ones(m, dtype=float) / float(m)

    # build full weight vector (numpy -> torch)
    w_full = np.zeros(N, dtype=float)
    w_full[S] = w_sub
    w_full_torch = torch.from_numpy(w_full.astype(np.float32)).to(DEVICE)

    if return_indices:
        return w_full_torch, S
    return w_full_torch


# --- UPDATED baseline_fair_admm: accept and return_history ----------
def baseline_fair_admm(
    P_tensor, sensitive_idx, outcome_idx, sigma, m,
    rho=1.0, lr_fair=0.1, proj_steps=20,
    ridge=1e-8, fairness_beta=0.0, dp_frequency=None, lambda_fair=1.0, return_history=False, verbose=False
):
    """
    Fair baseline pipeline:
     - fairness-aware greedy selection (alternate frequency controlled by dp_frequency)
     - ADMM refinement on the selected subset to enforce fairness approximately

    returns: (w_full_torch, S) or (w_full_torch, S, history) if return_history=True
    """
    P_np = P_tensor.cpu().numpy()
    S, _, K = weighted_kernel_herding_frank_wolfe_alternate(
        P_np, m, sigma=sigma, ridge=ridge, verbose=verbose,
        fairness_beta=fairness_beta, sensitive_mask=sensitive_idx,
        outcome_mask=outcome_idx, alternate_dp_freq=dp_frequency
    )

    if len(S) == 0:
        # fallback: uniform over dataset
        N = P_np.shape[0]
        w_full = np.ones(N, dtype=float) / float(N)
        w_full_torch = torch.from_numpy(w_full.astype(np.float32)).to(DEVICE)
        if return_history:
            return w_full_torch, S, {}
        return w_full_torch, S

    # prepare subset tensors and masks
    P_S = P_tensor[S, :].clone()
    K_SS_torch = rbf_kernel_torch(P_S, P_S, sigma=sigma).to(DEVICE)
    # compute k_S (target mean) as torch
    K_full_torch = rbf_kernel_torch(P_tensor, P_S, sigma=sigma).to(DEVICE)
    k_S_torch = K_full_torch.mean(dim=0)

    # masks restricted to S (torch bool)
    sens_np = sensitive_idx.cpu().numpy().astype(bool)
    out_np = outcome_idx.cpu().numpy().astype(bool)
    sens_sub = torch.from_numpy(sens_np[S]).to(DEVICE).bool()
    out_sub = torch.from_numpy(out_np[S]).to(DEVICE).bool()

    admm = FairADMMSubset(
        K_SS_torch, k_S_torch, sens_sub, out_sub,
        rho=rho, lambda_fair=lambda_fair,
        lr_fair=lr_fair, projection_steps=proj_steps, device=DEVICE
    )

    if return_history:
        w_sub_torch, z_sub_torch, history = admm.solve(iterations=200, return_history=True, verbose=verbose)
    else:
        w_sub_torch = admm.solve(iterations=200, return_history=False, verbose=verbose)

    # ensure nonneg and sum-to-one (ADMM may produce tiny negatives)
    w_sub = w_sub_torch.cpu().numpy()
    w_sub = np.maximum(w_sub, 0.0)
    s = w_sub.sum()
    if s <= 0:
        w_sub = np.ones_like(w_sub) / len(w_sub)
    else:
        w_sub = w_sub / s

    # build full weights
    w_full = np.zeros(P_np.shape[0], dtype=float)
    w_full[S] = w_sub
    w_full_torch = torch.from_numpy(w_full.astype(np.float32)).to(DEVICE)

    if return_history:
        return w_full_torch, S, history
    return w_full_torch, S



# Add this helper somewhere near your evaluation code
def print_group_fairness(weights, sensitive_idx, outcome_idx, name="Method"):
    """
    weights: torch tensor shape (N,) (float)
    sensitive_idx: torch bool mask shape (N,) (True = male in your pipeline)
    outcome_idx: torch bool mask shape (N,) (True = positive outcome)
    """
    eps = 1e-12
    w = weights.view(-1).float().cpu()
    sens = sensitive_idx.view(-1).cpu().bool()
    out = outcome_idx.view(-1).cpu().bool()

    male_mask = sens
    female_mask = ~sens

    male_count = int(male_mask.sum().item())
    female_count = int(female_mask.sum().item())

    # How many in each group are selected (nonzero weight)
    male_selected = int((w[male_mask] > 0).sum().item())
    female_selected = int((w[female_mask] > 0).sum().item())

    # Total weight mass per group
    male_weight_mass = float(w[male_mask].sum().item())
    female_weight_mass = float(w[female_mask].sum().item())
    total_weight_mass = float(w.sum().item())

    # Unweighted positive rates (simple dataset base rates)
    unweighted_male_pos = float(out[male_mask].float().mean().item()) if male_count > 0 else float('nan')
    unweighted_female_pos = float(out[female_mask].float().mean().item()) if female_count > 0 else float('nan')

    # Weighted positive rates (using weights as probabilities; normalize by group weight mass)
    weighted_male_pos = float((w * out.float() * male_mask.float()).sum().item() / (male_weight_mass + eps))
    weighted_female_pos = float((w * out.float() * female_mask.float()).sum().item() / (female_weight_mass + eps))

    # Top-k contributors (indices and weights)
    k = min(10, w.numel())
    topk_vals, topk_idx = torch.topk(w, k)
    topk_list = [(int(idx.item()), float(val.item())) for idx, val in zip(topk_idx, topk_vals)]

    print(f"\n=== {name} fairness summary ===")
    print(f"Group sizes: male={male_count}, female={female_count}")
    print(f"Selected (w>0): male={male_selected}, female={female_selected}  (total selected={(male_selected+female_selected)})")
    print(f"Weight mass: male={male_weight_mass:.6f}, female={female_weight_mass:.6f}, total={total_weight_mass:.6f}")
    print(f"Unweighted positive rates: male={unweighted_male_pos:.4f}, female={unweighted_female_pos:.4f}")
    print(f"Weighted positive rates:   male={weighted_male_pos:.4f}, female={weighted_female_pos:.4f}")
    print(f"Absolute DP gap (weighted): {abs(weighted_male_pos - weighted_female_pos):.6f}")
    print(f"Top-{k} weight contributors (idx, weight): {topk_list}")
    print("=================================\n")


def compute_group_stats(weights_torch, sensitive_idx, outcome_idx):
    """
    Returns dict with:
      - male_count, female_count (ints)
      - unweighted_male_pos, unweighted_female_pos (floats)
      - weighted_male_pos, weighted_female_pos (floats)
      - male_weight_mass, female_weight_mass (floats)
      - selected_counts: (male_selected, female_selected)
      - dp_gap_abs (float) = |weighted_male_pos - weighted_female_pos|
    """
    eps = 1e-12
    w = weights_torch.view(-1).float().cpu().numpy()
    sens = sensitive_idx.view(-1).cpu().numpy().astype(bool)
    out = outcome_idx.view(-1).cpu().numpy().astype(bool)

    male_mask = sens
    female_mask = ~sens

    male_count = int(male_mask.sum())
    female_count = int(female_mask.sum())

    male_selected = int((w[male_mask] > 0).sum())
    female_selected = int((w[female_mask] > 0).sum())

    male_weight_mass = float(w[male_mask].sum())
    female_weight_mass = float(w[female_mask].sum())

    # Unweighted positive rates (dataset base rates)
    unweighted_male_pos = float(out[male_mask].mean()) if male_count > 0 else np.nan
    unweighted_female_pos = float(out[female_mask].mean()) if female_count > 0 else np.nan

    # Weighted positive rates (normalize by group weight mass)
    weighted_male_pos = float((w * out * male_mask).sum() / (male_weight_mass + eps))
    weighted_female_pos = float((w * out * female_mask).sum() / (female_weight_mass + eps))

    dp_gap_abs = abs(weighted_male_pos - weighted_female_pos)

    # Unweighted positive rates among selected (w > 0)
    selected_mask = w > 0
    selected_male_mask = male_mask & selected_mask
    selected_female_mask = female_mask & selected_mask

    unweighted_male_pos_selected = float(out[selected_male_mask].mean()) if selected_male_mask.sum() > 0 else np.nan
    unweighted_female_pos_selected = float(out[selected_female_mask].mean()) if selected_female_mask.sum() > 0 else np.nan

    # Weighted positive rates among selected (normalize by selected group weight mass)
    male_weight_mass_selected = float(w[selected_male_mask].sum())
    female_weight_mass_selected = float(w[selected_female_mask].sum())

    weighted_male_pos_selected = float((w[selected_male_mask] * out[selected_male_mask]).sum() / (male_weight_mass_selected + eps)) if male_weight_mass_selected > 0 else np.nan
    weighted_female_pos_selected = float((w[selected_female_mask] * out[selected_female_mask]).sum() / (female_weight_mass_selected + eps)) if female_weight_mass_selected > 0 else np.nan

    return {
        'male_count': male_count,
        'female_count': female_count,
        'male_selected': male_selected,
        'female_selected': female_selected,
        'male_weight_mass': male_weight_mass,
        'female_weight_mass': female_weight_mass,
        'unweighted_male_pos': unweighted_male_pos,
        'unweighted_female_pos': unweighted_female_pos,
        'weighted_male_pos': weighted_male_pos,
        'weighted_female_pos': weighted_female_pos,
        'dp_gap_abs': dp_gap_abs,
        'unweighted_male_pos_selected': unweighted_male_pos_selected,
        'unweighted_female_pos_selected': unweighted_female_pos_selected,
        'weighted_male_pos_selected': weighted_male_pos_selected,
        'weighted_female_pos_selected': weighted_female_pos_selected
    }


def weighted_kernel_herding_frank_wolfe_alternate(
    X_np, m, sigma=1.0, ridge=1e-8, verbose=False, fairness_beta=0.0,
    sensitive_mask=None, outcome_mask=None, alternate_dp_freq=None,
    use_dp_penalty_in_mmd_steps=False
):
    """
    Greedy Franke-Wolfe style selection with optional alternation for fairness:
      - If alternate_dp_freq is set (e.g., 2=every other step), alternate between minimizing MMD and minimizing DP violation.
      - fairness_beta > 0 includes a small fairness penalty when evaluating candidates (penalizes DP violation) - optional in MMD steps.
      - returns selected indices (length m) and last intermediate weights on S
    """
    X = np.asarray(X_np)
    N = X.shape[0]
    if m < 1 or m > N:
        raise ValueError("m must be between 1 and N")

    K = rbf_kernel_np(X, X, sigma=sigma)  # full kernel (N,N)
    mu_pi = K.mean(axis=0)                # length N
    selected = []
    current_embedding = np.zeros(N, dtype=float)
    remaining = set(range(N))

    # small helper to compute candidate fairness violation given candidate set S and candidate weights (w)
    def dp_violation_for_weights(w_sub, S_idx):
        if sensitive_mask is None or outcome_mask is None:
            return 0.0
        # build full-length weights for DP calculation (zeros outside S)
        w_full = np.zeros(N, dtype=float)
        w_full[np.asarray(S_idx, dtype=int)] = w_sub
        # compute male/female masks (numpy)
        male_mask = np.asarray(sensitive_mask.cpu().numpy(), dtype=bool)
        female_mask = ~male_mask
        outcome_mask_np = np.asarray(outcome_mask.cpu().numpy(), dtype=bool)
        # avoid zero division
        eps = 1e-8
        p_male = w_full[male_mask & outcome_mask_np].sum() / (w_full[male_mask].sum() + eps)
        p_female = w_full[female_mask & outcome_mask_np].sum() / (w_full[female_mask].sum() + eps)
        return abs(p_male - p_female)

    for t in range(m):
        best_idx = None
        best_score = np.inf
        is_dp_step = (alternate_dp_freq is not None) and (t % alternate_dp_freq == 0) and (fairness_beta >= 0 or sensitive_mask is not None)

        # Evaluate each candidate (costly: O(N^2) inner worst-case)
        for j in remaining:
            S_candidate = selected + [j]
            S_arr = np.array(S_candidate, dtype=int)
            K_SS = K[np.ix_(S_arr, S_arr)]
            K_XS = K[:, S_arr]
            z = K_XS.mean(axis=0)
            # solve K_SS w = z (closed-form intermediate weights used for selection)
            try:
                A = K_SS + ridge * np.eye(K_SS.shape[0])
                w_candidate = np.linalg.solve(A, z)
            except np.linalg.LinAlgError:
                w_candidate = np.linalg.pinv(K_SS + ridge * np.eye(K_SS.shape[0])) @ z

            if is_dp_step:
                # DP-minimizing step: score is DP violation (lower is better)
                fair_pen = dp_violation_for_weights(w_candidate, S_arr)
                score = fair_pen  # minimize DP directly
            else:
                # MMD-minimizing step
                emb = K_XS @ w_candidate
                residual = emb - mu_pi
                score_mmd = float(residual @ residual)  # squared norm
                score = score_mmd
                # Optional: add small DP penalty even in MMD steps
                if use_dp_penalty_in_mmd_steps:
                    fair_pen = dp_violation_for_weights(w_candidate, S_arr)
                    score += fairness_beta * (fair_pen ** 2)

            if score < best_score:
                best_score = score
                best_idx = j

        selected.append(best_idx)
        remaining.discard(best_idx)
        # update current_embedding with current selected set's weights
        S = np.array(selected, dtype=int)
        K_SS = K[np.ix_(S, S)]
        K_XS = K[:, S]
        z = K_XS.mean(axis=0)
        try:
            A = K_SS + ridge * np.eye(K_SS.shape[0])
            w_current = np.linalg.solve(A, z)
        except np.linalg.LinAlgError:
            w_current = np.linalg.pinv(K_SS + ridge * np.eye(K_SS.shape[0])) @ z
        current_embedding = (K_XS @ w_current)

        if verbose:
            obj_type = "DP" if is_dp_step else "MMD"
            print(f"[sel {t+1}/{m}] picked {best_idx} | score {best_score:.6g} | objective: {obj_type} | fair_beta {fairness_beta} | alternate_freq {alternate_dp_freq}")

    # final weights on selected set (we'll re-optimize with QP or ADMM later)
    S_final = np.array(selected, dtype=int)
    K_SS = K[np.ix_(S_final, S_final)]
    K_XS = K[:, S_final]
    z = K_XS.mean(axis=0)
    try:
        final_w = np.linalg.solve(K_SS + ridge * np.eye(K_SS.shape[0]), z)
    except np.linalg.LinAlgError:
        final_w = np.linalg.pinv(K_SS + ridge * np.eye(K_SS.shape[0])) @ z

    return S_final, final_w, K


def baseline_simple_alternate(
    P_tensor, sensitive_idx, outcome_idx, sigma, m,
    ridge=1e-8, fairness_beta=0.0, dp_frequency=None,
    return_indices=False, verbose=False
):
    """
    Alternate-selection baseline (use FW's final weights, not uniform).
    The final_w coming from weighted_kernel_herding_frank_wolfe_alternate
    is clipped to be non-negative and normalized to sum to 1.
    """
    P_np = P_tensor.cpu().numpy()
    S, final_w, K = weighted_kernel_herding_frank_wolfe_alternate(
        P_np, m, sigma=sigma, ridge=ridge, verbose=verbose,
        fairness_beta=fairness_beta, sensitive_mask=sensitive_idx,
        outcome_mask=outcome_idx, alternate_dp_freq=dp_frequency,
        use_dp_penalty_in_mmd_steps=False
    )

    N = P_np.shape[0]
    w_full = np.zeros(N, dtype=float)

    if len(S) > 0:
        # final_w is the frank-wolfe closed-form weights on the subset (numpy)
        w_sub = np.asarray(final_w, dtype=float).copy()

        # enforce non-negativity and sum-to-one (simple clip + renormalize)
        w_sub = np.maximum(w_sub, 0.0)
        s = w_sub.sum()
        if s <= 0:
            # fallback: uniform on S
            w_sub = np.ones(len(S), dtype=float) / float(len(S))
        else:
            w_sub = w_sub / float(s)

        w_full[S] = w_sub

    w_full_torch = torch.from_numpy(w_full.astype(np.float32)).to(DEVICE)
    if return_indices:
        return w_full_torch, S
    return w_full_torch
# --- NEW baseline: alternate selection -> QP weights ------------------------
def baseline_alternate_qp(
    P_tensor, sensitive_idx, outcome_idx, sigma, m,
    ridge=1e-8, fairness_beta=0.0, dp_frequency=None,
    return_indices=False, verbose=False
):
    """
    Alternate-selection baseline followed by QP weight refinement on the subset.
    """
    P_np = P_tensor.cpu().numpy()
    S, _, K = weighted_kernel_herding_frank_wolfe_alternate(
        P_np, m, sigma=sigma, ridge=ridge, verbose=verbose,
        fairness_beta=fairness_beta, sensitive_mask=sensitive_idx,
        outcome_mask=outcome_idx, alternate_dp_freq=dp_frequency,
        use_dp_penalty_in_mmd_steps=False
    )

    # QP on selected subset
    if len(S) == 0:
        # fallback: uniform over dataset
        N = P_np.shape[0]
        w_full = np.ones(N, dtype=float) / float(N)
    else:
        K_SS = K[np.ix_(S, S)]
        K_XS = K[:, S]
        k_S = K_XS.mean(axis=0)
        w_sub = qp_weights_slsqp(K_SS, k_S, ridge=ridge)
        w_full = np.zeros(P_np.shape[0], dtype=float)
        w_full[S] = w_sub

    w_full_torch = torch.from_numpy(w_full.astype(np.float32)).to(DEVICE)
    if return_indices:
        return w_full_torch, S
    return w_full_torch



if __name__ == "__main__":
    # parameters
    N_SAMPLES = 1000
    SIGMA = 10.0        # sigma for rbf kernel
    CORESET_M = 50     # budget
    RHO = 1
    FAIR_BETA = 0.01    # small fairness penalty option in selection
    DP_FREQ = 2         # alternate every DP_FREQ steps (None means never alternate)

    # load data
    P, sensitive_idx, outcome_idx = get_adult_data(num_samples=N_SAMPLES)

    # compute full kernel and k_pq on torch for final evaluation
    print("Computing full kernel (torch) for final evaluation...")
    K_pp_torch = rbf_kernel_torch(P, P, sigma=SIGMA).to(DEVICE)
    k_pq_torch = K_pp_torch.mean(dim=1)

    # Run the six baselines
    results = {}
    timings = {}

    # 1) Random (uniform on subset)
    t0 = time.time()
    w_rand_torch, rand_indices = baseline_random(P, m=CORESET_M, seed=42, return_indices=True)
    timings['Random'] = time.time() - t0
    results['Random'] = (w_rand_torch, rand_indices)

    # 2) Standard WKH (greedy FW selection -> QP)
    t0 = time.time()
    w_std_torch, std_indices = baseline_standard_wkh(P, sigma=SIGMA, m=CORESET_M, ridge=1e-8, return_indices=True, verbose=True)
    timings['Standard WKH'] = time.time() - t0
    results['Standard WKH'] = (w_std_torch, std_indices)

    # # 3) Simple Alternate (selection only -> uniform on S)
    # t0 = time.time()
    # w_simple_alt, simple_alt_indices = baseline_simple_alternate(
    #     P, sensitive_idx, outcome_idx, sigma=SIGMA, m=CORESET_M,
    #     ridge=1e-8, fairness_beta=0.0, dp_frequency=DP_FREQ, return_indices=True, verbose=True
    # )
    # timings['Simple Alternate'] = time.time() - t0
    # results['Simple Alternate'] = (w_simple_alt, simple_alt_indices)

    # # 4) Alternate + QP (selection with alternation -> QP)
    # t0 = time.time()
    # w_alt_qp, alt_qp_indices = baseline_alternate_qp(
    #     P, sensitive_idx, outcome_idx, sigma=SIGMA, m=CORESET_M,
    #     ridge=1e-8, fairness_beta=0.0, dp_frequency=DP_FREQ, return_indices=True, verbose=True
    # )
    # timings['Alternate + QP'] = time.time() - t0
    # results['Alternate + QP'] = (w_alt_qp, alt_qp_indices)

    # 5) Alternate + ADMM (selection alternation -> ADMM, no fairness_beta during selection)
    t0 = time.time()
    w_alt_admm, alt_admm_indices, hist_admm = baseline_fair_admm(
        P, sensitive_idx, outcome_idx, sigma=SIGMA, m=CORESET_M,
        rho=1.0, lr_fair=0.02, proj_steps=40, ridge=1e-8,
        fairness_beta=0.0, dp_frequency=DP_FREQ, lambda_fair=0.05, return_history=True, verbose=True
    )
    timings['Alternate + ADMM'] = time.time() - t0
    results['Alternate + ADMM'] = (w_alt_admm, alt_admm_indices)

    # Evaluate all baselines
    names = ['Random', 'Standard WKH', #'Simple Alternate', 'Alternate + QP'#,
              'Alternate + ADMM']
    stats_list = []
    print("\n--- Evaluating final weights ---")
    for name in names:
        w_torch, S_idx = results[name]
        mmd_val, dp_val = calculate_metrics(w_torch, K_pp_torch, k_pq_torch, sensitive_idx, outcome_idx)
        stats = compute_group_stats(w_torch, sensitive_idx, outcome_idx)
        stats['mmd'] = mmd_val
        stats['dp_val'] = dp_val
        stats_list.append(stats)
        print(f"{name:<18} | MMD={mmd_val:12.6f} | DP gap={dp_val:12.6f} | time={timings[name]:.2f}s")

    # Prepare plotting arrays (adapted for 6 baselines)
    weighted_male = np.array([s['weighted_male_pos'] for s in stats_list])
    weighted_female = np.array([s['weighted_female_pos'] for s in stats_list])
    unweighted_male = stats_list[1]['unweighted_male_pos']  # dataset stat (from Standard WKH)
    unweighted_female = stats_list[1]['unweighted_female_pos']

    male_weight_mass = np.array([s['male_weight_mass'] for s in stats_list])
    female_weight_mass = np.array([s['female_weight_mass'] for s in stats_list])
    dp_gaps = np.array([s['dp_gap_abs'] for s in stats_list])

    x = np.arange(len(names))
    width = 0.35

    # Plot 1: weighted positive rates by group
    fig, ax = plt.subplots(figsize=(11, 5))
    rects1 = ax.bar(x - width/2, weighted_male, width, label='Male (weighted)')
    rects2 = ax.bar(x + width/2, weighted_female, width, label='Female (weighted)')
    ax.axhline(unweighted_male, linestyle='--', linewidth=1.2, label=f'Unweighted male = {unweighted_male:.3f}')
    ax.axhline(unweighted_female, linestyle=':', linewidth=1.2, label=f'Unweighted female = {unweighted_female:.3f}')
    ax.set_ylabel('Positive Rate (P[Y=1])')
    ax.set_title('Weighted Positive Rates by Group and Baseline')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha='right')
    ax.legend(loc='upper right', fontsize='small')

    def autolabel(rects, ax):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
    autolabel(rects1, ax)
    autolabel(rects2, ax)
    plt.tight_layout()

    # Plot 2: weight mass per group + DP gap
    fig2, ax2 = plt.subplots(figsize=(11, 5))
    rects_m = ax2.bar(x - width/4, male_weight_mass, width/2, label='Male weight mass')
    rects_f = ax2.bar(x + width/4, female_weight_mass, width/2, label='Female weight mass')
    ax3 = ax2.twinx()
    ax3.plot(x, dp_gaps, marker='o', linestyle='-', linewidth=1.5, label='|DP gap|', alpha=0.9)
    ax3.set_ylabel('Absolute DP gap (weighted)', color='tab:blue')
    ax2.set_ylabel('Total weight mass by group')
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=15, ha='right')
    ax2.set_title('Weight Mass by Group and Absolute DP Gap')
    ax2.legend(loc='upper left', fontsize='small')
    ax3.legend(loc='upper right', fontsize='small')

    for rect in rects_m + rects_f:
        h = rect.get_height()
        ax2.annotate(f'{h:.3f}', xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    plt.tight_layout()

    # Console summary
    print("\nDetailed group stats per baseline:")
    for name, s in zip(names, stats_list):
        print(f"\n{name}:")
        print(f"  group sizes: male={s['male_count']}, female={s['female_count']}")
        print(f"  selected counts (w>0): male={s['male_selected']}, female={s['female_selected']}")
        print(f"  weight mass: male={s['male_weight_mass']:.4f}, female={s['female_weight_mass']:.4f}")
        print(f"  unweighted pos rates: male={s['unweighted_male_pos_selected']:.4f}, female={s['unweighted_female_pos_selected']:.4f}")
        print(f"  weighted pos rates:   male={s['weighted_male_pos_selected']:.4f}, female={s['weighted_female_pos_selected']:.4f}")
        print(f"  |DP gap| = {s['dp_gap_abs']:.6f}")

    plt.show()
