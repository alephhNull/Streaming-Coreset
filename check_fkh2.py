"""
Standalone benchmark script for Fair Kernel Herding.

This file provides:
 - a corrected Adult dataset loader (`get_adult_data`)
 - exact MMD utility (`mmd2_exact_full_vs_coreset`)
 - reservoir baseline and a simple random baseline
 - plotting helpers to show MMD and group positive rates across baselines
 - a runnable `main()` which imports `FairKernelHerdingStreamer` (from
   fair_kernel_herding_streamer.py created earlier in the canvas), runs the
   streamer over the Adult dataset, runs baselines, and produces plots.

Run directly:
    python fair_kernel_herding_benchmark.py

Dependencies: numpy, pandas, torch, matplotlib, scikit-learn

"""

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt

# Import the streamer implemented in the other canvas file
from streamers.fkh_streamer2 import FairKernelHerdingStreamer as FKH1
from streamers.fair_herding_streamer import FairKernelHerdingStreamer as FKH2
from streamers.rff_wkh_streamer import WKHStreamingCoreset
from streamers.mirror_descent_streamer import MirrorDescentHerdingStreamer
from streamers.multisampler_streamer import MultiSamplerWKHStreamingCoreset

DEVICE = torch.device("cpu")


# ---------------------- Utilities ----------------------

def rbf_kernel_np(X, Y, sigma=1.0):
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    XX = np.sum(X * X, axis=1)[:, None]
    YY = np.sum(Y * Y, axis=1)[None, :]
    d2 = XX + YY - 2.0 * (X @ Y.T)
    gamma = 1.0 / (2.0 * (sigma ** 2)) if sigma > 0 else 1.0
    K = np.exp(-gamma * d2)
    return K


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def get_adult_data_fixed(num_samples=2000, seed=971944, scale_dummies=False):
    """
    Load Adult dataset and preprocess:
      - drop target and sensitive attributes before dummify
      - One-hot encode categorical columns
      - StandardScale only the original numeric columns (keep dummies 0/1)
    If scale_dummies=True, MaxAbsScaler-like behavior is applied (divides by 1 so effectively no-op).
    Returns: X_scaled (np.array), sensitive_cols_dict (pandas series), y (np.array)
    """
    from sklearn.datasets import fetch_openml

    print("Fetching Adult dataset...")
    oml = fetch_openml(name="adult", version=2, as_frame=True)
    df_original = oml.frame
    df_original.rename(columns={'class': 'income'}, inplace=True)

    # Clean
    df = df_original.replace([' ?', '?'], pd.NA).dropna().reset_index(drop=True)
    df['income'] = df['income'].str.contains('>50K')

    np.random.seed(seed)

    y = df['income'].astype(int)
    sensitive_cols = {
        'sex': df['sex'].str.strip()
    }

    # Keep copies of columns types BEFORE dropping sensitive/target
    df_features = df.drop(columns=['income', 'sex', 'fnlwgt', 'education-num'], errors='ignore')

    # Identify numeric columns (pandas dtypes)
    numeric_cols = df_features.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in df_features.columns if c not in numeric_cols]

    # One-hot encode categorical columns with pandas.get_dummies (drop_first optional)
    X_cat = pd.get_dummies(df_features[categorical_cols], drop_first=True)
    X_num = df_features[numeric_cols].reset_index(drop=True)

    # Subsample indices (stratify by outcome y) if requested
    if num_samples is not None and num_samples < len(df_features):
        indices = np.arange(len(df_features))
        train_idx, _ = train_test_split(indices, train_size=num_samples, stratify=y, random_state=seed)
        X_num = X_num.iloc[train_idx].reset_index(drop=True)
        X_cat = X_cat.iloc[train_idx].reset_index(drop=True)
        y_sel = y.iloc[train_idx].reset_index(drop=True)
        sens_sel = {k: v.iloc[train_idx].reset_index(drop=True) for k, v in sensitive_cols.items()}
    else:
        y_sel = y.reset_index(drop=True)
        sens_sel = {k: v.reset_index(drop=True) for k, v in sensitive_cols.items()}

    # Concatenate numeric and categorical parts
    # X_combined = np.hstack([X_num, X_cat])
    scaler = StandardScaler()
    # X_combined = scaler.fit_transform(X_combined)

    # Scale only numeric columns
    if X_num.shape[1] > 0:
        X_num_scaled = scaler.fit_transform(X_num)
    else:
        # no numeric columns -> empty array with appropriate rows
        X_num_scaled = np.zeros((X_cat.shape[0], 0))

    # Optionally scale dummies (not recommended). Here we leave them as 0/1 by default.
    if scale_dummies:
        # If you really want to scale one-hot columns, consider MaxAbsScaler-like transform.
        # But default here is to keep them as 0/1.
        from sklearn.preprocessing import MaxAbsScaler
        mab = MaxAbsScaler()
        X_cat_scaled = mab.fit_transform(X_cat)
    else:
        X_cat_scaled = X_cat.values

    # Concatenate numeric and categorical parts
    X_combined = np.hstack([X_num_scaled, X_cat_scaled])

    print(f"Loaded dataset: {X_combined.shape[0]} samples, {X_combined.shape[1]} features.")
    print("Numeric cols:", numeric_cols)
    print("Categorical dummies (first 10):", list(X_cat.columns[:10]))
    print("Sensitive attributes tracked:", list(sens_sel.keys()))

    return X_combined, sens_sel, y_sel.to_numpy(dtype=int)


# ---------------------- Dataset Loader (CORRECTED) ----------------------
def get_adult_data(num_samples=2000):
    """
    Robust loader for Adult dataset.
    Returns (X_np, sensitive_cols_dict, y_np).
    sensitive_cols_dict contains raw pandas Series for each sensitive attribute.
    """
    print("Fetching Adult dataset... (this may take a moment)")
    oml = fetch_openml(name="adult", version=2, as_frame=True)
    df_original = oml.frame
    df_original.rename(columns={'class': 'income'}, inplace=True)
    df_original = df_original.sample(frac=1, random_state=42).reset_index(drop=True)
    df = df_original.replace([' ?', '?'], pd.NA).dropna().reset_index(drop=True)
    df['income'] = df['income'].str.contains('>50K')



    y = df['income'].astype(int)
    sensitive_cols = {
        'sex': df['sex'].str.strip(),
        # optionally include race or others
        # 'race': df['race'].str.strip()
    }

    # Create features (drop sensitive and target before dummify)
    X = df.drop(columns=['income', 'sex', 'fnlwgt', 'education-num'], errors='ignore')
    X_dummified = pd.get_dummies(X, drop_first=True)

    # Subsample if requested (stratify by y)
    if num_samples is not None and num_samples < len(X_dummified):
        indices = np.arange(len(X_dummified))
        train_idx, _ = train_test_split(indices, train_size=num_samples, stratify=y)
        X_sel = X_dummified.iloc[train_idx].reset_index(drop=True)
        y_sel = y.iloc[train_idx].reset_index(drop=True)
        sens_sel = {k: v.iloc[train_idx].reset_index(drop=True) for k, v in sensitive_cols.items()}
    else:
        X_sel = X_dummified.reset_index(drop=True)
        y_sel = y.reset_index(drop=True)
        sens_sel = {k: v.reset_index(drop=True) for k, v in sensitive_cols.items()}

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sel)

    print(f"Loaded dataset: {X_scaled.shape[0]} samples, {X_scaled.shape[1]} features.")
    print("Sensitive attributes tracked:", list(sens_sel.keys()))

    return X_scaled, sens_sel, y_sel.to_numpy(dtype=int)


# ---------------------- MMD metric ----------------------
def mmd2_exact_full_vs_coreset(X_full, X_coreset, weights, sigma=1.0):
    X_full = np.asarray(X_full)
    X_coreset = np.asarray(X_coreset)
    n = X_full.shape[0]
    K_ff = rbf_kernel_np(X_full, X_full, sigma=sigma)
    K_fc = rbf_kernel_np(X_full, X_coreset, sigma=sigma)
    K_cc = rbf_kernel_np(X_coreset, X_coreset, sigma=sigma)
    term_ff = K_ff.sum() / (n * n)
    term_fc = 2.0 * (K_fc.sum(axis=0) @ weights) / n
    term_cc = weights @ (K_cc @ weights)
    return float(term_ff - term_fc + term_cc)


# ---------------------- Baselines ----------------------

def reservoir_baseline(N, m, random_state=None):
    rng = np.random.RandomState(random_state)
    idx = rng.choice(N, size=m, replace=False)
    weights = np.ones(m, dtype=float) / float(m)
    return idx, weights


def random_uniform_baseline(N, m, random_state=None):
    return reservoir_baseline(N, m, random_state=random_state)


# ---------------------- Group-rate evaluation ----------------------
def compute_group_positive_rates(sensitive_cols: dict, y_full: np.ndarray, coreset_idx: np.ndarray, coreset_weights: np.ndarray):
    """
    Returns a dict: {attr_name: {group_label: (full_rate, coreset_rate)}}
    coreset_idx are flat indices into the full dataset (0..N-1), weights are length=len(coreset_idx)
    """
    results = {}
    N = len(y_full)
    for attr, series in sensitive_cols.items():
        series_arr = np.asarray(series)
        groups = np.unique(series_arr)
        gdict = {}
        for g in groups:
            mask_full = (series_arr == g)
            if mask_full.sum() == 0:
                full_rate = 0.0
            else:
                full_rate = float(y_full[mask_full].sum()) / float(mask_full.sum())

            # coreset
            if len(coreset_idx) == 0:
                co_rate = 0.0
            else:
                # which coreset points belong to group g
                in_group = [i for i, idx in enumerate(coreset_idx) if series_arr[idx] == g]
                if len(in_group) == 0:
                    co_rate = 0.0
                else:
                    num = float((coreset_weights[in_group] * y_full[coreset_idx[in_group]]).sum())
                    den = float(coreset_weights[in_group].sum())
                    co_rate = num / (den + 1e-12)

            gdict[g] = (full_rate, co_rate)
        results[attr] = gdict
    return results


# ---------------------- Plot helpers ----------------------

def plot_mmd_bar(baseline_names, mmd_values, outpath=None):
    plt.figure(figsize=(8, 4))
    plt.bar(baseline_names, mmd_values)
    plt.ylabel('MMD^2 (full vs coreset)')
    plt.title('MMD^2 for different baselines')
    plt.tight_layout()
    if outpath:
        plt.savefig(outpath)
    else:
        plt.show()




def plot_group_rates(baseline_names, group_rates: dict, attr_name: str, outpath=None):
    """
    group_rates: dict mapping baseline -> {group: (full_rate, co_rate)}
    baseline_names: list of baselines in the order you want them plotted
    """
    groups = list(next(iter(group_rates.values())).keys())
    n_baselines = len(baseline_names)
    x = np.arange(n_baselines)

    # total horizontal space reserved for the bars at each baseline tick
    total_width = 0.8
    # number of small bars per baseline (two bars per group: full + coreset)
    n_bars_per_baseline = len(groups) * 2
    # width of each small bar
    bar_width = total_width / n_bars_per_baseline

    fig, ax = plt.subplots(figsize=(10, 5))

    # starting offset so the block of bars is centered at x
    start = - total_width / 2.0

    for i, g in enumerate(groups):
        # gather values in the same order as baseline_names
        full = [group_rates[b][g][0] for b in baseline_names]
        co = [group_rates[b][g][1] for b in baseline_names]

        # offset for this group's pair of bars
        offset = start + i * 2 * bar_width

        ax.bar(x + offset, full, width=bar_width, alpha=0.4, label=f"{g} (full)")
        ax.bar(x + offset + bar_width, co, width=bar_width, alpha=0.9, label=f"{g} (coreset)")

    ax.set_xticks(x)
    ax.set_xticklabels(baseline_names)
    ax.set_ylabel('Positive rate')
    ax.set_title(f'Group positive rates for attribute: {attr_name}')

    # avoid legend overlap with plot
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    fig.tight_layout()

    if outpath:
        fig.savefig(outpath)
    else:
        plt.show()

    return fig, ax



from sklearn.metrics import pairwise_distances
import numpy as np

def choose_gamma_median(X, subsample=2000, seed=0):
    """
    Compute a single gamma (and sigma) for the whole experiment using the median heuristic.
    X: numpy array (preprocessed), shape (N, d)
    Returns: gamma, sigma, median_d2
    """
    N = X.shape[0]
    rng = np.random.default_rng(seed)
    if N > subsample:
        idx = rng.choice(N, size=subsample, replace=False)
        Xs = X[idx]
    else:
        Xs = X
    # pairwise squared distances (efficient using sklearn or numpy)
    from sklearn.metrics import pairwise_distances
    d2 = pairwise_distances(Xs, metric="sqeuclidean")
    median_d2 = float(np.median(d2))
    # avoid zero
    median_d2 = max(median_d2, 1e-12)
    gamma = 1.0 / median_d2           # median heuristic -> gamma = 1 / median_d2
    sigma = np.sqrt(median_d2 / 2.0)  # sigma consistent with gamma via gamma = 1/(2*sigma^2)
    print(f"[median heuristic] median_d2={median_d2:.6g}, gamma={gamma:.6g}, sigma={sigma:.6g}")
    return gamma, sigma, median_d2


# ---------------------- Main experiment ----------------------

def main():
    # Configuration
    num_samples = 2000
    coreset_size = 50
    batch_size = 100
    buffer_capacity = 150

    
    # Load data
    X_full, sensitive_cols, y_full = get_adult_data_fixed(num_samples=num_samples, seed=666)

    GAMMA, SIGMA, MED_D2 = choose_gamma_median(X_full, subsample=2000, seed=42)
    print(f"Chosen hyperparameters: GAMMA={GAMMA}, SIGMA={SIGMA}, MED_D2={MED_D2}")

    print("Loaded dataset:", X_full.shape[0], "samples,", X_full.shape[1], "features.")
    N = X_full.shape[0]

    # Fit an RFF sampler for the streamer
    n_components = 1024
    sampler = RBFSampler(gamma=GAMMA, n_components=n_components, random_state=42)
    sampler.fit(X_full)

    # Instantiate fair streamer
    fkh1 = FKH1(
        coreset_size=coreset_size,
        buffer_capacity=buffer_capacity,
        sampler=sampler,
        batch_size=batch_size,
        select_alternate_freq=None,
        md_iterations=1000,
        eta=1,
        verbose=False,
    )

    wkh_streamer = WKHStreamingCoreset(
        coreset_size=coreset_size,
        buffer_capacity=buffer_capacity,
        sampler=sampler,
        batch_size=batch_size
    )

    # mdh_streamer = MirrorDescentHerdingStreamer(
    #     coreset_size=coreset_size,
    #     buffer_capacity=buffer_capacity,
    #     sampler=sampler,
    #     batch_size=batch_size,
    #     md_iterations=1000,
    #     eta=1,
    #     verbose=True
    # )

    multisampler_streamer = MultiSamplerWKHStreamingCoreset(
        coreset_size=coreset_size,
        buffer_capacity=buffer_capacity,
        sampler=sampler,
        batch_size=batch_size
    )


    # Stream through data in batches
    n_batches = int(np.ceil(N / batch_size))
    for b in range(n_batches):
        start = b * batch_size
        end = min(N, (b + 1) * batch_size)
        X_batch = X_full[start:end]
        y_batch = y_full[start:end]
        sens_batch = {k: v.iloc[start:end].to_numpy() for k, v in sensitive_cols.items()}
        print(f"Processing batch {b+1}/{n_batches} (size={len(X_batch)})")
        fkh1.process_batch(X_batch, y_batch, batch_idx=b, sensitive_batch=sens_batch)
        wkh_streamer.process_batch(X_batch, y_batch, batch_idx=b)
        # mdh_streamer.process_batch(X_batch, y_batch, batch_idx=b)
        multisampler_streamer.process_batch(X_batch, y_batch, batch_idx=b)



    # Get coreset from fair streamer
    fair_idx_flat1, fair_weights1, provenance = fkh1.get_final_coreset()
    X_fair1 = X_full[fair_idx_flat1]

    wkh_idx, wkh_w, _= wkh_streamer.get_final_coreset()
    X_wkh = X_full[wkh_idx]

    # mdh_idx, mdh_w, _= mdh_streamer.get_final_coreset()
    # X_mdh = X_full[mdh_idx]

    multisampler_idx, multsampler_q, _ = multisampler_streamer.get_final_coreset()
    X_multisampler = X_full[multisampler_idx]

    # Baseline: reservoir (random) selection
    res_idx, res_weights = reservoir_baseline(N, coreset_size, random_state=123)
    X_res = X_full[res_idx]

    # Evaluate MMDs
    mmd_fair1 = mmd2_exact_full_vs_coreset(X_full, X_fair1, fair_weights1, sigma=SIGMA)
    # mmd_fair2 = mmd2_exact_full_vs_coreset(X_full, X_fair2, fair_weights2, sigma=sigma)
    mmd_wkh = mmd2_exact_full_vs_coreset(X_full, X_wkh, wkh_w, sigma=SIGMA)
    # mmd_mdh = mmd2_exact_full_vs_coreset(X_full, X_mdh, mdh_w, sigma=SIGMA)
    mmd_res = mmd2_exact_full_vs_coreset(X_full, X_res, res_weights, sigma=SIGMA)
    mmd_multisampler = mmd2_exact_full_vs_coreset(X_full, X_multisampler, multsampler_q, sigma=SIGMA)

    print("MMD^2 - FairStreamer1:", mmd_fair1)
    # print("MMD^2 - FairStreamer2:", mmd_fair2)
    print("MMD^2 - WKH:", mmd_wkh)
    # print("MMD^2 - MDH:", mmd_mdh)
    print("MMD^2 - MultiSampler:", mmd_multisampler)
    print("MMD^2 - Random sampling:", mmd_res)

    baseline_names = ['FairStreamer1', #'FairStreamer2',
                       'WKH', #'MDH',
                         'MultiSampler', 'Random Sampling']
    mmd_values = [mmd_fair1, #mmd_fair2,
                  mmd_wkh, #mmd_mdh,
                    mmd_multisampler, mmd_res]

    plot_mmd_bar(baseline_names, mmd_values)

    # Compute group-wise positive rates
    fair_group_rates1 = compute_group_positive_rates(sensitive_cols, y_full, fair_idx_flat1, fair_weights1)
    # fair_group_rates2 = compute_group_positive_rates(sensitive_cols, y_full, fair_idx_flat2, fair_weights2)
    wkh_group_rates = compute_group_positive_rates(sensitive_cols, y_full, wkh_idx, wkh_w)
    # mdh_group_rates = compute_group_positive_rates(sensitive_cols, y_full, mdh_idx, mdh_w)
    multisampler_group_rates = compute_group_positive_rates(sensitive_cols, y_full, multisampler_idx, multsampler_q)
    res_group_rates = compute_group_positive_rates(sensitive_cols, y_full, res_idx, res_weights)

    group_rates = {
        'FairStreamer1': fair_group_rates1['sex'],
        #'FairStreamer2': fair_group_rates2['sex'],
        'WKH': wkh_group_rates['sex'],
        # 'MDH': mdh_group_rates['sex'],
        'MultiSampler': multisampler_group_rates['sex'],
        'Random Sampling': res_group_rates['sex']
    }

    plot_group_rates(baseline_names, group_rates, attr_name='sex')

    # Print a concise table of group rates
    # print("\nGroup positive rates (full vs coreset):")
    # for baseline in baseline_names:
    #     print(f"\n--- {baseline} ---")
    #     for g, (full_r, co_r) in (fair_group_rates['sex'].items() if baseline=='FairStreamer' else res_group_rates['sex'].items()):
    #         co = fair_group_rates['sex'][g] if baseline=='FairStreamer' else res_group_rates['sex'][g]
    #         print(f"Group {g}: full={co[0]:.4f} | coreset={co[1]:.4f}")


if __name__ == '__main__':
    main()
