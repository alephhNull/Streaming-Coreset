#!/usr/bin/env python3
"""
check_fair_herding_streamer.py

Quick runner that exercises streamers.fair_herding_streamer.FairKernelHerdingStreamer
on the UCI Adult dataset and compares to a reservoir baseline.

Usage:
    python check_fair_herding_streamer.py
"""

import numpy as np
import pandas as pd
import time
import sys
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.kernel_approximation import RBFSampler
import matplotlib.pyplot as plt

# Import your streamer (adjust path if necessary)
from streamers.fair_herding_streamer import FairKernelHerdingStreamer

# -----------------------
# Kernel / MMD utilities
# -----------------------
def rbf_kernel_np(X, Y, sigma=1.0):
    X = np.asarray(X)
    Y = np.asarray(Y)
    X2 = (X**2).sum(axis=1)[:, None]
    Y2 = (Y**2).sum(axis=1)[None, :]
    d2 = X2 + Y2 - 2 * np.dot(X, Y.T)
    d2 = np.maximum(d2, 0.0)
    return np.exp(-d2 / (2.0 * sigma**2))

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

# -----------------------
# Data loader (Adult)
# -----------------------
def load_adult_processed(subset_size=2000, seed=0):
    np.random.seed(seed)
    print("Loading Adult dataset (may download if not present)...")
    adult = fetch_openml('adult', version=2, as_frame=True)
    df = adult.data.copy()
    target = adult.target.copy()

    # basic cleaning
    df = df.replace('?', np.nan).dropna()
    target = target.loc[df.index].reset_index(drop=True)
    df = df.reset_index(drop=True)

    # take subset
    if subset_size is not None and subset_size < len(df):
        idx = np.random.choice(len(df), size=subset_size, replace=False)
        df = df.iloc[idx].reset_index(drop=True)
        target = target.iloc[idx].reset_index(drop=True)

    # sensitive cols to use
    sens_cols = [c for c in ['sex'] if c in df.columns]
    sens_df = df[sens_cols].copy() if len(sens_cols) else pd.DataFrame(index=df.index)

    # train/val split
    X_train_df, X_val_df, y_train_ser, y_val_ser, sens_train_df, sens_val_df = train_test_split(
        df, target, sens_df, test_size=0.2, random_state=42, stratify=target
    )

    numeric_cols = X_train_df.select_dtypes(include=['int64','float64']).columns.tolist()
    cat_cols = X_train_df.select_dtypes(include=['object','category']).columns.tolist()

    # one-hot encode categorical features
    if cat_cols:
        ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        ohe.fit(X_train_df[cat_cols])
        X_train_cat = ohe.transform(X_train_df[cat_cols])
        X_val_cat = ohe.transform(X_val_df[cat_cols])
    else:
        X_train_cat = np.empty((X_train_df.shape[0], 0))
        X_val_cat = np.empty((X_val_df.shape[0], 0))

    # scale numeric
    if numeric_cols:
        scaler = StandardScaler()
        scaler.fit(X_train_df[numeric_cols])
        X_train_num = scaler.transform(X_train_df[numeric_cols])
        X_val_num = scaler.transform(X_val_df[numeric_cols])
    else:
        X_train_num = np.empty((X_train_df.shape[0], 0))
        X_val_num = np.empty((X_val_df.shape[0], 0))

    X_train = np.hstack((X_train_num, X_train_cat))
    X_val = np.hstack((X_val_num, X_val_cat))

    le = LabelEncoder()
    y_train = le.fit_transform(y_train_ser)
    y_val = le.transform(y_val_ser)

    sensitive_train = {}
    sensitive_val = {}
    if not sens_train_df.empty:
        for c in sens_train_df.columns:
            sensitive_train[c] = sens_train_df[c].to_numpy()
        for c in sens_val_df.columns:
            sensitive_val[c] = sens_val_df[c].to_numpy()

    print(f"Loaded: X_train {X_train.shape}, X_val {X_val.shape}")
    return X_train, X_val, y_train, y_val, sensitive_train, sensitive_val

# -----------------------
# Reservoir baseline
# -----------------------
def reservoir_baseline(X, y, m, seed=0):
    np.random.seed(seed)
    n = X.shape[0]
    if m >= n:
        idx = np.arange(n, dtype=int)
    else:
        idx = np.random.choice(n, size=m, replace=False)
    w = np.ones(len(idx), dtype=float) / float(len(idx))
    return idx, w

# -----------------------
# Helpers for group rates
# -----------------------
def group_positive_rates_unweighted(indices, sensitive_cols, y):
    y = np.asarray(y)
    rates = {}
    for attr, col in sensitive_cols.items():
        col_arr = np.asarray(col)
        groups = np.unique(col_arr)
        rates[attr] = {}
        for g in groups:
            mask = (col_arr[indices] == g)
            if mask.sum() == 0:
                rates[attr][g] = np.nan
            else:
                rates[attr][g] = float(y[indices][mask].sum()) / float(mask.sum())
    return rates

def group_positive_rates_weighted(indices, weights, sensitive_cols, y):
    y = np.asarray(y)
    rates = {}
    for attr, col in sensitive_cols.items():
        col_arr = np.asarray(col)
        groups = np.unique(col_arr)
        rates[attr] = {}
        for g in groups:
            mask = (col_arr[indices] == g)
            wg = weights[mask]
            if wg.sum() <= 0:
                rates[attr][g] = np.nan
            else:
                rates[attr][g] = float((wg * y[indices][mask]).sum() / (wg.sum()))
    return rates

# -----------------------
# Simple stream runner using the FairKernelHerdingStreamer
# -----------------------
def run_with_streamer(X_train, y_train, sensitive_train,
                      streamer: FairKernelHerdingStreamer,
                      batch_size=32, arrival_interval_ms=None, verbose=False):
    n = X_train.shape[0]
    # create simple batches by slicing sequentially
    indices = np.arange(n)
    batches = [indices[i:i+batch_size] for i in range(0, n, batch_size)]

    # process stream
    for batch_idx, idxs in enumerate(batches):
        batch_X = X_train[idxs]
        batch_y = y_train[idxs]
        # prepare sensitive batch dict (if streamer accepts it)
        sens_batch = {k: sensitive_train[k][idxs] for k in sensitive_train.keys()} if sensitive_train else None
        # streamer.process_batch signature accepts sensitive_batch; pass it
        streamer.process_batch(batch_X, batch_y, batch_idx, sensitive_batch=sens_batch)
        if verbose and ((batch_idx+1) % 50 == 0):
            print(f"Processed batch {batch_idx+1}/{len(batches)}")

    return streamer.get_final_coreset()  # flat_indices, normalized_weights, global_ids

# -----------------------
# MAIN
# -----------------------
def main():
    # config
    subset = 1000
    coreset_size = 50
    buffer_capacity = 500
    n_rff = 1024
    gamma = 0.1  # RBFSampler's gamma ~ 1/(2*sigma^2) in some conventions; we keep this as 'gamma' passed to RBFSampler
    rbf_sigma = 10  # sigma used for exact RBF computations and in streamer construction
    seed = 8088
    batch_size = 32
    verbose = False

    # load data
    X_train, X_val, y_train, y_val, sensitive_train, sensitive_val = load_adult_processed(subset_size=subset, seed=seed)

    # Reservoir baseline
    t0 = time.time()
    r_idx, r_w = reservoir_baseline(X_train, y_train, coreset_size, seed=seed)
    t_res = time.time() - t0
    print(f"[Reservoir] time={t_res:.3f}s, coreset={len(r_idx)}")

    # Setup RBFSampler and FairKernelHerdingStreamer
    rbf = RBFSampler(gamma=gamma, n_components=n_rff, random_state=seed)
    rbf.fit(X_train)  # only needed for transformer internals

    streamer = FairKernelHerdingStreamer(
        batch_size=batch_size,
        m_coreset_size=coreset_size,
        n_rff_components=n_rff,
        buffer_capacity=buffer_capacity,
        rbf_sigma=rbf_sigma,
        random_seed=seed,
        device="cpu",
        select_alternate_freq=None,
        md_iterations=1000,
        md_eta=1,
        verbose=verbose,
    )
    streamer.set_rbf_sampler(rbf)

    # Run streaming with the fair streamer (this will call baseline_fair_mirror inside)
    t0 = time.time()
    fh_flat_idx, fh_w, fh_prov = run_with_streamer(X_train, y_train, sensitive_train, streamer, batch_size=batch_size, verbose=True)
    t_fh = time.time() - t0
    print(f"[FairStreamer] time={t_fh:.3f}s, coreset_points={len(fh_flat_idx)}")

    # Post-check: if get_final_coreset returned empty, print and exit
    if len(fh_flat_idx) == 0:
        print("Fair streamer returned empty coreset — check logs / parameters.")
        return

    # Compute MMD^2 (exact) between full dataset and coresets
    Xc_res = X_train[r_idx]
    Xc_fh = X_train[fh_flat_idx]
    mmd_res = mmd2_exact_full_vs_coreset(X_train, Xc_res, r_w, sigma=rbf_sigma)
    mmd_fh = mmd2_exact_full_vs_coreset(X_train, Xc_fh, fh_w, sigma=rbf_sigma)
    print(f"MMD^2 Reservoir: {mmd_res:.6g}")
    print(f"MMD^2 FairHerding: {mmd_fh:.6g}")

    # Compute group positive rates and print
    print("\nGroup positive rates (Full / Reservoir-weighted / Fair-weighted):")
    for attr, col in sensitive_train.items():
        groups = np.unique(col)
        # full
        full_rates = {g: float(y_train[col == g].sum()) / float((col == g).sum()) for g in groups}
        res_w_rates = group_positive_rates_weighted(r_idx, r_w, {attr: col}, y_train)[attr]
        fh_w_rates = group_positive_rates_weighted(fh_flat_idx, fh_w, {attr: col}, y_train)[attr]
        print(f"\nAttribute: {attr}")
        print(" Full:", full_rates)
        print(" Reservoir-weighted:", res_w_rates)
        print(" FairHerding-weighted:", fh_w_rates)

        # plot
        labels = [str(g) for g in groups]
        full_vals = [full_rates[g] for g in groups]
        res_vals = [res_w_rates.get(g, np.nan) for g in groups]
        fh_vals = [fh_w_rates.get(g, np.nan) for g in groups]

        x = np.arange(len(groups))
        width = 0.25
        plt.figure(figsize=(7,3.5))
        plt.bar(x - width, full_vals, width, label='Full')
        plt.bar(x, res_vals, width, label='Reservoir-weighted')
        plt.bar(x + width, fh_vals, width, label='FairHerding-weighted')
        plt.xticks(x, labels, rotation=45)
        plt.ylabel("Positive rate")
        plt.title(f"Positive rates for {attr}")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # MMD bar
    plt.figure(figsize=(4,3))
    plt.bar(['Reservoir', 'FairHerding'], [mmd_res, mmd_fh])
    plt.ylabel('MMD^2 to full')
    plt.title('MMD^2 comparison')
    plt.tight_layout()
    plt.show()

    # Print coreset provenance (if streamer stores it)
    try:
        print("\nFair streamer provenance / sample of coreset global ids:")
        print(fh_prov[:10] if isinstance(fh_prov, (list, tuple)) else fh_prov)
    except Exception:
        pass

    print("\nDone.")

if __name__ == "__main__":
    main()
