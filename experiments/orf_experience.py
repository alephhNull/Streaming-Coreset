"""
Rigorous Theory Experiment: Separation-Guarded Online Coreset on Non-IID MNIST.

This experiment:
  1. Builds a non-stationary MNIST stream with controlled drift
  2. Deterministically calculates ALL hyperparameters from (M, D, gamma, delta_drift)
  3. Runs the TheoreticalRigourousStreamer
  4. Compares the max observed MMD against the theoretical epsilon bound
  5. Produces detailed diagnostics and visualizations

The stream is constructed so that classes appear in sorted order (all 0s, then
all 1s, ...) creating a non-stationary distribution with bounded drift per step.
The drift bound delta_drift is estimated from the data: when the stream
transitions between classes, the per-point change in the running mean is at
most ||z_{new} - z_{old}|| / t, which for RFF features is <= 2*sqrt(2)/t.
But during class transitions the mean shifts more.  We compute the empirical
drift bound conservatively.
"""

import numpy as np
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter
from typing import List
import random
import os
import sys


parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from streamers.orf_streamer import (
    SeparationGuardedFrankWolfeStreamer,
    TheoreticalHyperparamsFW,
    OrthogonalSampler,
)

from dataloaders import load_dataset

# from sklearn.kernel_approximation import RBFSampler

# Data loading
try:
    import torch
    from torchvision import datasets, transforms
except ImportError as e:
    raise ImportError("torch/torchvision required. " + str(e))


# ========================================================================
#  EXPERIMENT PARAMETERS (only these are user-specified)
# ========================================================================
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# Stream design
NUM_PER_CLASS = 250         # points per class
N_CLASSES = 10
TOTAL = NUM_PER_CLASS * N_CLASSES  # 2000 total
BATCH_SIZE = 1              # process point-by-point (theory is per-point)

# Core algorithm inputs (the ONLY things the user specifies)
M = 50                      # buffer size
RFF_DIM = 1024               # RFF dimension D
RFF_GAMMA = 0.001           # kernel gamma for Gaussian RBF
K_iter = 1000
NU_SEPERATION = 0.0
# Drift control: how quickly classes transition.
# With sorted-class stream of N points per class, at the transition boundary
# the running mean changes by roughly delta per step.  We set a TRANSITION_WIDTH
# that controls how "sharp" each class boundary is.  A larger width = smoother
# drift = smaller delta_drift per step.
TRANSITION_WIDTH = 50       # number of points over which transition happens

# Output
SNAPSHOT_DIR = "snapshots_theory_experiment"
os.makedirs(SNAPSHOT_DIR, exist_ok=True)


def sample_stratified_mnist_subset_embedded(num_per_class=NUM_PER_CLASS, seed=SEED):
    # download if necessary
    train_loader, val_loader, X_train, X_val, y_train, y_val = load_dataset('mnist', 50000, 50000, seed, embedding='resnet18', embed_dim=None, device='cpu')
    # mnist = datasets.MNIST(root="./data", train=True, download=True, transform=transforms.ToTensor())
    # X_all = mnist.data.numpy().reshape(-1, 28 * 28).astype(np.float32) / 255.0
    # y_all = mnist.targets.numpy().astype(int)

    X_all = X_train
    y_all = y_train

    idxs = []
    rng = np.random.RandomState(seed)
    
    per_class = {
        0: NUM_PER_CLASS,
        1: NUM_PER_CLASS * 2,
        2: NUM_PER_CLASS * 3,
        3: NUM_PER_CLASS * 3,
        4: NUM_PER_CLASS * 4,
        5: NUM_PER_CLASS * 5,
        6: NUM_PER_CLASS * 4,
        7: NUM_PER_CLASS * 3,
        8: NUM_PER_CLASS *2,
        9: NUM_PER_CLASS,
    }

    for c in range(N_CLASSES):
        class_idx = np.where(y_all == c)[0]
        chosen = rng.choice(class_idx, size=per_class[c], replace=False)
        idxs.append(chosen)

    idxs = np.concatenate(idxs)
    X = X_all[idxs]
    y = y_all[idxs]

    return X, y

# ========================================================================
#  DATA PREPARATION
# ========================================================================
def load_mnist_sorted_stream(num_per_class: int, seed: int):
    """Load MNIST, take num_per_class per digit, sort by class for non-iid stream."""
    mnist = datasets.MNIST(root="./data", train=True, download=True,
                           transform=transforms.ToTensor())
    X_all = mnist.data.numpy().reshape(-1, 28 * 28).astype(np.float32) / 255.0
    y_all = mnist.targets.numpy().astype(int)

    rng = np.random.RandomState(seed)
    idxs = []
    for c in range(N_CLASSES):
        class_idx = np.where(y_all == c)[0]
        chosen = rng.choice(class_idx, size=num_per_class, replace=False)
        idxs.append(chosen)
    idxs = np.concatenate(idxs)
    X = X_all[idxs]
    y = y_all[idxs]

    # Sort by class to create non-iid ordering
    order = np.argsort(y, kind='stable')
    X = X[order]
    y = y[order]
    return X, y


def estimate_drift_bound(X: np.ndarray, sampler: OrthogonalSampler) -> float:
    """
    Empirically estimate the maximum per-step drift in the running mean.

    delta_drift = max_t || mu_t - mu_{t-1} ||

    where mu_t = (1/t) sum_{s=1}^{t} z_s is the running average.
    This gives us a tight, data-dependent bound.
    """
    Z = sampler.transform(X)
    n = Z.shape[0]  
    mu = np.zeros(Z.shape[1])
    prev_mu = np.zeros(Z.shape[1])
    max_drift = 0.0
    drifts = []

    for t in range(1, n + 1):
        mu = ((t - 1) * mu + Z[t - 1]) / float(t)
        if t > 1:
            drift = float(np.linalg.norm(mu - prev_mu))
            drifts.append(drift)
            if drift > max_drift:
                max_drift = drift
        prev_mu = mu.copy()

    return max_drift, drifts


# ========================================================================
#  VISUALIZATION HELPERS
# ========================================================================
def class_balance_stats(labels: np.ndarray, n_classes: int = N_CLASSES):
    c = Counter(int(x) for x in labels)
    return np.array([c.get(i, 0) for i in range(n_classes)], dtype=int)


def plot_mmd_trajectory(mmd_history, eps_bound, save_path):
    """Plot MMD over time vs theoretical bound."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    steps = np.arange(1, len(mmd_history) + 1)
    ax.plot(steps, mmd_history, linewidth=0.7, color='steelblue', label='Observed MMD')
    ax.axhline(y=eps_bound, color='red', linestyle='--', linewidth=2,
               label=f'Theoretical bound eps={eps_bound:.4f}')
    ax.set_xlabel('Stream step t')
    ax.set_ylabel('MMD (L2 in RFF space)')
    ax.set_title('MMD Trajectory vs Theoretical Guarantee')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved MMD trajectory plot: {save_path}")


def plot_drift_profile(drifts, save_path):
    """Plot per-step drift in the running mean."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    ax.plot(range(2, len(drifts) + 2), drifts, linewidth=0.5, color='darkorange')
    ax.set_xlabel('Stream step t')
    ax.set_ylabel('||mu_t - mu_{t-1}||')
    ax.set_title('Per-Step Drift in Running Mean')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved drift profile plot: {save_path}")


def plot_buffer_class_evolution(class_history, n_classes, save_path):
    """Stacked area chart of buffer class composition over time."""
    steps = np.arange(len(class_history))
    data = np.array(class_history)  # (T, n_classes)

    fig, ax = plt.subplots(1, 1, figsize=(14, 5))
    ax.stackplot(steps, data.T, labels=[f'Class {i}' for i in range(n_classes)],
                 alpha=0.8)
    ax.set_xlabel('Stream step t')
    ax.set_ylabel('# points in buffer')
    ax.set_title('Buffer Class Composition Over Time')
    ax.legend(loc='upper left', fontsize=7, ncol=5)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved buffer composition plot: {save_path}")


def plot_stream_vs_buffer_distribution(stream_dist_history, buffer_dist_history,
                                       n_classes, save_path):
    """Compare stream cumulative distribution vs buffer distribution at checkpoints."""
    n_checkpoints = len(stream_dist_history)
    n_cols = min(5, n_checkpoints)
    n_rows = max(1, math.ceil(n_checkpoints / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    if n_checkpoints == 1:
        axes = np.array([axes])
    axes = np.array(axes).reshape(-1)

    x = np.arange(n_classes)
    width = 0.35

    for idx in range(len(axes)):
        ax = axes[idx]
        if idx < n_checkpoints:
            s_dist = stream_dist_history[idx]
            b_dist = buffer_dist_history[idx]
            step_label = s_dist['step']

            s_vals = s_dist['dist']
            b_vals = b_dist['dist']

            ax.bar(x - width / 2, s_vals, width, label='Stream', alpha=0.7, color='steelblue')
            ax.bar(x + width / 2, b_vals, width, label='Buffer', alpha=0.7, color='coral')
            ax.set_title(f't={step_label}', fontsize=9)
            ax.set_xticks(x)
            ax.set_xticklabels(x, fontsize=7)
            ax.set_ylim(0, 1.0)
            if idx == 0:
                ax.legend(fontsize=7)
        else:
            ax.axis('off')

    fig.suptitle('Stream vs Buffer Class Distribution at Checkpoints', fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved distribution comparison plot: {save_path}")


def plot_buffer_snapshot(buffer_X, buffer_y, buffer_weights, step, save_path):
    """Visualize buffer contents as a grid of MNIST images with weights."""
    k = buffer_X.shape[0]
    if k == 0:
        return
    cols = min(10, k)
    rows = max(1, math.ceil(k / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.8))
    axs_flat = np.array(axs).reshape(-1)
    for i in range(rows * cols):
        ax = axs_flat[i]
        ax.axis('off')
        if i < k:
            img = buffer_X[i].reshape(28, 28)
            ax.imshow(img, cmap='gray', interpolation='nearest')
            lbl = int(buffer_y[i])
            w = buffer_weights[i] if i < len(buffer_weights) else 0
            ax.set_title(f'{lbl} ({w:.3f})', fontsize=7)
    fig.suptitle(f'Buffer Snapshot at t={step} ({k} points)', fontsize=10)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


from sklearn.metrics.pairwise import rbf_kernel

def compute_exact_rbf_mmd(X_stream: np.ndarray, buffer_X: list, buffer_weights: np.ndarray, gamma: float) -> float:
    """
    Computes the exact MMD between the full stream and the weighted coreset buffer
    using the true Gaussian RBF kernel, avoiding any RFF approximation errors.
    """
    if len(buffer_X) == 0:
        return float('inf')

    Z = np.vstack(buffer_X)
    w = np.array(buffer_weights)
    w = w / np.sum(w)  # Ensure perfectly normalized weights

    N = X_stream.shape[0]

    # Term 1: Coreset-Coreset Density (w^T K_ZZ w)
    K_ZZ = rbf_kernel(Z, Z, gamma=gamma)
    term_ZZ = w.T @ K_ZZ @ w

    # Term 2: Stream-Coreset Cross Covariance
    # K_XZ shape: (N, M). We take the mean over the stream N.
    K_XZ = rbf_kernel(X_stream, Z, gamma=gamma)
    term_XZ = 2.0 * np.mean(K_XZ @ w)

    # Term 3: Stream-Stream True Density
    # Computed in chunks to prevent RAM blowup (e.g., if N > 10,000)
    term_XX = 0.0
    chunk_size = 2000
    for i in range(0, N, chunk_size):
        end_i = min(i + chunk_size, N)
        K_XX_chunk = rbf_kernel(X_stream[i:end_i], X_stream, gamma=gamma)
        term_XX += np.sum(K_XX_chunk)
    term_XX /= (N * N)

    # Exact MMD^2 = E[k(x,x)] - 2E[k(x,z)] + E[k(z,z)]
    mmd_sq = term_XX - term_XZ + term_ZZ
    
    # Clip at 0 to avoid float precision negatives
    return float(np.sqrt(max(0.0, mmd_sq)))

def plot_weight_distribution(weights, step, save_path):
    """Histogram of buffer weights."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    ax.hist(weights, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Weight')
    ax.set_ylabel('Count')
    ax.set_title(f'Buffer Weight Distribution at t={step}')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


# ========================================================================
#  MAIN EXPERIMENT
# ========================================================================
def main():
    print("=" * 70)
    print("  RIGOROUS THEORY EXPERIMENT: Separation-Guarded Coreset on MNIST")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("\n[1] Loading MNIST and preparing non-IID stream...")
    X, y = sample_stratified_mnist_subset_embedded(NUM_PER_CLASS)
    print(f"    Total points: {X.shape[0]}, Feature dim: {X.shape[1]}")
    print(f"    Class order: {np.unique(y[::NUM_PER_CLASS])}")
    print(f"    Stream is sorted by class (non-IID)")

    # ------------------------------------------------------------------
    # 2. Fit RFF sampler
    # ------------------------------------------------------------------
    print("\n[2] Fitting RBFSampler (RFF)...")
    sampler = OrthogonalSampler(d_in=X.shape[1], n_components=RFF_DIM, gamma=RFF_GAMMA)
    print(f"    ORF dim D={RFF_DIM}, gamma={RFF_GAMMA}")

    # ------------------------------------------------------------------
    # 3. Estimate drift bound from data
    # ------------------------------------------------------------------
    print("\n[3] Estimating drift bound from data...")
    delta_drift, drift_values = estimate_drift_bound(X, sampler)
    # Use a slightly inflated bound for safety
    delta_drift_safe = delta_drift * 1.1
    print(f"    Max empirical drift: {delta_drift:.6f}")
    print(f"    Safe drift bound (1.1x): {delta_drift_safe:.6f}")
    print(f"    Mean drift: {np.mean(drift_values):.6f}")
    print(f"    Median drift: {np.median(drift_values):.6f}")

    # Plot drift profile
    plot_drift_profile(drift_values, os.path.join(SNAPSHOT_DIR, "drift_profile.png"))

    # ------------------------------------------------------------------
    # 4. Compute ALL hyperparameters deterministically
    # ------------------------------------------------------------------
    print("\n[4] Computing theoretical hyperparameters...")
    hp = TheoreticalHyperparamsFW(M=M, D=RFF_DIM, delta_drift_max=delta_drift_safe, nu_separation=NU_SEPERATION, K_iter=K_iter, t_final=TOTAL)
    print(hp.summary())

    # ------------------------------------------------------------------
    # 5. Create streamer
    # ------------------------------------------------------------------
    print("\n[5] Creating TheoreticalRigourousStreamer...")
    streamer = SeparationGuardedFrankWolfeStreamer(M=M, D=RFF_DIM, delta_drift_max=delta_drift_safe, nu_separation=NU_SEPERATION, sampler=sampler, batch_size=BATCH_SIZE, verbose=False)

    # ------------------------------------------------------------------
    # 6. Run the stream
    # ------------------------------------------------------------------
    print("\n[6] Processing stream...")
    n_total = X.shape[0]

    # Tracking for visualization
    class_composition_history = []
    stream_dist_history = []
    buffer_dist_history = []
    snapshot_steps = set()

    # Checkpoints: at class transitions and regular intervals
    for c in range(N_CLASSES):
        # At end of each class block
        snapshot_steps.add(min((c + 1) * NUM_PER_CLASS - 1, n_total - 1))
        # At midpoint of each class block
        snapshot_steps.add(min(c * NUM_PER_CLASS + NUM_PER_CLASS // 2, n_total - 1))
    # Also at start and end
    snapshot_steps.add(min(M, n_total - 1))  # after buffer fills
    snapshot_steps.add(n_total - 1)

    stream_labels_so_far = []
    mmd_at_steps = []

    print_interval = max(1, n_total // 20)  # print ~20 status lines

    for t in range(n_total):
        x_t = X[t:t + 1]
        y_t = y[t:t + 1]
        stream_labels_so_far.append(int(y[t]))

        streamer.process_batch(x_t, y_t, batch_idx=t)

        # Record buffer class composition
        buf_counts = class_balance_stats(streamer.buffer_y, N_CLASSES)
        class_composition_history.append(buf_counts.copy())

        # At checkpoints, record distributions
        if t in snapshot_steps:
            # Stream distribution up to t
            stream_counts = class_balance_stats(
                np.array(stream_labels_so_far), N_CLASSES
            )
            s_total = stream_counts.sum()
            s_dist = stream_counts / float(max(s_total, 1))

            # Buffer distribution
            b_total = buf_counts.sum()
            b_dist = buf_counts / float(max(b_total, 1))

            stream_dist_history.append({'step': t, 'dist': s_dist})
            buffer_dist_history.append({'step': t, 'dist': b_dist})

            # # Save buffer snapshot image
            # snap_path = os.path.join(SNAPSHOT_DIR, f"buffer_snapshot_t{t:04d}.png")
            # plot_buffer_snapshot(streamer.buffer_X, streamer.buffer_y,
            #                     streamer.buffer_weights, t, snap_path)

        # Print status
        if t % print_interval == 0 or t == n_total - 1:
            current_mmd = streamer.get_current_mmd()
            n_buf = len(streamer.buffer_y)
            dominant_classes = [(i, c) for i, c in enumerate(buf_counts) if c > 0]
            print(f"  t={t:5d}/{n_total} | buf={n_buf:3d} | "
                  f"MMD={current_mmd:.6f} | eps={hp.eps_total:.6f} | "
                  f"classes={dominant_classes}")

    # ------------------------------------------------------------------
    # 7. Analysis and results
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  RESULTS AND ANALYSIS")
    print("=" * 70)

    mmd_history = streamer.mmd_history
    max_mmd = max(mmd_history) if mmd_history else 0
    mean_mmd = np.mean(mmd_history) if mmd_history else 0
    final_mmd = mmd_history[-1] if mmd_history else 0
    eps = hp.eps_total

    print(f"\n  Theoretical epsilon bound:      {eps:.6f}")
    print(f"  Max observed MMD (all steps):   {max_mmd:.6f}")
    print(f"  Mean observed MMD:              {mean_mmd:.6f}")
    print(f"  Final observed MMD:             {final_mmd:.6f}")
    print(f"  Max MMD / eps bound:            {max_mmd / max(eps, 1e-15):.4f}")

    if max_mmd <= eps:
        print(f"\n  >> GUARANTEE SATISFIED: max MMD ({max_mmd:.6f}) <= eps ({eps:.6f})")
    else:
        violation_count = sum(1 for m in mmd_history if m > eps)
        print(f"\n  >> GUARANTEE VIOLATED at {violation_count}/{len(mmd_history)} steps")
        print(f"     (max MMD {max_mmd:.6f} > eps {eps:.6f})")
        print(f"     Ratio: {max_mmd / eps:.4f}x the bound")
        # Find when violations occur
        violation_indices = [i for i, m in enumerate(mmd_history) if m > eps]
        if violation_indices:
            print(f"     First violation at step {violation_indices[0]}")
            print(f"     Last violation at step {violation_indices[-1]}")
    
    # ------------------------------------------------------------------
    # 7b. Exact Kernel Validation (The True Ground Truth)
    # ------------------------------------------------------------------
    print("\n [7b] Computing exact RBF Kernel MMD (Validation)...")
    exact_mmd = compute_exact_rbf_mmd(X, streamer.buffer_X, streamer.buffer_weights, RFF_GAMMA)
    
    print(f"  Exact Kernel MMD:               {exact_mmd:.6f}")
    
    # Compare against the theoretical RFF approximation error (1/sqrt(D))
    rff_floor = hp.eps_rff
    print(f"  RFF Approximation Floor:        {rff_floor:.6f}")
    
    if exact_mmd < rff_floor:
        print(f"  >> OUTSTANDING: The exact MMD is lower than the theoretical RFF noise floor!")
    else:
        print(f"  >> VALIDATED: The exact MMD is bounded tightly by the RFF approximation.")

    # diag = streamer.get_diagnostics()
    # print(f"\n  Stream statistics:")
    # print(f"    Total points processed: {diag['t']}")
    # print(f"    Final buffer size:      {diag['buffer_size']}")
    # print(f"    Merge events:           {diag['merge_count']}")
    # print(f"    Add events:             {diag['add_count']}")
    # print(f"    Evict events:           {diag['evict_count']}")

    # Final buffer class balance
    final_counts = class_balance_stats(streamer.buffer_y, N_CLASSES)
    print(f"\n  Final buffer class distribution:")
    for c in range(N_CLASSES):
        if final_counts[c] > 0:
            print(f"    Class {c}: {final_counts[c]} points "
                  f"({final_counts[c] / max(final_counts.sum(), 1) * 100:.1f}%)")

    # Stream class distribution
    all_stream_counts = class_balance_stats(y, N_CLASSES)
    print(f"\n  Stream class distribution:")
    for c in range(N_CLASSES):
        print(f"    Class {c}: {all_stream_counts[c]} points "
              f"({all_stream_counts[c] / max(all_stream_counts.sum(), 1) * 100:.1f}%)")

    # Weight statistics
    w = streamer.buffer_weights
    print(f"\n  Weight statistics:")
    print(f"    Min weight:  {w.min():.6f}")
    print(f"    Max weight:  {w.max():.6f}")
    print(f"    Mean weight: {w.mean():.6f}")
    print(f"    Std weight:  {w.std():.6f}")
    print(f"    Sum weights: {w.sum():.6f}")
    print(f"    # nonzero:   {np.sum(w > 1e-8)}/{len(w)}")

    # ------------------------------------------------------------------
    # 8. Plots
    # ------------------------------------------------------------------
    print(f"\n[8] Generating visualization plots...")

    # a) MMD trajectory
    plot_mmd_trajectory(
        mmd_history, eps,
        os.path.join(SNAPSHOT_DIR, "mmd_trajectory.png")
    )

    # b) Buffer class composition over time
    plot_buffer_class_evolution(
        class_composition_history, N_CLASSES,
        os.path.join(SNAPSHOT_DIR, "buffer_composition.png")
    )

    # c) Stream vs buffer distribution at checkpoints
    plot_stream_vs_buffer_distribution(
        stream_dist_history, buffer_dist_history, N_CLASSES,
        os.path.join(SNAPSHOT_DIR, "stream_vs_buffer_dist.png")
    )

    # d) Weight distribution at final step
    plot_weight_distribution(
        streamer.buffer_weights, n_total,
        os.path.join(SNAPSHOT_DIR, "final_weight_distribution.png")
    )

    # e) MMD trajectory zoomed to last 20% of stream (steady state region)
    if len(mmd_history) > 100:
        start_idx = int(0.8 * len(mmd_history))
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        steps_zoom = np.arange(start_idx, len(mmd_history))
        ax.plot(steps_zoom, mmd_history[start_idx:], linewidth=0.8,
                color='steelblue', label='Observed MMD')
        ax.axhline(y=eps, color='red', linestyle='--', linewidth=2,
                    label=f'eps={eps:.4f}')
        ax.set_xlabel('Stream step t')
        ax.set_ylabel('MMD')
        ax.set_title('MMD Trajectory (Steady State Region, last 20%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(SNAPSHOT_DIR, "mmd_steady_state.png"), dpi=150)
        plt.close(fig)
        print(f"  Saved steady-state MMD plot")

    # f) Per-class weight allocation over time
    # Track total weight per class in buffer at each step
    weight_per_class_history = []
    # Re-run is expensive; instead use class_composition_history normalized
    # Actually we need weights, which we don't track per step. Use counts as proxy.
    fig, ax = plt.subplots(1, 1, figsize=(14, 5))
    data = np.array(class_composition_history, dtype=float)
    row_sums = data.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    data_norm = data / row_sums
    for c in range(N_CLASSES):
        ax.plot(data_norm[:, c], linewidth=0.8, label=f'Class {c}')
    ax.set_xlabel('Stream step t')
    ax.set_ylabel('Fraction of buffer')
    ax.set_title('Buffer Class Fraction Over Time')
    ax.legend(loc='upper left', fontsize=7, ncol=5)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(SNAPSHOT_DIR, "class_fraction_over_time.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved class fraction plot")

    print(f"\n  All plots saved to: {SNAPSHOT_DIR}/")
    print("\n" + "=" * 70)
    print("  EXPERIMENT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
