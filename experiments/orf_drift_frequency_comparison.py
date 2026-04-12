"""
Compare StreamingCoreset MMD (L2 in RFF space vs stream running mean) for two
orderings of the *same* multiset of points and the *same* final class histogram:

  1) Monotonic (orf_experience-style): classes appear as contiguous blocks 0..9.
  2) Chunk-interleaved: each class is split into NUM_SPLITS sub-blocks; blocks are
     emitted in round-robin order (many more label / concept transitions).

Total stream length and per-class counts match experiments/orf_experience.py
lines 97--108. Hyperparameters (M, RFF, gamma, ...) match that file unless noted.

Outputs a single plot: both MMD trajectories vs stream step t.
"""

from __future__ import annotations

import os
import random
import sys
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

from dataloaders import load_dataset
from streamers.orf_streamer import OrthogonalSampler, StreamingCoreset

try:
    import torch  # noqa: F401 — load_dataset may use it
except ImportError as e:
    raise ImportError("torch required. " + str(e))

# ---------------------------------------------------------------------------
# Same experiment knobs as orf_experience.py
# ---------------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

NUM_PER_CLASS = 250
N_CLASSES = 10

# Identical final distribution to orf_experience.py:97-108
PER_CLASS: Dict[int, int] = {
    0: NUM_PER_CLASS,
    1: NUM_PER_CLASS * 2,
    2: NUM_PER_CLASS * 3,
    3: NUM_PER_CLASS * 3,
    4: NUM_PER_CLASS * 4,
    5: NUM_PER_CLASS * 5,
    6: NUM_PER_CLASS * 4,
    7: NUM_PER_CLASS * 3,
    8: NUM_PER_CLASS * 2,
    9: NUM_PER_CLASS,
}

M = 50
RFF_DIM = 1024
RFF_GAMMA = 0.001
K_iter = 10
BATCH_SIZE = 1

# More splits => more chunk boundaries => more class transitions (but still
# long coherent mini-blocks, so the stream does not look IID / stationary).
NUM_SPLITS = 8

SNAPSHOT_DIR = "snapshots_drift_frequency_comparison"
os.makedirs(SNAPSHOT_DIR, exist_ok=True)


def sample_stratified_pool_embedded(seed: int = SEED) -> Tuple[np.ndarray, np.ndarray]:
    """Same sampling as orf_experience: stratified counts PER_CLASS, class blocks 0..9."""
    train_loader, val_loader, X_train, X_val, y_train, y_val = load_dataset(
        "mnist",
        50000,
        50000,
        seed,
        embedding="resnet18",
        embed_dim=None,
        device="cpu",
    )
    X_all = X_train
    y_all = y_train
    idxs: List[np.ndarray] = []
    rng = np.random.RandomState(seed)
    for c in range(N_CLASSES):
        class_idx = np.where(y_all == c)[0]
        chosen = rng.choice(class_idx, size=PER_CLASS[c], replace=False)
        idxs.append(chosen)
    idxs_arr = np.concatenate(idxs)
    return X_all[idxs_arr], y_all[idxs_arr]


def class_block_starts(per_class: Dict[int, int]) -> np.ndarray:
    starts = np.zeros(N_CLASSES, dtype=np.int64)
    acc = 0
    for c in range(N_CLASSES):
        starts[c] = acc
        acc += per_class[c]
    return starts


def interleaved_chunk_indices(per_class: Dict[int, int], num_splits: int) -> np.ndarray:
    """
    Split each class contiguous block into `num_splits` slices, then emit slices
    in round-robin class order (0,1,...,9,0,1,...). Preserves multiset of indices.
    """
    if num_splits < 1:
        raise ValueError("num_splits must be >= 1")
    starts = class_block_starts(per_class)
    by_class: List[List[np.ndarray]] = [[] for _ in range(N_CLASSES)]
    for c in range(N_CLASSES):
        n = per_class[c]
        base = starts[c]
        edges = np.linspace(0, n, num_splits + 1, dtype=np.int64)
        for s in range(num_splits):
            lo, hi = int(edges[s]), int(edges[s + 1])
            if lo < hi:
                by_class[c].append(np.arange(base + lo, base + hi, dtype=np.int64))
    max_rounds = max(len(by_class[c]) for c in range(N_CLASSES))
    out: List[np.ndarray] = []
    for r in range(max_rounds):
        for c in range(N_CLASSES):
            if r < len(by_class[c]):
                out.append(by_class[c][r])
    return np.concatenate(out) if out else np.array([], dtype=np.int64)


def count_label_transitions(y: np.ndarray) -> int:
    if len(y) <= 1:
        return 0
    return int(np.sum(y[1:] != y[:-1]))


def estimate_drift_bound(X: np.ndarray, sampler: OrthogonalSampler) -> Tuple[float, float]:
    """Max and mean per-step ||mu_t - mu_{t-1}|| in RFF space."""
    Z = sampler.transform(X)
    n = Z.shape[0]
    mu = np.zeros(Z.shape[1])
    prev_mu = np.zeros(Z.shape[1])
    max_drift = 0.0
    drifts: List[float] = []
    for t in range(1, n + 1):
        mu = ((t - 1) * mu + Z[t - 1]) / float(t)
        if t > 1:
            drift = float(np.linalg.norm(mu - prev_mu))
            drifts.append(drift)
            max_drift = max(max_drift, drift)
        prev_mu = mu.copy()
    mean_drift = float(np.mean(drifts)) if drifts else 0.0
    return max_drift, mean_drift


def run_streaming_coreset(
    X: np.ndarray,
    y: np.ndarray,
    sampler: OrthogonalSampler,
    delta_drift_max: float,
) -> List[float]:
    streamer = StreamingCoreset(
        M=M,
        D=RFF_DIM,
        delta_drift_max=delta_drift_max,
        sampler=sampler,
        batch_size=BATCH_SIZE,
        K_iter=K_iter,
        verbose=False,
    )
    n_total = X.shape[0]
    for t in range(n_total):
        x_t = X[t : t + 1]
        y_t = y[t : t + 1]
        streamer.process_batch(x_t, y_t, batch_idx=t)
    return list(streamer.mmd_history)


def plot_mmd_comparison(
    mmd_mono: np.ndarray,
    mmd_inter: np.ndarray,
    save_path: str,
    n_trans_mono: int,
    n_trans_inter: int,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    t = np.arange(1, len(mmd_mono) + 1)
    ax.plot(
        t,
        mmd_mono,
        linewidth=0.85,
        color="steelblue",
        label=f"Monotonic class blocks ({n_trans_mono} label transitions)",
    )
    ax.plot(
        t,
        mmd_inter,
        linewidth=0.85,
        color="darkorange",
        label=f"Chunk-interleaved (NUM_SPLITS={NUM_SPLITS}, {n_trans_inter} transitions)",
    )
    ax.set_xlabel("Stream step t")
    ax.set_ylabel(r"MMD $= \|\mu_t^{\mathrm{stream}} - \mu_t^{\mathrm{buffer}}\|_2$ (RFF)")
    ax.set_title(
        "StreamingCoreset: L2 distance in RFF space vs true cumulative stream mean"
    )
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def main() -> None:
    print("=" * 70)
    print("  DRIFT FREQUENCY COMPARISON (same counts, different order)")
    print("=" * 70)

    print("\n[1] Loading stratified MNIST (embedded) pool...")
    X_pool, y_pool = sample_stratified_pool_embedded(SEED)
    n_total = X_pool.shape[0]
    assert n_total == sum(PER_CLASS[c] for c in range(N_CLASSES))
    print(f"    Total points: {n_total}, feature dim: {X_pool.shape[1]}")

    idx_mono = np.arange(n_total, dtype=np.int64)
    idx_inter = interleaved_chunk_indices(PER_CLASS, NUM_SPLITS)
    assert np.array_equal(np.sort(idx_inter), np.sort(idx_mono))

    X_mono, y_mono = X_pool, y_pool
    X_inter, y_inter = X_pool[idx_inter], y_pool[idx_inter]

    n_tr_mono = count_label_transitions(y_mono)
    n_tr_inter = count_label_transitions(y_inter)
    print(f"    Label transitions (monotonic):   {n_tr_mono}")
    print(f"    Label transitions (interleaved):   {n_tr_inter}")

    print("\n[2] Orthogonal RFF sampler (shared across both runs)...")
    sampler = OrthogonalSampler(
        d_in=X_pool.shape[1], n_components=RFF_DIM, gamma=RFF_GAMMA
    )

    print("\n[3] Drift statistics in RFF space (per stream ordering)...")
    max_d_mono, mean_d_mono = estimate_drift_bound(X_mono, sampler)
    max_d_inter, mean_d_inter = estimate_drift_bound(X_inter, sampler)
    delta_mono = max_d_mono * 1.1
    delta_inter = max_d_inter * 1.1
    print(f"    Monotonic:   max step drift={max_d_mono:.6f}, mean={mean_d_mono:.6f} -> delta={delta_mono:.6f}")
    print(
        f"    Interleaved: max step drift={max_d_inter:.6f}, mean={mean_d_inter:.6f} -> delta={delta_inter:.6f}"
    )

    print("\n[4] Running StreamingCoreset (monotonic)...")
    mmd_mono = run_streaming_coreset(X_mono, y_mono, sampler, delta_mono)
    print("\n[5] Running StreamingCoreset (chunk-interleaved)...")
    mmd_inter = run_streaming_coreset(X_inter, y_inter, sampler, delta_inter)

    mmd_mono = np.asarray(mmd_mono, dtype=np.float64)
    mmd_inter = np.asarray(mmd_inter, dtype=np.float64)

    print("\n[6] Summary statistics (MMD in RFF space)")
    print(
        f"    Monotonic:   max={mmd_mono.max():.6f}, mean={mmd_mono.mean():.6f}, final={mmd_mono[-1]:.6f}"
    )
    print(
        f"    Interleaved: max={mmd_inter.max():.6f}, mean={mmd_inter.mean():.6f}, final={mmd_inter[-1]:.6f}"
    )

    out_png = os.path.join(SNAPSHOT_DIR, "mmd_monotonic_vs_interleaved.png")
    plot_mmd_comparison(mmd_mono, mmd_inter, out_png, n_tr_mono, n_tr_inter)

    # Optional: numpy archive for later analysis
    np.savez(
        os.path.join(SNAPSHOT_DIR, "mmd_curves.npz"),
        mmd_monotonic=mmd_mono,
        mmd_interleaved=mmd_inter,
        num_splits=NUM_SPLITS,
        n_trans_mono=n_tr_mono,
        n_trans_inter=n_tr_inter,
    )
    print(f"\n  Curves saved to {SNAPSHOT_DIR}/mmd_curves.npz")
    print("\n" + "=" * 70)
    print("  DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
