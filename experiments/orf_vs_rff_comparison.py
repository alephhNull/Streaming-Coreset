"""
Fair comparison: StreamingCoreset with Orthogonal Random Features (ORF)
vs. standard i.i.d. Random Fourier Features (sklearn RBFSampler).

Same multiset of points and class histogram, same buffer size M, same D, same
RBF gamma, same K_iter and batch_size. Stream *order* is controlled by NUM_SPLITS
(see experiments/orf_drift_frequency_comparison.py): NUM_SPLITS=1 is monotonic
class blocks; larger values interleave chunk-wise for more concept drift.

Drift bound delta_drift_max is the maximum of the two empirical per-step drift
estimates (times 1.1) so neither method gets a looser theory knob.

Metrics:
  1) L2 distance in *each method's own* RFF space: ||mu_t - sum w_i z_i|| (what
     get_current_mmd tracks) — plotted vs t for both runs.
  2) Exact Gaussian-kernel MMD (sqrt of population MMD^2) between the empirical
     stream prefix and the weighted coreset in input space (same gamma). Sparse
     checkpoints to keep O(N^2) kernel sums manageable (small stream length N).
"""

from __future__ import annotations

import os
import random
import sys
from typing import Dict, List, Optional, Protocol, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.kernel_approximation import RBFSampler
from sklearn.metrics.pairwise import rbf_kernel

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

from streamers.orf_streamer import OrthogonalSampler, StreamingCoreset

try:
    from dataloaders import load_dataset
except ImportError:
    load_dataset = None  # type: ignore


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# Smaller stream so exact kernel MMD (O(N^2) in stream length at each checkpoint)
# stays tractable; increase at your own CPU/RAM cost.
NUM_PER_CLASS = 40
N_CLASSES = 10
BATCH_SIZE = 1
M = 50
RFF_DIM = 1024
RFF_GAMMA = 0.001
K_iter = 10

# Compute exact RBF MMD every this many steps (1 = every step; only for small N).
EXACT_MMD_STRIDE = 5

# Concept-drift frequency: same multiset and class histogram as before, but reorder
# the stream. NUM_SPLITS=1 => monotonic class blocks 0..9 (default, original behavior).
# Larger NUM_SPLITS => each class block is split into more slices emitted in
# round-robin order (more label transitions). See experiments/orf_drift_frequency_comparison.py.
NUM_SPLITS = 1

SNAPSHOT_DIR = os.path.join("snapshots_orf_vs_rff", f"splits_{NUM_SPLITS}")
os.makedirs(SNAPSHOT_DIR, exist_ok=True)


class _Sampler(Protocol):
    n_components: int

    def transform(self, X: np.ndarray) -> np.ndarray: ...


def per_class_counts(num_per_class: int) -> Dict[int, int]:
    """Same relative histogram as orf_experience / orf_drift_frequency_comparison."""
    return {
        0: num_per_class,
        1: num_per_class * 2,
        2: num_per_class * 3,
        3: num_per_class * 3,
        4: num_per_class * 4,
        5: num_per_class * 5,
        6: num_per_class * 4,
        7: num_per_class * 3,
        8: num_per_class * 2,
        9: num_per_class,
    }


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
    in round-robin class order (0,1,...,9,0,1,...). Preserves multiset of row indices
    into the monotonic pool [class0 block | class1 block | ...].
    num_splits == 1 recovers strict monotonic class blocks.
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


def label_transition_steps_1based(y: np.ndarray) -> List[int]:
    """1-based stream step indices where the label changes vs the previous point."""
    out: List[int] = []
    for t in range(1, len(y)):
        if y[t] != y[t - 1]:
            out.append(t + 1)
    return out


def sample_stratified_mnist_subset_embedded(
    num_per_class: int = NUM_PER_CLASS,
    seed: int = SEED,
    num_splits: int = NUM_SPLITS,
) -> Tuple[np.ndarray, np.ndarray, Optional[List[int]], Dict[int, int]]:
    """
    Returns ordered stream (X, y), optional vertical-line steps for plots, and PER_CLASS.

    With num_splits==1, y is monotonic by class (original behavior). Vertical lines
    are major block boundaries (between classes). With num_splits>1, vertical lines
    are every label transition (can be dense).
    """
    if load_dataset is None:
        raise ImportError("dataloaders.load_dataset required for embedded MNIST")
    train_loader, val_loader, X_train, X_val, y_train, y_val = load_dataset(
        "mnist", 50000, 50000, seed, embedding="resnet18", embed_dim=None, device="cpu"
    )
    X_all = X_train
    y_all = y_train
    rng = np.random.RandomState(seed)
    per_class = per_class_counts(num_per_class)
    idxs: List[np.ndarray] = []
    for c in range(N_CLASSES):
        class_idx = np.where(y_all == c)[0]
        chosen = rng.choice(class_idx, size=per_class[c], replace=False)
        idxs.append(chosen)
    idxs_flat = np.concatenate(idxs)
    X_pool = X_all[idxs_flat]
    y_pool = y_all[idxs_flat]
    n_total = X_pool.shape[0]
    idx_mono = np.arange(n_total, dtype=np.int64)
    idx_order = interleaved_chunk_indices(per_class, num_splits)
    assert np.array_equal(np.sort(idx_order), idx_mono)

    X = X_pool[idx_order]
    y = y_pool[idx_order]

    if num_splits == 1:
        counts = np.array([per_class[c] for c in range(N_CLASSES)])
        class_change_steps = np.cumsum(counts)[:-1].tolist()
    else:
        class_change_steps = label_transition_steps_1based(y)

    return X, y, class_change_steps, per_class


def estimate_drift_bound(X: np.ndarray, sampler: _Sampler) -> Tuple[float, List[float]]:
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
            if drift > max_drift:
                max_drift = drift
        prev_mu = mu.copy()
    return max_drift, drifts


def compute_exact_rbf_mmd(
    X_stream: np.ndarray,
    buffer_X: list,
    buffer_weights: np.ndarray,
    gamma: float,
) -> float:
    """Exact MMD with Gaussian RBF kernel (same as orf_experience)."""
    if len(buffer_X) == 0:
        return float("inf")
    Z = np.vstack(buffer_X)
    w = np.array(buffer_weights, dtype=np.float64)
    w = w / np.sum(w)
    N = X_stream.shape[0]
    K_ZZ = rbf_kernel(Z, Z, gamma=gamma)
    term_ZZ = float(w.T @ K_ZZ @ w)
    K_XZ = rbf_kernel(X_stream, Z, gamma=gamma)
    term_XZ = 2.0 * float(np.mean(K_XZ @ w))
    term_XX = 0.0
    chunk_size = 2000
    for i in range(0, N, chunk_size):
        end_i = min(i + chunk_size, N)
        K_XX_chunk = rbf_kernel(X_stream[i:end_i], X_stream, gamma=gamma)
        term_XX += float(np.sum(K_XX_chunk))
    term_XX /= N * N
    mmd_sq = term_XX - term_XZ + term_ZZ
    return float(np.sqrt(max(0.0, mmd_sq)))


def plot_l2_vs_t(
    l2_orf: List[float],
    l2_rff: List[float],
    save_path: str,
    class_change_steps: Optional[List[int]] = None,
    title_suffix: str = "",
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    steps = np.arange(1, len(l2_orf) + 1)
    ax.plot(steps, l2_orf, linewidth=0.9, color="steelblue", label="ORF (orthogonal)")
    ax.plot(steps, l2_rff, linewidth=0.9, color="coral", label="RFF (i.i.d. RBFSampler)")
    if class_change_steps:
        max_vlines = 80
        lines = class_change_steps[:max_vlines] if len(class_change_steps) > max_vlines else class_change_steps
        alpha = 0.35 if len(class_change_steps) > 25 else 0.6
        for t in lines:
            if 1 <= t <= len(l2_orf):
                ax.axvline(x=t, color="gray", linestyle="--", linewidth=0.45, alpha=alpha)
    ax.set_xlabel("Stream step t")
    ax.set_ylabel(r"L2 $\|\mu_t - \sum_i w_i z_i\|$ (each method's RFF space)")
    ax.set_title("Approximate kernel MMD surrogate in feature space: ORF vs RFF" + title_suffix)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_exact_mmd_vs_t(
    steps_chk: np.ndarray,
    mmd_orf: np.ndarray,
    mmd_rff: np.ndarray,
    save_path: str,
    title_suffix: str = "",
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    ax.plot(steps_chk, mmd_orf, marker="o", ms=3, linewidth=1.0, color="steelblue", label="ORF coreset")
    ax.plot(steps_chk, mmd_rff, marker="s", ms=3, linewidth=1.0, color="coral", label="RFF coreset")
    ax.set_xlabel("Stream step t")
    ax.set_ylabel("Exact RBF MMD (input space)")
    ax.set_title(
        f"Exact kernel MMD vs t (gamma={RFF_GAMMA}, checkpoints every {EXACT_MMD_STRIDE})" + title_suffix
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def main() -> None:
    print("=" * 70)
    print("  ORF vs standard RFF: fair StreamingCoreset comparison")
    print("=" * 70)

    print("\n[1] Loading embedded MNIST stream...")
    X, y, class_change_steps, per_class = sample_stratified_mnist_subset_embedded(
        NUM_PER_CLASS, SEED, NUM_SPLITS
    )
    n_total = X.shape[0]
    d_in = X.shape[1]
    n_trans = count_label_transitions(y)
    print(f"    N={n_total}, d_in={d_in}, M={M}, D={RFF_DIM}, gamma={RFF_GAMMA}")
    print(f"    NUM_SPLITS={NUM_SPLITS}  |  label transitions={n_trans}")
    if len(class_change_steps or []) > 80:
        print(f"    (vertical lines on L2 plot: first {80} transition steps only)")

    # Independent draws: ORF uses global numpy RNG; RFF uses sklearn random_state.
    np.random.seed(SEED)
    sampler_orf = OrthogonalSampler(d_in=d_in, n_components=RFF_DIM, gamma=RFF_GAMMA)
    sampler_rff = RBFSampler(gamma=RFF_GAMMA, n_components=RFF_DIM, random_state=SEED + 12345)
    sampler_rff.fit(X[: min(10, n_total)])

    print("\n[2] Empirical drift bounds...")
    d_orf, _ = estimate_drift_bound(X, sampler_orf)
    d_rff, _ = estimate_drift_bound(X, sampler_rff)
    delta_safe = max(d_orf, d_rff) * 1.1
    print(f"    max drift ORF: {d_orf:.6f}")
    print(f"    max drift RFF: {d_rff:.6f}")
    print(f"    shared delta_drift_max (1.1 * max): {delta_safe:.6f}")

    print("\n[3] Run both streamers (one pass; exact MMD at checkpoints)...")
    streamer_orf = StreamingCoreset(
        M=M,
        D=RFF_DIM,
        delta_drift_max=delta_safe,
        sampler=sampler_orf,
        batch_size=BATCH_SIZE,
        K_iter=K_iter,
        verbose=False,
    )
    streamer_rff = StreamingCoreset(
        M=M,
        D=RFF_DIM,
        delta_drift_max=delta_safe,
        sampler=sampler_rff,
        batch_size=BATCH_SIZE,
        K_iter=K_iter,
        verbose=False,
    )

    mmd_exact_orf: List[float] = []
    mmd_exact_rff: List[float] = []
    chk_steps: List[int] = []

    for t in range(n_total):
        streamer_orf.process_batch(X[t : t + 1], y[t : t + 1], batch_idx=t)
        streamer_rff.process_batch(X[t : t + 1], y[t : t + 1], batch_idx=t)
        record = t % EXACT_MMD_STRIDE == 0 or t == n_total - 1
        if record:
            X_prefix = X[: t + 1]
            mmd_exact_orf.append(
                compute_exact_rbf_mmd(X_prefix, streamer_orf.buffer_X, streamer_orf.buffer_weights, RFF_GAMMA)
            )
            mmd_exact_rff.append(
                compute_exact_rbf_mmd(X_prefix, streamer_rff.buffer_X, streamer_rff.buffer_weights, RFF_GAMMA)
            )
            chk_steps.append(t)
            if t == 0 or t == n_total - 1:
                print(
                    f"    t={t} (1-based end step {t+1}): exact MMD ORF={mmd_exact_orf[-1]:.6f}  RFF={mmd_exact_rff[-1]:.6f}"
                )

    l2_orf = list(streamer_orf.mmd_history)
    l2_rff = list(streamer_rff.mmd_history)
    print(f"  Final L2 in ORF feature space: {l2_orf[-1]:.6f}")
    print(f"  Final L2 in RFF feature space: {l2_rff[-1]:.6f}")

    chk_steps_arr = np.array(chk_steps, dtype=int)

    suffix = f"  [NUM_SPLITS={NUM_SPLITS}, transitions={n_trans}]"
    print("\n[4] Plots...")
    plot_l2_vs_t(
        l2_orf,
        l2_rff,
        os.path.join(SNAPSHOT_DIR, "l2_rff_space_orf_vs_rff.png"),
        class_change_steps=class_change_steps,
        title_suffix=suffix,
    )
    plot_exact_mmd_vs_t(
        chk_steps_arr + 1,
        np.array(mmd_exact_orf),
        np.array(mmd_exact_rff),
        os.path.join(SNAPSHOT_DIR, "exact_rbf_mmd_orf_vs_rff.png"),
        title_suffix=suffix,
    )

    np.savez(
        os.path.join(SNAPSHOT_DIR, "curves.npz"),
        l2_orf=np.array(l2_orf),
        l2_rff=np.array(l2_rff),
        exact_mmd_steps=chk_steps_arr + 1,
        exact_mmd_orf=np.array(mmd_exact_orf),
        exact_mmd_rff=np.array(mmd_exact_rff),
        gamma=RFF_GAMMA,
        D=RFF_DIM,
        M=M,
        num_splits=NUM_SPLITS,
        n_label_transitions=n_trans,
        per_class=np.array([per_class[c] for c in range(N_CLASSES)], dtype=np.int32),
    )
    print(f"  Saved: {os.path.join(SNAPSHOT_DIR, 'curves.npz')}")

    print("\n" + "=" * 70)
    print("  DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
