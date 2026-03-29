"""
RQ1: Does STREAMCORE track the true stream mean under severe distribution shifts
without compounding error?

Split-CIFAR10 (class-incremental): stream tasks (0,1) -> (2,3) -> ... -> (8,9)
with 512-d ResNet18 embeddings of the CIFAR-10 *training* set. Full train embeddings
are cached once; the stream uses MAX_PER_CLASS points per class (10k total by default)
for fast exact-MMD iteration.

Compares STREAMCORE (orf_streamer.StreamingCoreset + OrthogonalSampler) against
optional baselines: uniform reservoir sampling and FIFO sliding window (buffer
size M=100 each). Enable/disable baselines via `--no-reservoir` / `--no-sliding-window`.

Metric: exact (finite-sample) RBF kernel MMD^2 between the buffer empirical measure
and the uniform empirical measure over all points seen so far, evaluated every
EVAL_EVERY steps and at task boundaries. The RBF bandwidth γ is set by the **median
heuristic** on the cached training embeddings (not chosen by hand).

Output: `rq1_drift_immunity_output/neurips_fig1_drift_immunity.pdf` (double-column friendly).
By default the figure window is shown after saving; use `--no-show` or `RQ1_HEADLESS=1` to disable.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Sequence, Set, Tuple

import joblib
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from tqdm import tqdm

import matplotlib

# Headless servers: set RQ1_HEADLESS=1. Otherwise prefer TkAgg so we can plt.show() after save.
if os.environ.get("RQ1_HEADLESS", "").lower() in ("1", "true", "yes"):
    matplotlib.use("Agg")
else:
    try:
        matplotlib.use("TkAgg")
    except Exception:
        matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Project root on path (same pattern as other experiments/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

from dataloaders import TransformDataset, extract_resnet18_features
from streamers.orf_streamer import OrthogonalSampler, StreamingCoreset

# ---------------------------------------------------------------------------
# Global experiment constants (paper scenario)
# ---------------------------------------------------------------------------
SEED = 42
M = 100  # buffer size for all methods
# γ for k(x,y)=exp(-γ||x-y||²) is computed via median heuristic after loading X_all (see main).
MEDIAN_HEURISTIC_SAMPLE_SIZE = 2000  # subsample for O(m²) pairwise distances
RFF_D = 1024  # Orthogonal RFF dimension D
K_ITER = 100
EVAL_EVERY = 1000  # less frequent exact MMD — dominant cost scales as O(t^2) per eval
MAX_PER_CLASS = 1000  # 2 classes × 5 tasks × 1000 = 10k stream points
FEATURE_CACHE_PATH = os.path.join(PROJECT_ROOT, "feature_cache", "cifar10_train_resnet18_rq1.pkl")
# All artifacts for this experiment live under experiments/rq1_drift_immunity_output/
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "rq1_drift_immunity_output")
FIGURE_PATH = os.path.join(OUTPUT_DIR, "neurips_fig1_drift_immunity.pdf")


@dataclass
class BaselineConfig:
    """Which baselines to run besides STREAMCORE (always on)."""

    use_reservoir: bool = True
    use_sliding_window: bool = True


# Split-CIFAR10: five tasks, two consecutive classes each (standard definition).
TASK_CLASS_PAIRS: List[Tuple[int, int]] = [
    (0, 1),
    (2, 3),
    (4, 5),
    (6, 7),
    (8, 9),
]


def set_global_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_median_heuristic_gamma(
    X: np.ndarray,
    sample_size: int = MEDIAN_HEURISTIC_SAMPLE_SIZE,
    seed: int = SEED,
) -> float:
    """
    Median heuristic for RBF bandwidth (standard in kernel two-sample / MMD work):

        γ = 1 / (2 · median_{i<j} ||x_i - x_j||²)

    Uses a random subsample when n > sample_size so pairwise distance cost stays tractable.
    Apply to the same feature matrix used for the stream (here, 512-d ResNet18 embeddings).
    """
    n = X.shape[0]
    rng = np.random.RandomState(seed)
    if n > sample_size:
        idx = rng.choice(n, size=sample_size, replace=False)
        X_sub = X[idx]
    else:
        X_sub = X

    D_sq = euclidean_distances(X_sub, squared=True)
    upper_tri_idx = np.triu_indices_from(D_sq, k=1)
    distances_sq = D_sq[upper_tri_idx]
    median_dist_sq = float(np.median(distances_sq))
    if median_dist_sq <= 0:
        raise ValueError("Median squared distance is non-positive; check embeddings.")
    return 1.0 / (2.0 * median_dist_sq)


# =============================================================================
# 1. Data: CIFAR-10 train only -> ResNet18 embeddings (512-d)
# =============================================================================


def load_cifar10_train_embeddings(
    device: torch.device,
    cache_path: str = FEATURE_CACHE_PATH,
    force_recompute: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load CIFAR-10 training split (50,000 images), embed with torchvision ResNet18
    (classification head removed -> 512-d), using dataloaders helpers.

    Cached to disk so reruns skip forward passes.
    """
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    if not force_recompute and os.path.exists(cache_path):
        print(f"Loading cached train embeddings from {cache_path}")
        data = joblib.load(cache_path)
        return data["X"].astype(np.float64), data["y"].astype(np.int64)

    print("Computing ResNet18 embeddings for CIFAR-10 train (50k) — one-time cost...")
    train_ds = datasets.CIFAR10(
        root=os.path.join(PROJECT_ROOT, "data"),
        train=True,
        download=True,
    )
    resnet_transform = transforms.Compose(
        [
            transforms.Resize(224, InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    dataset = TransformDataset(train_ds, resnet_transform)
    X, y = extract_resnet18_features(dataset, device, batch_size=256, num_workers=2)
    X = X.astype(np.float64)
    y = y.astype(np.int64)
    joblib.dump({"X": X, "y": y}, cache_path)
    print(f"Saved embeddings to {cache_path}")
    return X, y


def build_split_cifar10_stream_order(
    X: np.ndarray,
    y: np.ndarray,
    pairs: Sequence[Tuple[int, int]] = TASK_CLASS_PAIRS,
    seed: int = SEED,
    max_per_class: int = MAX_PER_CLASS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int]]:
    """
    Class-incremental stream: per task, up to max_per_class points from each class
    in the pair (random subsample), then sorted by (label, random tie-break) for a
    sharp jump at task boundaries.

    Returns
    -------
    X_stream, y_stream : arrays in stream order
    orig_indices : index into original X for each stream position (for debugging)
    segment_lengths : number of points per task (for transition lines)
    """
    rng = np.random.RandomState(seed)
    order_blocks: List[np.ndarray] = []

    for c0, c1 in pairs:
        idx_c0 = np.where(y == c0)[0]
        idx_c1 = np.where(y == c1)[0]

        idx_c0 = rng.choice(idx_c0, size=min(len(idx_c0), max_per_class), replace=False)
        idx_c1 = rng.choice(idx_c1, size=min(len(idx_c1), max_per_class), replace=False)

        idx = np.concatenate([idx_c0, idx_c1])

        keys = np.stack([y[idx], rng.rand(len(idx))], axis=1)
        sort_order = np.lexsort((keys[:, 1], keys[:, 0]))
        order_blocks.append(idx[sort_order])

    stream_idx = np.concatenate(order_blocks, axis=0)
    segment_lengths = [len(b) for b in order_blocks]
    return X[stream_idx], y[stream_idx], stream_idx, segment_lengths


# =============================================================================
# 2. Baselines: reservoir + sliding window (FIFO)
# =============================================================================


class ReservoirBuffer:
    """Classic reservoir sample: uniform random subset of all points seen (size M)."""

    def __init__(self, M: int, rng: np.random.RandomState):
        self.M = M
        self.rng = rng
        self.buffer: List[np.ndarray] = []
        self.t_seen = 0

    def observe(self, x: np.ndarray) -> None:
        self.t_seen += 1
        if len(self.buffer) < self.M:
            if not self.buffer:
                self._feat_dim = x.shape[0]
            self.buffer.append(x.copy())
            return
        j = self.rng.randint(0, self.t_seen)
        if j < self.M:
            self.buffer[j] = x.copy()

    def get_matrix_weights(self) -> Tuple[np.ndarray, np.ndarray]:
        if not self.buffer:
            d = int(getattr(self, "_feat_dim", 0))
            return np.empty((0, d), dtype=np.float64), np.empty(0, dtype=np.float64)
        Z = np.stack(self.buffer, axis=0)
        m = Z.shape[0]
        w = np.full(m, 1.0 / m, dtype=np.float64)
        return Z, w


class SlidingWindowBuffer:
    """FIFO queue of the last M points."""

    def __init__(self, M: int):
        self.M = M
        self.buf: Deque[np.ndarray] = deque(maxlen=M)

    def observe(self, x: np.ndarray) -> None:
        if not self.buf:
            self._feat_dim = x.shape[0]
        self.buf.append(x.copy())

    def get_matrix_weights(self) -> Tuple[np.ndarray, np.ndarray]:
        if len(self.buf) == 0:
            d = int(getattr(self, "_feat_dim", 0))
            return np.empty((0, d), dtype=np.float64), np.empty(0, dtype=np.float64)
        Z = np.stack(list(self.buf), axis=0)
        m = Z.shape[0]
        w = np.full(m, 1.0 / m, dtype=np.float64)
        return Z, w


# =============================================================================
# 3. Exact RBF MMD^2 (batched; avoids full n×n Gram in memory)
# =============================================================================


def _rbf_kernel_block(X: np.ndarray, Y: np.ndarray, gamma: float) -> np.ndarray:
    """Pairwise RBF kernel between rows of X (n×d) and Y (m×d)."""
    # ||x-y||^2 = ||x||^2 + ||y||^2 - 2 x y^T
    x2 = np.sum(X * X, axis=1, keepdims=True)
    y2 = np.sum(Y * Y, axis=1, keepdims=True).T
    dist_sq = np.maximum(x2 + y2 - 2.0 * (X @ Y.T), 0.0)
    return np.exp(-gamma * dist_sq).astype(np.float64, copy=False)


def term_ek_xx_batched(X: np.ndarray, gamma: float, block: int = 256) -> float:
    """
    E[k(X, X')] under the empirical measure on n points (uniform):
    (1/n^2) * sum_{i,j} k(x_i, x_j).
    """
    n = X.shape[0]
    if n == 0:
        return 0.0
    s = 0.0
    for i0 in range(0, n, block):
        i1 = min(i0 + block, n)
        Xi = X[i0:i1]
        for j0 in range(0, n, block):
            j1 = min(j0 + block, n)
            Xj = X[j0:j1]
            s += float(_rbf_kernel_block(Xi, Xj, gamma).sum())
    return s / float(n * n)


def term_ek_xz_batched(X: np.ndarray, Z: np.ndarray, w: np.ndarray, gamma: float, block: int = 512) -> float:
    """
    E[k(X, Z)] with X uniform on n stream points, Z weighted by w (sum w = 1):
    (1/n) * sum_i sum_j w_j k(x_i, z_j).
    """
    n = X.shape[0]
    m = Z.shape[0]
    if n == 0 or m == 0:
        return 0.0
    acc = 0.0
    for i0 in range(0, n, block):
        i1 = min(i0 + block, n)
        Xi = X[i0:i1]
        K = _rbf_kernel_block(Xi, Z, gamma)  # (b, m)
        acc += float((K @ w).sum())
    return acc / float(n)


def term_ek_zz(Z: np.ndarray, w: np.ndarray, gamma: float) -> float:
    """E[k(Z, Z')] = w^T K_ZZ w for discrete Z with weights w."""
    m = Z.shape[0]
    if m == 0:
        return 0.0
    K = _rbf_kernel_block(Z, Z, gamma)
    return float(w @ K @ w)


def exact_rbf_mmd_squared(
    X_prefix: np.ndarray,
    Z: np.ndarray,
    w: np.ndarray,
    gamma: float,
    block_xx: int = 256,
    block_xz: int = 512,
) -> float:
    """
    MMD^2 between P_n = (1/n) sum delta_{x_i} and Q = sum_j w_j delta_{z_j}
    in the RKHS of the Gaussian kernel with bandwidth gamma:

        MMD^2 = E_{x,x'~P_n}[k(x,x')] - 2 E_{x~P_n, z~Q}[k(x,z)] + E_{z,z'~Q}[k(z,z')].
    """
    n = X_prefix.shape[0]
    if n == 0 or Z.shape[0] == 0:
        return float("nan")
    exx = term_ek_xx_batched(X_prefix, gamma, block=block_xx)
    exz = term_ek_xz_batched(X_prefix, Z, w, gamma, block=block_xz)
    ezz = term_ek_zz(Z, w, gamma)
    return max(exx - 2.0 * exz + ezz, 0.0)


def streamcore_buffer_to_weighted_set(
    streamer: StreamingCoreset,
) -> Tuple[np.ndarray, np.ndarray]:
    """Raw embedding matrix and weights (sum to 1) from STREAMCORE."""
    if not streamer.buffer_X:
        d = 0
        return np.empty((0, d), dtype=np.float64), np.empty(0, dtype=np.float64)
    Z = np.stack([np.asarray(v, dtype=np.float64) for v in streamer.buffer_X], axis=0)
    w = np.asarray(streamer.buffer_weights, dtype=np.float64)
    return Z, w


# =============================================================================
# 4. Evaluation schedule + plotting
# =============================================================================


def evaluation_steps(T: int, segment_lengths: Sequence[int], every: int = EVAL_EVERY) -> Set[int]:
    """Steps t in {1..T} where we record exact MMD (1-based stream index)."""
    boundary_ends = set(int(x) for x in np.cumsum(segment_lengths)[:-1])
    steps: Set[int] = set()
    for t in range(every, T + 1, every):
        steps.add(t)
    steps |= boundary_ends
    steps.add(T)
    return steps


def apply_paper_style() -> None:
    """Matplotlib style suitable for double-column ML proceedings."""
    for name in ("seaborn-v0_8-paper", "seaborn-paper", "seaborn-whitegrid"):
        try:
            plt.style.use(name)
            return
        except OSError:
            continue


def plot_rq1(
    steps: Sequence[int],
    mmd_sc: Sequence[float],
    mmd_res: Optional[Sequence[float]],
    mmd_fifo: Optional[Sequence[float]],
    transition_x: Sequence[int],
    out_path: str,
    *,
    show: bool = True,
) -> None:
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(5.5, 3.2))

    ax.plot(
        steps,
        mmd_sc,
        color="#1f77b4",
        linestyle="-",
        linewidth=2.0,
        label="STREAMCORE",
    )
    if mmd_res is not None:
        ax.plot(
            steps,
            mmd_res,
            color="#ff7f0e",
            linestyle=":",
            linewidth=1.5,
            label="Reservoir",
        )
    if mmd_fifo is not None:
        ax.plot(
            steps,
            mmd_fifo,
            color="#d62728",
            linestyle="--",
            linewidth=1.5,
            label="Sliding window",
        )

    for xv in transition_x:
        ax.axvline(x=xv, color="0.45", linestyle="--", linewidth=0.9, alpha=0.85)

    ax.set_xlabel(r"Stream step $t$")
    ax.set_ylabel(r"Exact RBF $\mathrm{MMD}^2$")
    ax.set_yscale("log")
    ax.legend(frameon=True, fontsize=8, loc="best")
    ax.grid(True, which="both", ls="-", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    print(f"Saved figure to {out_path}")

    if show and os.environ.get("RQ1_HEADLESS", "").lower() not in ("1", "true", "yes"):
        try:
            plt.show(block=True)
        except Exception as exc:
            print(f"(Could not display figure interactively: {exc})")

    plt.close(fig)


# =============================================================================
# 5. Main experiment driver
# =============================================================================


def main(baselines: Optional[BaselineConfig] = None, show_figure: Optional[bool] = None) -> None:
    if baselines is None:
        baselines = BaselineConfig()
    if show_figure is None:
        show_figure = os.environ.get("RQ1_HEADLESS", "").lower() not in ("1", "true", "yes")

    set_global_seeds(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = get_device()
    print(f"Device for ResNet18 feature extraction: {device}")
    print(
        f"Baselines: reservoir={baselines.use_reservoir}, "
        f"sliding_window={baselines.use_sliding_window}"
    )

    X_all, y_all = load_cifar10_train_embeddings(device)
    d_in = X_all.shape[1]
    assert d_in == 512, f"Expected 512-d ResNet18 features, got {d_in}"

    rbf_gamma = compute_median_heuristic_gamma(
        X_all, sample_size=MEDIAN_HEURISTIC_SAMPLE_SIZE, seed=SEED
    )
    print(
        f"Median-heuristic RBF γ = {rbf_gamma:.6g} "
        f"(subsample n={min(MEDIAN_HEURISTIC_SAMPLE_SIZE, len(X_all))})"
    )

    X_stream, y_stream, _, segment_lengths = build_split_cifar10_stream_order(
        X_all, y_all, TASK_CLASS_PAIRS, seed=SEED, max_per_class=MAX_PER_CLASS
    )
    T = X_stream.shape[0]
    expected_T = len(TASK_CLASS_PAIRS) * 2 * MAX_PER_CLASS
    assert T == expected_T, f"Stream length {T} != {expected_T} (check class counts vs MAX_PER_CLASS)"

    # Orthogonal RFF for STREAMCORE: same γ as exact RBF MMD (median heuristic)
    sampler = OrthogonalSampler(d_in=d_in, n_components=RFF_D, gamma=rbf_gamma)
    streamcore = StreamingCoreset(
        M=M,
        D=RFF_D,
        delta_drift_max=1.0,
        sampler=sampler,
        batch_size=1,
        K_iter=K_ITER,
        verbose=False,
    )

    rng = np.random.RandomState(SEED)
    reservoir = ReservoirBuffer(M, rng) if baselines.use_reservoir else None
    fifo = SlidingWindowBuffer(M) if baselines.use_sliding_window else None

    eval_at = evaluation_steps(T, segment_lengths, every=EVAL_EVERY)
    transition_x = [int(x) for x in np.cumsum(segment_lengths)[:-1]]

    rec_steps: List[int] = []
    rec_sc: List[float] = []
    rec_r: List[float] = []
    rec_f: List[float] = []

    pbar = tqdm(range(T), desc="RQ1 stream", ncols=88)
    for t0 in pbar:
        x_t = X_stream[t0]
        y_t = int(y_stream[t0])

        # STREAMCORE expects a batch array
        streamcore.process_batch(x_t[np.newaxis, :], np.array([y_t], dtype=np.int64), batch_idx=t0)

        if reservoir is not None:
            reservoir.observe(x_t)
        if fifo is not None:
            fifo.observe(x_t)

        step = t0 + 1  # 1-based stream step
        if step not in eval_at:
            continue

        # Uniform empirical measure over all points seen: contiguous prefix of the stream
        X_prefix = X_stream[:step]

        Zs, ws = streamcore_buffer_to_weighted_set(streamcore)

        m_sc = exact_rbf_mmd_squared(X_prefix, Zs, ws, rbf_gamma)
        rec_steps.append(step)
        rec_sc.append(m_sc)

        if reservoir is not None:
            Zr, wr = reservoir.get_matrix_weights()
            rec_r.append(exact_rbf_mmd_squared(X_prefix, Zr, wr, rbf_gamma))
        if fifo is not None:
            Zf, wf = fifo.get_matrix_weights()
            rec_f.append(exact_rbf_mmd_squared(X_prefix, Zf, wf, rbf_gamma))

    plot_rq1(
        rec_steps,
        rec_sc,
        rec_r if baselines.use_reservoir else None,
        rec_f if baselines.use_sliding_window else None,
        transition_x,
        FIGURE_PATH,
        show=show_figure,
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RQ1 drift immunity experiment (STREAMCORE vs baselines).")
    p.add_argument(
        "--no-reservoir",
        action="store_true",
        help="Omit reservoir baseline (STREAMCORE only vs other enabled baselines).",
    )
    p.add_argument(
        "--no-sliding-window",
        action="store_true",
        help="Omit FIFO sliding-window baseline.",
    )
    p.add_argument(
        "--no-show",
        action="store_true",
        help="Save PDF only; do not open an interactive figure window.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg = BaselineConfig(
        use_reservoir=not args.no_reservoir,
        use_sliding_window=not args.no_sliding_window,
    )
    main(baselines=cfg, show_figure=not args.no_show)