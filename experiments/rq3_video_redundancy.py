"""
RQ3 (video): Redundancy penalty and anomaly retention.

No raw MP4 processing is used. The stream is a pre-extracted feature matrix
X_stream (e.g., C3D / I3D / CLIP video embeddings).

If --embedding-path is missing, this script auto-generates a dummy surveillance
stream with strongly redundant regions and a short anomaly burst:
    - scene 1 (t=1..800):   N(mu_1, 0.01)
    - anomaly (t=801..850): N(mu_anom, 0.5)
    - scene 2 (t=851..2000): N(mu_2, 0.01)

Compares:
    - STREAMCORE (M=50)
    - Reservoir (M=50)
    - Sliding window (M=50)

Output:
    experiments/rq3_video_redundancy_output/neurips_fig3_video_barcode.pdf
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import deque
from typing import Deque, List, Sequence, Tuple

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from tqdm import tqdm

import matplotlib

if os.environ.get("RQ3_VIDEO_HEADLESS", "").lower() in ("1", "true", "yes"):
    matplotlib.use("Agg")
else:
    try:
        matplotlib.use("TkAgg")
    except Exception:
        matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from streamers.orf_streamer import OrthogonalSampler, StreamingCoreset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEED = 42
M = 50
RFF_D = 1024
K_ITER = 100
MEDIAN_HEURISTIC_SAMPLE_SIZE = 2000

DEFAULT_EMBED_PATH = os.path.join(PROJECT_ROOT, "data", "sample_video_features.npy")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "rq3_video_redundancy_output")
FIGURE_PATH = os.path.join(OUTPUT_DIR, "neurips_fig3_video_barcode.pdf")

T_DUMMY = 2000
D_DUMMY = 1024
SCENE1_END = 800
ANOM_START = 801
ANOM_END = 850


class ReservoirBuffer:
    """Classic reservoir sample storing stream indices only."""

    def __init__(self, M: int, rng: np.random.RandomState):
        self.M = M
        self.rng = rng
        self.buffer_idx: List[int] = []
        self.t_seen = 0

    def observe(self, stream_idx_0based: int) -> None:
        self.t_seen += 1
        if len(self.buffer_idx) < self.M:
            self.buffer_idx.append(int(stream_idx_0based))
            return
        j = int(self.rng.randint(0, self.t_seen))
        if j < self.M:
            self.buffer_idx[j] = int(stream_idx_0based)

    def get_indices_weights(self) -> Tuple[np.ndarray, np.ndarray]:
        if not self.buffer_idx:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
        idx = np.asarray(self.buffer_idx, dtype=np.int64)
        w = np.full(idx.shape[0], 1.0 / idx.shape[0], dtype=np.float64)
        return idx, w


class SlidingWindowBuffer:
    """FIFO buffer over the most recent M points."""

    def __init__(self, M: int):
        self.buf_idx: Deque[int] = deque(maxlen=M)

    def observe(self, stream_idx_0based: int) -> None:
        self.buf_idx.append(int(stream_idx_0based))

    def get_indices_weights(self) -> Tuple[np.ndarray, np.ndarray]:
        if len(self.buf_idx) == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
        idx = np.asarray(list(self.buf_idx), dtype=np.int64)
        w = np.full(idx.shape[0], 1.0 / idx.shape[0], dtype=np.float64)
        return idx, w


def set_global_seeds(seed: int) -> None:
    np.random.seed(seed)


def compute_median_heuristic_gamma(
    X: np.ndarray,
    sample_size: int = MEDIAN_HEURISTIC_SAMPLE_SIZE,
    seed: int = SEED,
) -> float:
    n = X.shape[0]
    rng = np.random.RandomState(seed)
    if n > sample_size:
        idx = rng.choice(n, size=sample_size, replace=False)
        X_sub = X[idx]
    else:
        X_sub = X

    D_sq = euclidean_distances(X_sub, squared=True)
    tri = np.triu_indices_from(D_sq, k=1)
    med = float(np.median(D_sq[tri]))
    if med <= 0:
        raise ValueError("Median squared distance is non-positive; check embeddings.")
    return 1.0 / (2.0 * med)


def apply_paper_style() -> None:
    for name in ("seaborn-v0_8-paper", "seaborn-paper", "seaborn-whitegrid"):
        try:
            plt.style.use(name)
            return
        except OSError:
            continue


def generate_dummy_video_matrix(path: str, seed: int = SEED) -> np.ndarray:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rng = np.random.RandomState(seed)

    # Keep means far apart so anomaly burst is geometrically distinct.
    mu_1 = np.zeros(D_DUMMY, dtype=np.float64)
    mu_anom = np.full(D_DUMMY, 3.0, dtype=np.float64)
    mu_2 = np.full(D_DUMMY, -1.5, dtype=np.float64)

    n1 = SCENE1_END  # t=1..800
    n_anom = ANOM_END - ANOM_START + 1  # t=801..850
    n2 = T_DUMMY - n1 - n_anom  # t=851..2000

    X1 = rng.normal(loc=mu_1, scale=np.sqrt(0.01), size=(n1, D_DUMMY))
    Xa = rng.normal(loc=mu_anom, scale=np.sqrt(0.5), size=(n_anom, D_DUMMY))
    X2 = rng.normal(loc=mu_2, scale=np.sqrt(0.01), size=(n2, D_DUMMY))
    X = np.vstack([X1, Xa, X2]).astype(np.float64, copy=False)
    if X.shape != (T_DUMMY, D_DUMMY):
        raise RuntimeError(f"Unexpected dummy shape {X.shape}, expected {(T_DUMMY, D_DUMMY)}")

    np.save(path, X)
    print(f"Generated dummy video feature matrix at {path} with shape={X.shape}")
    return X


def load_video_feature_stream(path: str, seed: int = SEED) -> np.ndarray:
    if not os.path.exists(path):
        print(f"Embedding file not found at {path}; generating dummy matrix.")
        return generate_dummy_video_matrix(path, seed=seed)

    X = np.load(path)
    if X.ndim != 2:
        raise ValueError(f"Expected 2D embedding matrix, got shape={X.shape}")
    X = np.asarray(X, dtype=np.float64)
    print(f"Loaded embedding matrix from {path} with shape={X.shape}")
    return X


def _rbf_kernel_block(X: np.ndarray, Y: np.ndarray, gamma: float) -> np.ndarray:
    x2 = np.sum(X * X, axis=1, keepdims=True)
    y2 = np.sum(Y * Y, axis=1, keepdims=True).T
    dist_sq = np.maximum(x2 + y2 - 2.0 * (X @ Y.T), 0.0)
    return np.exp(-gamma * dist_sq).astype(np.float64, copy=False)


def precompute_kernel_and_prefix_terms(
    X_stream: np.ndarray,
    gamma: float,
    block: int = 256,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        K: full stream kernel matrix
        exx_prefix[t-1] = E_{x,x'~prefix_t}[k(x,x')]
        col_prefix[t-1, j] = sum_{i<=t} k(x_i, x_j)
    """
    T = X_stream.shape[0]
    K = np.empty((T, T), dtype=np.float64)
    for i0 in tqdm(range(0, T, block), desc="Kernel blocks", ncols=90):
        i1 = min(i0 + block, T)
        Xi = X_stream[i0:i1]
        for j0 in range(0, T, block):
            j1 = min(j0 + block, T)
            Xj = X_stream[j0:j1]
            K[i0:i1, j0:j1] = _rbf_kernel_block(Xi, Xj, gamma)

    S = np.cumsum(np.cumsum(K, axis=0), axis=1)
    denom = np.arange(1, T + 1, dtype=np.float64) ** 2
    exx_prefix = np.diag(S) / denom
    col_prefix = np.cumsum(K, axis=0)
    return K, exx_prefix, col_prefix


def exact_mmd2_from_precomputed(
    *,
    step: int,
    K: np.ndarray,
    exx_prefix: np.ndarray,
    col_prefix: np.ndarray,
    buf_idx: np.ndarray,
    buf_w: np.ndarray,
) -> float:
    if buf_idx.size == 0:
        return float("nan")
    t = int(step)
    exx = float(exx_prefix[t - 1])
    prefix_col_sums = col_prefix[t - 1, buf_idx]
    exz = float(np.dot(prefix_col_sums, buf_w) / t)
    Kzz = K[np.ix_(buf_idx, buf_idx)]
    ezz = float(buf_w @ Kzz @ buf_w)
    return max(exx - 2.0 * exz + ezz, 0.0)


def streamcore_indices_weights(streamer: StreamingCoreset) -> Tuple[np.ndarray, np.ndarray]:
    if not streamer.buffer_provenance:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    # batch_size=1 in this script, so stream index == batch_idx.
    idx = np.asarray([int(p[0]) for p in streamer.buffer_provenance], dtype=np.int64)
    w = np.asarray(streamer.buffer_weights, dtype=np.float64)
    return idx, w


def count_in_anomaly_zone(indices_0based: np.ndarray) -> int:
    if indices_0based.size == 0:
        return 0
    idx_1based = indices_0based + 1
    return int(np.sum((idx_1based >= ANOM_START) & (idx_1based <= ANOM_END)))


def plot_video_barcode(
    *,
    steps: np.ndarray,
    mmd_sc: np.ndarray,
    mmd_res: np.ndarray,
    mmd_sld: np.ndarray,
    final_idx_sc: np.ndarray,
    final_idx_res: np.ndarray,
    final_idx_sld: np.ndarray,
    out_path: str,
    show: bool = True,
) -> None:
    apply_paper_style()
    fig, axes = plt.subplots(
        2, 1, figsize=(7.0, 6.4), sharex=True, gridspec_kw={"height_ratios": [2.0, 1.2]}
    )

    ax = axes[0]
    ax.plot(steps, mmd_sc, color="#1f77b4", linewidth=1.9, label="STREAMCORE")
    ax.plot(steps, mmd_res, color="#ff7f0e", linestyle=":", linewidth=1.5, label="Reservoir")
    ax.plot(steps, mmd_sld, color="#d62728", linestyle="--", linewidth=1.5, label="Sliding window")
    ax.set_yscale("log")
    ax.set_ylabel(r"Exact RBF $\mathrm{MMD}^2$")
    ax.grid(True, which="both", ls="-", alpha=0.25)
    ax.legend(loc="best", fontsize=8, frameon=True)
    ax.axvspan(ANOM_START, ANOM_END, color="#ffcccc", alpha=0.35, zorder=0)

    ax = axes[1]
    y_map = {"Reservoir": 2, "Sliding window": 1, "STREAMCORE": 0}
    rows = [
        ("Reservoir", final_idx_res, "#ff7f0e"),
        ("Sliding window", final_idx_sld, "#d62728"),
        ("STREAMCORE", final_idx_sc, "#1f77b4"),
    ]
    for name, idx0, color in rows:
        y = y_map[name]
        x = np.sort(idx0 + 1)  # convert to 1-based stream step
        if x.size > 0:
            ax.vlines(x, y - 0.32, y + 0.32, color=color, linewidth=0.9, alpha=0.95)

    ax.axvspan(ANOM_START, ANOM_END, color="#ffcccc", alpha=0.35, zorder=0)
    ax.set_yticks([2, 1, 0])
    ax.set_yticklabels(["Reservoir", "Sliding window", "STREAMCORE"])
    ax.set_ylim(-0.7, 2.7)
    ax.set_xlabel(r"Stream step $t$")
    ax.set_ylabel("Final buffer")
    ax.grid(True, axis="x", ls="-", alpha=0.18)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    print(f"Saved figure to {out_path}")

    if show and os.environ.get("RQ3_VIDEO_HEADLESS", "").lower() not in ("1", "true", "yes"):
        try:
            plt.show(block=True)
        except Exception as exc:
            print(f"(Could not display figure interactively: {exc})")
    plt.close(fig)


def run_experiment(
    X_stream: np.ndarray,
    M: int = M,
    seed: int = SEED,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    T, d_in = X_stream.shape
    steps = np.arange(1, T + 1, dtype=np.int64)

    gamma = compute_median_heuristic_gamma(X_stream, sample_size=MEDIAN_HEURISTIC_SAMPLE_SIZE, seed=seed)
    print(f"Median-heuristic RBF gamma = {gamma:.6g}")

    print("Precomputing stream kernel matrix and prefix terms...")
    K, exx_prefix, col_prefix = precompute_kernel_and_prefix_terms(X_stream, gamma=gamma, block=256)

    sampler = OrthogonalSampler(d_in=d_in, n_components=RFF_D, gamma=gamma)
    sc = StreamingCoreset(M=M, D=RFF_D, sampler=sampler, batch_size=1, K_iter=K_ITER, verbose=False)

    rng = np.random.RandomState(seed)
    res = ReservoirBuffer(M=M, rng=rng)
    sld = SlidingWindowBuffer(M=M)

    mmd_sc = np.empty(T, dtype=np.float64)
    mmd_res = np.empty(T, dtype=np.float64)
    mmd_sld = np.empty(T, dtype=np.float64)

    pbar = tqdm(range(T), desc="RQ3 video stream", ncols=90)
    for t0 in pbar:
        x_t = X_stream[t0]
        sc.process_batch(x_t[np.newaxis, :], np.array([0], dtype=np.int64), batch_idx=t0)
        res.observe(t0)
        sld.observe(t0)

        idx_sc, w_sc = streamcore_indices_weights(sc)
        idx_res, w_res = res.get_indices_weights()
        idx_sld, w_sld = sld.get_indices_weights()

        step = t0 + 1
        mmd_sc[t0] = exact_mmd2_from_precomputed(
            step=step,
            K=K,
            exx_prefix=exx_prefix,
            col_prefix=col_prefix,
            buf_idx=idx_sc,
            buf_w=w_sc,
        )
        mmd_res[t0] = exact_mmd2_from_precomputed(
            step=step,
            K=K,
            exx_prefix=exx_prefix,
            col_prefix=col_prefix,
            buf_idx=idx_res,
            buf_w=w_res,
        )
        mmd_sld[t0] = exact_mmd2_from_precomputed(
            step=step,
            K=K,
            exx_prefix=exx_prefix,
            col_prefix=col_prefix,
            buf_idx=idx_sld,
            buf_w=w_sld,
        )

    final_idx_sc, _ = streamcore_indices_weights(sc)
    final_idx_res, _ = res.get_indices_weights()
    final_idx_sld, _ = sld.get_indices_weights()
    return steps, mmd_sc, mmd_res, mmd_sld, final_idx_sc, final_idx_res, final_idx_sld


def main(embedding_path: str, M_value: int = M, seed: int = SEED, show_figure: bool = True) -> None:
    set_global_seeds(seed)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"RQ3 video redundancy: M={M_value}, seed={seed}, embedding_path={embedding_path}")

    X_stream = load_video_feature_stream(embedding_path, seed=seed)
    if X_stream.shape[0] < ANOM_END:
        raise ValueError(
            f"Need at least {ANOM_END} frames for the requested anomaly region, got T={X_stream.shape[0]}"
        )
    print(f"Stream shape: T={X_stream.shape[0]}, d={X_stream.shape[1]}")

    steps, mmd_sc, mmd_res, mmd_sld, final_idx_sc, final_idx_res, final_idx_sld = run_experiment(
        X_stream, M=M_value, seed=seed
    )

    plot_video_barcode(
        steps=steps,
        mmd_sc=mmd_sc,
        mmd_res=mmd_res,
        mmd_sld=mmd_sld,
        final_idx_sc=final_idx_sc,
        final_idx_res=final_idx_res,
        final_idx_sld=final_idx_sld,
        out_path=FIGURE_PATH,
        show=show_figure,
    )

    print("\n" + "=" * 80)
    print("RQ3 video redundancy summary (final buffer anomaly retention)")
    print("=" * 80)
    print(f"Reservoir anomaly frames kept: {count_in_anomaly_zone(final_idx_res)} / {len(final_idx_res)}")
    print(f"Sliding window anomaly frames kept: {count_in_anomaly_zone(final_idx_sld)} / {len(final_idx_sld)}")
    print(f"STREAMCORE anomaly frames kept: {count_in_anomaly_zone(final_idx_sc)} / {len(final_idx_sc)}")
    print(f"Output figure: {FIGURE_PATH}")
    print("=" * 80 + "\n")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RQ3 video redundancy barcode experiment.")
    p.add_argument(
        "--embedding-path",
        type=str,
        default=DEFAULT_EMBED_PATH,
        help="Path to .npy video feature matrix (shape: T x d).",
    )
    p.add_argument("--M", type=int, default=M, help="Buffer size for all methods (default: 50).")
    p.add_argument("--seed", type=int, default=SEED, help="Random seed (default: 42).")
    p.add_argument(
        "--no-show",
        action="store_true",
        help="Save PDF only; do not open interactive figure window.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.M <= 0:
        raise ValueError("--M must be positive.")

    main(
        embedding_path=str(args.embedding_path),
        M_value=int(args.M),
        seed=int(args.seed),
        show_figure=not bool(args.no_show),
    )
