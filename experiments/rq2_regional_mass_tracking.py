"""
RQ2: Regional aggregate-weight tracking / coverage stability.

Research question
-----------------
Does STREAMCORE preserve mass at the level of semantic/geometric regions (here:
Split-CIFAR10 class regions), even when individual prototype weights are small?

Why this granularity matters
----------------------------
The key theoretical gap is that per-atom minimum weight is too weak to guarantee
representation quality under arbitrary stream order. A more plausible invariant is
regional aggregate mass: for region/class k at stream step t, compare:

    prefix_mass pi_k(t) = (1/t) * sum_{s<=t} 1[y_s = k]
    buffer_mass W_k(t)  = sum_{i: buffer_label_i = k} w_i(t)

If W_k(t) tracks pi_k(t), then low individual weights need not imply poor global
approximation: the relevant object is aggregate region mass.

Connection to the regional-lemma direction
------------------------------------------
This experiment does not prove the target bound, but it tests its qualitative
mechanism:
1) region-level mass tracking,
2) coverage failures concentrated in tiny-prefix-mass classes,
3) regional mismatch correlating with exact approximation error (RBF MMD^2).

Setup mirrors RQ1/RQ3:
- cached CIFAR-10 train ResNet18 embeddings (512-d)
- Split-CIFAR10 class-incremental stream
- M=100, ORF D=1024, K_iter=100
- same median-heuristic RBF gamma
- methods: STREAMCORE, reservoir, FIFO sliding window

Outputs under experiments/rq2_regional_mass_tracking_output/:
- rq2_regional_mass_tracking_per_class.csv
- rq2_regional_mass_tracking_summary.csv
- rq2_regional_mass_tracking.pdf
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Sequence, Set, Tuple

import joblib
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from tqdm import tqdm

import matplotlib

if os.environ.get("RQ2_REGIONAL_HEADLESS", "").lower() in ("1", "true", "yes"):
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

import torch
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

from dataloaders import TransformDataset, extract_resnet18_features
from streamers.orf_streamer import OrthogonalSampler, StreamingCoreset

# ---------------------------------------------------------------------------
# Constants (aligned with RQ1/RQ3 defaults)
# ---------------------------------------------------------------------------
SEED = 42
M = 100
RFF_D = 1024
K_ITER = 100
MAX_PER_CLASS = 1000
MEDIAN_HEURISTIC_SAMPLE_SIZE = 2000
EVAL_EVERY = 1000
NUM_CLASSES = 10
SIGNIFICANT_MASS_TAU = 0.05

FEATURE_CACHE_PATH = os.path.join(PROJECT_ROOT, "feature_cache", "cifar10_train_resnet18_rq1.pkl")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "rq2_regional_mass_tracking_output")
PER_CLASS_CSV = os.path.join(OUTPUT_DIR, "rq2_regional_mass_tracking_per_class.csv")
SUMMARY_CSV = os.path.join(OUTPUT_DIR, "rq2_regional_mass_tracking_summary.csv")
FIGURE_PATH = os.path.join(OUTPUT_DIR, "rq2_regional_mass_tracking.pdf")

TASK_CLASS_PAIRS: List[Tuple[int, int]] = [
    (0, 1),
    (2, 3),
    (4, 5),
    (6, 7),
    (8, 9),
]


@dataclass
class MethodSnapshot:
    name: str
    Z: np.ndarray
    yb: np.ndarray
    w: np.ndarray


class ReservoirBufferLabeled:
    """Classic reservoir sample with labels retained for diagnostics."""

    def __init__(self, M: int, rng: np.random.RandomState):
        self.M = M
        self.rng = rng
        self.buffer_x: List[np.ndarray] = []
        self.buffer_y: List[int] = []
        self.t_seen = 0

    def observe(self, x: np.ndarray, y: int) -> None:
        self.t_seen += 1
        if len(self.buffer_x) < self.M:
            self.buffer_x.append(x.copy())
            self.buffer_y.append(int(y))
            return
        j = self.rng.randint(0, self.t_seen)
        if j < self.M:
            self.buffer_x[j] = x.copy()
            self.buffer_y[j] = int(y)

    def get_matrix_labels_weights(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(self.buffer_x) == 0:
            return (
                np.empty((0, 0), dtype=np.float64),
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.float64),
            )
        Z = np.stack(self.buffer_x, axis=0).astype(np.float64, copy=False)
        yb = np.asarray(self.buffer_y, dtype=np.int64)
        m = Z.shape[0]
        w = np.full(m, 1.0 / m, dtype=np.float64)
        return Z, yb, w


class SlidingWindowBufferLabeled:
    """FIFO sliding window with label retention for diagnostics."""

    def __init__(self, M: int):
        self.buf_x: Deque[np.ndarray] = deque(maxlen=M)
        self.buf_y: Deque[int] = deque(maxlen=M)

    def observe(self, x: np.ndarray, y: int) -> None:
        self.buf_x.append(x.copy())
        self.buf_y.append(int(y))

    def get_matrix_labels_weights(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(self.buf_x) == 0:
            return (
                np.empty((0, 0), dtype=np.float64),
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.float64),
            )
        Z = np.stack(list(self.buf_x), axis=0).astype(np.float64, copy=False)
        yb = np.asarray(list(self.buf_y), dtype=np.int64)
        m = Z.shape[0]
        w = np.full(m, 1.0 / m, dtype=np.float64)
        return Z, yb, w


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


def load_cifar10_train_embeddings(
    device: torch.device,
    cache_path: str = FEATURE_CACHE_PATH,
    force_recompute: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
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
    tfm = transforms.Compose(
        [
            transforms.Resize(224, InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    dataset = TransformDataset(train_ds, tfm)
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
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    rng = np.random.RandomState(seed)
    blocks: List[np.ndarray] = []

    for c0, c1 in pairs:
        idx_c0 = np.where(y == c0)[0]
        idx_c1 = np.where(y == c1)[0]
        idx_c0 = rng.choice(idx_c0, size=min(len(idx_c0), max_per_class), replace=False)
        idx_c1 = rng.choice(idx_c1, size=min(len(idx_c1), max_per_class), replace=False)
        idx = np.concatenate([idx_c0, idx_c1])
        keys = np.stack([y[idx], rng.rand(len(idx))], axis=1)
        sort_order = np.lexsort((keys[:, 1], keys[:, 0]))
        blocks.append(idx[sort_order])

    stream_idx = np.concatenate(blocks, axis=0)
    segment_lengths = [len(b) for b in blocks]
    return X[stream_idx], y[stream_idx], segment_lengths


def evaluation_steps(T: int, segment_lengths: Sequence[int], every: int = EVAL_EVERY) -> Set[int]:
    boundary_ends = set(int(x) for x in np.cumsum(segment_lengths)[:-1])
    steps: Set[int] = set()
    for t in range(every, T + 1, every):
        steps.add(t)
    steps |= boundary_ends
    steps.add(T)
    return steps


def task_id_from_step(step: int, segment_lengths: Sequence[int]) -> int:
    boundaries = np.cumsum(np.asarray(segment_lengths, dtype=np.int64))
    return int(np.searchsorted(boundaries, step, side="left") + 1)


def _rbf_kernel_block(X: np.ndarray, Y: np.ndarray, gamma: float) -> np.ndarray:
    x2 = np.sum(X * X, axis=1, keepdims=True)
    y2 = np.sum(Y * Y, axis=1, keepdims=True).T
    dist_sq = np.maximum(x2 + y2 - 2.0 * (X @ Y.T), 0.0)
    return np.exp(-gamma * dist_sq).astype(np.float64, copy=False)


def term_ek_xx_batched(X: np.ndarray, gamma: float, block: int = 256) -> float:
    n = X.shape[0]
    if n == 0:
        return 0.0
    total = 0.0
    for i0 in range(0, n, block):
        i1 = min(i0 + block, n)
        Xi = X[i0:i1]
        for j0 in range(0, n, block):
            j1 = min(j0 + block, n)
            Xj = X[j0:j1]
            total += float(_rbf_kernel_block(Xi, Xj, gamma).sum())
    return total / float(n * n)


def term_ek_xz_batched(X: np.ndarray, Z: np.ndarray, w: np.ndarray, gamma: float, block: int = 512) -> float:
    n = X.shape[0]
    m = Z.shape[0]
    if n == 0 or m == 0:
        return 0.0
    acc = 0.0
    for i0 in range(0, n, block):
        i1 = min(i0 + block, n)
        Xi = X[i0:i1]
        K = _rbf_kernel_block(Xi, Z, gamma)
        acc += float((K @ w).sum())
    return acc / float(n)


def term_ek_zz(Z: np.ndarray, w: np.ndarray, gamma: float) -> float:
    if Z.shape[0] == 0:
        return 0.0
    K = _rbf_kernel_block(Z, Z, gamma)
    return float(w @ K @ w)


def exact_rbf_mmd_squared(
    X_prefix: np.ndarray,
    Z: np.ndarray,
    w: np.ndarray,
    gamma: float,
) -> float:
    if X_prefix.shape[0] == 0 or Z.shape[0] == 0:
        return float("nan")
    exx = term_ek_xx_batched(X_prefix, gamma=gamma, block=256)
    exz = term_ek_xz_batched(X_prefix, Z, w, gamma=gamma, block=512)
    ezz = term_ek_zz(Z, w, gamma=gamma)
    return max(exx - 2.0 * exz + ezz, 0.0)


def streamcore_snapshot(sc: StreamingCoreset) -> MethodSnapshot:
    if len(sc.buffer_X) == 0:
        return MethodSnapshot(
            name="streamcore",
            Z=np.empty((0, 0), dtype=np.float64),
            yb=np.empty(0, dtype=np.int64),
            w=np.empty(0, dtype=np.float64),
        )
    Z = np.stack([np.asarray(x, dtype=np.float64) for x in sc.buffer_X], axis=0)
    yb = np.asarray(sc.buffer_y, dtype=np.int64)
    w = np.asarray(sc.buffer_weights, dtype=np.float64)
    return MethodSnapshot(name="streamcore", Z=Z, yb=yb, w=w)


def per_class_mass_stats(y_prefix_counts: np.ndarray, t: int, yb: np.ndarray, w: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    prefix_mass = y_prefix_counts.astype(np.float64) / float(t)
    counts = np.bincount(yb, minlength=NUM_CLASSES).astype(np.int64) if yb.size else np.zeros(NUM_CLASSES, dtype=np.int64)
    agg = np.zeros(NUM_CLASSES, dtype=np.float64)
    if yb.size:
        for k in range(NUM_CLASSES):
            mask = yb == k
            if np.any(mask):
                agg[k] = float(w[mask].sum())
    return prefix_mass, counts, agg


def safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return float("nan")
    sx = float(np.std(x))
    sy = float(np.std(y))
    if sx <= 1e-15 or sy <= 1e-15:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def rankdata_average_ties(x: np.ndarray) -> np.ndarray:
    idx = np.argsort(x, kind="mergesort")
    sorted_x = x[idx]
    n = x.size
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        avg_rank = 0.5 * (i + j - 1) + 1.0
        ranks[idx[i:j]] = avg_rank
        i = j
    return ranks


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return float("nan")
    rx = rankdata_average_ties(x)
    ry = rankdata_average_ties(y)
    return safe_pearson(rx, ry)


def write_csv(path: str, rows: List[Dict[str, object]], columns: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Saved CSV to {path}")


def apply_paper_style() -> None:
    for name in ("seaborn-v0_8-paper", "seaborn-paper", "seaborn-whitegrid"):
        try:
            plt.style.use(name)
            return
        except OSError:
            continue


def plot_three_panel(
    per_class_rows: List[Dict[str, object]],
    summary_rows: List[Dict[str, object]],
    transition_x: Sequence[int],
    out_path: str,
    show: bool = True,
) -> None:
    apply_paper_style()
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.0))
    ax_a, ax_b, ax_c = axes

    # Panel A: STREAMCORE class-wise pi_k(t) dashed vs W_k(t) solid.
    sc_rows = [r for r in per_class_rows if r["method"] == "streamcore"]
    class_colors = [plt.cm.tab10(i) for i in range(NUM_CLASSES)]
    for k in range(NUM_CLASSES):
        rk = [r for r in sc_rows if int(r["class_id"]) == k]
        steps = np.array([int(r["step"]) for r in rk], dtype=np.int64)
        pi = np.array([float(r["prefix_mass"]) for r in rk], dtype=np.float64)
        wk = np.array([float(r["aggregate_weight"]) for r in rk], dtype=np.float64)
        color = class_colors[k]
        ax_a.plot(steps, pi, linestyle="--", color=color, linewidth=1.0, alpha=0.7)
        ax_a.plot(steps, wk, linestyle="-", color=color, linewidth=1.6, label=f"class {k}")
    ax_a.set_title("A) STREAMCORE class-mass tracking")
    ax_a.set_xlabel(r"Stream step $t$")
    ax_a.set_ylabel(r"Mass ($\pi_k(t)$ dashed, $W_k(t)$ solid)")
    ax_a.set_ylim(0.0, 1.02)
    ax_a.grid(True, ls="-", alpha=0.25)
    ax_a.legend(ncol=2, fontsize=7, frameon=True, loc="upper right")

    # Panel B: L1 mass gap over time for all methods.
    method_specs = {
        "streamcore": ("STREAMCORE", "#1f77b4", "-"),
        "reservoir": ("Reservoir", "#ff7f0e", ":"),
        "sliding_window": ("Sliding window", "#d62728", "--"),
    }
    for method, (label, color, ls) in method_specs.items():
        mr = [r for r in summary_rows if r["method"] == method]
        steps = np.array([int(r["step"]) for r in mr], dtype=np.int64)
        l1 = np.array([float(r["l1_mass_gap"]) for r in mr], dtype=np.float64)
        ax_b.plot(steps, l1, color=color, linestyle=ls, linewidth=1.7, label=label)
    for xv in transition_x:
        ax_b.axvline(x=xv, color="0.45", linestyle="--", linewidth=0.85, alpha=0.75)
    ax_b.set_title("B) Regional mismatch over time")
    ax_b.set_xlabel(r"Stream step $t$")
    ax_b.set_ylabel(r"$D_t^{(1)}=\sum_k |W_k-\pi_k|$")
    ax_b.grid(True, ls="-", alpha=0.25)
    ax_b.legend(fontsize=8, frameon=True, loc="best")

    # Panel C: scatter l1_mass_gap vs exact mmd2 with per-method Pearson.
    for method, (label, color, marker_ls) in method_specs.items():
        mr = [r for r in summary_rows if r["method"] == method]
        l1 = np.array([float(r["l1_mass_gap"]) for r in mr], dtype=np.float64)
        mmd2 = np.array([float(r["mmd2"]) for r in mr], dtype=np.float64)
        pearson = safe_pearson(l1, mmd2)
        legend_label = f"{label} (r={pearson:.2f})" if np.isfinite(pearson) else label
        marker = "o" if method == "streamcore" else ("s" if method == "reservoir" else "^")
        ax_c.scatter(l1, mmd2, color=color, marker=marker, alpha=0.75, s=20, label=legend_label)
    ax_c.set_title("C) Regional mismatch vs exact error")
    ax_c.set_xlabel(r"$D_t^{(1)}$")
    ax_c.set_ylabel(r"Exact RBF $\mathrm{MMD}^2$")
    ax_c.grid(True, ls="-", alpha=0.25)
    ax_c.legend(fontsize=7.5, frameon=True, loc="best")

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    print(f"Saved figure to {out_path}")

    if show and os.environ.get("RQ2_REGIONAL_HEADLESS", "").lower() not in ("1", "true", "yes"):
        try:
            plt.show(block=True)
        except Exception as exc:
            print(f"(Could not display figure interactively: {exc})")
    plt.close(fig)


def print_summary(summary_rows: List[Dict[str, object]]) -> None:
    methods = ["streamcore", "reservoir", "sliding_window"]
    print("\n" + "=" * 84)
    print("RQ2 regional aggregate-weight tracking summary")
    print("=" * 84)
    for method in methods:
        mr = [r for r in summary_rows if r["method"] == method]
        l1 = np.array([float(r["l1_mass_gap"]) for r in mr], dtype=np.float64)
        linf = np.array([float(r["linf_mass_gap"]) for r in mr], dtype=np.float64)
        unc = np.array([float(r["num_uncovered_significant_classes"]) for r in mr], dtype=np.float64)
        mmd2 = np.array([float(r["mmd2"]) for r in mr], dtype=np.float64)
        frac_unc = float(np.mean(unc > 0.0))
        pearson = safe_pearson(l1, mmd2)
        spearman = safe_spearman(l1, mmd2)
        print(f"[{method}]")
        print(f"  mean/max l1_mass_gap:   {l1.mean():.6e} / {l1.max():.6e}")
        print(f"  mean/max linf_mass_gap: {linf.mean():.6e} / {linf.max():.6e}")
        print(f"  frac eval with uncovered significant classes: {frac_unc:.4f}")
        if np.isfinite(pearson):
            print(f"  Pearson(l1_gap, mmd2):  {pearson:.6f}")
        else:
            print("  Pearson(l1_gap, mmd2):  nan")
        if np.isfinite(spearman):
            print(f"  Spearman(l1_gap, mmd2): {spearman:.6f}")
        else:
            print("  Spearman(l1_gap, mmd2): nan")
    print("=" * 84 + "\n")


def main(
    show_figure: Optional[bool] = None,
    max_per_class: int = MAX_PER_CLASS,
    eval_every: int = EVAL_EVERY,
) -> None:
    if show_figure is None:
        show_figure = os.environ.get("RQ2_REGIONAL_HEADLESS", "").lower() not in ("1", "true", "yes")

    set_global_seeds(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = get_device()
    print(f"Device for ResNet18 feature extraction: {device}")

    X_all, y_all = load_cifar10_train_embeddings(device)
    d_in = X_all.shape[1]
    assert d_in == 512, f"Expected 512-d ResNet18 features, got {d_in}"

    rbf_gamma = compute_median_heuristic_gamma(X_all, sample_size=MEDIAN_HEURISTIC_SAMPLE_SIZE, seed=SEED)
    print(f"Median-heuristic RBF gamma = {rbf_gamma:.6g}")

    X_stream, y_stream, segment_lengths = build_split_cifar10_stream_order(
        X_all, y_all, TASK_CLASS_PAIRS, seed=SEED, max_per_class=max_per_class
    )
    T = X_stream.shape[0]
    expected_T = len(TASK_CLASS_PAIRS) * 2 * max_per_class
    assert T == expected_T, f"Stream length {T} != {expected_T}"

    sampler = OrthogonalSampler(d_in=d_in, n_components=RFF_D, gamma=rbf_gamma)
    streamcore = StreamingCoreset(
        M=M,
        D=RFF_D,
        sampler=sampler,
        batch_size=1,
        K_iter=K_ITER,
        verbose=False,
    )
    reservoir = ReservoirBufferLabeled(M=M, rng=np.random.RandomState(SEED))
    sliding_window = SlidingWindowBufferLabeled(M=M)

    eval_at = evaluation_steps(T, segment_lengths, every=eval_every)
    transition_x = [int(x) for x in np.cumsum(segment_lengths)[:-1]]
    print(
        f"Stream length T={T}, evaluation points={len(eval_at)}, M={M}, "
        f"max_per_class={max_per_class}, eval_every={eval_every}"
    )

    y_prefix_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    per_class_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []

    pbar = tqdm(range(T), desc="RQ2 regional stream", ncols=90)
    for t0 in pbar:
        x_t = X_stream[t0]
        y_t = int(y_stream[t0])
        y_prefix_counts[y_t] += 1

        streamcore.process_batch(x_t[np.newaxis, :], np.array([y_t], dtype=np.int64), batch_idx=t0)
        reservoir.observe(x_t, y_t)
        sliding_window.observe(x_t, y_t)

        step = t0 + 1
        if step not in eval_at:
            continue

        task_id = task_id_from_step(step, segment_lengths)
        X_prefix = X_stream[:step]

        snapshots: List[MethodSnapshot] = [
            streamcore_snapshot(streamcore),
            MethodSnapshot("reservoir", *reservoir.get_matrix_labels_weights()),
            MethodSnapshot("sliding_window", *sliding_window.get_matrix_labels_weights()),
        ]

        for snap in snapshots:
            prefix_mass, buffer_counts, aggregate_weights = per_class_mass_stats(
                y_prefix_counts=y_prefix_counts,
                t=step,
                yb=snap.yb,
                w=snap.w,
            )
            abs_gap = np.abs(aggregate_weights - prefix_mass)
            l1_gap = float(abs_gap.sum())
            linf_gap = float(abs_gap.max())
            significant = prefix_mass >= SIGNIFICANT_MASS_TAU
            num_uncovered_sig = int(np.sum((buffer_counts == 0) & significant))
            mmd2 = exact_rbf_mmd_squared(X_prefix, snap.Z, snap.w, gamma=rbf_gamma)

            summary_rows.append(
                {
                    "step": int(step),
                    "task_id": int(task_id),
                    "method": snap.name,
                    "l1_mass_gap": l1_gap,
                    "linf_mass_gap": linf_gap,
                    "num_uncovered_significant_classes": int(num_uncovered_sig),
                    "mmd2": float(mmd2),
                }
            )

            for k in range(NUM_CLASSES):
                per_class_rows.append(
                    {
                        "step": int(step),
                        "task_id": int(task_id),
                        "method": snap.name,
                        "class_id": int(k),
                        "prefix_mass": float(prefix_mass[k]),
                        "buffer_count": int(buffer_counts[k]),
                        "aggregate_weight": float(aggregate_weights[k]),
                        "abs_mass_gap": float(abs_gap[k]),
                        "covered": int(buffer_counts[k] > 0),
                    }
                )

    write_csv(
        PER_CLASS_CSV,
        per_class_rows,
        columns=[
            "step",
            "task_id",
            "method",
            "class_id",
            "prefix_mass",
            "buffer_count",
            "aggregate_weight",
            "abs_mass_gap",
            "covered",
        ],
    )
    write_csv(
        SUMMARY_CSV,
        summary_rows,
        columns=[
            "step",
            "task_id",
            "method",
            "l1_mass_gap",
            "linf_mass_gap",
            "num_uncovered_significant_classes",
            "mmd2",
        ],
    )
    plot_three_panel(per_class_rows, summary_rows, transition_x, FIGURE_PATH, show=show_figure)
    print_summary(summary_rows)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RQ2 regional aggregate-weight tracking on Split-CIFAR10.")
    p.add_argument(
        "--no-show",
        action="store_true",
        help="Save figure only; do not display interactive window.",
    )
    p.add_argument(
        "--max-per-class",
        type=int,
        default=MAX_PER_CLASS,
        help=f"Points per class in each task block (default: {MAX_PER_CLASS}).",
    )
    p.add_argument(
        "--eval-every",
        type=int,
        default=EVAL_EVERY,
        help=f"Evaluate every N points plus task boundaries (default: {EVAL_EVERY}).",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(
        show_figure=not args.no_show,
        max_per_class=int(args.max_per_class),
        eval_every=int(args.eval_every),
    )
