"""
RQ2: Downstream utility — does a STREAMCORE buffer train a better classifier?

Split-CIFAR10 stream (10k points, same construction as RQ1). For each buffer size M,
run STREAMCORE, reservoir, and sliding window to completion, then train weighted
multiclass logistic regression on the final buffer and evaluate on raw 512-d
ResNet18 test embeddings (no extra L2 normalization).

Oracle: logistic regression on all 10k stream points (upper bound).

Output: rq2_downstream_output/neurips_fig2_downstream_accuracy.pdf
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import deque
from typing import Deque, List, Sequence, Tuple

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import euclidean_distances
from tqdm import tqdm

import matplotlib

if os.environ.get("RQ2_HEADLESS", "").lower() in ("1", "true", "yes"):
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
# Constants
# ---------------------------------------------------------------------------
SEED = 42
M_VALUES = [50, 100, 200, 400, 800]
MAX_PER_CLASS = 1000
MEDIAN_HEURISTIC_SAMPLE_SIZE = 2000
RFF_D = 1024
K_ITER = 100
# Reuse RQ1 train cache path so we do not re-extract 50k features.
TRAIN_FEATURE_CACHE = os.path.join(PROJECT_ROOT, "feature_cache", "cifar10_train_resnet18_rq1.pkl")
# Distinct filename so runs after dropping L2 norm do not load old normalized caches.
TEST_FEATURE_CACHE = os.path.join(PROJECT_ROOT, "feature_cache", "cifar10_test_resnet18_rq2_raw.pkl")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "rq2_downstream_output")
FIGURE_PATH = os.path.join(OUTPUT_DIR, "neurips_fig2_downstream_accuracy.pdf")

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
    """γ = 1 / (2 · median_{i<j} ||x_i - x_j||²)."""
    n = X.shape[0]
    rng = np.random.RandomState(seed)
    X_sub = X[rng.choice(n, size=min(n, sample_size), replace=False)] if n > sample_size else X
    D_sq = euclidean_distances(X_sub, squared=True)
    tri = np.triu_indices_from(D_sq, k=1)
    med = float(np.median(D_sq[tri]))
    if med <= 0:
        raise ValueError("Non-positive median squared distance.")
    return 1.0 / (2.0 * med)


# =============================================================================
# Data: train (50k), test (10k) — ResNet18 embeddings
# =============================================================================


def load_cifar10_train_embeddings(
    device: torch.device,
    cache_path: str = TRAIN_FEATURE_CACHE,
    force_recompute: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Same pipeline as RQ1: CIFAR-10 train, ResNet18, 512-d, joblib cache."""
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    if not force_recompute and os.path.exists(cache_path):
        print(f"Loading cached train embeddings from {cache_path}")
        data = joblib.load(cache_path)
        return data["X"].astype(np.float64), data["y"].astype(np.int64)

    print("Computing ResNet18 embeddings for CIFAR-10 train (50k)...")
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
    X, y = X.astype(np.float64), y.astype(np.int64)
    joblib.dump({"X": X, "y": y}, cache_path)
    print(f"Saved train embeddings to {cache_path}")
    return X, y


def load_cifar10_test_embeddings(
    device: torch.device,
    cache_path: str = TEST_FEATURE_CACHE,
    force_recompute: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    CIFAR-10 test split (10k), same ResNet18 pipeline as train (raw embeddings).
    Cached for fast reruns.
    """
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    if not force_recompute and os.path.exists(cache_path):
        print(f"Loading cached test embeddings from {cache_path}")
        data = joblib.load(cache_path)
        return data["X"].astype(np.float64), data["y"].astype(np.int64)

    print("Computing ResNet18 embeddings for CIFAR-10 test (10k)...")
    test_ds = datasets.CIFAR10(
        root=os.path.join(PROJECT_ROOT, "data"),
        train=False,
        download=True,
    )
    tfm = transforms.Compose(
        [
            transforms.Resize(224, InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    dataset = TransformDataset(test_ds, tfm)
    X, y = extract_resnet18_features(dataset, device, batch_size=256, num_workers=2)
    X, y = X.astype(np.float64), y.astype(np.int64)
    joblib.dump({"X": X, "y": y}, cache_path)
    print(f"Saved test embeddings to {cache_path}")
    return X, y


def build_split_cifar10_stream_order(
    X: np.ndarray,
    y: np.ndarray,
    pairs: Sequence[Tuple[int, int]] = TASK_CLASS_PAIRS,
    seed: int = SEED,
    max_per_class: int = MAX_PER_CLASS,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Identical stream construction to RQ1 (class-incremental split CIFAR-10)."""
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
    seg_lens = [len(b) for b in blocks]
    return X[stream_idx], y[stream_idx], seg_lens


# =============================================================================
# Labeled baselines (RQ1 buffers did not store labels)
# =============================================================================


class ReservoirBufferLabeled:
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

    def get_xy_weights(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.buffer_x:
            return (
                np.empty((0, 0), dtype=np.float64),
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.float64),
            )
        Xb = np.stack(self.buffer_x, axis=0)
        yb = np.asarray(self.buffer_y, dtype=np.int64)
        m = Xb.shape[0]
        w = np.full(m, 1.0 / m, dtype=np.float64)
        return Xb, yb, w


class SlidingWindowBufferLabeled:
    def __init__(self, M: int):
        self.buf_x: Deque[np.ndarray] = deque(maxlen=M)
        self.buf_y: Deque[int] = deque(maxlen=M)

    def observe(self, x: np.ndarray, y: int) -> None:
        self.buf_x.append(x.copy())
        self.buf_y.append(int(y))

    def get_xy_weights(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(self.buf_x) == 0:
            return (
                np.empty((0, 0), dtype=np.float64),
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.float64),
            )
        Xb = np.stack(list(self.buf_x), axis=0)
        yb = np.asarray(list(self.buf_y), dtype=np.int64)
        m = Xb.shape[0]
        w = np.full(m, 1.0 / m, dtype=np.float64)
        return Xb, yb, w


# =============================================================================
# Streaming + downstream evaluation
# =============================================================================


def streamcore_final_buffer(
    X_stream: np.ndarray,
    y_stream: np.ndarray,
    M: int,
    d_in: int,
    rbf_gamma: float,
    sampler_seed: int = SEED,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run STREAMCORE to t=T; return (X_buf, y_buf, weights summing to 1)."""
    # Same RFF map for every M (only buffer capacity changes across the sweep).
    set_global_seeds(sampler_seed)
    sampler = OrthogonalSampler(d_in=d_in, n_components=RFF_D, gamma=rbf_gamma)
    sc = StreamingCoreset(
        M=M,
        D=RFF_D,
        delta_drift_max=1.0,
        sampler=sampler,
        batch_size=1,
        K_iter=K_ITER,
        verbose=False,
    )
    T = len(X_stream)
    for t0 in range(T):
        x_t = X_stream[t0]
        y_t = int(y_stream[t0])
        sc.process_batch(x_t[np.newaxis, :], np.array([y_t], dtype=np.int64), batch_idx=t0)
    if not sc.buffer_X:
        raise RuntimeError("STREAMCORE buffer empty after full stream.")
    Xb = np.stack([np.asarray(v, dtype=np.float64) for v in sc.buffer_X], axis=0)
    yb = np.asarray(sc.buffer_y, dtype=np.int64)
    w = np.asarray(sc.buffer_weights, dtype=np.float64)
    w = w / (w.sum() + 1e-15)
    return Xb, yb, w


def reservoir_final_buffer(
    X_stream: np.ndarray,
    y_stream: np.ndarray,
    M: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    buf = ReservoirBufferLabeled(M, rng)
    for t0 in range(len(X_stream)):
        buf.observe(X_stream[t0], int(y_stream[t0]))
    return buf.get_xy_weights()


def sliding_window_final_buffer(
    X_stream: np.ndarray,
    y_stream: np.ndarray,
    M: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    buf = SlidingWindowBufferLabeled(M)
    for t0 in range(len(X_stream)):
        buf.observe(X_stream[t0], int(y_stream[t0]))
    return buf.get_xy_weights()


def train_lr_eval_accuracy(
    X_train: np.ndarray,
    y_train: np.ndarray,
    sample_weight: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> float:
    """
    Multinomial logistic regression with optional sample weights; accuracy on test.

    If the buffer is class-collapsed (e.g. sliding window after drift), sklearn cannot
    fit a boundary; we evaluate the degenerate predictor that always outputs the sole class.
    On balanced CIFAR-10 test that is ~10% when one of ten classes is memorized.
    """
    unique_classes = np.unique(y_train)
    if unique_classes.size == 0:
        return 0.0
    if unique_classes.size < 2:
        pred = np.full_like(y_test, unique_classes[0], dtype=np.int64)
        return float(accuracy_score(y_test, pred))

    lr = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        random_state=SEED,
    )
    lr.fit(X_train, y_train, sample_weight=sample_weight)
    pred = lr.predict(X_test)
    return float(accuracy_score(y_test, pred))


def oracle_accuracy_on_stream(
    X_stream: np.ndarray,
    y_stream: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> float:
    """Train on all stream points (uniform weight); evaluate on test."""
    n = X_stream.shape[0]
    sw = np.full(n, 1.0, dtype=np.float64)
    return train_lr_eval_accuracy(X_stream, y_stream, sw, X_test, y_test)


def apply_paper_style() -> None:
    for name in ("seaborn-v0_8-paper", "seaborn-paper", "seaborn-whitegrid"):
        try:
            plt.style.use(name)
            return
        except OSError:
            continue


def plot_rq2(
    M_list: Sequence[int],
    acc_sc: Sequence[float],
    acc_res: Sequence[float],
    acc_sw: Sequence[float],
    oracle_acc: float,
    out_path: str,
    *,
    show: bool = True,
) -> None:
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(5.5, 3.4))

    ax.plot(
        M_list,
        acc_sc,
        color="#1f77b4",
        linestyle="-",
        linewidth=2.0,
        marker="o",
        markersize=5,
        label="STREAMCORE",
    )
    ax.plot(
        M_list,
        acc_res,
        color="#ff7f0e",
        linestyle=":",
        linewidth=1.5,
        marker="s",
        markersize=5,
        label="Reservoir",
    )
    ax.plot(
        M_list,
        acc_sw,
        color="#d62728",
        linestyle="--",
        linewidth=1.5,
        marker="^",
        markersize=5,
        label="Sliding window",
    )

    ax.axhline(
        y=oracle_acc,
        color="black",
        linestyle="--",
        linewidth=1.2,
        alpha=0.7,
        label="Offline oracle (10k stream)",
    )

    ax.set_xscale("log")
    ax.set_xticks(list(M_list))
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v)}"))
    ax.set_xlabel(r"Buffer size $M$")
    ax.set_ylabel("Test accuracy")
    ax.set_ylim(bottom=0.0, top=1.02)
    ax.legend(frameon=True, fontsize=7.5, loc="lower right")
    ax.grid(True, which="both", ls="-", alpha=0.25)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    print(f"Saved figure to {out_path}")

    if show and os.environ.get("RQ2_HEADLESS", "").lower() not in ("1", "true", "yes"):
        try:
            plt.show(block=True)
        except Exception as exc:
            print(f"(Could not display figure: {exc})")
    plt.close(fig)


def main(show_figure: bool = True) -> None:
    set_global_seeds(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = get_device()
    print(f"Device (ResNet features): {device}")

    X_train_all, y_train_all = load_cifar10_train_embeddings(device)
    X_test, y_test = load_cifar10_test_embeddings(device)
    d_in = X_train_all.shape[1]
    assert d_in == 512

    # Median heuristic on full train embeddings (same spirit as RQ1).
    rbf_gamma = compute_median_heuristic_gamma(
        X_train_all, sample_size=MEDIAN_HEURISTIC_SAMPLE_SIZE, seed=SEED
    )
    print(f"Median-heuristic γ = {rbf_gamma:.6g}")

    X_stream, y_stream, _ = build_split_cifar10_stream_order(
        X_train_all, y_train_all, TASK_CLASS_PAIRS, seed=SEED, max_per_class=MAX_PER_CLASS
    )
    T = X_stream.shape[0]
    expected_T = len(TASK_CLASS_PAIRS) * 2 * MAX_PER_CLASS
    assert T == expected_T, f"Stream length {T} != {expected_T}"

    print("Fitting offline oracle (LR on full 10k stream)...")
    oracle_acc = oracle_accuracy_on_stream(X_stream, y_stream, X_test, y_test)
    print(f"Oracle test accuracy: {oracle_acc:.4f}")

    acc_sc: List[float] = []
    acc_res: List[float] = []
    acc_sw: List[float] = []

    for M in tqdm(M_VALUES, desc="Buffer size M"):
        seed_res = SEED + M * 1000 + 2  # reservoir randomness depends on M

        Xb, yb, w = streamcore_final_buffer(X_stream, y_stream, M, d_in, rbf_gamma, SEED)
        # sample_weight: use weights proportional to coreset mass (sum = M for numerical comfort)
        sw = w * float(len(w))
        acc_sc.append(train_lr_eval_accuracy(Xb, yb, sw, X_test, y_test))

        Xb, yb, w = reservoir_final_buffer(X_stream, y_stream, M, seed_res)
        sw = w * float(len(w))
        acc_res.append(train_lr_eval_accuracy(Xb, yb, sw, X_test, y_test))

        Xb, yb, w = sliding_window_final_buffer(X_stream, y_stream, M)
        sw = w * float(len(w))
        acc_sw.append(train_lr_eval_accuracy(Xb, yb, sw, X_test, y_test))

    plot_rq2(
        M_VALUES,
        acc_sc,
        acc_res,
        acc_sw,
        oracle_acc,
        FIGURE_PATH,
        show=show_figure,
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RQ2 downstream accuracy vs buffer size.")
    p.add_argument("--no-show", action="store_true", help="Save PDF only, no GUI.")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(show_figure=not args.no_show)
