"""
RQ3: Stability decomposition / one-step theorem validation for STREAMCORE.

Uses the same Split-CIFAR10 setup as RQ1:
- cached 512-d ResNet18 CIFAR-10 train embeddings
- same class-incremental task order
- same MAX_PER_CLASS and buffer size M
- same ORF/RFF dimensionality and median-heuristic gamma

At each evaluation step (task boundaries + every 500 points), logs:
- e_prev: ||mu_hat_{t-1} - mu_{t-1}||_2
- e_warm: ||mu_hat_t^(0) - mu_t||_2
- e_opt: ||mu_hat_t^opt - mu_t||_2
- e_final: ||mu_hat_t - mu_t||_2
- warm_ratio: e_warm / ((1-alpha_t) * e_prev + 1e-15)
- opt_gain: e_warm - e_opt
- evict_penalty: e_final - e_opt
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Dict, List, Sequence, Set, Tuple

import joblib
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from tqdm import tqdm

import matplotlib

if os.environ.get("RQ3_HEADLESS", "").lower() in ("1", "true", "yes"):
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
# Shared setup constants (kept aligned with RQ1)
# ---------------------------------------------------------------------------
SEED = 42
M = 100
RFF_D = 1024
K_ITER = 100
MAX_PER_CLASS = 1000
MEDIAN_HEURISTIC_SAMPLE_SIZE = 2000
EVAL_EVERY = 500

FEATURE_CACHE_PATH = os.path.join(PROJECT_ROOT, "feature_cache", "cifar10_train_resnet18_rq1.pkl")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "rq3_stability_output")
CSV_PATH = os.path.join(OUTPUT_DIR, "rq3_stability_decomposition.csv")
FIGURE_PATH = os.path.join(OUTPUT_DIR, "rq3_stability_decomposition.pdf")

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
        raise ValueError("Median squared distance is non-positive.")
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
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
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


def apply_paper_style() -> None:
    for name in ("seaborn-v0_8-paper", "seaborn-paper", "seaborn-whitegrid"):
        try:
            plt.style.use(name)
            return
        except OSError:
            continue


def save_csv(rows: List[Dict[str, float]], csv_path: str) -> None:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    columns = [
        "step",
        "task_id",
        "alpha_t",
        "e_prev",
        "e_warm",
        "e_opt",
        "e_final",
        "warm_ratio",
        "opt_gain",
        "evict_penalty",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Saved CSV to {csv_path}")


def plot_decomposition(
    rows: List[Dict[str, float]],
    transition_x: Sequence[int],
    out_path: str,
    *,
    show: bool = True,
) -> None:
    apply_paper_style()
    steps = np.array([int(r["step"]) for r in rows], dtype=np.int64)
    e_warm = np.array([float(r["e_warm"]) for r in rows], dtype=np.float64)
    e_opt = np.array([float(r["e_opt"]) for r in rows], dtype=np.float64)
    e_final = np.array([float(r["e_final"]) for r in rows], dtype=np.float64)
    warm_ratio = np.array([float(r["warm_ratio"]) for r in rows], dtype=np.float64)
    opt_gain = np.array([float(r["opt_gain"]) for r in rows], dtype=np.float64)
    evict_penalty = np.array([float(r["evict_penalty"]) for r in rows], dtype=np.float64)

    fig, axes = plt.subplots(3, 1, figsize=(7.0, 8.5), sharex=True)

    ax = axes[0]
    ax.plot(steps, e_warm, color="#ff7f0e", linewidth=1.6, label=r"$e_t^{(0)}$ (warm-start)")
    ax.plot(steps, e_opt, color="#2ca02c", linewidth=1.6, label=r"$e_t^{opt}$ (post-opt)")
    ax.plot(steps, e_final, color="#1f77b4", linewidth=1.8, label=r"$e_t$ (final)")
    if np.all(np.concatenate([e_warm, e_opt, e_final]) > 0):
        ax.set_yscale("log")
    ax.set_ylabel("Error norm")
    ax.grid(True, which="both", ls="-", alpha=0.25)
    ax.legend(loc="best", fontsize=8, frameon=True)

    ax = axes[1]
    ax.plot(steps, warm_ratio, color="#9467bd", linewidth=1.6, label="Warm ratio")
    ax.axhline(1.0, color="0.35", linestyle="--", linewidth=1.1, label="Identity = 1")
    ax.set_ylabel(r"$e_t^{(0)} / ((1-\alpha_t)e_{t-1})$")
    ax.grid(True, which="both", ls="-", alpha=0.25)
    ax.legend(loc="best", fontsize=8, frameon=True)

    ax = axes[2]
    ax.plot(steps, opt_gain, color="#2ca02c", linewidth=1.6, label="Optimization gain")
    ax.plot(steps, evict_penalty, color="#d62728", linewidth=1.6, label="Eviction penalty")
    ax.axhline(0.0, color="0.35", linestyle="--", linewidth=1.0)
    ax.set_ylabel("Delta")
    ax.set_xlabel(r"Stream step $t$")
    ax.grid(True, which="both", ls="-", alpha=0.25)
    ax.legend(loc="best", fontsize=8, frameon=True)

    for axis in axes:
        for xv in transition_x:
            axis.axvline(x=xv, color="0.45", linestyle="--", linewidth=0.85, alpha=0.8)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    print(f"Saved figure to {out_path}")

    if show and os.environ.get("RQ3_HEADLESS", "").lower() not in ("1", "true", "yes"):
        try:
            plt.show(block=True)
        except Exception as exc:
            print(f"(Could not display figure interactively: {exc})")

    plt.close(fig)


def print_summary(rows: List[Dict[str, float]]) -> None:
    warm_ratio = np.array([float(r["warm_ratio"]) for r in rows], dtype=np.float64)
    opt_gain = np.array([float(r["opt_gain"]) for r in rows], dtype=np.float64)
    evict_penalty = np.array([float(r["evict_penalty"]) for r in rows], dtype=np.float64)

    warm_dev = np.abs(warm_ratio - 1.0)
    print("\n" + "=" * 76)
    print("RQ3 stability decomposition summary")
    print("=" * 76)
    print(f"Mean warm_ratio deviation |ratio-1|: {warm_dev.mean():.6e}")
    print(f"Max  warm_ratio deviation |ratio-1|: {warm_dev.max():.6e}")
    print(f"Mean optimization gain (e_warm - e_opt): {opt_gain.mean():.6e}")
    print(f"Mean eviction penalty  (e_final - e_opt): {evict_penalty.mean():.6e}")
    print(f"Max  eviction penalty  (e_final - e_opt): {evict_penalty.max():.6e}")
    print("=" * 76 + "\n")


def main(show_figure: bool = True) -> None:
    set_global_seeds(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = get_device()
    print(f"Device for ResNet18 feature extraction: {device}")

    X_all, y_all = load_cifar10_train_embeddings(device)
    d_in = X_all.shape[1]
    assert d_in == 512, f"Expected 512-d ResNet18 features, got {d_in}"

    rbf_gamma = compute_median_heuristic_gamma(
        X_all, sample_size=MEDIAN_HEURISTIC_SAMPLE_SIZE, seed=SEED
    )
    print(
        f"Median-heuristic RBF gamma = {rbf_gamma:.6g} "
        f"(subsample n={min(MEDIAN_HEURISTIC_SAMPLE_SIZE, len(X_all))})"
    )

    X_stream, y_stream, segment_lengths = build_split_cifar10_stream_order(
        X_all, y_all, TASK_CLASS_PAIRS, seed=SEED, max_per_class=MAX_PER_CLASS
    )
    T = X_stream.shape[0]
    expected_T = len(TASK_CLASS_PAIRS) * 2 * MAX_PER_CLASS
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

    eval_at = evaluation_steps(T, segment_lengths, every=EVAL_EVERY)
    transition_x = [int(x) for x in np.cumsum(segment_lengths)[:-1]]
    print(
        f"Stream length T={T}, M={M}, RFF_D={RFF_D}, K_iter={K_ITER}, "
        f"evaluation points={len(eval_at)}"
    )

    rows: List[Dict[str, float]] = []

    # True prefix mean in the same ORF space, maintained recursively.
    mu_true = np.zeros(RFF_D, dtype=np.float64)
    # Previous-step final estimate and true mean (for e_prev at current t).
    prev_mu_hat_final = np.zeros(RFF_D, dtype=np.float64)
    prev_mu_true = np.zeros(RFF_D, dtype=np.float64)

    pbar = tqdm(range(T), desc="RQ3 stability stream", ncols=90)
    for t0 in pbar:
        x_t = X_stream[t0]
        y_t = int(y_stream[t0])
        step = t0 + 1

        # Map current point with the same sampler used by STREAMCORE, then update true mean.
        phi_t = streamcore.map_to_feature_space(x_t)[0]
        alpha_t = 1.0 / float(step)
        mu_true = (1.0 - alpha_t) * mu_true + alpha_t * phi_t

        streamcore.process_batch(x_t[np.newaxis, :], np.array([y_t], dtype=np.int64), batch_idx=t0)
        diag = streamcore.last_step_diagnostics

        # Consistency checks: recursive mean and internal diagnostics must match current step.
        if int(diag["t"]) != step:
            raise RuntimeError(f"Diagnostic step mismatch: got {diag['t']} expected {step}")
        if not np.allclose(diag["true_mean"], mu_true, atol=1e-9, rtol=1e-7):
            raise RuntimeError("True ORF prefix mean mismatch between external and internal updates.")

        if step in eval_at:
            e_prev = float(np.linalg.norm(prev_mu_hat_final - prev_mu_true))
            e_warm = float(np.linalg.norm(diag["warm_estimate"] - mu_true))
            e_opt = float(np.linalg.norm(diag["post_opt_estimate"] - mu_true))
            e_final = float(np.linalg.norm(diag["final_estimate"] - mu_true))
            warm_ratio = e_warm / ((1.0 - alpha_t) * e_prev + 1e-15)
            opt_gain = e_warm - e_opt
            evict_penalty = e_final - e_opt

            rows.append(
                {
                    "step": int(step),
                    "task_id": task_id_from_step(step, segment_lengths),
                    "alpha_t": float(alpha_t),
                    "e_prev": e_prev,
                    "e_warm": e_warm,
                    "e_opt": e_opt,
                    "e_final": e_final,
                    "warm_ratio": warm_ratio,
                    "opt_gain": opt_gain,
                    "evict_penalty": evict_penalty,
                }
            )

        prev_mu_hat_final = diag["final_estimate"].copy()
        prev_mu_true = mu_true.copy()

    save_csv(rows, CSV_PATH)
    plot_decomposition(rows, transition_x, FIGURE_PATH, show=show_figure)
    print_summary(rows)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RQ3 stability decomposition on Split-CIFAR10 stream.")
    p.add_argument(
        "--no-show",
        action="store_true",
        help="Save figure only; do not display interactive plot window.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(show_figure=not args.no_show)
