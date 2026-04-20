"""
RQ4: Forgetting and recovery under recurring classes.

Demonstrates that with small memory M, a class can disappear from the STREAMCORE
buffer during non-stationary phases, and later be reacquired when it returns.

Default recurring stream:
    phase 1: class 0 only, n_phase points
    phase 2: class 1 only, n_phase points
    phase 3: class 0 only, n_phase points

Optional 4-phase stream:
    0 -> 1 -> 2 -> 0

At regular checkpoints, records:
    - step / phase_id / buffer size
    - exact class histogram in buffer
    - ORF tracking error ||mu_hat_t - mu_t||_2
    - exact RBF MMD^2(buffer empirical, prefix empirical)
    - per-phase-class presence indicators in buffer
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import joblib
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from tqdm import tqdm

import matplotlib

if os.environ.get("RQ4_HEADLESS", "").lower() in ("1", "true", "yes"):
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


SEED = 42
M = 3
RFF_D = 1024
K_ITER = 100
N_PHASE = 1500
RECORD_EVERY = 20
MEDIAN_HEURISTIC_SAMPLE_SIZE = 2000

FEATURE_CACHE_PATH = os.path.join(PROJECT_ROOT, "feature_cache", "cifar10_train_resnet18_rq1.pkl")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "rq4_forgetting_recovery_output")
CSV_PATH = os.path.join(OUTPUT_DIR, "rq4_forgetting_recovery.csv")
FIGURE_PATH = os.path.join(OUTPUT_DIR, "rq4_forgetting_recovery.pdf")


@dataclass
class RQ4Config:
    n_phase: int = N_PHASE
    M: int = M
    record_every: int = RECORD_EVERY
    use_four_phase: bool = False
    no_show: bool = False


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

    print("Computing ResNet18 embeddings for CIFAR-10 train (50k) -- one-time cost...")
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


def build_recurring_stream(
    X_all: np.ndarray,
    y_all: np.ndarray,
    phase_classes: Sequence[int],
    n_phase: int,
    seed: int = SEED,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    rng = np.random.RandomState(seed)
    blocks: List[np.ndarray] = []

    for cls in phase_classes:
        idx = np.where(y_all == cls)[0]
        if len(idx) < n_phase:
            raise ValueError(f"Class {cls} has only {len(idx)} points; need n_phase={n_phase}.")
        chosen = rng.choice(idx, size=n_phase, replace=False)
        rng.shuffle(chosen)
        blocks.append(chosen)

    stream_idx = np.concatenate(blocks, axis=0)
    segment_lengths = [n_phase for _ in phase_classes]
    return X_all[stream_idx], y_all[stream_idx], segment_lengths


def phase_id_from_step(step: int, segment_lengths: Sequence[int]) -> int:
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
    s = 0.0
    for i0 in range(0, n, block):
        i1 = min(i0 + block, n)
        Xi = X[i0:i1]
        for j0 in range(0, n, block):
            j1 = min(j0 + block, n)
            Xj = X[j0:j1]
            s += float(_rbf_kernel_block(Xi, Xj, gamma).sum())
    return s / float(n * n)


def term_ek_xz_batched(
    X: np.ndarray,
    Z: np.ndarray,
    w: np.ndarray,
    gamma: float,
    block: int = 512,
) -> float:
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
) -> float:
    n = X_prefix.shape[0]
    if n == 0 or Z.shape[0] == 0:
        return float("nan")
    exx = term_ek_xx_batched(X_prefix, gamma, block=256)
    exz = term_ek_xz_batched(X_prefix, Z, w, gamma, block=512)
    ezz = term_ek_zz(Z, w, gamma)
    return max(exx - 2.0 * exz + ezz, 0.0)


def streamcore_buffer_to_weighted_set(streamer: StreamingCoreset) -> Tuple[np.ndarray, np.ndarray]:
    if not streamer.buffer_X:
        return np.empty((0, 0), dtype=np.float64), np.empty(0, dtype=np.float64)
    Z = np.stack([np.asarray(v, dtype=np.float64) for v in streamer.buffer_X], axis=0)
    w = np.asarray(streamer.buffer_weights, dtype=np.float64)
    return Z, w


def apply_paper_style() -> None:
    for name in ("seaborn-v0_8-paper", "seaborn-paper", "seaborn-whitegrid"):
        try:
            plt.style.use(name)
            return
        except OSError:
            continue


def save_csv(rows: List[Dict[str, float]], path: str, phase_classes: Sequence[int]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tracked_classes = sorted(set(int(c) for c in phase_classes))
    cols = [
        "step",
        "phase_id",
        "buffer_size",
        "tracking_error_l2",
        "exact_mmd2",
    ]
    cols += [f"buffer_count_class_{c}" for c in tracked_classes]
    cols += [f"class_{c}_present" for c in tracked_classes]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Saved CSV to {path}")


def plot_rq4(
    rows: List[Dict[str, float]],
    phase_classes: Sequence[int],
    phase_boundaries: Sequence[int],
    forgetting_step: int | None,
    recovery_step: int | None,
    out_path: str,
    *,
    show: bool = True,
) -> None:
    apply_paper_style()
    tracked_classes = sorted(set(int(c) for c in phase_classes))

    steps = np.array([int(r["step"]) for r in rows], dtype=np.int64)
    tracking_error = np.array([float(r["tracking_error_l2"]) for r in rows], dtype=np.float64)
    mmd2 = np.array([float(r["exact_mmd2"]) for r in rows], dtype=np.float64)
    composition = np.vstack(
        [
            np.array([float(r[f"buffer_count_class_{c}"]) for r in rows], dtype=np.float64)
            for c in tracked_classes
        ]
    )

    fig, axes = plt.subplots(3, 1, figsize=(8.2, 8.5), sharex=True)

    ax = axes[0]
    labels = [f"class {c}" for c in tracked_classes]
    ax.stackplot(steps, composition, labels=labels, alpha=0.85)
    ax.set_ylabel("Buffer composition")
    ax.legend(loc="upper right", fontsize=8, frameon=True)
    ax.grid(True, which="both", ls="-", alpha=0.25)

    ax = axes[1]
    ax.plot(steps, tracking_error, color="#1f77b4", linewidth=1.8)
    ax.set_ylabel(r"$\|\hat{\mu}_t - \mu_t\|_2$")
    ax.grid(True, which="both", ls="-", alpha=0.25)

    ax = axes[2]
    ax.plot(steps, mmd2, color="#d62728", linewidth=1.8)
    ax.set_ylabel(r"Exact RBF $\mathrm{MMD}^2$")
    ax.set_xlabel(r"Stream step $t$")
    ax.grid(True, which="both", ls="-", alpha=0.25)

    for axis in axes:
        for xv in phase_boundaries:
            axis.axvline(x=xv, color="0.45", linestyle="--", linewidth=0.9, alpha=0.85)
        if forgetting_step is not None:
            axis.axvline(x=forgetting_step, color="#ff7f0e", linestyle="--", linewidth=1.2, alpha=0.9)
        if recovery_step is not None:
            axis.axvline(x=recovery_step, color="#2ca02c", linestyle="--", linewidth=1.2, alpha=0.9)

    if forgetting_step is not None:
        axes[0].annotate(
            "forgetting",
            xy=(forgetting_step, axes[0].get_ylim()[1] * 0.8),
            xytext=(4, 0),
            textcoords="offset points",
            color="#ff7f0e",
            fontsize=8,
        )
    if recovery_step is not None:
        axes[0].annotate(
            "recovery",
            xy=(recovery_step, axes[0].get_ylim()[1] * 0.6),
            xytext=(4, 0),
            textcoords="offset points",
            color="#2ca02c",
            fontsize=8,
        )

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    print(f"Saved figure to {out_path}")

    if show and os.environ.get("RQ4_HEADLESS", "").lower() not in ("1", "true", "yes"):
        try:
            plt.show(block=True)
        except Exception as exc:
            print(f"(Could not display figure interactively: {exc})")
    plt.close(fig)


def summarize_phase_errors(rows: List[Dict[str, float]], num_phases: int) -> Dict[int, Tuple[float, float, float]]:
    out: Dict[int, Tuple[float, float, float]] = {}
    for p in range(1, num_phases + 1):
        vals = np.array([float(r["tracking_error_l2"]) for r in rows if int(r["phase_id"]) == p], dtype=np.float64)
        if vals.size == 0:
            out[p] = (float("nan"), float("nan"), float("nan"))
        else:
            out[p] = (float(vals.min()), float(vals.max()), float(vals.mean()))
    return out


def main(cfg: RQ4Config) -> None:
    set_global_seeds(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = get_device()
    print(f"Device for ResNet18 feature extraction: {device}")

    phase_classes = [0, 1, 2, 0] if cfg.use_four_phase else [0, 1, 0]
    print(
        f"RQ4 stream phases={phase_classes}, n_phase={cfg.n_phase}, M={cfg.M}, "
        f"record_every={cfg.record_every}, seed={SEED}"
    )

    X_all, y_all = load_cifar10_train_embeddings(device)
    d_in = X_all.shape[1]
    assert d_in == 512, f"Expected 512-d ResNet18 features, got {d_in}"

    rbf_gamma = compute_median_heuristic_gamma(X_all, sample_size=MEDIAN_HEURISTIC_SAMPLE_SIZE, seed=SEED)
    print(f"Median-heuristic RBF gamma = {rbf_gamma:.6g}")

    X_stream, y_stream, segment_lengths = build_recurring_stream(
        X_all, y_all, phase_classes=phase_classes, n_phase=cfg.n_phase, seed=SEED
    )
    T = X_stream.shape[0]
    phase_boundaries = [int(x) for x in np.cumsum(segment_lengths)[:-1]]

    sampler = OrthogonalSampler(d_in=d_in, n_components=RFF_D, gamma=rbf_gamma)
    streamcore = StreamingCoreset(
        M=cfg.M,
        D=RFF_D,
        sampler=sampler,
        batch_size=1,
        K_iter=K_ITER,
        verbose=False,
    )

    rows: List[Dict[str, float]] = []
    tracked_classes = sorted(set(phase_classes))
    phase3_start = int(np.sum(segment_lengths[:2])) + 1 if len(segment_lengths) >= 3 else None
    phase2_start = int(segment_lengths[0]) + 1 if len(segment_lengths) >= 2 else None
    forgetting_step: int | None = None
    recovery_step: int | None = None

    pbar = tqdm(range(T), desc="RQ4 recurring stream", ncols=90)
    for t0 in pbar:
        step = t0 + 1
        x_t = X_stream[t0]
        y_t = int(y_stream[t0])
        streamcore.process_batch(x_t[np.newaxis, :], np.array([y_t], dtype=np.int64), batch_idx=t0)

        if step % cfg.record_every != 0 and step != T:
            continue

        phase_id = phase_id_from_step(step, segment_lengths)
        counts = {int(c): 0 for c in tracked_classes}
        for lbl in streamcore.buffer_y:
            lbl_i = int(lbl)
            if lbl_i in counts:
                counts[lbl_i] += 1

        tracking_error = float(streamcore.get_current_mmd())
        Z, w = streamcore_buffer_to_weighted_set(streamcore)
        exact_mmd2 = exact_rbf_mmd_squared(X_stream[:step], Z, w, rbf_gamma)

        row: Dict[str, float] = {
            "step": int(step),
            "phase_id": int(phase_id),
            "buffer_size": int(len(streamcore.buffer_y)),
            "tracking_error_l2": tracking_error,
            "exact_mmd2": float(exact_mmd2),
        }
        for c in tracked_classes:
            cnt = int(counts[c])
            row[f"buffer_count_class_{c}"] = cnt
            row[f"class_{c}_present"] = int(cnt > 0)
        rows.append(row)

        c0_count = int(counts.get(0, 0))
        if phase2_start is not None and step >= phase2_start and forgetting_step is None and c0_count == 0:
            forgetting_step = step
        if phase3_start is not None and step >= phase3_start and recovery_step is None and c0_count > 0:
            recovery_step = step

    save_csv(rows, CSV_PATH, phase_classes=phase_classes)
    plot_rq4(
        rows,
        phase_classes=phase_classes,
        phase_boundaries=phase_boundaries,
        forgetting_step=forgetting_step,
        recovery_step=recovery_step,
        out_path=FIGURE_PATH,
        show=not cfg.no_show,
    )

    forgetting_occurred = forgetting_step is not None
    forgetting_interval = None
    if forgetting_step is not None and recovery_step is not None and recovery_step >= forgetting_step:
        forgetting_interval = recovery_step - forgetting_step
    recovery_lag = None
    if phase3_start is not None and recovery_step is not None:
        recovery_lag = recovery_step - phase3_start

    print("\n" + "=" * 80)
    print("RQ4 forgetting / recovery summary")
    print("=" * 80)
    print(f"Forgetting occurred (class 0 vanished): {'yes' if forgetting_occurred else 'no'}")
    print(f"Forgetting step: {forgetting_step if forgetting_step is not None else 'not detected'}")
    print(f"Recovery step: {recovery_step if recovery_step is not None else 'not detected'}")
    print(
        "Forgetting interval length: "
        f"{forgetting_interval if forgetting_interval is not None else 'not available'}"
    )
    print(f"Recovery lag from phase 3 start: {recovery_lag if recovery_lag is not None else 'not available'}")
    print("-" * 80)
    phase_stats = summarize_phase_errors(rows, num_phases=len(segment_lengths))
    for p in range(1, len(segment_lengths) + 1):
        mn, mx, mean = phase_stats[p]
        print(f"Phase {p}: tracking error min={mn:.6e}, max={mx:.6e}, mean={mean:.6e}")
    print("=" * 80 + "\n")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="RQ4 forgetting and recovery on recurring class stream (CIFAR-10 embeddings)."
    )
    p.add_argument("--n-phase", type=int, default=N_PHASE, help="Points per phase (default: 1500).")
    p.add_argument("--M", type=int, default=M, help="STREAMCORE buffer size (default: 20).")
    p.add_argument(
        "--record-every",
        type=int,
        default=RECORD_EVERY,
        help="Record metrics every N stream steps (default: 20).",
    )
    p.add_argument(
        "--four-phase",
        action="store_true",
        help="Use 4-phase recurring stream: 0 -> 1 -> 2 -> 0 (default is 0 -> 1 -> 0).",
    )
    p.add_argument(
        "--no-show",
        action="store_true",
        help="Save PDF only; do not open interactive figure window.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.n_phase <= 0:
        raise ValueError("--n-phase must be positive.")
    if args.M <= 0:
        raise ValueError("--M must be positive.")
    if args.record_every <= 0:
        raise ValueError("--record-every must be positive.")

    main(
        RQ4Config(
            n_phase=int(args.n_phase),
            M=int(args.M),
            record_every=int(args.record_every),
            use_four_phase=bool(args.four_phase),
            no_show=bool(args.no_show),
        )
    )
