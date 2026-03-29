"""
RQ3: Ablation — Kinematic warm-start + Pairwise FW vs Vanilla Frank–Wolfe baseline.

Split-CIFAR10 ResNet stream (10k). Training embeddings are L2 row-normalized before the
median-heuristic γ so the RBF landscape is sharp. STREAMCORE uses the default PFW loop
with kinematic weight warm-start; the baseline re-initializes to a 1-hot vertex on the
new point and runs standard Vanilla FW for K steps per arrival.

Output: rq3_ablation_output/neurips_fig3_ablation_warmstart.pdf
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Sequence, Set, Tuple

import joblib
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

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
SEED = 42
M = 100
RFF_D = 1024
K_ITER = 15  # strict per-step FW budget (ablation stress test)
EVAL_EVERY = 500
MAX_PER_CLASS = 1000
MEDIAN_HEURISTIC_SAMPLE_SIZE = 2000
TRAIN_FEATURE_CACHE = os.path.join(PROJECT_ROOT, "feature_cache", "cifar10_train_resnet18_rq1.pkl")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "rq3_ablation_output")
FIGURE_PATH = os.path.join(OUTPUT_DIR, "neurips_fig3_ablation_warmstart.pdf")

TASK_CLASS_PAIRS: List[Tuple[int, int]] = [
    (0, 1),
    (2, 3),
    (4, 5),
    (6, 7),
    (8, 9),
]


class VanillaFWStreamingCoreset(StreamingCoreset):
    """
    Baseline: same buffer and RFF geometry as STREAMCORE, but each step re-initializes
    weights to the vertex e_{new} (1-hot on the latest point) and runs **Vanilla**
    Frank–Wolfe (atom = argmin gradient) with exact line search — no pairwise / away steps.
    """

    def _process_point(self, x_raw, y_label, z_rff, batch_idx, local_idx):
        self.t += 1
        alpha = 1.0 / self.t
        self.mean_rff = (1.0 - alpha) * self.mean_rff + alpha * z_rff

        self.buffer_X.append(x_raw)
        self.buffer_y.append(y_label)
        self.buffer_provenance.append((batch_idx, local_idx))

        if len(self.buffer_Z) > 0:
            self.buffer_Z = np.vstack([self.buffer_Z, z_rff[np.newaxis, :]])
        else:
            self.buffer_Z = z_rff[np.newaxis, :]

        m_now = self.buffer_Z.shape[0]
        # 1-hot on the newest buffer point (cold start in weight space each step).
        self.buffer_weights = np.zeros(m_now, dtype=np.float64)
        self.buffer_weights[-1] = 1.0

        if m_now > 1:
            K_mat = self.buffer_Z @ self.buffer_Z.T
            linear_term = self.buffer_Z @ self.mean_rff
            weights = self.buffer_weights.copy()

            for _ in range(self.K_iter):
                grad = K_mat @ weights - linear_term
                idx_fw = int(np.argmin(grad))

                d = np.zeros(m_now, dtype=np.float64)
                d[idx_fw] = 1.0
                d -= weights

                denom = float(d @ K_mat @ d)
                if denom > 1e-10:
                    gamma_fw = -float(grad @ d) / denom
                else:
                    gamma_fw = 0.0
                gamma_fw = float(np.clip(gamma_fw, 0.0, 1.0))

                weights = weights + gamma_fw * d

            self.buffer_weights = weights

        if len(self.buffer_Z) > self.M:
            evict = np.argmin(self.buffer_weights)
            self.buffer_Z = np.delete(self.buffer_Z, evict, axis=0)
            self.buffer_weights = np.delete(self.buffer_weights, evict)
            del self.buffer_X[evict]
            del self.buffer_y[evict]
            del self.buffer_provenance[evict]

            s = np.sum(self.buffer_weights)
            if s > 1e-9:
                self.buffer_weights /= s

        self.mmd_history.append(self.get_current_mmd())


def set_global_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def l2_normalize_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Unit L2 norm per row (sharpens pairwise distances for median-heuristic γ and MMD)."""
    X = np.asarray(X, dtype=np.float64)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, eps)


def compute_median_heuristic_gamma(
    X: np.ndarray,
    sample_size: int = MEDIAN_HEURISTIC_SAMPLE_SIZE,
    seed: int = SEED,
) -> float:
    n = X.shape[0]
    rng = np.random.RandomState(seed)
    X_sub = X[rng.choice(n, size=min(n, sample_size), replace=False)] if n > sample_size else X
    D_sq = euclidean_distances(X_sub, squared=True)
    tri = np.triu_indices_from(D_sq, k=1)
    med = float(np.median(D_sq[tri]))
    if med <= 0:
        raise ValueError("Non-positive median squared distance.")
    return 1.0 / (2.0 * med)


def load_cifar10_train_embeddings(
    device: torch.device,
    cache_path: str = TRAIN_FEATURE_CACHE,
    force_recompute: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
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
    seg_lens = [len(b) for b in blocks]
    return X[stream_idx], y[stream_idx], seg_lens


# --- Exact RBF MMD (batched), same as RQ1 ---------------------------------


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
    n = X_prefix.shape[0]
    if n == 0 or Z.shape[0] == 0:
        return float("nan")
    exx = term_ek_xx_batched(X_prefix, gamma, block=block_xx)
    exz = term_ek_xz_batched(X_prefix, Z, w, gamma, block=block_xz)
    ezz = term_ek_zz(Z, w, gamma)
    return max(exx - 2.0 * exz + ezz, 0.0)


def streamcore_buffer_to_weighted_set(streamer: StreamingCoreset) -> Tuple[np.ndarray, np.ndarray]:
    if not streamer.buffer_X:
        return np.empty((0, 0), dtype=np.float64), np.empty(0, dtype=np.float64)
    Z = np.stack([np.asarray(v, dtype=np.float64) for v in streamer.buffer_X], axis=0)
    w = np.asarray(streamer.buffer_weights, dtype=np.float64)
    return Z, w


def evaluation_steps(T: int, segment_lengths: Sequence[int], every: int) -> Set[int]:
    boundary_ends = set(int(x) for x in np.cumsum(segment_lengths)[:-1])
    steps: Set[int] = set()
    for t in range(every, T + 1, every):
        steps.add(t)
    steps |= boundary_ends
    steps.add(T)
    return steps


def apply_paper_style() -> None:
    for name in ("seaborn-v0_8-paper", "seaborn-paper", "seaborn-whitegrid"):
        try:
            plt.style.use(name)
            return
        except OSError:
            continue


def run_stream_and_collect_mmd(
    streamer: StreamingCoreset,
    X_stream: np.ndarray,
    y_stream: np.ndarray,
    rbf_gamma: float,
    eval_at: Set[int],
) -> Tuple[List[int], List[float]]:
    rec_t: List[int] = []
    rec_mmd: List[float] = []
    T = len(X_stream)
    for t0 in range(T):
        x_t = X_stream[t0]
        y_t = int(y_stream[t0])
        streamer.process_batch(x_t[np.newaxis, :], np.array([y_t], dtype=np.int64), batch_idx=t0)
        step = t0 + 1
        if step not in eval_at:
            continue
        X_prefix = X_stream[:step]
        Z, w = streamcore_buffer_to_weighted_set(streamer)
        mmd = exact_rbf_mmd_squared(X_prefix, Z, w, rbf_gamma)
        rec_t.append(step)
        rec_mmd.append(mmd)
    return rec_t, rec_mmd


def print_console_insights(
    steps: Sequence[int],
    mmd_warm: Sequence[float],
    mmd_vanilla: Sequence[float],
    segment_lengths: Sequence[int],
    transition_x: Sequence[int],
) -> None:
    mw = np.asarray(mmd_warm, dtype=np.float64)
    mv = np.asarray(mmd_vanilla, dtype=np.float64)
    print("\n" + "=" * 72)
    print("RQ3 ABLATION — SUMMARY (exact RBF MMD² vs stream prefix)")
    print("=" * 72)

    print(
        f"\nGlobal: mean MMD²  warm-start PFW = {mw.mean():.6g}   "
        f"baseline Vanilla FW = {mv.mean():.6g}"
    )
    print(f"         final step T={steps[-1]}:  PFW = {mw[-1]:.6g}   Vanilla FW = {mv[-1]:.6g}")
    if mw[-1] > 1e-12:
        print(f"         Vanilla / PFW at T  = {mv[-1] / mw[-1]:.2f}×")

    cum = np.cumsum([0] + list(segment_lengths))
    for k, (a, b) in enumerate(zip(cum[:-1], cum[1:])):
        mask = (np.array(steps) > a) & (np.array(steps) <= b)
        if not np.any(mask):
            continue
        print(
            f"\n  Task {k + 1} (stream steps {a + 1}–{b}): "
            f"mean MMD² PFW={mw[mask].mean():.6g}  Vanilla={mv[mask].mean():.6g}"
        )

    print("\n  Values at first eval on/after each task boundary:")
    for xv in transition_x:
        after = [s for s in steps if s >= xv]
        if not after:
            continue
        s_eval = min(after)
        idx = steps.index(s_eval)
        print(
            f"    step ~{xv}→{s_eval}:  PFW={mmd_warm[idx]:.6g}  Vanilla={mmd_vanilla[idx]:.6g}  "
            f"(ratio Vanilla/PFW={mmd_vanilla[idx] / (mmd_warm[idx] + 1e-15):.2f})"
        )

    print(
        "\nInterpretation: With K_iter=%d, STREAMCORE’s kinematic warm-start + PFW tracks "
        "the running mean in RFF space efficiently; Vanilla FW from a 1-hot restart is a "
        "weaker optimizer under the same budget, especially after distribution shift."
        % K_ITER
    )
    print("=" * 72 + "\n")


def plot_rq3(
    steps: Sequence[int],
    mmd_warm: Sequence[float],
    mmd_vanilla: Sequence[float],
    transition_x: Sequence[int],
    out_path: str,
    *,
    show: bool = True,
) -> None:
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(5.5, 3.4))

    ax.plot(
        steps,
        mmd_warm,
        color="#1f77b4",
        linestyle="-",
        linewidth=2.0,
        label="STREAMCORE (Warm-Start PFW)",
    )
    ax.plot(
        steps,
        mmd_vanilla,
        color="#d62728",
        linestyle="--",
        linewidth=2.0,
        label="Baseline (Vanilla FW)",
    )

    for xv in transition_x:
        ax.axvline(x=xv, color="0.45", linestyle="--", linewidth=0.9, alpha=0.85)

    ax.set_xlabel(r"Stream step $t$")
    ax.set_ylabel(r"Exact RBF $\mathrm{MMD}^2$")
    ax.set_yscale("log")
    ax.legend(frameon=True, fontsize=7.5, loc="best")
    ax.grid(True, which="both", ls="-", alpha=0.25)

    ax.text(
        0.02,
        0.98,
        f"Computation budget: $K={K_ITER}$ iterations/step",
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="0.7", alpha=0.92),
    )

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    print(f"Saved figure to {out_path}")

    if show and os.environ.get("RQ3_HEADLESS", "").lower() not in ("1", "true", "yes"):
        try:
            plt.show(block=True)
        except Exception as exc:
            print(f"(Could not display figure: {exc})")
    plt.close(fig)


def main(show_figure: bool = True) -> None:
    set_global_seeds(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = get_device()
    print(f"Device (ResNet, if needed): {device}")

    X_train_all, y_train_all = load_cifar10_train_embeddings(device)
    d_in = X_train_all.shape[1]
    assert d_in == 512

    X_train_all = l2_normalize_rows(X_train_all)

    rbf_gamma = compute_median_heuristic_gamma(
        X_train_all, sample_size=MEDIAN_HEURISTIC_SAMPLE_SIZE, seed=SEED
    )
    print(f"Median-heuristic γ = {rbf_gamma:.6g}")

    X_stream, y_stream, segment_lengths = build_split_cifar10_stream_order(
        X_train_all, y_train_all, TASK_CLASS_PAIRS, seed=SEED, max_per_class=MAX_PER_CLASS
    )
    T = X_stream.shape[0]
    assert T == len(TASK_CLASS_PAIRS) * 2 * MAX_PER_CLASS

    eval_at = evaluation_steps(T, segment_lengths, every=EVAL_EVERY)
    transition_x = [int(x) for x in np.cumsum(segment_lengths)[:-1]]

    print(f"Stream length T={T}, buffer M={M}, RFF_D={RFF_D}, K_iter={K_ITER}")
    print(f"Recording exact MMD at {len(eval_at)} steps (every {EVAL_EVERY} + task ends + T).")

    set_global_seeds(SEED)
    sampler = OrthogonalSampler(d_in=d_in, n_components=RFF_D, gamma=rbf_gamma)

    warm = StreamingCoreset(
        M=M,
        D=RFF_D,
        delta_drift_max=1.0,
        sampler=sampler,
        batch_size=1,
        K_iter=K_ITER,
        verbose=False,
    )

    set_global_seeds(SEED)
    sampler_v = OrthogonalSampler(d_in=d_in, n_components=RFF_D, gamma=rbf_gamma)
    vanilla = VanillaFWStreamingCoreset(
        M=M,
        D=RFF_D,
        delta_drift_max=1.0,
        sampler=sampler_v,
        batch_size=1,
        K_iter=K_ITER,
        verbose=False,
    )

    print("\nRunning STREAMCORE (Warm-Start PFW)...")
    steps_w, mmd_w = run_stream_and_collect_mmd(warm, X_stream, y_stream, rbf_gamma, eval_at)
    print("Running baseline (Vanilla FW, 1-hot restart)...")
    steps_v, mmd_v = run_stream_and_collect_mmd(vanilla, X_stream, y_stream, rbf_gamma, eval_at)
    assert steps_w == steps_v

    print_console_insights(steps_w, mmd_w, mmd_v, segment_lengths, transition_x)

    plot_rq3(steps_w, mmd_w, mmd_v, transition_x, FIGURE_PATH, show=show_figure)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RQ3: PFW + warm-start vs Vanilla FW baseline.")
    p.add_argument("--no-show", action="store_true", help="Save PDF only.")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(show_figure=not args.no_show)
