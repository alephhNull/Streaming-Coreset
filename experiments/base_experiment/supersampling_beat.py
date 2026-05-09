from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Union, Any, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from embedded_data import load_embedded_train_split
from stream_builders import (
    EXACT_MMD_COMPATIBLE,
    count_label_transitions,
    infer_n_classes,
    logits_to_per_class_counts,
    stratified_sample_stream,
    blocks_to_counts,
    manual_sample_stream,
)
from streaming_coreset import StreamingCoreset
from unweighted_coreset import UnweightedStreamingCoreset
from reservoir_rff_baseline import ReservoirRFFBaseline
from super_sampling import SuperSamplingCoreset
from online_k_center import OnlineKCenterStreamingCoreset

DEFAULT_EXACT_MMD_MAX_STREAM = 2000


class MetricsMode(str, Enum):
    L2_RFF = "l2_rff"
    EXACT_MMD = "exact_mmd"
    BOTH = "both"


def _sanitize_key(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", name.strip()).strip("_").lower() or "run"


@dataclass
class BaseExperimentConfig:
    dataset_name: str = "mnist"
    label_logits: Dict[int, float] = field(default_factory=dict)
    stream_blocks: Optional[List[Tuple[int, float]]] = None
    n_classes: Optional[int] = None
    stream_length: Union[int, str] = 2000
    num_splits: int = 1
    metrics: MetricsMode = MetricsMode.BOTH
    rbf_gamma: float = 0.001
    exact_mmd_stride: int = 5
    seed: int = 42
    output_dir: str = "snapshots_base_experiment"
    data_subset_size: int = 50000
    embed_device: str = "cpu"


@dataclass
class StratifiedStream:
    X: np.ndarray
    y: np.ndarray
    class_change_steps: Optional[List[int]]
    per_class: Dict[int, int]
    n_classes: int

    @property
    def d_in(self) -> int:
        return int(self.X.shape[1])

    @property
    def n_total(self) -> int:
        return int(self.X.shape[0])


def resolve_n_classes(cfg: BaseExperimentConfig) -> int:
    if cfg.n_classes is not None:
        return int(cfg.n_classes)
    if cfg.stream_blocks:
        return int(max(b[0] for b in cfg.stream_blocks) + 1)
    if cfg.label_logits:
        return infer_n_classes(cfg.label_logits)
    ds = cfg.dataset_name.lower()
    if ds in ("mnist", "cifar10", "fashion_mnist", "svhn"):
        return 10
    if ds == "cifar100":
        return 100
    raise ValueError("Set BaseExperimentConfig.n_classes for this dataset.")


def resolve_stream_length(cfg: BaseExperimentConfig) -> int:
    spec = cfg.stream_length
    if spec == EXACT_MMD_COMPATIBLE:
        return DEFAULT_EXACT_MMD_MAX_STREAM
    if isinstance(spec, int):
        if spec < 1:
            raise ValueError("stream_length must be >= 1")
        return spec
    raise ValueError(
        f"stream_length must be int or EXACT_MMD_COMPATIBLE, got {spec!r}"
    )


def load_stratified_stream(cfg: BaseExperimentConfig) -> StratifiedStream:
    """Load embedded data and build the stratified / interleaved stream. Use ``d_in`` to build samplers and ``StreamingCoreset`` instances, then pass them to ``run_base_experiment``."""
    n_classes = resolve_n_classes(cfg)
    L = resolve_stream_length(cfg)

    print(f"[data] Loading {cfg.dataset_name} (embedded train)...")
    X_all, y_all = load_embedded_train_split(
        cfg.dataset_name, cfg.seed, subset_size=cfg.data_subset_size, device=cfg.embed_device
    )

    if cfg.stream_blocks:
        explicit_counts = blocks_to_counts(cfg.stream_blocks, L)
        X, y, class_change_steps, per_class_used = manual_sample_stream(
            X_all, y_all, explicit_counts, n_classes, cfg.seed
        )
    else:
        logits = dict(cfg.label_logits) if cfg.label_logits else {c: 1.0 for c in range(n_classes)}
        per_class = logits_to_per_class_counts(logits, n_classes, L)
        X, y, class_change_steps, per_class_used = stratified_sample_stream(
            X_all, y_all, per_class, n_classes, cfg.num_splits, cfg.seed
        )

    return StratifiedStream(
        X=X,
        y=y,
        class_change_steps=class_change_steps,
        per_class=per_class_used,
        n_classes=n_classes,
    )


def compute_exact_rbf_mmd(
    X_stream: np.ndarray,
    buffer_X: list,
    buffer_weights: np.ndarray,
    gamma: float,
) -> float:
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


def plot_l2_multi(
    series: Dict[str, List[float]],
    save_path: str,
    class_change_steps: Optional[List[int]] = None,
    title: str = "",
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    names = list(series.keys())
    steps = np.arange(1, len(next(iter(series.values()))) + 1)
    for i, name in enumerate(names):
        ax.plot(steps, series[name], linewidth=0.9, color=f"C{i % 10}", label=name)
    if class_change_steps:
        max_vlines = 80
        lines = class_change_steps[:max_vlines] if len(class_change_steps) > max_vlines else class_change_steps
        alpha = 0.35 if len(class_change_steps) > 25 else 0.6
        n0 = len(next(iter(series.values())))
        for t in lines:
            if 1 <= t <= n0:
                ax.axvline(x=t, color="gray", linestyle="--", linewidth=0.45, alpha=alpha)
    ax.set_xlabel("Stream step t")
    ax.set_ylabel(r"L2 $\|\mu_t - \sum_i w_i z_i\|$ (each method's RFF space)")
    ax.set_title(title or "L2 surrogate in each method's RFF space")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_exact_mmd_multi(
    steps_chk: np.ndarray,
    series: Dict[str, np.ndarray],
    save_path: str,
    title: str = "",
    gamma: float = 0.001,
    stride: int = 1,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    names = list(series.keys())
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    for i, name in enumerate(names):
        m = markers[i % len(markers)]
        ax.plot(
            steps_chk,
            series[name],
            marker=m,
            ms=3,
            linewidth=1.0,
            color=f"C{i % 10}",
            label=name,
        )
    ax.set_xlabel("Stream step t")
    ax.set_ylabel("Exact RBF MMD (input space)")
    ax.set_title(
        title
        or f"Exact kernel MMD vs t (gamma={gamma}, checkpoints every {stride})"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def run_base_experiment(
    cfg: BaseExperimentConfig,
    stream: StratifiedStream,
    streamers: Dict[str, StreamingCoreset],
) -> Dict[str, Any]:
    """
    Run one pass over ``stream.X`` / ``stream.y`` for each pre-built streamer.
    You construct ``StreamingCoreset`` (and inject samplers) yourself; this only
    drives the loop, metrics, and plots.
    """
    if not streamers:
        raise ValueError("streamers must be non-empty")

    os.makedirs(cfg.output_dir, exist_ok=True)
    X, y = stream.X, stream.y
    n_total = stream.n_total
    n_trans = count_label_transitions(y)

    any_st = next(iter(streamers.values()))
    M = any_st.M
    D = any_st.rff_dim

    print(
        f"  N={n_total}, d_in={stream.d_in}, M={M}, D={D}, gamma={cfg.rbf_gamma} | "
        f"splits={cfg.num_splits} | label transitions={n_trans}"
    )

    want_l2 = cfg.metrics in (MetricsMode.L2_RFF, MetricsMode.BOTH)
    want_exact = cfg.metrics in (MetricsMode.EXACT_MMD, MetricsMode.BOTH)

    names = list(streamers.keys())
    l2_hist: Dict[str, List[float]] = {name: [] for name in names}
    exact_hist: Dict[str, List[float]] = {name: [] for name in names}
    chk_steps: List[int] = []

    for t in range(n_total):
        for name in names:
            streamers[name].process_batch(X[t : t + 1], y[t : t + 1], batch_idx=t)
        record_exact = want_exact and (t % cfg.exact_mmd_stride == 0 or t == n_total - 1)
        if record_exact:
            X_prefix = X[: t + 1]
            for name in names:
                st = streamers[name]
                exact_hist[name].append(
                    compute_exact_rbf_mmd(
                        X_prefix, st.buffer_X, st.buffer_weights, cfg.rbf_gamma
                    )
                )
            chk_steps.append(t)

    for name in names:
        l2_hist[name] = list(streamers[name].mmd_history)

    stem = os.path.join(cfg.output_dir, "run")
    title_suffix = f"  [{cfg.dataset_name}, N={n_total}, splits={cfg.num_splits}]"

    if want_l2:
        plot_l2_multi(
            l2_hist,
            stem + "_l2_rff_space.png",
            class_change_steps=stream.class_change_steps,
            title="L2 in each method's RFF space" + title_suffix,
        )
        print(f"  Saved: {stem}_l2_rff_space.png")

    if want_exact and chk_steps:
        steps_arr = np.array(chk_steps, dtype=np.int64) + 1
        exact_arr = {k: np.array(v, dtype=np.float64) for k, v in exact_hist.items()}
        plot_exact_mmd_multi(
            steps_arr,
            exact_arr,
            stem + "_exact_rbf_mmd.png",
            title="Exact RBF MMD (input space)" + title_suffix,
            gamma=cfg.rbf_gamma,
            stride=cfg.exact_mmd_stride,
        )
        print(f"  Saved: {stem}_exact_rbf_mmd.png")

    save_dict: Dict[str, Any] = {
        "gamma": cfg.rbf_gamma,
        "D": D,
        "M": M,
        "num_splits": cfg.num_splits,
        "n_label_transitions": n_trans,
        "n_classes": stream.n_classes,
        "per_class": np.array([stream.per_class[c] for c in range(stream.n_classes)], dtype=np.int32),
        "dataset": cfg.dataset_name,
        "stream_length_resolved": n_total,
    }
    for name in names:
        key = _sanitize_key(name)
        save_dict[f"l2_{key}"] = np.array(l2_hist[name], dtype=np.float64)
        if want_exact:
            save_dict[f"exact_mmd_{key}"] = np.array(exact_hist[name], dtype=np.float64)
    if want_exact and chk_steps:
        save_dict["exact_mmd_steps"] = np.array(chk_steps, dtype=np.int64) + 1

    np.savez(os.path.join(cfg.output_dir, "curves.npz"), **save_dict)
    print(f"  Saved: {os.path.join(cfg.output_dir, 'curves.npz')}")

    return {
        "l2": l2_hist,
        "exact_mmd": exact_hist if want_exact else {},
        "chk_steps": chk_steps,
        "per_class": stream.per_class,
        "X_shape": X.shape,
        "class_change_steps": stream.class_change_steps,
    }


def main() -> None:
    from sklearn.kernel_approximation import RBFSampler

    # THE "TSUNAMI" ADVERSARIAL STREAM (Structural Erasure Trap):
    # We populate the stream with 9 distinct minority classes.
    # Then we flood it with an 800-point tsunami of a single class.
    # Super-Sampling (M=10) must use 8 slots to represent the tsunami's mass,
    # forcing it to permanently delete the geometric support of the other classes.
    explicit_blocks = [
        (1, 20), (2, 20), (3, 20), 
        (4, 20), (5, 20), (6, 20), 
        (7, 20), (8, 20), (9, 20), 
        (0, 820)  # The Tsunami (82% of total mass)
    ]

    cfg = BaseExperimentConfig(
        dataset_name="mnist",
        stream_blocks=explicit_blocks,
        stream_length=1000,
        num_splits=1,
        metrics=MetricsMode.BOTH,
        rbf_gamma=0.00009,      # Sharper kernel forces a huge penalty for missing classes
        exact_mmd_stride=10,  # High res tracking
        seed=42,
        output_dir=os.path.join(_PROJECT_ROOT, "stress_test_structural_erasure"),
    )
    stream = load_stratified_stream(cfg)

    # CRITICAL: Budget M=10
    M, D = 10, 1024
    
    np.random.seed(cfg.seed)
    sampler_rff = RBFSampler(gamma=cfg.rbf_gamma, n_components=D, random_state=cfg.seed + 12345)
    sampler_rff.fit(stream.X[: min(10, stream.n_total)])

    streamers = {
        # High K_iter guarantees optimal continuous weights to compress the Tsunami
        "Weighted RFF (Proposed)": StreamingCoreset(M, D, sampler_rff, batch_size=1, K_iter=300),
        "SuperSampling (MMD Reservoir)": SuperSamplingCoreset(M, D, sampler_rff, batch_size=1),
        "Reservoir Sampling": ReservoirRFFBaseline(M, D, sampler_rff, batch_size=1),
        # "Unweighted RFF (Naive FW)": UnweightedStreamingCoreset(M, D, sampler_rff, batch_size=1),
    }
    
    run_base_experiment(cfg, stream, streamers)

if __name__ == "__main__":
    main()