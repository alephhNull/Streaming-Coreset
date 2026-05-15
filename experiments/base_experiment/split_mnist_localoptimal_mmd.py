"""
Split-MNIST: StreamingCoreset vs LocalOptimal — exact MMD vs RFF L2
==================================================================
Same five-task Split-MNIST stream as ``split_mnist_comparison.py`` (ResNet-18
embeddings, ``task_based_sample_stream``). Runs **two** RFF streamers that share
one ``RBFSampler``:

- ``StreamingCoreset`` (Franklin/Wolfe-style updates from ``streaming_coreset.py``)
- ``LocalOptimalStreamingCoreset`` (exact simplex QP + eviction from
  ``localoptimal_streamer.py``)

Metrics (aligned with ``base_experiment.py``):

- **Exact MMD**: ``compute_exact_rbf_mmd`` — RBF MMD in input space vs stream prefix.
- **RFF L2**: ``mmd_history`` / ``get_current_mmd`` — Euclidean distance in each
  method's RFF feature map.
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import numpy as np
from sklearn.kernel_approximation import RBFSampler

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
for _p in (_PROJECT_ROOT, _SCRIPT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dataloaders import load_dataset
from stream_builders import task_based_sample_stream

from base_experiment import compute_exact_rbf_mmd, plot_exact_mmd_multi, plot_l2_multi
from localoptimal_streamer import LocalOptimalStreamingCoreset
from streaming_coreset import StreamingCoreset

DATASET_NAME = "mnist"
N_CLASSES = 10
_SPLIT_MNIST_TASKS = [
    [0, 1],
    [2, 3],
    [4, 5],
    [6, 7],
    [8, 9],
]

DEFAULT_OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "split_mnist_localoptimal_mmd")


def _sanitize_key(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", name.strip()).strip("_").lower() or "run"


@dataclass
class SplitMnistTwoStreamerMMDConfig:
    seed: int = 42
    stream_length: int = 1000
    subset_size: int = 70_000
    M: int = 50
    rbf_gamma: float = 0.0001
    rff_dim: int = 1024
    streaming_coreset_k_iter: int = 1000
    exact_mmd_stride: int = 5
    output_dir: str = DEFAULT_OUTPUT_DIR


def load_split_mnist_stream(
    cfg: SplitMnistTwoStreamerMMDConfig,
) -> Tuple[np.ndarray, np.ndarray, List[int], Dict[int, int], int]:
    _, _, X_train, _, y_train, _ = load_dataset(
        DATASET_NAME,
        cfg.subset_size,
        cfg.subset_size,
        cfg.seed,
        embedding="resnet18",
        embed_dim=None,
        device="cpu",
    )
    X_train = X_train.astype(np.float64)
    y_train = y_train.astype(np.int64)
    d_in = X_train.shape[1]

    samples_per_class = cfg.stream_length // N_CLASSES
    X_stream, y_stream, class_change_steps, per_class_used = task_based_sample_stream(
        X_train, y_train, _SPLIT_MNIST_TASKS, samples_per_class, N_CLASSES, cfg.seed
    )
    return X_stream, y_stream, class_change_steps or [], per_class_used, d_in


def run_experiment(cfg: SplitMnistTwoStreamerMMDConfig) -> Dict[str, Any]:
    os.makedirs(cfg.output_dir, exist_ok=True)
    np.random.seed(cfg.seed)

    print(f"[data] Split-MNIST stream (ResNet-18), target length={cfg.stream_length} ...")
    X, y, class_change_steps, per_class_used, d_in = load_split_mnist_stream(cfg)
    n_total = len(X)
    print(f"  X={X.shape}, d_in={d_in}, task boundaries (1-based ends): {class_change_steps}")
    print(f"  per_class_used={per_class_used}")

    sampler = RBFSampler(
        gamma=cfg.rbf_gamma, n_components=cfg.rff_dim, random_state=cfg.seed + 1
    )
    sampler.fit(X[: min(500, n_total)])

    streamers: Dict[str, Any] = {
        "StreamingCoreset": StreamingCoreset(
            cfg.M,
            cfg.rff_dim,
            sampler,
            batch_size=1,
            K_iter=cfg.streaming_coreset_k_iter,
            verbose=False,
        ),
        "LocalOptimal": LocalOptimalStreamingCoreset(
            cfg.M, cfg.rff_dim, sampler, batch_size=1, verbose=False
        ),
    }
    names = list(streamers.keys())

    chk_steps: List[int] = []
    exact_hist: Dict[str, List[float]] = {n: [] for n in names}

    for t in range(n_total):
        x1, y1 = X[t : t + 1], y[t : t + 1]
        for name in names:
            streamers[name].process_batch(x1, y1, batch_idx=t)

        if t % cfg.exact_mmd_stride == 0 or t == n_total - 1:
            chk_steps.append(t)
            X_prefix = X[: t + 1]
            for name in names:
                st = streamers[name]
                exact_hist[name].append(
                    compute_exact_rbf_mmd(
                        X_prefix, st.buffer_X, st.buffer_weights, cfg.rbf_gamma
                    )
                )

    l2_hist: Dict[str, List[float]] = {n: list(streamers[n].mmd_history) for n in names}

    stem = os.path.join(cfg.output_dir, "split_mnist_streaming_vs_localoptimal")
    title_suffix = (
        f"  [Split-MNIST, N={n_total}, M={cfg.M}, D={cfg.rff_dim}, "
        f"γ={cfg.rbf_gamma}, K_iter={cfg.streaming_coreset_k_iter}]"
    )

    plot_l2_multi(
        l2_hist,
        stem + "_l2_rff_space.png",
        class_change_steps=class_change_steps,
        title="RFF L2 surrogate (per method)" + title_suffix,
    )
    print(f"  Saved: {stem}_l2_rff_space.png")

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

    save_kw: Dict[str, Any] = {
        "gamma": cfg.rbf_gamma,
        "M": cfg.M,
        "D": cfg.rff_dim,
        "n_total": n_total,
        "K_iter": cfg.streaming_coreset_k_iter,
        "exact_mmd_stride": cfg.exact_mmd_stride,
        "class_change_steps": np.array(class_change_steps, dtype=np.int64),
        "exact_mmd_steps": steps_arr,
    }
    for name in names:
        key = _sanitize_key(name)
        save_kw[f"l2_{key}"] = np.array(l2_hist[name], dtype=np.float64)
        save_kw[f"exact_mmd_{key}"] = exact_arr[name]
    npz_path = os.path.join(cfg.output_dir, "curves.npz")
    np.savez(npz_path, **save_kw)
    print(f"  Saved: {npz_path}")

    return {
        "steps": steps_arr,
        "exact_mmd": exact_hist,
        "l2_rff": l2_hist,
        "class_change_steps": class_change_steps,
        "per_class_used": per_class_used,
    }


def main() -> None:
    run_experiment(SplitMnistTwoStreamerMMDConfig())


if __name__ == "__main__":
    main()
