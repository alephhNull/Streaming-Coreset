"""
Split-MNIST Streaming Coreset Comparison
=========================================
Compares StreamingCoreset (StreamCore) vs BilevelStreamingCoreset (Bilevel)
on the Split-MNIST continual learning benchmark.

Split-MNIST definition
----------------------
The standard 10-class MNIST dataset is split into 5 sequential binary tasks:
  Task 1: digits 0 & 1
  Task 2: digits 2 & 3
  Task 3: digits 4 & 5
  Task 4: digits 6 & 7
  Task 5: digits 8 & 9

Each task arrives as a contiguous block in the stream (hard concept-drift
boundaries), simulating a realistic online continual learning scenario.

Setup
-----
- Input features : ResNet-18 embeddings (512-dim) of raw MNIST images.
- Bilevel surrogate : nn.Linear(512, 10) — a linear head on top of the
  frozen ResNet-18 feature extractor; gradients through this head supply
  the implicit-gradient signal for greedy coreset selection.
- Buffer size M = 100.
- Data is streamed in mini-batches of CHUNK_SIZE points so that the
  Bilevel merge-reduce framework receives meaningful batches.

Evaluation
----------
For each algorithm the final coreset (≤ M points) is used to train four
downstream classifiers (RandomForest, SVC, Ridge, MLP):
  1. Overall 10-class accuracy on the full MNIST test split.
  2. Per-task binary accuracy (digits in {c, c+1} only).
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
for _p in (_PROJECT_ROOT, _SCRIPT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import SVC
from sklearn.kernel_approximation import RBFSampler

import torch
import torch.nn as nn
import torch.optim as optim

from dataloaders import load_dataset
from stream_builders import blocks_to_counts, manual_sample_stream
from streaming_coreset import StreamingCoreset
from bilevel_streamer import BilevelStreamingCoreset

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------
DATASET_NAME   = "mnist"
N_CLASSES      = 10
SEED           = 42
RBF_GAMMA      = 0.0001          # tuned for 512-dim ResNet-18 MNIST embeddings
RFF_DIM        = 1024
STREAM_LENGTH  = 2000           # 200 points per digit class
SUBSET_SIZE    = 70_000         # use all MNIST (train + test = 70 k)
M              = 50            # coreset / buffer size
CHUNK_SIZE     = 32            # mini-batch size for streaming
OUTPUT_DIR     = os.path.join(_PROJECT_ROOT, "split_mnist_results")

# Split-MNIST stream: 5 tasks, each with 2 classes arriving contiguously.
# Classes inside a task are interleaved so both digits appear together,
# creating within-task balance while preserving cross-task concept drift.
_SPLIT_MNIST_BLOCKS = [
    (0, 1),   # Task 1 — digit 0
    (1, 1),   # Task 1 — digit 1
    (2, 1),   # Task 2 — digit 2
    (3, 1),   # Task 2 — digit 3
    (4, 1),   # Task 3 — digit 4
    (5, 1),   # Task 3 — digit 5
    (6, 1),   # Task 4 — digit 6
    (7, 1),   # Task 4 — digit 7
    (8, 1),   # Task 5 — digit 8
    (9, 1),   # Task 5 — digit 9
]

# ---------------------------------------------------------------------------
# Lightweight PyTorch MLP for downstream evaluation
# ---------------------------------------------------------------------------

class _MLP(nn.Module):
    def __init__(self, in_dim: int, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _train_mlp(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    n_classes: int,
    epochs: int = 300,
    lr: float = 1e-3,
) -> _MLP:
    """Train a weighted MLP; weights must sum to len(weights) (average = 1)."""
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)
    w_t = torch.tensor(weights, dtype=torch.float32)
    model = _MLP(X.shape[1], n_classes)
    opt = optim.Adam(model.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss(reduction="none")
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = (ce(model(X_t), y_t) * w_t).mean()
        loss.backward()
        opt.step()
    return model


def _predict_mlp(model: _MLP, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float32)).argmax(1).numpy()


# ---------------------------------------------------------------------------
# Buffer extraction
# ---------------------------------------------------------------------------

def _extract_buffer(streamer) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (X_buf, y_buf, weights_buf) as numpy arrays."""
    X_buf = np.vstack(streamer.buffer_X) if len(streamer.buffer_X) > 0 else np.empty((0,))
    y_buf = np.array(
        streamer.buffer_y if hasattr(streamer, "buffer_y") else [],
        dtype=np.int64,
    ).flatten()
    weights_buf = np.array(streamer.buffer_weights, dtype=np.float64).flatten()
    return X_buf, y_buf, weights_buf


def _normalise_weights(weights: np.ndarray) -> np.ndarray:
    """Clip negatives, scale so sum == len(weights) (average weight = 1)."""
    w = np.clip(weights, 0.0, None)
    s = w.sum()
    if s > 1e-12:
        w = (w / s) * len(w)
    return w


# ---------------------------------------------------------------------------
# Downstream evaluation helpers
# ---------------------------------------------------------------------------

def _eval_downstream(
    X_buf: np.ndarray,
    y_buf: np.ndarray,
    weights: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_classes: int,
    seed: int,
) -> Dict[str, float]:
    """Train 4 classifiers on the coreset; return test accuracies."""
    results: Dict[str, float] = {}

    rf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
    rf.fit(X_buf, y_buf, sample_weight=weights)
    results["RandomForest"] = float(np.mean(rf.predict(X_test) == y_test))

    svc = SVC(kernel="rbf", random_state=seed)
    svc.fit(X_buf, y_buf, sample_weight=weights)
    results["SVC_RBF"] = float(np.mean(svc.predict(X_test) == y_test))

    ridge = RidgeClassifier(alpha=1.0)
    ridge.fit(X_buf, y_buf, sample_weight=weights)
    results["Ridge"] = float(np.mean(ridge.predict(X_test) == y_test))

    mlp = _train_mlp(X_buf, y_buf, weights, n_classes)
    results["MLP"] = float(np.mean(_predict_mlp(mlp, X_test) == y_test))

    return results


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.random.seed(SEED)

    # ---- Load MNIST with ResNet-18 embeddings ----------------------------
    print(f"[data] Loading {DATASET_NAME} (ResNet-18 embeddings) ...")
    _, _, X_train, X_val, y_train, y_val = load_dataset(
        DATASET_NAME,
        SUBSET_SIZE,
        SUBSET_SIZE,
        SEED,
        embedding="resnet18",
        embed_dim=None,
        device="cpu",
    )
    X_train = X_train.astype(np.float64)
    X_val   = X_val.astype(np.float64)
    y_train = y_train.astype(np.int64)
    y_val   = y_val.astype(np.int64)
    embed_dim = X_train.shape[1]   # 512 for ResNet-18
    print(f"  Train : {X_train.shape}  |  Test : {X_val.shape}  |  embed_dim={embed_dim}")

    # ---- Build Split-MNIST stream ----------------------------------------
    # blocks_to_counts maps relative weights → exact integer counts summing
    # to STREAM_LENGTH, preserving the block order (hard concept-drift).
    explicit_counts = blocks_to_counts(_SPLIT_MNIST_BLOCKS, STREAM_LENGTH)
    X_stream, y_stream, class_change_steps, per_class_used = manual_sample_stream(
        X_train, y_train, explicit_counts, N_CLASSES, SEED
    )
    print(f"  Stream shape  : {X_stream.shape}")
    print(f"  Block counts  : {explicit_counts}")
    print(f"  Drift steps   : {class_change_steps}")

    # ---- Fit RFF sampler (shared by StreamCore) --------------------------
    sampler_rff = RBFSampler(gamma=RBF_GAMMA, n_components=RFF_DIM, random_state=SEED + 1)
    sampler_rff.fit(X_stream[:min(500, len(X_stream))])

    # ---- ResNet-18 linear head surrogate for Bilevel ---------------------
    # The input to the surrogate is the 512-dim ResNet-18 feature vector.
    # Using a linear classifier as the surrogate captures the linear
    # separability geometry of the pre-trained embedding space, which is the
    # most relevant signal for coreset selection quality.
    surrogate = nn.Linear(embed_dim, N_CLASSES)

    # ---- Instantiate streamers -------------------------------------------
    streamers: Dict[str, Any] = {
        "StreamCore": StreamingCoreset(
            M, RFF_DIM, sampler_rff, batch_size=1, K_iter=1000
        ),
        "Bilevel": BilevelStreamingCoreset(
            buffer_capacity=M,
            surrogate_model=surrogate,
            n_classes=N_CLASSES,
            nr_slots=10,
        ),
    }

    # ---- Stream data in mini-batches ------------------------------------
    # Mini-batches give the Bilevel merge-reduce framework meaningful chunks
    # to compress (greedy selection on a single point is trivial).
    n_total = len(X_stream)
    n_batches = (n_total + CHUNK_SIZE - 1) // CHUNK_SIZE
    print(f"\n[stream] Streaming {n_total} points in {n_batches} chunks of {CHUNK_SIZE} ...")

    for batch_idx in range(n_batches):
        lo = batch_idx * CHUNK_SIZE
        hi = min(lo + CHUNK_SIZE, n_total)
        Xb = X_stream[lo:hi]
        yb = y_stream[lo:hi]

        for name, st in streamers.items():
            st.process_batch(Xb, yb, batch_idx=batch_idx)

        if (batch_idx + 1) % 5 == 0 or batch_idx == n_batches - 1:
            print(f"  chunk {batch_idx + 1}/{n_batches}  (pts {hi}/{n_total})")

    # ---- Extract final coresets and evaluate ----------------------------
    print("\n[eval] Extracting coresets ...")
    records: List[Dict] = []

    for algo_name, streamer in streamers.items():
        X_buf, y_buf, w_buf = _extract_buffer(streamer)

        assert len(X_buf) <= M, (
            f"[{algo_name}] Buffer size {len(X_buf)} exceeds M={M}"
        )

        n_unique = len(np.unique(y_buf))
        print(f"  [{algo_name}] buffer={len(X_buf)}, classes={n_unique}")

        if n_unique < 2:
            print(f"  [DEGENERATE] {algo_name}: only {n_unique} class(es), skipping.")
            continue

        # Normalise weights so average = 1 (sum = len)
        w_buf = _normalise_weights(w_buf)

        print(f"  [{algo_name}] 10-class evaluation on full test set ...")
        results = _eval_downstream(
            X_buf, y_buf, w_buf,
            X_val, y_val,
            n_classes=N_CLASSES,
            seed=SEED,
        )
        for model_name, acc in results.items():
            records.append(dict(Algorithm=algo_name, Model=model_name, Test_Accuracy=acc))
            print(f"    {model_name:15s}  {acc:.4f}")

    # ---- Save CSV --------------------------------------------------------
    df = pd.DataFrame(records)
    df.to_csv(os.path.join(OUTPUT_DIR, "split_mnist_results.csv"), index=False)
    print("\n--- Results ---")
    print(df.to_string(index=False))

    # ---- Plot ------------------------------------------------------------
    _plot_results(df, os.path.join(OUTPUT_DIR, "split_mnist_results.pdf"))

    print("\nDone. Results saved to:", OUTPUT_DIR)


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def _plot_results(df: pd.DataFrame, save_path: str) -> None:
    """Grouped bar chart: X = downstream model, groups = algorithm."""
    model_order = ["RandomForest", "SVC_RBF", "Ridge", "MLP"]
    algo_order  = sorted(df["Algorithm"].unique())
    n_models = len(model_order)
    n_algos  = len(algo_order)
    bar_w = 0.8 / n_algos
    x = np.arange(n_models)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, algo in enumerate(algo_order):
        vals = []
        for m in model_order:
            row = df[(df["Algorithm"] == algo) & (df["Model"] == m)]
            vals.append(float(row["Test_Accuracy"].values[0]) if len(row) else 0.0)
        offset = (i - n_algos / 2 + 0.5) * bar_w
        bars = ax.bar(x + offset, vals, bar_w * 0.9,
                      label=algo, color=f"C{i}", alpha=0.85,
                      edgecolor="black", linewidth=0.6)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(model_order, fontsize=11)
    ax.set_ylabel("Test Accuracy (10-class)")
    ax.set_xlabel("Downstream Model")
    ax.set_title(
        f"Split-MNIST: StreamCore vs Bilevel  (M={M}, ResNet-18 embeddings)",
        fontsize=12,
    )
    ax.legend(title="Algorithm")
    ax.set_ylim(0, min(1.05, ax.get_ylim()[1] + 0.08))
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


if __name__ == "__main__":
    main()
