"""
Split-MNIST Streaming Coreset Comparison
=========================================
Compares StreamingCoreset (StreamCore), BilevelStreamingCoreset (Bilevel),
OCSStreamingCoreset (OCS), and Gradient-Based Sample Selection (GSSStreamer)
from ``gss_coreset_streamer.py`` (actively trained surrogate) on Split-MNIST.

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
downstream classifiers (RandomForest, SVC, Ridge, MLP) on the full MNIST
test split (10-class accuracy).
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
from ocs_coreset import OCSStreamingCoreset
from gss_coreset_streamer import GSSStreamer

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
M              = 100           # coreset / buffer size
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
    """Return (X_buf, y_buf, weights_buf) as numpy arrays.

    Handles list-of-array buffers (StreamCore, Bilevel), tensor buffers (OCS),
    and list-of-tensor buffers (``gss_coreset_streamer.GSSStreamer``).
    """
    if isinstance(streamer, GSSStreamer):
        if not getattr(streamer, "buffer_X", None) or len(streamer.buffer_X) == 0:
            return (
                np.empty((0,)),
                np.array([], dtype=np.int64),
                np.array([], dtype=np.float64),
            )
        X_buf = (
            torch.stack(streamer.buffer_X).detach().cpu().numpy().astype(np.float64)
        )
        y_buf = (
            torch.stack(streamer.buffer_y).detach().cpu().numpy().astype(np.int64).flatten()
        )
        n = len(y_buf)
        weights = (
            np.full(n, 1.0 / n, dtype=np.float64)
            if n > 0
            else np.array([], dtype=np.float64)
        )
        return X_buf, y_buf, weights

    bX = streamer.buffer_X
    by = streamer.buffer_y if hasattr(streamer, "buffer_y") else []

    # OCS stores torch.Tensor directly; others use lists of numpy arrays
    if isinstance(bX, torch.Tensor):
        X_buf = bX.detach().cpu().numpy() if bX is not None and len(bX) > 0 else np.empty((0,))
    else:
        X_buf = np.vstack(bX) if len(bX) > 0 else np.empty((0,))

    if isinstance(by, torch.Tensor):
        y_buf = by.detach().cpu().numpy().flatten().astype(np.int64)
    else:
        y_buf = np.array(by, dtype=np.int64).flatten()

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
    torch.manual_seed(SEED)

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

    # ---- ResNet-18 linear head surrogate for Bilevel and OCS ------------
    # A linear classifier on 512-dim ResNet-18 features is the natural
    # surrogate: it captures linear separability geometry in the embedding
    # space and is cheap to differentiate through at each selection step.
    bilevel_surrogate = nn.Linear(embed_dim, N_CLASSES)
    ocs_surrogate     = nn.Linear(embed_dim, N_CLASSES)
    gss_surrogate = nn.Linear(embed_dim, N_CLASSES)

    # ---- Instantiate streamers -------------------------------------------
    streamers: Dict[str, Any] = {
        "StreamCore": StreamingCoreset(
            M, RFF_DIM, sampler_rff, batch_size=1, K_iter=1000
        ),
        "Bilevel": BilevelStreamingCoreset(
            buffer_capacity=M,
            surrogate_model=bilevel_surrogate,
            optimizer=optim.Adam(bilevel_surrogate.parameters(), lr=0.005),
            criterion=nn.CrossEntropyLoss(),
            epochs=5,
            n_classes=N_CLASSES,
            nr_slots=10,
        ),
        "OCS": OCSStreamingCoreset(
            buffer_capacity=M,
            surrogate_model=ocs_surrogate,
            criterion=nn.CrossEntropyLoss(),
            optimizer=optim.Adam(ocs_surrogate.parameters(), lr=0.005),
            device=torch.device("cpu"),
            tau=1000.0,
        ),
        "GSS": GSSStreamer(
            buffer_capacity=M,
            surrogate_model=gss_surrogate,
            criterion=nn.CrossEntropyLoss(),
            optimizer=optim.Adam(gss_surrogate.parameters(), lr=0.005),
            device=torch.device("cpu"),
            num_random_samples=10,
            rehearsal_iters=2,
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
    bar_path = os.path.join(OUTPUT_DIR, "split_mnist_bar_chart.pdf")
    radar_path = os.path.join(OUTPUT_DIR, "split_mnist_radar_chart.pdf")
    
    _plot_results(df, bar_path)
    _plot_radar_chart(df, radar_path)

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

    fig, ax = plt.subplots(figsize=(11, 5))
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
        f"Split-MNIST: StreamCore vs Bilevel vs OCS vs GSS  (M={M}, ResNet-18)",
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




def _plot_radar_chart(df: pd.DataFrame, save_path: str) -> None:
    """
    Creates a professional, paper-ready radar (spider) chart showing
    downstream generalization across different models.
    """
    from math import pi

    model_order = ["RandomForest", "SVC_RBF", "Ridge", "MLP"]
    algo_order  = sorted(df["Algorithm"].unique())
    
    # We must append the first model to the end of the list to 'close' the polygon
    categories = model_order + [model_order[0]]
    N = len(model_order)
    
    # Calculate angles for each axis
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    # Initialize the spider plot
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    
    # Rotate the plot so the first axis is at the top
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    
    # Draw one axe per variable + add labels
    plt.xticks(angles[:-1], model_order, color='black', size=12, weight='bold')
    
    # Dynamic Y-axis scaling to make the differences visible
    min_val = max(0.0, df["Test_Accuracy"].min() - 0.05)
    max_val = min(1.0, df["Test_Accuracy"].max() + 0.05)
    ax.set_ylim(min_val, max_val)
    
    # Define a professional, paper-ready color palette
    # Make StreamCore a bold crimson/magenta, and baselines cool blue/gray/amber
    color_map = {
        "StreamCore": "#D81B60",  # Bold Pink/Crimson
        "Bilevel": "#1E88E5",     # Cool Blue
        "OCS": "#FFC107",         # Amber/Yellow
        "GSS": "#7E57C2",         # Purple (distinct from others)
    }
    fallback_colors = plt.cm.Dark2.colors
    
    for i, algo in enumerate(algo_order):
        vals = []
        for m in model_order:
            row = df[(df["Algorithm"] == algo) & (df["Model"] == m)]
            vals.append(float(row["Test_Accuracy"].values[0]) if len(row) else 0.0)
            
        # Append first value to close the polygon
        vals += vals[:1]
        
        # Styling: Emphasize StreamCore visually
        is_ours = (algo == "StreamCore")
        color = color_map.get(algo, fallback_colors[i % len(fallback_colors)])
        linewidth = 3.0 if is_ours else 1.5
        alpha_fill = 0.20 if is_ours else 0.05
        zorder = 10 if is_ours else 5
        
        # Plot the outline
        ax.plot(angles, vals, linewidth=linewidth, linestyle='solid', 
                label=algo, color=color, zorder=zorder)
        # Fill the polygon
        ax.fill(angles, vals, color=color, alpha=alpha_fill, zorder=zorder)
        
    # Styling the grid
    ax.grid(color='#E0E0E0', linestyle='--', linewidth=1)
    ax.spines['polar'].set_color('#222222')
    ax.spines['polar'].set_linewidth(1.5)
    
    # Add legend outside the plot
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), 
              fontsize=11, frameon=False)
              
    plt.title("Downstream Generalization Profile", size=14, weight='bold', y=1.1)
    
    # Save as high-res PDF for LaTeX
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved Radar Chart: {save_path}")

if __name__ == "__main__":
    main()
