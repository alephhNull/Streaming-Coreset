"""
Task 3: Surrogate-Free Generalization Matrix
============================================
Evaluates downstream utility of streaming coreset algorithms across
fundamentally different ML architectures (RF, SVC, Ridge, MLP).

Run from the project root or from the experiments/base_experiment directory.
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

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import SVC
from sklearn.kernel_approximation import RBFSampler

import torch
import torch.nn as nn
import torch.optim as optim

from dataloaders import load_dataset
from stream_builders import manual_sample_stream, blocks_to_counts
from streaming_coreset import StreamingCoreset
from reservoir_rff_baseline import ReservoirRFFBaseline
from super_sampling import SuperSamplingCoreset
from bilevel_streamer import BilevelStreamingCoreset
from online_k_center import OnlineKCenterStreamingCoreset
from stream_fp_coreset import StreamFPCoreset, IdentityEmbedder
from ocs_coreset import OCSStreamingCoreset

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------
DATASET_NAME = "cifar10"
N_CLASSES = 10
SEED = 42
RBF_GAMMA = 0.00095
RFF_DIM = 1024
NUM_SPLITS = 1
STREAM_LENGTH = 2000
SUBSET_SIZE = 50_000
M_LIST = [10, 20, 50, 100, 200]
OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "generalization_results")

# Non-stationary stream: temporally-ordered contiguous blocks.
# Each tuple is (class_id, relative_weight). The stream arrives in this exact
# order — class 0 floods first, class 4/5 are rare micro-clusters in the middle,
# class 9 floods last — creating hard concept-drift boundaries for the streamers.
_STREAM_BLOCKS = [
    (0, 5),   # Block 01 — large flood
    (1, 4),   # Block 02
    (2, 3),   # Block 03
    (3, 2),   # Block 04
    (4, 1),   # Block 05 — rare micro-cluster
    (5, 1),   # Block 06 — rare micro-cluster
    (6, 2),   # Block 07
    (7, 3),   # Block 08
    (8, 4),   # Block 09
    (9, 5),   # Block 10 — large flood
]

# ---------------------------------------------------------------------------
# PyTorch MLP with weighted cross-entropy
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


def train_weighted_mlp(
    X_buf: np.ndarray,
    y_buf: np.ndarray,
    weights_buf: np.ndarray,
    n_classes: int,
    epochs: int = 200,
    lr: float = 1e-3,
    device: str = "cpu",
) -> _MLP:
    """Train a 2-layer PyTorch MLP with per-sample weights on the cross-entropy loss.

    Expects weights_buf to already sum to len(weights_buf) (i.e. average weight = 1),
    matching the sklearn convention used for RF / SVC / Ridge. With that scaling,
    .mean() over (loss * w) yields the correct weighted expectation without
    introducing a spurious factor-of-M shrinkage on the gradients.
    """
    X_t = torch.tensor(X_buf, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_buf, dtype=torch.long, device=device)
    # Weights already sum to M (passed in from the shared normalisation block).
    w_t = torch.tensor(weights_buf, dtype=torch.float32, device=device)

    model = _MLP(X_buf.shape[1], n_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # reduction='none' so we multiply by per-sample weights explicitly.
    ce_loss = nn.CrossEntropyLoss(reduction="none")

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(X_t)
        per_sample_loss = ce_loss(logits, y_t)
        # sum(w_t) == M, so .mean() == weighted_sum / M == correct weighted expectation.
        loss = (per_sample_loss * w_t).mean()
        loss.backward()
        optimizer.step()

    return model


def predict_mlp(model: _MLP, X: np.ndarray, device: str = "cpu") -> np.ndarray:
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32, device=device)
        preds = model(X_t).argmax(dim=1).cpu().numpy()
    return preds


# ---------------------------------------------------------------------------
# Helper: extract buffer arrays from a streamer
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


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation(
    X_stream: np.ndarray,
    y_stream: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    sampler_rff: RBFSampler,
    M_list: List[int],
    seed: int = 42,
    n_classes: int = 10,
) -> pd.DataFrame:

    records: List[Dict] = []
    model_names = ["RandomForest", "SVC_RBF", "Ridge", "GradientBoosting", "MLP"]

    for M in M_list:
        print(f"\n{'='*60}")
        print(f"  Buffer size M = {M}")
        print(f"{'='*60}")

        embed_dim = X_stream.shape[1]

        # ResNet-18 linear head surrogate for Bilevel (operates on pre-extracted
        # ResNet-18 embeddings, so a linear classifier is the natural surrogate).
        bilevel_surrogate = nn.Linear(embed_dim, n_classes)

        # Build fresh streamers for each M
        algo_streamers: Dict[str, Any] = {
            "StreamCore": StreamingCoreset(
                M, RFF_DIM, sampler_rff, batch_size=1, K_iter=1000
            ),
            "Bilevel": BilevelStreamingCoreset(
                buffer_capacity=M,
                surrogate_model=bilevel_surrogate,
                n_classes=n_classes,
                nr_slots=10,
            ),
            # ---- baselines commented out for StreamCore vs Bilevel comparison ----
            # "OnlineKCenter": OnlineKCenterStreamingCoreset(
            #     M, RFF_DIM, sampler_rff, batch_size=1
            # ),
            # "StreamFP": StreamFPCoreset(
            #     buffer_size=M,
            #     coreset_ratio=0.5,
            #     feature_extractor=fp_embedder,
            #     fingerprints=fp_tensor,
            # ),
            # "OCS": OCSStreamingCoreset(
            #     capacity=M,
            #     model=ocs_model,
            #     criterion=ocs_criterion,
            #     num_classes=n_classes,
            #     tau=1000.0,
            #     r2c_iter=100,
            #     device="cpu",
            #     selection_ratio=0.5,
            # ),
            # "Reservoir": ReservoirRFFBaseline(
            #     M, RFF_DIM, sampler_rff, seed=seed, batch_size=1
            # ),
            # "SuperSampling": SuperSamplingCoreset(
            #     M, RFF_DIM, sampler_rff, batch_size=1
            # ),
        }

        # --- Stream all data ---
        print(f"  Streaming {len(X_stream)} points ...")
        for t in range(len(X_stream)):
            for name, st in algo_streamers.items():
                st.process_batch(X_stream[t : t + 1], y_stream[t : t + 1], batch_idx=t)

        # --- Evaluate each algorithm ---
        for algo_name, streamer in algo_streamers.items():
            X_buf, y_buf, weights_buf = _extract_buffer(streamer)

            # ---------- Capacity assertion ----------
            assert len(X_buf) <= M, (
                f"[{algo_name}, M={M}] Buffer exceeds capacity: {len(X_buf)} > {M}"
            )

            # ---------- Simplex / weight normalisation assertion ----------
            w_sum = float(np.sum(weights_buf))
            # StreamCore normalises to 1; baselines return 1/k weights → also sum 1
            assert np.isclose(w_sum, 1.0, atol=1e-4) or np.isclose(w_sum, float(len(weights_buf)), atol=1e-4), (
                f"[{algo_name}, M={M}] Weights do not sum to 1 or M; got {w_sum}"
            )
            # Normalise to sum-1 for consistent downstream training
            if not np.isclose(w_sum, 1.0, atol=1e-4):
                weights_buf = weights_buf / w_sum

            n_unique = len(np.unique(y_buf))
            if n_unique < 2:
                print(
                    f"  [DEGENERATE] {algo_name} M={M}: only {n_unique} class(es) in buffer — "
                    "recording 10% accuracy for all models."
                )
                for mname in model_names:
                    records.append(
                        dict(Algorithm=algo_name, M=M, Model=mname, Test_Accuracy=0.10)
                    )
                continue

            # Clip weights to non-negative (PFW guarantees ≥0 but float rounding)
            weights_buf = np.clip(weights_buf, 0.0, None)
            # Scale so average weight == 1 (sum == M).
            # This keeps sklearn's regularisation strength (C, alpha, …) at face
            # value — passing weights that sum to 1 would implicitly divide C by M
            # and make SVC / Ridge massively under-regularised.
            weights_buf = (weights_buf / weights_buf.sum()) * len(weights_buf)

            print(f"  [{algo_name}] buffer={len(X_buf)}, classes={n_unique}, |w|₁={weights_buf.sum():.1f} (==M)")

            # ---- RandomForest ----
            rf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
            rf.fit(X_buf, y_buf, sample_weight=weights_buf)
            rf_acc = float(np.mean(rf.predict(X_test) == y_test))
            records.append(dict(Algorithm=algo_name, M=M, Model="RandomForest", Test_Accuracy=rf_acc))
            print(f"    RandomForest     acc = {rf_acc:.4f}")

            # ---- SVC (RBF kernel) ----
            svc = SVC(kernel="rbf", probability=False, random_state=seed)
            svc.fit(X_buf, y_buf, sample_weight=weights_buf)
            svc_acc = float(np.mean(svc.predict(X_test) == y_test))
            records.append(dict(Algorithm=algo_name, M=M, Model="SVC_RBF", Test_Accuracy=svc_acc))
            print(f"    SVC_RBF          acc = {svc_acc:.4f}")

            # ---- Ridge Classifier ----
            ridge = RidgeClassifier(alpha=1.0)
            ridge.fit(X_buf, y_buf, sample_weight=weights_buf)
            ridge_acc = float(np.mean(ridge.predict(X_test) == y_test))
            records.append(dict(Algorithm=algo_name, M=M, Model="Ridge", Test_Accuracy=ridge_acc))
            print(f"    Ridge            acc = {ridge_acc:.4f}")

            # ---- Gradient Boosting (HistGBM) ----
            # Axis-aligned splits — orthogonal to RKHS geometry, strengthens
            # the "surrogate-free" claim when StreamCore still wins here.
            hgb = HistGradientBoostingClassifier(random_state=seed)
            hgb.fit(X_buf, y_buf, sample_weight=weights_buf)
            hgb_acc = float(np.mean(hgb.predict(X_test) == y_test))
            records.append(dict(Algorithm=algo_name, M=M, Model="GradientBoosting", Test_Accuracy=hgb_acc))
            print(f"    GradientBoosting acc = {hgb_acc:.4f}")

            # ---- PyTorch MLP ----
            # weights_buf now sums to M (average == 1), so .mean() inside the
            # training loop gives the correct weighted expectation without
            # double-dividing by M.
            mlp = train_weighted_mlp(
                X_buf, y_buf, weights_buf, n_classes=n_classes, epochs=300, lr=1e-3
            )
            mlp_preds = predict_mlp(mlp, X_test)
            mlp_acc = float(np.mean(mlp_preds == y_test))
            records.append(dict(Algorithm=algo_name, M=M, Model="MLP", Test_Accuracy=mlp_acc))
            print(f"    MLP              acc = {mlp_acc:.4f}")

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_generalization_matrix(
    df: pd.DataFrame,
    M_target: int,
    save_path: str,
) -> None:
    """Grouped bar chart for a fixed M: X=Downstream Model, groups=Algorithm."""
    sub = df[df["M"] == M_target].copy()
    if sub.empty:
        print(f"  No results for M={M_target}, skipping plot.")
        return

    model_order = ["RandomForest", "SVC_RBF", "Ridge", "GradientBoosting", "MLP"]
    algo_order = sorted(sub["Algorithm"].unique())

    n_models = len(model_order)
    n_algos = len(algo_order)
    bar_width = 0.8 / n_algos
    x = np.arange(n_models)

    colors = [f"C{i}" for i in range(n_algos)]

    fig, ax = plt.subplots(figsize=(13, 6))
    for i, algo in enumerate(algo_order):
        vals = []
        for m in model_order:
            row = sub[(sub["Algorithm"] == algo) & (sub["Model"] == m)]
            vals.append(float(row["Test_Accuracy"].values[0]) if len(row) > 0 else 0.0)
        offset = (i - n_algos / 2 + 0.5) * bar_width
        bars = ax.bar(
            x + offset,
            vals,
            bar_width * 0.9,
            label=algo,
            color=colors[i],
            alpha=0.85,
            edgecolor="black",
            linewidth=0.6,
        )
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(model_order, fontsize=11)
    ax.set_ylabel("Test Accuracy", fontsize=12)
    ax.set_xlabel("Downstream Model", fontsize=12)
    ax.set_title(
        f"Surrogate-Free Generalization Matrix  (M={M_target}, CIFAR-10 ResNet-18 embeddings)",
        fontsize=12,
    )
    ax.legend(title="Algorithm", fontsize=10)
    ax.set_ylim(0, min(1.05, ax.get_ylim()[1] + 0.08))
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_accuracy_vs_M(df: pd.DataFrame, save_path: str) -> None:
    """Line plot of test accuracy vs M for each (algorithm, model) combination."""
    model_order = ["RandomForest", "SVC_RBF", "Ridge", "GradientBoosting", "MLP"]
    algo_order = sorted(df["Algorithm"].unique())

    fig, axes = plt.subplots(1, len(model_order), figsize=(20, 4), sharey=False)
    linestyles = ["-", "--", ":"]

    for ax, mname in zip(axes, model_order):
        for i, algo in enumerate(algo_order):
            sub = df[(df["Algorithm"] == algo) & (df["Model"] == mname)]
            sub = sub.sort_values("M")
            ax.plot(
                sub["M"],
                sub["Test_Accuracy"],
                marker="o",
                linestyle=linestyles[i % len(linestyles)],
                label=algo,
                color=f"C{i}",
            )
        ax.set_title(mname, fontsize=11)
        ax.set_xlabel("Buffer size M")
        ax.set_ylabel("Test Accuracy")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Test Accuracy vs Buffer Size M  (CIFAR-10 ResNet-18)", fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Radar (spider) chart
# ---------------------------------------------------------------------------

def plot_radar_chart(df: pd.DataFrame, M_target: int, save_path: str) -> None:
    """Spider / radar chart of Test Accuracy across downstream models for a fixed M."""
    sub = df[df["M"] == M_target].copy()
    if sub.empty:
        print(f"  No results for M={M_target}, skipping radar plot.")
        return

    model_order = ["RandomForest", "SVC_RBF", "Ridge", "GradientBoosting", "MLP"]
    algo_order = sorted(sub["Algorithm"].unique())

    num_vars = len(model_order)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors = [f"C{i}" for i in range(len(algo_order))]

    for i, algo in enumerate(algo_order):
        vals = []
        for m in model_order:
            row = sub[(sub["Algorithm"] == algo) & (sub["Model"] == m)]
            vals.append(float(row["Test_Accuracy"].values[0]) if len(row) > 0 else 0.0)
        vals += vals[:1]  # close the polygon

        ax.plot(angles, vals, color=colors[i], linewidth=2.5, label=algo)
        ax.fill(angles, vals, color=colors[i], alpha=0.1)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), model_order, fontsize=12, weight="bold")
    ax.set_rlabel_position(0)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=9)
    plt.title(
        f"Surrogate-Free Generalization  (M={M_target})",
        size=14, y=1.1, weight="bold",
    )
    plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), title="Algorithm")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.random.seed(SEED)

    # ---- Load data --------------------------------------------------------
    print(f"[data] Loading {DATASET_NAME} embeddings (ResNet-18) ...")
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

    print(f"  Train (stream) : {X_train.shape}, Test : {X_val.shape}")

    # ---- Build non-stationary block stream --------------------------------
    # blocks_to_counts converts relative weights → exact integer counts that
    # sum precisely to STREAM_LENGTH, then manual_sample_stream emits them as
    # contiguous temporal blocks (no interleaving), producing hard concept-drift
    # boundaries that stress-test memory-limited streaming algorithms.
    explicit_counts = blocks_to_counts(_STREAM_BLOCKS, STREAM_LENGTH)
    X_stream, y_stream, class_change_steps, per_class_used = manual_sample_stream(
        X_train, y_train, explicit_counts, N_CLASSES, SEED
    )
    print(f"  Stream shape   : {X_stream.shape}")
    print(f"  Block counts   : {explicit_counts}")
    print(f"  Drift steps    : {class_change_steps}")

    # ---- Fit RFF sampler (shared across all streamers) -------------------
    sampler_rff = RBFSampler(gamma=RBF_GAMMA, n_components=RFF_DIM, random_state=SEED + 12345)
    sampler_rff.fit(X_stream[: min(10, len(X_stream))])

    # ---- Run evaluation --------------------------------------------------
    df = run_evaluation(
        X_stream, y_stream,
        X_val, y_val,
        sampler_rff,
        M_LIST,
        seed=SEED,
        n_classes=N_CLASSES,
    )

    # ---- Save results ----------------------------------------------------
    csv_path = os.path.join(OUTPUT_DIR, "generalization_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n  Saved results CSV: {csv_path}")

    print("\n--- Full Results Table ---")
    print(df.to_string(index=False))

    # ---- Plots -----------------------------------------------------------
    bar_path = os.path.join(OUTPUT_DIR, "surrogate_free_generalization.pdf")
    plot_generalization_matrix(df, M_target=50, save_path=bar_path)

    radar_path = os.path.join(OUTPUT_DIR, "generalization_radar.pdf")
    plot_radar_chart(df, M_target=50, save_path=radar_path)

    acc_vs_m_path = os.path.join(OUTPUT_DIR, "accuracy_vs_M.pdf")
    plot_accuracy_vs_M(df, save_path=acc_vs_m_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
