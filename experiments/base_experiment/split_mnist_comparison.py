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

Each task arrives as one contiguous segment in the stream (hard concept-drift
boundaries between tasks); within a task, the two digit classes are randomly
interleaved so both appear throughout that segment.

Setup
-----
- Input features : ResNet-18 embeddings (512-dim) of raw MNIST images.
- Bilevel surrogate : nn.Linear(512, 10) — a linear head on top of the
  frozen ResNet-18 feature extractor; gradients through this head supply
  the implicit-gradient signal for greedy coreset selection.
- Data is streamed in mini-batches of CHUNK_SIZE points so that the
  Bilevel merge-reduce framework receives meaningful batches.

Evaluation
----------
For each algorithm the final coreset (≤ M points) is evaluated under (i)
architecture-agnostic supervised classifiers — RF, SVC (RBF), Ridge, MLP —
and geometric KNN; and (ii) task-agnostic probes — K-Means + adjusted Rand
index on test embeddings (labels used only for scoring), and Isolation
Forest anomaly separation vs uniform noise (ROC-AUC).

StreamCore uses only feature geometry ($X$) for coreset construction;
gradient-based baselines are cross-entropy–driven on $y$.
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

from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import adjusted_rand_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.kernel_approximation import RBFSampler

import torch
import torch.nn as nn
import torch.optim as optim

from dataloaders import load_dataset
from stream_builders import task_based_sample_stream
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
CHUNK_SIZE     = 32             # mini-batch size for streaming
OUTPUT_DIR     = os.path.join(_PROJECT_ROOT, "split_mnist_results")

# Array of buffer sizes to test
M_VALUES       = [20, 50, 100]

SUPERVISED_MODELS = ["RandomForest", "SVC_RBF", "Ridge", "MLP"]
TASK_AGNOSTIC_MODELS = ["KNN", "KMeans_ARI", "Anomaly_AUC"]

# Split-MNIST stream: 5 sequential tasks; each task mixes two classes (shuffled).
_SPLIT_MNIST_TASKS = [
    [0, 1],
    [2, 3],
    [4, 5],
    [6, 7],
    [8, 9],
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

    X_buf = X_buf.astype(np.float64)
    y_buf = y_buf.astype(np.int64)
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
    """Train / fit downstream models on the coreset; return scalar scores."""

    rng = np.random.default_rng(seed + 913)

    results: Dict[str, float] = {}

    # --- Architecture agnosticity (supervised classification) ---
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

    # --- Task agnosticity (geometry / no label supervision in the probe) ---
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_buf, y_buf)
    results["KNN"] = float(np.mean(knn.predict(X_test) == y_test))

    # K-Means fitted on coreset features only; ARI compares cluster ids to held-out labels.
    kmeans = KMeans(n_clusters=n_classes, random_state=seed, n_init=10)
    try:
        kmeans.fit(X_buf)
    except TypeError:
        kmeans.fit(X_buf)
    test_clusters = kmeans.predict(X_test)
    results["KMeans_ARI"] = float(adjusted_rand_score(y_test, test_clusters))

    # C. Anomaly Detection (Isolation Forest)
    iso_forest = IsolationForest(random_state=seed)
    try:
        iso_forest.fit(X_buf, sample_weight=weights)
    except TypeError:
        iso_forest.fit(X_buf)

    # Make Hard Anomalies: Perturb the real test data by 1.5x its standard deviation
    feature_std = np.std(X_test, axis=0)
    # Ensure no zero standard deviations
    feature_std = np.where(feature_std == 0, 1e-6, feature_std) 
    
    noise_shift = rng.normal(loc=0.0, scale=1.5 * feature_std, size=X_test.shape)
    hard_anomalies = X_test + noise_shift

    scores_real = iso_forest.decision_function(X_test)
    scores_anomaly = iso_forest.decision_function(hard_anomalies)
    
    y_true_anomaly = np.concatenate([np.ones(len(X_test)), np.zeros(len(hard_anomalies))])
    y_scores_anomaly = np.concatenate([scores_real, scores_anomaly])
    try:
        results["Anomaly_AUC"] = float(roc_auc_score(y_true_anomaly, y_scores_anomaly))
    except ValueError:
        results["Anomaly_AUC"] = 0.5

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
    samples_per_class = STREAM_LENGTH // N_CLASSES
    X_stream, y_stream, class_change_steps, per_class_used = task_based_sample_stream(
        X_train, y_train, _SPLIT_MNIST_TASKS, samples_per_class, N_CLASSES, SEED
    )
    print(f"  Stream shape      : {X_stream.shape}")
    print(f"  Samples / class   : {samples_per_class}")
    print(f"  Per-class used    : {per_class_used}")
    print(f"  Task drift steps  : {class_change_steps}")

    # ---- Fit RFF sampler (shared by StreamCore) --------------------------
    sampler_rff = RBFSampler(gamma=RBF_GAMMA, n_components=RFF_DIM, random_state=SEED + 1)
    sampler_rff.fit(X_stream[:min(500, len(X_stream))])

    records: List[Dict] = []
    
    # Iterate through all M values
    for M in M_VALUES:
        print(f"\n{'='*50}")
        print(f" Running experiments for M = {M}")
        print(f"{'='*50}")
        
        # Instantiate fresh surrogates for each M to avoid state-bleed
        bilevel_surrogate = nn.Linear(embed_dim, N_CLASSES)
        ocs_surrogate     = nn.Linear(embed_dim, N_CLASSES)
        gss_surrogate     = nn.Linear(embed_dim, N_CLASSES)

        # ---- Instantiate streamers -------------------------------------------
        streamers: Dict[str, Any] = {
            "StreamCore": StreamingCoreset(
                M, RFF_DIM, sampler_rff, K_iter=10
            ),
            "Bilevel": BilevelStreamingCoreset(
                buffer_capacity=M,
                surrogate_model=bilevel_surrogate,
                optimizer=optim.Adam(bilevel_surrogate.parameters(), lr=0.005),
                criterion=nn.CrossEntropyLoss(),
                epochs=10,
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
                rehearsal_iters=10,
            ),
        }

        # ---- Stream data in mini-batches ------------------------------------
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

            print(f"  [{algo_name}] downstream evaluation (supervised + task-agnostic) ...")
            results = _eval_downstream(
                X_buf, y_buf, w_buf,
                X_val, y_val,
                n_classes=N_CLASSES,
                seed=SEED,
            )
            for model_name, score in results.items():
                records.append(dict(Algorithm=algo_name, M=M, Model=model_name, Test_Accuracy=score))
                print(f"    {model_name:18s}  {score:.4f}")

    # ---- Save CSV --------------------------------------------------------
    df = pd.DataFrame(records)
    df.to_csv(os.path.join(OUTPUT_DIR, "split_mnist_results.csv"), index=False)
    print("\n--- Results Saved to CSV ---")

    # ---- Generate LaTeX Table --------------------------------------------
    table_path = os.path.join(OUTPUT_DIR, "split_mnist_table.tex")
    _generate_latex_table(df, M_VALUES, table_path)

    # ---- Plot (Filters to M=100 so original layout doesn't break) --------
    df_m100 = df[df["M"] == 100].copy()
    if not df_m100.empty:
        bar_path = os.path.join(OUTPUT_DIR, "split_mnist_bar_chart.pdf")
        radar_path = os.path.join(OUTPUT_DIR, "split_mnist_radar_chart.pdf")
        
        _plot_results(df_m100, bar_path, title_m=100)
        _plot_radar_chart(df_m100, radar_path)

    print("\nDone. Results saved to:", OUTPUT_DIR)


# ---------------------------------------------------------------------------
# Visualisation & Output Helpers
# ---------------------------------------------------------------------------

import re
import pandas as pd

def _generate_latex_table(df: pd.DataFrame, m_values: list, output_path: str):
    """
    Takes a flat DataFrame of experiment results, pivots it into a multi-column
    layout, and formats it exactly to match the target IEEE/ACM booktabs style.
    """
    df_copy = df.copy()
    
    # 1. Clean up model names to match the paper's desired column headers
    rename_models = {
        "RandomForest": "RF", "SVC_RBF": "SVC", "Ridge": "Ridge", 
        "MLP": "MLP", "KNN": "KNN", "KMeans_ARI": "ARI", "Anomaly_AUC": "AUC"
    }
    df_copy["Model"] = df_copy["Model"].replace(rename_models)
    
    # 2. Pivot the table: Algorithms as rows; M and Model as nested columns
    pivot_df = df_copy.pivot(index="Algorithm", columns=["M", "Model"], values="Test_Accuracy")
    
    # 3. Enforce the exact column order
    model_order = ["RF", "SVC", "Ridge", "MLP", "KNN", "ARI", "AUC"]
    expected_cols = pd.MultiIndex.from_product([m_values, model_order])
    pivot_df = pivot_df.reindex(columns=expected_cols)
    
    # CRITICAL: Strip index and column names so pandas doesn't print them
    pivot_df.index.name = None
    pivot_df.columns.names = [None, None]
    
    # 4. Style: Bold the maximum value in each column
    styler = pivot_df.style \
        .format(precision=2) \
        .highlight_max(axis=0, props="textbf:--rwrap;") 
    
    # Generate the base LaTeX string with \toprule, \midrule, \bottomrule
    latex_str = styler.to_latex(hrules=True)
    
    # ---------------------------------------------------------
    # String Manipulations to match the exact target LaTeX shape
    # ---------------------------------------------------------
    
    num_models = len(model_order)
    
    # A. Remove vertical lines, group by M (e.g., l *{7}{c} *{7}{c} *{7}{c})
    tabular_fmt = "l" + "".join([f" *{{{num_models}}}{{c}}" for _ in m_values])
    latex_str = re.sub(r'\\begin\{tabular\}\{.*?\}', f'\\\\begin{{tabular}}{{{tabular_fmt}}}', latex_str)
    
    # B. Format the M headers and build the \cmidrule string
    cmidrules = []
    start_col = 2
    for m in m_values:
        # ROBUST REGEX: Catch any alignment pandas uses (l, r, or c) and force to 'c' with math mode
        pattern = r"\\multicolumn\{" + str(num_models) + r"\}\{[lrc]\}\{" + str(m) + r"\}"
        replacement = r"\\multicolumn{" + str(num_models) + r"}{c}{$M=" + str(m) + r"$}"
        latex_str = re.sub(pattern, replacement, latex_str)
        
        # Calculate start and end columns for \cmidrule
        end_col = start_col + num_models - 1
        cmidrules.append(f"\\cmidrule(lr){{{start_col}-{end_col}}}")
        start_col = end_col + 1
        
    cmidrule_str = "".join(cmidrules)
    
    # C. Inject \multirow and \cmidrule into the header lines
    lines = latex_str.split('\n')
    new_lines = []
    for line in lines:
        if "\\multicolumn" in line:
            # Replace the leading empty cell with \multirow
            line = re.sub(r'^\s*&\s*\\multicolumn', r'\\multirow{2}{*}{\\textbf{Method}} & \\multicolumn', line)
            new_lines.append(line)
            # Insert the horizontal dividers immediately after the top header row
            new_lines.append(cmidrule_str)
        else:
            new_lines.append(line)
            
    latex_str = "\n".join(new_lines)
    
    # D. Wrap in the table environment with \arraystretch and \resizebox
    wrapper = (
        "\\begin{table*}[htbp]\n"
        "\\centering\n"
        "\\renewcommand{\\arraystretch}{1.2}\n"
        "\\caption{Split-MNIST Streaming Coreset Comparison}\n"
        "\\label{tab:split_mnist_results}\n"
        "\\resizebox{\\linewidth}{!}{\n"
        "\\begin{tabular}"
    )
    latex_str = latex_str.replace("\\begin{tabular}", wrapper)
    
    footer = (
        "\\end{tabular}\n"
        "}\n"
        "\\end{table*}"
    )
    latex_str = latex_str.replace("\\end{tabular}", footer)
    
    # 5. Save to disk
    with open(output_path, "w") as f:
        f.write(latex_str)
    
    print(f"  Saved formatted LaTeX Table: {output_path}")


def _plot_results(df: pd.DataFrame, save_path: str, title_m: int) -> None:
    """Two grouped bar charts: supervised accuracy vs geometric / unsupervised probes."""
    algo_order = sorted(df["Algorithm"].unique())
    n_algos = len(algo_order)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # --- Panel A: supervised (uses labels during training only for these probes) ---
    ax0 = axes[0]
    model_order_sup = SUPERVISED_MODELS
    n_models = len(model_order_sup)
    bar_w = 0.8 / max(n_algos, 1)
    x = np.arange(n_models)
    for i, algo in enumerate(algo_order):
        vals = []
        for m in model_order_sup:
            row = df[(df["Algorithm"] == algo) & (df["Model"] == m)]
            vals.append(float(row["Test_Accuracy"].values[0]) if len(row) else 0.0)
        offset = (i - n_algos / 2 + 0.5) * bar_w
        bars = ax0.bar(x + offset, vals, bar_w * 0.9, label=algo, color=f"C{i}",
                       alpha=0.85, edgecolor="black", linewidth=0.6)
        for bar, v in zip(bars, vals):
            ax0.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax0.set_xticks(x)
    ax0.set_xticklabels(model_order_sup, fontsize=10, rotation=12, ha="right")
    ax0.set_ylabel("Accuracy", fontsize=11)
    ax0.set_title("Architecture agnostic (supervised)", fontsize=12)
    ax0.set_ylim(0, 1.05)
    ax0.yaxis.grid(True, alpha=0.3)
    ax0.legend(title="Coreset alg.", fontsize=9)
    ax0.set_axisbelow(True)

    # --- Panel B: task-agnostic (KNN geometric; clustering & anomaly without CE loss) ---
    ax1 = axes[1]
    model_order_task = TASK_AGNOSTIC_MODELS
    n_t = len(model_order_task)
    x1 = np.arange(n_t)
    for i, algo in enumerate(algo_order):
        vals = []
        for m in model_order_task:
            row = df[(df["Algorithm"] == algo) & (df["Model"] == m)]
            vals.append(float(row["Test_Accuracy"].values[0]) if len(row) else 0.0)
        offset = (i - n_algos / 2 + 0.5) * bar_w
        bars = ax1.bar(x1 + offset, vals, bar_w * 0.9, label=algo, color=f"C{i}",
                       alpha=0.85, edgecolor="black", linewidth=0.6)
        for bar, v in zip(bars, vals):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                     f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax1.set_xticks(x1)
    ax1.set_xticklabels(["KNN (acc.)", "K-Means (ARI)", "IsoForest (ROC-AUC)"],
                       fontsize=10, rotation=12, ha="right")
    ax1.set_ylabel("Score", fontsize=11)
    ax1.set_title("Task agnostic (geometry & unsupervised probes)", fontsize=12)
    y_min = min(0.0, df[df["Model"].isin(model_order_task)]["Test_Accuracy"].min())
    ax1.set_ylim(y_min - 0.06, min(1.08, df[df["Model"].isin(model_order_task)]["Test_Accuracy"].max() + 0.12))
    ax1.axhline(0.0, color="#999999", linewidth=0.8, linestyle=":")
    ax1.yaxis.grid(True, alpha=0.3)
    ax1.legend(title="Coreset alg.", fontsize=9)
    ax1.set_axisbelow(True)

    fig.suptitle(
        f"Split-MNIST coreset → downstream probes  (M={title_m}, ResNet-18 feats)",
        fontsize=13,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved Bar Chart: {save_path}")


def _plot_radar_chart(df: pd.DataFrame, save_path: str) -> None:
    """
    Creates a single, unified 7-axis radar chart combining both 
    supervised and task-agnostic downstream evaluations.
    """
    from math import pi

    # Combine all 7 metrics
    model_order = ["RandomForest", "SVC_RBF", "Ridge", "MLP", "KNN", "KMeans_ARI", "Anomaly_AUC"]
    algo_order  = sorted(df["Algorithm"].unique())
    
    N = len(model_order)
    
    # Calculate angles for each axis
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles_closed = angles + angles[:1]
    
    # Initialize the spider plot
    fig, ax = plt.subplots(figsize=(7.5, 7.5), subplot_kw=dict(polar=True))
    
    # Rotate the plot so the first axis is at the top
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    
    # Clean up the labels for the paper
    display_labels = {
        "RandomForest": "Random\nForest",
        "SVC_RBF": "SVC",
        "KMeans_ARI": "KMeans\n(ARI)",
        "Anomaly_AUC": "Anomaly\n(AUC)"
    }
    lbls = [display_labels.get(m, m) for m in model_order]
    
    # Draw axes and labels
    ax.set_xticks(angles)
    ax.set_xticklabels(lbls, color='black', size=11, weight='bold')
    
    # Lock the Y-axis from slightly below 0 (for ARI) to slightly above 1
    ax.set_ylim(-0.1, 1.05)
    
    # Paper-ready color palette
    color_map = {
        "StreamCore": "#D81B60",  # Bold Pink/Crimson
        "Bilevel": "#1E88E5",     # Cool Blue
        "OCS": "#FFC107",         # Amber/Yellow
        "GSS": "#7E57C2",         # Purple
    }
    fallback_colors = plt.cm.Dark2.colors
    
    for i, algo in enumerate(algo_order):
        vals = []
        for m in model_order:
            row = df[(df["Algorithm"] == algo) & (df["Model"] == m)]
            # Default to 0.0 if a metric is missing to prevent crashes
            vals.append(float(row["Test_Accuracy"].values[0]) if len(row) else 0.0)
            
        # Append first value to close the polygon
        vals_closed = vals + vals[:1]
        
        # Styling: Emphasize StreamCore visually
        is_ours = (algo == "StreamCore")
        color = color_map.get(algo, fallback_colors[i % len(fallback_colors)])
        linewidth = 3.5 if is_ours else 1.5
        alpha_fill = 0.20 if is_ours else 0.05
        zorder = 10 if is_ours else 5
        
        # Plot the outline
        ax.plot(angles_closed, vals_closed, linewidth=linewidth, linestyle='solid', 
                label=algo, color=color, zorder=zorder)
        # Fill the polygon
        ax.fill(angles_closed, vals_closed, color=color, alpha=alpha_fill, zorder=zorder)
        
    # Styling the grid
    ax.grid(color='#CCCCCC', linestyle='--', linewidth=1)
    ax.spines['polar'].set_color('#222222')
    ax.spines['polar'].set_linewidth(1.5)
    
    # Add an inner circle at y=0 to clearly show where metrics (like ARI) drop
    ax.plot(np.linspace(0, 2*pi, 100), np.zeros(100), color='#222222', linestyle=':', linewidth=1.5, zorder=1)
    
    # Add legend outside the plot
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), 
              fontsize=11, frameon=False, title="Algorithm")
              
    plt.title("Universal Task Generalization Profile", size=15, weight='bold', y=1.1)
    
    # Save as high-res PDF for LaTeX
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved Unified Radar Chart: {save_path}")


if __name__ == "__main__":
    main()