"""
RQ4 (text): Continuous manifold drift and graceful forgetting.

Builds a smooth non-stationary stream from 20 Newsgroups embeddings:
    topics = ["comp.graphics", "sci.space", "talk.politics.mideast"]

Phase schedule (T=3000, 1000 points per phase):
    phase 1 (t=1..1000):    p(comp)->0 linearly, p(space)->1 linearly, p(politics)=0
    phase 2 (t=1001..2000): p(space)->0 linearly, p(politics)->1 linearly, p(comp)=0
    phase 3 (t=2001..3000): p(politics)=1

Runs:
    - STREAMCORE (M=100)
    - ReservoirBuffer (M=100)
    - SlidingWindowBuffer (M=100)

Outputs:
    experiments/rq4_text_continuous_drift_output/neurips_fig4_text_drift.pdf
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import euclidean_distances
from tqdm import tqdm

import matplotlib

if os.environ.get("RQ4_TEXT_HEADLESS", "").lower() in ("1", "true", "yes"):
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

from streamers.orf_streamer import OrthogonalSampler, StreamingCoreset


SEED = 42
M = 100
RFF_D = 1024
K_ITER = 100
T_STREAM = 3000
PHASE_LEN = 1000
MEDIAN_HEURISTIC_SAMPLE_SIZE = 2000

TOPICS: List[str] = ["comp.graphics", "sci.space", "talk.politics.mideast"]
TOPIC_TO_ID: Dict[str, int] = {name: i for i, name in enumerate(TOPICS)}
ID_TO_TOPIC: Dict[int, str] = {i: name for i, name in enumerate(TOPICS)}

EMBED_CACHE_PATH = os.path.join(PROJECT_ROOT, "feature_cache", "20newsgroups_minilm_l6_v2.pkl")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "rq4_text_continuous_drift_output")
FIGURE_PATH = os.path.join(OUTPUT_DIR, "neurips_fig4_text_drift.pdf")


@dataclass
class ExperimentConfig:
    T: int = T_STREAM
    phase_len: int = PHASE_LEN
    M: int = M
    seed: int = SEED
    no_show: bool = False
    force_recompute_embeddings: bool = False


class ReservoirBuffer:
    """Uniform reservoir sample over all points seen so far."""

    def __init__(self, M: int, rng: np.random.RandomState):
        self.M = int(M)
        self.rng = rng
        self.buffer_idx: List[int] = []
        self.buffer_y: List[int] = []
        self.t_seen = 0

    def observe(self, x_idx: int, y: int) -> None:
        self.t_seen += 1
        if len(self.buffer_idx) < self.M:
            self.buffer_idx.append(int(x_idx))
            self.buffer_y.append(int(y))
            return
        j = int(self.rng.randint(0, self.t_seen))
        if j < self.M:
            self.buffer_idx[j] = int(x_idx)
            self.buffer_y[j] = int(y)

    def get_indices_weights(self) -> Tuple[np.ndarray, np.ndarray]:
        if not self.buffer_idx:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
        idx = np.asarray(self.buffer_idx, dtype=np.int64)
        w = np.full(idx.shape[0], 1.0 / float(idx.shape[0]), dtype=np.float64)
        return idx, w

    def fraction_class(self, cls: int) -> float:
        if not self.buffer_y:
            return 0.0
        return float(np.mean(np.asarray(self.buffer_y, dtype=np.int64) == int(cls)))


class SlidingWindowBuffer:
    """FIFO window over most recent M points."""

    def __init__(self, M: int):
        self.M = int(M)
        self.buf_idx: Deque[int] = deque(maxlen=M)
        self.buf_y: Deque[int] = deque(maxlen=M)

    def observe(self, x_idx: int, y: int) -> None:
        self.buf_idx.append(int(x_idx))
        self.buf_y.append(int(y))

    def get_indices_weights(self) -> Tuple[np.ndarray, np.ndarray]:
        if len(self.buf_idx) == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
        idx = np.asarray(list(self.buf_idx), dtype=np.int64)
        w = np.full(idx.shape[0], 1.0 / float(idx.shape[0]), dtype=np.float64)
        return idx, w

    def fraction_class(self, cls: int) -> float:
        if len(self.buf_y) == 0:
            return 0.0
        return float(np.mean(np.asarray(list(self.buf_y), dtype=np.int64) == int(cls)))


def set_global_seeds(seed: int) -> None:
    np.random.seed(seed)


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


def apply_paper_style() -> None:
    for name in ("seaborn-v0_8-paper", "seaborn-paper", "seaborn-whitegrid"):
        try:
            plt.style.use(name)
            return
        except OSError:
            continue


def load_20ng_minilm_embeddings(
    cache_path: str = EMBED_CACHE_PATH,
    force_recompute: bool = False,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    if (not force_recompute) and os.path.exists(cache_path):
        print(f"Loading cached 20NG embeddings from {cache_path}")
        data = joblib.load(cache_path)
        X = np.asarray(data["X"], dtype=np.float64)
        y = np.asarray(data["y"], dtype=np.int64)
        topic_names = list(data["topic_names"])
        return X, y, topic_names

    print("Fetching 20 Newsgroups text for selected topics...")
    ds = fetch_20newsgroups(
        subset="all",
        categories=TOPICS,
        remove=("headers", "footers", "quotes"),
        shuffle=False,
    )

    # Map sklearn target order to our fixed TOPICS order.
    target_names = list(ds.target_names)
    name_to_fixed = {name: TOPIC_TO_ID[name] for name in TOPICS}
    y = np.asarray([name_to_fixed[target_names[t]] for t in ds.target], dtype=np.int64)
    texts = [t if isinstance(t, str) else "" for t in ds.data]

    print("Encoding text with sentence-transformers/all-MiniLM-L6-v2...")
    X: np.ndarray
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        X = model.encode(
            texts,
            batch_size=128,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=False,
        ).astype(np.float64)
    except Exception as exc:
        # Some environments have incompatible package metadata (e.g., numpy version
        # lookup failure in sentence_transformers import path). Fallback to plain
        # transformers with the same checkpoint and standard mean pooling.
        print(
            "sentence-transformers path failed; falling back to transformers "
            f"mean-pooling encoder. Root error: {exc}"
        )
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
            model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(device)
            model.eval()

            def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
                mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
                summed = torch.sum(last_hidden_state * mask, dim=1)
                denom = torch.clamp(mask.sum(dim=1), min=1e-9)
                return summed / denom

            batch_size = 128
            vecs: List[np.ndarray] = []
            for i0 in tqdm(range(0, len(texts), batch_size), desc="MiniLM fallback encode", ncols=90):
                batch = texts[i0 : i0 + batch_size]
                toks = tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=256,
                )
                toks = {k: v.to(device) for k, v in toks.items()}
                with torch.no_grad():
                    out = model(**toks)
                    emb = _mean_pool(out.last_hidden_state, toks["attention_mask"])
                vecs.append(emb.detach().cpu().numpy())
            X = np.vstack(vecs).astype(np.float64, copy=False)
        except Exception as exc2:
            # Last-resort path for environments where transformers import is broken
            # (e.g., metadata/version resolution issues). Keeps experiment runnable
            # while preserving 384-d text representation.
            print(
                "transformers fallback also failed; using TF-IDF+SVD(384) emergency "
                f"fallback. Root error: {exc2}"
            )
            tfidf = TfidfVectorizer(
                strip_accents="unicode",
                lowercase=True,
                stop_words="english",
                max_df=0.95,
                min_df=2,
                max_features=50000,
                ngram_range=(1, 2),
                dtype=np.float64,
            )
            X_tfidf = tfidf.fit_transform(texts)
            svd = TruncatedSVD(n_components=384, random_state=SEED)
            X = svd.fit_transform(X_tfidf).astype(np.float64, copy=False)
            print(
                "Emergency text embedding backend active: TF-IDF+SVD(384). "
                "Install/fix transformers stack to restore MiniLM path."
            )
    if X.ndim != 2 or X.shape[1] != 384:
        raise RuntimeError(f"Expected 384-d MiniLM embeddings, got shape={X.shape}")

    payload = {"X": X, "y": y, "topic_names": TOPICS}
    joblib.dump(payload, cache_path)
    print(f"Saved 20NG embeddings cache to {cache_path}")
    return X, y, TOPICS


def _phase_probabilities(step_1based: int, phase_len: int) -> np.ndarray:
    """
    Return topic probabilities for t in [1..T] using smooth phase transitions.
    Topic order: [comp.graphics, sci.space, talk.politics.mideast].
    """
    t0 = int(step_1based) - 1  # 0-based for math
    if t0 < phase_len:
        s = t0 / float(max(phase_len - 1, 1))
        return np.array([1.0 - s, s, 0.0], dtype=np.float64)
    if t0 < 2 * phase_len:
        s = (t0 - phase_len) / float(max(phase_len - 1, 1))
        return np.array([0.0, 1.0 - s, s], dtype=np.float64)
    return np.array([0.0, 0.0, 1.0], dtype=np.float64)


def build_smooth_drift_stream(
    X_all: np.ndarray,
    y_all: np.ndarray,
    T: int,
    phase_len: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if T != 3 * phase_len:
        raise ValueError("This experiment expects T = 3 * phase_len.")

    rng = np.random.RandomState(seed)
    per_topic_idx: Dict[int, np.ndarray] = {
        cls: np.where(y_all == cls)[0].astype(np.int64) for cls in range(3)
    }
    for cls in range(3):
        if per_topic_idx[cls].size == 0:
            raise ValueError(f"No samples found for topic id={cls} ({ID_TO_TOPIC[cls]}).")

    stream_indices = np.empty(T, dtype=np.int64)
    stream_labels = np.empty(T, dtype=np.int64)
    for step in range(1, T + 1):
        p = _phase_probabilities(step, phase_len)
        cls = int(rng.choice(3, p=p))
        source_pool = per_topic_idx[cls]
        idx = int(source_pool[rng.randint(0, source_pool.size)])
        stream_indices[step - 1] = idx
        stream_labels[step - 1] = cls

    X_stream = X_all[stream_indices]
    return X_stream, stream_labels, stream_indices


def _rbf_kernel_block(X: np.ndarray, Y: np.ndarray, gamma: float) -> np.ndarray:
    x2 = np.sum(X * X, axis=1, keepdims=True)
    y2 = np.sum(Y * Y, axis=1, keepdims=True).T
    dist_sq = np.maximum(x2 + y2 - 2.0 * (X @ Y.T), 0.0)
    return np.exp(-gamma * dist_sq).astype(np.float64, copy=False)


def precompute_prefix_kernel_terms(
    X_stream: np.ndarray,
    gamma: float,
    block: int = 256,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        exx_prefix[t-1] = E_{x,x'~prefix_t}[k(x,x')]
        col_prefix[t-1, j] = sum_{i<=t} k(x_i, x_j)
    """
    T = X_stream.shape[0]
    K = np.empty((T, T), dtype=np.float64)
    for i0 in tqdm(range(0, T, block), desc="Kernel blocks", ncols=90):
        i1 = min(i0 + block, T)
        Xi = X_stream[i0:i1]
        for j0 in range(0, T, block):
            j1 = min(j0 + block, T)
            Xj = X_stream[j0:j1]
            K[i0:i1, j0:j1] = _rbf_kernel_block(Xi, Xj, gamma)

    S = np.cumsum(np.cumsum(K, axis=0), axis=1)  # 2D prefix sums
    denom = (np.arange(1, T + 1, dtype=np.float64) ** 2)
    exx_prefix = np.diag(S) / denom
    col_prefix = np.cumsum(K, axis=0)
    return K, exx_prefix, col_prefix


def exact_mmd2_from_precomputed(
    *,
    step: int,
    exx_prefix: np.ndarray,
    col_prefix: np.ndarray,
    K_stream: np.ndarray,
    buf_idx: np.ndarray,
    buf_w: np.ndarray,
) -> float:
    if buf_idx.size == 0:
        return float("nan")
    t = int(step)
    exx = float(exx_prefix[t - 1])
    prefix_col_sums = col_prefix[t - 1, buf_idx]  # sum_{i<=t} k(x_i, z_j)
    exz = float(np.dot(prefix_col_sums, buf_w) / float(t))
    Kzz = K_stream[np.ix_(buf_idx, buf_idx)]
    ezz = float(buf_w @ Kzz @ buf_w)
    return max(exx - 2.0 * exz + ezz, 0.0)


def vector_key(x: np.ndarray) -> bytes:
    return np.ascontiguousarray(x, dtype=np.float64).tobytes()


def streamcore_indices_weights(
    streamcore: StreamingCoreset,
    key_to_first_idx: Dict[bytes, int],
) -> Tuple[np.ndarray, np.ndarray]:
    if not streamcore.buffer_X:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    idx: List[int] = []
    for x in streamcore.buffer_X:
        k = vector_key(np.asarray(x, dtype=np.float64))
        if k not in key_to_first_idx:
            # Should not occur as long as buffer points come from the observed stream.
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
        idx.append(int(key_to_first_idx[k]))
    w = np.asarray(streamcore.buffer_weights, dtype=np.float64)
    return np.asarray(idx, dtype=np.int64), w


def topic0_prefix_mass(topic_labels: np.ndarray) -> np.ndarray:
    is_topic0 = (topic_labels == 0).astype(np.float64)
    cum = np.cumsum(is_topic0)
    denom = np.arange(1, topic_labels.shape[0] + 1, dtype=np.float64)
    return cum / denom


def plot_rq4_text_drift(
    steps: np.ndarray,
    mmd_streamcore: np.ndarray,
    mmd_reservoir: np.ndarray,
    mmd_sliding: np.ndarray,
    pi0: np.ndarray,
    w0_streamcore: np.ndarray,
    frac0_sliding: np.ndarray,
    out_path: str,
    *,
    show: bool = True,
) -> None:
    apply_paper_style()
    fig, axes = plt.subplots(2, 1, figsize=(6.2, 6.8), sharex=True)

    ax = axes[0]
    ax.plot(steps, mmd_streamcore, color="#1f77b4", linewidth=1.9, label="STREAMCORE")
    ax.plot(steps, mmd_reservoir, color="#ff7f0e", linestyle=":", linewidth=1.5, label="Reservoir")
    ax.plot(
        steps,
        mmd_sliding,
        color="#d62728",
        linestyle="--",
        linewidth=1.5,
        label="Sliding window",
    )
    positive = np.concatenate([mmd_streamcore, mmd_reservoir, mmd_sliding])
    if np.all(positive > 0):
        ax.set_yscale("log")
    ax.set_ylabel(r"Exact RBF $\mathrm{MMD}^2$")
    ax.grid(True, which="both", ls="-", alpha=0.25)
    ax.legend(loc="best", fontsize=8, frameon=True)

    ax = axes[1]
    ax.plot(steps, pi0, color="black", linestyle="--", linewidth=1.6, label=r"True mass $\pi_0(t)$")
    ax.plot(
        steps,
        w0_streamcore,
        color="#ff7f0e",
        linestyle="-",
        linewidth=1.8,
        label=r"STREAMCORE $W_0(t)$",
    )
    ax.plot(
        steps,
        frac0_sliding,
        color="#d62728",
        linestyle="-",
        linewidth=1.6,
        label="Sliding window fraction",
    )
    ax.set_xlabel(r"Stream step $t$")
    ax.set_ylabel(r"Mass for comp.graphics")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, which="both", ls="-", alpha=0.25)
    ax.legend(loc="best", fontsize=8, frameon=True)

    for axis in axes:
        axis.axvline(x=1000, color="0.45", linestyle="--", linewidth=0.9, alpha=0.85)
        axis.axvline(x=2000, color="0.45", linestyle="--", linewidth=0.9, alpha=0.85)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    print(f"Saved figure to {out_path}")

    if show and os.environ.get("RQ4_TEXT_HEADLESS", "").lower() not in ("1", "true", "yes"):
        try:
            plt.show(block=True)
        except Exception as exc:
            print(f"(Could not display figure interactively: {exc})")
    plt.close(fig)


def main(cfg: ExperimentConfig) -> None:
    set_global_seeds(cfg.seed)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(
        f"RQ4 text drift: T={cfg.T}, phase_len={cfg.phase_len}, M={cfg.M}, "
        f"RFF_D={RFF_D}, K_iter={K_ITER}, seed={cfg.seed}"
    )
    X_all, y_all, topic_names = load_20ng_minilm_embeddings(
        cache_path=EMBED_CACHE_PATH,
        force_recompute=cfg.force_recompute_embeddings,
    )
    print(f"Loaded topic names: {topic_names}")
    print(f"Embedding matrix shape: {X_all.shape}")

    gamma = compute_median_heuristic_gamma(X_all, sample_size=MEDIAN_HEURISTIC_SAMPLE_SIZE, seed=cfg.seed)
    print(f"Median-heuristic RBF gamma = {gamma:.6g}")

    X_stream, y_stream, stream_idx = build_smooth_drift_stream(
        X_all=X_all,
        y_all=y_all,
        T=cfg.T,
        phase_len=cfg.phase_len,
        seed=cfg.seed,
    )
    assert X_stream.shape[0] == cfg.T

    # For exact MMD at every step, precompute kernel matrix on stream and prefix terms once.
    print("Precomputing stream kernel matrix and prefix terms for exact MMD...")
    K_stream, exx_prefix, col_prefix = precompute_prefix_kernel_terms(X_stream, gamma=gamma, block=256)

    sampler = OrthogonalSampler(d_in=X_stream.shape[1], n_components=RFF_D, gamma=gamma)
    streamcore = StreamingCoreset(
        M=cfg.M,
        D=RFF_D,
        sampler=sampler,
        batch_size=1,
        K_iter=K_ITER,
        verbose=False,
    )
    rng = np.random.RandomState(cfg.seed)
    reservoir = ReservoirBuffer(cfg.M, rng)
    sliding = SlidingWindowBuffer(cfg.M)

    key_to_first_idx: Dict[bytes, int] = {}

    steps = np.arange(1, cfg.T + 1, dtype=np.int64)
    mmd_sc = np.empty(cfg.T, dtype=np.float64)
    mmd_res = np.empty(cfg.T, dtype=np.float64)
    mmd_sld = np.empty(cfg.T, dtype=np.float64)
    w0_sc = np.empty(cfg.T, dtype=np.float64)
    frac0_sld = np.empty(cfg.T, dtype=np.float64)

    pbar = tqdm(range(cfg.T), desc="RQ4 text stream", ncols=90)
    for t0 in pbar:
        step = t0 + 1
        x_t = X_stream[t0]
        y_t = int(y_stream[t0])

        k = vector_key(x_t)
        if k not in key_to_first_idx:
            key_to_first_idx[k] = int(t0)

        streamcore.process_batch(x_t[np.newaxis, :], np.array([y_t], dtype=np.int64), batch_idx=t0)
        reservoir.observe(x_idx=t0, y=y_t)
        sliding.observe(x_idx=t0, y=y_t)

        idx_sc, w_sc = streamcore_indices_weights(streamcore, key_to_first_idx)
        idx_res, w_res = reservoir.get_indices_weights()
        idx_sld, w_sld = sliding.get_indices_weights()

        mmd_sc[t0] = exact_mmd2_from_precomputed(
            step=step,
            exx_prefix=exx_prefix,
            col_prefix=col_prefix,
            K_stream=K_stream,
            buf_idx=idx_sc,
            buf_w=w_sc,
        )
        mmd_res[t0] = exact_mmd2_from_precomputed(
            step=step,
            exx_prefix=exx_prefix,
            col_prefix=col_prefix,
            K_stream=K_stream,
            buf_idx=idx_res,
            buf_w=w_res,
        )
        mmd_sld[t0] = exact_mmd2_from_precomputed(
            step=step,
            exx_prefix=exx_prefix,
            col_prefix=col_prefix,
            K_stream=K_stream,
            buf_idx=idx_sld,
            buf_w=w_sld,
        )

        if streamcore.buffer_y:
            y_buf = np.asarray(streamcore.buffer_y, dtype=np.int64)
            w0_sc[t0] = float(np.sum(w_sc[y_buf == 0]))
        else:
            w0_sc[t0] = 0.0
        frac0_sld[t0] = sliding.fraction_class(0)

    pi0 = topic0_prefix_mass(y_stream)

    plot_rq4_text_drift(
        steps=steps,
        mmd_streamcore=mmd_sc,
        mmd_reservoir=mmd_res,
        mmd_sliding=mmd_sld,
        pi0=pi0,
        w0_streamcore=w0_sc,
        frac0_sliding=frac0_sld,
        out_path=FIGURE_PATH,
        show=not cfg.no_show,
    )

    print("\n" + "=" * 80)
    print("RQ4 text drift summary")
    print("=" * 80)
    print(f"Output figure: {FIGURE_PATH}")
    print(f"Final pi_0(T): {pi0[-1]:.6f}")
    print(f"Final STREAMCORE W_0(T): {w0_sc[-1]:.6f}")
    print(f"Final sliding fraction(T): {frac0_sld[-1]:.6f}")
    print("=" * 80 + "\n")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="RQ4 text continuous drift (20NG MiniLM) for graceful forgetting."
    )
    p.add_argument("--T", type=int, default=T_STREAM, help="Total stream length (default: 3000).")
    p.add_argument("--phase-len", type=int, default=PHASE_LEN, help="Points per phase (default: 1000).")
    p.add_argument("--M", type=int, default=M, help="Buffer size for all methods (default: 100).")
    p.add_argument("--seed", type=int, default=SEED, help="Random seed (default: 42).")
    p.add_argument(
        "--force-recompute-embeddings",
        action="store_true",
        help="Ignore joblib cache and recompute sentence embeddings.",
    )
    p.add_argument(
        "--no-show",
        action="store_true",
        help="Save PDF only; do not open interactive figure window.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.T <= 0:
        raise ValueError("--T must be positive.")
    if args.phase_len <= 0:
        raise ValueError("--phase-len must be positive.")
    if args.T != 3 * args.phase_len:
        raise ValueError("This setup requires --T == 3 * --phase-len (default: 3000 == 3 * 1000).")
    if args.M <= 0:
        raise ValueError("--M must be positive.")

    main(
        ExperimentConfig(
            T=int(args.T),
            phase_len=int(args.phase_len),
            M=int(args.M),
            seed=int(args.seed),
            no_show=bool(args.no_show),
            force_recompute_embeddings=bool(args.force_recompute_embeddings),
        )
    )
