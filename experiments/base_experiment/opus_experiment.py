from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Union, Any
import json

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
)
from instrument import StreamingCoreset

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
    logits = dict(cfg.label_logits) if cfg.label_logits else {c: 1.0 for c in range(n_classes)}
    L = resolve_stream_length(cfg)
    per_class = logits_to_per_class_counts(logits, n_classes, L)

    print(f"[data] Loading {cfg.dataset_name} (embedded train)...")
    X_all, y_all = load_embedded_train_split(
        cfg.dataset_name, cfg.seed, subset_size=cfg.data_subset_size, device=cfg.embed_device
    )
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

    ws = [1, 300, 0, 0, 0, 0, 0, 0, 0, 0]
    logits = {i: float(ws[i]) for i in range(len(ws))}
    logits = {0: 10.0, 1: 20.0, 2: 10.0}
    cfg = BaseExperimentConfig(
        dataset_name="cifar10",
        label_logits=logits,
        stream_length=2000,
        num_splits=1,
        metrics=MetricsMode.L2_RFF,
        seed=42,
        output_dir=os.path.join(_PROJECT_ROOT, "snapshots_base_experiment", "instrument_streamer"),
    )
    stream = load_stratified_stream(cfg)

    M, D, gamma = 50, 1024, 0.001
    np.random.seed(cfg.seed)
    sampler_rff = RBFSampler(gamma=gamma, n_components=D, random_state=cfg.seed + 12345)
    sampler_rff.fit(stream.X[: min(10, stream.n_total)])

    instrument_streamer = StreamingCoreset(M, D, sampler_rff, batch_size=1, K_iter=100)

    streamers = {
        "RFF K_iter=100": instrument_streamer,
    }

    run_base_experiment(cfg, stream, streamers)

    max_points = cfg.stream_length

    instrument_streamer.finalize()
    logs = instrument_streamer.diag_logs

    # Convert to arrays for analysis
    ts = np.array([l["t"] for l in logs])
    labels = np.array([l["y_label"] for l in logs])
    pre_fw = np.array([l["pre_fw_err"] for l in logs])
    post_fw = np.array([l["post_fw_err"] for l in logs])
    post_evict = np.array([l["post_evict_err"] for l in logs])
    taus = np.array([l["evicted_weight_tau"] for l in logs])
    tau_ratios = np.array([l["tau_over_1mtau"] for l in logs])
    evict_labels = np.array([l["evicted_label"] for l in logs])
    evict_is_new = np.array([l["evicted_is_new"] for l in logs])
    evict_dist = np.array([l["evicted_dist_to_mu"] for l in logs])
    new_pt_w = np.array([l["new_point_weight_post_fw"] for l in logs])
    golden_exact = np.array([l["golden_exact_err"] for l in logs])
    golden_cs = np.array([l["golden_cs_bound"] for l in logs])
    cross_terms = np.array([l["cross_term"] for l in logs])
    fw_gaps = np.array([l["fw_gap"] for l in logs])
    fw_iters = np.array([l["fw_iters"] for l in logs])

    # --- Print summary tables ---
    print("=" * 80)
    print(f"DIAGNOSTIC EXPERIMENT: {cfg.dataset_name}, M={M}")
    print(f"Total points processed: {len(logs)}")
    print("=" * 80)

    # Identify concept transitions
    transitions = []
    for i in range(1, len(labels)):
        if labels[i] != labels[i - 1]:
            transitions.append(i)
    print(f"\nConcept transitions at t = {[ts[i] for i in transitions]}")

    # --- Table 1: Error dynamics around transitions ---
    print("\n--- TABLE 1: Error dynamics around concept transitions ---")
    window = 20
    for tr in transitions:
        lo = max(0, tr - window)
        hi = min(len(logs), tr + window)
        print(f"\n  Transition at t={ts[tr]} (label {labels[tr-1]} -> {labels[tr]}):")
        print(f"  {'t':>6} {'label':>5} {'pre_fw':>10} {'post_fw':>10} {'post_ev':>10} "
              f"{'tau':>10} {'tau/(1-t)':>10} {'ev_dist':>10} {'ev_lbl':>6} {'ev_new':>6} "
              f"{'gold_ex':>10} {'gold_cs':>10} {'cross':>12}")
        for i in range(lo, hi):
            l = logs[i]
            print(f"  {l['t']:>6d} {l['y_label']:>5d} {l['pre_fw_err']:>10.6f} "
                  f"{l['post_fw_err']:>10.6f} {l['post_evict_err']:>10.6f} "
                  f"{l['evicted_weight_tau']:>10.7f} {l['tau_over_1mtau']:>10.7f} "
                  f"{l['evicted_dist_to_mu']:>10.6f} {l['evicted_label']:>6d} "
                  f"{int(l['evicted_is_new']):>6d} "
                  f"{l['golden_exact_err']:>10.6f} {l['golden_cs_bound']:>10.6f} "
                  f"{l['cross_term']:>12.8f}")

    # --- Table 2: tau statistics by phase ---
    print("\n--- TABLE 2: tau statistics by phase ---")
    # Phase = which concept block we're in
    phase_starts = [0] + transitions + [len(logs)]
    for p in range(len(phase_starts) - 1):
        lo, hi = phase_starts[p], phase_starts[p + 1]
        phase_taus = taus[lo:hi]
        phase_taus_pos = phase_taus[phase_taus > 1e-15]
        if len(phase_taus_pos) > 0:
            print(f"  Phase {p} (t={ts[lo]}-{ts[hi-1]}, label={labels[lo]}): "
                  f"tau mean={np.mean(phase_taus_pos):.8f}, "
                  f"median={np.median(phase_taus_pos):.8f}, "
                  f"max={np.max(phase_taus_pos):.8f}, "
                  f"min={np.min(phase_taus_pos):.8f}, "
                  f"count={len(phase_taus_pos)}/{hi-lo}")

    # --- Table 3: Golden equation verification ---
    print("\n--- TABLE 3: Golden equation verification (post-buffer phase) ---")
    mask = (ts > M) & (golden_exact > -0.5)
    if np.any(mask):
        actual = post_evict[mask]
        predicted = golden_exact[mask]
        cs_bound = golden_cs[mask]
        rel_err = np.abs(actual - predicted) / (actual + 1e-15)
        print(f"  Golden exact vs actual: max_rel_err={np.max(rel_err):.2e}, "
              f"mean_rel_err={np.mean(rel_err):.2e}")
        print(f"  CS bound tight? actual/cs_bound: "
              f"mean={np.mean(actual/(cs_bound+1e-15)):.4f}, "
              f"min={np.min(actual/(cs_bound+1e-15)):.4f}")

    # --- Table 4: Recurrence verification ---
    print("\n--- TABLE 4: Recurrence e_t <= ((t-1)/t)*e_{t-1} + tau/(1-tau)*dist ---")
    mask2 = (ts > M + 1) & (taus > 1e-15)
    if np.any(mask2):
        idxs = np.where(mask2)[0]
        violations = 0
        max_slack = 0.0
        for i in idxs:
            if i == 0:
                continue
            t_val = ts[i]
            bound = ((t_val - 1) / t_val) * post_evict[i - 1] + tau_ratios[i] * evict_dist[i]
            actual_val = post_evict[i]
            slack = actual_val - bound
            if slack > 1e-9:
                violations += 1
            max_slack = max(max_slack, slack)
        print(f"  Violations: {violations}/{len(idxs)}, max_slack={max_slack:.2e}")

    # --- Table 5: Buffer composition over time ---
    print("\n--- TABLE 5: Buffer composition snapshots ---")
    snapshot_times = list(range(M, min(len(logs), max_points), max(1, (max_points - M) // 30)))
    for i in snapshot_times:
        l = logs[i]
        print(f"  t={l['t']:>6d} label={l['y_label']:>3d} comp={l['buffer_composition']} "
              f"err={l['post_evict_err']:.6f} tau={l['evicted_weight_tau']:.8f}")

    # --- Table 6: New point weight dynamics ---
    print("\n--- TABLE 6: New point FW weight at transitions ---")
    for tr in transitions:
        lo = max(0, tr - 5)
        hi = min(len(logs), tr + 40)
        print(f"\n  Transition at t={ts[tr]}:")
        print(f"  {'t':>6} {'label':>5} {'new_w':>12} {'tau':>12} {'evict_new':>10}")
        for i in range(lo, hi):
            l = logs[i]
            print(f"  {l['t']:>6d} {l['y_label']:>5d} {l['new_point_weight_post_fw']:>12.8f} "
                  f"{l['evicted_weight_tau']:>12.8f} {int(l['evicted_is_new']):>10d}")

    # --- Table 7: Cross-term sign analysis ---
    print("\n--- TABLE 7: Cross-term sign (negative = favorable) ---")
    mask3 = (ts > M) & (np.abs(cross_terms) > 1e-15)
    if np.any(mask3):
        ct = cross_terms[mask3]
        print(f"  Negative fraction: {np.mean(ct < 0):.4f}")
        print(f"  Mean cross_term: {np.mean(ct):.2e}")
        print(f"  When negative, mean magnitude: {np.mean(np.abs(ct[ct<0])):.2e}")
        print(f"  When positive, mean magnitude: {np.mean(np.abs(ct[ct>0])):.2e}")

    # --- Save raw logs for further analysis ---
    output_path = os.path.join(_PROJECT_ROOT, "experiments", "diagnostic_logs.json")
    # Convert numpy types for JSON
    serializable_logs = []
    for l in logs:
        sl = {}
        for k, v in l.items():
            if isinstance(v, (np.integer,)):
                sl[k] = int(v)
            elif isinstance(v, (np.floating,)):
                sl[k] = float(v)
            elif isinstance(v, np.bool_):
                sl[k] = bool(v)
            elif isinstance(v, dict):
                sl[k] = {str(kk): int(vv) for kk, vv in v.items()}
            else:
                sl[k] = v
        serializable_logs.append(sl)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(serializable_logs, f)
    print(f"\nRaw logs saved to {output_path}")

    return logs

if __name__ == "__main__":
    main()