from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

EXACT_MMD_COMPATIBLE = "exact_mmd_compatible"


def logits_to_per_class_counts(
    label_logits: Dict[int, float],
    n_classes: int,
    stream_length: int,
) -> Dict[int, int]:
    w = np.array([float(label_logits.get(c, 0.0)) for c in range(n_classes)], dtype=np.float64)
    if np.sum(w) <= 0:
        w[:] = 1.0
    w = w / np.sum(w)
    raw = stream_length * w
    counts = np.floor(raw).astype(np.int64)
    deficit = int(stream_length - int(np.sum(counts)))
    rem = raw - counts.astype(np.float64)
    for _ in range(deficit):
        j = int(np.argmax(rem))
        counts[j] += 1
        rem[j] -= 1.0
    return {c: int(counts[c]) for c in range(n_classes)}


def class_block_starts(per_class: Dict[int, int], n_classes: int) -> np.ndarray:
    starts = np.zeros(n_classes, dtype=np.int64)
    acc = 0
    for c in range(n_classes):
        starts[c] = acc
        acc += per_class[c]
    return starts


def interleaved_chunk_indices(
    per_class: Dict[int, int], n_classes: int, num_splits: int
) -> np.ndarray:
    if num_splits < 1:
        raise ValueError("num_splits must be >= 1")
    starts = class_block_starts(per_class, n_classes)
    by_class: List[List[np.ndarray]] = [[] for _ in range(n_classes)]
    for c in range(n_classes):
        n = per_class[c]
        base = starts[c]
        edges = np.linspace(0, n, num_splits + 1, dtype=np.int64)
        for s in range(num_splits):
            lo, hi = int(edges[s]), int(edges[s + 1])
            if lo < hi:
                by_class[c].append(np.arange(base + lo, base + hi, dtype=np.int64))
    max_rounds = max(len(by_class[c]) for c in range(n_classes))
    out: List[np.ndarray] = []
    for r in range(max_rounds):
        for c in range(n_classes):
            if r < len(by_class[c]):
                out.append(by_class[c][r])
    return np.concatenate(out) if out else np.array([], dtype=np.int64)


def label_transition_steps_1based(y: np.ndarray) -> List[int]:
    out: List[int] = []
    for t in range(1, len(y)):
        if y[t] != y[t - 1]:
            out.append(t + 1)
    return out


def stratified_sample_stream(
    X_all: np.ndarray,
    y_all: np.ndarray,
    per_class: Dict[int, int],
    n_classes: int,
    num_splits: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, Optional[List[int]], Dict[int, int]]:
    rng = np.random.RandomState(seed)
    idxs: List[np.ndarray] = []
    for c in range(n_classes):
        n_want = per_class[c]
        if n_want == 0:
            idxs.append(np.array([], dtype=np.int64))
            continue
        class_idx = np.where(y_all == c)[0]
        if len(class_idx) < n_want:
            raise ValueError(
                f"Not enough points for class {c}: need {n_want}, have {len(class_idx)}"
            )
        chosen = rng.choice(class_idx, size=n_want, replace=False)
        idxs.append(chosen)
    idxs_flat = np.concatenate(idxs)
    X_pool = X_all[idxs_flat]
    y_pool = y_all[idxs_flat]
    n_total = X_pool.shape[0]
    idx_mono = np.arange(n_total, dtype=np.int64)
    idx_order = interleaved_chunk_indices(per_class, n_classes, num_splits)
    assert np.array_equal(np.sort(idx_order), idx_mono)

    X = X_pool[idx_order]
    y = y_pool[idx_order]

    if num_splits == 1:
        counts = np.array([per_class[c] for c in range(n_classes)])
        class_change_steps = np.cumsum(counts)[:-1].tolist()
    else:
        class_change_steps = label_transition_steps_1based(y)

    return X, y, class_change_steps, per_class


def count_label_transitions(y: np.ndarray) -> int:
    if len(y) <= 1:
        return 0
    return int(np.sum(y[1:] != y[:-1]))


def infer_n_classes(label_logits: Dict[int, float]) -> int:
    if not label_logits:
        return 10
    return int(max(label_logits.keys()) + 1)
