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


def blocks_to_counts(
    stream_blocks: List[Tuple[int, float]], stream_length: int
) -> List[Tuple[int, int]]:
    """Converts a sequence of (class_idx, relative_weight) blocks to exact counts summing to stream_length."""
    if not stream_blocks:
        return []
    
    w = np.array([float(b[1]) for b in stream_blocks], dtype=np.float64)
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
    
    return [(stream_blocks[i][0], int(counts[i])) for i in range(len(stream_blocks))]


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


def manual_sample_stream(
    X_all: np.ndarray,
    y_all: np.ndarray,
    explicit_counts: List[Tuple[int, int]],
    n_classes: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, Optional[List[int]], Dict[int, int]]:
    """Builds a stream explicitly following the block sequence without reusing data points."""
    rng = np.random.RandomState(seed)
    
    # Shuffle available indices per class
    available_idxs = {}
    for c in range(n_classes):
        idx_c = np.where(y_all == c)[0]
        rng.shuffle(idx_c)
        available_idxs[c] = idx_c
        
    used_counters = {c: 0 for c in range(n_classes)}
    per_class_used = {c: 0 for c in range(n_classes)}
    
    X_list = []
    y_list = []
    class_change_steps = []
    current_step = 0
    
    for c, count in explicit_counts:
        if count <= 0:
            continue
        start = used_counters[c]
        end = start + count
        
        if end > len(available_idxs[c]):
            raise ValueError(
                f"Not enough points for class {c}: need {end}, have {len(available_idxs[c])}"
            )
            
        idxs = available_idxs[c][start:end]
        X_list.append(X_all[idxs])
        y_list.append(y_all[idxs])
        
        used_counters[c] = end
        per_class_used[c] += count
        
        current_step += count
        class_change_steps.append(current_step)
        
    # The last block boundary coincides with stream completion, so drop it from transition lines
    if class_change_steps:
        class_change_steps.pop()
        
    if not X_list:
        return np.array([]), np.array([]), [], per_class_used
        
    X = np.concatenate(X_list)
    y = np.concatenate(y_list)
    
    return X, y, class_change_steps, per_class_used


def task_based_sample_stream(
    X_all: np.ndarray,
    y_all: np.ndarray,
    task_groups: Sequence[Sequence[int]],
    samples_per_class: int,
    n_classes: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, Optional[List[int]], Dict[int, int]]:
    """Build a stream where each task gathers multiple classes, then shuffles them together.

    Indices are drawn without replacement from per-class pools (shuffled once), consuming
    ``samples_per_class`` from each class appearing in each task. Task boundaries are
    recorded in ``class_change_steps`` (1-based end indices of each task, excluding the
    final stream end), matching ``manual_sample_stream``.
    """
    rng = np.random.RandomState(seed)

    available_idxs: Dict[int, np.ndarray] = {}
    for c in range(n_classes):
        idx_c = np.where(y_all == c)[0]
        rng.shuffle(idx_c)
        available_idxs[c] = idx_c

    used_counters = {c: 0 for c in range(n_classes)}
    per_class_used = {c: 0 for c in range(n_classes)}

    X_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []
    class_change_steps: List[int] = []
    current_step = 0

    for task_classes in task_groups:
        task_X: List[np.ndarray] = []
        task_y: List[np.ndarray] = []
        for c in task_classes:
            start = used_counters[c]
            end = start + samples_per_class
            if end > len(available_idxs[c]):
                raise ValueError(
                    f"Not enough points for class {c}: need {end}, have {len(available_idxs[c])}"
                )
            idxs = available_idxs[c][start:end]
            task_X.append(X_all[idxs])
            task_y.append(y_all[idxs])
            used_counters[c] = end
            per_class_used[c] += samples_per_class

        task_X_arr = np.concatenate(task_X)
        task_y_arr = np.concatenate(task_y)
        shuffle_idx = rng.permutation(len(task_y_arr))
        X_list.append(task_X_arr[shuffle_idx])
        y_list.append(task_y_arr[shuffle_idx])

        current_step += len(task_y_arr)
        class_change_steps.append(current_step)

    if class_change_steps:
        class_change_steps.pop()

    if not X_list:
        return np.array([]), np.array([]), [], per_class_used

    X = np.concatenate(X_list)
    y = np.concatenate(y_list)
    return X, y, class_change_steps, per_class_used


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