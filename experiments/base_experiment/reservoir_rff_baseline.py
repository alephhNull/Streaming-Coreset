from __future__ import annotations

from typing import Any, List, Tuple

import numpy as np


class ReservoirRFFBaseline:
    """
    Uniform reservoir sample of size M in input space; metric uses the same RFF map
    as the given ``sampler`` (e.g. sklearn RBFSampler): running stream mean in feature
    space vs the mean RFF embedding of the reservoir.
    """

    def __init__(
        self,
        M: int,
        D: int,
        sampler: Any,
        seed: int = 0,
        batch_size: int = 1,
    ):
        self.M = M
        self.D = D
        self.sampler = sampler
        self.batch_size = batch_size
        self.rff_dim = int(sampler.n_components)
        self._rng = np.random.RandomState(seed)

        self.buffer_X: List[np.ndarray] = []
        self.buffer_provenance: List[Tuple[int, int]] = []  # Added to track indices
        self.t = 0
        self.mean_rff = np.zeros(self.rff_dim, dtype=np.float64)
        self.mmd_history: List[float] = []

    @property
    def buffer_weights(self) -> np.ndarray:
        k = len(self.buffer_X)
        if k == 0:
            return np.empty(0, dtype=np.float64)
        return np.ones(k, dtype=np.float64) / float(k)

    def process_batch(self, X_batch: np.ndarray, y_batch: np.ndarray, batch_idx: int) -> None:
        Z_batch = self.sampler.transform(X_batch)
        for i in range(X_batch.shape[0]):
            self.t += 1
            z = Z_batch[i]
            alpha = 1.0 / self.t
            self.mean_rff = (1.0 - alpha) * self.mean_rff + alpha * z
            x = X_batch[i]

            if len(self.buffer_X) < self.M:
                self.buffer_X.append(np.asarray(x, dtype=np.float64).copy())
                self.buffer_provenance.append((batch_idx, i))  # Track added item
            else:
                j = int(self._rng.randint(0, self.t))
                if j < self.M:
                    self.buffer_X[j] = np.asarray(x, dtype=np.float64).copy()
                    self.buffer_provenance[j] = (batch_idx, i)  # Track replaced item

            if len(self.buffer_X) == 0:
                self.mmd_history.append(1.0)
            else:
                Z_buf = self.sampler.transform(np.vstack(self.buffer_X))
                mu_buf = np.mean(Z_buf, axis=0)
                self.mmd_history.append(float(np.linalg.norm(self.mean_rff - mu_buf)))