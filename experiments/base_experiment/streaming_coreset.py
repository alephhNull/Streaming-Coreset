import sys
import os
from typing import Any, List, Tuple

import numpy as np

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from streamers.abstract_streamer import AbstractStreamingCoreset


class StreamingCoreset(AbstractStreamingCoreset):
    def __init__(
        self,
        M: int,
        D: int,
        sampler: Any,
        batch_size: int = 1,
        K_iter: int = 100,
        verbose: bool = False,
    ):
        self.M = M
        self.D = D
        self.sampler = sampler
        self.batch_size = batch_size
        self.verbose = verbose
        self.K_iter = K_iter

        self.rff_dim = sampler.n_components
        self.buffer_X: List[np.ndarray] = []
        self.buffer_y: List[int] = []
        self.buffer_Z = np.empty((0, self.rff_dim), dtype=np.float64)
        self.buffer_weights = np.empty(0, dtype=np.float64)
        self.buffer_provenance: List[Tuple[int, int]] = []

        self.mean_rff = np.zeros(self.rff_dim)
        self.num_points_seen = 0
        self.t = 0
        self._finalized = False
        self.mmd_history: List[float] = []

    def _process_point(self, x_raw, y_label, z_rff, batch_idx, local_idx):
        self.t += 1
        alpha = 1.0 / self.t
        self.mean_rff = (1.0 - alpha) * self.mean_rff + alpha * z_rff

        self.buffer_X.append(x_raw)
        self.buffer_y.append(y_label)
        self.buffer_provenance.append((batch_idx, local_idx))

        if len(self.buffer_Z) > 0:
            self.buffer_Z = np.vstack([self.buffer_Z, z_rff[np.newaxis, :]])
            self.buffer_weights *= 1.0 - alpha
            self.buffer_weights = np.append(self.buffer_weights, alpha)
        else:
            self.buffer_Z = z_rff[np.newaxis, :]
            self.buffer_weights = np.array([1.0])

        if len(self.buffer_Z) > 1:
            K_mat = self.buffer_Z @ self.buffer_Z.T
            linear_term = self.buffer_Z @ self.mean_rff

            for _ in range(self.K_iter):
                grad = K_mat @ self.buffer_weights - linear_term
                idx_s = np.argmin(grad)

                active = np.where(self.buffer_weights > 1e-9)[0]
                if len(active) == 0:
                    break
                idx_v = active[np.argmax(grad[active])]

                gap = grad[idx_v] - grad[idx_s]
                if gap < 1e-7:
                    break

                hess = K_mat[idx_s, idx_s] - 2 * K_mat[idx_s, idx_v] + K_mat[idx_v, idx_v]
                gamma = gap / hess if hess > 1e-10 else 1.0
                gamma = min(gamma, self.buffer_weights[idx_v])

                self.buffer_weights[idx_s] += gamma
                self.buffer_weights[idx_v] -= gamma

        if len(self.buffer_Z) > self.M:
            evict = np.argmin(self.buffer_weights)
            self.buffer_Z = np.delete(self.buffer_Z, evict, axis=0)
            self.buffer_weights = np.delete(self.buffer_weights, evict)
            del self.buffer_X[evict]
            del self.buffer_y[evict]
            del self.buffer_provenance[evict]

            s = np.sum(self.buffer_weights)
            if s > 1e-9:
                self.buffer_weights /= s

        self.mmd_history.append(self.get_current_mmd())

    def process_batch(self, X_batch, y_batch, batch_idx):
        if self._finalized:
            return
        Z_batch = self.sampler.transform(X_batch)
        for i in range(X_batch.shape[0]):
            self._process_point(X_batch[i], int(y_batch[i]), Z_batch[i], batch_idx, i)

    def get_current_mmd(self) -> float:
        if len(self.buffer_Z) == 0:
            return 1.0
        return float(np.linalg.norm(self.mean_rff - (self.buffer_Z.T @ self.buffer_weights)))

    def get_final_coreset(self):
        self._finalized = True
        indices = np.array([p[0] * self.batch_size + p[1] for p in self.buffer_provenance])
        return indices, self.buffer_weights.copy(), self.buffer_provenance

    def print_coreset_provenance(self):
        pass
