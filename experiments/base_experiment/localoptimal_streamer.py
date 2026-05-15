import sys
import os
from typing import Any, List, Tuple

import numpy as np
from scipy.optimize import minimize

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from streamers.abstract_streamer import AbstractStreamingCoreset
from qpsolvers import solve_qp

class LocalOptimalStreamingCoreset(AbstractStreamingCoreset):
    def __init__(
        self,
        M: int,
        D: int,
        sampler: Any,
        batch_size: int = 1,
        verbose: bool = False,
        **kwargs  # Captures K_iter or others to maintain API compatibility
    ):
        self.M = M
        self.D = D
        self.sampler = sampler
        self.batch_size = batch_size
        self.verbose = verbose

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


    def _solve_exact_simplex_weights(self, Z: np.ndarray, mu: np.ndarray) -> Tuple[np.ndarray, float]:
        N = Z.shape[0]
        if N == 0: return np.array([]), float('inf')
        if N == 1: return np.array([1.0]), np.sum((Z[0] - mu)**2)

        P = Z @ Z.T
        # Add a tiny ridge for numerical stability so P is strictly positive definite
        P += np.eye(N) * 1e-9 
        q = -(Z @ mu)

        # Constraint: Sum of weights = 1 (A x = b)
        A = np.ones(N)
        b = np.array([1.0])

        # Constraint: 0 <= w_i <= 1 (lb <= x <= ub)
        lb = np.zeros(N)
        ub = np.ones(N)

        # Use 'quadprog' or 'osqp'
        w_opt = solve_qp(P, q, A=A, b=b, lb=lb, ub=ub, solver='quadprog')

        if w_opt is None:
            # Fallback if solver fails
            w_opt = np.ones(N) / N 

        w_opt = np.clip(w_opt, 0.0, 1.0)
        w_opt /= np.sum(w_opt)
        
        err = np.sum((Z.T @ w_opt - mu)**2)
        return w_opt, err

    def _process_point(self, x_raw, y_label, z_rff, batch_idx, local_idx):
        self.t += 1
        alpha = 1.0 / self.t
        self.mean_rff = (1.0 - alpha) * self.mean_rff + alpha * z_rff

        self.buffer_X.append(x_raw)
        self.buffer_y.append(y_label)
        self.buffer_provenance.append((batch_idx, local_idx))

        if len(self.buffer_Z) > 0:
            self.buffer_Z = np.vstack([self.buffer_Z, z_rff[np.newaxis, :]])
        else:
            self.buffer_Z = z_rff[np.newaxis, :]

        # --- Exact QP & Local Optimal Eviction Logic ---
        if len(self.buffer_Z) > self.M:
            best_evict_idx = -1
            best_weights = None
            min_err = float('inf')

            # Evaluate leaving out each point and optimizing exactly
            for i in range(len(self.buffer_Z)):
                Z_candidate = np.delete(self.buffer_Z, i, axis=0)
                w_opt, err = self._solve_exact_simplex_weights(Z_candidate, self.mean_rff)
                
                if err < min_err:
                    min_err = err
                    best_weights = w_opt
                    best_evict_idx = i

            # Evict the point that yields the lowest optimally-weighted error
            self.buffer_Z = np.delete(self.buffer_Z, best_evict_idx, axis=0)
            del self.buffer_X[best_evict_idx]
            del self.buffer_y[best_evict_idx]
            del self.buffer_provenance[best_evict_idx]
            self.buffer_weights = best_weights

        else:
            # Buffer not full yet, ensure current weights are perfectly optimal
            self.buffer_weights, _ = self._solve_exact_simplex_weights(self.buffer_Z, self.mean_rff)

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