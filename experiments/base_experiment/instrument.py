import sys
import os
from typing import Any, List, Tuple, Dict

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

        # === DIAGNOSTIC LOGS ===
        self.diag_logs: List[Dict] = []

    def _process_point(self, x_raw, y_label, z_rff, batch_idx, local_idx):
        self.t += 1
        t = self.t
        alpha = 1.0 / t
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

        # --- pre-FW error ---
        pre_fw_mu_hat = self.buffer_Z.T @ self.buffer_weights
        pre_fw_residual = self.mean_rff - pre_fw_mu_hat
        pre_fw_err = float(np.linalg.norm(pre_fw_residual))

        fw_gap_final = 0.0
        fw_iters_used = 0

        if len(self.buffer_Z) > 1:
            K_mat = self.buffer_Z @ self.buffer_Z.T
            linear_term = self.buffer_Z @ self.mean_rff

            for it in range(self.K_iter):
                grad = K_mat @ self.buffer_weights - linear_term
                idx_s = np.argmin(grad)

                active = np.where(self.buffer_weights > 1e-9)[0]
                if len(active) == 0:
                    fw_iters_used = it + 1
                    break
                idx_v = active[np.argmax(grad[active])]

                gap = grad[idx_v] - grad[idx_s]
                if gap < 1e-7:
                    fw_gap_final = float(gap)
                    fw_iters_used = it + 1
                    break

                hess = K_mat[idx_s, idx_s] - 2 * K_mat[idx_s, idx_v] + K_mat[idx_v, idx_v]
                gamma = gap / hess if hess > 1e-10 else 1.0
                gamma = min(gamma, self.buffer_weights[idx_v])

                self.buffer_weights[idx_s] += gamma
                self.buffer_weights[idx_v] -= gamma
                fw_gap_final = float(gap)
                fw_iters_used = it + 1

        # --- post-FW, pre-eviction quantities ---
        post_fw_mu_hat = self.buffer_Z.T @ self.buffer_weights
        post_fw_residual = self.mean_rff - post_fw_mu_hat
        post_fw_err = float(np.linalg.norm(post_fw_residual))

        # defaults
        evicted_label = -1
        evicted_weight = 0.0
        evicted_dist_to_mu = 0.0
        evicted_is_new = False
        new_point_weight_post_fw = float(self.buffer_weights[-1])
        post_evict_err = post_fw_err
        cross_term = 0.0
        golden_exact_err = -1.0
        golden_cs_bound = -1.0
        tau_over_1mtau = 0.0

        if len(self.buffer_Z) > self.M:
            evict = np.argmin(self.buffer_weights)
            evicted_weight = float(self.buffer_weights[evict])
            evicted_label = int(self.buffer_y[evict])
            z_evict = self.buffer_Z[evict].copy()
            evicted_dist_to_mu = float(np.linalg.norm(self.mean_rff - z_evict))
            evicted_is_new = (evict == len(self.buffer_Z) - 1)

            # --- compute golden equation terms BEFORE eviction ---
            tau = evicted_weight
            if tau < 1.0 - 1e-12:
                tau_over_1mtau = tau / (1.0 - tau)

                # cross term: <mu_t - hat_mu_t^(B), mu_t - z_j>
                diff_mu_zj = self.mean_rff - z_evict
                cross_term = float(np.dot(post_fw_residual, diff_mu_zj))

                # exact golden equation (algebraic identity):
                # ||mu - tilde_mu||^2 = (||r||^2 + 2*tau*<r, mu-z_j> + tau^2*||mu-z_j||^2) / (1-tau)^2
                # where r = post_fw_residual = mu - hat_mu^(B)
                numer = (post_fw_err**2
                         + 2.0 * tau * cross_term
                         + tau**2 * evicted_dist_to_mu**2)
                golden_exact_err = float(np.sqrt(max(0.0, numer))) / (1.0 - tau)

                # Cauchy-Schwarz upper bound:
                # e_t <= (e^(B) + tau*||mu-z_j||) / (1-tau)
                golden_cs_bound = (post_fw_err + tau * evicted_dist_to_mu) / (1.0 - tau)

            # --- perform eviction ---
            self.buffer_Z = np.delete(self.buffer_Z, evict, axis=0)
            self.buffer_weights = np.delete(self.buffer_weights, evict)
            del self.buffer_X[evict]
            del self.buffer_y[evict]
            del self.buffer_provenance[evict]

            s = np.sum(self.buffer_weights)
            if s > 1e-9:
                self.buffer_weights /= s

            post_evict_mu_hat = self.buffer_Z.T @ self.buffer_weights
            post_evict_err = float(np.linalg.norm(self.mean_rff - post_evict_mu_hat))

        self.mmd_history.append(post_evict_err)

        # --- buffer composition ---
        unique_labels, label_counts = np.unique(self.buffer_y, return_counts=True)
        buffer_composition = {int(l): int(c) for l, c in zip(unique_labels, label_counts)}

        # --- log everything ---
        log_entry = {
            "t": t,
            "y_label": int(y_label),
            "pre_fw_err": pre_fw_err,
            "fw_gap": fw_gap_final,
            "fw_iters": fw_iters_used,
            "post_fw_err": post_fw_err,
            "evicted_label": evicted_label,
            "evicted_weight_tau": evicted_weight,
            "tau_over_1mtau": tau_over_1mtau,
            "evicted_dist_to_mu": evicted_dist_to_mu,
            "evicted_is_new": evicted_is_new,
            "new_point_weight_post_fw": new_point_weight_post_fw,
            "cross_term": cross_term,
            "golden_exact_err": golden_exact_err,
            "golden_cs_bound": golden_cs_bound,
            "post_evict_err": post_evict_err,
            "buffer_composition": buffer_composition,
            "buffer_size": len(self.buffer_Z),
        }
        self.diag_logs.append(log_entry)

    def process_batch(self, X_batch, y_batch, batch_idx):
        Z_batch = self.sampler.transform(X_batch)
        for i in range(len(X_batch)):
            self.num_points_seen += 1
            self._process_point(X_batch[i], y_batch[i], Z_batch[i], batch_idx, i)

    def finalize(self):
        self._finalized = True
        return {
            "buffer_X": self.buffer_X,
            "buffer_y": self.buffer_y,
            "buffer_Z": self.buffer_Z,
            "buffer_weights": self.buffer_weights,
            "mean_rff": self.mean_rff,
            "mmd_history": self.mmd_history,
        }
    
    def get_final_coreset(self):
        pass

    def print_coreset_provenance(self):
        pass
