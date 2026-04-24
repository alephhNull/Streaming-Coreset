import numpy as np
from typing import List, Tuple
from streamers.abstract_streamer import AbstractStreamingCoreset


class SuperSamplingCoreset(AbstractStreamingCoreset):
    """
    Implementation of:
    "Super-Sampling with a Reservoir" (Paige et al.)

    Uses exact nearest neighbor (O(MD)) instead of RP-tree.
    """

    def __init__(
        self,
        M: int,
        D: int,
        sampler,
        batch_size: int = 1,
        verbose: bool = False,
    ):
        self.M = M
        self.D = D
        self.sampler = sampler
        self.batch_size = batch_size
        self.verbose = verbose

        self.rff_dim = sampler.n_components

        # Reservoir
        self.buffer_X: List[np.ndarray] = []
        self.buffer_y: List[int] = []
        self.buffer_Z = np.empty((0, self.rff_dim))
        self.buffer_provenance: List[Tuple[int, int]] = []

        # Mean embedding μ̂_n
        self.mean_rff = np.zeros(self.rff_dim)
        self.t = 0

        self._finalized = False
        self.mmd_history: List[float] = []

    def _update_mean(self, z):
        """μ̂_n update (Eq. 18)"""
        self.t += 1
        alpha = 1.0 / self.t
        self.mean_rff = (1 - alpha) * self.mean_rff + alpha * z

    def _compute_current_nu(self):
        """ν̂_M = mean of buffer"""
        if len(self.buffer_Z) == 0:
            return np.zeros(self.rff_dim)
        return np.mean(self.buffer_Z, axis=0)

    @property
    def buffer_weights(self) -> np.ndarray:
        """
        Dynamically return uniform weights so external evaluation scripts 
        (like compute_exact_rbf_mmd) can read st.buffer_weights transparently.
        """
        n = len(self.buffer_Z)
        if n == 0:
            return np.empty(0, dtype=np.float64)
        return np.ones(n, dtype=np.float64) / n

    def _process_point(self, x_raw, y_label, z_rff, batch_idx, local_idx):
        # Update μ̂_n
        self._update_mean(z_rff)

        # If buffer not full → just insert
        if len(self.buffer_Z) < self.M:
            self.buffer_X.append(x_raw)
            self.buffer_y.append(y_label)
            self.buffer_Z = (
                np.vstack([self.buffer_Z, z_rff])
                if len(self.buffer_Z) > 0
                else z_rff[np.newaxis, :]
            )
            self.buffer_provenance.append((batch_idx, local_idx))

            self.mmd_history.append(self.get_current_mmd())
            return

        # ---- Super-sampling logic ----

        nu_prev = self._compute_current_nu()

        # φ* = φ(x_n) + M(ν_{n-1} - μ_n)   (Eq. 23)
        phi_star = z_rff + self.M * (nu_prev - self.mean_rff)

        # Find nearest neighbor in feature space (Eq. 24)
        dists = np.linalg.norm(self.buffer_Z - phi_star, axis=1)
        idx_drop = np.argmin(dists)

        # Candidate set: replace or not?
        # Try replacing → compute new ν
        Z_new = self.buffer_Z.copy()
        Z_new[idx_drop] = z_rff
        nu_new = np.mean(Z_new, axis=0)

        # Compute MMDs
        mmd_old = np.linalg.norm(self.mean_rff - nu_prev)
        mmd_new = np.linalg.norm(self.mean_rff - nu_new)

        # Accept only if improves MMD
        if mmd_new < mmd_old:
            self.buffer_Z[idx_drop] = z_rff
            self.buffer_X[idx_drop] = x_raw
            self.buffer_y[idx_drop] = y_label
            self.buffer_provenance[idx_drop] = (batch_idx, local_idx)

        self.mmd_history.append(self.get_current_mmd())

    def process_batch(self, X_batch_np, y_batch_np, batch_idx):
        if self._finalized:
            return

        Z_batch = self.sampler.transform(X_batch_np)

        for i in range(X_batch_np.shape[0]):
            self._process_point(
                X_batch_np[i],
                int(y_batch_np[i]),
                Z_batch[i],
                batch_idx,
                i,
            )

    def get_current_mmd(self):
        if len(self.buffer_Z) == 0:
            return 1.0
        nu = np.mean(self.buffer_Z, axis=0)
        return float(np.linalg.norm(self.mean_rff - nu))

    def get_final_coreset(self):
        self._finalized = True
        indices = np.array(
            [p[0] * self.batch_size + p[1] for p in self.buffer_provenance]
        )
        weights = np.ones(len(indices)) / len(indices)
        return indices, weights, self.buffer_provenance

    def print_coreset_provenance(self):
        indices, weights, prov = self.get_final_coreset()
        print("\n--- SuperSampling Coreset ---")
        for i, (idx, w) in enumerate(zip(indices, weights)):
            b, j = prov[i]
            print(f"Point {i}: Batch {b}, Idx {j}, Weight {w:.4f}")
        print("-----------------------------")
