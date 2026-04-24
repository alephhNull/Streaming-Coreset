import sys
import os
from typing import Any, List, Tuple

import numpy as np

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from streamers.abstract_streamer import AbstractStreamingCoreset


class OnlineKCenterStreamingCoreset(AbstractStreamingCoreset):
    def __init__(
        self,
        M: int,
        D: int,
        sampler: Any,
        batch_size: int = 1,
        verbose: bool = False,
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
        
        # Spatial radius for the Doubling Algorithm
        self.R = 0.0

    def _recluster(self):
        """
        Greedily merges points in the buffer that are within distance R
        of each other to reduce the coreset size.
        """
        n_points = len(self.buffer_Z)
        active_mask = np.ones(n_points, dtype=bool)
        
        new_Z = []
        new_weights = []
        new_X = []
        new_y = []
        new_prov = []
        
        for i in range(n_points):
            if not active_mask[i]:
                continue
            
            # Select point i as a center
            active_mask[i] = False
            new_Z.append(self.buffer_Z[i])
            w = self.buffer_weights[i]
            
            # Find all other active points within distance R
            dists = np.linalg.norm(self.buffer_Z - self.buffer_Z[i], axis=1)
            merge_indices = np.where(active_mask & (dists <= self.R))[0]
            
            if len(merge_indices) > 0:
                # Absorb their weights
                w += np.sum(self.buffer_weights[merge_indices])
                active_mask[merge_indices] = False
                
            new_weights.append(w)
            new_X.append(self.buffer_X[i])
            new_y.append(self.buffer_y[i])
            new_prov.append(self.buffer_provenance[i])
            
        self.buffer_Z = np.array(new_Z)
        self.buffer_weights = np.array(new_weights)
        self.buffer_X = new_X
        self.buffer_y = new_y
        self.buffer_provenance = new_prov

    def _process_point(self, x_raw, y_label, z_rff, batch_idx, local_idx):
        self.t += 1
        alpha = 1.0 / self.t
        
        # Track the true streaming mean
        self.mean_rff = (1.0 - alpha) * self.mean_rff + alpha * z_rff

        if len(self.buffer_Z) > 0:
            # Decay existing weights
            self.buffer_weights *= (1.0 - alpha)

        if len(self.buffer_Z) < self.M:
            # Buffer not full, append freely
            if len(self.buffer_Z) == 0:
                self.buffer_Z = z_rff[np.newaxis, :]
            else:
                self.buffer_Z = np.vstack([self.buffer_Z, z_rff[np.newaxis, :]])
            
            self.buffer_weights = np.append(self.buffer_weights, alpha)
            self.buffer_X.append(x_raw)
            self.buffer_y.append(y_label)
            self.buffer_provenance.append((batch_idx, local_idx))
        else:
            # Buffer is full, apply K-Center logic
            distances = np.linalg.norm(self.buffer_Z - z_rff, axis=1)
            closest_idx = int(np.argmin(distances))
            d_min = distances[closest_idx]

            # Initialize radius R based on the minimum pairwise distance of current buffer
            if self.R == 0.0:
                if len(self.buffer_Z) > 1:
                    pwd = np.linalg.norm(self.buffer_Z[:, np.newaxis] - self.buffer_Z, axis=2)
                    np.fill_diagonal(pwd, np.inf)
                    self.R = max(np.min(pwd), 1e-9)
                else:
                    self.R = 1e-9

            if d_min <= self.R:
                # Point is covered by an existing center, merge its weight
                self.buffer_weights[closest_idx] += alpha
            else:
                # Point is outside coverage, must be added
                self.buffer_Z = np.vstack([self.buffer_Z, z_rff[np.newaxis, :]])
                self.buffer_weights = np.append(self.buffer_weights, alpha)
                self.buffer_X.append(x_raw)
                self.buffer_y.append(y_label)
                self.buffer_provenance.append((batch_idx, local_idx))

                # Double the radius and re-cluster until buffer is within budget M
                while len(self.buffer_Z) > self.M:
                    self.R *= 2.0
                    self._recluster()

        # Re-normalize to avoid float drift over a long stream
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