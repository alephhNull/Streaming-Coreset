import sys
import os
from typing import Any, List, Tuple
import numpy as np

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from streamers.abstract_streamer import AbstractStreamingCoreset

class UnweightedStreamingCoreset(AbstractStreamingCoreset):
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
        self.buffer_provenance: List[Tuple[int, int]] = []

        self.mean_rff = np.zeros(self.rff_dim)
        self.num_points_seen = 0
        self.t = 0
        self._finalized = False
        self.mmd_history: List[float] = []

    def _process_point(self, x_raw, y_label, z_rff, batch_idx, local_idx):
        self.t += 1
        alpha = 1.0 / self.t
        
        # Track the true cumulative moving average \mu_t
        self.mean_rff = (1.0 - alpha) * self.mean_rff + alpha * z_rff

        self.buffer_X.append(x_raw)
        self.buffer_y.append(y_label)
        self.buffer_provenance.append((batch_idx, local_idx))

        if len(self.buffer_Z) > 0:
            self.buffer_Z = np.vstack([self.buffer_Z, z_rff[np.newaxis, :]])
        else:
            self.buffer_Z = z_rff[np.newaxis, :]

        # Evict if we exceed capacity M
        if len(self.buffer_Z) > self.M:
            # Evaluate the gradient at the current uniform weights
            buffer_mean = np.mean(self.buffer_Z, axis=0)
            error_vector = buffer_mean - self.mean_rff
            
            # Find the point that aligns most with the gradient of the loss
            # Removing it pulls the new buffer mean closer to the true stream mean
            alignments = self.buffer_Z @ error_vector
            evict_idx = np.argmax(alignments)
            
            self.buffer_Z = np.delete(self.buffer_Z, evict_idx, axis=0)
            del self.buffer_X[evict_idx]
            del self.buffer_y[evict_idx]
            del self.buffer_provenance[evict_idx]

        self.mmd_history.append(self.get_current_mmd())

    def process_batch(self, X_batch, y_batch, batch_idx):
        if self._finalized:
            return
        Z_batch = self.sampler.transform(X_batch)
        for i in range(X_batch.shape[0]):
            self._process_point(X_batch[i], int(y_batch[i]), Z_batch[i], batch_idx, i)

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

    def get_current_mmd(self) -> float:
        if len(self.buffer_Z) == 0:
            return 1.0
        buffer_mean = np.mean(self.buffer_Z, axis=0)
        return float(np.linalg.norm(self.mean_rff - buffer_mean))

    def get_final_coreset(self):
        self._finalized = True
        indices = np.array([p[0] * self.batch_size + p[1] for p in self.buffer_provenance])
        return indices, self.buffer_weights.copy(), self.buffer_provenance

    def print_coreset_provenance(self):
        pass