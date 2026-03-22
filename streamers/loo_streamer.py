import numpy as np
from typing import Optional, List, Tuple, Dict, Any
from scipy.optimize import minimize
from streamers.abstract_streamer import AbstractStreamingCoreset

# ==============================================================================
# 1. THE ORTHOGONAL SAMPLER (Ensuring Theory Meets Practice)
# ==============================================================================

class OrthogonalSampler:
    def __init__(self, d_in: int, n_components: int, gamma: float):
        self.d_in = d_in
        self.n_components = n_components
        self.gamma = gamma
        
        # 1. We need n_components features. 
        # Since each orthogonal block is size d_in x d_in, we stack blocks.
        nb_blocks = int(np.ceil(n_components / d_in))
        W_blocks = []
        
        for _ in range(nb_blocks):
            # Generate a random Gaussian matrix
            G = np.random.randn(d_in, d_in)
            # Compute QR to get an orthogonal matrix Q
            Q, _ = np.linalg.qr(G)
            W_blocks.append(Q)
            
        # 2. Stack and truncate to exactly n_components
        # W_ortho shape will be (n_components, d_in)
        W_ortho = np.vstack(W_blocks)[:n_components, :]
        
        # 3. Scale by sqrt(2*gamma) to approximate the RBF kernel correctly
        # This ensures the spectral frequencies match the Gaussian distribution
        self.W = W_ortho * np.sqrt(2 * gamma)
        
        # 4. Bias term must match n_components (D)
        self.b = np.random.uniform(0, 2 * np.pi, n_components)

    def transform(self, X: np.ndarray) -> np.ndarray:
        # X: (N, d_in)
        # self.W.T: (d_in, D)
        # Result: (N, D)
        projection = X @ self.W.T + self.b
        
        # Apply the cosine activation
        return np.sqrt(2.0 / self.n_components) * np.cos(projection)

# ==============================================================================
# 2. THE STREAMER
# ==============================================================================

class LOOStreamer(AbstractStreamingCoreset):
    def __init__(
        self,
        M: int,
        D: int,
        delta_drift_max: float,
        sampler,
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
        self.buffer_X = []
        self.buffer_y = []
        self.buffer_Z = np.empty((0, self.rff_dim), dtype=np.float64)
        self.buffer_weights = np.empty(0, dtype=np.float64)
        self.buffer_provenance = []

        self.mean_rff = np.zeros(self.rff_dim)
        self.num_points_seen = 0
        self.t = 0
        self._finalized = False
        self.mmd_history: List[float] = []

    def _solve_qp(self, Z: np.ndarray, mean_rff: np.ndarray) -> np.ndarray:
        """
        Solves the QP to find optimal simplex weights for the given set of features Z.
        Minimizes || Z^T w - mean_rff ||^2 subject to w >= 0, sum(w) = 1.
        """
        n = Z.shape[0]
        K = Z @ Z.T
        l = Z @ mean_rff
        
        # Objective: 0.5 * w^T K w - l^T w
        def obj(w):
            return 0.5 * w.T @ K @ w - l.T @ w
            
        def jac(w):
            return K @ w - l
            
        cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
        bounds = [(0.0, 1.0) for _ in range(n)]
        
        # Uniform initialization
        w0 = np.ones(n) / n
        
        res = minimize(obj, w0, method='SLSQP', jac=jac, bounds=bounds, constraints=cons)
        return res.x

    def _process_point(self, x_raw, y_label, z_rff, batch_idx, local_idx):
        self.t += 1
        alpha = 1.0 / self.t
        self.mean_rff = (1.0 - alpha) * self.mean_rff + alpha * z_rff

        self.buffer_X.append(x_raw)
        self.buffer_y.append(y_label)
        self.buffer_provenance.append((batch_idx, local_idx))

        if len(self.buffer_Z) > 0:
            self.buffer_Z = np.vstack([self.buffer_Z, z_rff[np.newaxis, :]])
            self.buffer_weights *= (1.0 - alpha)
            self.buffer_weights = np.append(self.buffer_weights, alpha) 
        else:
            self.buffer_Z = z_rff[np.newaxis, :]
            self.buffer_weights = np.array([1.0])

        # Standard PFW Optimization (runs while buffer is filling up)
        if 1 < len(self.buffer_Z) <= self.M:
            K_mat = self.buffer_Z @ self.buffer_Z.T
            linear_term = self.buffer_Z @ self.mean_rff
            
            for _ in range(self.K_iter):
                grad = K_mat @ self.buffer_weights - linear_term
                idx_s = np.argmin(grad)
                
                active = np.where(self.buffer_weights > 1e-9)[0]
                if len(active) == 0: break
                idx_v = active[np.argmax(grad[active])]
                
                gap = grad[idx_v] - grad[idx_s]
                if gap < 1e-7: break
                
                # Line Search
                hess = K_mat[idx_s, idx_s] - 2*K_mat[idx_s, idx_v] + K_mat[idx_v, idx_v]
                gamma = gap / hess if hess > 1e-10 else 1.0
                gamma = min(gamma, self.buffer_weights[idx_v])
                
                self.buffer_weights[idx_s] += gamma
                self.buffer_weights[idx_v] -= gamma

        # LOO Eviction and QP Re-optimization
        if len(self.buffer_Z) > self.M:
            best_error = float('inf')
            best_evict_idx = -1
            best_weights = None
            
            # Iterate through all M+1 points to find the best one to drop
            for i in range(len(self.buffer_Z)):
                # Temporarily fix weight w_i = 0 by removing point i
                Z_loo = np.delete(self.buffer_Z, i, axis=0)
                
                # Solve QP for the remaining M points
                w_opt = self._solve_qp(Z_loo, self.mean_rff)
                
                # Calculate exact MMD for this LOO configuration
                current_mmd = np.linalg.norm(self.mean_rff - (Z_loo.T @ w_opt))
                
                # Track the configuration with the lowest error
                if current_mmd < best_error:
                    best_error = current_mmd
                    best_evict_idx = i
                    best_weights = w_opt
            
            # Permanently execute the eviction that gave the lowest error
            self.buffer_Z = np.delete(self.buffer_Z, best_evict_idx, axis=0)
            self.buffer_weights = best_weights  # These already sum to 1 from the QP
            
            # Clean up tracking lists
            del self.buffer_X[best_evict_idx]
            del self.buffer_y[best_evict_idx]
            del self.buffer_provenance[best_evict_idx]

        self.mmd_history.append(self.get_current_mmd())

    def process_batch(self, X_batch, y_batch, batch_idx):
        if self._finalized: return
        Z_batch = self.sampler.transform(X_batch)
        for i in range(X_batch.shape[0]):
            self._process_point(X_batch[i], int(y_batch[i]), Z_batch[i], batch_idx, i)

    def get_current_mmd(self) -> float:
        if len(self.buffer_Z) == 0: return 1.0
        return np.linalg.norm(self.mean_rff - (self.buffer_Z.T @ self.buffer_weights))

    def get_final_coreset(self):
        self._finalized = True
        indices = np.array([p[0] * self.batch_size + p[1] for p in self.buffer_provenance])
        return indices, self.buffer_weights.copy(), self.buffer_provenance
    
    def print_coreset_provenance(self):
        pass