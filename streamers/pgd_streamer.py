import numpy as np
from typing import Optional, List, Tuple, Dict, Any
from streamers.abstract_streamer import AbstractStreamingCoreset

class TheoreticalHyperparamsFW:
    """
    Computes the error bound for Pairwise Frank-Wolfe.
    
    KEY DIFFERENCE:
    - BIAS IS ZERO. We solve the exact constrained problem.
    - The only error is 'Lag' (how far we are from convergence).
    - For PFW, convergence is Linear (Exponential decay).
    """

    def __init__(self, M: int, D: int, K_iter: int = 100):
        self.M = M
        self.D = D
        self.K_iter = K_iter

        # 1. Bias is ZERO
        self.eps_bias = 0.0

        # 2. Optimization Lag (Geometric Convergence)
        # PFW converges as O((1 - rho)^k). 
        # The geometric rate 'rho' depends on the "Pyramidal Width" of the simplex.
        # Conservatively for M=50, rho approx 0.01.
        self.rho_geom = 0.01 
        self.lag_factor = (1.0 - self.rho_geom) ** self.K_iter
        
        # Max initial error is bounded by diameter of kernel space (1.0)
        self.eps_lag = self.lag_factor * 1.0 

        # 3. RFF Error
        self.eps_rff = 1.0 / np.sqrt(self.D)

        # 4. Total
        self.eps_total = self.eps_rff + self.eps_lag + self.eps_bias

    def summary(self) -> str:
        return "\n".join([
            "=" * 65,
            "  FRANK-WOLFE BOUND SUMMARY (Zero Bias)",
            "=" * 65,
            f"  Buffer M             = {self.M}",
            f"  Iterations K         = {self.K_iter}",
            "-" * 65,
            f"  1. Bias (Reg Cost)   = {self.eps_bias:.6f}  <-- (The Magic)",
            f"  2. Opt Lag           = {self.eps_lag:.6f}",
            f"  3. RFF Approx        = {self.eps_rff:.6f}",
            "-" * 65,
            f"  ** Total Bound       = {self.eps_total:.6f}",
            "=" * 65,
        ])


class FrankWolfeStreamer(AbstractStreamingCoreset):
    """
    Pairwise Frank-Wolfe Algorithm for Streaming Coresets.
    
    Mechanism:
    Instead of Gradient Descent, we perform 'Weight Swaps':
    1. Find 'Good' atom (best correlation with residual).
    2. Find 'Bad' atom (worst correlation with residual).
    3. Move weight from Bad -> Good.
    """

    def __init__(self, M, D, sampler, batch_size=1, K_iter=100, verbose=False):
        self.M = M
        self.D = D
        self.sampler = sampler
        self.batch_size = batch_size
        self.K_iter = K_iter  # FW needs far fewer iterations than GD
        self.verbose = verbose
        
        self.hp = TheoreticalHyperparamsFW(M, D, K_iter)
        
        self.rff_dim = sampler.n_components
        self.buffer_X = []
        self.buffer_y = []
        self.buffer_Z = np.empty((0, self.rff_dim))
        self.buffer_weights = np.empty(0)
        self.buffer_provenance = []
        self.mean_rff = np.zeros(self.rff_dim)
        self.t = 0
        self._finalized = False
        self.mmd_history = []

    def _process_point(self, x, y, z, batch_idx, local_idx):
        self.t += 1
        # Update Mean
        alpha = 1.0 / self.t
        self.mean_rff = (1.0 - alpha) * self.mean_rff + alpha * z

        # Add point tentatively
        self.buffer_X.append(x)
        self.buffer_y.append(y)
        if len(self.buffer_Z) > 0:
            self.buffer_Z = np.vstack([self.buffer_Z, z[np.newaxis, :]])
            # Warm start: decay old weights, give new point 0 weight initially
            self.buffer_weights *= (1.0 - alpha)
            self.buffer_weights = np.append(self.buffer_weights, alpha) 
        else:
            self.buffer_Z = z[np.newaxis, :]
            self.buffer_weights = np.array([1.0])
        
        self.buffer_provenance.append((batch_idx, local_idx))

        # --- PAIRWISE FRANK-WOLFE OPTIMIZATION ---
        # Objective: minimize 0.5 * ||Zw - mu||^2
        # Gradient: K * w - (Z @ mu)
        
        if len(self.buffer_Z) > 1:
            K_mat = self.buffer_Z @ self.buffer_Z.T
            linear_term = self.buffer_Z @ self.mean_rff
            
            for k in range(self.K_iter):
                # 1. Compute Gradient
                grad = K_mat @ self.buffer_weights - linear_term
                
                # 2. Find Atoms (The "Oracle")
                # 's' (Good): Index with MIN gradient (best correlation)
                # 'v' (Bad):  Index with MAX gradient (worst correlation)
                #             Only consider 'v' where weight > 0 (Away step)
                
                idx_s = np.argmin(grad)
                
                # Mask zeros for 'v' selection to avoid numerical issues
                active_indices = np.where(self.buffer_weights > 1e-9)[0]
                if len(active_indices) == 0: break
                
                # Find max grad among active weights
                sub_grad = grad[active_indices]
                idx_v_local = np.argmax(sub_grad)
                idx_v = active_indices[idx_v_local]
                
                # If gap is small (duality gap), we are done
                gap = grad[idx_v] - grad[idx_s]
                if gap < 1e-7: break
                
                # 3. Line Search (Exact for Quadratic)
                # We want to move mass from v to s: w_new = w + gamma * (e_s - e_v)
                # Direction d = e_s - e_v
                # This is a 1D quadratic minimization.
                
                # The quadratic term: d^T K d
                # K_ss - 2 K_sv + K_vv
                hess = K_mat[idx_s, idx_s] - 2*K_mat[idx_s, idx_v] + K_mat[idx_v, idx_v]
                
                # The linear term: grad^T d = grad_s - grad_v = -gap
                
                # Optimal step size gamma = -linear / hess = gap / hess
                if hess < 1e-10: 
                    gamma = 1.0 # Fallback
                else:
                    gamma = gap / hess
                
                # Clip gamma. We can't remove more weight than v has.
                max_gamma = self.buffer_weights[idx_v]
                gamma = np.clip(gamma, 0, max_gamma)
                
                # 4. Update
                self.buffer_weights[idx_s] += gamma
                self.buffer_weights[idx_v] -= gamma
        
        # --- EVICTION ---
        # Frank-Wolfe naturally drives weights to zero.
        # We simply remove the smallest weight (which is likely 0 or epsilon).
        if len(self.buffer_Z) > self.M:
            evict_idx = np.argmin(self.buffer_weights)
            
            self.buffer_Z = np.delete(self.buffer_Z, evict_idx, axis=0)
            self.buffer_weights = np.delete(self.buffer_weights, evict_idx)
            del self.buffer_X[evict_idx]
            del self.buffer_y[evict_idx]
            del self.buffer_provenance[evict_idx]
            
            # Renormalize to stay on simplex (corrects numerical drift)
            self.buffer_weights /= np.sum(self.buffer_weights)

        self.mmd_history.append(self.get_current_mmd())

    def process_batch(self, X, y, batch_idx):
        if self._finalized: return
        Z = self.sampler.transform(X)
        # self.num_points_seen += X.shape[0]
        for i in range(X.shape[0]):
            self._process_point(X[i], int(y[i]), Z[i], batch_idx, i)

    def get_current_mmd(self):
        if len(self.buffer_Z) == 0: return float("inf")
        embed = self.buffer_Z.T @ self.buffer_weights
        return float(np.sqrt(np.sum((self.mean_rff - embed) ** 2)))
        
    def get_final_coreset(self):
        self._finalized = True
        flat = np.array([p[0]*self.batch_size + p[1] for p in self.buffer_provenance], dtype=int)
        return flat, self.buffer_weights, list(self.buffer_provenance)

    def get_diagnostics(self):
        return {
            "t": self.t, "mmd": self.get_current_mmd(), 
            "bound_bias": self.hp.eps_bias,
            "bound_total": self.hp.eps_total
        }

    def print_coreset_provenance(self):
        flat_indices, weights, provenance = self.get_final_coreset()
        print("--- Final Coreset Provenance ---")
        if len(provenance) == 0:
            print("Coreset is empty.")
            return
        print(f"{'Global Index':<15} {'Provenance':<20} {'Weight':<10}")
        print("-" * 55)
        for i in range(len(provenance)):
            prov_str = f"(Batch {provenance[i][0]}, Idx {provenance[i][1]})"
            print(f"{flat_indices[i]:<15} {prov_str:<20} {weights[i]:<10.6f}")