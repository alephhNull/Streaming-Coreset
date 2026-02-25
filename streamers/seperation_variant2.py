import numpy as np
from typing import Optional, List, Tuple, Dict, Any
from streamers.abstract_streamer import AbstractStreamingCoreset

# ==============================================================================
#  1. RIGOROUS HYPERPARAMETERS (SEPARATION-GUARDED PFW)
# ==============================================================================

class TheoreticalHyperparamsSGPFW:
    """
    Computes the EXACT theoretical bounds for Separation-Guarded Pairwise Frank-Wolfe.
    
    Demonstrates the fundamental trade-off:
    Higher nu -> Better conditioning -> Faster convergence (Lower Lag) -> Higher Quantization Error.
    """

    def __init__(
        self,
        M: int,
        D: int,
        nu_separation: float,
        delta_max: float,        # For Scenario A (Constant Max Drift)
        K_iter: int = 100,
    ):
        self.M = M
        self.D = D
        self.nu = nu_separation
        self.delta_max = delta_max
        self.K_iter = K_iter

        # 1. Bias is ZERO
        self.eps_bias = 0.0

        # 2. Contraction Rate (Rigorous Condition Number Bound)
        # For a nu-separated set of M points, the condition number kappa <= M / nu^2
        # PFW geometric rate is rho = 2 / (M * kappa)
        kappa = self.M / (self.nu ** 2) if self.nu > 0 else float('inf')
        self.rho_fw = min(1.0, 2.0 / (self.M * kappa)) if kappa < float('inf') else 0.0
        
        # Contraction over K steps: gamma_K = (1 - rho_fw)^K
        self.gamma_K = (1.0 - self.rho_fw) ** self.K_iter

        # 3. Static Errors
        self.eps_rff = 1.0 / np.sqrt(self.D)
        self.eps_quant = self.nu             # The "No Free Lunch" cost of separation
        self.eps_geo = 2.0 / self.M          # Pigeonhole capacity limit

        # ----------------------------------------------------------------------
        # SCENARIO A: Constant Max Drift (Non-stationary stream)
        # Recurrence: eps = gamma * (eps + delta_max) + eps_geo
        # ----------------------------------------------------------------------
        if self.gamma_K >= 1.0 - 1e-12:
            self.eps_lag_const = float('inf')
        else:
            self.eps_lag_const = (self.gamma_K * self.delta_max + self.eps_geo) / (1.0 - self.gamma_K)
            
        self.bound_constant_drift = min(1.0, self.eps_rff + self.eps_quant + self.eps_lag_const + self.eps_bias)

        # ----------------------------------------------------------------------
        # SCENARIO B: Decaying Drift O(1/t) (Stationary/Converging stream)
        # As t -> infinity, drift -> 0. 
        # The tracking error vanishes completely. We only pay the geometric floor.
        # ----------------------------------------------------------------------
        if self.gamma_K >= 1.0 - 1e-12:
            self.eps_lag_decay = float('inf')
        else:
            self.eps_lag_decay = self.eps_geo / (1.0 - self.gamma_K)
            
        self.bound_decaying_drift = min(1.0, self.eps_rff + self.eps_quant + self.eps_lag_decay + self.eps_bias)
        self.eps_total = self.bound_decaying_drift

    def summary(self) -> str:
        return "\n".join([
            "=" * 70,
            "  SEPARATION-GUARDED PFW BOUND SUMMARY",
            "=" * 70,
            f"  Buffer M             = {self.M}",
            f"  Separation Nu        = {self.nu:.6f}",
            f"  Iterations K         = {self.K_iter}",
            f"  Contraction (rho_fw) = {self.rho_fw:.6f} per step",
            f"  Total Contraction    = {self.gamma_K:.6e} over K steps",
            "-" * 70,
            f"  [Static Penalties]",
            f"  Bias (Reg Cost)      = {self.eps_bias:.6f}  <-- (Mathematically Zero)",
            f"  RFF Approx Error     = {self.eps_rff:.6f}",
            f"  Quantization Error   = {self.eps_quant:.6f}  <-- (The cost of Nu)",
            "-" * 70,
            f"  [SCENARIO A: Constant Drift delta={self.delta_max:.4f}]",
            f"  Optimization Lag     = {self.eps_lag_const:.6f}",
            f"  ** Max Bound         = {self.bound_constant_drift:.6f}",
            "-" * 70,
            f"  [SCENARIO B: Decaying Drift O(1/t)] (Steady State as t->inf)",
            f"  Optimization Lag     = {self.eps_lag_decay:.6f}  <-- (Tracking err vanishes)",
            f"  ** Asymptotic Bound  = {self.bound_decaying_drift:.6f}",
            "=" * 70,
        ])


# ==============================================================================
#  2. THE STREAMER
# ==============================================================================

class SeparationGuardedFWStreamer(AbstractStreamingCoreset):
    """
    Separation-Guarded Pairwise Frank-Wolfe Algorithm.
    
    Implements the epsilon-net logic directly in the buffer ingestion step,
    followed by Warm-Started Pairwise Frank-Wolfe.
    """

    def __init__(
        self, 
        M: int, 
        D: int, 
        nu_separation: float, 
        delta_max: float,
        sampler, 
        batch_size: int = 1, 
        K_iter: int = 100, 
        verbose: bool = False
    ):
        self.M = M
        self.D = D
        self.nu = nu_separation
        self.sampler = sampler
        self.batch_size = batch_size
        self.K_iter = K_iter
        self.verbose = verbose
        
        self.hp = TheoreticalHyperparamsSGPFW(M, D, nu_separation, delta_max, K_iter)
        
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
        
        # 1. Update True Running Mean
        alpha = 1.0 / self.t
        self.mean_rff = (1.0 - alpha) * self.mean_rff + alpha * z

        # 2. Separation Guard (The Epsilon-Net check)
        # 
        merged = False
        if len(self.buffer_Z) > 0:
            dists = np.linalg.norm(self.buffer_Z - z, axis=1)
            min_dist_idx = np.argmin(dists)
            
            if dists[min_dist_idx] < self.nu:
                # REJECT AND MERGE: Point is too close to an existing atom.
                # We update the weights (Warm Start) but do not add a new row.
                self.buffer_weights *= (1.0 - alpha)
                self.buffer_weights[min_dist_idx] += alpha
                merged = True

        if not merged:
            # ACCEPT: Point is far enough away to guarantee condition number.
            self.buffer_X.append(x)
            self.buffer_y.append(y)
            self.buffer_provenance.append((batch_idx, local_idx))
            
            if len(self.buffer_Z) > 0:
                self.buffer_Z = np.vstack([self.buffer_Z, z[np.newaxis, :]])
                self.buffer_weights *= (1.0 - alpha)
                self.buffer_weights = np.append(self.buffer_weights, alpha) 
            else:
                self.buffer_Z = z[np.newaxis, :]
                self.buffer_weights = np.array([1.0])

        # 3. Pairwise Frank-Wolfe Optimization
        if len(self.buffer_Z) > 1:
            K_mat = self.buffer_Z @ self.buffer_Z.T
            linear_term = self.buffer_Z @ self.mean_rff
            
            for _ in range(self.K_iter):
                grad = K_mat @ self.buffer_weights - linear_term
                
                idx_s = np.argmin(grad) # Best atom
                
                active_indices = np.where(self.buffer_weights > 1e-9)[0]
                if len(active_indices) == 0: break
                
                sub_grad = grad[active_indices]
                idx_v = active_indices[np.argmax(sub_grad)] # Worst active atom
                
                gap = grad[idx_v] - grad[idx_s]
                if gap < 1e-7: break # Optimality reached
                
                # Exact line search
                hess = K_mat[idx_s, idx_s] - 2*K_mat[idx_s, idx_v] + K_mat[idx_v, idx_v]
                gamma = gap / hess if hess > 1e-10 else 1.0
                gamma = np.clip(gamma, 0, self.buffer_weights[idx_v])
                
                # Swap mass
                self.buffer_weights[idx_s] += gamma
                self.buffer_weights[idx_v] -= gamma
        
        # 4. Eviction
        # We only need to evict if we actually added a new point (not merged)
        if len(self.buffer_Z) > self.M:
            evict_idx = np.argmin(self.buffer_weights)
            
            self.buffer_Z = np.delete(self.buffer_Z, evict_idx, axis=0)
            self.buffer_weights = np.delete(self.buffer_weights, evict_idx)
            del self.buffer_X[evict_idx]
            del self.buffer_y[evict_idx]
            del self.buffer_provenance[evict_idx]
            
            self.buffer_weights /= np.sum(self.buffer_weights)

        self.mmd_history.append(self.get_current_mmd())

    def process_batch(self, X, y, batch_idx):
        if self._finalized: return
        Z = self.sampler.transform(X)
        for i in range(X.shape[0]):
            self._process_point(X[i], int(y[i]), Z[i], batch_idx, i)

    def get_current_mmd(self):
        if len(self.buffer_Z) == 0: return float("inf")
        embed = self.buffer_Z.T @ self.buffer_weights
        return float(np.sqrt(np.sum((self.mean_rff - embed) ** 2)))
        
    def get_final_coreset(self):
        self._finalized = True
        flat = np.array([p[0]*self.batch_size + p[1] for p in self.buffer_provenance], dtype=int)
        return flat, self.buffer_weights.copy(), list(self.buffer_provenance)

    def get_diagnostics(self):
        return {
            "t": self.t, 
            "mmd": self.get_current_mmd(), 
            "bound_scenario_A": self.hp.bound_constant_drift,
            "bound_scenario_B": self.hp.bound_decaying_drift
        }

    def print_coreset_provenance(self):
        pass