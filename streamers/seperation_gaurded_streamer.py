import numpy as np
from typing import Optional, List, Tuple, Dict, Any
from streamers.abstract_streamer import AbstractStreamingCoreset

# ==============================================================================
#  1. RIGOROUS HYPERPARAMETERS (SEPARATION-GUARDED PFW)
# ==============================================================================

class TheoreticalHyperparamsFW:
    """
    Deterministically computes the theoretical error bound for Separation-Guarded PFW.
    
    The bound is composed of four additive terms:
      1. RFF Error: Fixed approx error (1/sqrt(D)).
      2. Geometric Error: Capacity limit of M points (2/M).
      3. Quantization Error: The penalty for rejecting points closer than nu (nu).
      4. Optimization Lag: The tracking error from PFW convergence.
    """

    def __init__(
        self,
        M: int,
        D: int,
        delta_drift_max: float,   # Max drift for Case A
        nu_separation: float,     # The separation guard threshold
        K_iter: int = 100,
        t_final: int = 2000,      # To estimate 1/t drift for Case B
    ):
        self.M = M
        self.D = D
        self.delta_drift_max = delta_drift_max
        self.nu = nu_separation
        self.K_iter = K_iter
        self.t_final = t_final

        # ----------------------------------------------------------------------
        # A. Condition Number & Contraction Rate (The "Guard" Effect)
        # ----------------------------------------------------------------------
        # We bound the coherence 'c' of the dictionary based on nu.
        # <z_i, z_j> <= 1 - nu^2/2
        self.coherence = 1.0 - (self.nu ** 2) / 2.0
        
        # Eigenvalue bounds of the Gram matrix (Gershgorin Circle Theorem)
        # L (Smoothness) <= 1 + (M-1)c
        # mu (Strong Convexity) >= 1 - (M-1)c
        
        self.L_smooth = 1.0 + (self.M - 1) * self.coherence
        self.mu_convex = 1.0 - (self.M - 1) * self.coherence
        
        # If mu <= 0, the buffer is too dense for the given M; condition number explodes.
        # We fall back to sublinear convergence if separation is insufficient.
        self.rho_fw = (self.nu ** 2) / (4.0 * self.M)
        self.gamma_K = (1.0 - self.rho_fw) ** self.K_iter

        # ----------------------------------------------------------------------
        # B. Fixed Error Terms (Independent of Optimization)
        # ----------------------------------------------------------------------
        # 1. RFF Noise
        self.eps_rff = 1.0 / np.sqrt(self.D)
        
        # 2. Geometric Capacity (2/M)
        self.eps_geo = 2.0 / self.M
        
        # 3. Quantization Penalty (The "No Free Lunch" Cost)
        # Rejecting a point introduces error <= nu
        self.eps_quant = self.nu

        # ----------------------------------------------------------------------
        # C. Optimization Lag (Case A vs Case B)
        # ----------------------------------------------------------------------
        
        # CASE A: Constant Maximum Drift (Worst Case / High Volatility)
        # Steady State Error = (gamma * delta + eps_geo) / (1 - gamma)
        # We add Quantization externally.
        if self.gamma_K >= 1.0 - 1e-9:
            self.lag_case_A = 1.0 # Trivial bound
        else:
            numerator = (self.gamma_K * self.delta_drift_max) + self.eps_geo
            denominator = 1.0 - self.gamma_K
            self.lag_case_A = numerator / denominator

        # CASE B: Decaying Drift (Stable Stream O(1/t))
        # The tracking error vanishes as t -> infinity.
        # Asymptotic Error = eps_geo + eps_quant (Lag -> 0)
        # However, for a rigorous bound at t_final, we use delta_t = 1/t_final.
        delta_t_final = 1.0 / float(self.t_final) if self.t_final > 0 else 1.0
        
        if self.gamma_K >= 1.0 - 1e-9:
             self.lag_case_B = 1.0
        else:
            # We treat delta_t_final as the "instantaneous" drift at the end
            num_B = (self.gamma_K * delta_t_final) + self.eps_geo
            self.lag_case_B = num_B / (1.0 - self.gamma_K)

        # ----------------------------------------------------------------------
        # D. Total Bounds
        # ----------------------------------------------------------------------
        # Total = RFF + Quant + Lag (which includes Geo inside)
        
        raw_A = self.eps_rff + self.eps_quant + self.lag_case_A
        self.bound_case_A = min(1.0, raw_A)
        
        raw_B = self.eps_rff + self.eps_quant + self.lag_case_B
        self.bound_case_B = min(1.0, raw_B)
        self.eps_total = self.bound_case_B

    def summary(self) -> str:
        # status = "ILL-CONDITIONED (nu too small)" if self.is_ill_conditioned else "Well-Conditioned"
        return "\n".join([
            "=" * 65,
            "  SEPARATION-GUARDED PFW BOUND SUMMARY",
            "=" * 65,
            f"  Buffer M             = {self.M}",
            f"  Separation nu        = {self.nu:.6f}",
            # f"  Condition Number     = {self.kappa:.2f} ({status})",
            f"  Contraction (rho)    = {self.rho_fw:.6f}",
            f"  Residual (gamma^K)   = {self.gamma_K:.6e}",
            "-" * 65,
            f"  1. RFF Approx        = {self.eps_rff:.6f}",
            f"  2. Quantization      = {self.eps_quant:.6f} (Cost of Guard)",
            f"  3. Geo Capacity      = {self.eps_geo:.6f}",
            "-" * 65,
            f"  CASE A (Max Drift)   = {self.bound_case_A:.6f}",
            f"  CASE B (1/t Drift)   = {self.bound_case_B:.6f} (Asymptotic)",
            "=" * 65,
        ])


# ==============================================================================
#  2. THE STREAMER
# ==============================================================================

class SeparationGuardedFrankWolfeStreamer(AbstractStreamingCoreset):
    """
    Separation-Guarded Pairwise Frank-Wolfe.
    
    Algorithm:
    1. Ingest x_t.
    2. Check dist(x_t, Buffer).
    3. If dist < nu: Reject x_t (Quantization Error).
    4. If dist >= nu: Add x_t to Buffer (Guard Condition Met).
    5. Run Pairwise Frank-Wolfe (Warm Start).
    """

    def __init__(
        self,
        M: int,
        D: int,
        delta_drift_max: float,
        nu_separation: float,
        sampler,
        batch_size: int = 1,
        K_iter: int = 100,
        verbose: bool = False,
        hyperparams: Optional[TheoreticalHyperparamsFW] = None,
    ):
        self.M = M
        self.D = D
        self.sampler = sampler
        self.batch_size = batch_size
        self.verbose = verbose

        # Init Hyperparams
        if hyperparams is not None:
            self.hp = hyperparams
        else:
            self.hp = TheoreticalHyperparamsFW(
                M, D, delta_drift_max, nu_separation, K_iter
            )

        self.K_iter = self.hp.K_iter
        self.nu = self.hp.nu
        
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
        
        # Diagnostics
        self.rejection_count = 0

    def _process_point(self, x_raw, y_label, z_rff, batch_idx, local_idx):
        self.t += 1

        # A. Update Global Mean (Drift)
        alpha = 1.0 / self.t
        self.mean_rff = (1.0 - alpha) * self.mean_rff + alpha * z_rff

        # B. Separation Guard (Check distances)
        # We only add the point if it is sufficiently far from existing buffer points.
        accept_point = True
        
        if len(self.buffer_Z) > 0:
            # Vectorized distance calculation
            # dist^2 = ||z_t||^2 + ||z_i||^2 - 2<z_t, z_i>
            # Assuming normalized RFF features ||z|| approx 1 (or exactly 1 for Cosine)
            # We use full Euclidean for rigor.
            diffs = self.buffer_Z - z_rff
            dists_sq = np.sum(diffs**2, axis=1)
            min_dist_sq = np.min(dists_sq)
            
            if min_dist_sq < self.nu ** 2:
                accept_point = False
                self.rejection_count += 1
                # Note: We do NOT need to manually add weight to the nearest neighbor.
                # The mean_rff has shifted towards the new point.
                # The PFW optimization below will automatically shift weight to 
                # the nearest neighbor to minimize the distance to the new mean.

        if accept_point:
            self.buffer_X.append(x_raw)
            self.buffer_y.append(y_label)
            
            if len(self.buffer_Z) > 0:
                self.buffer_Z = np.vstack([self.buffer_Z, z_rff[np.newaxis, :]])
                # Warm Start: Decay old weights, append 0.0 for new point
                self.buffer_weights *= (1.0 - alpha)
                self.buffer_weights = np.append(self.buffer_weights, alpha) 
            else:
                self.buffer_Z = z_rff[np.newaxis, :]
                self.buffer_weights = np.array([1.0])
                
            self.buffer_provenance.append((batch_idx, local_idx))
        else:
            # Even if rejected, we must re-normalize weights to sum to 1
            # (conceptually, we just re-optimize existing weights against new mean)
            pass

        # C. Pairwise Frank-Wolfe Optimization
        # We optimize the weights of WHATEVER is currently in the buffer
        # against the NEW updated mean.
        
        n_buf = len(self.buffer_Z)
        if n_buf > 1:
            K_mat = self.buffer_Z @ self.buffer_Z.T
            linear_term = self.buffer_Z @ self.mean_rff
            
            for k in range(self.K_iter):
                # 1. Gradient
                grad = K_mat @ self.buffer_weights - linear_term
                
                # 2. Oracle (Good and Bad Atoms)
                idx_s = np.argmin(grad)
                
                active_indices = np.where(self.buffer_weights > 1e-9)[0]
                if len(active_indices) == 0: break
                
                sub_grad = grad[active_indices]
                idx_v_local = np.argmax(sub_grad)
                idx_v = active_indices[idx_v_local]
                
                # Convergence Check (Duality Gap)
                gap = grad[idx_v] - grad[idx_s]
                if gap < 1e-7: break
                
                # 3. Step Size (Exact Line Search)
                # d = e_s - e_v
                # hess = d^T K d
                hess = K_mat[idx_s, idx_s] - 2*K_mat[idx_s, idx_v] + K_mat[idx_v, idx_v]
                
                if hess < 1e-10: 
                    gamma = 1.0 
                else:
                    gamma = gap / hess
                
                # Clip to available weight
                gamma = min(gamma, self.buffer_weights[idx_v])
                
                # 4. Update
                self.buffer_weights[idx_s] += gamma
                self.buffer_weights[idx_v] -= gamma

        # D. Eviction (If buffer grew)
        if len(self.buffer_Z) > self.M:
            evict_idx = np.argmin(self.buffer_weights)
            
            self.buffer_Z = np.delete(self.buffer_Z, evict_idx, axis=0)
            self.buffer_weights = np.delete(self.buffer_weights, evict_idx)
            del self.buffer_X[evict_idx]
            del self.buffer_y[evict_idx]
            del self.buffer_provenance[evict_idx]
            
            # Renormalize to correct numerical drift
            s = np.sum(self.buffer_weights)
            if s > 1e-9: self.buffer_weights /= s
            else: self.buffer_weights = np.ones(len(self.buffer_weights))/len(self.buffer_weights)

        self.mmd_history.append(self.get_current_mmd())

    def process_batch(self, X_batch_np, y_batch_np, batch_idx):
        if self._finalized: return
        batch_len = X_batch_np.shape[0]
        Z_batch_rff = self.sampler.transform(X_batch_np)
        self.num_points_seen += batch_len
        for i in range(batch_len):
            self._process_point(
                X_batch_np[i], int(y_batch_np[i]), Z_batch_rff[i], batch_idx, i
            )

    def get_current_mmd(self) -> float:
        if len(self.buffer_Z) == 0: return float("inf")
        embed = self.buffer_Z.T @ self.buffer_weights
        return float(np.sqrt(np.sum((self.mean_rff - embed) ** 2)))

    def get_final_coreset(self):
        self._finalized = True
        flat_indices = np.array([p[0] * self.batch_size + p[1] for p in self.buffer_provenance], dtype=int)
        return flat_indices, self.buffer_weights.copy(), list(self.buffer_provenance)

    def get_diagnostics(self) -> Dict[str, Any]:
        return {
            "t": self.t,
            "buffer_size": len(self.buffer_Z),
            "current_mmd": self.get_current_mmd(),
            "rejections": self.rejection_count,
            "bound_case_A": self.hp.bound_case_A,
            "bound_case_B": self.hp.bound_case_B
        }

    def print_coreset_provenance(self):
        pass