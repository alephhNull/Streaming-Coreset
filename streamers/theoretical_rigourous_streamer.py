from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from streamers.abstract_streamer import AbstractStreamingCoreset

class TheoreticalHyperparams:
    """
    Deterministically computes the theoretical error bound in Distance Space.
    
    CORRECTED LOGIC (Rigorous):
    1. Separates Optimization Gap (contractible) from Regularization Bias (fixed).
    2. Models recurrence on the Energy Gap (J - J*), not absolute Distance.
    3. Adds the bias term: sqrt(2 * lambda * ln(M)).
    """

    def __init__(
        self,
        M: int,
        D: int,
        gamma: float,       # Unused in rigorous calculation, derived internally
        delta_drift: float,
        lambda_reg: float = 0.01,
        K_iter: int = 500,
    ):
        self.M = M
        self.D = D
        self.delta_drift = delta_drift
        self.lambda_reg = lambda_reg
        self.K_iter = K_iter

        # 1. Optimization Constants
        self.L_f = 1.0
        self.eta = 1.0 / self.L_f
        
        # Energy Contraction Rate (per step)
        # For Strongly Convex objective: rho = 1 - eta * lambda
        self.rho_energy_step = 1.0 - (self.eta * self.lambda_reg)
        
        # Total Contraction over K steps
        self.rho_K = self.rho_energy_step ** self.K_iter

        # 2. Energy Shocks (Squared space)
        # Eviction Shock in Energy: approx (2/M)^2 / 2  -> conservative estimate derived from distance
        # We convert the user's distance-based eps_geo (2/M) into an energy shock estimate.
        dist_geo = 2.0 / self.M
        self.energy_shock_geo = (dist_geo ** 2) / 2.0
        
        # Drift Shock in Energy: approx (delta)^2 / 2
        self.energy_shock_drift = (self.delta_drift ** 2) / 2.0

        # 3. Steady State Gap (Energy Space)
        # Recurrence: Gap_new <= rho^K * (Gap_old + shock_drift) + shock_geo
        # Steady State Gap_ss = (shock_geo + rho^K * shock_drift) / (1 - rho^K)
        
        numerator = self.energy_shock_geo + (self.rho_K * self.energy_shock_drift)
        denominator = 1.0 - self.rho_K
        
        self.gap_ss_energy = numerator / denominator
        
        # Convert Gap to Distance (The Optimization Lag + Geo Error)
        self.eps_lag_geo = np.sqrt(2.0 * self.gap_ss_energy)

        # 4. Regularization Bias (The Price of Entropy)
        # Bound: sqrt(2 * lambda * max(Entropy))
        # max(Entropy) on simplex is ln(M)
        self.eps_bias = np.sqrt(2.0 * self.lambda_reg * np.log(self.M))

        # 5. RFF Approx Error
        self.eps_rff = 1.0 / np.sqrt(self.D)

        # 6. Total Bound
        self.eps_total = self.eps_rff + self.eps_lag_geo + self.eps_bias

    def summary(self) -> str:
        return "\n".join([
            "=" * 65,
            "  RIGOROUS THEORETICAL BOUND SUMMARY",
            "=" * 65,
            f"  Buffer M             = {self.M}",
            f"  RFF Dim D            = {self.D}",
            f"  Lambda (Reg)         = {self.lambda_reg}",
            f"  Contraction (rho^K)  = {self.rho_K:.6f}",
            "-" * 65,
            f"  1. Opt Lag + Geo     = {self.eps_lag_geo:.6f}",
            f"  2. Reg Bias          = {self.eps_bias:.6f} (Entropy Cost)",
            f"  3. RFF Approx        = {self.eps_rff:.6f}",
            "-" * 65,
            f"  ** Total Error Bound = {self.eps_total:.6f}",
            "=" * 65,
        ])

# The Streamer class remains largely the same, 
# but effectively uses the updated Hyperparams for reporting.
class TheoreticalRigourousStreamer(AbstractStreamingCoreset):
    """
    Separation-Guarded Online Coreset with Entropic Mirror Descent.
    """

    def __init__(
        self,
        M: int,
        D: int,
        gamma: float, # Kept for API compatibility, but ignored by rigorous math
        delta_drift: float,
        sampler,
        batch_size: int = 1,
        verbose: bool = False,
        hyperparams: Optional[TheoreticalHyperparams] = None,
    ):
        self.M = M
        self.D = D
        self.delta_drift = delta_drift
        self.sampler = sampler
        self.batch_size = batch_size
        self.verbose = verbose

        # Use the RIGOROUS hyperparams by default
        if hyperparams is not None:
            self.hp = hyperparams
        else:
            self.hp = TheoreticalHyperparams(M, D, gamma, delta_drift)

        self.eta = self.hp.eta
        self.K_iter = self.hp.K_iter
        self.lambda_reg = self.hp.lambda_reg

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
        self.merge_count = 0
        self.add_count = 0
        self.evict_count = 0

    def _process_point(
        self,
        x_raw: np.ndarray,
        y_label: int,
        z_rff: np.ndarray,
        batch_idx: int,
        local_idx: int,
    ):
        self.t += 1

        # A. Update Global Mean
        alpha = 1.0 / self.t
        self.mean_rff = (1.0 - alpha) * self.mean_rff + alpha * z_rff

        # B. Add Point
        self.buffer_X.append(x_raw)
        self.buffer_y.append(y_label)
        
        if len(self.buffer_Z) > 0:
            self.buffer_Z = np.vstack([self.buffer_Z, z_rff[np.newaxis, :]])
        else:
            self.buffer_Z = z_rff[np.newaxis, :]

        # Warm Start
        if len(self.buffer_weights) > 0:
            self.buffer_weights *= (1.0 - alpha)
            self.buffer_weights = np.append(self.buffer_weights, alpha)
        else:
            self.buffer_weights = np.array([1.0])
            
        self.buffer_provenance.append((batch_idx, local_idx))
        self.add_count += 1

        # C. Optimization (Entropic Mirror Descent)
        n_buf = len(self.buffer_Z)
        if n_buf > 0:
            K_mat = self.buffer_Z @ self.buffer_Z.T
            linear_term = self.buffer_Z @ self.mean_rff
            
            w = self.buffer_weights.copy()
            eps_safe = 1e-20

            # Using log-space updates for numerical stability with entropy
            for _ in range(self.K_iter):
                grad_mmd = K_mat @ w - linear_term
                
                log_w = np.log(np.maximum(w, eps_safe))
                # decay = 1 - eta * lambda (implements entropic regularization)
                decay = 1.0 - (self.eta * self.lambda_reg)
                log_w_new = (decay * log_w) - (self.eta * grad_mmd)
                
                log_w_new -= np.max(log_w_new) # Stability shift
                w_unnorm = np.exp(log_w_new)
                w = w_unnorm / np.sum(w_unnorm)

            self.buffer_weights = w

        # D. Eviction
        if len(self.buffer_Z) > self.M:
            evict_idx = np.argmin(self.buffer_weights)
            self.evict_count += 1
            
            self.buffer_Z = np.delete(self.buffer_Z, evict_idx, axis=0)
            self.buffer_weights = np.delete(self.buffer_weights, evict_idx)
            del self.buffer_X[evict_idx]
            del self.buffer_y[evict_idx]
            del self.buffer_provenance[evict_idx]
            
            self.buffer_weights /= np.sum(self.buffer_weights)

        self.mmd_history.append(self.get_current_mmd())

    def process_batch(
        self, X_batch_np: np.ndarray, y_batch_np: np.ndarray, batch_idx: int
    ) -> None:
        if self._finalized:
            return
        batch_len = X_batch_np.shape[0]
        Z_batch_rff = self.sampler.transform(X_batch_np)
        self.num_points_seen += batch_len
        for i in range(batch_len):
            self._process_point(
                x_raw=X_batch_np[i],
                y_label=int(y_batch_np[i]),
                z_rff=Z_batch_rff[i],
                batch_idx=batch_idx,
                local_idx=i,
            )

    def get_final_coreset(self) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
        self._finalized = True
        if not self.buffer_provenance:
             return np.array([], dtype=int), np.array([], dtype=float), []
        flat_indices = np.array(
            [p[0] * self.batch_size + p[1] for p in self.buffer_provenance], dtype=int
        )
        return flat_indices, self.buffer_weights.copy(), list(self.buffer_provenance)

    def print_coreset_provenance(self) -> None:
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

    def get_current_mmd(self) -> float:
        if len(self.buffer_Z) == 0:
            return float("inf")
        embed = self.buffer_Z.T @ self.buffer_weights
        return float(np.sqrt(np.sum((self.mean_rff - embed) ** 2)))

    def get_diagnostics(self) -> Dict[str, Any]:
        return {
            "t": self.t,
            "buffer_size": len(self.buffer_Z),
            "current_mmd_empirical": self.get_current_mmd(),
            "bound_lag_geo": self.hp.eps_lag_geo,
            "bound_bias": self.hp.eps_bias,
            "bound_rff": self.hp.eps_rff,
            "bound_total": self.hp.eps_total
        }