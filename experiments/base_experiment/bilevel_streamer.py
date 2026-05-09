import numpy as np
import random
from typing import List, Tuple
from streamers.abstract_streamer import AbstractStreamingCoreset

class BilevelStreamingCoreset(AbstractStreamingCoreset):
    """
    Implementation of "Coresets via Bilevel Optimization for Continual Learning and Streaming" (Borsos et al., 2020).
    Uses Merge-Reduce for maintaining a streaming replay buffer and Kernel Ridge Regression (KRR) proxy for subset selection.
    """
    
    def __init__(self, total_coreset_size: int, sampler_rff, num_slots: int = 10, gamma: float = 5e-4, lambda_reg: float = 1e-3, candidate_sample_size: int = 200):
        """
        Args:
            total_coreset_size (int): The total capacity of the streaming buffer.
            sampler_rff: RFF sampler for computing L2 surrogate in RFF space.
            num_slots (int): The number of slots for the merge-reduce framework (default is s=10 as per streaming experiments).
            gamma (float): RBF kernel parameter.
            lambda_reg (float): Regularization strength for the KRR proxy inner objective.
            candidate_sample_size (int): Number of greedy candidates sampled per forward selection step.
        """
        self.num_slots = num_slots
        # To maintain the total size across 'num_slots', each slot has capacity total_coreset_size // num_slots.
        self.m_slot = max(1, total_coreset_size // num_slots) 
        self.gamma = gamma
        self.lambda_reg = lambda_reg
        self.candidate_sample_size = candidate_sample_size
        self.sampler_rff = sampler_rff
        
        # Buffer stores tuples of: (X, y, provenance, beta)
        self.buffer = []
        self._mmd_history = []
        self._cumulative_phi = None
        self._n_seen = 0

    @property
    def buffer_X(self) -> List[np.ndarray]:
        """Compatibility property for run_base_experiment."""
        return [X for X, _, _, _ in self.buffer]

    @property
    def buffer_y(self) -> np.ndarray:
        """Compatibility property for run_base_experiment. Flattens all labels in the buffer."""
        if not self.buffer:
            return np.array([])
        all_y = [y for _, y, _, _ in self.buffer]
        return np.concatenate(all_y, axis=0)

    @property
    def buffer_weights(self) -> np.ndarray:
        """Compatibility property for run_base_experiment. Returns uniform weights across all points."""
        if not self.buffer:
            return np.array([])
        total_points = sum(len(X) for X, _, _, _ in self.buffer)
        return np.ones(total_points) / total_points

    @property
    def mmd_history(self) -> List[float]:
        """Compatibility property for run_base_experiment. Bilevel doesn't track internal MMD, so we return zeros."""
        if not hasattr(self, "_mmd_history"):
            return []
        return self._mmd_history

    @property
    def M(self) -> int:
        """Compatibility property for run_base_experiment."""
        return self.m_slot * self.num_slots

    @property
    def rff_dim(self) -> int:
        """Compatibility property for run_base_experiment. Bilevel works in input space."""
        if not self.buffer:
            return 0
        return self.buffer[0][0].shape[1]

    def _rbf_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Computes the RBF Kernel between X1 and X2."""
        # Equivalent to exp(-gamma * ||X1 - X2||^2)
        X1_sq = np.sum(X1**2, axis=1, keepdims=True)
        X2_sq = np.sum(X2**2, axis=1)
        dist_sq = X1_sq + X2_sq - 2 * np.dot(X1, X2.T)
        return np.exp(-self.gamma * np.clip(dist_sq, 0, None))

    def _to_one_hot(self, y: np.ndarray) -> np.ndarray:
        """Dynamically one-hot encodes integer labels for the KRR proxy."""
        if y.ndim > 1 and y.shape[1] > 1:
            return y
        y_int = y.astype(int).flatten()
        num_classes = np.max(y_int) + 1
        return np.eye(num_classes)[y_int]

    def _construct_coreset(self, X: np.ndarray, y: np.ndarray, prov: List[Tuple[int, int]], m: int) -> Tuple[np.ndarray, np.ndarray, List]:
        """
        Algorithm 1: Coresets via Bilevel Optimization (Proxy Formulation with greedy forward selection).
        """
        n = X.shape[0]
        if n <= m:
            return X, y, prov

        # Standardize features for kernel as per Appendix E data preprocessing rules
        X_std = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8)
        Y_one_hot = self._to_one_hot(y)
        K = self._rbf_kernel(X_std, X_std)

        # Initialize with a random point
        start_idx = random.randint(0, n - 1)
        S = [start_idx]
        rem_indices = set(range(n)) - {start_idx}

        # Greedy forward selection
        for t in range(2, m + 1):
            if not rem_indices:
                break
                
            # Random sample of candidate points to speed up outer objective evaluation
            num_to_sample = min(self.candidate_sample_size, len(rem_indices))
            C_candidates = random.sample(list(rem_indices), num_to_sample)
            
            best_k = -1
            best_loss = float('inf')
            
            for k in C_candidates:
                S_cand = S + [k]
                K_cand_inner = K[np.ix_(S_cand, S_cand)]
                Y_cand = Y_one_hot[S_cand]
                
                # Closed form inner optimization for KRR (Equation 6 equivalent / Section 6.1)
                try:
                    alpha = np.linalg.solve(K_cand_inner + self.lambda_reg * np.eye(len(S_cand)), Y_cand)
                except np.linalg.LinAlgError:
                    alpha = np.linalg.pinv(K_cand_inner + self.lambda_reg * np.eye(len(S_cand))) @ Y_cand
                
                # Evaluate outer objective (validation on full batch)
                K_all_cand = K[:, S_cand]
                y_pred = K_all_cand @ alpha
                loss = np.mean((Y_one_hot - y_pred) ** 2)
                
                if loss < best_loss:
                    best_loss = loss
                    best_k = k
                    
            S.append(best_k)
            rem_indices.remove(best_k)
            
        return X[S], y[S], [prov[i] for i in S]

    def _select_index(self) -> int:
        """
        Algorithm 3: select_index for determining which buffer slots to merge.
        """
        s = len(self.buffer) - 1
        if s == 1 or self.buffer[s-1][3] > self.buffer[s][3]:
            return s - 1
        else:
            # Finding argmin_i (beta_i == beta_i+1) translates to finding the first matching weight
            for i in range(s):
                if self.buffer[i][3] == self.buffer[i+1][3]:
                    return i
            return s - 1 

    def process_batch(self, X_batch_np: np.ndarray, y_batch_np: np.ndarray, batch_idx: int) -> None:
        """
        Algorithm 2: Streaming coresets via merge-reduce.
        """
        # Update cumulative mean embedding in RFF space for L2 surrogate tracking
        phi_batch = self.sampler_rff.transform(X_batch_np)
        batch_sum_phi = np.sum(phi_batch, axis=0)
        if self._cumulative_phi is None:
            self._cumulative_phi = batch_sum_phi
        else:
            self._cumulative_phi += batch_sum_phi
        self._n_seen += X_batch_np.shape[0]
        mu_t = self._cumulative_phi / self._n_seen

        # Create local provenance mapping
        prov = [(batch_idx, i) for i in range(X_batch_np.shape[0])]
        
        # Construct summary for the new batch
        C_X, C_y, C_prov = self._construct_coreset(X_batch_np, y_batch_np, prov, self.m_slot)
        beta_t = X_batch_np.shape[0]
        
        self.buffer.append((C_X, C_y, C_prov, beta_t))
        
        # Merge-reduce loop if buffer exceeds slot limits
        while len(self.buffer) > self.num_slots:
            k = self._select_index()
            
            X_k, y_k, prov_k, beta_k = self.buffer[k]
            X_k1, y_k1, prov_k1, beta_k1 = self.buffer[k+1]
            
            # Merge
            X_merged = np.vstack([X_k, X_k1])
            y_merged = np.concatenate([y_k, y_k1], axis=0) if y_k.ndim == 1 else np.vstack([y_k, y_k1])
            prov_merged = prov_k + prov_k1
            
            # Reduce
            C_X_prime, C_y_prime, C_prov_prime = self._construct_coreset(X_merged, y_merged, prov_merged, self.m_slot)
            beta_prime = beta_k + beta_k1
            
            # Replace
            del self.buffer[k+1]
            self.buffer[k] = (C_X_prime, C_y_prime, C_prov_prime, beta_prime)

        # Compute L2 surrogate for the current coreset
        all_X = []
        for X_slot, _, _, _ in self.buffer:
            all_X.append(X_slot)
        Z = np.vstack(all_X)
        phi_Z = self.sampler_rff.transform(Z)
        # Bilevel uses uniform weights for its final coreset
        w = np.ones(len(Z)) / len(Z)
        mu_coreset = w @ phi_Z
        l2_surrogate = np.linalg.norm(mu_t - mu_coreset)
        self._mmd_history.append(float(l2_surrogate))

    def get_final_coreset(self) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
        """
        Extracts and flattens the contents of the slots and applies uniform distribution weights.
        """
        if not self.buffer:
            return np.array([]), np.array([]), []
            
        all_X = []
        all_prov = []
        
        for X_slot, _, prov_slot, _ in self.buffer:
            all_X.append(X_slot)
            all_prov.extend(prov_slot)
            
        total_points = len(all_prov)
        weights = np.ones(total_points) / total_points
        
        # In a real dynamic stream, mapping to a global flat index requires a stateful counter tracking stream elements.
        # Without access to total global indices mapped arbitrarily, it is represented as sequential markers mapping local slots.
        flat_indices = np.arange(total_points) 
        
        return flat_indices, weights, all_prov

    def print_coreset_provenance(self) -> None:
        """
        Prints the batch and local indices representing the history of elements currently kept alive in the summary.
        """
        _, weights, provenance = self.get_final_coreset()
        print("Coreset Size:", len(provenance))
        for w, (b_idx, l_idx) in zip(weights, provenance):
            print(f"Weight: {w:.4f} | Source Batch: {b_idx} | Local Index: {l_idx}")
