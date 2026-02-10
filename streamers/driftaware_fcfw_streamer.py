import numpy as np

class DriftAwareHerdingStreamer:
    def __init__(self, budget, sampler, alpha=0.5, lookahead=10.0):
        self.budget = budget
        self.sampler = sampler
        self.alpha = alpha
        self.lookahead = lookahead
        
        # Buffers for raw data (for visualization)
        self.buffer_X = np.empty((0, 784))
        self.buffer_y = np.empty(0)
        self.buffer_provenance = []
        
        # Buffers for feature space (for optimization)
        self.buffer_phi = np.empty((0, sampler.n_components))
        
        # State for Drift Tracking
        self.target_mu = np.zeros(sampler.n_components)
        self.prev_target_mu = None
        self.drift_vec = np.zeros(sampler.n_components)
        self.t = 0

    def process_batch(self, Xb, yb, bidx):
        # 1. Map to Feature Space
        phi_batch = self.sampler.transform(Xb)
        n_batch = Xb.shape[0]
        
        for i in range(n_batch):
            self.t += 1
            curr_phi = phi_batch[i]
            
            # 2. Update Running Mean (Target Mu)
            if self.prev_target_mu is not None:
                # Update drift estimate: current_mean - previous_mean
                # Using an EMA for the drift vector to handle high-dimensional noise
                curr_drift = self.target_mu - self.prev_target_mu
                self.drift_vec = 0.8 * self.drift_vec + 0.2 * curr_drift
            
            self.prev_target_mu = self.target_mu.copy()
            self.target_mu = (1 - 1/self.t) * self.target_mu + (1/self.t) * curr_phi
            
            # 3. Add to Buffer
            self.buffer_X = np.vstack([self.buffer_X, Xb[i]])
            self.buffer_y = np.append(self.buffer_y, yb[i])
            self.buffer_phi = np.vstack([self.buffer_phi, curr_phi])
            self.buffer_provenance.append((bidx, i))
            
            # 4. Prune if over budget
            if len(self.buffer_y) > self.budget:
                self._prune()

    def _prune(self):
        # Solve for Current Weights (approximation via projection/herding)
        # For efficiency in a stream, we use the "Combined Utility" logic
        
        # Current utility: how well does the point match the current target?
        # We use a simple dot-product score as a proxy for 'weight' in herding
        w_curr = self.buffer_phi @ self.target_mu
        
        # Future utility: project target mean forward using drift
        target_future = self.target_mu + (self.drift_vec * self.lookahead)
        w_future = self.buffer_phi @ target_future
        
        # Combined Score
        utility = (1 - self.alpha) * w_curr + self.alpha * w_future
        
        # Remove the point that contributes LEAST to the current and future mean
        drop_idx = np.argmin(utility)
        
        self.buffer_X = np.delete(self.buffer_X, drop_idx, axis=0)
        self.buffer_y = np.delete(self.buffer_y, drop_idx)
        self.buffer_phi = np.delete(self.buffer_phi, drop_idx, axis=0)
        del self.buffer_provenance[drop_idx]

    def get_final_coreset(self):
        # Returns indices (dummy for compatibility), weights (uniform for herding), and provenance
        n = len(self.buffer_y)
        return np.arange(n), np.ones(n)/n, self.buffer_provenance