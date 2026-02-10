import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.kernel_approximation import RBFSampler
from qpsolvers import solve_qp
import warnings

warnings.filterwarnings('ignore')

# --- 1. CORE ALGORITHMS (Same logic, refined for metric tracking) ---

def weighted_kernel_herding_rff_qp(mu_pi, X_candidate, sampler, m):
    # (Same solver as before)
    n_candidates = X_candidate.shape[0]
    D = mu_pi.shape[0]
    X_candidate_rff = sampler.transform(X_candidate)
    
    # Greedy Selection
    selected_indices = []
    current_embedding = np.zeros(D)
    for k in range(m):
        residual = mu_pi - current_embedding
        search_values = X_candidate_rff @ residual
        if len(selected_indices) > 0:
            search_values[selected_indices] = -np.inf
        best_x_idx = np.argmax(search_values)
        selected_indices.append(best_x_idx)
        current_embedding = (k * current_embedding + X_candidate_rff[best_x_idx]) / (k + 1)

    # QP Weight Optimization
    coreset_rff = X_candidate_rff[selected_indices]
    K_rff = coreset_rff @ coreset_rff.T
    z_rff = coreset_rff @ mu_pi
    
    eps = 1e-8
    P = (K_rff + K_rff.T) / 2.0 + eps * np.eye(K_rff.shape[0])
    q = -z_rff
    A = np.ones((1, len(selected_indices)))
    b = np.array([1.0])
    G = -np.eye(len(selected_indices))
    h = np.zeros(len(selected_indices))

    try:
        weights = solve_qp(P, q, G, h, A, b, solver="quadprog")
        if weights is None: weights = np.ones(m)/m
    except:
        weights = np.ones(m)/m
        
    return np.array(selected_indices), weights

class StreamingCoresetBase:
    def __init__(self, coreset_size, buffer_capacity, sampler):
        self.coreset_size = coreset_size
        self.buffer_capacity = buffer_capacity
        self.sampler = sampler
        self.buffer_X = np.empty((0, sampler.random_weights_.shape[0]))
        self.buffer_y = np.empty(0, dtype=int)
        self.buffer_w = np.empty(0)
    
    def get_mmd_error(self, target_X):
        # Calculate MMD between Coreset and a Target Validation Set
        if len(self.buffer_X) == 0: return 1.0
        
        # Mean of Coreset
        phi_coreset = self.sampler.transform(self.buffer_X)
        mu_coreset = np.average(phi_coreset, axis=0, weights=self.buffer_w)
        
        # Mean of Target
        phi_target = self.sampler.transform(target_X)
        mu_target = np.mean(phi_target, axis=0)
        
        # L2 Distance in RKHS
        return np.linalg.norm(mu_coreset - mu_target)

class WKH_Benchmark(StreamingCoresetBase):
    def __init__(self, B, sampler):
        super().__init__(B, B, sampler)
        self.mean_global = np.zeros(sampler.n_components)
        self.n_seen = 0
        
    def process(self, X_batch, y_batch):
        # Update Global Mean
        X_rff = self.sampler.transform(X_batch)
        batch_mean = np.mean(X_rff, axis=0)
        n = len(X_batch)
        if self.n_seen == 0: self.mean_global = batch_mean
        else: self.mean_global = (self.n_seen/(self.n_seen+n))*self.mean_global + (n/(self.n_seen+n))*batch_mean
        self.n_seen += n
        
        # Buffer
        self.buffer_X = np.vstack([self.buffer_X, X_batch])
        self.buffer_y = np.concatenate([self.buffer_y, y_batch])
        
        if len(self.buffer_X) > self.buffer_capacity:
            inds, w = weighted_kernel_herding_rff_qp(self.mean_global, self.buffer_X, self.sampler, self.coreset_size)
            self.buffer_X = self.buffer_X[inds]
            self.buffer_y = self.buffer_y[inds]
            self.buffer_w = w
        else:
            self.buffer_w = np.ones(len(self.buffer_X))/len(self.buffer_X)

class MSKH_Proposed(StreamingCoresetBase):
    def __init__(self, B, sampler, alpha=0.3, lam=0.5): # TUNED LAMBDA HERE
        super().__init__(B, B, sampler)
        self.mean_global = np.zeros(sampler.n_components)
        self.mean_fast = np.zeros(sampler.n_components)
        self.n_seen = 0
        self.alpha = alpha
        self.lam = lam
        
    def process(self, X_batch, y_batch):
        X_rff = self.sampler.transform(X_batch)
        batch_mean = np.mean(X_rff, axis=0)
        n = len(X_batch)
        
        # Update Means
        if self.n_seen == 0:
            self.mean_global = batch_mean
            self.mean_fast = batch_mean
        else:
            self.mean_global = (self.n_seen/(self.n_seen+n))*self.mean_global + (n/(self.n_seen+n))*batch_mean
            self.mean_fast = (1-self.alpha)*self.mean_fast + self.alpha*batch_mean
        self.n_seen += n
        
        # Buffer
        self.buffer_X = np.vstack([self.buffer_X, X_batch])
        self.buffer_y = np.concatenate([self.buffer_y, y_batch])
        
        if len(self.buffer_X) > self.buffer_capacity:
            # Composite Target
            target = (self.mean_global + self.lam * self.mean_fast) / (1 + self.lam)
            
            inds, w = weighted_kernel_herding_rff_qp(target, self.buffer_X, self.sampler, self.coreset_size)
            self.buffer_X = self.buffer_X[inds]
            self.buffer_y = self.buffer_y[inds]
            self.buffer_w = w
        else:
            self.buffer_w = np.ones(len(self.buffer_X))/len(self.buffer_X)

# --- 2. EXPERIMENT ---

def run_experiment():
    print("Loading MNIST and setting up Stream...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X_all = mnist.data.astype('float32') / 255.0
    y_all = mnist.target.astype('int')
    
    # Create Targets for Metrics (Validation Sets)
    # We want to measure error on "Digit 0" (History) vs "Digit 4" (Current)
    X_val_0 = X_all[y_all == 0][:100]
    X_val_4 = X_all[y_all == 4][:100]
    
    # Create Ordered Stream 0 -> 1 -> 2 -> 3 -> 4
    stream_X, stream_y = [], []
    points_per_class = 300
    for i in range(5):
        idx = np.where(y_all == i)[0][:points_per_class]
        stream_X.append(X_all[idx])
        stream_y.append(y_all[idx])
    X_stream = np.vstack(stream_X)
    y_stream = np.concatenate(stream_y)
    
    # Setup
    B = 25
    BATCH = 10
    RFF_DIM = 500
    sampler = RBFSampler(gamma=0.01, n_components=RFF_DIM, random_state=42)
    sampler.fit(X_stream[:1000])
    
    wkh = WKH_Benchmark(B, sampler)
    mskh = MSKH_Proposed(B, sampler, alpha=0.2, lam=0.5) # Lambda=0.6 (Balanced)
    
    # Metric Logs
    err_current_wkh, err_current_mskh = [], []
    err_hist_wkh, err_hist_mskh = [], []
    
    print("Streaming...")
    n_batches = len(y_stream) // BATCH
    
    for i in range(n_batches):
        s, e = i*BATCH, (i+1)*BATCH
        xb, yb = X_stream[s:e], y_stream[s:e]
        
        wkh.process(xb, yb)
        mskh.process(xb, yb)
        
        # Metric 1: Responsiveness (Error on Current Batch)
        # We use the actual current batch as proxy for "Current Distribution"
        e_cur_w = wkh.get_mmd_error(xb)
        e_cur_m = mskh.get_mmd_error(xb)
        
        # Metric 2: Retention (Error on Digit 0 - The oldest history)
        e_his_w = wkh.get_mmd_error(X_val_0)
        e_his_m = mskh.get_mmd_error(X_val_0)
        
        err_current_wkh.append(e_cur_w)
        err_current_mskh.append(e_cur_m)
        err_hist_wkh.append(e_his_w)
        err_hist_mskh.append(e_his_m)

    return (err_current_wkh, err_current_mskh, err_hist_wkh, err_hist_mskh, 
            wkh.buffer_y, mskh.buffer_y)

# Run
res = run_experiment()
e_cur_w, e_cur_m, e_his_w, e_his_m, buf_w, buf_m = res

# --- 3. VISUALIZATION ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Responsiveness (Lower is Better)
ax1.plot(e_cur_w, label='Benchmark WKH', color='gray', linestyle='--')
ax1.plot(e_cur_m, label='Proposed MS-KH', color='green')
ax1.set_title("Metric 1: Responsiveness (MMD to Current Batch)")
ax1.set_xlabel("Time Steps")
ax1.set_ylabel("Error (Lower is Better)")
ax1.legend()
ax1.grid(alpha=0.3)

# Plot 2: Retention (Lower is Better)
ax2.plot(e_his_w, label='Benchmark WKH', color='gray', linestyle='--')
ax2.plot(e_his_m, label='Proposed MS-KH', color='green')
ax2.set_title("Metric 2: Retention (MMD to Digit '0')")
ax2.set_xlabel("Time Steps")
ax2.set_ylabel("Error (Lower is Better)")
ax2.legend()
ax2.grid(alpha=0.3)

# Add Phase Lines
phase_len = 300 // 10
for ax in [ax1, ax2]:
    for i in range(5):
        ax.axvline(i*phase_len, color='k', alpha=0.1)

plt.tight_layout()
plt.show()

# Print Composition
print("\n--- Final Composition (Balanced Target: [5,5,5,5,5]) ---")
print(f"Benchmark: {np.bincount(buf_w, minlength=5)}")
print(f"Proposed:  {np.bincount(buf_m, minlength=5)}")