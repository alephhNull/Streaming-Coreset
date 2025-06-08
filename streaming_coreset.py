import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.kernel_approximation import RBFSampler
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import torch.nn.functional as F
import time

from bcsstreamer import BilevelCoresetSelector
from ocsstreamer import OCSStreamer # For tracking processing time

# --- Your provided helper functions (load_adult_data, train_classifier, calculate_mmd2_exact) ---
# (Assuming these are defined as in your prompt)
def load_adult_data(subset_size=500):
  adult = fetch_openml('adult', version=2, as_frame=True)
  data_df = adult.data
  target = adult.target

  # Handle missing values
  data_df = data_df.replace('?', np.nan)
  data_df = data_df.dropna()
  target = target.loc[data_df.index]

  subset_indices = np.random.choice(len(data_df), subset_size, replace=False)
  data_df = data_df.iloc[subset_indices, :]
  target = target.iloc[subset_indices]

  # Encode target ('<=50K' and '>50K' to 0 and 1)
  le_target = LabelEncoder()
  y = le_target.fit_transform(target)

  # Split into training and validation sets (80% train, 20% validation)
  X_train, X_val, y_train, y_val = train_test_split(data_df, y, test_size=0.2, random_state=42, stratify=y)

  # Identify numerical and categorical columns
  numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
  categorical_cols = X_train.select_dtypes(include=['category']).columns
#   categorical_cols = []

  ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
  ohe.fit(X_train[categorical_cols])
  X_train_cat = ohe.transform(X_train[categorical_cols])
  X_val_cat = ohe.transform(X_val[categorical_cols])

  # Standardize numerical features
  scaler = StandardScaler()
  scaler.fit(X_train[numerical_cols])
  X_train_num = scaler.transform(X_train[numerical_cols])
  X_val_num = scaler.transform(X_val[numerical_cols])

  # Combine processed features
  X_train_processed = np.hstack((X_train_num, X_train_cat))
  X_val_processed = np.hstack((X_val_num, X_val_cat))

  print(f'Shapes of X_train: {X_train_processed.shape}, X_val: {X_val_processed.shape}')
  return X_train_processed, X_val_processed, y_train, y_val


def train_classifier(X_train: np.ndarray, X_val: np.ndarray, y_train: np.ndarray, y_val: np.ndarray) -> float:
    unique_classes = np.unique(y_train)
    
    if len(unique_classes) < 2:
        print(f"  Warning: Training data contains only one class (label: {unique_classes[0]}). Using constant prediction.")
        # Predict the same class for all validation samples
        y_pred = np.full_like(y_val, fill_value=unique_classes[0])
        acc = accuracy_score(y_val, y_pred)
        return acc

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    return acc

def calculate_mmd2_exact(X_full, X_coreset, coreset_weights, kernel_gamma):
    if len(X_coreset) == 0 or len(X_full) == 0:
        return np.inf

    X_full_torch = torch.from_numpy(X_full).float()
    X_coreset_torch = torch.from_numpy(X_coreset).float()
    w_torch = torch.from_numpy(coreset_weights).float()

    n = X_full_torch.shape[0]
    m = X_coreset_torch.shape[0]

    # RBF kernel function
    def rbf_kernel(X1, X2, gamma):
        dist_sq = torch.cdist(X1, X2)**2
        return torch.exp(-gamma * dist_sq)

    K_full_full = rbf_kernel(X_full_torch, X_full_torch, kernel_gamma)
    K_coreset_coreset = rbf_kernel(X_coreset_torch, X_coreset_torch, kernel_gamma)
    K_full_coreset = rbf_kernel(X_full_torch, X_coreset_torch, kernel_gamma)

    term1 = torch.mean(K_full_full) # E_{x,x' ~ P}[k(x, x')]
    term2 = w_torch @ K_coreset_coreset @ w_torch # E_{y,y' ~ Q}[k(y, y')]
    term3 = -2 * torch.mean(K_full_coreset @ w_torch) # -2 * E_{x ~ P, y ~ Q}[k(x, y)]

    mmd2 = term1 + term2 + term3
    return max(0, mmd2.item()) # MMD^2 should be non-negative


# Assuming ReservoirSamplerBatchStreamer is defined as in the previous response
class ReservoirSamplerBatchStreamer:
    def __init__(self, coreset_size, random_seed=42):
        self.m = coreset_size; self.random_seed=random_seed; np.random.seed(random_seed)
        self.reservoir_global_indices = []; self._items_processed_count = 0
    def process_batch(self, batch_global_start_idx, batch_size):
        if self.m == 0: return
        for i in range(batch_size):
            current_item_global_idx = batch_global_start_idx + i
            self._items_processed_count += 1
            if len(self.reservoir_global_indices) < self.m: self.reservoir_global_indices.append(current_item_global_idx)
            else:
                j = np.random.randint(0, self._items_processed_count)
                if j < self.m: self.reservoir_global_indices[j] = current_item_global_idx
    def get_coreset_indices(self): return np.array(self.reservoir_global_indices, dtype=int)
    def get_final_coreset_details(self, all_stream_data_accumulator_np, all_stream_labels_accumulator_np=None):
        indices = self.get_coreset_indices()
        valid_indices = indices[indices < len(all_stream_data_accumulator_np)]
        coreset_X = all_stream_data_accumulator_np[valid_indices]
        weights = np.ones(len(valid_indices)) / (len(valid_indices) if len(valid_indices)>0 else 1)
        if all_stream_labels_accumulator_np is not None:
            coreset_y = all_stream_labels_accumulator_np[valid_indices]
            return coreset_X, coreset_y, weights
        return coreset_X, weights
    

# The offline method remains for baseline comparison
def get_coreset_rff_sampler_log(X_train_np, m, kernel_gamma, n_rff_components=500, n_epochs=1000, lr=0.001, lambda_log=1e-4, epsilon=1e-6, random_seed=42):
    # (Implementation from your prompt)
    torch.manual_seed(random_seed); np.random.seed(random_seed)
    n = X_train_np.shape[0]
    if m > n: m = n
    if m == 0 or n == 0: return np.array([], dtype=int), np.array([]), [], []
    rbf_map = RBFSampler(gamma=kernel_gamma, n_components=n_rff_components, random_state=random_seed)
    Z_train = torch.from_numpy(rbf_map.fit_transform(X_train_np)).float()
    w_logits = nn.Parameter(torch.full((n,), 0.1, dtype=torch.float32, requires_grad=True))
    opt = optim.Adam([w_logits], lr=lr)
    mean_Z_full = torch.mean(Z_train, dim=0)
    mmd_hist, loss_hist = [], []
    for _ in range(n_epochs):
        opt.zero_grad()
        weights = torch.relu(w_logits)
        norm_weights = weights / (weights.sum() + 1e-9)
        mean_Z_core = torch.sum(norm_weights.unsqueeze(1) * Z_train, dim=0)
        mmd2 = torch.sum((mean_Z_full - mean_Z_core)**2)
        log_p = lambda_log * torch.sum(torch.log(epsilon + weights))
        loss = mmd2 + log_p # Note: Your original code had +log_penalty. This encourages sparsity.
        loss.backward(); opt.step()
        mmd_hist.append(mmd2.item()); loss_hist.append(loss.item())
    final_weights = torch.relu(w_logits).detach()
    k_top = min(m, len(final_weights))
    if final_weights.sum() < 1e-9 or k_top == 0:
        idx = np.random.choice(n, m, replace=False) if n >= m else np.arange(n)
        w = np.ones(len(idx)) / (len(idx) if len(idx)>0 else 1)
        return idx, w, mmd_hist, loss_hist
    vals, idx = torch.topk(final_weights, k=k_top)
    idx_np = idx.cpu().numpy()
    w_np = (vals / (vals.sum() + 1e-9)).cpu().numpy()
    return idx_np, w_np, mmd_hist, loss_hist


class OnlineMMDPlusStreamer:
    """ An intelligent, stateful streamer for coreset selection with persistent weights and smart buffer management. """
    def __init__(self, m_coreset_size, n_rff_components, buffer_capacity,
                 n_epochs_online=30, lr_online=0.01, lambda_log_online=1e-5, random_seed=42):
        # Core parameters
        self.m = m_coreset_size
        self.n_rff_components = n_rff_components
        self.buffer_capacity = buffer_capacity

        # Optimization parameters
        self.n_epochs_online = n_epochs_online
        self.lr_online = lr_online
        self.lambda_log = lambda_log_online
        self.epsilon = 1e-6

        # State tracking
        self.rbf_sampler = None
        self.random_seed = random_seed
        self.num_points_seen = 0
        self._current_batch_idx = -1
        self._sum_rff_full_stream = torch.zeros(self.n_rff_components, dtype=torch.float32)
        self.mean_rff_full_stream_torch = torch.zeros(self.n_rff_components, dtype=torch.float32)
        self.optimizer = None
        self.sparsity_history = []

        # The core data structure: a buffer of point objects (dicts with persistent logits)
        self.point_buffer = []

        if random_seed is not None:
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)

    def set_rbf_sampler(self, rbf_sampler_instance):
        self.rbf_sampler = rbf_sampler_instance

    def _optimize_weights(self):
        """ Runs optimization on the current point_buffer's logits. """
        if not self.point_buffer:
            return

        trainable_logits = [p['logit'] for p in self.point_buffer]
        if not trainable_logits: return

        self.optimizer = optim.Adam(trainable_logits, lr=self.lr_online)
        candidate_rffs = torch.stack([p['rff'] for p in self.point_buffer])

        for _ in range(self.n_epochs_online):
            self.optimizer.zero_grad()
            
            # This is the corrected part: torch.cat now works on 1-D tensors
            current_weights = torch.relu(torch.cat(trainable_logits))

            sum_weights = current_weights.sum()
            normalized_weights = current_weights / (sum_weights + 1e-9)
            mean_Z_coreset = torch.sum(normalized_weights.unsqueeze(1) * candidate_rffs, dim=0)
            mmd2 = torch.sum((self.mean_rff_full_stream_torch - mean_Z_coreset)**2)

            log_penalty = self.lambda_log * torch.sum(torch.log(self.epsilon + current_weights))
            loss = mmd2 + log_penalty
            loss.backward()
            self.optimizer.step()

        # Update sparsity count
        final_weights = torch.relu(torch.cat([p['logit'].detach() for p in self.point_buffer]))
        num_nonzero = torch.sum(final_weights > 1e-9).item()
        print("Number of non-zero weights after optimization:", num_nonzero, 'Length of point buffer:', len(self.point_buffer))
        self.sparsity_history.append(num_nonzero / len(self.point_buffer))


    def _prune_buffer(self):
        """ Intelligently prunes the buffer down to its capacity based on weight and redundancy. """
        num_to_prune = len(self.point_buffer) - self.buffer_capacity
        if num_to_prune <= 0: return

        with torch.no_grad():
            weights = torch.cat([torch.relu(p['logit'].detach()) for p in self.point_buffer])

        sorted_indices = torch.argsort(weights, descending=True)
        provisional_coreset_indices = sorted_indices[:self.m]
        pruning_candidate_indices = sorted_indices[self.m:]

        if len(pruning_candidate_indices) <= num_to_prune:
            indices_to_remove = set(pruning_candidate_indices.numpy())
        else:
            provisional_coreset_rffs = torch.stack([self.point_buffer[i]['rff'] for i in provisional_coreset_indices])
            pruning_rffs = torch.stack([self.point_buffer[i]['rff'] for i in pruning_candidate_indices])

            cosine_sim = F.cosine_similarity(pruning_rffs.unsqueeze(1), provisional_coreset_rffs.unsqueeze(0), dim=-1)
            redundancy_scores, _ = torch.max(cosine_sim, dim=1)
            
            indices_to_prune_from_candidates = torch.argsort(redundancy_scores, descending=True)[:num_to_prune]
            final_indices_to_prune = pruning_candidate_indices[indices_to_prune_from_candidates]
            indices_to_remove = set(final_indices_to_prune.numpy())

        self.point_buffer = [p for i, p in enumerate(self.point_buffer) if i not in indices_to_remove]


    def process_batch(self, X_batch_np):
        if self.rbf_sampler is None: raise RuntimeError("RBFSampler not set.")
        self._current_batch_idx += 1
        batch_size = X_batch_np.shape[0]
        if batch_size == 0: return

        for i in range(batch_size):
            rff = torch.from_numpy(self.rbf_sampler.transform(X_batch_np[i:i+1])).float().squeeze(0)
            point_info = {
                "rff": rff,
                "global_id": (self._current_batch_idx, i),
                # ❗ **THE FIX** ❗ Initialize as a 1-D tensor
                "logit": nn.Parameter(torch.tensor([0.1], dtype=torch.float32))
            }
            self.point_buffer.append(point_info)

        Z_batch_stacked = torch.stack([p['rff'] for p in self.point_buffer[-batch_size:]])
        self._sum_rff_full_stream += torch.sum(Z_batch_stacked, dim=0)
        self.num_points_seen += batch_size
        self.mean_rff_full_stream_torch = self._sum_rff_full_stream / self.num_points_seen
        
        self._optimize_weights()

        if len(self.point_buffer) > self.buffer_capacity:
            self._prune_buffer()

        sparsity_ratio = self.sparsity_history[-1] if self.sparsity_history else 0
        print(f"  Batch {self._current_batch_idx} processed. Sparsity: {sparsity_ratio:.2%}")


    def get_final_coreset(self):
        """ Extracts the final coreset based on the highest weights in the buffer. """
        if not self.point_buffer:
            return [], [], []

        with torch.no_grad():
            weights = torch.cat([torch.relu(p['logit'].detach()) for p in self.point_buffer])

        k_topk = min(self.m, len(self.point_buffer))
        if k_topk == 0: return [], [], []

        top_k_weights, top_k_indices_in_buffer = torch.topk(weights, k=k_topk)
        normalized_weights = top_k_weights / (top_k_weights.sum() + 1e-9)
        coreset_points_info = [self.point_buffer[i] for i in top_k_indices_in_buffer]
        coreset_global_ids = [p['global_id'] for p in coreset_points_info]

        return coreset_global_ids, normalized_weights.cpu().numpy()

    def print_coreset_provenance(self, BATCH_SIZE):
        """ Prints the origin of each coreset point for analysis. """
        if not self.point_buffer:
            print("Coreset is empty.")
            return np.array([], dtype=int)

        coreset_global_ids, coreset_weights = self.get_final_coreset()
        print("\n--- Final Coreset Provenance ---")
        flat_indices = []
        for i, gid in enumerate(coreset_global_ids):
            flat_idx = gid[0] * BATCH_SIZE + gid[1]
            flat_indices.append(flat_idx)
            print(f"  Point {i}: From Batch {gid[0]}, Idx {gid[1]} (Flat Index: {flat_idx}) -> Weight: {coreset_weights[i]:.4f}")

        batch_indices = [gid[0] for gid in coreset_global_ids]
        batch_counts = {b: batch_indices.count(b) for b in sorted(list(set(batch_indices)))}
        print("\nCoreset points per batch:", batch_counts, "\n------------------------------")
        return np.array(flat_indices, dtype=int)


def run_intelligent_streaming_experiment():
    # --- Configuration ---
    DATASET_SUBSET_SIZE = 2500
    BATCH_SIZE = 50
    N_RFF_COMPONENTS = 100
    KERNEL_GAMMA = 0.1
    
    # OnlineMMDPlusStreamer params
    # Buffer capacity will be a multiple of the coreset size
    BUFFER_CAPACITY_FACTOR = 3 
    N_EPOCHS_ONLINE_UPDATE = 20
    LR_ONLINE = 0.1
    LAMBDA_LOG_ONLINE = 5e-5 # Lambda can be tuned based on observed sparsity

    coreset_sizes_to_test = np.arange(20, 40, 5) #[10, 20, 30, 40, 50, 60, 70]

    # --- Load Data ---
    print("Loading data...")
    np.random.seed(23874291)  # For reproducibility
    X_train_full, X_val, y_train_full, y_val = load_adult_data(DATASET_SUBSET_SIZE)
    num_total_stream_points = X_train_full.shape[0]
    num_batches = (num_total_stream_points + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"Total stream points: {num_total_stream_points}, Batch size: {BATCH_SIZE}, Num batches: {num_batches}")

    # --- Global RBF Sampler ---
    global_rbf_sampler = RBFSampler(gamma=KERNEL_GAMMA, n_components=N_RFF_COMPONENTS, random_state=42)
    global_rbf_sampler.fit(X_train_full)
    print("Global RBFSampler fitted.")

    # --- Baseline: Whole Dataset ---
    acc_whole = train_classifier(X_train_full, X_val, y_train_full, y_val)
    print(f"\nBaseline (whole dataset) accuracy: {acc_whole:.4f}")

    # --- Results Storage ---
    results = { 'coreset_size': [], 'OnlineMMDPlus_Acc': [], 'OnlineMMDPlus_MMD': [], 
               'Reservoir_Acc': [], 'Reservoir_MMD': [], 'OCS_Acc': [], 'OCS_MMD': [],
                'BCSR_Acc': [], 'BCSR_MMD': [] }

    # --- Iterate Over Coreset Sizes ---
    for m_coreset in coreset_sizes_to_test:
        buffer_capacity = 150
        print(f"\n--- Processing for Coreset Size m = {m_coreset} (Buffer Capacity = {buffer_capacity}) ---")
        results['coreset_size'].append(m_coreset)

        # --- Method 1: OnlineMMDPlusStreamer ---
        print("  Streaming with OnlineMMDPlusStreamer...")
        online_plus_streamer = OnlineMMDPlusStreamer(
            m_coreset_size=m_coreset, n_rff_components=N_RFF_COMPONENTS,
            buffer_capacity=buffer_capacity, n_epochs_online=N_EPOCHS_ONLINE_UPDATE,
            lr_online=LR_ONLINE, lambda_log_online=LAMBDA_LOG_ONLINE, random_seed=42
        )
        online_plus_streamer.set_rbf_sampler(global_rbf_sampler)

        for i in range(num_batches):
            start_idx = i * BATCH_SIZE
            end_idx = min(start_idx + BATCH_SIZE, num_total_stream_points)
            X_batch = X_train_full[start_idx:end_idx]
            if X_batch.shape[0] > 0:
                online_plus_streamer.process_batch(X_batch)
        
        # Final Evaluation for OnlineMMDPlus
        coreset_flat_indices = online_plus_streamer.print_coreset_provenance(BATCH_SIZE)
        _, w_core_online = online_plus_streamer.get_final_coreset()
        X_core_online = X_train_full[coreset_flat_indices]
        y_core_online = y_train_full[coreset_flat_indices]
        
        acc_online = train_classifier(X_core_online, X_val, y_core_online, y_val)
        mmd_exact_online = calculate_mmd2_exact(X_train_full, X_core_online, w_core_online, KERNEL_GAMMA)
        results['OnlineMMDPlus_Acc'].append(acc_online)
        results['OnlineMMDPlus_MMD'].append(mmd_exact_online)
        print(f"  OnlineMMDPlus: Acc={acc_online:.4f}, Exact MMD={mmd_exact_online:.6f}")
        
                # --- Method 2: Reservoir Sampling ---
        print("  Streaming with Reservoir Sampling...")

        num_trials = 10  # Number of trials, can be adjusted as needed

        acc_trials = []
        mmd_exact_trials = []

        for trial in range(num_trials):
            reservoir_streamer = ReservoirSamplerBatchStreamer(coreset_size=m_coreset, random_seed=42 + trial)
            for i in range(num_batches):
                start_idx = i * BATCH_SIZE
                end_idx = min(start_idx + BATCH_SIZE, num_total_stream_points)
                if end_idx > start_idx:
                    reservoir_streamer.process_batch(start_idx, end_idx - start_idx)
            
            X_core_res, y_core_res, w_core_res = reservoir_streamer.get_final_coreset_details(X_train_full, y_train_full)
            acc_res = train_classifier(X_core_res, X_val, y_core_res, y_val)
            mmd_exact_res = calculate_mmd2_exact(X_train_full, X_core_res, w_core_res, KERNEL_GAMMA)
            acc_trials.append(acc_res)
            mmd_exact_trials.append(mmd_exact_res)

        # Compute averages across trials
        average_acc = np.mean(acc_trials)
        average_mmd_exact = np.mean(mmd_exact_trials)

        # Store averaged results
        results['Reservoir_Acc'].append(average_acc)
        results['Reservoir_MMD'].append(average_mmd_exact)

        print(f"  Reservoir (averaged over {num_trials} trials): Acc={average_acc:.4f}, Exact MMD={average_mmd_exact:.6f}")

                # --- Method 3: Online Coreset Selection (OCS) ---
        print("  Streaming with OCSStreamer...")
        # Note: You'll need to pass y_train_full to the streamer
        # --- Inside your loop for Method 3: Online Coreset Selection (OCS) ---

        ocs_streamer = OCSStreamer(m_coreset_size=m_coreset, batch_size=BATCH_SIZE, tau=1.0, random_seed=42)

        for i in range(num_batches):
            start_idx = i * BATCH_SIZE
            end_idx = min(start_idx + BATCH_SIZE, num_total_stream_points)
            X_batch = X_train_full[start_idx:end_idx]
            y_batch = y_train_full[start_idx:end_idx]
            batch_indices = np.arange(start_idx, end_idx)
            if X_batch.shape[0] > 0:
                # CORRECTED: Pass X_train_full and y_train_full to the method
                ocs_streamer.process_batch(X_batch, y_batch, batch_indices, X_train_full, y_train_full)

        # Final Evaluation for OCS
        X_core_ocs, y_core_ocs, w_core_ocs = ocs_streamer.get_final_coreset_details(X_train_full, y_train_full)
        acc_ocs = train_classifier(X_core_ocs, X_val, y_core_ocs, y_val)
        mmd_exact_ocs = calculate_mmd2_exact(X_train_full, X_core_ocs, w_core_ocs, KERNEL_GAMMA)

        # Add results to your results dictionary
        results['OCS_Acc'].append(acc_ocs)
        results['OCS_MMD'].append(mmd_exact_ocs)
        print(f"OCS: Acc={acc_ocs:.4f}, Exact MMD={mmd_exact_ocs:.6f}")


        # --- Method 3: Bilevel Coreset Selection (BCSR) ---
        print("    Streaming with Bilevel Coreset Selection...")
        input_dim = X_train_full.shape[1] # Number of features in the dataset
        
        # Initialize BCSR streamer with tunable parameters.
        # These values are reasonable starting points but can be further tuned.
        bcsr_streamer = BilevelCoresetSelector(
            input_dim=input_dim,
            m_coreset_size=m_coreset,
            outer_loops=5,  # J: Number of outer iterations for 'w'
            inner_loops=5,  # N: Number of inner iterations for 'theta' (model_cs)
            lr_outer=0.01,  # Learning rate for coreset weights (w)
            lr_inner=0.001, # Learning rate for inner model (model_cs)
            lambda_reg=0.001, # Regularization strength for 'smoothed top-K' (sparsity)
            random_seed=42
        )
        # Set the full dataset for upper-level objective evaluation
        bcsr_streamer.set_full_data(X_train_full, y_train_full)

        # Stream batches to the BCSR algorithm
        for i in range(num_batches):
            start_idx = i * BATCH_SIZE
            end_idx = min(start_idx + BATCH_SIZE, num_total_stream_points)
            X_batch = X_train_full[start_idx:end_idx]
            y_batch = y_train_full[start_idx:end_idx] # Pass corresponding labels

            if X_batch.shape[0] > 0:
                bcsr_streamer.process_batch(X_batch, y_batch) # Process batch with labels

        # Final Evaluation for BCSR
        coreset_flat_indices_bcsr, w_core_bcsr = bcsr_streamer.get_final_coreset()
        
        if len(coreset_flat_indices_bcsr) > 0:
            X_core_bcsr = X_train_full[coreset_flat_indices_bcsr]
            y_core_bcsr = y_train_full[coreset_flat_indices_bcsr]
            
            acc_bcsr = train_classifier(X_core_bcsr, X_val, y_core_bcsr, y_val)
            mmd_exact_bcsr = calculate_mmd2_exact(X_train_full, X_core_bcsr, w_core_bcsr, KERNEL_GAMMA)
        else:
            acc_bcsr = 0.0
            mmd_exact_bcsr = float('inf')

        results['BCSR_Acc'].append(acc_bcsr)
        results['BCSR_MMD'].append(mmd_exact_bcsr)
        print(f"    BCSR: Acc={acc_bcsr:.4f}, Exact MMD={mmd_exact_bcsr:.6f}")

    # --- Plotting Results ---
    # (Plotting code would be similar to before, comparing OnlineMMDPlus_Acc/MMD with Reservoir_Acc/MMD)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    ax1.plot(results['coreset_size'], results['OnlineMMDPlus_Acc'], marker='o', linestyle='-', label='OnlineMMDPlus Streamer')
    ax1.plot(results['coreset_size'], results['Reservoir_Acc'], marker='s', linestyle='--', label='Reservoir Sampling')
    ax1.plot(results['coreset_size'], results['OCS_Acc'], marker='^', linestyle=':', label='OCS Streamer')
    ax1.plot(results['coreset_size'], results['BCSR_Acc'], marker='x', linestyle='-.', label='BCSR Streamer')
    ax1.axhline(acc_whole, color='k', linestyle='-', label=f'Whole Dataset ({acc_whole:.3f})')
    ax1.set_xlabel('Coreset Size (m)'); ax1.set_ylabel('Validation Accuracy')
    ax1.set_title('Accuracy vs. Coreset Size'); ax1.legend(); ax1.grid(True)
    
    ax2.plot(results['coreset_size'], results['OnlineMMDPlus_MMD'], marker='o', linestyle='-', label='OnlineMMDPlus Streamer')
    ax2.plot(results['coreset_size'], results['Reservoir_MMD'], marker='s', linestyle='--', label='Reservoir Sampling')
    ax2.plot(results['coreset_size'], results['OCS_MMD'], marker='^', linestyle=':', label='OCS Streamer')
    ax2.plot(results['coreset_size'], results['BCSR_MMD'], marker='x', linestyle='-.', label='BCSR Streamer')
    ax2.set_xlabel('Coreset Size (m)'); ax2.set_ylabel('Exact MMD² (RBF Kernel)')
    ax2.set_title('Exact MMD² vs. Coreset Size'); ax2.legend(); ax2.grid(True); ax2.set_yscale('log')
    plt.tight_layout(); plt.show()

if __name__ == '__main__':
    run_intelligent_streaming_experiment()