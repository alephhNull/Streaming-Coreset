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
import time # For tracking processing time

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


# --- Downstream Classification Model Training ---
def train_classifier(X_train: np.ndarray, X_val: np.ndarray, y_train: np.ndarray, y_val: np.ndarray) -> float:
    if len(np.unique(y_train)) < 2:
        print(f"  Warning: Training data for classifier contains only {len(np.unique(y_train))} unique class(es) (labels: {np.unique(y_train)}). Skipping training and returning NaN.")
        return np.nan
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

# Your offline coreset selection method
def get_coreset_rff_sampler_log(X_train_np, m, kernel_gamma, n_rff_components=500, n_epochs=1000, lr=0.001, lambda_log=1e-4, epsilon=1e-6, random_seed=42):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    mmd_history = []
    loss_history = []

    n = X_train_np.shape[0]
    if m > n:
        m = n
    if m == 0:
        return np.array([], dtype=int), np.array([]), mmd_history, loss_history
    if n == 0:
        return np.array([], dtype=int), np.array([]), mmd_history, loss_history

    rbf_feature_map = RBFSampler(gamma=kernel_gamma, n_components=n_rff_components, random_state=random_seed)
    Z_train_np = rbf_feature_map.fit_transform(X_train_np)
    Z_train = torch.from_numpy(Z_train_np).float()

    w_logits = nn.Parameter(torch.full((n,), 0.1, dtype=torch.float32, requires_grad=True))
    optimizer = optim.Adam([w_logits], lr=lr)
    mean_Z_full = torch.mean(Z_train, dim=0)

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        current_learned_weights = torch.relu(w_logits)
        sum_current_learned_weights = current_learned_weights.sum()
        if sum_current_learned_weights < 1e-9:
            normalized_eval_weights = torch.zeros_like(current_learned_weights)
        else:
            normalized_eval_weights = current_learned_weights / sum_current_learned_weights
        mean_Z_coreset = torch.sum(normalized_eval_weights.unsqueeze(1) * Z_train, dim=0)
        mmd2_objective = torch.sum((mean_Z_full - mean_Z_coreset)**2)
        log_penalty = lambda_log * torch.sum(torch.log(epsilon + current_learned_weights)) # Maximize this term by minimizing its negative
        loss = mmd2_objective + log_penalty # Maximize log_penalty = minimize -log_penalty

        mmd_history.append(mmd2_objective.item())
        loss_history.append(loss.item())
        loss.backward()
        optimizer.step()

    final_learned_weights = torch.relu(w_logits).detach()
    if final_learned_weights.sum() < 1e-9:
        coreset_indices_np = np.random.choice(n, m, replace=False) if n >= m else np.arange(n)
        weights_coreset_np = np.ones(len(coreset_indices_np)) / len(coreset_indices_np) if len(coreset_indices_np) > 0 else np.array([])
        return coreset_indices_np, weights_coreset_np.flatten(), mmd_history, loss_history

    k_topk = min(m, len(final_learned_weights))
    if k_topk == 0:
        return np.array([], dtype=int), np.array([]), mmd_history, loss_history

    top_k_selected_values, top_k_indices_torch = torch.topk(final_learned_weights, k=k_topk)
    coreset_indices_np = top_k_indices_torch.cpu().numpy()
    selected_coreset_weights_torch = top_k_selected_values.cpu()

    sum_selected_coreset_weights = selected_coreset_weights_torch.sum()
    if sum_selected_coreset_weights < 1e-9 :
        weights_coreset_normalized_torch = torch.ones_like(selected_coreset_weights_torch) / k_topk if k_topk > 0 else torch.empty(0)
    else:
        weights_coreset_normalized_torch = selected_coreset_weights_torch / sum_selected_coreset_weights
    weights_coreset_np = weights_coreset_normalized_torch.numpy()
    return coreset_indices_np, weights_coreset_np.flatten(), mmd_history, loss_history



def get_coreset_rff_online_update(
    candidate_rff_features_torch: torch.Tensor, # RFFs of points in the candidate set
    mean_rff_full_stream_torch: torch.Tensor, # Target mean of the whole stream
    m: int, # Desired coreset size
    n_epochs_online: int = 50, # Fewer epochs for online updates
    lr: float = 0.01,
    lambda_log: float = 1e-5, # Adjusted lambda
    epsilon: float = 1e-6,
    random_seed: int = None # Can pass seed for consistency if needed per call
) -> tuple[np.ndarray, np.ndarray]: # (selected_indices_in_candidates, selected_weights_np)

    if random_seed is not None:
        torch.manual_seed(random_seed)

    num_candidates = candidate_rff_features_torch.shape[0]

    mmd_history = []
    loss_history = []

    if num_candidates == 0:
        return np.array([], dtype=int), np.array([]), mmd_history, loss_history
    if m == 0:
        return np.array([], dtype=int), np.array([]), mmd_history, loss_history
    if m >= num_candidates: # If coreset size is >= candidates, take all with uniform weights (or learned if preferred)
        # For simplicity, let's use learned weights even if m >= num_candidates, or just return all
        pass # Proceed with optimization

    # Initialize learnable logits for weights for the candidates
    w_logits = nn.Parameter(torch.full((num_candidates,), 0.1, dtype=torch.float32, requires_grad=True))
    optimizer = optim.Adam([w_logits], lr=lr)

    # Target mean is provided
    mean_Z_full_stream = mean_rff_full_stream_torch

    for epoch in range(n_epochs_online):
        optimizer.zero_grad()
        current_learned_weights = torch.relu(w_logits) # Ensure positive weights

        # Normalize weights for MMD calculation (sum to 1)
        sum_current_learned_weights = current_learned_weights.sum()
        if sum_current_learned_weights < 1e-9: # Avoid division by zero
            normalized_eval_weights = torch.zeros_like(current_learned_weights)
        else:
            normalized_eval_weights = current_learned_weights / sum_current_learned_weights

        # Compute E_Q[Z(Y)] for the coreset based on current candidates
        mean_Z_coreset_candidates = torch.sum(normalized_eval_weights.unsqueeze(1) * candidate_rff_features_torch, dim=0)

        mmd2_objective = torch.sum((mean_Z_full_stream - mean_Z_coreset_candidates)**2)
        log_penalty = lambda_log * torch.sum(torch.log(epsilon + current_learned_weights))
        loss = mmd2_objective + log_penalty # trying to maximize log penalty (less sparse) or minimize -log_penalty (more sparse)

        mmd_history.append(mmd2_objective.item())
        loss_history.append(loss.item())

        loss.backward()
        optimizer.step()

    final_learned_weights = torch.relu(w_logits).detach()
    # final_learned_weights_np = final_learned_weights.cpu().numpy()
    # print('Top final weights:', np.sort(final_learned_weights_np)[::-1][:100])

    actual_m = min(m, num_candidates) # Number of points to select

    if final_learned_weights.sum() < 1e-9 or actual_m == 0: # Fallback if all weights are zero or m is 0
        if actual_m > 0 and num_candidates >0: # Pick random if weights failed
            selected_indices_in_candidates = np.random.choice(num_candidates, actual_m, replace=False)
            selected_weights_np = np.ones(actual_m) / actual_m
        else:
            selected_indices_in_candidates = np.array([], dtype=int)
            selected_weights_np = np.array([])
        return selected_indices_in_candidates, selected_weights_np, mmd_history, loss_history


    top_k_selected_values, top_k_indices_torch = torch.topk(final_learned_weights, k=actual_m)
    selected_indices_in_candidates = top_k_indices_torch.cpu().numpy()
    selected_coreset_weights_torch = top_k_selected_values.cpu()

    # Normalize selected weights to sum to 1
    sum_selected_coreset_weights = selected_coreset_weights_torch.sum()
    if sum_selected_coreset_weights < 1e-9 :
        weights_coreset_normalized_torch = torch.ones_like(selected_coreset_weights_torch) / actual_m if actual_m > 0 else torch.empty(0)
    else:
        weights_coreset_normalized_torch = selected_coreset_weights_torch / sum_selected_coreset_weights

    selected_weights_np = weights_coreset_normalized_torch.numpy()

    return selected_indices_in_candidates, selected_weights_np, mmd_history, loss_history


class OnlineMMDRFFBatchStreamer:
    def __init__(self, m_coreset_size, n_rff_components,
                 aux_buffer_capacity_factor=2,
                 n_sample_from_aux_buffer=10,
                 n_epochs_online=30, lr_online=0.01, lambda_log_online=1e-5, random_seed=42):
        self.m = m_coreset_size
        self.n_rff_components = n_rff_components
        self.n_epochs_online = n_epochs_online
        self.lr_online = lr_online
        self.lambda_log_online = lambda_log_online
        self.random_seed = random_seed

        self.rbf_sampler = None # Must be fitted and set externally

        # Current coreset (RFFs and global indices)
        self.coreset_global_indices = np.array([], dtype=int)
        self.coreset_rffs = torch.empty((0, self.n_rff_components), dtype=torch.float32)
        self.coreset_weights = np.array([])

        # For online mean embedding of the entire stream
        self._sum_rff_full_stream = torch.zeros(self.n_rff_components, dtype=torch.float32)
        self.mean_rff_full_stream_torch = torch.zeros(self.n_rff_components, dtype=torch.float32)
        self.num_points_seen = 0
        self._current_global_idx_offset = 0 # Tracks starting global index for current batch

        # Auxiliary buffer
        self.aux_buffer_capacity = int(aux_buffer_capacity_factor * self.m)
        self.n_sample_from_aux_buffer = min(n_sample_from_aux_buffer, self.aux_buffer_capacity)
        self.aux_buffer = [] # Stores tuples of (rff_tensor, global_stream_idx)

        if random_seed is not None:
            np.random.seed(random_seed) # For np operations like choice

    def set_rbf_sampler(self, rbf_sampler_instance):
        self.rbf_sampler = rbf_sampler_instance
        # Reset sums if sampler changes (though it should be fixed for a run)
        self._sum_rff_full_stream = torch.zeros(self.n_rff_components, dtype=torch.float32)
        self.mean_rff_full_stream_torch = torch.zeros(self.n_rff_components, dtype=torch.float32)

    def _add_to_aux_buffer(self, rff_tensor, global_idx):
        if self.aux_buffer_capacity == 0: return # No aux buffer
        if len(self.aux_buffer) >= self.aux_buffer_capacity:
            # Evict a random element to make space
            idx_to_remove = np.random.randint(0, len(self.aux_buffer))
            self.aux_buffer.pop(idx_to_remove)
        self.aux_buffer.append((rff_tensor.clone(), global_idx)) # Store a clone

    def process_batch(self, X_batch_np):
        if self.rbf_sampler is None:
            raise RuntimeError("RBFSampler not set. Call set_rbf_sampler first.")

        batch_size = X_batch_np.shape[0]
        if batch_size == 0:
            return

        # Get RFFs for the current batch
        Z_batch_rff_list = [torch.from_numpy(self.rbf_sampler.transform(X_batch_np[j:j+1])).float().squeeze(0) for j in range(batch_size)]
        Z_batch_stacked = torch.stack(Z_batch_rff_list) if Z_batch_rff_list else torch.empty((0, self.n_rff_components), dtype=torch.float32)

        # Update overall stream mean RFF
        self._sum_rff_full_stream += torch.sum(Z_batch_stacked, dim=0)
        self.num_points_seen += batch_size
        if self.num_points_seen > 0:
            self.mean_rff_full_stream_torch = self._sum_rff_full_stream / self.num_points_seen
        else: # Should not happen if batch_size > 0
            self.mean_rff_full_stream_torch.zero_()


        # --- Construct Candidate Pool ---
        candidate_rffs_list = []
        candidate_global_indices_list = [] # Global indices of points in candidate_rffs_list

        # 1. Add current coreset to candidates
        if self.coreset_rffs.nelement() > 0: # If coreset is not empty
            for i in range(self.coreset_rffs.shape[0]):
                candidate_rffs_list.append(self.coreset_rffs[i])
                candidate_global_indices_list.append(self.coreset_global_indices[i])

        # 2. Add new batch points to candidates
        current_batch_global_indices = []
        for i in range(batch_size):
            global_idx = self._current_global_idx_offset + i
            current_batch_global_indices.append(global_idx)
            candidate_rffs_list.append(Z_batch_rff_list[i])
            candidate_global_indices_list.append(global_idx)

        # 3. Add samples from auxiliary buffer to candidates
        actual_samples_from_aux = min(self.n_sample_from_aux_buffer, len(self.aux_buffer))
        if actual_samples_from_aux > 0:
            sampled_indices_in_aux_buffer = np.random.choice(len(self.aux_buffer), size=actual_samples_from_aux, replace=False)
            for aux_idx in sampled_indices_in_aux_buffer:
                rff_aux, global_idx_aux = self.aux_buffer[aux_idx]
                # Avoid adding duplicates if an aux point happens to be in current coreset (via global_idx check)
                if global_idx_aux not in candidate_global_indices_list:
                    candidate_rffs_list.append(rff_aux)
                    candidate_global_indices_list.append(global_idx_aux)

        if not candidate_rffs_list:
            self._current_global_idx_offset += batch_size
            return # No candidates to process

        candidate_rffs_torch_stacked = torch.stack(candidate_rffs_list)
        candidate_global_indices_np = np.array(candidate_global_indices_list, dtype=int)

        # --- Optimize Coreset ---
        num_candidates = candidate_rffs_torch_stacked.shape[0]
        new_coreset_global_indices_set = set()

        if num_candidates == 0: # Should be caught by "if not candidate_rffs_list"
             pass
        elif num_candidates <= self.m : # Not enough candidates, take all
            self.coreset_global_indices = candidate_global_indices_np
            self.coreset_rffs = candidate_rffs_torch_stacked
            if num_candidates > 0:
                self.coreset_weights = np.ones(num_candidates) / num_candidates
            else:
                self.coreset_weights = np.array([])
            new_coreset_global_indices_set.update(self.coreset_global_indices)
        else:
            opt_seed = (self.random_seed + self.num_points_seen) if self.random_seed is not None else None
            selected_indices_in_candidates, selected_weights_np, mmd_history, loss_history = get_coreset_rff_online_update(
                candidate_rff_features_torch=candidate_rffs_torch_stacked,
                mean_rff_full_stream_torch=self.mean_rff_full_stream_torch,
                m=self.m,
                n_epochs_online=self.n_epochs_online,
                lr=self.lr_online,
                lambda_log=self.lambda_log_online,
                random_seed=opt_seed
            )
            # plt.plot(mmd_history)
            # plt.xlabel('Epoch')
            # plt.ylabel('MMD^2')
            # plt.title('MMD^2 Over Epochs')
            # plt.grid(True)
            # plt.show()
            self.coreset_global_indices = candidate_global_indices_np[selected_indices_in_candidates]
            self.coreset_weights = selected_weights_np
            self.coreset_rffs = candidate_rffs_torch_stacked[selected_indices_in_candidates]
            new_coreset_global_indices_set.update(self.coreset_global_indices)
        
        # --- Update Auxiliary Buffer ---
        # Add points from the current batch that were NOT selected for the new coreset
        for i in range(batch_size):
            global_idx = self._current_global_idx_offset + i # This is the global index from current_batch_global_indices
            if global_idx not in new_coreset_global_indices_set:
                self._add_to_aux_buffer(Z_batch_rff_list[i], global_idx)

        # Remove points from aux_buffer that are now in the coreset (if they were sampled and selected)
        self.aux_buffer = [(rff, idx) for rff, idx in self.aux_buffer if idx not in new_coreset_global_indices_set]

        self._current_global_idx_offset += batch_size


    def get_current_rff_mmd(self):
        if self.num_points_seen == 0 or self.coreset_rffs.nelement() == 0:
            return np.inf
        weights_torch = torch.from_numpy(self.coreset_weights).float().to(self.coreset_rffs.device)
        if weights_torch.shape[0] != self.coreset_rffs.shape[0]: # Mismatch, coreset might be empty
            return np.inf
            
        mean_Z_coreset = torch.sum(weights_torch.unsqueeze(1) * self.coreset_rffs, dim=0)
        mmd2_rff = torch.sum((self.mean_rff_full_stream_torch - mean_Z_coreset)**2)
        return mmd2_rff.item()

    # This method now requires the full accumulated stream data for evaluation purposes only
    def get_final_coreset_details(self, all_stream_data_accumulator_np, all_stream_labels_accumulator_np=None):
        if len(self.coreset_global_indices) == 0:
            # Try to determine feature dimension if possible, otherwise default to 0 or handle error
            feature_dim = all_stream_data_accumulator_np.shape[1] if all_stream_data_accumulator_np.ndim > 1 and all_stream_data_accumulator_np.shape[0] > 0 else 0
            coreset_X = np.empty((0, feature_dim))
            coreset_y = np.array([]) if all_stream_labels_accumulator_np is not None else None
        else:
            # Ensure indices are within bounds of the accumulator
            valid_indices = self.coreset_global_indices[self.coreset_global_indices < len(all_stream_data_accumulator_np)]
            if len(valid_indices) != len(self.coreset_global_indices):
                print(f"Warning: Some coreset indices {self.coreset_global_indices} out of bounds for accumulator size {len(all_stream_data_accumulator_np)}. Using valid subset.")
            
            coreset_X = all_stream_data_accumulator_np[valid_indices]
            if all_stream_labels_accumulator_np is not None:
                coreset_y = all_stream_labels_accumulator_np[valid_indices]
            else:
                coreset_y = None
            # Adjust weights if some indices were invalid (though ideally this shouldn't happen with correct offset logic)
            # For simplicity, we assume indices are correct and weights correspond.

        if all_stream_labels_accumulator_np is not None:
            return coreset_X, coreset_y, self.coreset_weights
        return coreset_X, self.coreset_weights
    

class ReservoirSamplerBatchStreamer:
    def __init__(self, coreset_size, random_seed=42):
        self.m = coreset_size
        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)

        self.reservoir_global_indices = [] # Stores global stream indices
        self._items_processed_count = 0 # Internal counter for reservoir logic

    def process_batch(self, batch_global_start_idx, batch_size):
        if self.m == 0: return # No coreset to build

        for i in range(batch_size):
            current_item_global_idx = batch_global_start_idx + i
            self._items_processed_count += 1

            if len(self.reservoir_global_indices) < self.m:
                self.reservoir_global_indices.append(current_item_global_idx)
            else:
                # Probability of selecting current item is m / _items_processed_count
                # So, pick a random slot if random number < m
                j = np.random.randint(0, self._items_processed_count) # j is from 0 to count-1
                if j < self.m:
                    self.reservoir_global_indices[j] = current_item_global_idx

    def get_coreset_indices(self):
        return np.array(self.reservoir_global_indices, dtype=int)

    def get_final_coreset_details(self, all_stream_data_accumulator_np, all_stream_labels_accumulator_np=None):
        indices = self.get_coreset_indices()
        if len(indices) == 0:
            feature_dim = all_stream_data_accumulator_np.shape[1] if all_stream_data_accumulator_np.ndim > 1 and all_stream_data_accumulator_np.shape[0] > 0 else 0
            coreset_X = np.empty((0, feature_dim))
            weights = np.array([])
            coreset_y = np.array([]) if all_stream_labels_accumulator_np is not None else None
        else:
             # Ensure indices are within bounds of the accumulator
            valid_indices = indices[indices < len(all_stream_data_accumulator_np)]
            if len(valid_indices) != len(indices):
                 print(f"Warning: Some reservoir indices {indices} out of bounds for accumulator size {len(all_stream_data_accumulator_np)}. Using valid subset.")

            coreset_X = all_stream_data_accumulator_np[valid_indices]
            weights = np.ones(len(valid_indices)) / len(valid_indices) if len(valid_indices) > 0 else np.array([])
            if all_stream_labels_accumulator_np is not None:
                coreset_y = all_stream_labels_accumulator_np[valid_indices]
            else:
                coreset_y = None
        
        if all_stream_labels_accumulator_np is not None:
            return coreset_X, coreset_y, weights
        return coreset_X, weights


def run_batched_streaming_experiment():
    # --- Configuration ---
    DATASET_SUBSET_SIZE = 2500
    BATCH_SIZE = 50
    N_RFF_COMPONENTS = 100
    KERNEL_GAMMA = .1
    # OnlineMMDRFFBatchStreamer params
    AUX_BUFFER_FACTOR = 2 # aux_buffer_capacity = factor * m_coreset
    N_SAMPLE_FROM_AUX = 60 # Number of points to sample from aux buffer
    N_EPOCHS_ONLINE_UPDATE = 200 # Fewer epochs for faster batch processing
    LR_ONLINE = .1
    LAMBDA_LOG_ONLINE = 1e-6

    coreset_sizes_to_test = [10, 12, 14, 16, 18] #[18, 19, 20, 21, 22]
    STREAM_LOG_INTERVAL_BATCHES = 5 # Log MMD every N batches

    # --- Load Data ---
    print("Loading data...")
    np.random.seed(38942102)
    X_train_full_original, X_val, y_train_full_original, y_val = load_adult_data(DATASET_SUBSET_SIZE)
    
    if X_train_full_original.shape[0] == 0:
        print("No training data loaded. Exiting.")
        return
    
    num_total_stream_points = X_train_full_original.shape[0]
    num_batches = (num_total_stream_points + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"Total stream points: {num_total_stream_points}, Batch size: {BATCH_SIZE}, Num batches: {num_batches}")


    # --- Baseline: Whole Dataset ---
    print("\nCalculating baseline (whole dataset) accuracy...")
    acc_whole = train_classifier(X_train_full_original, X_val, y_train_full_original, y_val)
    print(f"Baseline (whole dataset) accuracy: {acc_whole:.4f}")

    # --- Results Storage ---
    results = {
        'coreset_size': [],
        'OnlineMMD_RFF_Acc': [], 'OnlineMMD_RFF_MMD_Exact': [], 'OnlineMMD_RFF_Time': [], 'OnlineMMD_RFF_MMD_Stream': [],
        'Reservoir_Acc': [], 'Reservoir_MMD_Exact': [], 'Reservoir_Time': [], 'Reservoir_MMD_Stream': [],
        'OfflineRFF_Acc': [], 'OfflineRFF_MMD_Exact': [], 'OfflineRFF_Time': [],
        'Random_Acc': [], 'Random_MMD_Exact': [], 'Random_Time': []
    }
    
    # --- Global RBF Sampler ---
    # Fit on the first batch, or a larger initial portion if available, or whole for perfect comparison
    # For true streaming, fit on an initial segment. Here, fit on all for feature consistency.
    global_rbf_sampler = RBFSampler(gamma=KERNEL_GAMMA, n_components=N_RFF_COMPONENTS, random_state=42)
    global_rbf_sampler.fit(X_train_full_original) # Fit on all data for stable RFFs across runs
    print("Global RBFSampler fitted.")

    # --- Iterate Over Coreset Sizes ---
    for m_coreset in coreset_sizes_to_test:
        print(f"\n--- Processing for Coreset Size m = {m_coreset} ---")
        results['coreset_size'].append(m_coreset)

        # Accumulators for full stream data (for evaluation ONLY)
        current_X_stream_accumulator = []
        current_y_stream_accumulator = []
        
        # --- Method 1: Online-MMD-RFF Batch Streamer ---
        print("  Streaming with Online-MMD-RFF (Batched)...")
        online_streamer = OnlineMMDRFFBatchStreamer(
            m_coreset_size=m_coreset, n_rff_components=N_RFF_COMPONENTS,
            aux_buffer_capacity_factor=AUX_BUFFER_FACTOR,
            n_sample_from_aux_buffer=N_SAMPLE_FROM_AUX,
            n_epochs_online=N_EPOCHS_ONLINE_UPDATE, lr_online=LR_ONLINE,
            lambda_log_online=LAMBDA_LOG_ONLINE, random_seed=42  )
        online_streamer.set_rbf_sampler(global_rbf_sampler)

        start_time_online = time.time()
        current_mmds_online_stream_log = []
        processed_batches_count_online = 0

        for i in range(num_batches):
            start_idx = i * BATCH_SIZE
            end_idx = min((i + 1) * BATCH_SIZE, num_total_stream_points)
            X_batch = X_train_full_original[start_idx:end_idx]
            # y_batch = y_train_full_original[start_idx:end_idx] # Not needed by streamer itself

            if X_batch.shape[0] > 0:
                online_streamer.process_batch(X_batch)
                processed_batches_count_online +=1

                # For evaluation, accumulate data (NOT PART OF ALGORITHM'S MEMORY)
                current_X_stream_accumulator.extend(X_batch)
                # current_y_stream_accumulator.extend(y_batch)


            if (processed_batches_count_online % STREAM_LOG_INTERVAL_BATCHES == 0) or (processed_batches_count_online == num_batches):
                mmd_rff_val = online_streamer.get_current_rff_mmd()
                current_mmds_online_stream_log.append(mmd_rff_val)
        
        end_time_online = time.time()
        results['OnlineMMD_RFF_Time'].append(end_time_online - start_time_online)
        results['OnlineMMD_RFF_MMD_Stream'].append(current_mmds_online_stream_log)
        
        # Final evaluation for OnlineMMD
        # Convert accumulator to numpy array for indexing
        final_X_stream_acc_np = np.array(current_X_stream_accumulator)
        final_y_stream_acc_np = y_train_full_original[:len(current_X_stream_accumulator)] # Use original y directly based on accumulated length

        X_core_online, y_core_online, w_core_online = online_streamer.get_final_coreset_details(final_X_stream_acc_np, final_y_stream_acc_np)
        acc_online = train_classifier(X_core_online, X_val, y_core_online, y_val)
        mmd_exact_online = calculate_mmd2_exact(final_X_stream_acc_np, X_core_online, w_core_online, KERNEL_GAMMA)
        results['OnlineMMD_RFF_Acc'].append(acc_online)
        results['OnlineMMD_RFF_MMD_Exact'].append(mmd_exact_online)
        print(f"  Online-MMD-RFF (Batched): Acc={acc_online:.4f}, Exact MMD={mmd_exact_online:.6f}, Time={end_time_online - start_time_online:.2f}s")


    #     # --- Method 2: Reservoir Sampling Batch Streamer ---
        print("  Streaming with Reservoir Sampling (Batched)...")

        num_trials = 10  # Number of trials, can be adjusted as needed

        mmd_logs_trials = []
        acc_trials = []
        mmd_exact_trials = []
        time_trials = []

        for trial in range(num_trials):
            reservoir_streamer = ReservoirSamplerBatchStreamer(coreset_size=m_coreset, random_seed=42 + trial)
            start_time_reservoir = time.time()
            current_mmds_reservoir_stream_log = []
            processed_batches_count_reservoir = 0

            # For MMD RFF calculation of reservoir during stream
            temp_sum_rff_reservoir_mmd = torch.zeros(N_RFF_COMPONENTS, dtype=torch.float32)
            temp_points_seen_reservoir_mmd = 0

            for i in range(num_batches):
                start_idx = i * BATCH_SIZE
                end_idx = min((i + 1) * BATCH_SIZE, num_total_stream_points)
                X_batch = X_train_full_original[start_idx:end_idx]

                if X_batch.shape[0] > 0:
                    reservoir_streamer.process_batch(start_idx, X_batch.shape[0])
                    processed_batches_count_reservoir += 1

                    # Update sum and count for RFF MMD calculation
                    Z_batch_rff_torch = torch.from_numpy(global_rbf_sampler.transform(X_batch)).float()
                    temp_sum_rff_reservoir_mmd += torch.sum(Z_batch_rff_torch, dim=0)
                    temp_points_seen_reservoir_mmd += X_batch.shape[0]

                if (processed_batches_count_reservoir % STREAM_LOG_INTERVAL_BATCHES == 0) or (processed_batches_count_reservoir == num_batches):
                    if temp_points_seen_reservoir_mmd > 0:
                        mean_rff_stream_for_reservoir = temp_sum_rff_reservoir_mmd / temp_points_seen_reservoir_mmd
                        res_indices = reservoir_streamer.get_coreset_indices()
                        if len(res_indices) > 0:
                            X_core_res_temp = X_train_full_original[res_indices]
                            if X_core_res_temp.shape[0] > 0:
                                rff_features_coreset_res = torch.from_numpy(global_rbf_sampler.transform(X_core_res_temp)).float()
                                mean_Z_coreset_res = torch.mean(rff_features_coreset_res, dim=0)  # Uniform weights
                                mmd_rff_val_res = torch.sum((mean_rff_stream_for_reservoir - mean_Z_coreset_res)**2).item()
                                current_mmds_reservoir_stream_log.append(mmd_rff_val_res)
                            else:
                                current_mmds_reservoir_stream_log.append(np.inf)
                        else:
                            current_mmds_reservoir_stream_log.append(np.inf)
                    else:
                        current_mmds_reservoir_stream_log.append(np.inf)

            end_time_reservoir = time.time()
            time_trials.append(end_time_reservoir - start_time_reservoir)
            mmd_logs_trials.append(current_mmds_reservoir_stream_log)

            # Final evaluation for this trial
            X_core_res, y_core_res, w_core_res = reservoir_streamer.get_final_coreset_details(final_X_stream_acc_np, final_y_stream_acc_np)
            acc_res = train_classifier(X_core_res, X_val, y_core_res, y_val)
            mmd_exact_res = calculate_mmd2_exact(final_X_stream_acc_np, X_core_res, w_core_res, KERNEL_GAMMA)
            acc_trials.append(acc_res)
            mmd_exact_trials.append(mmd_exact_res)

        # Compute averages across trials
        # For MMD logs, average across trials for each logging point
        num_log_points = len(mmd_logs_trials[0])
        average_mmd_stream_log = [np.mean([mmd_logs_trials[trial][i] for trial in range(num_trials)]) for i in range(num_log_points)]

        # For final metrics
        average_acc = np.mean(acc_trials)
        average_mmd_exact = np.mean(mmd_exact_trials)
        average_time = np.mean(time_trials)

        # Store averaged results
        results['Reservoir_Time'].append(average_time)
        results['Reservoir_MMD_Stream'].append(average_mmd_stream_log)
        results['Reservoir_Acc'].append(average_acc)
        results['Reservoir_MMD_Exact'].append(average_mmd_exact)

        print(f"  Reservoir (Batched, averaged over {num_trials} trials): Acc={average_acc:.4f}, Exact MMD={average_mmd_exact:.6f}, Time={average_time:.2f}s")


            # --- Method 4: Random Sampling (at the end, on full original data) ---
        print("  Processing with Random Sampling (End)...")

        num_trials = 10  # Number of trials, can be adjusted as needed

        acc_trials = []
        mmd_exact_trials = []
        time_trials = []

        for trial in range(num_trials):
            np.random.seed(42 + trial)  # Set seed for reproducibility across trials
            start_time_random = time.time()
            if m_coreset > 0 and m_coreset <= num_total_stream_points:
                random_idx = np.random.choice(num_total_stream_points, m_coreset, replace=False)
                X_core_random = X_train_full_original[random_idx]
                y_core_random = y_train_full_original[random_idx]
                w_core_random = np.ones(m_coreset) / m_coreset
            elif m_coreset > num_total_stream_points:  # take all
                X_core_random = X_train_full_original
                y_core_random = y_train_full_original
                w_core_random = np.ones(num_total_stream_points) / num_total_stream_points
            else:  # m_coreset is 0
                X_core_random = np.empty((0, X_train_full_original.shape[1]))
                y_core_random = np.array([])
                w_core_random = np.array([])
            end_time_random = time.time()
            time_trials.append(end_time_random - start_time_random)
            acc_random = train_classifier(X_core_random, X_val, y_core_random, y_val)
            mmd_exact_random = calculate_mmd2_exact(X_train_full_original, X_core_random, w_core_random, KERNEL_GAMMA)
            acc_trials.append(acc_random)
            mmd_exact_trials.append(mmd_exact_random)

        # Compute averages across trials
        average_acc = np.mean(acc_trials)
        average_mmd_exact = np.mean(mmd_exact_trials)
        average_time = np.mean(time_trials)

        # Store averaged results
        results['Random_Time'].append(average_time)
        results['Random_Acc'].append(average_acc)
        results['Random_MMD_Exact'].append(average_mmd_exact)

        print(f"  Random Sampling (End, averaged over {num_trials} trials): Acc={average_acc:.4f}, Exact MMD={average_mmd_exact:.6f}, Time={average_time:.2f}s")

    # --- Plotting Results (similar to before, ensure titles reflect batched nature if desired) ---
    # Accuracy vs. Coreset Size
    plt.figure(figsize=(10, 6))
    plt.plot(results['coreset_size'], results['OnlineMMD_RFF_Acc'], marker='o', label='OnlineMMD-RFF (Batch)')
    plt.plot(results['coreset_size'], results['Reservoir_Acc'], marker='s', linestyle='--', label='Reservoir (Batch)')
    plt.plot(results['coreset_size'], results['Random_Acc'], marker='x', linestyle=':', label='Random (End)')
    plt.axhline(acc_whole, color='k', linestyle='-', label=f'Whole Dataset ({acc_whole:.3f})')
    plt.xlabel('Coreset Size (m)'); plt.ylabel('Validation Accuracy')
    plt.title(f'Accuracy vs. Coreset Size (Batched Stream, Size {DATASET_SUBSET_SIZE})')
    plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

    # Exact MMD vs. Coreset Size
    plt.figure(figsize=(10, 6))
    plt.plot(results['coreset_size'], results['OnlineMMD_RFF_MMD_Exact'], marker='o', label='OnlineMMD-RFF (Batch)')
    plt.plot(results['coreset_size'], results['Reservoir_MMD_Exact'], marker='s', linestyle='--', label='Reservoir (Batch)')
    plt.plot(results['coreset_size'], results['Random_MMD_Exact'], marker='x', linestyle=':', label='Random (End)')
    plt.xlabel('Coreset Size (m)'); plt.ylabel('Exact MMD² (RBF Kernel)')
    plt.title(f'Exact MMD² vs. Coreset Size (Batched Stream, Size {DATASET_SUBSET_SIZE})')
    plt.legend(); plt.grid(True); plt.yscale('log'); plt.tight_layout(); plt.show()
    
    # MMD_RFF evolution for a specific coreset size
    if results['coreset_size'] and results['OnlineMMD_RFF_MMD_Stream']:
        plot_coreset_idx = len(results['coreset_size']) // 2
        target_m_for_plot = results['coreset_size'][plot_coreset_idx]
        
        mmd_stream_online = results['OnlineMMD_RFF_MMD_Stream'][plot_coreset_idx]
        mmd_stream_reservoir = results['Reservoir_MMD_Stream'][plot_coreset_idx]
        
        # Number of logging points might not be exactly num_batches / STREAM_LOG_INTERVAL_BATCHES
        # if the last log point is triggered by end of stream.
        # Create x-axis based on actual number of logged MMDs.
        num_logs = len(mmd_stream_online)
        stream_log_points_xaxis = np.linspace(0, num_batches * BATCH_SIZE, num_logs, endpoint=True)


        plt.figure(figsize=(10, 6))
        if mmd_stream_online : plt.plot(stream_log_points_xaxis, mmd_stream_online, marker='o', linestyle='-', markersize=4, label='OnlineMMD-RFF (RFF MMD)')
        if mmd_stream_reservoir : plt.plot(stream_log_points_xaxis[:len(mmd_stream_reservoir)], mmd_stream_reservoir, marker='s', linestyle='--', markersize=4, label='Reservoir (RFF MMD)') # Reservoir might have diff num logs if logic varies
        plt.xlabel(f'Number of Stream Points Processed (Approx.)')
        plt.ylabel('RFF MMD² (Approximation)')
        plt.title(f'RFF MMD² Evolution During Batched Stream (Target m={target_m_for_plot})')
        plt.legend(); plt.grid(True); plt.yscale('log'); plt.tight_layout(); plt.show()


if __name__ == '__main__':
    run_batched_streaming_experiment()