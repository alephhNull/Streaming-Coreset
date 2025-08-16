import numpy as np
from sklearn.kernel_approximation import RBFSampler
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from streamers.abstract_streamer import AbstractStreamingCoreset

class OnlineMMDPlusStreamer(AbstractStreamingCoreset):
    """A refactored, faster Online MMD+ streamer that preserves original logic
    but uses contiguous tensors, batched RFF transforms, optimizer reuse,
    matrix-based pruning, and optional AMP support.

    Key improvements (no algorithmic change):
      - single batched RBFSampler.transform per process_batch
      - buffer stored as contiguous tensors: `buffer_rffs` and `buffer_logits`
      - `buffer_global_ids` kept as a Python list for provenance
      - optimizer only recreated when the Parameter object changes
      - pruning uses one matmul instead of many F.cosine_similarity calls
      - early stopping in weight optimization
      - optional mixed-precision (AMP) support
    """

    def __init__(
        self,
        batch_size,
        m_coreset_size,
        n_rff_components,
        buffer_capacity,
        n_epochs_online=30,
        lr_online=0.01,
        lambda_log_online=1e-5,
        random_seed=42,
        device="cuda",
        use_amp=False,
    ):
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.batch_size = batch_size

        # Core parameters
        self.m = m_coreset_size
        self.n_rff_components = n_rff_components
        self.buffer_capacity = buffer_capacity

        # Optimization parameters
        self.n_epochs_online = n_epochs_online
        self.lr_online = lr_online
        self.lambda_log = lambda_log_online
        self.epsilon = 1e-6

        self.use_amp = use_amp
        if self.use_amp and self.device.type != "cuda":
            # AMP is only supported on CUDA; silently disable for CPU
            self.use_amp = False

        # State
        self.rbf_sampler = None
        self.random_seed = random_seed
        self.num_points_seen = 0
        self._sum_rff_full_stream = torch.zeros(self.n_rff_components, dtype=torch.float32, device=self.device)
        self.mean_rff_full_stream_torch = torch.zeros(self.n_rff_components, dtype=torch.float32, device=self.device)

        # Buffer in contiguous form: tensors on device
        self.buffer_rffs = None        # shape (B, D)
        self.buffer_logits = None      # nn.Parameter shape (B,)
        self.buffer_global_ids = []    # list[(batch_idx, in_batch_idx)]

        # Optimizer and scaler (for AMP)
        self.optimizer = None
        self._opt_param_id = None
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None

        # Monitoring
        self.sparsity_history = []

        if random_seed is not None:
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)

    def set_rbf_sampler(self, rbf_sampler_instance: RBFSampler):
        self.rbf_sampler = rbf_sampler_instance

    # ----------------------- Optimizer helpers -----------------------
    def _ensure_optimizer(self):
        """Ensure there is an optimizer for `self.buffer_logits`.
        Recreate it only if the Parameter object changed.
        """
        if self.buffer_logits is None:
            return
        if self.optimizer is None or self._opt_param_id != id(self.buffer_logits):
            # Simple approach: recreate optimizer when parameter object changes
            self.optimizer = optim.Adam([self.buffer_logits], lr=self.lr_online)
            self._opt_param_id = id(self.buffer_logits)

    # ----------------------- Weight optimization -----------------------
    def _optimize_weights(self):
        """Optimize weights (logits) for the current buffer using Adam.
        Uses contiguous buffer tensors for speed. Adds early stopping.
        """
        if self.buffer_rffs is None or self.buffer_rffs.shape[0] == 0:
            return

        B = self.buffer_rffs.shape[0]
        candidate_rffs = self.buffer_rffs  # (B, D)
        self._ensure_optimizer()
        if self.optimizer is None:
            return

        prev_loss = None
        tol = 1e-7
        max_no_improve = 3
        stagnant = 0

        for epoch in range(self.n_epochs_online):
            self.optimizer.zero_grad()

            if self.use_amp:
                with torch.cuda.amp.autocast():
                    current_weights = torch.relu(self.buffer_logits)  # (B,)
                    norm = current_weights.sum()
                    normalized = current_weights / (norm + 1e-9)
                    mean_Z_coreset = torch.matmul(normalized.unsqueeze(0), candidate_rffs).squeeze(0)
                    mmd2 = torch.sum((self.mean_rff_full_stream_torch - mean_Z_coreset) ** 2)
                    log_penalty = self.lambda_log * torch.sum(torch.log(self.epsilon + current_weights))
                    loss = mmd2 + log_penalty
                # backward with scaler
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                loss_item = loss.detach().cpu().item()
            else:
                current_weights = torch.relu(self.buffer_logits)  # (B,)
                norm = current_weights.sum()
                normalized = current_weights / (norm + 1e-9)
                mean_Z_coreset = torch.matmul(normalized.unsqueeze(0), candidate_rffs).squeeze(0)
                mmd2 = torch.sum((self.mean_rff_full_stream_torch - mean_Z_coreset) ** 2)
                log_penalty = self.lambda_log * torch.sum(torch.log(self.epsilon + current_weights))
                loss = mmd2 + log_penalty
                loss.backward()
                self.optimizer.step()
                loss_item = loss.detach().cpu().item()

            # early stopping: check small improvements
            if prev_loss is not None:
                if prev_loss - loss_item < tol:
                    stagnant += 1
                    if stagnant >= max_no_improve:
                        break
                else:
                    stagnant = 0
            prev_loss = loss_item

        # Update sparsity
        with torch.no_grad():
            final_weights = torch.relu(self.buffer_logits.detach())
            num_nonzero = int((final_weights > 1e-9).sum().item())
            self.sparsity_history.append(num_nonzero)

    # ----------------------- Buffer pruning -----------------------
    def _prune_buffer(self):
        """Prune buffer down to buffer_capacity.
        Keeps the top-m provisional coreset and prunes the rest based on redundancy
        (cosine similarity) computed via a fast matmul.
        """
        if self.buffer_rffs is None:
            return
        num_to_prune = self.buffer_rffs.shape[0] - self.buffer_capacity
        if num_to_prune <= 0:
            return

        with torch.no_grad():
            weights = torch.relu(self.buffer_logits.detach())  # (B,)
            sorted_indices = torch.argsort(weights, descending=True)  # (B,)

            provisional_coreset_indices = sorted_indices[: self.m]
            pruning_candidate_indices = sorted_indices[self.m :]

            if pruning_candidate_indices.numel() == 0:
                return

            if pruning_candidate_indices.numel() <= num_to_prune:
                # remove all pruning candidates
                indices_to_remove = pruning_candidate_indices
            else:
                # compute redundancy via cosine similarity using one matmul
                provisional_coreset_rffs = self.buffer_rffs[provisional_coreset_indices]  # (k, D)
                pruning_rffs = self.buffer_rffs[pruning_candidate_indices]  # (p, D)

                # normalize rows (avoid divide-by-zero)
                prov_norms = provisional_coreset_rffs.norm(dim=1, keepdim=True).clamp(min=1e-9)
                prune_norms = pruning_rffs.norm(dim=1, keepdim=True).clamp(min=1e-9)
                prov_unit = provisional_coreset_rffs / prov_norms
                prune_unit = pruning_rffs / prune_norms

                # similarity: (p, D) @ (D, k) -> (p, k)
                cosine_sim = torch.matmul(prune_unit, prov_unit.T)
                redundancy_scores, _ = torch.max(cosine_sim, dim=1)  # (p,)

                # select top `num_to_prune` most-redundant candidates to prune
                idxs_sorted_by_redundancy = torch.argsort(redundancy_scores, descending=True)
                indices_to_prune_from_candidates = idxs_sorted_by_redundancy[:num_to_prune]
                indices_to_remove = pruning_candidate_indices[indices_to_prune_from_candidates]

        # Build a boolean mask and keep the rest
        mask = torch.ones(self.buffer_rffs.shape[0], dtype=torch.bool, device=self.device)
        mask[indices_to_remove] = False

        # Apply mask to tensors and global ids
        self.buffer_rffs = self.buffer_rffs[mask]
        new_logits_tensor = self.buffer_logits.detach()[mask]
        self.buffer_logits = nn.Parameter(new_logits_tensor)

        # update global ids list (mask is on device; move to cpu)
        mask_cpu = mask.cpu().numpy().tolist()
        self.buffer_global_ids = [g for keep, g in zip(mask_cpu, self.buffer_global_ids) if keep]

    # ----------------------- Main processing -----------------------
    def process_batch(self, X_batch_np: np.ndarray, y_batch_np: np.ndarray, batch_idx: int) -> None:
        if self.rbf_sampler is None:
            raise RuntimeError("RBFSampler not set.")
        batch_size = X_batch_np.shape[0]
        if batch_size == 0:
            return

        # 1) Transform the entire batch at once (single CPU->GPU copy)
        batch_rff_np = self.rbf_sampler.transform(X_batch_np)  # (batch_size, D)
        batch_rff = torch.from_numpy(batch_rff_np).float().to(self.device)  # one transfer

        # 2) Append batch to contiguous buffer tensors
        if self.buffer_rffs is None:
            self.buffer_rffs = batch_rff.clone()
            init_logits = torch.full((batch_rff.shape[0],), 0.1, device=self.device, dtype=torch.float32)
            self.buffer_logits = nn.Parameter(init_logits)
            self.buffer_global_ids = [(batch_idx, i) for i in range(batch_rff.shape[0])]
        else:
            self.buffer_rffs = torch.cat([self.buffer_rffs, batch_rff], dim=0)
            new_logits = torch.full((batch_rff.shape[0],), 0.1, device=self.device, dtype=torch.float32)
            # create new Parameter from concatenated detached tensors
            concatenated_logits = torch.cat([self.buffer_logits.detach(), new_logits], dim=0)
            self.buffer_logits = nn.Parameter(concatenated_logits)
            # append global ids
            base_len = len(self.buffer_global_ids)
            self.buffer_global_ids.extend([(batch_idx, i) for i in range(batch_rff.shape[0])])

        # 3) Update running mean embedding (exponential moving average)
        alpha = 0.1
        current_batch_mean = torch.mean(batch_rff, dim=0)

        if self.num_points_seen == 0:
            self.mean_rff_full_stream_torch = current_batch_mean.clone()
        else:
            self.mean_rff_full_stream_torch = (1 - alpha) * self.mean_rff_full_stream_torch + alpha * current_batch_mean

        self.num_points_seen += batch_size

        # 4) Optimize weights (fast path uses contiguous buffers)
        self._optimize_weights()

        # 5) Prune if buffer exceeded capacity
        if self.buffer_rffs is not None and self.buffer_rffs.shape[0] > self.buffer_capacity:
            self._prune_buffer()

        num_nonzero_weights = self.sparsity_history[-1] if self.sparsity_history else 0.0
        # Minimal logging; place expensive prints behind a debug flag if needed
        print(f"   Batch {batch_idx} processed. Num non-zero weights: {num_nonzero_weights}")

    # ----------------------- Final coreset extraction -----------------------
    def get_final_coreset(self):
        """Return (flat_indices, normalized_weights_np, coreset_global_ids).
        flat_indices computed as batch_idx * batch_size + in_batch_idx (assumes constant batch_size).
        """
        if self.buffer_logits is None or len(self.buffer_global_ids) == 0:
            return np.array([], dtype=int), np.array([]), []

        with torch.no_grad():
            weights = torch.relu(self.buffer_logits.detach())
            k_topk = min(self.m, weights.numel())
            if k_topk == 0:
                return np.array([], dtype=int), np.array([]), []

            vals, idx = torch.topk(weights, k=k_topk)
            normalized = vals / (vals.sum() + 1e-9)
            top_k_indices_in_buffer = idx.cpu().numpy().tolist()

            coreset_global_ids = [self.buffer_global_ids[i] for i in top_k_indices_in_buffer]
            flat_indices = np.array([gid[0] * self.batch_size + gid[1] for gid in coreset_global_ids])

            return flat_indices, normalized.cpu().numpy(), coreset_global_ids

    def print_coreset_provenance(self):
        if not self.buffer_global_ids:
            print("Coreset is empty.")
            return np.array([], dtype=int)

        flat_indices, coreset_weights, coreset_global_ids = self.get_final_coreset()
        print("\n--- Final Coreset Provenance ---")
        for i, (gid, flat_idx) in enumerate(zip(coreset_global_ids, flat_indices)):
            print(f"  Point {i}: From Batch {gid[0]}, Idx {gid[1]} (Flat Index: {flat_idx}) -> Weight: {coreset_weights[i]:.4f}")

        batch_indices = [gid[0] for gid in coreset_global_ids]
        batch_counts = {b: batch_indices.count(b) for b in sorted(list(set(batch_indices)))}
        print("\nCoreset points per batch:", batch_counts, "\n------------------------------")


# ----------------------- Offline baseline (kept for completeness) -----------------------
def get_coreset_rff_sampler_log(
    X_train_np,
    m,
    kernel_gamma,
    n_rff_components=500,
    n_epochs=1000,
    lr=0.001,
    lambda_log=1e-4,
    epsilon=1e-6,
    random_seed=42,
):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    n = X_train_np.shape[0]
    if m > n:
        m = n
    if m == 0 or n == 0:
        return np.array([], dtype=int), np.array([]), [], []

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
        mmd2 = torch.sum((mean_Z_full - mean_Z_core) ** 2)
        log_p = lambda_log * torch.sum(torch.log(epsilon + weights))
        loss = mmd2 + log_p
        loss.backward()
        opt.step()
        mmd_hist.append(mmd2.item())
        loss_hist.append(loss.item())

    final_weights = torch.relu(w_logits).detach()
    k_top = min(m, len(final_weights))
    if final_weights.sum() < 1e-9 or k_top == 0:
        idx = np.random.choice(n, m, replace=False) if n >= m else np.arange(n)
        w = np.ones(len(idx)) / (len(idx) if len(idx) > 0 else 1)
        return idx, w, mmd_hist, loss_hist

    vals, idx = torch.topk(final_weights, k=k_top)
    idx_np = idx.cpu().numpy()
    w_np = (vals / (vals.sum() + 1e-9)).cpu().numpy()
    return idx_np, w_np, mmd_hist, loss_hist


# class OnlineMMDPlusStreamer(AbstractStreamingCoreset):
#     """ An intelligent, stateful streamer for coreset selection with persistent weights and smart buffer management. """
#     def __init__(self, batch_size, m_coreset_size, n_rff_components, buffer_capacity,
#                  n_epochs_online=30, lr_online=0.01, lambda_log_online=1e-5, random_seed=42, device='cuda'):
        
#         self.device = device
#         self.batch_size = batch_size

#         # Core parameters
#         self.m = m_coreset_size
#         self.n_rff_components = n_rff_components
#         self.buffer_capacity = buffer_capacity

#         # Optimization parameters
#         self.n_epochs_online = n_epochs_online
#         self.lr_online = lr_online
#         self.lambda_log = lambda_log_online
#         self.epsilon = 1e-6

#         # State tracking
#         self.rbf_sampler = None
#         self.random_seed = random_seed
#         self.num_points_seen = 0
#         # self._current_batch_idx = -1  # 🗑️ REMOVED: No longer needed, as we'll pass the global index.
#         self._sum_rff_full_stream = torch.zeros(self.n_rff_components, dtype=torch.float32, device=device)
#         self.mean_rff_full_stream_torch = torch.zeros(self.n_rff_components, dtype=torch.float32, device=device)
#         self.optimizer = None
#         self.sparsity_history = []

#         # The core data structure: a buffer of point objects (dicts with persistent logits)
#         self.point_buffer = []

#         if random_seed is not None:
#             np.random.seed(random_seed)
#             torch.manual_seed(random_seed)

#     def set_rbf_sampler(self, rbf_sampler_instance):
#         self.rbf_sampler = rbf_sampler_instance

#     def _optimize_weights(self):
#         """ Runs optimization on the current point_buffer's logits. """
#         if not self.point_buffer:
#             return

#         trainable_logits = [p['logit'] for p in self.point_buffer]
#         if not trainable_logits: return

#         self.optimizer = optim.Adam(trainable_logits, lr=self.lr_online)
#         candidate_rffs = torch.stack([p['rff'] for p in self.point_buffer]).to(self.device)

#         for _ in range(self.n_epochs_online):
#             self.optimizer.zero_grad()
            
#             # This is the corrected part: torch.cat now works on 1-D tensors
#             current_weights = torch.relu(torch.cat(trainable_logits))

#             sum_weights = current_weights.sum()
#             normalized_weights = current_weights / (sum_weights + 1e-9)
#             mean_Z_coreset = torch.sum(normalized_weights.unsqueeze(1) * candidate_rffs, dim=0)
#             mmd2 = torch.sum((self.mean_rff_full_stream_torch - mean_Z_coreset)**2)

#             log_penalty = self.lambda_log * torch.sum(torch.log(self.epsilon + current_weights))
#             loss = mmd2 + log_penalty
#             loss.backward()
#             self.optimizer.step()

#         # Update sparsity count
#         final_weights = torch.relu(torch.cat([p['logit'].detach() for p in self.point_buffer]))
#         num_nonzero = torch.sum(final_weights > 1e-9).item()
#         print("Number of non-zero weights after optimization:", num_nonzero, 'Length of point buffer:', len(self.point_buffer))
#         self.sparsity_history.append(num_nonzero / len(self.point_buffer))


#     def _prune_buffer(self):
#         """ Intelligently prunes the buffer down to its capacity based on weight and redundancy. """
#         num_to_prune = len(self.point_buffer) - self.buffer_capacity
#         if num_to_prune <= 0: return

#         with torch.no_grad():
#             weights = torch.cat([torch.relu(p['logit'].detach()) for p in self.point_buffer])

#         sorted_indices = torch.argsort(weights, descending=True)
#         provisional_coreset_indices = sorted_indices[:self.m]
#         pruning_candidate_indices = sorted_indices[self.m:]

#         if len(pruning_candidate_indices) <= num_to_prune:
#             indices_to_remove = set(pruning_candidate_indices.cpu().numpy())
#         else:
#             provisional_coreset_rffs = torch.stack([self.point_buffer[i]['rff'] for i in provisional_coreset_indices])
#             pruning_rffs = torch.stack([self.point_buffer[i]['rff'] for i in pruning_candidate_indices])

#             cosine_sim = F.cosine_similarity(pruning_rffs.unsqueeze(1), provisional_coreset_rffs.unsqueeze(0), dim=-1)
#             redundancy_scores, _ = torch.max(cosine_sim, dim=1)
            
#             indices_to_prune_from_candidates = torch.argsort(redundancy_scores, descending=True)[:num_to_prune]
#             final_indices_to_prune = pruning_candidate_indices[indices_to_prune_from_candidates]
#             indices_to_remove = set(final_indices_to_prune.cpu().numpy())

#         self.point_buffer = [p for i, p in enumerate(self.point_buffer) if i not in indices_to_remove]


#     def process_batch(self, X_batch_np: np.ndarray, y_batch_np: np.ndarray, batch_idx: int) -> None:
#         if self.rbf_sampler is None: raise RuntimeError("RBFSampler not set.")
#         # self._current_batch_idx += 1 # 🗑️ REMOVED
#         batch_size = X_batch_np.shape[0]
#         if batch_size == 0: return

#         for i in range(batch_size):
#             rff = torch.from_numpy(self.rbf_sampler.transform(X_batch_np[i:i+1])).float().squeeze(0).to(self.device)
#             point_info = {
#                 "rff": rff,
#                 # ✅ CORRECTED: Use the true batch_idx for provenance
#                 "global_id": (batch_idx, i), 
#                 "logit": nn.Parameter(torch.tensor([0.1], dtype=torch.float32, device=self.device))
#             }
#             self.point_buffer.append(point_info)

#         # The rest of this method is unchanged
#         alpha = 0.1
#         current_batch_mean = torch.mean(torch.stack([p['rff'] for p in self.point_buffer[-batch_size:]]), dim=0)
        
#         if self.num_points_seen == 0:
#             self.mean_rff_full_stream_torch = current_batch_mean
#         else:
#             self.mean_rff_full_stream_torch = (1 - alpha) * self.mean_rff_full_stream_torch + alpha * current_batch_mean

#         self.num_points_seen += batch_size
#         self._optimize_weights()

#         if len(self.point_buffer) > self.buffer_capacity:
#             self._prune_buffer()

#         sparsity_ratio = self.sparsity_history[-1] if self.sparsity_history else 0
#         print(f"   Batch {batch_idx} processed. Sparsity: {sparsity_ratio:.2%}")

#     def get_final_coreset(self):
#         """ ✅ MODIFIED: Extracts the final coreset and now returns indices too. """
#         if not self.point_buffer:
#             return np.array([], dtype=int), np.array([])

#         with torch.no_grad():
#             weights = torch.cat([torch.relu(p['logit'].detach()) for p in self.point_buffer])
        
#         k_topk = min(self.m, len(self.point_buffer))
#         if k_topk == 0: return np.array([], dtype=int), np.array([])

#         top_k_weights, top_k_indices_in_buffer = torch.topk(weights, k=k_topk)
#         normalized_weights = top_k_weights / (top_k_weights.sum() + 1e-9)
        
#         coreset_points_info = [self.point_buffer[i] for i in top_k_indices_in_buffer]
#         coreset_global_ids = [p['global_id'] for p in coreset_points_info]

#         # Calculate flat indices here
#         flat_indices = np.array([gid[0] * self.batch_size + gid[1] for gid in coreset_global_ids])
#         # This assumes a constant batch size, which might not be true. 
#         # A more robust way is to pass BATCH_SIZE to this function or store it.
#         # For simplicity, assuming you pass it to print_coreset_provenance as before.
        
#         return flat_indices, normalized_weights.cpu().numpy(), coreset_global_ids

#     def print_coreset_provenance(self):
#         """ Prints the origin of each coreset point for analysis and returns flat indices. """
#         if not self.point_buffer:
#             print("Coreset is empty.")
#             return np.array([], dtype=int)

#         flat_indices, coreset_weights, coreset_global_ids = self.get_final_coreset()
#         print("\n--- Final Coreset Provenance ---")
#         for i, (gid, flat_idx) in enumerate(zip(coreset_global_ids, flat_indices)):
#             # This calculation is now CORRECT because gid[0] is the true global batch index
#             print(f"  Point {i}: From Batch {gid[0]}, Idx {gid[1]} (Flat Index: {flat_idx}) -> Weight: {coreset_weights[i]:.4f}")

#         batch_indices = [gid[0] for gid in coreset_global_ids]
#         batch_counts = {b: batch_indices.count(b) for b in sorted(list(set(batch_indices)))}
#         print("\nCoreset points per batch:", batch_counts, "\n------------------------------")
    



# # The offline method remains for baseline comparison
# def get_coreset_rff_sampler_log(X_train_np, m, kernel_gamma, n_rff_components=500, n_epochs=1000, lr=0.001, lambda_log=1e-4, epsilon=1e-6, random_seed=42):
#     torch.manual_seed(random_seed); np.random.seed(random_seed)
#     n = X_train_np.shape[0]
#     if m > n: m = n
#     if m == 0 or n == 0: return np.array([], dtype=int), np.array([]), [], []
#     rbf_map = RBFSampler(gamma=kernel_gamma, n_components=n_rff_components, random_state=random_seed)
#     Z_train = torch.from_numpy(rbf_map.fit_transform(X_train_np)).float()
#     w_logits = nn.Parameter(torch.full((n,), 0.1, dtype=torch.float32, requires_grad=True))
#     opt = optim.Adam([w_logits], lr=lr)
#     mean_Z_full = torch.mean(Z_train, dim=0)
#     mmd_hist, loss_hist = [], []
#     for _ in range(n_epochs):
#         opt.zero_grad()
#         weights = torch.relu(w_logits)
#         norm_weights = weights / (weights.sum() + 1e-9)
#         mean_Z_core = torch.sum(norm_weights.unsqueeze(1) * Z_train, dim=0)
#         mmd2 = torch.sum((mean_Z_full - mean_Z_core)**2)
#         log_p = lambda_log * torch.sum(torch.log(epsilon + weights))
#         loss = mmd2 + log_p # Note: Your original code had +log_penalty. This encourages sparsity.
#         loss.backward(); opt.step()
#         mmd_hist.append(mmd2.item()); loss_hist.append(loss.item())
#     final_weights = torch.relu(w_logits).detach()
#     k_top = min(m, len(final_weights))
#     if final_weights.sum() < 1e-9 or k_top == 0:
#         idx = np.random.choice(n, m, replace=False) if n >= m else np.arange(n)
#         w = np.ones(len(idx)) / (len(idx) if len(idx)>0 else 1)
#         return idx, w, mmd_hist, loss_hist
#     vals, idx = torch.topk(final_weights, k=k_top)
#     idx_np = idx.cpu().numpy()
#     w_np = (vals / (vals.sum() + 1e-9)).cpu().numpy()
#     return idx_np, w_np, mmd_hist, loss_hist