import numpy as np
from typing import Any, Dict, List, Optional, Tuple
import torch

# -------------------------
# Utilities
# -------------------------
def project_to_simplex(v: np.ndarray) -> np.ndarray:
    """Project vector v to probability simplex (non-negative, sum=1)."""
    v = np.asarray(v, dtype=float).copy()
    n = v.size
    if n == 0:
        return v
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho_mask = u * np.arange(1, n + 1) > (cssv - 1.0)
    if not np.any(rho_mask):
        return np.ones_like(v) / float(n)
    rho = np.nonzero(rho_mask)[0][-1]
    theta = (cssv[rho] - 1.0) / float(rho + 1)
    w = np.maximum(v - theta, 0.0)
    if w.sum() <= 0:
        return np.ones_like(w) / float(n)
    return w / w.sum()


def mirror_descent_weights_on_candidates_rff(
    X_candidate_rff: np.ndarray,
    mu_pi: np.ndarray,
    md_iterations: int = 200,
    eta: float = 0.1,
    ridge: float = 1e-8,
    initial_weights: Optional[np.ndarray] = None,
    tol: float = 1e-6,
    verbose: bool = False,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Run entropic mirror descent to find weights on the simplex
    for the provided candidate RFFs (full-length weight vector).
    Uses PyTorch autograd for efficient gradient computation.
    Returns w_full (length = number of candidates) and history dict.

    X_candidate_rff: (N, D)
    mu_pi: (D,)
    """
    N = X_candidate_rff.shape[0]
    device = torch.device("cuda" if torch.cuda.is_available() and N > 5000 else "cpu")  # Threshold for GPU; tune based on your hardware
    X = torch.from_numpy(X_candidate_rff).float().to(device)
    mu = torch.from_numpy(mu_pi).float().to(device)
    _, D = X.shape
    eps = 1e-12

    if initial_weights is None:
        z = torch.full((N,), 1.0 / float(N), dtype=torch.float, device=device)
    else:
        z_np = np.asarray(initial_weights, dtype=float).copy()
        if z_np.size != N:
            z_np = np.zeros(N, dtype=float)
            z_np[:min(N, len(z_np))] = initial_weights[:min(N, len(initial_weights))]
        z = torch.from_numpy(z_np).float().to(device)
        z = torch.clamp(z, min=eps)
        z = z / z.sum()

    history = {"md_iters": 0}

    for it in range(md_iterations):
        old_z = z.clone()

        # Compute gradient via autograd
        z.requires_grad_(True)
        embed = torch.matmul(X.t(), z)  # (D,) = X.t() (D,N) @ z (N,)
        loss = torch.sum((mu - embed) ** 2) + ridge * torch.sum(z ** 2)
        grad = torch.autograd.grad(loss, z, create_graph=False)[0]
        z.requires_grad_(False)  # Detach for manual update

        # Mirror (exponentiated gradient) step
        log_z = torch.log(z + eps) - eta * grad
        log_z -= torch.max(log_z)  # Stability
        z = torch.exp(log_z)
        z_sum = z.sum()
        if z_sum <= 0:
            z = torch.full_like(z, 1.0 / float(N))
        else:
            z /= (z_sum + eps)

        history["md_iters"] = it + 1
        if torch.max(torch.abs(z - old_z)) < tol:
            if verbose:
                print(f"MD converged in {it+1} iters for N={N}")
            break

    return z.cpu().numpy(), history

# Updated wrapper (unchanged)
def mirror_descent_wrapper_for_stream(
    X_candidate_rff: np.ndarray,
    mu_pi: np.ndarray,
    md_iterations: int = 200,
    eta: float = 0.1,
    ridge: float = 1e-8,
    initial_weights: Optional[np.ndarray] = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Wrapper that runs mirror descent on candidate RFFs and returns full-length weights.
    X_candidate_rff is already RFF-transformed.
    """
    w, hist = mirror_descent_weights_on_candidates_rff(
        X_candidate_rff,
        mu_pi,
        md_iterations=md_iterations,
        eta=eta,
        ridge=ridge,
        initial_weights=initial_weights,
        tol=1e-6,
        verbose=verbose,
    )
    return w, hist


# -------------------------
# Weighted kernel herding selection (select up to m points)
# -------------------------
def weighted_kernel_herding_rff_md(
    mu_pi: np.ndarray,
    X_candidate: np.ndarray,
    sampler,
    m: int,
    md_iterations: int = 200,
    eta: float = 0.1,
    ridge: float = 1e-8,
    verbose: bool = False,
    initial_indices: Optional[np.ndarray] = None,
    initial_weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Greedy selection of up to m indices from X_candidate using RFF + weighted fully-corrective MD warm-start.
    Returns selected_indices (relative to X_candidate) and final normalized weights (len = selected).
    This implementation follows your earlier greedy-herding with MD weight refinement but accepts warm-start.
    """
    # Precompute RFFs
    X_candidate_rff = sampler.transform(X_candidate)  # (N, D)
    N, D = X_candidate_rff.shape

    if m <= 0 or N == 0:
        return np.array([], dtype=int), np.array([], dtype=float)

    dtype = mu_pi.dtype
    selected_indices: List[int] = []
    coreset_rff = np.zeros((m, D), dtype=float)  # stores selected features
    weights = np.array([], dtype=float)

    eps = 1e-12
    tol = 1e-6
    max_iter = md_iterations

    # if initial_indices provided, include them first in the order (soft protection)
    forced_indices = np.asarray(initial_indices, dtype=int) if initial_indices is not None else np.array([], dtype=int)
    forced_inserted = 0

    # Insert forced indices first (but do not exceed m)
    for idx in forced_indices:
        if forced_inserted >= m:
            break
        if idx < 0 or idx >= N:
            continue
        if idx in selected_indices:
            continue
        selected_indices.append(int(idx))
        coreset_rff[forced_inserted] = X_candidate_rff[idx]
        forced_inserted += 1

    s = forced_inserted
    # Initialize weights for forced set if provided
    if s > 0:
        if initial_weights is not None and initial_weights.size >= s:
            w_init = np.asarray(initial_weights, dtype=float)[:s].copy()
            if w_init.sum() > 0:
                w_init = w_init / (w_init.sum() + eps)
            else:
                w_init = np.ones(s) / float(s)
            weights = w_init
        else:
            weights = np.ones(s, dtype=float) / float(s)
    else:
        weights = np.array([], dtype=float)

    # Greedy add remaining up to m
    for k in range(s, m):
        # compute current embedding
        if weights.size == 0:
            current_embedding = np.zeros(D, dtype=float)
        else:
            current_embedding = np.dot(weights, coreset_rff[: weights.size])

        # pick next by max inner product with residual
        residual = mu_pi - current_embedding  # (D,)
        # compute scores for candidates not selected
        scores = X_candidate_rff.dot(residual)  # (N,)
        # mask out already selected
        if len(selected_indices) > 0:
            scores[selected_indices] = -np.inf
        best_idx = int(np.argmax(scores))
        if scores[best_idx] == -np.inf:
            # no candidates left
            break
        selected_indices.append(best_idx)
        coreset_rff[k] = X_candidate_rff[best_idx]
        s_now = k + 1

        # Now compute fully-corrective weights for the selected s_now candidates via MD
        # Use the scalable autograd version instead of explicit K
        slice_rff = coreset_rff[:s_now]
        # Warm start vector for MD on the small set
        if s_now == 1:
            z = np.array([1.0], dtype=float)
        else:
            z = np.zeros(s_now, dtype=float)
            if weights.size > 0:
                # place previous weights into the first positions corresponding to previous picks
                z[: weights.size] = weights * (1.0 - eps)
            # small mass for new
            z[s_now - 1] = eps
            # normalize
            z = project_to_simplex(z)

        # Call autograd MD
        z, _ = mirror_descent_weights_on_candidates_rff(
            slice_rff,
            mu_pi,
            md_iterations=max_iter,
            eta=eta,
            ridge=ridge,
            initial_weights=z,
            tol=tol,
            verbose=verbose,
        )

        weights = z

        if verbose:
            current_embedding = np.dot(weights, slice_rff)
            mmd_rff = float(np.sum((mu_pi - current_embedding) ** 2))
            print(f"[Greedy+MD {k+1}/{m}] picked {best_idx} | rff-mmd={mmd_rff:.6g}")

    # finalize selected indices & weights
    if len(selected_indices) == 0:
        return np.array([], dtype=int), np.array([], dtype=float)

    final_weights = np.maximum(weights, 0.0)
    ssum = final_weights.sum()
    if ssum > 0:
        final_weights /= (ssum + 1e-12)
    else:
        final_weights = np.ones_like(final_weights) / float(len(final_weights))

    return np.array(selected_indices[: len(final_weights)], dtype=int), final_weights


# -------------------------
# Removal delta computation
# -------------------------
def compute_removal_deltas(mu_pi: np.ndarray, current_embedding: np.ndarray, X_buf_rff: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Compute exact immediate increase in squared error (MMD^2) if we remove each point j
    and renormalize remaining weights (no re-optimization).
    Returns deltas array length Nbuf.
    """
    r = mu_pi - current_embedding
    mu = current_embedding
    phis = X_buf_rff
    w = weights
    one_minus_w = 1.0 - w
    eps = 1e-12
    unstable_mask = one_minus_w <= 1e-8

    mu_minus_phi = mu[None, :] - phis  # (N, D)
    norm2 = np.sum(mu_minus_phi * mu_minus_phi, axis=1)  # (N,)
    r_dot = np.dot(mu_minus_phi, r)  # (N,)

    numer = w * w
    denom = np.maximum(one_minus_w * one_minus_w, eps)
    term1 = numer / denom * norm2
    term2 = 2.0 * (w / np.maximum(one_minus_w, eps)) * r_dot
    deltas = term1 - term2
    deltas[unstable_mask] = np.inf
    return deltas


import numpy as np
from typing import Any, Dict, List, Optional, Tuple

# (Assume mirror_descent_wrapper_for_stream and compute_removal_deltas are defined as in your code above)

class MirrorDescentHerdingStreamer:
    def __init__(
        self,
        coreset_size: int,
        buffer_capacity: int,
        sampler,
        batch_size: int,
        md_iterations: int = 200,
        eta: float = 0.1,
        ridge: float = 1e-8,
        verbose: bool = False,
        removal_batch_size: Optional[int] = None,  # NEW: control how many items to remove per reopt
    ) -> None:
        assert coreset_size <= buffer_capacity, "coreset_size must be <= buffer_capacity"
        self.coreset_size = coreset_size
        self.buffer_capacity = buffer_capacity
        self.sampler = sampler
        self.batch_size = batch_size
        self.md_iterations = md_iterations
        self.eta = eta
        self.ridge = ridge
        self.verbose = verbose

        # NEW: how many items to remove per iterative re-opt step.
        # If None -> default to remove all excess in one shot (old behavior).
        # If k (int) -> remove k items per reopt iteration until buffer fits.
        self.removal_batch_size = removal_batch_size

        # derived dims
        self.rff_dim = sampler.n_components
        try:
            self.feature_dim = sampler.random_weights_.shape[1]
        except Exception:
            self.feature_dim = None

        # buffer
        self.buffer_X = np.empty((0, self.feature_dim)) if self.feature_dim is not None else np.empty((0, 0))
        self.buffer_y = np.empty(0, dtype=int)
        self.buffer_weights = np.empty(0, dtype=float)
        self.buffer_provenance: List[Tuple[int, int]] = []

        # running mean rff
        self.mean_rff_full_stream = np.zeros(self.rff_dim)
        self.num_points_seen = 0
        self._finalized = False
        self.last_history: Optional[Dict[str, Any]] = None

    def process_batch(self, X_batch_np: np.ndarray, y_batch_np: np.ndarray, batch_idx: int) -> None:
        if self._finalized:
            if self.verbose:
                print("Warning: streamer finalized, ignoring incoming batch")
            return

        batch_len = X_batch_np.shape[0]
        # update running mean in RFF space
        X_batch_rff = self.sampler.transform(X_batch_np)
        current_batch_mean = np.mean(X_batch_rff, axis=0)
        if self.num_points_seen == 0:
            self.mean_rff_full_stream = current_batch_mean.copy()
        else:
            alpha = batch_len / float(self.num_points_seen + batch_len)
            self.mean_rff_full_stream = (1 - alpha) * self.mean_rff_full_stream + alpha * current_batch_mean
        self.num_points_seen += batch_len

        # Append new batch to buffer arrays (and provenance)
        if self.buffer_X.size == 0:
            self.buffer_X = np.asarray(X_batch_np).copy()
        else:
            self.buffer_X = np.vstack([self.buffer_X, X_batch_np])
        self.buffer_y = np.concatenate([self.buffer_y, np.asarray(y_batch_np, dtype=int)])
        self.buffer_provenance.extend([(batch_idx, i) for i in range(batch_len)])

        # update buffer weights: previous weights decay, new points get small mass
        if self.buffer_weights.size == 0:
            # If buffer was empty, initialize new weights proportional to current running count
            new_weights = np.full(batch_len, 1.0 / float(self.num_points_seen))
            self.buffer_weights = new_weights
        else:
            alpha = float(batch_len) / float(self.num_points_seen)
            self.buffer_weights *= (1 - alpha)    # decay old mass
            new_weights = np.full(batch_len, alpha / float(batch_len))  # equally distribute new mass
            self.buffer_weights = np.concatenate([self.buffer_weights, new_weights])

        # check overflow
        if len(self.buffer_X) > self.buffer_capacity:
            if self.verbose:
                print(f"Buffer overflow (size={len(self.buffer_X)} > cap={self.buffer_capacity}) - running fully-corrective unherding")

            # Precompute RFFs for buffer once
            X_candidate = self.buffer_X
            X_candidate_rff = self.sampler.transform(X_candidate)
            Nbuf = X_candidate_rff.shape[0]

            # Run MD on entire buffer with warm start = current buffer_weights (full-length)
            initial_weights_full = self.buffer_weights.copy() if self.buffer_weights.size == Nbuf else None
            w_full, history = mirror_descent_wrapper_for_stream(
                X_candidate_rff,
                self.mean_rff_full_stream,
                md_iterations=self.md_iterations,
                eta=self.eta,
                ridge=self.ridge,
                initial_weights=initial_weights_full,
                verbose=self.verbose,
            )
            self.last_history = {"md_full_initial": history}

            # Normalize
            if w_full.sum() > 0:
                w_full = w_full / (w_full.sum() + 1e-12)
            else:
                w_full = np.ones_like(w_full) / float(len(w_full))

            # Decide how many to remove in total and per mini-iteration
            total_remove = int(Nbuf - self.buffer_capacity)
            if total_remove <= 0:
                # nothing to remove (shouldn't happen)
                return

            # Determine mini-batch size (k) for iterative removals:
            # - if user supplied removal_batch_size, use it
            # - else default to total_remove (one-shot) to preserve current fast behavior
            if self.removal_batch_size is None:
                k = total_remove
            else:
                k = int(max(1, self.removal_batch_size))

            # Initialize working arrays for iterative removal loop
            X_buf_rff = X_candidate_rff.copy()
            X_buf_raw = X_candidate.copy()
            buf_y = self.buffer_y.copy()
            prov = list(self.buffer_provenance)
            w = w_full.copy()

            removed_count = 0
            # Loop until we've removed total_remove items
            while removed_count < total_remove:
                # compute current embedding and deltas on current kept set
                current_embedding = np.dot(w, X_buf_rff)  # (D,)
                deltas = compute_removal_deltas(self.mean_rff_full_stream, current_embedding, X_buf_rff, w)

                # choose how many to remove this iteration
                to_remove_now = min(k, total_remove - removed_count)
                # pick indices of smallest deltas (they are indices into the current buffer)
                remove_idx = np.argsort(deltas)[:to_remove_now]

                if self.verbose:
                    print(f"Removing {to_remove_now} items (indices {remove_idx}) out of {len(w)} candidates; removed so far {removed_count}")

                # Remove chosen indices
                keep_mask = np.ones(len(w), dtype=bool)
                keep_mask[remove_idx] = False

                X_buf_rff = X_buf_rff[keep_mask]
                X_buf_raw = X_buf_raw[keep_mask]
                buf_y = buf_y[keep_mask]
                prov = [p for i, p in enumerate(prov) if keep_mask[i]]
                w = w[keep_mask]

                removed_count += to_remove_now

                # Quick renormalize (useful as warm-start for MD)
                if w.sum() > 0:
                    w = w / (w.sum() + 1e-12)
                else:
                    # fallback uniform
                    w = np.ones(len(w), dtype=float) / float(len(w))

                # Re-run mirror-descent on the current retained set to be fully-corrective (use fewer iterations when streaming)
                reopt_iters = max(30, int(self.md_iterations // 6))
                w, hist2 = mirror_descent_wrapper_for_stream(
                    X_buf_rff,
                    self.mean_rff_full_stream,
                    md_iterations=reopt_iters,
                    eta=self.eta,
                    ridge=self.ridge,
                    initial_weights=w,
                    verbose=self.verbose,
                )

                # normalize again after MD
                if w.sum() > 0:
                    w = w / (w.sum() + 1e-12)
                else:
                    w = np.ones_like(w) / float(len(w))

                if self.verbose:
                    # quick monitor of current MMD^2
                    emb = np.dot(w, X_buf_rff)
                    mmd_now = float(np.sum((self.mean_rff_full_stream - emb) ** 2))
                    print(f"After removal loop step: retained {len(w)} points, MMD^2={mmd_now:.6g}")

            # Update buffer with final kept set
            self.buffer_X = X_buf_raw
            self.buffer_y = buf_y
            self.buffer_provenance = prov
            self.buffer_weights = w

            if self.verbose:
                print(f"After unherding prune: buffer size = {len(self.buffer_X)} (cap={self.buffer_capacity})")

    def _finalize_coreset(self) -> None:
        if self._finalized:
            return

        # If buffer > target, apply the same unherding pruning to capacity (should be <= capacity)
        if len(self.buffer_X) > self.buffer_capacity:
            # reuse process by calling a dummy batch append of size 0 to trigger pruning
            # but simpler: just call process_batch with empty batch won't change mean; here we call the same removal loop:
            pass

        # Now produce final m-sized coreset using fully-corrective weighted kernel herding
        if len(self.buffer_X) > self.coreset_size:
            # For final selection, we want exactly coreset_size points selected from buffer
            # We'll run weighted_kernel_herding_rff_md on the buffer and ask for m = coreset_size
            S_indices_rel, sel_weights = weighted_kernel_herding_rff_md(
                self.mean_rff_full_stream,
                self.buffer_X,
                self.sampler,
                self.coreset_size,
                md_iterations=max(200, self.md_iterations),
                eta=self.eta,
                ridge=self.ridge,
                verbose=self.verbose,
                # warm-start using current buffer weights mapped to selected positions:
                initial_indices=None,
                initial_weights=None,
            )
            # If no indices returned (numerical), fallback to top-k by buffer_weights
            if S_indices_rel.size == 0:
                order = np.argsort(-self.buffer_weights)[: self.coreset_size]
                S_indices_rel = order
                sel_weights = self.buffer_weights[S_indices_rel]
                if sel_weights.sum() > 0:
                    sel_weights = sel_weights / (sel_weights.sum() + 1e-12)
                else:
                    sel_weights = np.ones(len(S_indices_rel), dtype=float) / float(len(S_indices_rel))

            # update buffer to final coreset
            self.buffer_X = self.buffer_X[S_indices_rel]
            self.buffer_y = self.buffer_y[S_indices_rel]
            self.buffer_weights = sel_weights
            self.buffer_provenance = [self.buffer_provenance[i] for i in S_indices_rel]

        # final normalization / mark done
        if self.buffer_weights.sum() > 0:
            self.buffer_weights = self.buffer_weights / (self.buffer_weights.sum() + 1e-12)
        else:
            self.buffer_weights = np.ones(len(self.buffer_X), dtype=float) / max(1, len(self.buffer_X))

        self._finalized = True

    def get_final_coreset(self) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
        self._finalize_coreset()
        flat_indices = np.array([p[0] * self.batch_size + p[1] for p in self.buffer_provenance], dtype=int)
        return flat_indices, self.buffer_weights, list(self.buffer_provenance)

    def print_coreset_provenance(self) -> None:
        flat_indices, weights, provenance = self.get_final_coreset()
        print("--- Final Coreset Provenance (Fully-corrective Unherding - RFF) ---")
        if not provenance:
            print("Coreset is empty.")
            return
        print(f"{'Global Index':<15} {'Provenance':<20} {'Weight':<10}")
        print("-" * 55)
        for i in range(len(provenance)):
            prov_str = f"(Batch {provenance[i][0]}, Idx {provenance[i][1]})"
            print(f"{flat_indices[i]:<15} {prov_str:<20} {weights[i]:<10.6f}")
        print(f"\nTotal points in coreset: {len(provenance)}")
        print(f"Total points seen in stream: {self.num_points_seen}")