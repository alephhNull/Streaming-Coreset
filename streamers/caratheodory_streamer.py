import numpy as np
from typing import Tuple, List, Optional, Callable

from streamers.abstract_streamer import AbstractStreamingCoreset



# class CaratheoderyStreamingCoreset(AbstractStreamingCoreset):
#     """
#     Streaming coreset that maintains an (approximate) representation of the
#     running *feature-space* mean using constructive Carathéodory reductions.

#     Parameters
#     ----------
#     coreset_size : int
#         Desired maximum number of coreset points to keep (M).
#     buffer_capacity : int
#         Buffer capacity (how many points we allow to pile up before running
#         reduction). Typically equals coreset_size.
#     rbf_sampler : object
#         An sklearn-like transformer implementing .transform(X) that maps a
#         (B, D) batch of inputs to (B, d) feature vectors (e.g. sklearn.kernel_approximation.RBFSampler).
#     batch_size : int
#         Declared batch size of the stream. Used to compute flat/global indices
#         as ``global_idx = batch_idx * batch_size + local_idx`` so provenance
#         remains meaningful if batches are dropped or skipped.
#     eps : float
#         Numerical tolerance for zeroing tiny weights.
#     """

#     def __init__(self,
#                  coreset_size: int,
#                  buffer_capacity: int,
#                  rbf_sampler,
#                  batch_size: int,
#                  eps: float = 1e-12):
#         assert coreset_size >= 1, "coreset_size must be >= 1"
#         assert buffer_capacity >= 1, "buffer_capacity must be >= 1"
#         assert batch_size >= 1, "batch_size must be >= 1"

#         self.coreset_size = int(coreset_size)
#         self.buffer_capacity = int(buffer_capacity)
#         self.rbf_sampler = rbf_sampler
#         self.batch_size = int(batch_size)
#         self.eps = float(eps)

#         # internal state
#         self._features: List[np.ndarray] = []   # list of feature vectors (d,)
#         self._weights = np.zeros(0, dtype=float)   # convex weights summing to ~1
#         self._provenance: List[Tuple[int,int]] = []
#         self._global_indices: List[int] = []
#         self._total_seen = 0

#         # inferred feature dimension (set on first transform)
#         self._feature_dim: Optional[int] = None

#     # ---------------- public API -----------------
#     def process_batch(self, X_batch_np: np.ndarray, y_batch_np: np.ndarray, batch_idx: int) -> None:
#         """Process a batch of raw inputs. The rbf_sampler is used to transform
#         the whole batch to feature space (vectorized).
#         Each sample is still processed sequentially to maintain the running mean
#         update exactly (weights are scaled by previous_count / new_count).
#         """
#         if X_batch_np is None:
#             return
#         X = np.asarray(X_batch_np)
#         if X.ndim == 1:
#             X = X[np.newaxis, :]

#         # optionally transform whole batch to features (vectorized). We attempt
#         # to use rbf_sampler.transform for speed; if it fails we fall back to per-sample calls.
#         psi_batch = None
#         if self.rbf_sampler is not None:
#             try:
#                 psi_batch = np.asarray(self.rbf_sampler.transform(X))
#             except Exception:
#                 psi_batch = None

#         B = X.shape[0]
#         for local_idx in range(B):
#             x = X[local_idx]
#             if psi_batch is not None:
#                 psi = psi_batch[local_idx]
#             else:
#                 # call transform on single sample, if available
#                 if self.rbf_sampler is not None:
#                     try:
#                         psi = np.asarray(self.rbf_sampler.transform(x[np.newaxis, :]))[0]
#                     except Exception:
#                         # last resort: use raw x
#                         psi = np.asarray(x, dtype=float).ravel()
#                 else:
#                     psi = np.asarray(x, dtype=float).ravel()

#             self._process_single_feature(psi, batch_idx, local_idx)

#     def get_final_coreset(self, uniform_weights: bool = False) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
#         """Return final coreset flat indices, weights and provenance.

#         If `uniform_weights` is True each returned point gets weight 1/|S|.
#         Otherwise the convex weights (that represent the running mean in the
#         feature space) are returned.
#         """
#         if len(self._weights) == 0:
#             return np.array([], dtype=int), np.array([]), []

#         if uniform_weights:
#             k = len(self._weights)
#             return np.array(self._global_indices, dtype=int), np.full(k, 1.0 / k), list(self._provenance)
#         else:
#             w = np.asarray(self._weights, dtype=float)
#             s = w.sum()
#             if s <= 0:
#                 k = len(w)
#                 w = np.full(k, 1.0 / k)
#             else:
#                 w = w / s
#             return np.array(self._global_indices, dtype=int), w, list(self._provenance)

#     def print_coreset_provenance(self) -> None:
#         flat_indices, weights, provenance = self.get_final_coreset()
#         print("Coreset size:", len(weights))
#         for i, (gidx, w, prov) in enumerate(zip(flat_indices, weights, provenance)):
#             print(f"[{i}] global_idx={gidx}, weight={w:.6g}, provenance={prov}")

#     # ---------------- internal helpers -----------------
#     def _process_single_feature(self, psi: np.ndarray, batch_idx: int, local_idx: int) -> None:
#         """Process a single *feature* vector psi (already transformed).

#         We maintain the running mean exactly by scaling existing weights and
#         inserting the new point with weight 1/new_t. After insertion we allow
#         the support to grow up to buffer_capacity; once buffer_capacity is
#         exceeded we run Carathéodory reductions until the support is reduced
#         at least down to the target (see notes below).
#         """
#         psi = np.asarray(psi, dtype=float).ravel()
#         if self._feature_dim is None:
#             self._feature_dim = psi.shape[0]
#             # warn if requested coreset_size is impossible to achieve exactly
#             if self.coreset_size < (self._feature_dim + 1):
#                 print(f"[Warning] requested coreset_size={self.coreset_size} < feature_dim+1={self._feature_dim+1}. "
#                       "An exact representation of the running mean with that small "
#                       "coreset_size is impossible in general. The algorithm will "
#                       "reduce support to at most feature_dim+1 points.")

#         prev_t = self._total_seen
#         new_t = prev_t + 1

#         # scale existing weights
#         if len(self._weights) > 0:
#             self._weights = self._weights * (prev_t / new_t)

#         # append new feature and weight
#         self._features.append(psi.copy())
#         self._weights = np.concatenate([self._weights, np.array([1.0 / new_t])])
#         self._provenance.append((batch_idx, local_idx))
#         # compute flat/global index using batch_idx and batch_size
#         flat_idx = int(batch_idx) * int(self.batch_size) + int(local_idx)
#         self._global_indices.append(flat_idx)

#         self._total_seen = new_t

#         # If support exceeds buffer_capacity, reduce until acceptable target
#         if len(self._weights) > self.buffer_capacity:
#             self._reduce_to_target()

#         # numerical cleanup
#         self._cleanup_weights()

#     def _reduce_to_target(self) -> None:
#         """Reduce support repeatedly using Carathéodory steps until the
#         support size is <= target_support, where

#           target_support = max(self.coreset_size, self._feature_dim+1)

#         This respects the mathematical limit that you cannot generally represent
#         an arbitrary point in R^d using fewer than d+1 affinely independent
#         points. If the user requested a smaller coreset than that, we stop at
#         d+1 and warn (the running mean cannot be represented exactly with
#         fewer points in general).
#         """
#         if self._feature_dim is None:
#             return
#         target_support = max(self.coreset_size, self._feature_dim + 1)

#         # reduce while we have strictly more than target_support
#         while len(self._weights) > target_support:
#             reduced = self._caratheodory_step()
#             if not reduced:
#                 # cannot reduce further (numerical or rank issues)
#                 break

#         if target_support > self.coreset_size and len(self._weights) > self.coreset_size:
#             print(f"[Warning] final support size={len(self._weights)} > requested coreset_size={self.coreset_size}; "
#                   "exact representation to the requested size was impossible given feature dimension.")

#     def _caratheodory_step(self) -> bool:
#         """Perform a single Carathéodory reduction step. Returns True if the
#         support size decreased, False otherwise.

#         This finds a non-trivial alpha in the nullspace of A = [psi; 1] and
#         moves along u - theta * alpha to zero at least one positive coordinate.
#         """
#         k = len(self._weights)
#         d = self._feature_dim
#         if k <= d + 1:
#             # no non-trivial affine dependence guaranteed
#             return False

#         # build A (d+1, k)
#         A = np.zeros((d + 1, k), dtype=float)
#         for j in range(k):
#             A[:d, j] = self._features[j]
#             A[d, j] = 1.0

#         # compute a basis vector for nullspace via SVD (right-singular vector with smallest singular value)
#         try:
#             U, s, Vt = np.linalg.svd(A, full_matrices=False)
#             alpha = Vt.T[:, -1]
#         except np.linalg.LinAlgError:
#             ATA = A.T @ A
#             try:
#                 w, v = np.linalg.eigh(ATA)
#                 alpha = v[:, 0]
#             except Exception:
#                 return False

#         if np.all(np.abs(alpha) <= self.eps):
#             return False

#         # ensure alpha has both positive and negative entries (affine dependence)
#         pos_idx = np.where(alpha > self.eps)[0]
#         neg_idx = np.where(alpha < -self.eps)[0]
#         if pos_idx.size == 0 or neg_idx.size == 0:
#             # numerical problem
#             return False

#         thetas = self._weights[pos_idx] / alpha[pos_idx]
#         theta = float(np.min(thetas))
#         if not (np.isfinite(theta) and theta > 0):
#             return False

#         new_weights = self._weights - theta * alpha
#         new_weights[np.abs(new_weights) < self.eps] = 0.0

#         # indices to keep
#         keep_indices = np.where(new_weights > self.eps)[0]
#         drop_indices = np.where(new_weights <= self.eps)[0]

#         if drop_indices.size == 0:
#             # choose one index to drop (the one that achieved min theta)
#             minpos = pos_idx[int(np.argmin(thetas))]
#             drop_indices = np.array([minpos])
#             keep_indices = np.setdiff1d(np.arange(k), drop_indices)
#             new_weights[minpos] = 0.0

#         # apply keep/drop
#         self._features = [self._features[i] for i in keep_indices]
#         self._provenance = [self._provenance[i] for i in keep_indices]
#         self._global_indices = [self._global_indices[i] for i in keep_indices]
#         self._weights = new_weights[keep_indices]

#         # renormalize
#         s = float(np.sum(self._weights))
#         if s <= 0:
#             # fallback to uniform on remaining
#             k2 = len(self._weights)
#             if k2 == 0:
#                 return False
#             self._weights = np.full(k2, 1.0 / k2)
#         else:
#             self._weights = self._weights / s

#         return True

#     def _cleanup_weights(self) -> None:
#         if len(self._weights) == 0:
#             return
#         self._weights = np.asarray(self._weights, dtype=float)
#         self._weights[np.abs(self._weights) < self.eps] = 0.0
#         s = float(self._weights.sum())
#         if s <= 0:
#             k = len(self._weights)
#             self._weights = np.full(k, 1.0 / k)
#         else:
#             self._weights = self._weights / s


class CaratheoderyStreamingCoreset(AbstractStreamingCoreset):
    """
    Streaming coreset that maintains an (approximate) representation of the
    running *feature-space* mean using constructive Carathéodory reductions.

    Public attributes provided for experiment code compatibility:
      - buffer_X : np.ndarray of raw inputs currently stored in the buffer (k x D)
      - buffer_y : np.ndarray of labels for those inputs (k,)
      - buffer_provenance : list of (batch_idx, local_idx) tuples for each buffered point

    Parameters
    ----------
    coreset_size : int
        Desired maximum number of coreset points to keep (M).
    buffer_capacity : int
        Buffer capacity (how many points we allow to pile up before running
        reduction). Typically equals coreset_size.
    rbf_sampler : object
        An sklearn-like transformer implementing .transform(X) that maps a
        (B, D) batch of inputs to (B, d) feature vectors (e.g. sklearn.kernel_approximation.RBFSampler).
    batch_size : int
        Declared batch size of the stream. Used to compute flat/global indices
        as ``global_idx = batch_idx * batch_size + local_idx`` so provenance
        remains meaningful if batches are dropped or skipped.
    eps : float
        Numerical tolerance for zeroing tiny weights.
    """

    def __init__(self,
                 coreset_size: int,
                 buffer_capacity: int,
                 rbf_sampler,
                 batch_size: int,
                 eps: float = 1e-12):
        assert coreset_size >= 1, "coreset_size must be >= 1"
        assert buffer_capacity >= 1, "buffer_capacity must be >= 1"
        assert batch_size >= 1, "batch_size must be >= 1"

        self.coreset_size = int(coreset_size)
        self.buffer_capacity = int(buffer_capacity)
        self.rbf_sampler = rbf_sampler
        self.batch_size = int(batch_size)
        self.eps = float(eps)

        # internal state (feature-space)
        self._features: List[np.ndarray] = []   # list of feature vectors (d,)
        self._weights = np.zeros(0, dtype=float)   # convex weights summing to ~1
        self._provenance: List[Tuple[int,int]] = []
        self._global_indices: List[int] = []
        self._total_seen = 0

        # keep the original/raw inputs and labels for visualization/experiment
        self._raw_X: List[np.ndarray] = []
        self._raw_y: List[int] = []

        # public-friendly buffer views (kept in sync)
        self.buffer_X: np.ndarray = np.zeros((0,))
        self.buffer_y: np.ndarray = np.zeros((0,), dtype=int)
        self.buffer_provenance: List[Tuple[int,int]] = []

        # inferred feature dimension (set on first transform)
        self._feature_dim: Optional[int] = None

    # ---------------- public API -----------------
    def process_batch(self, X_batch_np: np.ndarray, y_batch_np: np.ndarray, batch_idx: int) -> None:
        """Process a batch of raw inputs. The rbf_sampler is used to transform
        the whole batch to feature space (vectorized).
        Each sample is still processed sequentially to maintain the running mean
        update exactly (weights are scaled by previous_count / new_count).
        """
        if X_batch_np is None:
            return
        X = np.asarray(X_batch_np)
        if X.ndim == 1:
            X = X[np.newaxis, :]

        y_arr = None
        if y_batch_np is not None:
            y_arr = np.asarray(y_batch_np)
            if y_arr.ndim == 0:
                y_arr = np.array([y_arr])

        # attempt vectorized transform
        psi_batch = None
        if self.rbf_sampler is not None:
            try:
                psi_batch = np.asarray(self.rbf_sampler.transform(X))
            except Exception:
                psi_batch = None

        B = X.shape[0]
        for local_idx in range(B):
            x = X[local_idx]
            yval = int(y_arr[local_idx]) if y_arr is not None else -1
            if psi_batch is not None:
                psi = psi_batch[local_idx]
            else:
                # call transform on single sample if available
                if self.rbf_sampler is not None:
                    try:
                        psi = np.asarray(self.rbf_sampler.transform(x[np.newaxis, :]))[0]
                    except Exception:
                        psi = np.asarray(x, dtype=float).ravel()
                else:
                    psi = np.asarray(x, dtype=float).ravel()

            self._process_single_feature(psi, x, yval, batch_idx, local_idx)

    def get_final_coreset(self, uniform_weights: bool = False) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
        """Return final coreset flat indices, weights and provenance.

        If `uniform_weights` is True each returned point gets weight 1/|S|.
        Otherwise the convex weights (that represent the running mean in the
        feature space) are returned.
        """
        if len(self._weights) == 0:
            return np.array([], dtype=int), np.array([]), []

        if uniform_weights:
            k = len(self._weights)
            return np.array(self._global_indices, dtype=int), np.full(k, 1.0 / k), list(self._provenance)
        else:
            w = np.asarray(self._weights, dtype=float)
            s = w.sum()
            if s <= 0:
                k = len(w)
                w = np.full(k, 1.0 / k)
            else:
                w = w / s
            return np.array(self._global_indices, dtype=int), w, list(self._provenance)

    def print_coreset_provenance(self) -> None:
        flat_indices, weights, provenance = self.get_final_coreset()
        print("Coreset size:", len(weights))
        for i, (gidx, w, prov) in enumerate(zip(flat_indices, weights, provenance)):
            print(f"[{i}] global_idx={gidx}, weight={w:.6g}, provenance={prov}")

    # ---------------- internal helpers -----------------
    def _process_single_feature(self, psi: np.ndarray, x_raw: np.ndarray, yval: int, batch_idx: int, local_idx: int) -> None:
        """Process a single *feature* vector psi (already transformed).

        We maintain the running mean exactly by scaling existing weights and
        inserting the new point with weight 1/new_t. After insertion we allow
        the support to grow up to buffer_capacity; once buffer_capacity is
        exceeded we run Carathéodory reductions until the support is reduced
        at least down to the target (see notes below).
        """
        psi = np.asarray(psi, dtype=float).ravel()
        if self._feature_dim is None:
            self._feature_dim = psi.shape[0]
            # warn if requested coreset_size is impossible to achieve exactly
            if self.coreset_size < (self._feature_dim + 1):
                print(f"[Warning] requested coreset_size={self.coreset_size} < feature_dim+1={self._feature_dim+1}. "
                      "An exact representation of the running mean with that small "
                      "coreset_size is impossible in general. The algorithm will "
                      "reduce support to at most feature_dim+1 points.")

        prev_t = self._total_seen
        new_t = prev_t + 1

        # scale existing weights
        if len(self._weights) > 0:
            self._weights = self._weights * (prev_t / new_t)

        # append new feature and weight
        self._features.append(psi.copy())
        self._weights = np.concatenate([self._weights, np.array([1.0 / new_t])])
        self._provenance.append((batch_idx, local_idx))
        # compute flat/global index using batch_idx and batch_size
        flat_idx = int(batch_idx) * int(self.batch_size) + int(local_idx)
        self._global_indices.append(flat_idx)

        # store raw input and label in parallel buffers (for visualization)
        self._raw_X.append(np.asarray(x_raw).copy())
        self._raw_y.append(int(yval))

        # keep public views in sync
        self._sync_public_buffers()

        self._total_seen = new_t

        # If support exceeds buffer_capacity, reduce until acceptable target
        if len(self._weights) > self.buffer_capacity:
            self._reduce_to_target()

        # numerical cleanup
        self._cleanup_weights()

    def _sync_public_buffers(self) -> None:
        """Update public buffer views (buffer_X, buffer_y, buffer_provenance).
        Called whenever the internal lists change.
        """
        if len(self._raw_X) == 0:
            self.buffer_X = np.zeros((0,))
            self.buffer_y = np.zeros((0,), dtype=int)
            self.buffer_provenance = []
        else:
            # stack raw X into 2D array if shapes consistent; otherwise keep as object array
            try:
                self.buffer_X = np.vstack(self._raw_X)
            except Exception:
                self.buffer_X = np.array(self._raw_X, dtype=object)
            self.buffer_y = np.array(self._raw_y, dtype=int)
            self.buffer_provenance = list(self._provenance)

    def _reduce_to_target(self) -> None:
        """Reduce support repeatedly using Carathéodory steps until the
        support size is <= target_support, where

          target_support = max(self.coreset_size, self._feature_dim+1)

        This respects the mathematical limit that you cannot generally represent
        an arbitrary point in R^d using fewer than d+1 affinely independent
        points. If the user requested a smaller coreset than that, we stop at
        d+1 and warn (the running mean cannot be represented exactly with
        fewer points in general).
        """
        if self._feature_dim is None:
            return
        target_support = max(self.coreset_size, self._feature_dim + 1)

        # reduce while we have strictly more than target_support
        while len(self._weights) > target_support:
            reduced = self._caratheodory_step()
            # after each reduction, keep public buffers in sync
            self._sync_public_buffers()
            if not reduced:
                # cannot reduce further (numerical or rank issues)
                break

        if target_support > self.coreset_size and len(self._weights) > self.coreset_size:
            print(f"[Warning] final support size={len(self._weights)} > requested coreset_size={self.coreset_size}; "
                  "exact representation to the requested size was impossible given feature dimension.")

    def _caratheodory_step(self) -> bool:
        """Perform a single Carathéodory reduction step. Returns True if the
        support size decreased, False otherwise.

        This finds a non-trivial alpha in the nullspace of A = [psi; 1] and
        moves along u - theta * alpha to zero at least one positive coordinate.
        When coordinates are dropped we must also drop the corresponding raw
        inputs, labels and provenance entries to keep buffers consistent.
        """
        k = len(self._weights)
        d = self._feature_dim
        if k <= d + 1:
            # no non-trivial affine dependence guaranteed
            return False

        # build A (d+1, k)
        A = np.zeros((d + 1, k), dtype=float)
        for j in range(k):
            A[:d, j] = self._features[j]
            A[d, j] = 1.0

        # compute a basis vector for nullspace via SVD (right-singular vector with smallest singular value)
        try:
            U, s, Vt = np.linalg.svd(A, full_matrices=False)
            alpha = Vt.T[:, -1]
        except np.linalg.LinAlgError:
            ATA = A.T @ A
            try:
                w, v = np.linalg.eigh(ATA)
                alpha = v[:, 0]
            except Exception:
                return False

        if np.all(np.abs(alpha) <= self.eps):
            return False

        # ensure alpha has both positive and negative entries (affine dependence)
        pos_idx = np.where(alpha > self.eps)[0]
        neg_idx = np.where(alpha < -self.eps)[0]
        if pos_idx.size == 0 or neg_idx.size == 0:
            # numerical problem
            return False

        thetas = self._weights[pos_idx] / alpha[pos_idx]
        theta = float(np.min(thetas))
        if not (np.isfinite(theta) and theta > 0):
            return False

        new_weights = self._weights - theta * alpha
        new_weights[np.abs(new_weights) < self.eps] = 0.0

        # indices to keep/drop
        keep_indices = np.where(new_weights > self.eps)[0]
        drop_indices = np.where(new_weights <= self.eps)[0]

        if drop_indices.size == 0:
            # choose one index to drop (the one that achieved min theta)
            minpos = pos_idx[int(np.argmin(thetas))]
            drop_indices = np.array([minpos])
            keep_indices = np.setdiff1d(np.arange(k), drop_indices)
            new_weights[minpos] = 0.0

        # apply keep/drop to ALL parallel arrays
        self._features = [self._features[i] for i in keep_indices]
        self._provenance = [self._provenance[i] for i in keep_indices]
        self._global_indices = [self._global_indices[i] for i in keep_indices]
        self._raw_X = [self._raw_X[i] for i in keep_indices]
        self._raw_y = [self._raw_y[i] for i in keep_indices]
        self._weights = new_weights[keep_indices]

        # renormalize
        s = float(np.sum(self._weights))
        if s <= 0:
            # fallback to uniform on remaining
            k2 = len(self._weights)
            if k2 == 0:
                return False
            self._weights = np.full(k2, 1.0 / k2)
        else:
            self._weights = self._weights / s

        return True

    def _cleanup_weights(self) -> None:
        if len(self._weights) == 0:
            return
        self._weights = np.asarray(self._weights, dtype=float)
        self._weights[np.abs(self._weights) < self.eps] = 0.0
        s = float(self._weights.sum())
        if s <= 0:
            k = len(self._weights)
            self._weights = np.full(k, 1.0 / k)
        else:
            self._weights = self._weights / s
        # keep public buffers consistent after numeric fixes
        self._sync_public_buffers()
