import numpy as np
from typing import Tuple, List
from streamers.abstract_streamer import AbstractStreamingCoreset
from sklearn.kernel_approximation import RBFSampler


class SupersamplingCoreset(AbstractStreamingCoreset):
    """
    Implements a buffered version of "Super-Sampling with a Reservoir".

    This class uses a buffer to accumulate streaming data. When the buffer is full,
    it runs a selection algorithm to distill the buffered points into a smaller,
    representative coreset that minimizes the Maximum Mean Discrepancy (MMD).
    """

    def __init__(self, target_coreset_size: int, buffer_size: int, batch_size:int, input_dim: int, num_features: int = 200, gamma: float = 1.0, seed: int = 42):
        """
        Initializes the SupersamplingCoreset selector.

        Args:
            target_coreset_size (int): The final number of points (M) for the coreset.
            buffer_size (int): The maximum number of points to hold in memory before
                               running selection. Must be >= target_coreset_size.
            input_dim (int): The dimensionality of the input data points.
            num_features (int): The number of random Fourier features (D) to use.
            gamma (float): The lengthscale parameter for the RBF kernel.
            seed (int): A random seed for reproducibility.
        """
        if target_coreset_size > buffer_size:
            raise ValueError("Coreset size cannot exceed buffer capacity.")

        self.target_coreset_size = target_coreset_size
        self.buffer_size = buffer_size
        self.input_dim = input_dim
        self.num_features = num_features
        self.gamma = gamma
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)

        # Use scikit-learn's RBFSampler to compute RFF features.
        # We call fit on a dummy array to ensure internal random weights are initialized
        # in all scikit-learn versions.
        self.rff_sampler = RBFSampler(gamma=self.gamma, n_components=self.num_features, random_state=seed)
        self.rff_sampler.fit(np.zeros((2, self.input_dim)))

        # In-memory buffer for incoming data
        self.buffer_X = np.zeros((self.buffer_size, self.input_dim))
        self.buffer_provenance: List[Tuple[int, int]] = [(-1, -1)] * self.buffer_size
        self.buffer_fill_count = 0

        # Coreset storage (final selections)
        self.coreset_X = np.zeros((self.target_coreset_size, self.input_dim))
        self.coreset_phi = np.zeros((self.target_coreset_size, self.num_features))
        self.coreset_provenance: List[Tuple[int, int]] = [(-1, -1)] * self.target_coreset_size

        # Mean embeddings
        self.mu_hat = np.zeros(self.num_features)
        self.nu_hat = np.zeros(self.num_features)

        # State tracking
        self.n_processed_total = 0
        self.is_initialized = False

    def _compute_rff(self, X: np.ndarray) -> np.ndarray:
        """
        Compute Random Fourier Features using scikit-learn's RBFSampler.

        Args:
            X (np.ndarray): (N, D) array of inputs.

        Returns:
            phi (np.ndarray): (N, n_components) RFF features.
        """
        # RBFSampler.transform returns a (N, n_components) array
        return self.rff_sampler.transform(X)

    def _reduce_buffer(self):
        """
        Runs the streaming selection algorithm on the points currently in the buffer.
        This distills the buffer's contents down to the target coreset size.
        """
        # --- Initialization Step (First time buffer is processed) ---
        if not self.is_initialized:
            # Take the first M points from the buffer as the initial coreset
            self.coreset_X = self.buffer_X[:self.target_coreset_size].copy()
            self.coreset_provenance = self.buffer_provenance[:self.target_coreset_size][:]
            self.coreset_phi = self._compute_rff(self.coreset_X)

            # All points processed so far are in this initial coreset
            all_phi = self._compute_rff(self.buffer_X[:self.buffer_fill_count])
            self.mu_hat = np.mean(all_phi, axis=0)
            self.nu_hat = np.mean(self.coreset_phi, axis=0)

            self.is_initialized = True

            # The remaining points in the buffer are treated as the "stream"
            stream_start_index = self.target_coreset_size
        else:
            stream_start_index = self.target_coreset_size
            # The first M items in the buffer are the previous coreset
            self.coreset_X = self.buffer_X[:self.target_coreset_size].copy()
            self.coreset_provenance = self.buffer_provenance[:self.target_coreset_size][:]
            self.coreset_phi = self._compute_rff(self.coreset_X)


        # --- Streaming Selection on Buffer Contents ---
        for i in range(stream_start_index, self.buffer_fill_count):
            n = self.n_processed_total - self.buffer_fill_count + i + 1
            x_n = self.buffer_X[i:i+1]
            phi_n = self._compute_rff(x_n).flatten()

            # The global mu_hat was already updated when filling the buffer
            # Here we simulate the streaming update for the selection logic

            temp_mu_hat = ((n - 1) / n) * self.mu_hat + (1 / n) * phi_n
            phi_star = phi_n + self.target_coreset_size * (self.nu_hat - temp_mu_hat)

            candidate_phis = np.vstack([self.coreset_phi, phi_n])
            distances_sq = np.sum((candidate_phis - phi_star) ** 2, axis=1)
            j_drop = np.argmin(distances_sq)

            if j_drop < self.target_coreset_size:
                phi_to_drop = self.coreset_phi[j_drop]
                self.nu_hat += (1 / self.target_coreset_size) * (phi_n - phi_to_drop)

                # assign rows properly: x_n is shape (1, D), so take the first row
                self.coreset_X[j_drop] = x_n[0]
                self.coreset_phi[j_drop] = phi_n
                self.coreset_provenance[j_drop] = self.buffer_provenance[i]

        # --- Reset Buffer State ---
        # The new coreset now occupies the start of the buffer
        self.buffer_X[:self.target_coreset_size] = self.coreset_X
        self.buffer_provenance[:self.target_coreset_size] = self.coreset_provenance
        self.buffer_fill_count = self.target_coreset_size


    def process_batch(self, X_batch_np: np.ndarray, y_batch_np: np.ndarray, batch_idx: int) -> None:
        """
        Processes a new batch of data points, updating the coreset.
        """
        if X_batch_np.shape[0] == 0:
            return

        batch_cursor = 0
        while batch_cursor < X_batch_np.shape[0]:
            space_in_buffer = self.buffer_size - self.buffer_fill_count
            num_to_take = min(space_in_buffer, X_batch_np.shape[0] - batch_cursor)

            if num_to_take > 0:
                # Copy data into buffer
                start, end = self.buffer_fill_count, self.buffer_fill_count + num_to_take
                batch_start, batch_end = batch_cursor, batch_cursor + num_to_take
                
                self.buffer_X[start:end] = X_batch_np[batch_start:batch_end]
                for i in range(num_to_take):
                    self.buffer_provenance[start + i] = (batch_idx, batch_start + i)

                phi_batch_segment = self._compute_rff(X_batch_np[batch_start:batch_end])
                for i in range(num_to_take):
                    self.n_processed_total += 1
                    n = self.n_processed_total
                    self.mu_hat = ((n - 1) / n) * self.mu_hat + (1 / n) * phi_batch_segment[i]

                self.buffer_fill_count += num_to_take
                batch_cursor += num_to_take

            # Handle buffer full case
            if self.buffer_fill_count == self.buffer_size:
                self._reduce_buffer()
                # This ensures we don't loop forever
                if self.buffer_fill_count == self.buffer_size and num_to_take == 0:
                    break


    def get_final_coreset(self) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
        """
        Finalizes the selection process and returns the coreset.
        """
        # Process any remaining items in the buffer that didn't trigger a full reduction
        if self.buffer_fill_count > self.target_coreset_size:
            self._reduce_buffer()
        elif not self.is_initialized and self.buffer_fill_count > 0:
            # Handle case where stream ends before first buffer is full
            self.coreset_X = self.buffer_X[:self.buffer_fill_count].copy()
            self.coreset_provenance = self.buffer_provenance[:self.buffer_fill_count][:]


        num_in_coreset = min(self.n_processed_total, self.target_coreset_size)
        final_provenance = self.coreset_provenance[:num_in_coreset]

        # Flatten (batch_idx, local_idx) → global index
        flattened_indices = np.array([
            b_idx * self.batch_size + l_idx
            for b_idx, l_idx in final_provenance
        ], dtype=int)

        weights = np.ones(num_in_coreset) / num_in_coreset

        return flattened_indices, weights, final_provenance

    def print_coreset_provenance(self) -> None:
        """
        Prints the provenance and weight of each point in the final coreset.
        """
        coreset_X, weights, provenance = self.get_final_coreset()
        print(f"--- Coreset Provenance (Size: {len(coreset_X)}) ---")
        if not provenance:
            print("Coreset is empty.")
            return

        for i, (p, w) in enumerate(zip(provenance, weights)):
            print(f"Point {i:>3}: Provenance (batch_idx, local_idx) = {p}, Weight = {w:.4f}")