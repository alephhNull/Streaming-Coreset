import numpy as np
import torch
from abc import ABC, abstractmethod
from typing import Tuple, List
import geomloss
from scipy.optimize import minimize

from streamers.abstract_streamer import AbstractStreamingCoreset


class CO2Streamer(AbstractStreamingCoreset):
    """
    Implements the CO2 coreset selection algorithm for a streaming scenario.

    This class maintains a buffer of incoming data points. When the buffer exceeds
    its capacity, it uses the CO2-recombination algorithm to select a smaller,
    weighted subset (a coreset) that represents the buffered data. This coreset
    then replaces the buffer's contents.

    [cite_start]The CO2 algorithm is based on the provided paper[cite: 1, 2]. It approximates the
    Sinkhorn divergence with a second-order expansion, which reduces the problem to
    Maximum Mean Discrepancy (MMD) minimization. This is solved by:
    1.  [cite_start]Computing the Sinkhorn kernel (Hadamard operator)[cite: 249].
    2.  [cite_start]Using a Nyström method to find its dominant eigenfunctions[cite: 111, 208].
    3.  [cite_start]Selecting points and weights via a recombination (moment-matching) procedure[cite: 110, 208].
    """

    def __init__(self, wanted_coreset_size: int, buffer_capacity: int, reg: float = 1.0, theta: int = 3):
        """
        Initializes the CO2Streamer.

        Args:
            wanted_coreset_size (int): The desired number of points (r) in the coreset.
            buffer_capacity (int): The maximum number of points to hold before compression.
            [cite_start]reg (float): The regularization parameter ε for the Sinkhorn divergence[cite: 215].
            [cite_start]theta (int): The oversampling parameter for the Nyström approximation[cite: 115, 208].
        """
        if buffer_capacity < wanted_coreset_size:
            raise ValueError("buffer_capacity must be greater than or equal to wanted_coreset_size")

        self.wanted_coreset_size = wanted_coreset_size
        self.buffer_capacity = buffer_capacity
        self.reg = reg
        self.theta = theta

        # Buffer to store data points as they arrive
        self.buffer: np.ndarray = np.array([])
        
        # Provenance lists to track the origin of each point in the buffer
        self.provenance: List[Tuple[int, int]] = []
        self.global_indices: List[int] = []

        # Weights associated with points in the buffer (initially uniform)
        self.weights: np.ndarray = np.array([])
        
        # Counter for assigning a unique global index to each incoming data point
        self.global_idx_counter = 0

    def process_batch(self, X_batch_np: np.ndarray, batch_idx: int) -> None:
        """
        Processes a new batch of data, adding it to the buffer and running
        compression if the buffer exceeds its capacity.
        """
        batch_size = X_batch_np.shape[0]
        
        # Generate provenance for the new batch
        new_provenance = [(batch_idx, i) for i in range(batch_size)]
        new_global_indices = list(range(self.global_idx_counter, self.global_idx_counter + batch_size))
        self.global_idx_counter += batch_size

        if self.buffer.shape[0] == 0:
            self.buffer = X_batch_np
            self.provenance = new_provenance
            self.global_indices = new_global_indices
        else:
            self.buffer = np.vstack([self.buffer, X_batch_np])
            self.provenance.extend(new_provenance)
            self.global_indices.extend(new_global_indices)

        print(f"Processed batch {batch_idx}. Buffer size: {self.buffer.shape[0]}/{self.buffer_capacity}")

        # If buffer exceeds capacity, compress it
        if self.buffer.shape[0] > self.buffer_capacity:
            self._compress_buffer()

    def _compress_buffer(self):
        """
        Compresses the current buffer down to the `wanted_coreset_size` using the CO2 algorithm.
        """
        n = self.buffer.shape[0]
        r = self.wanted_coreset_size
        
        print(f"Buffer full. Compressing {n} points down to {r}...")

        # Run the CO2 algorithm
        selected_indices, weights = self._run_co2(self.buffer, r)

        # Update the buffer and its associated metadata
        self.buffer = self.buffer[selected_indices]
        self.weights = weights
        self.provenance = [self.provenance[i] for i in selected_indices]
        self.global_indices = [self.global_indices[i] for i in selected_indices]
        
        print(f"Compression complete. New buffer size: {self.buffer.shape[0]}")

    def _run_co2(self, X: np.ndarray, r: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Executes the core CO2-recombination algorithm on the input data X.

        Args:
            X (np.ndarray): The (n x d) data matrix to compress.
            r (int): The target coreset size.

        Returns:
            A tuple containing:
            - np.ndarray: The indices of the selected coreset points within X.
            - np.ndarray: The weights of the selected points.
        """
        n = X.shape[0]
        if n <= r:
            # No need to compress if data is smaller than target coreset size
            return np.arange(n), np.ones(n) / n

        # 1. Kernel Selection: Compute the Sinkhorn kernel matrix
        # As suggested by Lemma 12 and Section 4.3, we can use the entropic
        # [cite_start]transport plan πε_Pn,Pn as an efficient approximation of the kernel[cite: 261, 274].
        sinkhorn_kernel = self._compute_sinkhorn_kernel(X)

        # 2. Kernel Compression: Nyström Approximation
        # [cite_start]Use Nyström to get the top r eigenvectors and eigenvalues[cite: 111, 208].
        # [cite_start]This is Algorithm 4 in the paper[cite: 447].
        U, S = self._nystrom_approx(sinkhorn_kernel, r, self.theta)
        
        # 3. Kernel Compression: Recombination
        # Find points and weights that match the moments of the full data with
        # [cite_start]respect to the eigenfunctions U[cite: 110]. We use leverage score
        # sampling for point selection and constrained optimization for weights.
        coreset_indices, coreset_weights = self._find_coreset_points_and_weights(U, r)

        return coreset_indices, coreset_weights

    def _compute_sinkhorn_kernel(self, X: np.ndarray) -> np.ndarray:
        """Computes the Sinkhorn kernel matrix (transport plan) using geomloss."""
        X_t = torch.from_numpy(X).float()
        # The blur parameter in geomloss is sigma = sqrt(epsilon)
        blur = np.sqrt(self.reg)
        
        # Define the Sinkhorn divergence routine
        loss = geomloss.SamplesLoss(
            loss="sinkhorn", p=2, blur=blur, backend="tensorized", potentials=True
        )
        
        # To get the transport plan, we compute the potentials F and G
        F, G = loss(X_t, X_t)
        
        # The transport plan is given by π = exp((F(x) + G(y) - C(x,y))/ε) * α(x)β(y)
        # Here, α and β are uniform, so we can get π from the potentials.
        C = (X_t[:, None, :] - X_t[None, :, :]).norm(2, dim=2) ** 2
        log_pi = (F[:, None] + G[None, :] - C) / self.reg
        
        # The kernel is n * pi
        kernel = X.shape[0] * torch.exp(log_pi)
        
        return kernel.numpy()


    def _nystrom_approx(self, K: np.ndarray, r: int, theta: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        [cite_start]Performs Nyström approximation based on Algorithm 4 from the paper[cite: 447].

        Args:
            K (np.ndarray): The (n x n) kernel matrix.
            r (int): The target rank.
            theta (int): The oversampling parameter.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Eigenvectors U and eigenvalues S.
        """
        n = K.shape[0]
        l = min(r * theta, n) # Oversampled rank

        # Random projection matrix
        Omega = np.random.randn(n, l)
        
        # Form the sketched matrix Y = K @ Omega
        Y = K @ Omega
        
        # QR decomposition of Y for a stable basis
        Q, _ = np.linalg.qr(Y)
        
        # Project K onto the smaller space
        B = Q.T @ K @ Q
        
        # Eigendecomposition of the small matrix B
        try:
            eigvals, eigvecs = np.linalg.eigh(B)
        except np.linalg.LinAlgError:
            # Fallback for non-positive-semidefinite matrices due to numerical precision
            print("Warning: Eigendecomposition failed, using SVD as fallback.")
            _, eigvals, eigvecs_t = np.linalg.svd(B)
            eigvecs = eigvecs_t.T

        # Sort eigenvalues and eigenvectors in descending order
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        
        # Truncate to the desired rank r
        eigvals = eigvals[:r]
        eigvecs = eigvecs[:, :r]

        # Reconstruct the full-size eigenvectors U
        U = Q @ eigvecs
        S = eigvals

        return U, S
        
    def _find_coreset_points_and_weights(self, U: np.ndarray, r: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Selects coreset points and computes their weights to match moments.

        This implementation uses leverage score sampling to select points, as high-
        leverage points are most influential in the kernel space. Weights are then
        found by solving a constrained optimization problem to ensure they are
        non-negative and sum to one, while matching the moments of the full data.
        """
        n, k = U.shape
        
        # 1. Select points using leverage scores
        leverage_scores = np.sum(U**2, axis=1)
        probabilities = leverage_scores / np.sum(leverage_scores)
        
        # Ensure we select unique points
        selected_indices = np.random.choice(n, size=r, replace=False, p=probabilities)
        U_coreset = U[selected_indices, :]

        # 2. Compute moments of the full dataset
        # The moment of the j-th eigenfunction is the average value of U[:, j]
        b = np.mean(U, axis=0)

        # 3. Solve for weights w
        # We want to find w such that U_coreset.T @ w ≈ b, with w >= 0.
        # This is a constrained least squares problem. We formulate it as a
        # Quadratic Program: min ||U_coreset.T @ w - b||^2 s.t. w >= 0
        
        # Objective function for the optimizer
        def objective(w):
            return np.linalg.norm(U_coreset.T @ w - b)**2

        # Constraints: weights must be non-negative
        constraints = ({'type': 'ineq', 'fun': lambda w: w})
        
        # Initial guess: uniform weights
        w0 = np.ones(r) / r
        
        # Bounds for weights (non-negative)
        bounds = [(0, None) for _ in range(r)]

        # Solve the QP
        solution = minimize(objective, w0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        weights = solution.x
        
        # Normalize weights to sum to 1 for a proper probability distribution
        if np.sum(weights) > 1e-6:
            weights /= np.sum(weights)
        else:
            # Fallback to uniform weights if optimization fails
            print("Warning: Optimization for weights failed. Using uniform weights.")
            weights = np.ones(r) / r
            
        return selected_indices, weights


    def get_final_coreset(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Tuple[int, int]]]:
        """
        Returns the final coreset after potentially running one last compression.
        """
        # If the final buffer is still larger than the target, compress it one last time.
        if self.buffer.shape[0] > self.wanted_coreset_size:
            self._compress_buffer()

        # If weights haven't been computed yet (e.g., buffer never filled), assign uniform weights.
        if self.weights.shape[0] != self.buffer.shape[0]:
            n_final = self.buffer.shape[0]
            self.weights = np.ones(n_final) / n_final if n_final > 0 else np.array([])
            
        return (
            self.buffer,
            np.array(self.global_indices),
            self.weights,
            self.provenance
        )

    def print_coreset_provenance(self) -> None:
        """
        Prints the detailed origin and weight of each point in the final coreset.
        """
        coreset, global_ids, weights, provenance = self.get_final_coreset()

        if coreset.shape[0] == 0:
            print("Coreset is empty.")
            return

        print("\n--- Final Coreset Provenance ---")
        print(f"Total points in coreset: {coreset.shape[0]}")
        print("-" * 30)
        header = f"{'Global ID':<12} {'Batch ID':<10} {'Index in Batch':<16} {'Weight':<15}"
        print(header)
        print("=" * len(header))

        for i in range(coreset.shape[0]):
            global_id = global_ids[i]
            batch_id, local_id = provenance[i]
            weight = weights[i]
            print(f"{global_id:<12} {batch_id:<10} {local_id:<16} {weight:<15.6f}")
        print("-" * 30)


# --- Example Usage ---
if __name__ == '__main__':
    # --- Configuration ---
    D = 10  # Dimension of data
    NUM_BATCHES = 20
    BATCH_SIZE = 100
    
    # Coreset parameters
    WANTED_CORESET_SIZE = 50
    BUFFER_CAPACITY = 200
    [cite_start]SINKHORN_REG = 2.0 * D # As suggested in the paper's experiments [cite: 284]

    # --- Data Stream Simulation ---
    # Create a dummy data stream from a mixture of Gaussians
    np.random.seed(42)
    # The stream will have 3 clusters
    centers = [np.random.randn(D) * 5 for _ in range(3)]
    
    def data_generator(num_batches, batch_size):
        for i in range(num_batches):
            # Pick a random center for this batch
            center = centers[i % len(centers)]
            batch_data = np.random.randn(batch_size, D) + center
            yield batch_data, i

    # --- Initialize and run the streamer ---
    streamer = CO2Streamer(
        wanted_coreset_size=WANTED_CORESET_SIZE,
        buffer_capacity=BUFFER_CAPACITY,
        reg=SINKHORN_REG
    )

    data_stream = data_generator(NUM_BATCHES, BATCH_SIZE)

    for batch, batch_idx in data_stream:
        streamer.process_batch(batch, batch_idx)

    # --- Get and print the final results ---
    streamer.print_coreset_provenance()

    # You can also get the final data directly
    final_coreset_points, _, final_weights, _ = streamer.get_final_coreset()
    print(f"\nShape of the final coreset data: {final_coreset_points.shape}")
    print(f"Shape of the final coreset weights: {final_weights.shape}")
    print(f"Sum of final weights: {np.sum(final_weights):.2f}")