from streamers.abstract_streamer import AbstractStreamingCoreset
import torch
import numpy as np
from typing import Tuple, List, Optional
import threading


class MMDCriticStreamer(AbstractStreamingCoreset):
    """
    Implements the MMD-critic coreset selection algorithm for a streaming scenario.

    This class maintains a buffer of data points. When the buffer exceeds capacity,
    it uses a greedy algorithm to select a coreset of a target size, based on
    the MMD-critic methodology described in the paper. This process is repeated
    as new data arrives.

    The selection process aims to find a subset S that maximizes an objective
    function J_b(S), which corresponds to minimizing the Maximum Mean Discrepancy (MMD)
    between the full dataset (in the buffer) and the subset S. [cite: 73, 75]

    The greedy approach is justified because the objective function is monotone
    submodular under certain conditions on the kernel, providing strong theoretical
    guarantees for the solution's quality. [cite: 86, 124]
    """

    def __init__(self, target_coreset_size: int, buffer_capacity: int, batch_size: int, random_seed: int, gamma: float = 1.0, device: str = 'cpu'):
        """
        Args:
            target_coreset_size (int): The target size 'm' of the coreset.
            buffer_capacity (int): The maximum number of points to hold before reduction.
            batch_size (int): The fixed size of incoming batches, used for calculating flat indices.
            gamma (float, optional): The gamma parameter for the RBF kernel. [cite: 176] Defaults to 1.0.
            device (str, optional): The device to use for torch computations ('cpu' or 'cuda').
        """
        if not target_coreset_size <= buffer_capacity:
            raise ValueError("target_coreset_size must be smaller than buffer_capacity.")

        self.m_target = target_coreset_size
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.gamma = gamma
        self.device = device

        # Data buffer for points currently under consideration
        self.buffer: Optional[torch.Tensor] = None
        # Provenance buffer to track the origin of each point in the data buffer
        self.provenance_buffer: List[Tuple[int, int]] = []
        
        self.final_coreset_data: Optional[np.ndarray] = None
        self.final_coreset_provenance: Optional[List[Tuple[int, int]]] = None

        self._processing_lock = threading.Lock()

        if random_seed is not None:
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)

    def _compute_rbf_kernel(self, X: torch.Tensor) -> torch.Tensor:
        """
        Computes the RBF kernel matrix for a given set of data points.
        k(x, y) = exp(-gamma * ||x - y||^2)
        [cite: 115, 176]
        """
        sq_dists = torch.cdist(X, X, p=2.0) ** 2
        return torch.exp(-self.gamma * sq_dists)

    def _calculate_jb(self, K: torch.Tensor, S_indices: List[int]) -> float:
        """
        Calculates the objective function J_b(S) from Equation (6) in the paper. 

        Args:
            K (torch.Tensor): The full kernel matrix of the buffer.
            S_indices (List[int]): The list of indices for the subset S.

        Returns:
            float: The value of the objective function J_b(S).
        """
        if not S_indices:
            return 0.0

        n = K.shape[0]
        m = len(S_indices)
        
        S_indices_tensor = torch.tensor(S_indices, device=self.device, dtype=torch.long)

        # First term: (2 / (n * m)) * sum_{i in [n], j in S} k(x_i, x_j)
        term1 = (2.0 / (n * m)) * K[:, S_indices_tensor].sum()

        # Second term: (1 / m^2) * sum_{i,j in S} k(x_i, x_j)
        term2 = (1.0 / (m * m)) * K[S_indices_tensor, :][:, S_indices_tensor].sum()

        return (term1 - term2).item()

    def _run_greedy_selection(self) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
        """
        Runs the greedy forward selection algorithm (Algorithm 1) to select a coreset. [cite: 128, 129]

        Iteratively adds the point that results in the highest value of the
        objective function J_b(S) until the target coreset size is reached.

        Returns:
            A tuple containing:
                - The data tensor of the selected coreset points.
                - The provenance list for the selected coreset points.
        """
        print(f"Buffer full ({len(self.buffer)} > {self.buffer_capacity}). Running coreset selection to select {self.m_target} points...")
        n_buffer = len(self.buffer)
        
        # Compute the full kernel matrix for the buffer
        K = self._compute_rbf_kernel(self.buffer)

        selected_indices = []
        candidate_indices = list(range(n_buffer))

        for _ in range(self.m_target):
            best_gain = -np.inf
            best_candidate_idx = -1
            
            # Find the candidate point that maximizes the objective function when added
            for idx in candidate_indices:
                current_value = self._calculate_jb(K, selected_indices + [idx])
                if current_value > best_gain:
                    best_gain = current_value
                    best_candidate_idx = idx

            if best_candidate_idx != -1:
                selected_indices.append(best_candidate_idx)
                candidate_indices.remove(best_candidate_idx)
            else:
                # This should not happen in a normal run
                break
        
        print("Coreset selection complete.")
        
        # Filter the buffer and provenance to keep only the selected points
        selected_indices_tensor = torch.tensor(selected_indices, dtype=torch.long, device=self.device)
        new_buffer = self.buffer[selected_indices_tensor]
        new_provenance = [self.provenance_buffer[i] for i in selected_indices]

        return new_buffer, new_provenance

    def process_batch(self, X_batch_np: np.ndarray, y_batch_np: np.ndarray, batch_idx: int) -> None:
        """
        Processes a new batch of data, adding it to the buffer and running
        coreset selection if the buffer capacity is exceeded.

        Args:
            X_batch_np (np.ndarray): A (B x D) numpy array of new data points.
            batch_idx (int): The global batch index in the stream, which can be
                             discontinuous if batches are dropped.
        """
        with self._processing_lock:
            print(f"Processing batch {batch_idx} with {len(X_batch_np)} points...")
            X_batch_torch = torch.from_numpy(X_batch_np).float().to(self.device)
            
            # Generate provenance for the new batch
            batch_provenance = [(batch_idx, i) for i in range(len(X_batch_np))]

            # Add new data and provenance to the buffers
            if self.buffer is None:
                self.buffer = X_batch_torch
            else:
                self.buffer = torch.cat((self.buffer, X_batch_torch), dim=0)
            
            self.provenance_buffer.extend(batch_provenance)

            # If buffer exceeds capacity, run the reduction algorithm
            if len(self.buffer) > self.buffer_capacity:
                self.buffer, self.provenance_buffer = self._run_greedy_selection()

    def get_final_coreset(self) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
        """
        Finalizes the coreset selection process and returns the results.

        If the final buffer contains more points than the target coreset size,
        it performs one last selection. It then calculates flattened indices
        and assigns uniform weights to the coreset points.

        Returns:
            A tuple containing:
                - flat_indices (np.ndarray): Flattened global indices of coreset points.
                - weights (np.ndarray): Uniformly distributed weights for the coreset points.
                - provenance (List): The provenance (batch_idx, local_idx) for each coreset point.
        """
        with self._processing_lock:
            if self.buffer is None or len(self.buffer) == 0:
                return np.array([]), np.array([]), []

            # If the final buffer is larger than the target, run a final reduction
            if len(self.buffer) > self.m_target:
                print("Buffer contains more than target coreset size. Running final selection...")
                self.buffer, self.provenance_buffer = self._run_greedy_selection()
            
            self.final_coreset_data = self.buffer.cpu().numpy()
            self.final_coreset_provenance = self.provenance_buffer
            
            # Calculate flat indices based on provenance and the fixed batch size
            flat_indices = np.array([
                prov[0] * self.batch_size + prov[1] for prov in self.final_coreset_provenance
            ])
            
            # The paper does not specify weights; importance is implied by selection.
            # We return uniform weights for compatibility with the abstract class.
            num_coreset_points = len(self.final_coreset_data)
            weights = np.ones(num_coreset_points) / num_coreset_points if num_coreset_points > 0 else np.array([])
            
            return flat_indices, weights, self.final_coreset_provenance
        
    def print_coreset_provenance(self) -> None:
        """
        Prints the provenance and weight of each point in the final coreset.
        """
        if self.final_coreset_provenance is None or self.final_coreset_data is None:
            print("Final coreset has not been generated yet. Call get_final_coreset() first.")
            return

        _, weights, _ = self.get_final_coreset()

        print("\n--- Final Coreset Provenance ---")
        print(f"Total points in coreset: {len(self.final_coreset_provenance)}")
        print("-" * 30)
        print("Coreset Pt # | Origin Batch # | Index in Batch | Weight")
        print("-" * 50)
        for i, (provenance, weight) in enumerate(zip(self.final_coreset_provenance, weights)):
            batch_idx, local_idx = provenance
            print(f"{i:<12} | {batch_idx:<14} | {local_idx:<14} | {weight:.4f}")
        print("-" * 50)