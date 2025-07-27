import numpy as np
from abstract_streamer import AbstractStreamingCoreset

class ReservoirSamplerBatchStreamer(AbstractStreamingCoreset):
    """
    Implements batch-wise reservoir sampling for a data stream.

    It maintains a reservoir of a fixed size, ensuring that after processing the
    entire stream, each point has had an equal probability of being included
    in the final coreset. It correctly handles dropped batches by tracking
    the global index of each point.
    """
    def __init__(self, coreset_size, batch_size, random_seed=None):
        """
        Args:
            coreset_size (int): The desired size of the final coreset (the reservoir size).
            random_seed (int, optional): Seed for the random number generator for reproducibility.
        """
        self.m = coreset_size
        self.batch_size = batch_size
        self.reservoir_indices = []  # Stores the flat global indices of points in the reservoir
        self.points_seen = 0
        self.rng = np.random.default_rng(random_seed) # Use modern numpy random generator

    def process_batch(self, X_batch_np, batch_idx):
        """
        Processes a single batch of data using the reservoir sampling algorithm.

        Args:
            X_batch_np (np.ndarray): The data from the current batch.
            batch_idx (int): The global index of the batch.
        """
        batch_size = X_batch_np.shape[0]
        # BATCH_SIZE must be known for index calculation. We assume it's constant.
        # A more robust implementation might get it from train_loader.batch_size.
        
        for i in range(batch_size):
            # Calculate the true, unique index of the point in the entire dataset
            # Note: This assumes a constant batch size.
            global_point_idx = batch_idx * batch_size + i
            
            self.points_seen += 1
            
            if len(self.reservoir_indices) < self.m:
                # Reservoir is not full, add the point's index
                self.reservoir_indices.append(global_point_idx)
            else:
                # Reservoir is full, decide whether to replace an existing point
                # The probability of replacement is m / points_seen
                j = self.rng.integers(0, self.points_seen)
                if j < self.m:
                    self.reservoir_indices[j] = global_point_idx
    
    def get_final_coreset(self):
        """
        Returns the final coreset indices and uniform weights.

        Returns:
            tuple: (indices, weights)
                - indices (np.ndarray): The flat global indices of the coreset points.
                - weights (np.ndarray): Uniform weights for the coreset points.
        """
        if not self.reservoir_indices:
            return np.array([], dtype=int), np.array([])
            
        final_indices = np.array(self.reservoir_indices, dtype=int)
        num_indices = len(final_indices)
        weights = np.ones(num_indices) / num_indices if num_indices > 0 else np.array([])
        
        return final_indices, weights, None

    def print_coreset_provenance(self):
        """
        Prints the origin of each coreset point and returns the flat indices.
        This is useful for debugging and analysis.

        Args:
            BATCH_SIZE (int): The batch size used during streaming, for back-calculation.

        Returns:
            np.ndarray: The flat global indices of the coreset points.
        """
        indices, weights, _ = self.get_final_coreset()
        if indices.size == 0:
            print("Coreset is empty.")
            return np.array([], dtype=int)

        print("\n--- Final Coreset Provenance (Reservoir) ---")
        for i, flat_idx in enumerate(indices):
            # Back-calculate for verification
            origin_batch = flat_idx // self.batch_size
            origin_idx_in_batch = flat_idx % self.batch_size
            print(f"  Point {i}: From Batch {origin_batch}, Idx {origin_idx_in_batch} (Flat Index: {flat_idx}) -> Weight: {weights[i]:.4f}")
        print("------------------------------------------")
