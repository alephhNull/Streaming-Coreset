from typing import Tuple, List
import numpy as np
import torch
from scipy.spatial.distance import cdist
import threading
from streamers.abstract_streamer import AbstractStreamingCoreset


class CAMELStreamer(AbstractStreamingCoreset):
    """
    Implements the CAMEL coreset selection algorithm for streaming data.

    This class uses a merge-reduce strategy to maintain a buffer of representative
    points from the stream. When new data arrives, it's merged with the buffer.
    If the buffer exceeds its capacity, a greedy coreset selection algorithm
    (as described in Algorithm 1 of the paper) is run to reduce it back to size.
    """

    def __init__(self, buffer_capacity: int, coreset_size: int, batch_size: int, random_seed: int, weight_clip_range: Tuple[float, float] = (0.0, 1e9)):
        """
        Initializes the CAMELStreamer.

        Args:
            buffer_capacity (int): The maximum number of points to keep in the buffer.
            coreset_size (int): The target size of the final coreset.
            weight_clip_range (Tuple[float, float]): The min and max values for weight clipping.
        """
        if buffer_capacity < coreset_size:
            raise ValueError("Buffer capacity must be greater than or equal to the coreset size.")

        self.buffer_capacity = buffer_capacity
        self.coreset_size = coreset_size
        self.weight_clip_range = weight_clip_range
        self.random_seed = random_seed
        self.batch_size = batch_size

        self.buffer = np.array([])
        self.buffer_provenance: List[Tuple[int, int]] = []

        self._final_coreset_indices: np.ndarray = np.array([])
        self._final_coreset_weights: np.ndarray = np.array([])
        self._final_coreset_provenance: List[Tuple[int, int]] = []

        self._is_processing = False
        self._lock = threading.Lock()

        if random_seed is not None:
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)

    def _greedy_coreset_selection(self, data: np.ndarray, target_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Selects a coreset from the given data using a greedy approach.

        This implements a simplified version of Algorithm 1 from the paper,
        which minimizes the distance between the full dataset and the coreset.

        Args:
            data (np.ndarray): The data to select a coreset from.
            target_size (int): The desired size of the coreset.

        Returns:
            Tuple containing the indices of the selected coreset points and their calculated weights.
        """
        if len(data) <= target_size:
            # If the data is smaller than the target size, all points are selected
            indices = np.arange(len(data))
            weights = np.ones(len(data))
            return indices, weights


        num_points = data.shape[0]
        # Normalize features to unit length as mentioned in the paper
        data_normalized = data / (np.linalg.norm(data, axis=1, keepdims=True) + 1e-9)

        # Calculate all pairwise distances once
        all_distances = cdist(data_normalized, data_normalized, 'euclidean')

        # This will store the minimum distance from each point to a selected coreset point
        min_distances = np.full(num_points, np.inf)
        selected_indices = []

        for _ in range(target_size):
            marginal_gains = np.zeros(num_points)
            # Create a mask for points not yet selected
            not_selected_mask = np.ones(num_points, dtype=bool)
            if selected_indices:
                not_selected_mask[selected_indices] = False
            
            candidate_indices = np.where(not_selected_mask)[0]

            for idx in candidate_indices:
                # Calculate the potential new minimum distances if this point is added
                potential_new_min_distances = np.minimum(min_distances, all_distances[:, idx])
                # The gain is the reduction in the sum of distances
                marginal_gains[idx] = np.sum(min_distances - potential_new_min_distances)
            
            # Select the point with the highest marginal gain
            best_candidate_idx = np.argmax(marginal_gains)
            selected_indices.append(best_candidate_idx)

            # Update the minimum distances with the newly selected point
            min_distances = np.minimum(min_distances, all_distances[:, best_candidate_idx])
            
        selected_indices = np.array(selected_indices)
        
        # Calculate weights: for each coreset point, count how many data points are closest to it.
        # This corresponds to w_j = |V_j| in the paper.
        closest_coreset_point = np.argmin(all_distances[:, selected_indices], axis=1)
        weights = np.bincount(closest_coreset_point, minlength=target_size)
        
        return selected_indices, weights.astype(float)


    def process_batch(self, X_batch_np: np.ndarray, y_batch_np: np.ndarray, batch_idx: int) -> None:
        """
        Processes a single batch of data from the stream using the Merge-Reduce strategy.
        """
        if self._is_processing:
            print(f"Processor is busy. Dropping batch {batch_idx}.")
            return

        with self._lock:
            self._is_processing = True

        if self.batch_size == -1:
            self.batch_size = X_batch_np.shape[0]

        print(f"Processing batch {batch_idx}...")

        # --- Merge Step ---
        new_provenance = [(batch_idx, i) for i in range(X_batch_np.shape[0])]
        if self.buffer.size == 0:
             self.buffer = X_batch_np
        else:
             self.buffer = np.vstack([self.buffer, X_batch_np])
        self.buffer_provenance.extend(new_provenance)

        # --- Reduce Step ---
        if len(self.buffer) > self.buffer_capacity:
            print(f"Buffer full ({len(self.buffer)} > {self.buffer_capacity}). Reducing...")
            
            # Run coreset selection to reduce the buffer back to capacity
            indices_to_keep, _ = self._greedy_coreset_selection(self.buffer, self.buffer_capacity)
            
            self.buffer = self.buffer[indices_to_keep]
            self.buffer_provenance = [self.buffer_provenance[i] for i in indices_to_keep]

        print(f"Finished processing batch {batch_idx}. Buffer size: {len(self.buffer)}")
        with self._lock:
            self._is_processing = False

    def get_final_coreset(self) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
        """
        Selects the final coreset from the current buffer and computes weights and provenance.
        """
        with self._lock:
            if len(self.buffer) == 0:
                return np.array([]), np.array([]), []

            print(f"\nSelecting final coreset of size {self.coreset_size} from buffer of size {len(self.buffer)}...")
            
            # Select the final coreset from the current buffer
            coreset_indices_in_buffer, weights = self._greedy_coreset_selection(self.buffer, self.coreset_size)

            self._final_coreset_indices = coreset_indices_in_buffer
            self._final_coreset_provenance = [self.buffer_provenance[i] for i in coreset_indices_in_buffer]

            # As per the paper, scale and clip weights for fair comparison in training
            if len(coreset_indices_in_buffer)>0:
                sample_ratio = len(coreset_indices_in_buffer) / len(self.buffer)
                scaled_weights = weights * sample_ratio
                self._final_coreset_weights = np.clip(scaled_weights, self.weight_clip_range[0], self.weight_clip_range[1])
            else:
                 self._final_coreset_weights = np.array([])


            # Calculate flat indices
            flat_indices = np.array([batch_idx * self.batch_size + local_idx for batch_idx, local_idx in self._final_coreset_provenance])

        return flat_indices, self._final_coreset_weights, self._final_coreset_provenance

    def print_coreset_provenance(self) -> None:
        """
        Prints the provenance and weight of each point in the final coreset.
        """
        if len(self._final_coreset_provenance) == 0:
            print("\nFinal coreset is empty. Call get_final_coreset() first.")
            return

        print("\n--- Final Coreset Provenance ---")
        for i, (provenance, weight) in enumerate(zip(self._final_coreset_provenance, self._final_coreset_weights)):
            batch_id, local_id = provenance
            print(f"Point {i}: From Batch {batch_id}, Index {local_id} -> Weight: {weight:.4f}")
        print("---------------------------------")


if __name__ == '__main__':
    # --- Example Usage ---
    # Create a dummy data stream
    NUM_BATCHES = 20
    BATCH_SIZE = 100
    FEATURE_DIM = 10
    
    np.random.seed(42)
    # Create some clusters to make the data redundant
    centers = np.random.rand(5, FEATURE_DIM) * 10
    data_stream = []
    for _ in range(NUM_BATCHES):
        # Each batch is a mixture of points from the clusters
        batch_centers = centers[np.random.choice(5, BATCH_SIZE)]
        batch = batch_centers + np.random.randn(BATCH_SIZE, FEATURE_DIM) * 0.5
        data_stream.append(batch)

    # Initialize the streamer
    BUFFER_CAPACITY = 500
    CORESET_SIZE = 50
    camel_streamer = CAMELStreamer(buffer_capacity=BUFFER_CAPACITY, coreset_size=CORESET_SIZE)

    # Process the stream
    for i, batch_data in enumerate(data_stream):
        camel_streamer.process_batch(batch_data, batch_idx=i)
        
    # Get and print the final coreset
    flat_indices, weights, provenance = camel_streamer.get_final_coreset()
    camel_streamer.print_coreset_provenance()

    print(f"\nTotal points in final coreset: {len(flat_indices)}")
    print(f"Shape of flat indices: {flat_indices.shape}")
    print(f"Shape of weights: {weights.shape}")