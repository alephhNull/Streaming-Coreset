from typing import Tuple, List
import numpy as np
import torch
from sklearn.cluster import KMeans
from streamers.abstract_streamer import AbstractStreamingCoreset

class FreeSelStreamer(AbstractStreamingCoreset):
    """
    A streaming coreset selector based on the FreeSel paper.

    This class implements a coreset selection algorithm for streaming data. It uses a buffer
    to accumulate data and then applies a selection strategy inspired by the FreeSel paper,
    which involves identifying semantic patterns (via clustering) and then selecting a diverse
    subset based on distance-based sampling.

    Attributes:
        buffer_capacity (int): The maximum number of data points to store in the buffer
            before running the coreset selection.
        coreset_size (int): The target size of the coreset to be selected from the buffer.
        batch_size (int): The number of data points in each incoming batch.
        num_semantic_patterns (int): The number of clusters (semantic patterns) to identify
            in the buffered data. This is analogous to the 'K' parameter in the paper.
        sampling_strategy (str): The strategy for sampling from the semantic patterns.
            Can be 'prob' for probabilistic distance-based sampling or 'FDS' for
            farthest-distance sampling.
    """

    def __init__(self, buffer_capacity: int, coreset_size: int, batch_size: int, random_seed: int,
                 num_semantic_patterns: int = 5, sampling_strategy: str = 'prob'):
        """
        Initializes the FreeSelStreamer.

        Args:
            buffer_capacity (int): The maximum size of the data buffer.
            coreset_size (int): The desired size of the coreset.
            batch_size (int): The size of incoming data batches.
            num_semantic_patterns (int): The number of semantic patterns (clusters) to find.
            sampling_strategy (str): The sampling strategy to use ('prob' or 'FDS').
        """
        if coreset_size > buffer_capacity:
            raise ValueError("Coreset size cannot be larger than buffer capacity.")
        
        self.buffer_capacity = buffer_capacity
        self.coreset_size = coreset_size
        self.batch_size = batch_size
        self.num_semantic_patterns = num_semantic_patterns
        self.sampling_strategy = sampling_strategy
        self.random_seed = random_seed

        self._buffer = np.array([])
        self._provenance_buffer = []
        self._final_coreset = []
        self._final_provenance = []

        if random_seed is not None:
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)

    def process_batch(self, X_batch_np: np.ndarray, batch_idx: int) -> None:
        """
        Processes a single batch of data from the stream.

        This method adds the new data to a buffer. If the buffer exceeds its capacity,
        it triggers the coreset selection process to reduce the buffer size.

        Args:
            X_batch_np (np.ndarray): The new batch of data points.
            batch_idx (int): The index of the current batch.
        """
        if self._buffer.size == 0:
            self._buffer = X_batch_np
        else:
            self._buffer = np.vstack([self._buffer, X_batch_np])

        self._provenance_buffer.extend([(batch_idx, i) for i in range(X_batch_np.shape[0])])

        if len(self._provenance_buffer) >= self.buffer_capacity:
            self._select_coreset_from_buffer()


    def _select_coreset_from_buffer(self) -> None:
        """
        Selects a coreset from the current buffer using the FreeSel-inspired methodology.
        """
        # 1. Semantic Pattern Extraction (via K-Means clustering)
        num_points = self._buffer.shape[0]
        # Ensure n_clusters is not greater than the number of samples
        n_clusters = min(self.num_semantic_patterns, num_points)
        if n_clusters == 0:
            return # Cannot select from an empty buffer
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto')
        point_to_pattern_map = kmeans.fit_predict(self._buffer)
        semantic_patterns = kmeans.cluster_centers_

        # 2. Distance-based Sampling (Constructive Selection)
        if self.sampling_strategy == 'prob':
            selected_indices = self._probabilistic_sampling(point_to_pattern_map, semantic_patterns)
        elif self.sampling_strategy == 'FDS':
            selected_indices = self._farthest_distance_sampling(point_to_pattern_map, semantic_patterns)
        else:
            raise ValueError("Unknown sampling strategy. Choose 'prob' or 'FDS'.")
        
        # Update buffer with the selected coreset
        self._buffer = self._buffer[selected_indices]
        self._provenance_buffer = [self._provenance_buffer[i] for i in selected_indices]


    
    def _probabilistic_sampling(self, point_to_pattern_map: np.ndarray, semantic_patterns: np.ndarray) -> np.ndarray:
        """
        Performs probabilistic distance-based sampling, aligned with the paper's constructive algorithm.
        """
        num_points = self._buffer.shape[0]
        # Create a set of available indices to sample from
        unselected_indices = set(range(num_points))

        # Start with a single random point
        start_idx = np.random.choice(num_points)
        selected_indices = [start_idx]
        unselected_indices.remove(start_idx)

        # The pool of selected patterns is represented by the patterns of the selected data points
        selected_patterns = self._buffer[selected_indices]

        while len(selected_indices) < self.coreset_size and unselected_indices:
            # Calculate squared distances from each unselected point to the nearest selected point
            distances_sq = np.array([
                np.min(np.sum((self._buffer[i] - selected_patterns)**2, axis=1))
                for i in unselected_indices
            ])

            # Select next point with probability proportional to its squared distance
            probabilities = distances_sq / np.sum(distances_sq)
            
            # Sample from the list of *unselected* indices
            next_idx_local = np.random.choice(len(list(unselected_indices)), p=probabilities)
            next_idx = list(unselected_indices)[next_idx_local]

            selected_indices.append(next_idx)
            unselected_indices.remove(next_idx)
            
            # Update the pool of selected patterns for the next iteration
            selected_patterns = self._buffer[selected_indices]

        return np.array(selected_indices)


    def _farthest_distance_sampling(self, point_to_pattern_map: np.ndarray, semantic_patterns: np.ndarray) -> np.ndarray:
        """
        Performs farthest-distance sampling (FDS), aligned with the paper's constructive algorithm.
        """
        num_points = self._buffer.shape[0]
        # Create a set of available indices to sample from
        unselected_indices = set(range(num_points))
        
        # Start with a single random point
        start_idx = np.random.choice(num_points)
        selected_indices = [start_idx]
        unselected_indices.remove(start_idx)
        
        # The pool of selected patterns is represented by the patterns of the selected data points
        selected_patterns = self._buffer[selected_indices]
        
        while len(selected_indices) < self.coreset_size and unselected_indices:
            # For each unselected point, find the minimum distance to any already selected point
            distances = np.array([
                np.min(np.linalg.norm(self._buffer[i] - selected_patterns, axis=1))
                for i in unselected_indices
            ])
            
            # Find the point that is farthest away
            farthest_idx_local = np.argmax(distances)
            next_idx = list(unselected_indices)[farthest_idx_local]

            selected_indices.append(next_idx)
            unselected_indices.remove(next_idx)

            # Update the pool of selected patterns for the next iteration
            selected_patterns = self._buffer[selected_indices]

        return np.array(selected_indices)


    def get_final_coreset(self) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
        """
        Returns the final coreset after processing the entire stream.

        This method processes any remaining data in the buffer and then returns the
        final coreset indices, weights, and provenance.

        Returns:
            A tuple containing the flattened indices, weights, and provenance of the coreset.
        """
        # Process any remaining data in the buffer
        if len(self._provenance_buffer) > self.coreset_size:
            self._select_coreset_from_buffer()
        
        self._final_coreset = self._buffer
        self._final_provenance = self._provenance_buffer

        flat_indices = np.array([p[0] * self.batch_size + p[1] for p in self._final_provenance])
        weights = np.ones(len(self._final_provenance)) / len(self._final_provenance) # Uniform weights

        return flat_indices, weights, self._final_provenance

    def print_coreset_provenance(self) -> None:
        """
        Prints the provenance of each point in the final coreset.
        """
        if not self._final_provenance:
            self.get_final_coreset()

        print("Final Coreset Provenance:")
        for i, (batch_idx, local_idx) in enumerate(self._final_provenance):
            print(f"  Point {i}: from Batch {batch_idx}, Index {local_idx}")