from abc import ABC, abstractmethod
from typing import Tuple, List
import numpy as np
import torch
from sklearn.cluster import KMeans

class AbstractStreamingCoreset(ABC):
    """
    Abstract base class for streaming coreset selectors.
    All implementations must define how batches are processed, coresets are returned, and provenance is printed.
    """

    @abstractmethod
    def process_batch(self, X_batch_np: np.ndarray, batch_idx: int) -> None:
        """
        Process a batch of data from the stream.

        Args:
            X_batch_np (np.ndarray): A (B x D) numpy array of new data points.
            batch_idx (int): The global batch index in the stream.
        """
        pass

    @abstractmethod
    def get_final_coreset(self) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
        """
        Returns the final coreset.

        Returns:
            Tuple:
                - flat_indices (np.ndarray of shape (M,)): flattened indices of coreset points.
                - weights (np.ndarray of shape (M,)): normalized weights of selected coreset points.
                - provenance (List of Tuple[int, int]): list of (batch_idx, local_idx) for each coreset point.
        """
        pass

    @abstractmethod
    def print_coreset_provenance(self) -> None:
        """
        Prints the provenance (origin) of each point in the final coreset.
        Typically includes batch ID, point index within batch, and its weight.
        """
        pass


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

    def __init__(self, buffer_capacity: int, coreset_size: int, batch_size: int, 
                 num_semantic_patterns: int = 10, sampling_strategy: str = 'prob'):
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

        self._buffer = np.array([])
        self._provenance_buffer = []
        self._final_coreset = []
        self._final_provenance = []

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

        This method first applies K-Means clustering to find semantic patterns and then uses
        a distance-based sampling strategy to select a diverse coreset of the desired size.
        """
        # 1. Semantic Pattern Extraction (via K-Means clustering)
        kmeans = KMeans(n_clusters=self.num_semantic_patterns, random_state=0, n_init=10)
        labels = kmeans.fit_predict(self._buffer)
        semantic_patterns = kmeans.cluster_centers_

        # 2. Distance-based Sampling
        if self.sampling_strategy == 'prob':
            selected_indices = self._probabilistic_sampling(labels, semantic_patterns)
        elif self.sampling_strategy == 'FDS':
            selected_indices = self._farthest_distance_sampling(labels, semantic_patterns)
        else:
            raise ValueError("Unknown sampling strategy. Choose 'prob' or 'FDS'.")
        
        # Update buffer with the selected coreset
        self._buffer = self._buffer[selected_indices]
        self._provenance_buffer = [self._provenance_buffer[i] for i in selected_indices]


    def _probabilistic_sampling(self, labels: np.ndarray, semantic_patterns: np.ndarray) -> np.ndarray:
        """
        Performs probabilistic distance-based sampling.

        This method selects data points with a probability proportional to their distance
        from the nearest already selected semantic pattern.

        Args:
            labels (np.ndarray): The cluster labels for each point in the buffer.
            semantic_patterns (np.ndarray): The cluster centers (semantic patterns).

        Returns:
            np.ndarray: The indices of the selected coreset points.
        """
        selected_indices = []
        selected_pattern_indices = []

        # Start with a random data point
        initial_idx = np.random.choice(len(self._provenance_buffer))
        selected_indices.append(initial_idx)
        selected_pattern_indices.append(labels[initial_idx])

        while len(selected_indices) < self.coreset_size:
            min_distances = np.full(self.num_semantic_patterns, np.inf)
            for i in range(self.num_semantic_patterns):
                if i not in selected_pattern_indices:
                    # Calculate distance to the nearest selected pattern
                    distances_to_selected = np.linalg.norm(semantic_patterns[i] - semantic_patterns[selected_pattern_indices], axis=1)
                    min_distances[i] = np.min(distances_to_selected)

            # Select next pattern with probability proportional to squared distance
            probabilities = min_distances**2
            probabilities /= np.sum(probabilities)
            next_pattern_idx = np.random.choice(self.num_semantic_patterns, p=probabilities)
            
            # Find a data point from the selected pattern
            candidate_indices = np.where(labels == next_pattern_idx)[0]
            if len(candidate_indices) > 0:
                # Add a random point from this new cluster
                new_idx = np.random.choice(candidate_indices)
                if new_idx not in selected_indices:
                    selected_indices.append(new_idx)
                    selected_pattern_indices.append(next_pattern_idx)

        return np.array(selected_indices)


    def _farthest_distance_sampling(self, labels: np.ndarray, semantic_patterns: np.ndarray) -> np.ndarray:
        """
        Performs farthest-distance sampling (FDS).

        This method greedily selects the data point whose corresponding semantic pattern is
        farthest from any already selected patterns.

        Args:
            labels (np.ndarray): The cluster labels for each point in the buffer.
            semantic_patterns (np.ndarray): The cluster centers (semantic patterns).

        Returns:
            np.ndarray: The indices of the selected coreset points.
        """
        selected_indices = []
        selected_pattern_indices = []
        
        # Start with a random data point
        initial_idx = np.random.choice(len(self._provenance_buffer))
        selected_indices.append(initial_idx)
        selected_pattern_indices.append(labels[initial_idx])
        
        while len(selected_indices) < self.coreset_size:
            max_min_dist = -1
            next_pattern_idx = -1
            
            for i in range(self.num_semantic_patterns):
                if i not in selected_pattern_indices:
                    # Calculate distance to the nearest selected pattern
                    distances_to_selected = np.linalg.norm(semantic_patterns[i] - semantic_patterns[selected_pattern_indices], axis=1)
                    min_dist = np.min(distances_to_selected)
                    
                    if min_dist > max_min_dist:
                        max_min_dist = min_dist
                        next_pattern_idx = i
            
            if next_pattern_idx != -1:
                # Find a data point from the selected pattern
                candidate_indices = np.where(labels == next_pattern_idx)[0]
                if len(candidate_indices) > 0:
                    new_idx = np.random.choice(candidate_indices)
                    if new_idx not in selected_indices:
                        selected_indices.append(new_idx)
                        selected_pattern_indices.append(next_pattern_idx)
            else:
                # No more patterns to select
                break
                
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