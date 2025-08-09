from abc import ABC, abstractmethod
from typing import Tuple, List
import numpy as np
import torch


class AbstractStreamingCoreset(ABC):
    """
    Abstract base class for streaming coreset selectors.
    """

    @abstractmethod
    def process_batch(self, X_batch_np: np.ndarray, y_batch_np: np.ndarray, batch_idx: int) -> None:
        """
        Processes a new batch of data, adding it to the buffer and running
        coreset selection if the buffer capacity is exceeded.

        Args:
            X_batch_np (np.ndarray): A (B x D) numpy array of new data points.
            batch_idx (int): The global batch index in the stream, which can be
                             discontinuous if batches are dropped.
        """
        pass

    @abstractmethod
    def get_final_coreset(self) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
        """
        Returns:
            A tuple containing:
                - flat_indices (np.ndarray): Flattened global indices of coreset points.
                - weights (np.ndarray): Uniformly distributed weights for the coreset points.
                - provenance (List): The provenance (batch_idx, local_idx) for each coreset point.
        """
        pass

    @abstractmethod
    def print_coreset_provenance(self) -> None:
        """
        Prints the provenance and weight of each point in the final coreset.
        """
        pass