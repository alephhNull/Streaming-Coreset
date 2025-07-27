from abc import ABC, abstractmethod
from typing import Tuple, List
import numpy as np
import torch


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
