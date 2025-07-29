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
        pass

    @abstractmethod
    def get_final_coreset(self) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
        pass

    @abstractmethod
    def print_coreset_provenance(self) -> None:
        pass