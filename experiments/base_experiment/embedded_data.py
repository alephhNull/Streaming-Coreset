from __future__ import annotations

import os
import sys
from typing import Tuple

import numpy as np

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

try:
    from dataloaders import load_dataset
except ImportError:
    load_dataset = None


def load_embedded_train_split(
    dataset_name: str,
    seed: int,
    subset_size: int = 50000,
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray]:
    if load_dataset is None:
        raise ImportError("dataloaders.load_dataset is required for embedded datasets")
    ds = dataset_name.lower().strip()
    if ds not in ("mnist", "cifar10", "fashion_mnist", "cifar100"):
        raise ValueError(
            f"Unsupported embedded dataset '{dataset_name}'. "
            "Extend embedded_data.load_embedded_train_split for more."
        )
    train_loader, val_loader, X_train, X_val, y_train, y_val = load_dataset(
        ds,
        subset_size,
        subset_size,
        seed,
        embedding="resnet18",
        embed_dim=None,
        device=device,
    )
    return X_train.astype(np.float64), y_train.astype(np.int64)
