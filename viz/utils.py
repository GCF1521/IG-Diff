"""Visualization utility functions."""

import torch
import numpy as np


def to_numpy(t: torch.Tensor) -> np.ndarray:
    """Convert tensor to numpy, avoiding .numpy() which fails with numpy 2.x."""
    return np.array(t.detach().cpu().tolist(), dtype=np.float32)