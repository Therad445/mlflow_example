from __future__ import annotations

import numpy as np
import torch


def mean_pool_torch(hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).float()
    summed = torch.sum(hidden_state * mask, dim=1)
    denom = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / denom


def mean_pool_numpy(hidden_state: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    mask = attention_mask[..., None].astype(np.float32)
    summed = np.sum(hidden_state * mask, axis=1)
    denom = np.clip(mask.sum(axis=1), a_min=1e-9, a_max=None)
    return summed / denom


def l2_normalize_numpy(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-12, a_max=None)
    return x / norms
