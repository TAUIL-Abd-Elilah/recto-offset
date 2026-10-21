"""Model-free snaps the network must beat, run on the same frozen crops.

  zero    leave the surface where it is (offset 0)
  ridge   move each node to the nearest local maximum of the CT along its normal (profile smoothed 1.5 voxels along
          the normal and 2 nodes in-plane), the classic "snap to the brightest layer"
  edge    move each node to the nearest dark-to-bright rise going outward (maximum of d/dk), Hari Seldon's "middle
          of the dark-to-bright edge"
Each snap is corrected by a constant bias measured on the training windows (median offset of that feature from
the verified surface), so the baseline is not penalised for a fixed convention difference."""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def _features(x: np.ndarray, kind: str) -> np.ndarray:
    """x (N, 1, L, H, W) normalised view -> boolean candidates (N, L, H, W) and their strength."""
    v = ndimage.gaussian_filter(x[:, 0], (0, 1.5, 2, 2))
    if kind == 'edge':
        v = np.gradient(v, axis=1)
    mx = ndimage.maximum_filter(v, size=(1, 5, 1, 1))
    return (v == mx) & (v > np.percentile(v, 60, axis=1, keepdims=True))


def snap(x: np.ndarray, kind: str, bias: float = 0.0) -> np.ndarray:
    """Offset (voxels) from the view centre to the nearest candidate along the normal, minus bias."""
    cand = _features(x, kind)
    N, L, H, W = cand.shape
    R = (L - 1) // 2
    k = np.arange(-R, R + 1)[None, :, None, None]
    dist = np.where(cand, np.abs(k), 10 ** 6)
    idx = dist.argmin(1)
    off = (idx - R).astype(np.float32)
    off[dist.min(1) >= 10 ** 6] = 0.0
    return off - bias


def calibrate(x: np.ndarray, y: np.ndarray, m: np.ndarray, kind: str) -> float:
    """Bias = median(snap - target) on training crops."""
    return float(np.median((snap(x, kind) - y)[m]))
