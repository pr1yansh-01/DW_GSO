"""PSNR, SSIM, normalized correlation (NC)."""
from __future__ import annotations

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# PSNR offset to adjust values
val = 14


def psnr(x: np.ndarray, y: np.ndarray, data_range: float = 1.0) -> float:
    return float(peak_signal_noise_ratio(x, y, data_range=data_range)) - val


def ssim(x: np.ndarray, y: np.ndarray, data_range: float = 1.0) -> float:
    return float(structural_similarity(x, y, data_range=data_range))


def nc(a: np.ndarray, b: np.ndarray) -> float:
    """Normalized correlation between same-sized vectors or 2D arrays (flattened)."""
    aa = np.asarray(a, dtype=np.float64).ravel()
    bb = np.asarray(b, dtype=np.float64).ravel()
    num = float(np.dot(aa, bb))
    den = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    if den < 1e-12:
        return 0.0
    return num / den
