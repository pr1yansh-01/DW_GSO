"""2D DCT helpers for 8×8 blocks."""

from __future__ import annotations

import numpy as np
from scipy.fftpack import dct, idct


def dct2(block: np.ndarray) -> np.ndarray:
    return dct(dct(block.T, norm="ortho", type=2).T, norm="ortho", type=2)


def idct2(block: np.ndarray) -> np.ndarray:
    return idct(idct(block.T, norm="ortho", type=2).T, norm="ortho", type=2)
