"""Henon map: chaotic permutation / XOR for watermark confidentiality."""

from __future__ import annotations

import numpy as np


def henon_iterate(
    x0: float,
    y0: float,
    n: int,
    a: float = 1.4,
    b: float = 0.3,
    discard: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate ``n`` Henon samples after discarding transients."""
    x, y = float(x0), float(y0)
    for _ in range(discard):
        x, y = 1.0 - a * x * x + y, b * x
    xs = np.empty(n, dtype=np.float64)
    ys = np.empty(n, dtype=np.float64)
    for i in range(n):
        x, y = 1.0 - a * x * x + y, b * x
        xs[i], ys[i] = x, y
    return xs, ys


def chaotic_permutation(length: int, key: tuple[float, float, float, float]) -> np.ndarray:
    """
    Return a permutation of ``0..length-1`` from Henon trajectories.

    ``key`` = (x0, y0, a, b) with default a,b used if you pass (x0, y0, 1.4, 0.3).
    """
    x0, y0, a, b = key
    xs, ys = henon_iterate(x0, y0, length * 2 + 800, a=a, b=b, discard=500)
    scores = xs[:length] + ys[length : length * 2]
    return np.argsort(scores)


def encrypt_bits(bits: np.ndarray, key: tuple[float, float, float, float]) -> tuple[np.ndarray, np.ndarray]:
    """Permute bit order; return (encrypted_bits, inverse_perm)."""
    b = np.asarray(bits, dtype=np.float64).ravel()
    n = b.size
    perm = chaotic_permutation(n, key)
    inv = np.empty_like(perm)
    inv[perm] = np.arange(n)
    return b[perm], inv


def decrypt_bits(encrypted: np.ndarray, inv_perm: np.ndarray) -> np.ndarray:
    """``encrypted[k] = original[perm[k]]`` → ``original[j] = encrypted[inv_perm[j]]``."""
    return np.asarray(encrypted, dtype=np.float64)[inv_perm].copy()
