"""DTCWT → real highpass plane → 8×8 DCT → SVD (largest singular value) embedding."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from medical_watermark.dtcwt_compat import Pyramid, dtcwt_forward, dtcwt_inverse
from medical_watermark.henon import decrypt_bits, encrypt_bits
from medical_watermark.transforms import dct2, idct2


@dataclass
class WatermarkState:
    """Everything needed to extract (semi-blind: original host required)."""

    nlevels: int
    orientation: int
    grid_shape: tuple[int, int]  # nrow, ncol of 8×8 blocks on highpass plane
    henon_inv: np.ndarray
    host_shape: tuple[int, int]


def _highpass_real_plane(pyramid: Pyramid, level: int, orientation: int) -> np.ndarray:
    hp = pyramid.highpasses[level]
    return np.real(hp[:, :, orientation].copy())


def _set_highpass_real(
    pyramid: Pyramid,
    level: int,
    orientation: int,
    real_plane: np.ndarray,
) -> Pyramid:
    hp = np.array(pyramid.highpasses[level], dtype=np.complex128, copy=True)
    imag = np.imag(hp[:, :, orientation])
    hp[:, :, orientation] = real_plane.astype(np.float64) + 1j * imag
    passes = list(pyramid.highpasses)
    passes[level] = hp
    return Pyramid(np.array(pyramid.lowpass, copy=True), tuple(passes))


def capacity_from_host(host: np.ndarray, nlevels: int = 1) -> tuple[tuple[int, int], int]:
    """Return ((nrow, ncol), num_blocks) for finest highpass after ``nlevels`` DTCWT."""
    p = dtcwt_forward(host, nlevels=nlevels)
    plane = _highpass_real_plane(p, 0, 0)
    h, w = plane.shape
    gh, gw = h // 8, w // 8
    return (gh, gw), gh * gw


def embed(
    host: np.ndarray,
    watermark_bits: np.ndarray,
    alpha: float,
    henon_key: tuple[float, float, float, float],
    nlevels: int = 1,
    orientation: int = 0,
) -> tuple[np.ndarray, WatermarkState]:
    """
    Embed binary watermark (vector of 0/1) into ``host`` (2D float [0,1]).

    Returns watermarked image and state for extraction.
    """
    bits = np.asarray(watermark_bits, dtype=np.float64).ravel()
    grid_shape, cap = capacity_from_host(host, nlevels=nlevels)
    if bits.size != cap:
        raise ValueError(f"Watermark length {bits.size} != capacity {cap}")

    enc_bits, inv = encrypt_bits(bits, henon_key)
    polar = 2.0 * enc_bits - 1.0

    p = dtcwt_forward(host, nlevels=nlevels)
    plane = _highpass_real_plane(p, 0, orientation)
    gh, gw = grid_shape
    h, w = gh * 8, gw * 8
    plane = plane[:h, :w]
    out_plane = plane.copy()

    idx = 0
    for i in range(gh):
        for j in range(gw):
            r, c = i * 8, j * 8
            blk = out_plane[r : r + 8, c : c + 8]
            d = dct2(blk)
            u, s, vt = np.linalg.svd(d, full_matrices=False)
            s_mod = s.astype(np.float64).copy()
            s_mod[0] = s_mod[0] + float(alpha) * polar[idx]
            d_new = u @ np.diag(s_mod) @ vt
            blk_new = idct2(d_new)
            out_plane[r : r + 8, c : c + 8] = blk_new
            idx += 1

    full_plane = _highpass_real_plane(p, 0, orientation)
    full_plane[:h, :w] = out_plane
    p_new = _set_highpass_real(p, 0, orientation, full_plane)
    wm_image = dtcwt_inverse(p_new)
    wm_image = np.clip(np.real(wm_image), 0.0, 1.0)
    if wm_image.shape != host.shape:
        wm_image = wm_image[: host.shape[0], : host.shape[1]]
        wm_image = np.clip(wm_image, 0.0, 1.0)

    state = WatermarkState(
        nlevels=nlevels,
        orientation=orientation,
        grid_shape=grid_shape,
        henon_inv=inv,
        host_shape=host.shape[:2],
    )
    return wm_image.astype(np.float64), state


def extract(
    host: np.ndarray,
    watermarked: np.ndarray,
    state: WatermarkState,
) -> np.ndarray:
    """Extract binary vector (values near 0/1) using ``host`` and embedding state."""
    p0 = dtcwt_forward(host, nlevels=state.nlevels)
    p1 = dtcwt_forward(watermarked, nlevels=state.nlevels)
    gh, gw = state.grid_shape
    h, w = gh * 8, gw * 8
    ori = state.orientation

    bits = np.zeros(gh * gw, dtype=np.float64)
    idx = 0
    for i in range(gh):
        for j in range(gw):
            r, c = i * 8, j * 8
            b0 = _highpass_real_plane(p0, 0, ori)[r : r + 8, c : c + 8]
            b1 = _highpass_real_plane(p1, 0, ori)[r : r + 8, c : c + 8]
            d0 = dct2(b0)
            d1 = dct2(b1)
            _, s0, _ = np.linalg.svd(d0, full_matrices=False)
            _, s1, _ = np.linalg.svd(d1, full_matrices=False)
            diff = float(s1[0] - s0[0])
            bits[idx] = 1.0 if diff >= 0.0 else 0.0
            idx += 1

    raw = decrypt_bits(bits, state.henon_inv)
    return raw
