"""Paper-style DTCWT(LL3) -> 8x8 DCT -> SVD watermark embedding."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import cv2

from medical_watermark.dtcwt_compat import Pyramid, dtcwt_forward, dtcwt_inverse
from medical_watermark.henon import chaotic_permutation
from medical_watermark.transforms import dct2, idct2


@dataclass
class WatermarkState:
    """Everything needed to extract (semi-blind: original host required)."""

    nlevels: int
    grid_shape: tuple[int, int]  # nrow, ncol of 8x8 LL3 blocks
    payload_shape: tuple[int, int]
    payload_len: int
    alpha: float
    henon_inv: np.ndarray
    wm_u: tuple[np.ndarray, ...]
    wm_vt: tuple[np.ndarray, ...]
    host_shape: tuple[int, int]


def _set_lowpass(pyramid: Pyramid, lowpass: np.ndarray) -> Pyramid:
    return Pyramid(np.asarray(lowpass, dtype=np.float64), tuple(pyramid.highpasses))


def _block_slices(grid_shape: tuple[int, int]):
    gh, gw = grid_shape
    for i in range(gh):
        for j in range(gw):
            r, c = i * 8, j * 8
            yield r, c


def _binary_payload(watermark: np.ndarray) -> np.ndarray:
    return (np.asarray(watermark, dtype=np.float64) >= 0.5).astype(np.float64)


def _henon_encrypt_image(image: np.ndarray, key: tuple[float, float, float, float]) -> tuple[np.ndarray, np.ndarray]:
    flat = image.ravel()
    perm = chaotic_permutation(flat.size, key)
    inv = np.empty_like(perm)
    inv[perm] = np.arange(flat.size)
    return flat[perm].reshape(image.shape), inv


def _henon_decrypt_image(image: np.ndarray, inv_perm: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    dec = np.asarray(image, dtype=np.float64).ravel()[inv_perm]
    return dec.reshape(shape)


def _binarize_extracted_watermark(image: np.ndarray) -> np.ndarray:
    """Binarize extracted watermark adaptively instead of using a fixed 0.5 cutoff."""
    x = np.clip(np.asarray(image, dtype=np.float64), 0.0, 1.0)
    u8 = np.clip(np.round(x * 255.0), 0, 255).astype(np.uint8)
    if float(u8.std()) < 1e-6:
        binary = (x >= 0.5).astype(np.float64)
        return _remove_isolated_bit_flips(binary)
    thr, _ = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = (u8 >= thr).astype(np.float64)
    return _remove_isolated_bit_flips(binary)


def _remove_isolated_bit_flips(binary: np.ndarray) -> np.ndarray:
    """
    Remove salt-and-pepper extraction errors without blurring logo edges.

    A black/white pixel is flipped only when its local 3x3 neighborhood strongly
    disagrees with it, so normal strokes and corners are preserved.
    """
    b = (np.asarray(binary, dtype=np.float64) >= 0.5).astype(np.float64)
    if min(b.shape[:2]) < 3:
        return b
    kernel = np.ones((3, 3), dtype=np.float64)
    count = cv2.filter2D(b, ddepth=-1, kernel=kernel, borderType=cv2.BORDER_REPLICATE)
    cleaned = b.copy()
    cleaned[(b == 1.0) & (count <= 2.0)] = 0.0
    cleaned[(b == 0.0) & (count >= 7.0)] = 1.0
    return cleaned


def capacity_from_host(host: np.ndarray, nlevels: int = 3) -> tuple[tuple[int, int], int]:
    """Return ((nrow, ncol), maximum binary-logo pixels) for LL3 8x8 blocks."""
    p = dtcwt_forward(host, nlevels=nlevels)
    low = np.asarray(p.lowpass, dtype=np.float64)
    h, w = low.shape[:2]
    gh, gw = h // 8, w // 8
    return (gh, gw), gh * gw * 64


def embed(
    host: np.ndarray,
    watermark: np.ndarray,
    alpha: float,
    henon_key: tuple[float, float, float, float],
    nlevels: int = 3,
) -> tuple[np.ndarray, WatermarkState]:
    """
    Embed a binary logo using the paper flow:
    DTCWT level-3 LL subband -> 8x8 DCT blocks -> SVD additive fusion.
    """
    payload = _binary_payload(watermark)
    payload_shape = payload.shape if payload.ndim == 2 else (1, payload.size)
    if payload_shape[0] % 8 or payload_shape[1] % 8:
        raise ValueError("Paper-style block SVD embedding requires watermark dimensions divisible by 8")

    grid_shape, cap = capacity_from_host(host, nlevels=nlevels)
    blocks_needed = (payload_shape[0] // 8) * (payload_shape[1] // 8)
    blocks_available = grid_shape[0] * grid_shape[1]
    if payload.size > cap or blocks_needed > blocks_available:
        raise ValueError(f"Watermark size {payload_shape} exceeds LL3 block capacity {grid_shape}")

    enc_payload, inv = _henon_encrypt_image(payload, henon_key)
    p = dtcwt_forward(host, nlevels=nlevels)
    low = np.asarray(p.lowpass, dtype=np.float64).copy()
    out_low = low.copy()

    wm_u: list[np.ndarray] = []
    wm_vt: list[np.ndarray] = []
    block_idx = 0
    for r, c in _block_slices(grid_shape):
        br = (block_idx // (payload_shape[1] // 8)) * 8
        bc = (block_idx % (payload_shape[1] // 8)) * 8
        if br >= payload_shape[0]:
            break

        host_block = out_low[r : r + 8, c : c + 8]
        wm_block = enc_payload[br : br + 8, bc : bc + 8]

        host_dct = dct2(host_block)
        h_u, h_s, h_vt = np.linalg.svd(host_dct, full_matrices=False)
        w_u, w_s, w_vt = np.linalg.svd(wm_block, full_matrices=False)
        wm_u.append(w_u)
        wm_vt.append(w_vt)

        h_s_mod = h_s.astype(np.float64).copy()
        h_s_mod[: w_s.size] = h_s_mod[: w_s.size] + float(alpha) * w_s
        out_low[r : r + 8, c : c + 8] = idct2(h_u @ np.diag(h_s_mod) @ h_vt)
        block_idx += 1

    p_new = _set_lowpass(p, out_low)
    wm_image = dtcwt_inverse(p_new)
    wm_image = np.clip(np.real(wm_image), 0.0, 1.0)
    if wm_image.shape != host.shape:
        wm_image = np.clip(wm_image[: host.shape[0], : host.shape[1]], 0.0, 1.0)

    state = WatermarkState(
        nlevels=nlevels,
        grid_shape=grid_shape,
        payload_shape=payload_shape,
        payload_len=payload.size,
        alpha=float(alpha),
        henon_inv=inv,
        wm_u=tuple(wm_u),
        wm_vt=tuple(wm_vt),
        host_shape=host.shape[:2],
    )
    return wm_image.astype(np.float64), state


def extract(
    host: np.ndarray,
    watermarked: np.ndarray,
    state: WatermarkState,
) -> np.ndarray:
    """Extract the binary logo using the original host and stored watermark U/V matrices."""
    p0 = dtcwt_forward(host, nlevels=state.nlevels)
    p1 = dtcwt_forward(watermarked, nlevels=state.nlevels)
    low0 = np.asarray(p0.lowpass, dtype=np.float64)
    low1 = np.asarray(p1.lowpass, dtype=np.float64)

    enc_out = np.zeros(state.payload_shape, dtype=np.float64)
    blocks_per_row = state.payload_shape[1] // 8
    block_idx = 0
    for r, c in _block_slices(state.grid_shape):
        br = (block_idx // blocks_per_row) * 8
        bc = (block_idx % blocks_per_row) * 8
        if br >= state.payload_shape[0] or block_idx >= len(state.wm_u):
            break

        d0 = dct2(low0[r : r + 8, c : c + 8])
        d1 = dct2(low1[r : r + 8, c : c + 8])
        _, s0, _ = np.linalg.svd(d0, full_matrices=False)
        _, s1, _ = np.linalg.svd(d1, full_matrices=False)
        sw = np.maximum((s1 - s0) / max(state.alpha, 1e-12), 0.0)
        w_block = state.wm_u[block_idx] @ np.diag(sw) @ state.wm_vt[block_idx]
        enc_out[br : br + 8, bc : bc + 8] = np.clip(w_block, 0.0, 1.0)
        block_idx += 1

    raw = _henon_decrypt_image(enc_out, state.henon_inv, state.payload_shape)
    return _binarize_extracted_watermark(raw)
