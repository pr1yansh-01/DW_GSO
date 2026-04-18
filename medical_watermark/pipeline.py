"""Paper-style DTCWT(LL3) -> 8x8 DCT -> SVD watermark embedding."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from medical_watermark.crypto import decrypt_bit_payload_aes, encrypt_bit_payload_aes
from medical_watermark.dtcwt_compat import Pyramid, dtcwt_forward, dtcwt_inverse
from medical_watermark.henon import chaotic_permutation
from medical_watermark.transforms import dct2, idct2


@dataclass
class WatermarkState:
    """Everything needed to extract the watermark for one embedding configuration."""

    nlevels: int
    grid_shape: tuple[int, int]  # nrow, ncol of 8x8 LL3 blocks
    payload_shape: tuple[int, int]
    payload_len: int
    alpha: float
    alpha_blocks: np.ndarray
    adaptive_alpha: bool
    blind: bool
    use_aes: bool
    henon_inv: np.ndarray
    wm_u: tuple[np.ndarray, ...]
    wm_vt: tuple[np.ndarray, ...]
    host_s: tuple[np.ndarray, ...]
    aes_nonce: bytes | None
    protected_watermark: np.ndarray
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


def _alpha_block_map(
    lowpass: np.ndarray,
    grid_shape: tuple[int, int],
    alpha: float,
    adaptive_alpha: bool,
    gain_range: tuple[float, float],
) -> np.ndarray:
    """Return per-block embedding strengths for 8x8 LL blocks."""
    base = float(alpha)
    gh, gw = grid_shape
    if not adaptive_alpha:
        return np.full((gh, gw), base, dtype=np.float64)

    var_map = np.empty((gh, gw), dtype=np.float64)
    for i in range(gh):
        for j in range(gw):
            r, c = i * 8, j * 8
            block = lowpass[r : r + 8, c : c + 8]
            var_map[i, j] = float(np.var(block))

    lo, hi = gain_range
    if hi <= lo:
        raise ValueError(f"Invalid adaptive alpha gain range {gain_range}")

    vmin = float(var_map.min())
    vmax = float(var_map.max())
    if vmax - vmin < 1e-12:
        gains = np.ones_like(var_map)
    else:
        norm = (var_map - vmin) / (vmax - vmin)
        gains = lo + (hi - lo) * norm
        gains /= max(float(np.mean(gains)), 1e-12)
        gains = np.clip(gains, lo, hi)
    return base * gains


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
    """Baseline-friendly fixed-threshold watermark binarization."""
    return (np.asarray(image, dtype=np.float64) >= 0.5).astype(np.float64)


def _protect_payload(
    payload: np.ndarray,
    henon_key: tuple[float, float, float, float],
    *,
    use_aes: bool,
    aes_key: bytes | None,
    aes_nonce: bytes | None,
) -> tuple[np.ndarray, np.ndarray, bytes | None]:
    """Apply modified-path AES then Henon protection, or paper-style Henon only."""
    work = payload
    nonce = aes_nonce
    if use_aes:
        if aes_key is None:
            raise ValueError("Modified AES protection requires an AES key")
        if nonce is None:
            raise ValueError("Modified AES protection requires an AES nonce")
        work = encrypt_bit_payload_aes(work, aes_key, nonce).reshape(payload.shape)
    protected, inv = _henon_encrypt_image(work, henon_key)
    return protected, inv, nonce


def _recover_payload(protected: np.ndarray, state: WatermarkState, aes_key: bytes | None) -> np.ndarray:
    """Reverse Henon scrambling and optional AES encryption."""
    work = _henon_decrypt_image(protected, state.henon_inv, state.payload_shape)
    work_bits = _binarize_extracted_watermark(work)
    if not state.use_aes:
        return work_bits
    if aes_key is None or state.aes_nonce is None:
        raise ValueError("Modified AES extraction requires the AES key and stored nonce")
    dec = decrypt_bit_payload_aes(work_bits, aes_key, state.aes_nonce)
    return dec.reshape(state.payload_shape)


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
    adaptive_alpha: bool = False,
    alpha_gain_range: tuple[float, float] = (0.7, 1.3),
    use_aes: bool = False,
    aes_key: bytes | None = None,
    aes_nonce: bytes | None = None,
    blind: bool = False,
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

    enc_payload, inv, nonce = _protect_payload(
        payload,
        henon_key,
        use_aes=use_aes,
        aes_key=aes_key,
        aes_nonce=aes_nonce,
    )
    p = dtcwt_forward(host, nlevels=nlevels)
    low = np.asarray(p.lowpass, dtype=np.float64).copy()
    out_low = low.copy()
    alpha_blocks = _alpha_block_map(low, grid_shape, alpha, adaptive_alpha, alpha_gain_range)

    wm_u: list[np.ndarray] = []
    wm_vt: list[np.ndarray] = []
    host_s: list[np.ndarray] = []
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
        host_s.append(h_s.astype(np.float64).copy())

        h_s_mod = h_s.astype(np.float64).copy()
        alpha_block = float(alpha_blocks[r // 8, c // 8])
        h_s_mod[: w_s.size] = h_s_mod[: w_s.size] + alpha_block * w_s
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
        alpha_blocks=alpha_blocks,
        adaptive_alpha=bool(adaptive_alpha),
        blind=bool(blind),
        use_aes=bool(use_aes),
        henon_inv=inv,
        wm_u=tuple(wm_u),
        wm_vt=tuple(wm_vt),
        host_s=tuple(host_s),
        aes_nonce=nonce,
        protected_watermark=enc_payload.copy(),
        host_shape=host.shape[:2],
    )
    return wm_image.astype(np.float64), state


def extract(
    host: np.ndarray | None,
    watermarked: np.ndarray,
    state: WatermarkState,
    aes_key: bytes | None = None,
) -> np.ndarray:
    """Extract a watermark using baseline semi-blind or modified blind side information."""
    if not state.blind and host is None:
        raise ValueError("Baseline extraction requires the original host image")

    p0 = dtcwt_forward(host, nlevels=state.nlevels) if host is not None else None
    p1 = dtcwt_forward(watermarked, nlevels=state.nlevels)
    low0 = np.asarray(p0.lowpass, dtype=np.float64) if p0 is not None else None
    low1 = np.asarray(p1.lowpass, dtype=np.float64)

    enc_out = np.zeros(state.payload_shape, dtype=np.float64)
    blocks_per_row = state.payload_shape[1] // 8
    block_idx = 0
    for r, c in _block_slices(state.grid_shape):
        br = (block_idx // blocks_per_row) * 8
        bc = (block_idx % blocks_per_row) * 8
        if br >= state.payload_shape[0] or block_idx >= len(state.wm_u):
            break

        d0 = dct2(low0[r : r + 8, c : c + 8]) if low0 is not None else None
        d1 = dct2(low1[r : r + 8, c : c + 8])
        if d0 is None:
            s0 = state.host_s[block_idx]
        else:
            _, s0, _ = np.linalg.svd(d0, full_matrices=False)
        _, s1, _ = np.linalg.svd(d1, full_matrices=False)
        alpha_block = float(state.alpha_blocks[r // 8, c // 8])
        sw = np.maximum((s1 - s0) / max(alpha_block, 1e-12), 0.0)
        w_block = state.wm_u[block_idx] @ np.diag(sw) @ state.wm_vt[block_idx]
        enc_out[br : br + 8, bc : bc + 8] = np.clip(w_block, 0.0, 1.0)
        block_idx += 1

    raw = _recover_payload(enc_out, state, aes_key)
    return _binarize_extracted_watermark(raw)
