"""Paper-style DTCWT(LL3) -> 8x8 DCT -> SVD watermark embedding."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

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
    alpha_lookup_blocks: np.ndarray
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


@dataclass
class PreparedEmbedding:
    """Reusable host/watermark SVD data for many alpha trials."""

    host_shape: tuple[int, int]
    nlevels: int
    grid_shape: tuple[int, int]
    payload_shape: tuple[int, int]
    payload_len: int
    adaptive_alpha: bool
    blind: bool
    use_aes: bool
    henon_inv: np.ndarray
    aes_nonce: bytes | None
    protected_watermark: np.ndarray
    pyramid: Pyramid
    lowpass: np.ndarray
    alpha_unit_blocks: np.ndarray
    alpha_unit_lookup_blocks: np.ndarray
    block_positions: tuple[tuple[int, int, int, int], ...]
    host_u: tuple[np.ndarray, ...]
    host_s: tuple[np.ndarray, ...]
    host_vt: tuple[np.ndarray, ...]
    wm_u: tuple[np.ndarray, ...]
    wm_s: tuple[np.ndarray, ...]
    wm_vt: tuple[np.ndarray, ...]


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
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-block embedding strengths for 8x8 LL blocks."""
    base = float(alpha)
    gh, gw = grid_shape
    if not adaptive_alpha:
        full = np.full((gh, gw), base, dtype=np.float64)
        return full, full

    alpha_grid_shape = (max(1, (gh + 1) // 2), max(1, (gw + 1) // 2))
    var_map = np.empty(alpha_grid_shape, dtype=np.float64)
    for i in range(alpha_grid_shape[0]):
        for j in range(alpha_grid_shape[1]):
            r, c = i * 16, j * 16
            block = lowpass[r : r + 16, c : c + 16]
            var_map[i, j] = float(np.var(block))

    mean_var = float(np.mean(var_map))
    if mean_var < 1e-12:
        per_block_scale = np.ones_like(var_map)
    else:
        per_block_scale = var_map / mean_var
    alpha_blocks = base * per_block_scale
    alpha_lookup_blocks = np.repeat(np.repeat(alpha_blocks, 2, axis=0), 2, axis=1)[:gh, :gw]
    return alpha_blocks, alpha_lookup_blocks


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


def _remove_small_components(mask: np.ndarray, max_area: int, max_span: int) -> np.ndarray:
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=bool))
    if count == 0:
        return mask
    sizes = np.bincount(labels.ravel())
    remove = np.zeros(count + 1, dtype=bool)
    objects = ndimage.find_objects(labels)
    for label, obj in enumerate(objects, start=1):
        if obj is None or sizes[label] > max_area:
            continue
        height = obj[0].stop - obj[0].start
        width = obj[1].stop - obj[1].start
        if height <= max_span and width <= max_span:
            remove[label] = True
    cleaned = mask.copy()
    cleaned[remove[labels]] = False
    return cleaned


def _despeckle_binary_watermark(image: np.ndarray) -> np.ndarray:
    """Remove tiny salt-and-pepper islands from an extracted binary logo."""
    bits = (np.asarray(image, dtype=np.float64) >= 0.5)
    max_area = max(4, int(round(bits.size * 0.002)))
    max_span = 3

    ones = _remove_small_components(bits, max_area=max_area, max_span=max_span)
    zeros = _remove_small_components(~ones, max_area=max_area, max_span=max_span)
    cleaned = ~zeros
    return cleaned.astype(np.float64)


def _remove_off_center_dark_fragments(image: np.ndarray) -> np.ndarray:
    """Remove dark fragments outside the centered logo area."""
    bits = (np.asarray(image, dtype=np.float64) >= 0.5)
    dark = ~bits
    labels, count = ndimage.label(dark, structure=np.ones((3, 3), dtype=bool))
    if count == 0:
        return bits.astype(np.float64)

    h, w = bits.shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    ry, rx = max(1.0, h * 0.58), max(1.0, w * 0.42)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    main_label = int(np.argmax(sizes))
    main_obj = ndimage.find_objects(labels)[main_label - 1]
    if main_obj is None:
        return bits.astype(np.float64)
    main_x0, main_x1 = main_obj[1].start, main_obj[1].stop
    main_y0, main_y1 = main_obj[0].start, main_obj[0].stop
    main_width = max(1, main_x1 - main_x0)
    main_height = max(1, main_y1 - main_y0)
    min_keep_area = max(10, int(round(bits.size * 0.004)))
    objects = ndimage.find_objects(labels)
    cleaned_dark = dark.copy()

    for label, obj in enumerate(objects, start=1):
        if obj is None or label == main_label:
            continue
        y_mid = (obj[0].start + obj[0].stop - 1) / 2.0
        x_mid = (obj[1].start + obj[1].stop - 1) / 2.0
        inside_logo_area = ((x_mid - cx) / rx) ** 2 + ((y_mid - cy) / ry) ** 2 <= 1.0
        outside_main_band = (
            x_mid < main_x0
            or x_mid > main_x1 + 0.15 * main_width
            or y_mid > main_y1 + 0.15 * main_height
        )
        if outside_main_band or (sizes[label] < min_keep_area and not inside_logo_area):
            cleaned_dark[labels == label] = False

    return (~cleaned_dark).astype(np.float64)


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


def prepare_embedding(
    host: np.ndarray,
    watermark: np.ndarray,
    henon_key: tuple[float, float, float, float],
    nlevels: int = 3,
    adaptive_alpha: bool = False,
    use_aes: bool = False,
    aes_key: bytes | None = None,
    aes_nonce: bytes | None = None,
    blind: bool = False,
) -> PreparedEmbedding:
    """Precompute all embedding pieces that do not depend on alpha."""
    payload = _binary_payload(watermark)
    payload_shape = payload.shape if payload.ndim == 2 else (1, payload.size)
    if payload_shape[0] % 8 or payload_shape[1] % 8:
        raise ValueError("Paper-style block SVD embedding requires watermark dimensions divisible by 8")

    p = dtcwt_forward(host, nlevels=nlevels)
    low = np.asarray(p.lowpass, dtype=np.float64).copy()
    h, w = low.shape[:2]
    grid_shape = (h // 8, w // 8)
    cap = grid_shape[0] * grid_shape[1] * 64
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
    alpha_unit_blocks, alpha_unit_lookup_blocks = _alpha_block_map(low, grid_shape, 1.0, adaptive_alpha)

    block_positions: list[tuple[int, int, int, int]] = []
    host_u: list[np.ndarray] = []
    host_s: list[np.ndarray] = []
    host_vt: list[np.ndarray] = []
    wm_u: list[np.ndarray] = []
    wm_s: list[np.ndarray] = []
    wm_vt: list[np.ndarray] = []
    block_idx = 0
    for r, c in _block_slices(grid_shape):
        br = (block_idx // (payload_shape[1] // 8)) * 8
        bc = (block_idx % (payload_shape[1] // 8)) * 8
        if br >= payload_shape[0]:
            break

        host_block = low[r : r + 8, c : c + 8]
        wm_block = enc_payload[br : br + 8, bc : bc + 8]

        host_dct = dct2(host_block)
        h_u, h_s, h_vt = np.linalg.svd(host_dct, full_matrices=False)
        w_u, w_s, w_vt = np.linalg.svd(wm_block, full_matrices=False)
        block_positions.append((r, c, br, bc))
        host_u.append(h_u)
        host_s.append(h_s.astype(np.float64).copy())
        host_vt.append(h_vt)
        wm_u.append(w_u)
        wm_s.append(w_s.astype(np.float64).copy())
        wm_vt.append(w_vt)
        block_idx += 1

    return PreparedEmbedding(
        host_shape=host.shape[:2],
        nlevels=nlevels,
        grid_shape=grid_shape,
        payload_shape=payload_shape,
        payload_len=payload.size,
        adaptive_alpha=bool(adaptive_alpha),
        blind=bool(blind),
        use_aes=bool(use_aes),
        henon_inv=inv,
        aes_nonce=nonce,
        protected_watermark=enc_payload.copy(),
        pyramid=p,
        lowpass=low,
        alpha_unit_blocks=alpha_unit_blocks,
        alpha_unit_lookup_blocks=alpha_unit_lookup_blocks,
        block_positions=tuple(block_positions),
        host_u=tuple(host_u),
        host_s=tuple(host_s),
        host_vt=tuple(host_vt),
        wm_u=tuple(wm_u),
        wm_s=tuple(wm_s),
        wm_vt=tuple(wm_vt),
    )


def embed_prepared(prepared: PreparedEmbedding, alpha: float) -> tuple[np.ndarray, WatermarkState]:
    """Embed using reusable data from :func:`prepare_embedding`."""
    out_low = prepared.lowpass.copy()
    alpha = float(alpha)
    alpha_blocks = prepared.alpha_unit_blocks * alpha
    alpha_lookup_blocks = prepared.alpha_unit_lookup_blocks * alpha

    for block_idx, (r, c, _br, _bc) in enumerate(prepared.block_positions):
        h_s_mod = prepared.host_s[block_idx].copy()
        w_s = prepared.wm_s[block_idx]
        alpha_block = float(alpha_lookup_blocks[r // 8, c // 8])
        h_s_mod[: w_s.size] = h_s_mod[: w_s.size] + alpha_block * w_s
        out_low[r : r + 8, c : c + 8] = idct2(
            prepared.host_u[block_idx] @ np.diag(h_s_mod) @ prepared.host_vt[block_idx]
        )

    p_new = _set_lowpass(prepared.pyramid, out_low)
    wm_image = dtcwt_inverse(p_new)
    wm_image = np.clip(np.real(wm_image), 0.0, 1.0)
    if wm_image.shape != prepared.host_shape:
        wm_image = np.clip(wm_image[: prepared.host_shape[0], : prepared.host_shape[1]], 0.0, 1.0)

    state = WatermarkState(
        nlevels=prepared.nlevels,
        grid_shape=prepared.grid_shape,
        payload_shape=prepared.payload_shape,
        payload_len=prepared.payload_len,
        alpha=alpha,
        alpha_blocks=alpha_blocks,
        alpha_lookup_blocks=alpha_lookup_blocks,
        adaptive_alpha=prepared.adaptive_alpha,
        blind=prepared.blind,
        use_aes=prepared.use_aes,
        henon_inv=prepared.henon_inv,
        wm_u=prepared.wm_u,
        wm_vt=prepared.wm_vt,
        host_s=prepared.host_s,
        aes_nonce=prepared.aes_nonce,
        protected_watermark=prepared.protected_watermark.copy(),
        host_shape=prepared.host_shape,
    )
    return wm_image.astype(np.float64), state


def embed(
    host: np.ndarray,
    watermark: np.ndarray,
    alpha: float,
    henon_key: tuple[float, float, float, float],
    nlevels: int = 3,
    adaptive_alpha: bool = False,
    use_aes: bool = False,
    aes_key: bytes | None = None,
    aes_nonce: bytes | None = None,
    blind: bool = False,
) -> tuple[np.ndarray, WatermarkState]:
    """
    Embed a binary logo using the paper flow:
    DTCWT level-3 LL subband -> 8x8 DCT blocks -> SVD additive fusion.
    """
    prepared = prepare_embedding(
        host,
        watermark,
        henon_key,
        nlevels=nlevels,
        adaptive_alpha=adaptive_alpha,
        use_aes=use_aes,
        aes_key=aes_key,
        aes_nonce=aes_nonce,
        blind=blind,
    )
    return embed_prepared(prepared, alpha)


def extract(
    host: np.ndarray | None,
    watermarked: np.ndarray,
    state: WatermarkState,
    aes_key: bytes | None = None,
) -> np.ndarray:
    """Extract a watermark using baseline semi-blind or modified blind side information."""
    if not state.blind and host is None:
        raise ValueError("Baseline extraction requires the original host image")

    p1 = dtcwt_forward(watermarked, nlevels=state.nlevels)
    low1 = np.asarray(p1.lowpass, dtype=np.float64)

    enc_out = np.zeros(state.payload_shape, dtype=np.float64)
    blocks_per_row = state.payload_shape[1] // 8
    block_idx = 0
    for r, c in _block_slices(state.grid_shape):
        br = (block_idx // blocks_per_row) * 8
        bc = (block_idx % blocks_per_row) * 8
        if br >= state.payload_shape[0] or block_idx >= len(state.wm_u):
            break

        d1 = dct2(low1[r : r + 8, c : c + 8])
        s0 = state.host_s[block_idx]
        _, s1, _ = np.linalg.svd(d1, full_matrices=False)
        alpha_block = float(state.alpha_lookup_blocks[r // 8, c // 8])
        sw = np.maximum((s1 - s0) / max(alpha_block, 1e-12), 0.0)
        w_block = state.wm_u[block_idx] @ np.diag(sw) @ state.wm_vt[block_idx]
        enc_out[br : br + 8, bc : bc + 8] = np.clip(w_block, 0.0, 1.0)
        block_idx += 1

    raw = _recover_payload(enc_out, state, aes_key)
    extracted = _binarize_extracted_watermark(raw)
    if state.use_aes:
        return _remove_off_center_dark_fragments(_despeckle_binary_watermark(extracted))
    return extracted
