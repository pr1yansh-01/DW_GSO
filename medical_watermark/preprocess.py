"""Host and watermark preprocessing for DTCWT (level-1) and 8×8 DCT blocks."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class HostMeta:
    """Geometry after preprocessing (even sizes, divisible by 16 for this pipeline)."""

    shape: tuple[int, int]  # H, W
    nlevels: int


def _to_gray_float01(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    x = img.astype(np.float64)
    if x.max() > 1.5:
        x = x / 255.0
    return np.clip(x, 0.0, 1.0)


def preprocess_host(
    image_bgr_or_gray: np.ndarray,
    max_side: int = 512,
    nlevels: int = 1,
) -> tuple[np.ndarray, HostMeta]:
    """
    Resize/crop so that after ``nlevels`` DTCWT splits, the finest highpass is
    divisible by 8 (8×8 DCT blocks). With ``nlevels == 1``, H and W must be
    multiples of 16.
    """
    x = _to_gray_float01(image_bgr_or_gray)
    h, w = x.shape[:2]
    scale = min(max_side / max(h, w), 1.0)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    nh = max(16, (nh // 16) * 16)
    nw = max(16, (nw // 16) * 16)
    x = cv2.resize(x, (nw, nh), interpolation=cv2.INTER_AREA)
    meta = HostMeta(shape=(nh, nw), nlevels=nlevels)
    return x, meta


def preprocess_watermark_bitmap(
    watermark_bgr_or_gray: np.ndarray,
    grid_shape: tuple[int, int],
    *,
    sharp_binary: bool = True,
) -> np.ndarray:
    """
    Resize watermark to ``grid_shape`` (nrow, ncol) of embedding blocks and
    return a flat binary vector in row-major order, values in {0, 1}.

    Logos are often downscaled by a large factor (e.g. 128² → 16²). With
    ``sharp_binary=True``, area downscaling is followed by Otsu thresholding so
    edges stay black/white instead of gray mush at 0.5 cut-off.
    """
    wm = _to_gray_float01(watermark_bgr_or_gray)
    gh, gw = grid_shape
    small = cv2.resize(wm, (gw, gh), interpolation=cv2.INTER_AREA)
    if sharp_binary:
        u8 = np.clip(small * 255.0, 0, 255).astype(np.uint8)
        if float(u8.std()) < 1e-3:
            bits = (small >= 0.5).astype(np.float64).ravel()
        else:
            thr, _ = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            bits = (u8 >= thr).astype(np.float64).ravel()
    else:
        bits = (small >= 0.5).astype(np.float64).ravel()
    return bits


def watermark_display_preview(
    watermark_bgr_or_gray: np.ndarray,
    max_side: int = 160,
) -> np.ndarray:
    """Resize watermark for on-screen reference only (does not affect embedding)."""
    wm = _to_gray_float01(watermark_bgr_or_gray)
    h, w = wm.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return wm
    s = max_side / m
    nh, nw = max(1, int(round(h * s))), max(1, int(round(w * s)))
    return cv2.resize(wm, (nw, nh), interpolation=cv2.INTER_AREA)


def synthetic_shepp_logan(size: int = 512) -> np.ndarray:
    """Simple elliptical phantom (grayscale, [0,1]) for demos without DICOM."""
    s = size
    yy, xx = np.mgrid[-1 : 1 : s * 1j, -1 : 1 : s * 1j]
    r = np.sqrt(xx**2 + yy**2)
    img = np.zeros((s, s), dtype=np.float64)
    img += 0.35 * (r < 0.72)
    img -= 0.12 * (r < 0.55)
    img += 0.08 * ((xx / 0.35) ** 2 + ((yy - 0.12) / 0.5) ** 2 < 1)
    img += 0.06 * (((xx + 0.22) / 0.2) ** 2 + ((yy + 0.08) / 0.12) ** 2 < 1)
    return np.clip(img, 0.0, 1.0)
