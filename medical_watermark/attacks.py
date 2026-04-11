"""Common integrity attacks on the watermarked image ([0,1] float, grayscale)."""

from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import rotate, shift


def _to_uint8(img: np.ndarray) -> np.ndarray:
    return np.clip(np.round(img * 255.0), 0, 255).astype(np.uint8)


def _from_uint8(img: np.ndarray) -> np.ndarray:
    return img.astype(np.float64) / 255.0


def attack_jpeg(img: np.ndarray, quality: int = 70) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", _to_uint8(img), [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return img.copy()
    dec = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    return _from_uint8(dec)


def attack_gaussian_noise(
    img: np.ndarray,
    sigma: float = 0.02,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    rng = rng or np.random.default_rng()
    noise = rng.standard_normal(img.shape).astype(np.float64) * sigma
    return np.clip(img + noise, 0.0, 1.0)


def attack_rotation(img: np.ndarray, degrees: float, border: float = 0.5) -> np.ndarray:
    """Rotate in-plane; output same shape (edges filled with ``border``)."""
    rot = rotate(img, degrees, reshape=False, order=1, mode="constant", cval=border)
    return np.clip(rot, 0.0, 1.0)


def attack_scaling(img: np.ndarray, factor: float = 0.95) -> np.ndarray:
    h, w = img.shape[:2]
    nh, nw = max(8, int(round(h * factor))), max(8, int(round(w * factor)))
    small = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def attack_translation(img: np.ndarray, dx: int = 3, dy: int = 2) -> np.ndarray:
    return np.clip(shift(img, (dy, dx), order=1, mode="constant", cval=0.5), 0.0, 1.0)


def make_attack_registry(noise_rng: np.random.Generator | None = None) -> dict[str, object]:
    return {
        "jpeg": lambda im: attack_jpeg(im, 70),
        "noise": lambda im: attack_gaussian_noise(im, 0.02, rng=noise_rng or np.random.default_rng(0)),
        "rotation": lambda im: attack_rotation(im, 5.0),
        "scaling": lambda im: attack_scaling(im, 0.95),
        "translation": lambda im: attack_translation(im, 4, 2),
    }


ATTACK_REGISTRY = make_attack_registry(np.random.default_rng(0))


def apply_attack(name: str, img: np.ndarray) -> np.ndarray:
    if name not in ATTACK_REGISTRY:
        raise KeyError(name)
    fn = ATTACK_REGISTRY[name]
    return fn(img)  # type: ignore[operator]
