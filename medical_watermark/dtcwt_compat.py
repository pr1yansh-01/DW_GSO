"""NumPy 2 compatibility for the ``dtcwt`` package and thin DTCWT helpers."""

from __future__ import annotations

import numpy as np

from dtcwt.numpy.common import Pyramid
from dtcwt.numpy import Transform2d


def apply_dtcwt_numpy_shims() -> None:
    """Patch removed NumPy 1.x aliases used by ``dtcwt`` (works on NumPy 2.x)."""
    if not hasattr(np, "asfarray"):

        def asfarray(X, dtype=None):
            X = np.asarray(X)
            return np.asarray(X, dtype=dtype if dtype is not None else X.dtype)

        np.asfarray = asfarray  # type: ignore[attr-defined]

    if not hasattr(np, "issubsctype"):

        def issubsctype(arg, T):
            dt = arg.dtype if hasattr(arg, "dtype") else np.dtype(arg)
            return np.issubdtype(dt, T)

        np.issubsctype = issubsctype  # type: ignore[attr-defined]


apply_dtcwt_numpy_shims()

_TRANSFORM = Transform2d()


def dtcwt_forward(image: np.ndarray, nlevels: int = 1) -> Pyramid:
    """2D DTCWT forward; ``image`` is 2D float."""
    return _TRANSFORM.forward(np.asarray(image, dtype=np.float64), nlevels=nlevels)


def dtcwt_inverse(pyramid: Pyramid) -> np.ndarray:
    """2D DTCWT inverse."""
    return _TRANSFORM.inverse(pyramid)
