"""Aggregate objective for optimizer: imperceptibility + robustness."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from medical_watermark.attacks import make_attack_registry
from medical_watermark.metrics import nc, psnr, ssim
from medical_watermark.pipeline import embed, extract


@dataclass
class EvaluationResult:
    alpha: float
    psnr: float
    ssim: float
    nc_clean: float
    nc_by_attack: dict[str, float]
    mean_nc_attacks: float
    fitness: float


def evaluate_alpha(
    host: np.ndarray,
    watermark_bits: np.ndarray,
    alpha: float,
    henon_key: tuple[float, float, float, float],
    attack_names: list[str] | None = None,
    nlevels: int = 3,
    w_psnr: float = 0.25,
    w_ssim: float = 0.25,
    w_nc: float = 0.5,
) -> EvaluationResult:
    """
    Embed with ``alpha``, measure PSNR/SSIM vs host, NC vs original bits without
    attacks, and mean NC after each attack in ``attack_names``.
    """
    noise_rng = np.random.default_rng(42)
    registry = make_attack_registry(noise_rng)
    if attack_names is None:
        attack_names = list(registry.keys())

    wm_img, state = embed(host, watermark_bits, alpha, henon_key, nlevels=nlevels)
    p = psnr(host, wm_img)
    s = ssim(host, wm_img)
    ext_clean = extract(host, wm_img, state)
    n0 = nc(watermark_bits, ext_clean)

    nc_att: dict[str, float] = {}
    for name in attack_names:
        attacked = registry[name](wm_img)  # type: ignore[operator]
        if attacked.shape != host.shape:
            attacked = np.asarray(attacked, dtype=np.float64)
            attacked = attacked[: host.shape[0], : host.shape[1]]
        ext = extract(host, attacked, state)
        nc_att[name] = nc(watermark_bits, ext)

    mean_nc = float(np.mean(list(nc_att.values()))) if nc_att else 0.0
    # Map PSNR to [0,1] assuming 28–48 dB is a useful working range
    psnr_n = float(np.clip((p - 28.0) / 20.0, 0.0, 1.0))
    fit = w_psnr * psnr_n + w_ssim * s + w_nc * mean_nc

    return EvaluationResult(
        alpha=float(alpha),
        psnr=p,
        ssim=s,
        nc_clean=n0,
        nc_by_attack=nc_att,
        mean_nc_attacks=mean_nc,
        fitness=fit,
    )


def make_fitness_fn(
    host: np.ndarray,
    watermark_bits: np.ndarray,
    henon_key: tuple[float, float, float, float],
    attack_names: list[str] | None = None,
    nlevels: int = 3,
    w_psnr: float = 0.25,
    w_ssim: float = 0.25,
    w_nc: float = 0.5,
):
    def f(alpha: float) -> float:
        r = evaluate_alpha(
            host,
            watermark_bits,
            alpha,
            henon_key,
            attack_names=attack_names,
            nlevels=nlevels,
            w_psnr=w_psnr,
            w_ssim=w_ssim,
            w_nc=w_nc,
        )
        return r.fitness

    return f
