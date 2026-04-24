"""Aggregate objective for optimizer: imperceptibility + robustness."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from medical_watermark.attacks import correction_candidates_for_extraction, make_attack_registry
from medical_watermark.metrics import nc, psnr, ssim
from medical_watermark.pipeline import PreparedEmbedding, embed, embed_prepared, extract, prepare_embedding


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
    adaptive_alpha: bool = False,
    use_aes: bool = False,
    aes_key: bytes | None = None,
    aes_nonce: bytes | None = None,
    blind: bool = False,
    prepared: PreparedEmbedding | None = None,
) -> EvaluationResult:
    """
    Embed with ``alpha``, measure PSNR/SSIM vs host, NC vs original bits without
    attacks, and mean NC after each attack in ``attack_names``.
    """
    noise_rng = np.random.default_rng(42)
    registry = make_attack_registry(noise_rng)
    if attack_names is None:
        attack_names = list(registry.keys())

    if prepared is None:
        wm_img, state = embed(
            host,
            watermark_bits,
            alpha,
            henon_key,
            nlevels=nlevels,
            adaptive_alpha=adaptive_alpha,
            use_aes=use_aes,
            aes_key=aes_key,
            aes_nonce=aes_nonce,
            blind=blind,
        )
    else:
        wm_img, state = embed_prepared(prepared, alpha)
    p = psnr(host, wm_img)
    s = ssim(host, wm_img)
    ext_clean = extract(None if blind else host, wm_img, state, aes_key=aes_key)
    n0 = nc(watermark_bits, ext_clean)

    nc_att: dict[str, float] = {}
    for name in attack_names:
        attacked = registry[name](wm_img)  # type: ignore[operator]
        if attacked.shape != host.shape:
            attacked = np.asarray(attacked, dtype=np.float64)
            attacked = attacked[: host.shape[0], : host.shape[1]]
        best_nc = -1.0
        for extraction_input in correction_candidates_for_extraction(name, attacked):
            ext = extract(None if blind else host, extraction_input, state, aes_key=aes_key)
            best_nc = max(best_nc, nc(watermark_bits, ext))
        nc_att[name] = best_nc

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
    adaptive_alpha: bool = False,
    use_aes: bool = False,
    aes_key: bytes | None = None,
    aes_nonce: bytes | None = None,
    blind: bool = False,
    prepared: PreparedEmbedding | None = None,
):
    if prepared is None:
        prepared = prepare_embedding(
            host,
            watermark_bits,
            henon_key,
            nlevels=nlevels,
            adaptive_alpha=adaptive_alpha,
            use_aes=use_aes,
            aes_key=aes_key,
            aes_nonce=aes_nonce,
            blind=blind,
        )

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
            adaptive_alpha=adaptive_alpha,
            use_aes=use_aes,
            aes_key=aes_key,
            aes_nonce=aes_nonce,
            blind=blind,
            prepared=prepared,
        )
        return r.fitness

    return f
