#!/usr/bin/env python3
"""
Run the same watermarking + attack + metric pipeline twice:
once optimizing embedding strength with PSO, once with GWO.

Examples:
    python run_comparison.py
    python run_comparison.py --host mri.png --wm logoo.png
    python run_comparison.py --host mri.png --wm logoo.png --display
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from medical_watermark.fitness import evaluate_alpha, make_fitness_fn
from medical_watermark.henon import encrypt_bits
from medical_watermark.optimizers.gwo import gwo_maximize
from medical_watermark.optimizers.pso import pso_maximize
from medical_watermark.pipeline import capacity_from_host, embed, extract
from medical_watermark.preprocess import (
    preprocess_host,
    preprocess_watermark_image,
    synthetic_shepp_logan,
    watermark_display_preview,
)


def _display_pipeline(
    host: np.ndarray,
    watermark: np.ndarray,
    encrypted_watermark: np.ndarray,
    alpha_pso: float,
    alpha_gwo: float,
    henon_key: tuple[float, float, float, float],
    nlevels: int,
    title_prefix: str = "",
    wm_preview: np.ndarray | None = None,
) -> None:
    """Show host, original logo, embedded logo, encrypted logo, watermarked, extracted."""
    import matplotlib.pyplot as plt

    wm_pso, st_pso = embed(host, watermark, alpha_pso, henon_key, nlevels=nlevels)
    ex_pso = extract(host, wm_pso, st_pso)

    wm_gwo, st_gwo = embed(host, watermark, alpha_gwo, henon_key, nlevels=nlevels)
    ex_gwo = extract(host, wm_gwo, st_gwo)

    ncols = 6 if wm_preview is not None else 5
    fig, axes = plt.subplots(2, ncols, figsize=(2.6 * ncols, 6))
    gh, gw = watermark.shape
    fig.suptitle(f"{title_prefix}DTCWT-DCT-SVD binary logo watermark ({gh}x{gw})", fontsize=11)

    titles = ["Host"]
    if wm_preview is not None:
        titles.append("Original logo\n(reference)")
    titles.extend([f"Binary logo\n({gh}x{gw})", "Encrypted\n(Henon)", "Watermarked", "Extracted logo"])

    row_data = [
        (alpha_pso, wm_pso, ex_pso, "PSO"),
        (alpha_gwo, wm_gwo, ex_gwo, "GWO"),
    ]

    for row, (alpha, wmarked, extracted, name) in enumerate(row_data):
        c = 0
        ax = axes[row, c]
        ax.imshow(host, cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        if row == 0:
            ax.set_title(titles[c], fontsize=9)
        c += 1

        if wm_preview is not None:
            ax = axes[row, c]
            ax.imshow(wm_preview, cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(titles[c], fontsize=9)
            c += 1

        ax = axes[row, c]
        ax.imshow(watermark, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        if row == 0:
            ax.set_title(titles[c], fontsize=9)
        c += 1

        ax = axes[row, c]
        ax.imshow(encrypted_watermark, cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        if row == 0:
            ax.set_title(titles[c], fontsize=9)
        c += 1

        ax = axes[row, c]
        ax.imshow(wmarked, cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        if row == 0:
            ax.set_title(titles[c], fontsize=9)
        c += 1

        ax = axes[row, c]
        ax.imshow(extracted, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        if row == 0:
            ax.set_title(titles[c], fontsize=9)

        axes[row, 0].set_ylabel(f"{name}\nalpha={alpha:.4f}", fontsize=9)

    plt.tight_layout()
    plt.show()


def _load_image(path: str | None) -> np.ndarray:
    import cv2

    if not path:
        return synthetic_shepp_logan(512)
    p = Path(path)
    im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if im is None:
        raise FileNotFoundError(path)
    return im


def main() -> None:
    ap = argparse.ArgumentParser(description="PSO vs GWO for DTCWT-DCT-SVD logo watermarking")
    ap.add_argument("--host", type=str, default=None, help="Host image path (default: synthetic phantom)")
    ap.add_argument("--wm", type=str, default=None, help="Watermark/logo image path (default: random logo)")
    ap.add_argument("--max-side", type=int, default=512, help="Preprocessed host max side (larger = more LL3 capacity)")
    ap.add_argument("--wm-side", type=int, default=128, help="Preferred longest side for binary logo payload")
    ap.add_argument("--nlevels", type=int, default=3, help="DTCWT levels; paper uses 3")
    ap.add_argument(
        "--preserve-host-aspect",
        action="store_true",
        help="Keep host aspect ratio instead of paper-style square MxM resize",
    )
    ap.add_argument("--alpha-low", type=float, default=0.01)
    ap.add_argument("--alpha-high", type=float, default=0.35)
    ap.add_argument("--pso-alpha-low", type=float, default=None, help="Optional PSO-only lower alpha bound")
    ap.add_argument("--pso-alpha-high", type=float, default=None, help="Optional PSO-only upper alpha bound")
    ap.add_argument("--gwo-alpha-low", type=float, default=None, help="Optional GWO-only lower alpha bound")
    ap.add_argument("--gwo-alpha-high", type=float, default=0.12, help="Upper alpha bound for modified GWO")
    ap.add_argument("--particles", type=int, default=12, help="PSO particles / GWO wolves")
    ap.add_argument("--iters", type=int, default=12, help="Optimizer iterations")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-json", type=str, default=None, help="Write summary JSON")
    ap.add_argument("--display", action="store_true", help="Show host, logo, encrypted logo, watermarked, extracted")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    host_raw = _load_image(args.host)
    host, meta = preprocess_host(
        host_raw,
        max_side=args.max_side,
        nlevels=args.nlevels,
        square=not args.preserve_host_aspect,
    )
    grid_shape, cap = capacity_from_host(host, nlevels=meta.nlevels)
    max_wm_shape = (grid_shape[0] * 8, grid_shape[1] * 8)

    wm_preview: np.ndarray | None = None
    if args.wm:
        import cv2

        wm_raw = cv2.imread(args.wm, cv2.IMREAD_UNCHANGED)
        if wm_raw is None:
            raise FileNotFoundError(args.wm)
        watermark = preprocess_watermark_image(
            wm_raw,
            cap,
            preferred_side=args.wm_side,
            max_shape=max_wm_shape,
        )
        wm_preview = watermark_display_preview(wm_raw, max_side=160)
    else:
        side = min(args.wm_side, int(np.floor(np.sqrt(cap))))
        side = max(8, (side // 8) * 8)
        watermark = rng.random((side, side), dtype=np.float64)

    henon_key = (0.1 + 0.01 * args.seed, 0.2, 1.4, 0.3)
    pso_bounds = (
        args.alpha_low if args.pso_alpha_low is None else args.pso_alpha_low,
        args.alpha_high if args.pso_alpha_high is None else args.pso_alpha_high,
    )
    gwo_bounds = (
        args.alpha_low if args.gwo_alpha_low is None else args.gwo_alpha_low,
        args.gwo_alpha_high,
    )
    attack_names = ["jpeg", "noise", "rotation", "scaling", "translation"]

    fitness_pso = make_fitness_fn(
        host,
        watermark,
        henon_key,
        attack_names=attack_names,
        nlevels=meta.nlevels,
    )
    fitness_gwo = make_fitness_fn(
        host,
        watermark,
        henon_key,
        attack_names=attack_names,
        nlevels=meta.nlevels,
        w_psnr=0.15,
        w_ssim=0.15,
        w_nc=0.70,
    )

    print("Host shape:", host.shape)
    print("DTCWT levels:", meta.nlevels, "| LL3 block grid:", grid_shape, "| binary payload capacity:", cap, "bits")
    print("Logo payload shape:", watermark.shape)
    print("Optimizing with PSO ...")
    best_pso, fit_pso, hist_pso = pso_maximize(
        fitness_pso,
        pso_bounds,
        n_particles=args.particles,
        n_iter=args.iters,
        seed=args.seed,
    )
    print("Optimizing with GWO ...")
    best_gwo, fit_gwo, hist_gwo = gwo_maximize(
        fitness_gwo,
        gwo_bounds,
        n_wolves=max(3, args.particles),
        n_iter=args.iters,
        seed=args.seed + 7,
    )

    rep_pso = evaluate_alpha(host, watermark, best_pso, henon_key, attack_names=attack_names, nlevels=meta.nlevels)
    rep_gwo = evaluate_alpha(
        host,
        watermark,
        best_gwo,
        henon_key,
        attack_names=attack_names,
        nlevels=meta.nlevels,
        w_psnr=0.15,
        w_ssim=0.15,
        w_nc=0.70,
    )

    def block(title: str, r) -> None:
        print(f"\n=== {title} (alpha={r.alpha:.5f}, fitness={r.fitness:.4f}) ===")
        print(f"  PSNR: {r.psnr:.2f} dB   SSIM: {r.ssim:.4f}   NC (no attack): {r.nc_clean:.4f}")
        print("  NC under attacks:")
        for k, v in r.nc_by_attack.items():
            print(f"    {k:12s} {v:.4f}")
        print(f"  Mean NC (attacks): {r.mean_nc_attacks:.4f}")

    block("PSO", rep_pso)
    block("GWO", rep_gwo)

    summary = {
        "host_shape": list(host.shape),
        "dtcwt_levels": meta.nlevels,
        "ll3_block_grid": list(grid_shape),
        "payload_capacity_bits": int(cap),
        "watermark_shape": list(watermark.shape),
        "pso": {
            "alpha": rep_pso.alpha,
            "fitness": rep_pso.fitness,
            "psnr": rep_pso.psnr,
            "ssim": rep_pso.ssim,
            "nc_clean": rep_pso.nc_clean,
            "mean_nc_attacks": rep_pso.mean_nc_attacks,
            "nc_by_attack": rep_pso.nc_by_attack,
        },
        "gwo": {
            "modified_approach": "GWO with robustness-weighted fitness and bounded alpha search",
            "alpha": rep_gwo.alpha,
            "fitness": rep_gwo.fitness,
            "psnr": rep_gwo.psnr,
            "ssim": rep_gwo.ssim,
            "nc_clean": rep_gwo.nc_clean,
            "mean_nc_attacks": rep_gwo.mean_nc_attacks,
            "nc_by_attack": rep_gwo.nc_by_attack,
        },
    }
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nWrote {args.out_json}")

    if args.display:
        enc, _inv = encrypt_bits(watermark.ravel(), henon_key)
        encrypted_watermark = enc.reshape(watermark.shape)
        _display_pipeline(
            host,
            watermark,
            encrypted_watermark,
            best_pso,
            best_gwo,
            henon_key,
            meta.nlevels,
            wm_preview=wm_preview,
        )


if __name__ == "__main__":
    main()
