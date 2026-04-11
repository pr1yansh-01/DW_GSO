#!/usr/bin/env python3
"""
Run the same watermarking + attack + metric pipeline twice:
once optimizing embedding strength with PSO, once with GWO.

Usage (from ``Priyansh`` directory)::

    python run_comparison.py
    python run_comparison.py --host path/to/image.png --wm path/to/wm.png
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
from medical_watermark.optimizers.gwo import gwo_maximize
from medical_watermark.optimizers.pso import pso_maximize
from medical_watermark.henon import encrypt_bits
from medical_watermark.pipeline import capacity_from_host, embed, extract
from medical_watermark.preprocess import (
    preprocess_host,
    preprocess_watermark_bitmap,
    synthetic_shepp_logan,
    watermark_display_preview,
)


def _bits_to_grid(bits: np.ndarray, grid_shape: tuple[int, int]) -> np.ndarray:
    """Row-major 2D bitmap in [0,1] for display."""
    g = np.asarray(bits, dtype=np.float64).ravel()[: grid_shape[0] * grid_shape[1]]
    return g.reshape(grid_shape)


def _display_pipeline(
    host: np.ndarray,
    wm_bits: np.ndarray,
    enc_bits: np.ndarray,
    grid_shape: tuple[int, int],
    alpha_pso: float,
    alpha_gwo: float,
    henon_key: tuple[float, float, float, float],
    title_prefix: str = "",
    wm_preview: np.ndarray | None = None,
) -> None:
    """Show host, optional original-logo preview, payload grid, encrypted, watermarked, extracted."""
    import matplotlib.pyplot as plt

    wm_grid = _bits_to_grid(wm_bits, grid_shape)
    enc_grid = _bits_to_grid(enc_bits, grid_shape)

    wm_pso, st_pso = embed(host, wm_bits, alpha_pso, henon_key)
    ex_pso = extract(host, wm_pso, st_pso)
    ex_pso_g = _bits_to_grid(ex_pso, grid_shape)
    ex_pso_bin = (ex_pso_g >= 0.5).astype(np.float64)

    wm_gwo, st_gwo = embed(host, wm_bits, alpha_gwo, henon_key)
    ex_gwo = extract(host, wm_gwo, st_gwo)
    ex_gwo_g = _bits_to_grid(ex_gwo, grid_shape)
    ex_gwo_bin = (ex_gwo_g >= 0.5).astype(np.float64)

    ncols = 6 if wm_preview is not None else 5
    fig, axes = plt.subplots(2, ncols, figsize=(2.6 * ncols, 6))
    gh, gw = grid_shape
    fig.suptitle(
        f"{title_prefix}Pipeline — payload is only {gh}×{gw} bits (not full logo resolution)",
        fontsize=11,
    )

    def col_titles_row0() -> list[str]:
        t = ["Host"]
        if wm_preview is not None:
            t.append("Original logo\n(reference)")
        t.append(f"Embedded payload\n({gh}×{gw} bits)")
        t.extend(["Encrypted\n(permuted)", "Watermarked", "Extracted"])
        return t

    titles = col_titles_row0()

    row_data = [
        (alpha_pso, wm_pso, ex_pso_bin, "PSO"),
        (alpha_gwo, wm_gwo, ex_gwo_bin, "GWO"),
    ]

    for row, (alpha, wmarked, ext_bin, name) in enumerate(row_data):
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
            ax.imshow(wm_preview, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(titles[c], fontsize=9)
            c += 1

        ax = axes[row, c]
        ax.imshow(wm_grid, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        if row == 0:
            ax.set_title(titles[c], fontsize=9)
        c += 1

        ax = axes[row, c]
        ax.imshow(enc_grid, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
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
        ax.imshow(ext_bin, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        if row == 0:
            ax.set_title(titles[c], fontsize=9)

        axes[row, 0].set_ylabel(f"{name}\nα={alpha:.4f}", fontsize=9)

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
    ap = argparse.ArgumentParser(description="PSO vs GWO for DTCWT-DCT-SVD watermarking")
    ap.add_argument("--host", type=str, default=None, help="Host image path (default: synthetic phantom)")
    ap.add_argument("--wm", type=str, default=None, help="Watermark image path (default: random pattern)")
    ap.add_argument("--max-side", type=int, default=256, help="Preprocessed square-ish max side (smaller = faster)")
    ap.add_argument("--alpha-low", type=float, default=0.01)
    ap.add_argument("--alpha-high", type=float, default=0.35)
    ap.add_argument("--particles", type=int, default=12, help="PSO particles / GWO wolves")
    ap.add_argument("--iters", type=int, default=12, help="Optimizer iterations (fitness calls scale with this)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-json", type=str, default=None, help="Write summary JSON")
    ap.add_argument(
        "--display",
        action="store_true",
        help="Show figures: host, optional original-logo preview, tiny payload grid, encrypted, watermarked, extracted",
    )
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    host_raw = _load_image(args.host)
    host, _meta = preprocess_host(host_raw, max_side=args.max_side, nlevels=1)
    grid_shape, cap = capacity_from_host(host, nlevels=_meta.nlevels)

    wm_preview: np.ndarray | None = None
    if args.wm:
        import cv2

        wm_raw = cv2.imread(args.wm, cv2.IMREAD_UNCHANGED)
        if wm_raw is None:
            raise FileNotFoundError(args.wm)
        wm_bits = preprocess_watermark_bitmap(wm_raw, grid_shape)
        wm_preview = watermark_display_preview(wm_raw, max_side=160)
    else:
        wm_bits = rng.integers(0, 2, size=cap, dtype=np.int64).astype(np.float64)

    if wm_bits.size != cap:
        raise SystemExit(f"Watermark capacity mismatch: got {wm_bits.size}, need {cap}")

    henon_key = (0.1 + 0.01 * args.seed, 0.2, 1.4, 0.3)
    bounds = (args.alpha_low, args.alpha_high)
    attack_names = ["jpeg", "noise", "rotation", "scaling", "translation"]

    fitness = make_fitness_fn(
        host,
        wm_bits,
        henon_key,
        attack_names=attack_names,
    )

    print("Host shape:", host.shape, "| Capacity (bits):", cap)
    print("Optimizing with PSO ...")
    best_pso, fit_pso, hist_pso = pso_maximize(
        fitness,
        bounds,
        n_particles=args.particles,
        n_iter=args.iters,
        seed=args.seed,
    )
    print("Optimizing with GWO ...")
    best_gwo, fit_gwo, hist_gwo = gwo_maximize(
        fitness,
        bounds,
        n_wolves=max(3, args.particles),
        n_iter=args.iters,
        seed=args.seed + 7,
    )

    rep_pso = evaluate_alpha(host, wm_bits, best_pso, henon_key, attack_names=attack_names)
    rep_gwo = evaluate_alpha(host, wm_bits, best_gwo, henon_key, attack_names=attack_names)

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
        "capacity": int(cap),
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
        gh, gw = grid_shape
        print(
            f"\nDisplay: embedded payload is {gh}×{gw} = {cap} bits "
            f"(logo is downsampled to this grid — not full file resolution)."
        )
        enc_bits, _inv = encrypt_bits(wm_bits, henon_key)
        _display_pipeline(
            host,
            wm_bits,
            enc_bits,
            grid_shape,
            best_pso,
            best_gwo,
            henon_key,
            wm_preview=wm_preview,
        )


if __name__ == "__main__":
    main()
