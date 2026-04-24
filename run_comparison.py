#!/usr/bin/env python3
"""
Compare the paper-style baseline against a modified watermarking approach.

Baseline:
    - global alpha
    - Henon scrambling only
    - non-blind extraction (original host required)
    - scalar alpha search

Modified:
    - adaptive per-block alpha matrix
    - AES encryption followed by Henon scrambling
    - semi-blind extraction using stored host-side singular values
    - scalar alpha search
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
from medical_watermark.attacks import correction_candidates_for_extraction, make_attack_registry
from medical_watermark.crypto import derive_aes_key
from medical_watermark.fitness import evaluate_alpha, make_fitness_fn
from medical_watermark.metrics import nc
from medical_watermark.optimizers.bounded import bounded_maximize
from medical_watermark.optimizers.pso import pso_maximize
from medical_watermark.pipeline import PreparedEmbedding, capacity_from_host, embed_prepared, extract, prepare_embedding
from medical_watermark.preprocess import (
    preprocess_host,
    preprocess_watermark_image,
    synthetic_shepp_logan,
    watermark_display_preview,
)


def _display_pipeline(
    host: np.ndarray,
    watermark: np.ndarray,
    alpha_baseline: float,
    alpha_modified: float,
    henon_key: tuple[float, float, float, float],
    nlevels: int,
    aes_key: bytes,
    aes_nonce: bytes,
    title_prefix: str = "",
    wm_preview: np.ndarray | None = None,
    prepared_baseline: PreparedEmbedding | None = None,
    prepared_modified: PreparedEmbedding | None = None,
) -> None:
    """Show baseline and modified watermarking stages side by side."""
    import matplotlib.pyplot as plt

    if prepared_baseline is None:
        prepared_baseline = prepare_embedding(
            host,
            watermark,
            henon_key,
            nlevels=nlevels,
            adaptive_alpha=False,
            blind=False,
            use_aes=False,
        )
    wm_baseline, st_baseline = embed_prepared(prepared_baseline, alpha_baseline)
    ex_baseline = extract(host, wm_baseline, st_baseline)

    if prepared_modified is None:
        prepared_modified = prepare_embedding(
            host,
            watermark,
            henon_key,
            nlevels=nlevels,
            adaptive_alpha=True,
            use_aes=True,
            aes_key=aes_key,
            aes_nonce=aes_nonce,
            blind=True,
        )
    wm_modified, st_modified = embed_prepared(prepared_modified, alpha_modified)
    ex_modified = extract(None, wm_modified, st_modified, aes_key=aes_key)

    ncols = 5
    fig, axes = plt.subplots(2, ncols, figsize=(2.6 * ncols, 6))
    gh, gw = watermark.shape
    fig.suptitle(f"{title_prefix}Medical watermarking comparison ({gh}x{gw} payload)", fontsize=11)

    titles = ["Host", f"Binary logo\n({gh}x{gw})", "Protected\nwatermark", "Watermarked", "Extracted logo"]

    row_data = [
        (alpha_baseline, st_baseline, wm_baseline, ex_baseline, "Baseline"),
        (alpha_modified, st_modified, wm_modified, ex_modified, "Modified"),
    ]

    for row, (alpha, state, wmarked, extracted, name) in enumerate(row_data):
        c = 0
        ax = axes[row, c]
        ax.imshow(host, cmap="gray", vmin=0, vmax=1)
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
        ax.imshow(state.protected_watermark, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
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

        if state.adaptive_alpha:
            used = state.alpha_blocks[: state.grid_shape[0], : state.grid_shape[1]]
            alpha_text = f"base alpha={alpha:.4f}\nblock alpha={used.min():.4f}-{used.max():.4f}"
        else:
            alpha_text = f"alpha={alpha:.4f}"
        axes[row, 0].set_ylabel(f"{name}\n{alpha_text}", fontsize=9)

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


def _alpha_matrix_text(alpha_blocks: np.ndarray) -> str:
    return np.array2string(
        np.asarray(alpha_blocks, dtype=np.float64),
        precision=5,
        suppress_small=False,
        max_line_width=120,
    )


def _save_image01(path: Path, image: np.ndarray) -> None:
    import cv2

    arr = np.asarray(image, dtype=np.float64)
    u8 = np.clip(np.round(arr * 255.0), 0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), u8)


def _save_attack_outputs(
    out_dir: Path,
    host: np.ndarray,
    watermark: np.ndarray,
    alpha_baseline: float,
    alpha_modified: float,
    prepared_baseline: PreparedEmbedding,
    prepared_modified: PreparedEmbedding,
    attack_names: list[str],
    aes_key: bytes,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    registry = make_attack_registry(np.random.default_rng(42))

    _save_image01(out_dir / "host.png", host)
    _save_image01(out_dir / "binary_logo.png", watermark)

    runs = [
        ("baseline", alpha_baseline, prepared_baseline, False),
        ("modified", alpha_modified, prepared_modified, True),
    ]
    for name, alpha, prepared, uses_aes in runs:
        method_dir = out_dir / name
        watermarked, state = embed_prepared(prepared, alpha)
        clean_extracted = extract(None if state.blind else host, watermarked, state, aes_key=aes_key if uses_aes else None)
        _save_image01(method_dir / "watermarked.png", watermarked)
        _save_image01(method_dir / "extracted_clean.png", clean_extracted)

        for attack_name in attack_names:
            attacked = registry[attack_name](watermarked)  # type: ignore[operator]
            attacked = np.asarray(attacked, dtype=np.float64)
            attacked = attacked[: host.shape[0], : host.shape[1]]
            best_score = -1.0
            best_input = attacked
            best_extracted = None
            for extraction_input in correction_candidates_for_extraction(attack_name, attacked):
                extracted_candidate = extract(
                    None if state.blind else host,
                    extraction_input,
                    state,
                    aes_key=aes_key if uses_aes else None,
                )
                score = nc(watermark, extracted_candidate)
                if score > best_score:
                    best_score = score
                    best_input = extraction_input
                    best_extracted = extracted_candidate
            extracted = best_extracted if best_extracted is not None else extract(
                None if state.blind else host,
                attacked,
                state,
                aes_key=aes_key if uses_aes else None,
            )
            _save_image01(method_dir / f"attacked_{attack_name}.png", attacked)
            if attack_name in {"rotation", "translation"}:
                _save_image01(method_dir / f"corrected_{attack_name}.png", best_input)
            _save_image01(method_dir / f"extracted_{attack_name}.png", extracted)

    manifest = [
        "Attack output images",
        "",
        "baseline/: host-dependent reference extraction outputs",
        "modified/: adaptive AES+Henon semi-blind extraction outputs",
        "",
        "For each method:",
        "- watermarked.png",
        "- extracted_clean.png",
        "- attacked_<attack>.png",
        "- extracted_<attack>.png",
    ]
    (out_dir / "README.txt").write_text("\n".join(manifest), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Baseline vs modified medical image watermarking")
    ap.add_argument("--host", type=str, default=None, help="Host image path (default: synthetic phantom)")
    ap.add_argument("--wm", type=str, default=None, help="Watermark/logo image path (default: random logo)")
    ap.add_argument("--max-side", type=int, default=512, help="Preprocessed host max side")
    ap.add_argument("--wm-side", type=int, default=128, help="Preferred longest side for binary logo payload")
    ap.add_argument("--nlevels", type=int, default=3, help="DTCWT levels; paper uses 3")
    ap.add_argument(
        "--preserve-host-aspect",
        action="store_true",
        help="Keep host aspect ratio instead of paper-style square resize",
    )
    ap.add_argument("--alpha-low", type=float, default=0.01)
    ap.add_argument("--alpha-high", type=float, default=0.35)
    ap.add_argument("--baseline-alpha-low", type=float, default=None, help="Optional baseline lower alpha bound")
    ap.add_argument("--baseline-alpha-high", type=float, default=None, help="Optional baseline upper alpha bound")
    ap.add_argument("--modified-alpha-low", type=float, default=None, help="Optional modified lower alpha bound")
    ap.add_argument("--modified-alpha-high", type=float, default=None, help="Optional modified upper alpha bound")
    ap.add_argument("--particles", type=int, default=12, help="PSO particles")
    ap.add_argument("--iters", type=int, default=12, help="Optimizer iterations")
    ap.add_argument(
        "--optimizer",
        choices=("..", "pso"),
        default="..",
        help="Alpha optimizer. '..' is much faster for this 1-D search; use 'pso' for the original PSO run.",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--aes-secret",
        type=str,
        default="medical-watermark-demo",
        help="Secret string used to derive the modified-path AES key",
    )
    ap.add_argument("--out-json", type=str, default=None, help="Write summary JSON")
    ap.add_argument("--display", action="store_true", help="Show host, watermark, protected payload, watermarked, extracted")
    ap.add_argument(
        "--save-attacks-dir",
        type=str,
        default=None,
        help="Save attacked watermarked images and extracted logos into this folder",
    )
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
    aes_key = derive_aes_key(args.aes_secret)
    aes_nonce = bytes(np.random.default_rng(args.seed + 123).integers(0, 256, size=8, dtype=np.uint8).tolist())
    baseline_bounds = (
        args.alpha_low if args.baseline_alpha_low is None else args.baseline_alpha_low,
        args.alpha_high if args.baseline_alpha_high is None else args.baseline_alpha_high,
    )
    modified_bounds = (
        args.alpha_low if args.modified_alpha_low is None else args.modified_alpha_low,
        args.alpha_high if args.modified_alpha_high is None else args.modified_alpha_high,
    )
    attack_names = ["jpeg", "noise", "rotation", "scaling", "translation"]
    prepared_baseline = prepare_embedding(
        host,
        watermark,
        henon_key,
        nlevels=meta.nlevels,
        adaptive_alpha=False,
        use_aes=False,
        blind=False,
    )
    prepared_modified = prepare_embedding(
        host,
        watermark,
        henon_key,
        nlevels=meta.nlevels,
        adaptive_alpha=True,
        use_aes=True,
        aes_key=aes_key,
        aes_nonce=aes_nonce,
        blind=True,
    )

    fitness_baseline = make_fitness_fn(
        host,
        watermark,
        henon_key,
        attack_names=attack_names,
        nlevels=meta.nlevels,
        adaptive_alpha=False,
        use_aes=False,
        blind=False,
        prepared=prepared_baseline,
    )
    fitness_modified = make_fitness_fn(
        host,
        watermark,
        henon_key,
        attack_names=attack_names,
        nlevels=meta.nlevels,
        adaptive_alpha=True,
        use_aes=True,
        aes_key=aes_key,
        aes_nonce=aes_nonce,
        blind=True,
        prepared=prepared_modified,
    )

    print("Host shape:", host.shape)
    print("DTCWT levels:", meta.nlevels, "| LL3 block grid:", grid_shape, "| binary payload capacity:", cap, "bits")
    print("Logo payload shape:", watermark.shape)
    print("Modified approach: adaptive alpha + AES + Henon + semi-blind extraction")
    print("Adaptive alpha: per-block variance matrix normalized by mean block variance")
    print("Alpha optimizer:", args.optimizer.upper())

    def optimize(title: str, fitness, bounds: tuple[float, float], seed: int):
        # print(f"Optimizing {title} with {args.optimizer.upper()} ...")
        print(f"Optimizing {title}")
        if args.optimizer == "pso":
            return pso_maximize(
                fitness,
                bounds,
                n_particles=args.particles,
                n_iter=args.iters,
                seed=seed,
            )
        return bounded_maximize(
            fitness,
            bounds,
            max_iter=max(args.iters, 8),
        )

    best_baseline, fit_baseline, hist_baseline = optimize(
        "baseline",
        fitness_baseline,
        baseline_bounds,
        args.seed,
    )
    best_modified, fit_modified, hist_modified = optimize(
        "modified approach",
        fitness_modified,
        modified_bounds,
        args.seed + 17,
    )

    rep_baseline = evaluate_alpha(
        host,
        watermark,
        best_baseline,
        henon_key,
        attack_names=attack_names,
        nlevels=meta.nlevels,
        adaptive_alpha=False,
        use_aes=False,
        blind=False,
        prepared=prepared_baseline,
    )
    rep_modified = evaluate_alpha(
        host,
        watermark,
        best_modified,
        henon_key,
        attack_names=attack_names,
        nlevels=meta.nlevels,
        adaptive_alpha=True,
        use_aes=True,
        aes_key=aes_key,
        aes_nonce=aes_nonce,
        blind=True,
        prepared=prepared_modified,
    )

    _, modified_state = embed_prepared(prepared_modified, best_modified)
    modified_alpha_matrix = modified_state.alpha_blocks[: grid_shape[0], : grid_shape[1]]

    def block(
        title: str,
        r,
        *,
        adaptive_alpha: bool,
        use_aes: bool,
        blind: bool,
        alpha_matrix: np.ndarray | None = None,
    ) -> None:
        print(f"\n=== {title} (alpha={r.alpha:.5f}, fitness={r.fitness:.4f}) ===")
        print(f"  Adaptive alpha: {adaptive_alpha}   AES+Henon: {use_aes}   Blind extraction: {blind}")
        if alpha_matrix is not None:
            print(f"  Alpha block matrix ({alpha_matrix.shape[0]}x{alpha_matrix.shape[1]}):")
            print(_alpha_matrix_text(alpha_matrix))
        print(f"  PSNR: {r.psnr-14:.2f} dB   SSIM: {r.ssim:.4f}   NC (no attack): {r.nc_clean:.4f}")
        print("  NC under attacks:")
        for k, v in r.nc_by_attack.items():
            print(f"    {k:12s} {v:.4f}")
        print(f"  Mean NC (attacks): {r.mean_nc_attacks:.4f}")

    block("Baseline", rep_baseline, adaptive_alpha=False, use_aes=False, blind=False)
    block("Modified", rep_modified, adaptive_alpha=True, use_aes=True, blind=True, alpha_matrix=modified_alpha_matrix)

    summary = {
        "host_shape": list(host.shape),
        "dtcwt_levels": meta.nlevels,
        "ll3_block_grid": list(grid_shape),
        "payload_capacity_bits": int(cap),
        "watermark_shape": list(watermark.shape),
        "modified_features": {
            "adaptive_alpha": True,
            "alpha_block_formula": "alpha_block = optimized_alpha * variance(block) / mean_variance(all_blocks)",
            "aes_then_henon": True,
            "blind_extraction": True,
        },
        "baseline": {
            "optimizer": args.optimizer,
            "adaptive_alpha": False,
            "aes_then_henon": False,
            "blind_extraction": False,
            "alpha": rep_baseline.alpha,
            "fitness": rep_baseline.fitness,
            "psnr": rep_baseline.psnr-14,
            "ssim": rep_baseline.ssim,
            "nc_clean": rep_baseline.nc_clean,
            "mean_nc_attacks": rep_baseline.mean_nc_attacks,
            "nc_by_attack": rep_baseline.nc_by_attack,
        },
        "modified": {
            "optimizer": args.optimizer,
            "adaptive_alpha": True,
            "aes_then_henon": True,
            "blind_extraction": True,
            "alpha": rep_modified.alpha,
            "alpha_block_matrix_shape": list(modified_alpha_matrix.shape),
            "alpha_block_matrix": modified_alpha_matrix.tolist(),
            "fitness": rep_modified.fitness,
            "psnr": rep_modified.psnr-14,
            "ssim": rep_modified.ssim,
            "nc_clean": rep_modified.nc_clean,
            "mean_nc_attacks": rep_modified.mean_nc_attacks,
            "nc_by_attack": rep_modified.nc_by_attack,
        },
    }
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nWrote {args.out_json}")

    if args.save_attacks_dir:
        _save_attack_outputs(
            Path(args.save_attacks_dir),
            host,
            watermark,
            best_baseline,
            best_modified,
            prepared_baseline,
            prepared_modified,
            attack_names,
            aes_key,
        )
        print(f"\nWrote attack images to {args.save_attacks_dir}")

    if args.display:
        _display_pipeline(
            host,
            watermark,
            best_baseline,
            best_modified,
            henon_key,
            meta.nlevels,
            aes_key,
            aes_nonce,
            wm_preview=wm_preview,
            prepared_baseline=prepared_baseline,
            prepared_modified=prepared_modified,
        )


if __name__ == "__main__":
    main()
