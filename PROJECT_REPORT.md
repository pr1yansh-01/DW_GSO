# Adaptive-Strength Medical Image Watermarking Using DTCWT-DCT-SVD and Optimizer-Based Parameter Selection

## Abstract

This project implements and extends a medical image watermarking framework based on multi-resolution transform-domain embedding. The baseline pipeline uses a three-level dual-tree complex wavelet transform (DTCWT), block-wise discrete cosine transform (DCT), singular value decomposition (SVD), Henon-map-based watermark permutation, and particle swarm optimization (PSO) to select the embedding strength parameter. The main goal is to preserve medical image quality while maintaining watermark recoverability under common signal processing attacks.

In addition to reproducing the paper-style flow, this project introduces practical modifications aimed at improving experimental robustness and making the method more suitable for a course-level comparative study. These modifications include a robustness-weighted objective function, a Grey Wolf Optimizer (GWO) based modified variant, improved watermark preprocessing and extraction cleanup, and a texture-aware adaptive alpha strategy in which smooth regions receive lower embedding strength while textured regions receive higher strength. The resulting system supports baseline-versus-modified comparisons, visualization of embedding and extraction stages, and attack-based evaluation using PSNR, SSIM, and normalized correlation (NC).

The implementation is organized as a compact research prototype in Python, with separate modules for preprocessing, embedding, extraction, attacks, metrics, fitness evaluation, and optimization. The project demonstrates that meaningful improvements in watermark robustness can be explored without abandoning the original transform-domain structure. The work also highlights the tradeoff between imperceptibility and robustness, especially when adaptive or more aggressive embedding strategies are used.

## 1. Introduction

Medical image watermarking is used to embed identifying or authentication information into diagnostic imagery while preserving visual fidelity. A good watermarking method should satisfy three competing requirements:

- imperceptibility, so the host image remains visually acceptable for diagnosis
- robustness, so the watermark can still be recovered after attacks or distortions
- security, so unauthorized users cannot easily detect, remove, or forge the watermark

Transform-domain watermarking methods are commonly used because they can distribute watermark information in frequency sub-bands that are less sensitive to direct pixel-level tampering. In this project, the medical image watermark is embedded in the lowpass component of a three-level DTCWT representation, then further processed using block DCT and SVD. A binary logo watermark is first encrypted using a Henon chaotic permutation and then fused into the host transform coefficients.

The original paper behind this project focuses on combining DTCWT, DCT, SVD, chaotic encryption, and PSO-based parameter optimization. Our work keeps that core idea but adds a clearer baseline-versus-modified comparison and introduces one meaningful modification that is easy to defend academically: adaptive embedding strength based on local texture variation. We also compare PSO and GWO, improve the fitness design, and clean up extraction postprocessing.

## 2. Baseline Method

The baseline approach implemented in this project follows the same high-level logic as the reference paper:

1. preprocess the host medical image and convert it to grayscale floating-point form
2. preprocess the watermark logo into a binary payload that fits embedding capacity
3. encrypt or permute the watermark using a Henon chaotic map
4. apply a three-level DTCWT to the host image and work on the LL3 lowpass band
5. partition the LL3 band into `8x8` blocks
6. apply 2D DCT to each block
7. use SVD to modify the host singular values using watermark block singular values
8. reconstruct the watermarked image using inverse DCT and inverse DTCWT
9. use PSO to search for an embedding strength `alpha` that balances image quality and recovery quality

This baseline is semi-blind rather than fully blind. During extraction, the original host image is still required in order to estimate the embedded watermark from coefficient differences. While this makes extraction more reliable, it also reduces deployment flexibility in settings where the original image is unavailable.

## 3. Proposed Modifications

This project introduces four practical extensions over the baseline implementation.

### 3.1 Modification 1: Baseline vs Modified Experimental Framing

The project was restructured so the comparison is easier to explain in both the report and the presentation. Instead of presenting the output only as "PSO versus GWO," the system now distinguishes between:

- **Baseline approach**: paper-style embedding with a global alpha parameter selected by PSO
- **Modified approach**: robustness-weighted optimization with GWO and optional adaptive alpha

This framing makes the study more meaningful because the first configuration acts as a reproducible reference point, while the second acts as the improved variant.

### 3.2 Modification 2: Robustness-Weighted Fitness Design

The baseline problem of optimizing a single scalar `alpha` becomes much more useful when evaluated under both image-quality and robustness criteria. Instead of relying only on one output metric, this project uses a weighted objective containing:

- PSNR for fidelity
- SSIM for structural similarity
- mean NC over multiple attacks for watermark robustness

For the modified approach, robustness is emphasized more strongly by increasing the weight assigned to NC under attacks. This makes the optimizer less likely to over-prioritize imperceptibility at the cost of extraction reliability.

### 3.3 Modification 3: Improved Watermark Preprocessing and Extraction Cleanup

The watermark processing was made more practical by:

- preserving watermark aspect ratio during resizing
- constraining watermark dimensions to fit the LL3 block capacity
- binarizing extracted logos adaptively using Otsu thresholding
- removing isolated bit flips after extraction to clean up salt-and-pepper type errors

These changes do not alter the core embedding theory, but they make the implementation more stable and improve the quality of extracted binary logos.

### 3.4 Modification 4: Adaptive Embedding Strength

The most important algorithmic change in this project is adaptive alpha. In the baseline, the same scalar `alpha` is used in every LL3 block. In the modified method, local variance is measured over each `8x8` lowpass block and converted into a gain map:

- smooth blocks receive lower effective alpha
- textured blocks receive higher effective alpha

This is motivated by the observation that distortions are more visible in smooth regions than in complex textured regions. By keeping the average gain centered around the base alpha, the method preserves a meaningful global search parameter while still adapting locally to image content.

Conceptually, the modified embedding strength for block `(i, j)` is:

`alpha_ij = alpha_base * gain_ij`

where `gain_ij` is derived from the normalized local variance of the corresponding LL3 block.

## 4. Implementation Overview

The codebase is organized into small modules, which makes the project easier to explain and extend:

- `run_comparison.py`: main experiment driver for baseline and modified comparisons
- `medical_watermark/pipeline.py`: DTCWT-DCT-SVD embedding and extraction
- `medical_watermark/preprocess.py`: host and watermark preprocessing
- `medical_watermark/henon.py`: chaotic watermark permutation
- `medical_watermark/fitness.py`: multi-metric objective function
- `medical_watermark/attacks.py`: attack simulation
- `medical_watermark/metrics.py`: PSNR, SSIM, and NC
- `medical_watermark/optimizers/pso.py`: baseline PSO optimizer
- `medical_watermark/optimizers/gwo.py`: modified GWO optimizer

This modular organization is one of the strengths of the project because it clearly separates the algorithmic stages and makes ablation-style analysis possible.

## 5. Experimental Setup

### 5.1 Inputs

The experiments use:

- a grayscale medical host image
- a binary or grayscale logo watermark converted into a binary payload

For quick development, the project includes sample files such as `mri.png` and `logoo.png`.

### 5.2 Preprocessing

- host images are converted to grayscale and normalized to `[0, 1]`
- host geometry is resized to dimensions compatible with multi-level DTCWT and `8x8` block DCT
- watermarks are resized to fit the embedding capacity of the LL3 block grid
- watermark output is forced to dimensions divisible by `8`

### 5.3 Attacks

The current implementation evaluates watermark recovery under the following attacks:

- JPEG compression
- Gaussian noise
- rotation
- scaling
- translation

These attacks are used to compute mean NC under distortions.

### 5.4 Metrics

The following evaluation metrics are used:

- **PSNR**: measures the fidelity of the watermarked image relative to the host
- **SSIM**: measures structural preservation
- **NC**: measures similarity between original and extracted watermark

The modified fitness combines these metrics into a single scalar objective.

## 6. Results and Discussion

### 6.1 Qualitative Results

The visualization pipeline shows:

- host image
- original watermark
- binarized watermark payload
- encrypted watermark
- watermarked output
- extracted watermark

These intermediate outputs are useful because they make the algorithm easier to explain during the presentation and also confirm that the watermark survives the embedding and extraction stages.

### 6.2 Quantitative Comparison

The following table can be used to present final results after running the experiments with your preferred settings.

| Method | Optimizer | Adaptive Alpha | PSNR (dB) | SSIM | NC Clean | Mean NC Under Attacks |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Baseline | PSO | No | [fill] | [fill] | [fill] | [fill] |
| Modified | GWO | Yes/No | [fill] | [fill] | [fill] | [fill] |

For a quick development-scale run using reduced image size and fewer optimizer iterations, we observed a pattern like the following:

- baseline retained slightly higher PSNR
- modified adaptive embedding improved some attack recoveries, especially for JPEG/noise/scaling in preliminary runs
- the gains were modest rather than dramatic, which is consistent with realistic transform-domain watermarking tradeoffs

This is an important point for academic honesty: a believable improvement usually comes with a cost. In this case, the modified method can increase robustness while slightly reducing imperceptibility.

### 6.3 Interpretation of the Tradeoff

The baseline approach is more conservative because it uses a global alpha. This often leads to higher PSNR and visually cleaner watermarked images. However, it may under-utilize textured regions where stronger embedding could be tolerated.

The modified adaptive approach is more flexible because it redistributes watermark energy according to local texture. This can improve robustness under some attacks, but because certain blocks receive stronger embedding, PSNR may decrease modestly.

This makes the project’s main conclusion very clear:

- if image fidelity is the top priority, the baseline is attractive
- if attack robustness is more important, the modified approach is stronger

## 7. Advantages of the Proposed Work

- maintains the paper’s transform-domain structure rather than replacing it entirely
- introduces a clear baseline-versus-modified comparison
- uses multiple quality and robustness metrics instead of only one
- supports practical attack simulation
- improves watermark preprocessing and extraction stability
- adds adaptive alpha, which is a defensible and meaningful algorithmic enhancement

## 8. Limitations

Despite the improvements, the current project still has several limitations:

- extraction is semi-blind and requires the original host image
- attack coverage is limited to a moderate set of geometric and intensity distortions
- optimization is computationally expensive because embedding and extraction are repeatedly evaluated inside the optimizer loop
- robustness against geometric attacks such as rotation and translation remains a challenge
- experimental results are still sensitive to optimizer settings, image size, and alpha bounds

These limitations are normal for a student research prototype and provide natural directions for future work.

## 9. Future Work

The project can be improved further in several directions:

- extend attack coverage with blur, cropping, gamma correction, histogram equalization, and contrast adjustment
- add ablation experiments: baseline only, baseline plus improved extraction, baseline plus adaptive alpha, full modified method
- cache repeated transform computations to reduce optimization time
- investigate block-wise or learned alpha selection instead of simple variance-based gains
- explore blind or near-blind extraction strategies
- test the system on a larger set of medical images rather than a single example image

## 10. Conclusion

This project successfully implements a DTCWT-DCT-SVD based medical image watermarking pipeline and extends it with several practical improvements. The work preserves the original transform-domain foundation while making the comparison clearer and the evaluation more meaningful. The most important technical contribution is adaptive embedding strength, which makes the watermarking process sensitive to local image texture rather than relying on a single global embedding parameter.

The overall outcome is a stronger course project because it now contains:

- a working baseline
- a modified approach
- measurable tradeoffs
- implementation-level improvements
- enough structure for a clear report and presentation

Even when the modified method does not dramatically outperform the baseline on every metric, it remains a valuable contribution because it demonstrates a realistic engineering tradeoff between imperceptibility and robustness. That is an appropriate and defensible result for an academic project in digital watermarking.

## 11. Suggested Figures to Insert

You can add the following figures to strengthen the final report:

- system block diagram of the baseline method
- system block diagram of the modified method
- host image, watermark, encrypted watermark, watermarked output, extracted watermark
- comparison of baseline versus modified extracted watermarks
- bar chart of mean NC under attacks
- table of PSNR, SSIM, and NC

## 12. Commands Used for Final Results

Example commands for generating report-ready outputs:

```powershell
python run_comparison.py --host mri.png --wm logoo.png --out-json baseline.json
python run_comparison.py --host mri.png --wm logoo.png --adaptive-alpha --out-json modified.json
python run_comparison.py --host mri.png --wm logoo.png --adaptive-alpha --display
```

For quick tests:

```powershell
python run_comparison.py --host mri.png --wm logoo.png --particles 4 --iters 3 --max-side 256
python run_comparison.py --host mri.png --wm logoo.png --particles 4 --iters 3 --max-side 256 --adaptive-alpha
```

## 13. References

1. Reference paper on medical image watermarking based on DTCWT and PSO optimization.
2. Standard literature on DCT, SVD, and wavelet-domain watermarking.
3. Literature on chaotic encryption and Henon-map-based permutation in image security.
