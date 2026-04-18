# Presentation Draft: Medical Image Watermarking Project

## Slide 1: Title Slide

**Adaptive-Strength Medical Image Watermarking Using DTCWT-DCT-SVD**

- Baseline: PSO-optimized global alpha
- Modified: GWO-based robustness optimization with adaptive alpha
- Course: Digital Image Processing / Multimedia Security
- Name(s): [Add names]
- Date: [Add date]

## Slide 2: Problem Statement

- Medical images require authentication and ownership protection
- Watermarking must preserve diagnostic quality
- A good method must balance imperceptibility, robustness, and security
- Transform-domain methods are preferred because they are less fragile than direct spatial embedding

**Talk track:** We are not just hiding data inside an image. We are trying to do it in a way that keeps the medical image visually reliable while still allowing the watermark to survive common distortions.

## Slide 3: Reference Paper and Baseline Idea

- Reference method uses DTCWT, DCT, SVD, Henon encryption, and PSO
- Host image is decomposed using 3-level DTCWT
- Watermark is encrypted and embedded into LL3 `8x8` DCT blocks
- PSO searches for the best embedding strength `alpha`

**Insert:** simple baseline flow diagram

## Slide 4: Baseline Pipeline

- preprocess host image
- preprocess and binarize watermark
- apply Henon permutation to watermark
- apply DTCWT on host
- split LL3 into `8x8` blocks and apply DCT
- modify singular values using watermark block singular values
- reconstruct watermarked image
- extract watermark using original host image

**Insert:** screenshot or diagram of your pipeline

## Slide 5: Limitations of the Baseline

- same alpha used in every block
- may be too weak in textured regions
- may be too strong in smooth regions
- optimizer can over-favor fidelity if robustness is not weighted enough
- semi-blind extraction still depends on the original host image

**Talk track:** The baseline works, but it treats all blocks equally. That is the main weakness we targeted.

## Slide 6: Our Modifications

- clearer baseline vs modified comparison
- robustness-weighted fitness function
- improved watermark preprocessing and extraction cleanup
- adaptive alpha based on local texture variance
- GWO-based modified optimizer path

**Insert:** table with baseline and modified columns

## Slide 7: Key Modification - Adaptive Alpha

- local variance is computed for each LL3 block
- smooth blocks get lower effective alpha
- textured blocks get higher effective alpha
- average alpha remains centered around the optimized base value

Equation idea:

`alpha_ij = alpha_base * gain_ij`

**Why it helps:**

- reduces visible distortion in smooth regions
- uses textured regions more effectively for robustness

## Slide 8: Fitness Function

- objective uses PSNR, SSIM, and mean NC under attacks
- baseline uses standard weighting
- modified approach increases emphasis on robustness

Conceptual form:

`Fitness = w1 * PSNR_norm + w2 * SSIM + w3 * Mean_NC`

**Talk track:** This turns the project into a realistic optimization problem rather than selecting alpha manually.

## Slide 9: Experimental Setup

- host image: medical grayscale image
- watermark: binary logo image
- transforms: DTCWT + block DCT + SVD
- attacks: JPEG, Gaussian noise, rotation, scaling, translation
- metrics: PSNR, SSIM, NC

**Insert:** small table of settings

## Slide 10: Visual Results

- host image
- watermark logo
- encrypted watermark
- watermarked image
- extracted watermark

**Insert:** screenshot from `--display`

Recommended comparison:

- row 1: Baseline
- row 2: Modified

## Slide 11: Quantitative Comparison

Use a table like this:

| Method | Optimizer | Adaptive Alpha | PSNR | SSIM | NC Clean | Mean NC |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Baseline | PSO | No | [fill] | [fill] | [fill] | [fill] |
| Modified | GWO | Yes | [fill] | [fill] | [fill] | [fill] |

**Talk track:** Baseline may preserve slightly higher PSNR, while the modified method can improve robustness in selected attacks.

## Slide 12: Attack Robustness Discussion

- JPEG and scaling are handled relatively well
- noise performance is also reasonable
- rotation and translation remain challenging
- this shows the classic tradeoff in transform-domain watermarking

**Insert:** bar chart of NC by attack for both methods

## Slide 13: Advantages, Limitations, and Future Work

**Advantages**

- strong modular implementation
- defensible algorithmic modification
- multiple evaluation metrics
- baseline vs modified comparison is easy to explain

**Limitations**

- semi-blind extraction
- computationally expensive optimization
- geometric attacks still difficult

**Future Work**

- more attacks such as blur and cropping
- ablation study
- faster optimization
- blind extraction

## Slide 14: Conclusion

- Successfully implemented a DTCWT-DCT-SVD watermarking pipeline
- Built a clear baseline and a modified approach
- Adaptive alpha is the main contribution
- Modified method offers a meaningful robustness-vs-imperceptibility tradeoff
- Project is suitable for report, demo, and presentation

## Slide 15: Demo / Commands

```powershell
python run_comparison.py --host mri.png --wm logoo.png --display
python run_comparison.py --host mri.png --wm logoo.png --adaptive-alpha --display
python run_comparison.py --host mri.png --wm logoo.png --adaptive-alpha --out-json modified.json
```

## Notes for Final Polish

- replace `[fill]` values with your final measured metrics
- add one screenshot of the display window
- add one table comparing baseline and modified results
- keep the story consistent: baseline first, modified second
- do not overclaim improvements if some metrics decrease
