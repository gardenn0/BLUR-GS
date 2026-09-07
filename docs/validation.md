# Validation record

Validation is performed on the local Windows CPU environment with Python 3.12 and
PyTorch 2.6.0+cpu. This file records algorithm/software checks, not published benchmark
reproduction. The official pretrained IAAI checkpoint, real-scene dataset, and a CUDA
training environment were not supplied for this implementation run.

The test suite checks:

- Finite first and second derivatives of the SE(3) exponential at zero rotation.
- Camera-motion Jacobians against finite-difference perspective projection for all six axes.
- Weighted least-squares recovery with unequal fx/fy, arbitrary principal points, masked outliers,
  and invalid depth/flow; translation inverse-depth dependence and rotation depth independence.
- Motion initialization with a nonidentity world-to-camera pose and midpoint anchoring.
- Linear endpoint/interior interpolation, gradients to both endpoint controls, exact zero
  twist acceleration, and nine exposure renders independently of the two learned endpoints.
- Versioned linear checkpoint metadata, legacy midpoint rendering, and explicit rejection
  of incompatible Bezier optimizer state when resuming linear training.
- Image-level temporal reversal and transport of start-referenced flow to midpoint geometry.
- Differentiable zero-confidence loss, nonlinear path length, and crop restoration/vector scaling.
- RGB/expected-depth rendering with nonzero finite camera and scene gradients.
- Static-camera reblurring equivalence and optional CUDA pose/depth gradient checks.
- Exact freezing of the inactive parameter group, motion gradients to both groups,
  checkpoint resume continuity, depth warm-up, and decreasing RGB loss on a synthetic scene.
- COLMAP text/binary camera import and rejection of unsupported distorted calibration.

The CUDA test is skipped unless both CUDA and gsplat are available. Pure PyTorch rendering
uses full Gaussian support whereas CUDA uses tile culling and alpha thresholds; exact
pixel-for-pixel backend equality is not asserted. The CUDA adapter's half-pixel convention
is explicitly converted, but numerical GPU execution still needs verification on a GPU.

Reproduce checks with:

```bash
python -m pytest -q
python -m ruff check .
python -m blur_gs synthetic --output data/smoke --size 32 --views 3
python -m blur_gs train --data data/smoke/scene.json --config configs/smoke.yaml --output outputs/smoke
python -m blur_gs evaluate --data data/smoke/scene.json --checkpoint outputs/smoke/checkpoint.pt --output outputs/smoke-eval
```

Use new output directories when rerunning fixture generation or starting a new training
run. Local generated artifacts are excluded from Git.

## Linear implementation run (2026-09-07)

- **27 passed, 1 skipped** (CUDA-only check), 13.49 seconds. The initial attempt encountered
  access-denied errors in the pre-existing Windows pytest temporary/cache directories.
  The complete suite passed with a fresh temporary directory and the cache plugin disabled.
- Ruff lint and formatting checks passed.
- Linear interpolation, endpoint gradients, zero twist acceleration, both alternating phases,
  joint training, loss decrease, checkpoint saving/resume, and rendering are covered.
- Verified exactly nine exposure renders plus one reference-depth pass with the default
  configuration, independently of the two learned endpoint controls. The smoke config
  continues to use five exposure samples.
- Format-version-2 linear checkpoints are identified explicitly; legacy Bezier midpoint
  rendering is tested and legacy training resume is rejected with an actionable message.
- The synthetic fixture retains its mildly curved ground-truth motion analytically; it
  was not made linear merely to match the new learner. No new real-data/GPU benchmark was run.

## Initial Bezier implementation run (2026-09-07, before the linear change)

- `pytest -q`: **20 passed, 1 skipped** (CUDA-only check), 31.17 seconds.
- `ruff check .`: passed. `ruff format --check .`: all 25 Python files formatted.
- CLI fixture: 3 views, 32x32 RGB, 25 anisotropic Gaussians, analytic midpoint flow.
- CLI training: 24 iterations, both alternating phases and final joint phase completed,
  with finite losses and no motion-initialization fallback.
- Saved checkpoints at iterations 12 and 24, final checkpoint, standard DC-SH PLY,
  three sharp renders, per-step JSONL metrics, and a summary.
- CLI evaluation against synthetic sharp **training-view** references completed:
  mean PSNR 13.4227 dB, mean SSIM 0.3766. These low-iteration smoke metrics test the
  output/evaluation plumbing; they are not held-out novel-view or real-image results.
- A separate deterministic single-view regression test verifies decreasing RGB loss,
  and another checks exact resume continuity with Adam optimizer state.

Real-data IAAI checkpoint inference and CUDA execution remain unverified. They require
the official weights, prepared images/COLMAP geometry, and a suitable GPU environment.
