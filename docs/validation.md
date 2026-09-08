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
  twist acceleration, and ten exposure renders independently of the two learned endpoints.
- Versioned linear checkpoint metadata, legacy midpoint rendering, and explicit rejection
  of incompatible Bezier optimizer state when resuming linear training.
- Image-level temporal reversal and transport of start-referenced flow to midpoint geometry.
- Differentiable zero-confidence loss, nonlinear path length, and crop restoration/vector scaling.
- RGB/expected-depth rendering with nonzero finite camera and scene gradients.
- Static-camera reblurring equivalence and optional CUDA pose/depth gradient checks.
- Exact freezing of the inactive parameter group, motion gradients to both groups,
  checkpoint resume continuity, depth warm-up, and decreasing RGB loss on a synthetic scene.
- Gaussian split/prune topology changes and checkpoint resume after a changed Gaussian count.
- Rotation-equivariant anisotropic splits, accumulated multi-view refinement scores, retained
  Adam/AMSGrad state, and exact CPU continuation through another split for all trajectory types.
- Real-scene preflight checks for motion cache coverage and runtime/backend compatibility.
- COLMAP text/binary camera import and rejection of unsupported distorted calibration.

The CUDA renderer test is skipped unless both CUDA and gsplat are available; CUDA RNG
restoration has a separate test that requires CUDA only. Pure PyTorch rendering
uses full Gaussian support whereas CUDA uses tile culling and alpha thresholds; exact
pixel-for-pixel backend equality is not asserted. The CUDA adapter's half-pixel convention
is explicitly converted, but numerical GPU execution still needs verification on a GPU.

Reproduce checks with:

```bash
python -m pytest -q
python -m ruff check .
python scripts/make_synthetic.py --output data/smoke --size 32 --views 3
python train.py --data data/smoke/scene.json --config configs/smoke.yaml --output outputs/smoke
python metrics.py --data data/smoke/scene.json --checkpoint outputs/smoke/checkpoint.pt --output outputs/smoke-eval
```

Use new output directories when rerunning fixture generation or starting a new training
run. Local generated artifacts are excluded from Git.

## Refinement correctness fixes (2026-09-09)

- **58 passed, 2 skipped**, 79.80 seconds, with a fresh Windows temporary directory and
  pytest's cache plugin disabled. Skips cover CUDA rendering and CUDA RNG restoration.
- Ruff lint, formatting of changed Python files, and Git whitespace checks passed.
- A rotated anisotropic parent's split follows its covariance axes; both children remain
  on the expected parent ellipsoid instead of moving along unrotated world axes.
- With the default schedule and 25 views, refinement uses observations from all 25 views
  even though every refinement boundary falls on view 24 (zero-based).
- Pruning/splitting preserves unchanged Gaussian Adam/AMSGrad moments and group options;
  split children have zero moments. All-pruned fallback respects the Gaussian capacity.
- CPU checkpoint continuation through another split exactly matches uninterrupted state
  for linear, spline, and ODE trajectories, including pending statistics, RNG and Adam state.
- Legacy checkpoints still load and issue a continuity warning when missing saved state
  would affect future refinement. Training, rendering and evaluation CLI tests also pass.
- These are software regressions, not real-scene performance or CUDA validation. The
  CoMoGaussian latent ODE/CMR/pixel-weighting architecture remains outside this implementation.

## Research-layout refactor (2026-09-07)

- **37 passed, 1 skipped** (CUDA-only check), 85.23 seconds. Used a fresh Windows
  temporary directory and disabled the pytest cache plugin to avoid pre-existing ACL issues.
- Ruff lint and formatting checks passed.
- Direct `train.py`, `render.py`, `metrics.py`, and the three preparation scripts expose
  working help even when invoked by absolute path from another working directory.
- A subprocess integration test generates a synthetic scene, trains through `train.py`,
  renders through `render.py`, evaluates through `metrics.py`, and resumes through the
  legacy `python -m blur_gs train` command.
- Prepared-directory and explicit-manifest input, `-s`/`-m` and legacy option aliases,
  default nine exposure samples, and configuration overrides are covered by tests.
- Compared 52 function/class ASTs against the pre-refactor Git HEAD: all were unchanged.
  The deliberate exceptions are CLI-aware config overrides and an updated preparation
  script path in a missing-cache error message. Imports and entry-point code were relocated.
- Built a wheel without installing it into the existing environment. In an isolated
  Python process, all 18 selected entry-point and implementation modules loaded directly
  from that wheel, not from the checkout.
- Dataset/cache schemas, checkpoint version 2 and optimization equations remain unchanged.
  No densification/pruning, high-order SH, external CUDA source, or new GPU validation
  was added as part of this structural refactor.

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
