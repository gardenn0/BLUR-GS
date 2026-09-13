# Validation record

Date: 2026-09-13

## Executed locally

- Windows, Python 3.12.14, PyTorch 2.6.0+cpu in a workspace-local virtualenv.
- `python -m pytest -q`: **25 passed, 1 skipped**.
- `python -m compileall -q blur_gs precompute_blur_flow.py train.py`: passed.
- `python precompute_blur_flow.py --help`: passed.
- `git diff --check`: passed (Git emitted only line-ending conversion notices).

Tests cover analytic translation, rotation/depth independence, depth and pose
finite-difference gradchecks, feature-depth gradients with a nonrigid transform,
synthetic depth optimization, pixel/crop/resize conventions, global sign
selection, empty masks, occlusion rejection, cache validation, phase boundaries,
and gradient clearing/freezing. Integration tests exercise the full supervisor
flow-loss path with a synthetic depth provider and the precompute CLI with a
stub estimator. Automatic startup tests exercise exact loader camera names,
subprocess inference, cache reuse without weights, content-change invalidation,
explicit baseline selection, and inference-failure propagation. The stub is explicit: it does not validate real model inference.

Default como-mode tests check unchanged optimizer/gradient lifecycle across
phase boundaries and exact zero-flow optimizer parity in a small CPU objective.
The synthetic supervisor test verifies nonzero geometry and trajectory gradients
in the first active flow step. These tests do not establish full-scene parity.

## Not yet executed

- The included real CUDA rasterizer finite-difference test was skipped because
  the validation virtualenv is CPU-only. A local NVIDIA RTX 3060 (12 GB) and CUDA
  toolkit are present, but the full CoMo CUDA environment was not built.
- Official Image-as-an-IMU checkpoint inference: weights are external and were
  not downloaded or redistributed for this implementation.
- Full-scene CoMoGaussian baseline or BLUR-GS reconstruction training.
- Peak GPU memory, runtime, held-out PSNR/SSIM/LPIPS or reconstruction improvement.

CI tests target the upstream PyTorch 2.3.0/Python 3.10 CPU versions. A successful
CI run establishes core Python compatibility, not CUDA training correctness.

## GPU acceptance procedure

1. Install the CoMo environment and compile both extensions as documented.
2. Run `python -m pytest -q -m cuda`; missing extensions on a CUDA-enabled host
   cause a failure rather than a silent skip. Investigate a finite-difference
   mismatch before enabling geometry-flow updates.
3. Run official IAAI inference, inspect flow magnitude/direction and source crop.
4. Reproduce CoMoGaussian without `--flow_cache` on one small scene.
5. Run trajectory-only flow supervision and inspect valid coverage/gradients.
6. Enable alternating geometry updates and compare the same evaluation split.

This repository contains an implementation with tested core geometry, not a
claim of empirically validated deblurring performance.
