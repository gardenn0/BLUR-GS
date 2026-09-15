# BAD-BLUR-GS: standalone 3DGS port

This is a **BAD-style reimplementation**, not the official BAD-Gaussians code or
a claim of reproducing its published scores. `train_bad.py` uses standard 3DGS
means, SH colors, log scales, quaternions and opacity, with a plain PyTorch loop.
There is no Nerfstudio runtime, CoMoKernel, Neural ODE, learned pixel weighting,
blur mask, or 3D smoothing filter. The separate `train.py` remains CoMo-based.

## Install

Use a separate environment so the existing CoMo experiments remain reproducible:

```bash
conda create -n bad-blur-gs python=3.10 -y
conda activate bad-blur-gs
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements-bad.txt
```

Use a PyTorch CUDA build compatible with your driver and locally installed CUDA
toolkit. `gsplat==1.5.3` builds CUDA kernels on first use and requires a working
C++/CUDA compiler. **Do not install the CoMo rasterizer/simple-knn for this path.**
An environment with those extensions is not sufficient unless gsplat and the
new requirements are also installed.

IAAI can use your existing working `comogaussians` environment via
`--iaai_python /absolute/path/to/comogaussians/bin/python`. This is only the frozen
inference subprocess; reconstruction runs in the new environment. Alternatively,
install the official IAAI package and its requirements in the new environment.
The checkpoint is still the pretrained IAAI checkpoint, not ImageNet backbone weights.

## Baseline and flow experiments

```bash
# BAD-style no-flow baseline (no IAAI package or weights required)
python train_bad.py -s /path/to/blurfactory -m /outputs/bad_baseline/blurfactory \
  --eval -r 1 --llffhold 8 --baseline

# Same reconstruction settings, plus the existing BLUR-GS endpoint flow loss
python train_bad.py -s /path/to/blurfactory -m /outputs/bad_blur_gs/blurfactory \
  --eval -r 1 --llffhold 8 \
  --iaai_checkpoint checkpoints/image_as_imu.pth \
  --iaai_python /absolute/path/to/comogaussians/bin/python \
  --flow_mode como --flow_weight 0.01

# Use a different output directory for lambda=0.001 and trajectory-only flow.
# --flow_mode trajectory blocks the auxiliary depth's geometry gradient only;
# RGB continues to update both geometry and the exposure poses.
```

Here `--flow_mode como` is retained for compatibility with the older experiment
commands: it means **joint flow gradients**, not a CoMo architecture. Only `como`
and `trajectory` are supported. Alternating freeze schedules are deliberately
rejected rather than silently changing BAD's 25-step camera accumulation.

`--baseline` and `--flow_weight 0` both bypass IAAI preparation and auxiliary
rendering. The baseline has exactly the same pose model, optimizers and schedule.
IAAI runs automatically for nonzero flow, before allocating the reconstruction
scene on the GPU; compatible existing caches are reused. Cache observations use
training images only, not held-out views. `--flow_cache` accepts an existing
validated cache. `inspect_blur_flow.py -m /outputs/bad_blur_gs/blurfactory` works.

No implicit data download, new COLMAP reconstruction, GT pose substitution, or
test-view pose optimization takes place. Input is an undistorted COLMAP scene
(`sparse/0` binary or text and images). `images` is preferred; if absent, `images_1`
is used. `-i images_1` selects explicitly. Resolution `-r 1/2/4/8` downscales those
actual source images, not a different dataset version. Other formats and nonzero
lens distortion are rejected. Sorted images at indices 0,8,16,... are held out.

## Pipeline

1. Read the original COLMAP points/intrinsics/poses, split train/test, and record
   content hashes and a single coordinate normalization for exact resume checks.
2. Prepare/reuse frozen IAAI flow. No fine-tuning is performed.
3. Initialize vanilla 3DGS parameters and two (linear) or four (cubic) local pose
   controls per training image.
4. Sample ten poses including exposure endpoints, render each, average RGB.
5. Compute BAD-style L1 + SSIM + scheduled scale regularization.
6. After flow warmup, render endpoint alpha-normalized z-depth with detached
   camera matrices; use explicit pose reprojection for the flow gradient. Masks,
   crop/resize units, direction selection and robust loss reuse `blur_gs`.
7. One backward pass; update Gaussians every step and camera controls after each
   block of 25 accumulated gradients. Perform BAD-style refinement callbacks.

## Explicit schedule and upstream provenance

Reference: [BAD-Gaussians bdd8b3e](https://github.com/WU-CVGL/BAD-Gaussians/tree/bdd8b3e2ba068aa7be8e3ab6fb5277abd40909bc)
and its documented Nerfstudio **v1.0.3**, especially `splatfacto.py` and
`engine/trainer.py`. Settings below are for the default synthetic/linear setup;
upstream real-scene commands sometimes override them.

| Item | This port's default |
|---|---|
| Training length | 30,001 optimizer iterations; zero-based steps **0..30000** |
| Exposure trajectory | Linear translation + SO(3) interpolation; optional cumulative cubic rotation spline |
| Virtual views | 10, uniformly spaced including t=0 and t=1 |
| Pose composition | COLMAP c2w multiplied by local learned SE(3) delta |
| Initial controls | Nonzero se(3) noise, standard deviation 1e-5 |
| Blur integration | Mean of ten clipped RGB renders |
| RGB loss | 0.8 L1 + 0.2 (1-SSIM), pytorch-msssim SSIM |
| Scale regularizer | 0.1 mean(max(max_scale/min_scale - 10, 0)), every ten steps |
| Gaussian update | Every step, including final step |
| Camera update | Steps 24,49,...; **sum**, not mean, of 25 gradients |
| Final partial camera block | Not flushed, matching upstream; pending gradient saved |
| Means LR | 1.6e-4 to 1.6e-6, exponential, 30,000-step scheduler |
| Camera LR | 1e-3 to 1e-5, exponential, 30,000-step scheduler |
| SH/DC/rest LR | 0.0025 / (0.0025/20) |
| Opacity / scale / quaternion LR | 0.05 / 0.005 / 0.001 |
| Adam epsilon | 1e-15 |
| SH degree | 0 initially; increases each 1,000 steps, maximum 3 |
| Refinement | Every 100 steps, after step 500, split/cull before 15,000 |
| Densification statistics | Last RGB virtual view, matching upstream BAD; depth renders excluded |
| Gradient threshold | 4e-4 / virtual-view count; upstream screen-size normalization retained |
| Duplicate/split boundary | Scale 0.01 in normalized scene units; two split samples, scale divided by 1.6 |
| Alpha cull / reset cap | 0.005 / 0.01 |
| Opacity reset | 3,000-step periods, offset +100; after warmup: 3100,6100,9100,12100 |
| Densification pause after reset | Upstream modulo condition: step%3000 > train_count+100 |
| Large-scale cull | Scale >0.5 after first reset period |
| Screen split/cull | Radius/image size >0.05 / >0.15, only before step 4,000 |
| Background | Random per training image, shared across its exposure samples |
| Resolution | Full by default; configurable number of downscales and 250-step schedule |
| Evaluation / save | Every 500 / 2,000 steps, plus final step |
| Added flow schedule | Existing BLUR-GS: after 4000, linear ramp of 2000, weight 0.01 |

Learning-rate calls preserve the upstream scheduler-after-step convention.
The flow schedule is an added BLUR-GS choice, not an official BAD parameter.

## Differences that matter for comparison

- **Renderer:** gsplat **1.5.3**, rather than upstream gsplat 0.x. We preserve
  antialiased EWA splatting and epsilon 0.3, but do not claim numerical or gradient
  equivalence between renderer versions. No CoMo smoothing kernel is used.
- **Coordinates/data:** COLMAP rotations and intrinsics are preserved, including
  off-center principal points. A documented translation/scale normalization uses
  training camera centers (max absolute centered coordinate becomes 0.25).
  Upstream Nerfstudio also auto-orients/centers its data; that exact parser is not
  reproduced here. The same normalization is applied to points and camera centers.
  The inverse is recorded in `bad_config.json`. Never compare to an upstream run
  that uses a different re-rendered synthetic dataset or different COLMAP inputs.
- **Gaussian initialization:** BAD-style mean three-neighbor distance and random
  unit quaternions, computed with SciPy KDTree; no `simple-knn` extension.
- **Evaluation:** held-out input poses stay fixed. PSNR uses unquantized [0,1]
  images; LPIPS uses AlexNet and [-1,1] input. `SSIM` retains the existing CoMo
  skimage [-1,1], data_range=2 convention explicitly. `SSIM_BAD` also reports
  pytorch-msssim [0,1] SSIM. They must not be mixed in one comparison column.
- **Auxiliary depth:** separate [z,1,0] feature render, normalized by alpha from
  that same render, with the same antialiasing. Never use exposure-averaged alpha
  to normalize an endpoint depth. No clamp-to-one is applied to depth features.
- **Scope:** the original PyTorch 3DGS program structure/representation is kept,
  while gsplat supplies a differentiable CUDA rasterizer. This is not the untouched
  original Inria rasterizer or a bit-exact reproduction of official BAD.

Use this port's `--baseline` as the **primary ablation control**. Official BAD and
CoMo scores remain external references. Changes of renderer/parser/trajectory and
schedule together cannot establish that Neural ODE alone caused a difference.

## Resume, render and diagnostics

```bash
# Repeat the original training arguments and append:
# --start_checkpoint /outputs/bad_blur_gs/blurfactory/bad_chkpnt2000.pth

python render_bad.py -m /outputs/bad_blur_gs/blurfactory \
  --checkpoint /outputs/bad_blur_gs/blurfactory/bad_chkpnt30000.pth
```

Checkpoints include Gaussians, both optimizers, controls/base poses, partially
accumulated camera gradients, refinement statistics, camera-sampling stack and
Python/NumPy/CPU/CUDA RNG states. Changing data, calibration, flow observations,
baseline status or training settings on resume is rejected. Load only trusted
checkpoints. CoMo and Gaussian-only checkpoints are not compatible.

Outputs: TensorBoard, per-view and mean `metrics.json`, test renders/GT/errors,
standard Gaussian `point_cloud.ply`, and full `bad_chkpnt*.pth`. `render_bad.py`
also saves training-view midpoint sharp renders, predicted blur, exposure samples
and normalized c2w trajectories. PLY alone cannot resume motion optimization and
does not encode antialiasing metadata; use the supplied renderer for comparisons.

## Validation

Implementation validation on 2026-09-15: **43 CPU tests passed** (four CUDA tests
excluded across the repository). A local CUDA run was attempted with PyTorch
2.6.0+cu126 and an RTX 3060, but Windows Device Guard blocked `cudafe++.exe` while
building gsplat. Consequently, **actual CUDA rendering/training has not yet been
validated**, and no Deblur-NeRF benchmark result is claimed. Run the CUDA tests
on the Linux training server before launching long experiments:

```bash
python -m pytest -q -m 'not cuda'
python -m pytest -q tests/test_bad_cuda.py
```

Then use a separate disposable output directory for a short pipeline check:

```bash
python train_bad.py -s /path/to/blurfactory -m /outputs/bad_smoke/blurfactory \
  --eval -r 4 --baseline --iterations 51 --eval_every 25 --save_every 25
python render_bad.py -m /outputs/bad_smoke/blurfactory \
  --checkpoint /outputs/bad_smoke/blurfactory/bad_chkpnt50.pth --train_limit 1
```

This checks loading, actual rendering, two camera updates, evaluation, checkpoint
and PLY output; it is not a quality experiment. The CUDA unit tests also activate
flow and refinement at shortened schedules using explicitly synthetic fixtures.

CPU tests check analytical interpolation, rigid poses, explicit intrinsics,
gradient routing, no-flow equivalence, refinement/Adam migration, COLMAP holdout,
and exact continuation across partial accumulation. CUDA tests exercise actual
rasterization, pose finite differences, normalized depth gradients and short
joint/trajectory-flow training with refinement. CPU success is not a substitute
for CUDA tests. Neither test suite demonstrates benchmark performance.
