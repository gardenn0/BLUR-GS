# BLUR-GS

**Geometry-coupled blur-flow supervision on top of CoMoGaussian.**

Experimental implementation based on discussions of BLUR-GS/BlurTraj-GS and
*Image as an IMU*. This is research code, not a validated reproduction of a
published BLUR-GS result. No quality improvement over CoMoGaussian is claimed.

CoMoGaussian remains the backbone: its Neural ODE/CMR trajectory, Gaussian scene,
RGB renderer, blur weights/mask, and original losses are preserved. We add a
frozen observed-flow cache and a differentiable geometry-motion consistency loss.

```text
blurry images -> frozen Image-as-an-IMU -> cached endpoint flow
                                             |
CoMoKernel -> exposure cameras                |
     |                  |                    |
     |       Gaussians -> auxiliary z-depth   |
     |                  |                    |
     +--------> backproject / reproject -> flow loss
     |                                       |
     +-> original RGB renders -> blur loss   |
                                 |           |
                         optimize scene / trajectory
```

## What is implemented

- `train.py` automatically prepares and reuses frozen Image-as-an-IMU flow.
- BLUR-GS is the default; `--baseline` explicitly selects original CoMoGaussian.
- Crop/resize-aware flow caching, safe NPZ loading, missing-camera validation.
- Normalized z-depth via an auxiliary `[z, 1, 0]` feature render; no raw CUDA
  distance-depth supervision and no CUDA source changes.
- Independently computed forward/backward endpoint flow, global direction
  selection, validity/occlusion masks, robust flow loss and coverage logging.
- Trajectory-prior warm-up, alternating geometry/trajectory updates and a joint
  ablation with explicit gradient routing.
- Full BLUR-GS checkpoints including the CoMo model/optimizer and RNG state.
- CPU geometry/gradient tests plus an optional actual CUDA renderer gradient test.

Implementation details and limitations: [docs/IMPLEMENTATION.md](docs/IMPLEMENTATION.md).
Original backbone instructions: [README_CoMoGaussian.md](README_CoMoGaussian.md).
Attributions and component licenses: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## 1. CoMoGaussian environment

Use a CUDA development environment with a compatible C++ compiler. The upstream
reference setup uses Python 3.10, PyTorch 2.3.0 and CUDA 12.1. Linux is the primary
upstream workflow; Windows extension compilation is not validated here.

```bash
git clone https://github.com/gardenn0/BLUR-GS.git
cd BLUR-GS
conda create -n blur-gs python=3.10 -y
conda activate blur-gs
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install matplotlib pytest
pip install --no-build-isolation -e ./submodules/diff-gaussian-rasterization-pose-backprop/
pip install --no-build-isolation -e ./submodules/simple-knn/
```

Use the same source data and SfM camera preparation as CoMoGaussian. Keep datasets,
weights and generated caches outside Git. The inherited dependency pins are
preserved; use the Python version specified above for those older packages.

## 2. Install the flow estimator once

Install the official [Image-as-an-IMU](https://github.com/jerredchen/image-as-an-imu)
and obtain its official checkpoint. These are one-time dependencies, just like
installing the rasterizer; model weights are not distributed in this repository.
The existing CoMoGaussian environment can continue to run training.

A separate estimator environment avoids changing CoMo's PyTorch dependencies:

```bash
conda create -n image-as-imu python=3.10 -y
conda activate image-as-imu
git clone https://github.com/jerredchen/image-as-an-imu.git
cd image-as-an-imu
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install -e .
```

In your training shell, configure its Python executable and checkpoint once
(adjust these paths for your installation; they can be saved in a shell profile):

```bash
export BLUR_GS_IAAI_PYTHON=/home/user/miniconda3/envs/image-as-imu/bin/python
export BLUR_GS_IAAI_CHECKPOINT=/models/image_as_imu.pth
conda activate blur-gs  # or your existing CoMoGaussian environment
```

`BLUR_GS_IAAI_ROOT` is optional when the package was installed with `pip install -e`.
Alternatively supply `--iaai_python`, `--iaai_checkpoint`, and `--iaai_root` to
`train.py`. Without overrides, inference uses the current Python and looks for
`checkpoints/image_as_imu.pth` relative to the working directory. Using the same
environment requires that the official estimator dependencies are compatible;
this combination has not been validated.

## 3. Train with one command

```bash
python train.py -s /data/scene -m output/blur_gs --eval -r 1
```

BLUR-GS is now the default. `train.py` takes the exact training images selected
by CoMo's data loader, generates their flow automatically if needed, releases
the estimator process, and starts reconstruction training. No manual precompute
command is needed. Use `-r 4` for CoMo's real Deblur-NeRF data as upstream directs.

Automatic caches live under `/data/scene/flow_cache/<image-content-signature>`.
Changed image contents or training splits select a new cache. Existing caches
are checked for missing/corrupt observations; a supplied checkpoint must match
the cache checkpoint hash. A different checkpoint requires a fresh
`--flow_cache /path/to/new/cache`. Cached runs need no estimator environment or
checkpoint. An explicit existing `--flow_cache` is treated as a user-supplied
observation set; ensure it corresponds to the current images.
Missing dependencies or failed inference stop training with an error; they do
not silently disable BLUR-GS. The dataset must be writable for the default
cache; use `--flow_cache` to choose another location if needed.

### Original CoMoGaussian baseline

```bash
python train.py -s /data/scene -m output/baseline --eval -r 1 --baseline
```

Migration: previously omitting `--flow_cache` selected the baseline. Add
`--baseline` to old baseline commands. It cannot be combined with `--flow_cache`.
The standalone `precompute_blur_flow.py` remains available for optional offline
preparation, but is no longer required in the normal training workflow.

### BLUR-GS: trajectory prior followed by alternating refinement

```bash
python train.py -s /data/scene -m output/blur_gs --eval -r 1 \
  --flow_cache /data/scene/flow_cache \
  --flow_weight 0.01 --geometry_flow_weight 0.01 \
  --flow_start 4000 --geometry_start 20000 --flow_ramp 2000 \
  --flow_mode alternating --alternate_every 50
```

These are starting hyperparameters, not validated optimal settings. The flow
penalty is measured in the estimator's 320x224 pixel units. Training logs report
raw loss, selected direction, valid fraction, skipped loss and effective weight.
If most flow steps are skipped, inspect depth/alpha, coordinate alignment and
domain mismatch; a zero loss under no valid pixels is not evidence of success.

### Ablations

```bash
# RGB updates geometry; flow only supervises the trajectory throughout.
python train.py -s /data/scene -m output/trajectory_prior --eval -r 1 \
  --flow_cache /data/scene/flow_cache --flow_mode trajectory

# Joint updates after geometry_start (auxiliary depth camera gradients blocked).
python train.py -s /data/scene -m output/joint --eval -r 1 \
  --flow_cache /data/scene/flow_cache --flow_mode joint
```

Additional controls:

| Argument | Default | Meaning |
|---|---:|---|
| `--flow_direction` | auto | auto / forward / backward; auto selects globally |
| `--flow_min_alpha` | 0.5 | Minimum source and target accumulated opacity |
| `--flow_occlusion_tolerance` | 0.1 | Relative depth agreement; <=0 disables this check |
| `--flow_min_valid_fraction` | 0.05 | Skip flow loss below this valid coverage |

## 4. Resume and evaluate

Saving iterations also produce `blur_chkpntITER.pth`. Resume with the **same
training flags and iteration budget**, adding:

```bash
--start_checkpoint output/blur_gs/blur_chkpnt20000.pth
```

Only load trusted checkpoints. Original Gaussian-only CoMo checkpoints cannot
resume BLUR-GS, because they omit its CoMo trajectory model. Sharp novel views
need no Image-as-an-IMU or flow cache:

```bash
python render.py -m output/blur_gs
python metrics.py -m output/blur_gs
```

Compare identical splits, resolutions and evaluation pose-refinement settings.
Check RGB metrics, flow coverage, trajectory visualizations and depth quality;
flow consistency alone does not establish better sharp reconstruction.

## Tests and current validation

The pure-PyTorch geometry tests do not require CoMo's compiled extensions:

```bash
pip install torch numpy pillow pytest
python -m pytest -q -m "not cuda"
```

On a fully installed CUDA environment, run the actual rasterizer check:

```bash
python -m pytest -q -m cuda
```

CPU tests verify translation/rotation projection, depth/pose autograd finite
differences, depth recovery in a synthetic optimization, z-feature gradients,
crop/resize alignment, direction selection, masks/cache validation, and phase
freezing. The actual CUDA test compares an auxiliary depth derivative with
finite differences. See [docs/VALIDATION.md](docs/VALIDATION.md) for results and
the exact unverified portions. Full-scene GPU training is not yet validated.

## Scope

IAAI predicts endpoint displacement, not a sampled nonlinear exposure path.
We do not invent path, confidence, or separate observed translation/rotation
heads. True time direction and metric scale remain ambiguous. CoMo's CMR is
preserved and is not claimed to be a strict SE(3) camera trajectory. These limits
are explicit in the [implementation contract](docs/IMPLEMENTATION.md).
