# Linear trajectory, ten virtual poses

The default and CPU smoke configurations use ten uniform samples at `t=k/9`,
`k=0,...,9`, each weighted by 0.1. The two learned endpoint twists define
`T(t)=T0 exp((1-t) xi0+t xi1)`. A separate `t=0.5` pass supplies depth.
The supplied Blur_GS.pdf permits this parameterization (Sec. 1.3); nonlinear
trajectories, Neural ODEs, and learned shutter weights are not needed for this run.

## Prepare and train

Use a CUDA-capable PyTorch installation, a compatible CUDA toolkit/compiler, and
`python -m pip install -e '.[cuda,dev]'`. The CUDA backend is pinned to gsplat 1.5.3.
See the README and `docs/image_as_imu.md` for installation and official weights.
Motion inference can run in a separate environment from reconstruction if the
upstream network dependencies conflict: transfer the prepared scene and caches.

```bash
python scripts/import_colmap.py --model /path/to/undistorted/sparse/0 --images /path/to/undistorted/images --output data/my-scene --downscale 4
python scripts/prepare_motion.py --data data/my-scene/scene.json --checkpoint /path/to/official-checkpoint.pth --device cuda
python scripts/check_training.py -s data/my-scene --config configs/default.yaml
python train.py -s data/my-scene -m outputs/my-scene --config configs/default.yaml
python render.py -s data/my-scene -m outputs/my-scene --backend gsplat --device cuda
python metrics.py -s data/my-scene -m outputs/my-scene --backend gsplat --device cuda --split test
```

Evaluation requires explicit `sharp_image` references and a test split in the
manifest. It does not treat the blurry training images as sharp ground truth.
Start at reduced resolution; ten exposure graphs are retained for backpropagation,
so full-resolution memory requirements depend on the scene and Gaussian count.

## Reconstruction controls

Defaults are starting hyperparameters, not a reproduced benchmark recipe.

| YAML setting | Default | Meaning |
| --- | --- | --- |
| `sh_degree` | 3 | Maximum appearance degree; DC-only is 0 |
| `sh_interval` | 1000 | Global iterations between SH degree increases |
| `densify_from` / `densify_until` | 500 / 15000 | Global iteration window for refinement |
| `densify_interval` | 100 | Geometry/joint optimizer updates between refinements |
| `densify_grad_threshold` | 0.0002 | Mean normalized-screen gradient threshold |
| `max_gaussians` | 1000000 | Growth cap; does not truncate a larger input cloud |
| `prune_opacity` | 0.005 | Remove low-opacity Gaussians, keeping at least one |
| `opacity_reset_interval` | 3000 | Geometry/joint updates between opacity resets; 0 disables |

Set `densify_until: 0` to disable both refinement and opacity resets. Statistics
include only RGB exposure passes and only geometry/joint phases. Gradient norms
are accumulated across exposure views to avoid cancellation, corrected for the
uniform exposure weight, and converted from pixels to normalized screen units.
Small high-gradient Gaussians are cloned; large ones are split symmetrically along
their longest covariance axis and shrunk by 1.6. The size threshold is 1% of camera
extent, with a cloud-extent fallback for a single camera. This deterministic policy
is an engineering choice, not a claim of identical upstream densification.

Adam moments are preserved for retained Gaussians and zeroed for new children.
Opacity reset also clears its Adam moments. View-dependent appearance adds real
SH coefficients to bounded DC RGB, with conventional coefficient ordering in PLY.
Scale initialization uses a CPU KD-tree rather than quadratic pairwise distances.

An ambiguous image's temporal direction is selected at the first supported,
nonzero-motion objective evaluation and then locked. Zero-motion or unsupported
views defer the selection. This resolves the per-image loss convention, not the
unobservable absolute direction of time or inter-image temporal synchronization.

## Resume

```bash
python train.py -s data/my-scene -m outputs/my-scene --config configs/default.yaml --resume outputs/my-scene/checkpoint.pt --iterations 40000
```

Checkpoints save the dynamic scene, SH degree, density statistics, fixed directions,
geometry update count, and both optimizers. Existing format-2 checkpoints without
`training_state_version: 1` remain renderable but cannot resume into this changed
optimizer topology. Keep training settings and input data unchanged when resuming;
only iteration/log/checkpoint limits and device may change.

## References and verification boundary

The implementation was checked against the local 11-page Blur_GS.pdf.
[Deblur-GS training](https://github.com/Chaphlagical/Deblur-GS/blob/main/train.py)
informed exposure-view statistics and density/SH scheduling; the
[gsplat 1.5.3 renderer](https://github.com/nerfstudio-project/gsplat/blob/v1.5.3/gsplat/rendering.py)
defines the metadata contract. Source from these projects is not vendored.
[CoMoGaussian](https://github.com/Jho-Yonsei/CoMoGaussian) is an architectural
reference; its Neural ODE/CMR/learned weights are not implemented here.

CPU regression tests cover training, density changes, optimizer state migration,
fixed orientations, SH export, and exact deterministic resume. Official IAAI
checkpoint inference and real-scene CUDA quality still require the training machine
and data. This repository does not claim measured real-scene PSNR/SSIM/LPIPS.
