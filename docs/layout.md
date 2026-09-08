# Research-style layout migration

This is a file-organization and execution-interface refactor, not a change of method
or an adoption of another paper's training recipe. It follows the recurring top-level
layout seen in [Deblur-GS](https://github.com/Chaphlagical/Deblur-GS),
[DeblurGS](https://github.com/taekkii/deblurgs),
[CoMoGaussian](https://github.com/Jho-Yonsei/CoMoGaussian), and
[BAGS](https://github.com/snldmt/BAGS). Their source code is not copied.

## Python imports

| Previous module | New location |
| --- | --- |
| `blur_gs.training.TrainConfig/read_config` | `arguments.TrainConfig/read_config` |
| `blur_gs.training.Trainer/train/mix_depth` | `train.Trainer/train/mix_depth` |
| `blur_gs.evaluation` | `render` |
| `blur_gs.geometry` | `utils.pose_utils` |
| `blur_gs.trajectory` | `scene.trajectory` |
| `blur_gs.scene` | `scene.gaussian_model` |
| `blur_gs.data.Frame/validate_camera` | `scene.cameras` |
| `blur_gs.data.load_dataset/load_manifest/read_frame` | `scene.dataset_readers` |
| `blur_gs.data.read_image/save_image` | `utils.image_utils` |
| `blur_gs.colmap` | `scene.colmap_loader` |
| `blur_gs.motion` | `scene.motion_prior` |
| `blur_gs.rendering.*Renderer/make_renderer/RenderResult` | `gaussian_renderer` |
| `blur_gs.rendering.render_blur` | `gaussian_renderer.blur_renderer` |
| `blur_gs.rendering.srgb_to_linear/linear_to_srgb` | `utils.image_utils` |
| `blur_gs.losses.rgb_loss/ssim/charbonnier/weighted_mean` | `utils.loss_utils` |
| `blur_gs.losses.motion_loss/path_consistency` | `utils.motion_loss_utils` |
| `blur_gs.synthetic.make_synthetic` | `scripts.make_synthetic.make_synthetic` |

The old Python submodule imports are intentionally not duplicated. Only the legacy
CLI dispatcher remains in `blur_gs/`; all actual algorithm code lives in the new layout.
After updating an existing editable installation, run `python -m pip install -e .`
so the console script and package discovery refer to the new modules. With dependencies
already installed, the direct entry scripts also work from a checkout without that step.

## Data and checkpoint compatibility

Dataset manifests, cache paths, format-version-2 linear state dictionaries and optimizer
state are unchanged. Continue a linear run with `train.py --resume <checkpoint>`, using
the same optimization configuration and ordered frames. Legacy Bezier checkpoints remain
render-only. The adapter still uses the separately installed official `iaai` package.

A scene directory supplied through `-s` resolves to `scene.motion.json`, or `scene.json`
when no motion manifest exists. This is not automatic import of raw COLMAP data. Use
`scripts/import_colmap.py` and `scripts/prepare_motion.py` first for real scenes.
`-m` selects a training output directory, not a different checkpoint file format.

GPU rendering uses the external `gsplat` dependency without vendored CUDA kernels.
The subsequent training extension adds `scene/density.py` and `utils/sh_utils.py` for
adaptive Gaussians and degree-3 SH. Default and smoke exposure samples are both ten.
See `docs/training.md` for schedules and checkpoint compatibility.
