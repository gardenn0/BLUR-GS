# Periodic test evaluation and TensorBoard

Install the logging extra in the existing training environment:

```bash
python -m pip install '.[logging]'
```

TensorBoard is recorded automatically when installed; no `--tensorboard` flag is
needed. If it is unavailable, training continues with JSON logs and prints a
notice. Use `--no-tensorboard` to disable it explicitly. An explicit
`--tensorboard` remains supported and requires the dependency to be installed.

Add `--eval` to evaluate held-out test images as ground truth, using a separate
`sharp_image` when provided. The prepared manifest's train/test split is retained;
this option does not import raw COLMAP data or create a new split. The default
evaluation interval is 1000 steps, plus the final training step. Override the
interval with `--eval-every N`.
For CoMoGaussian-style explicit evaluation steps, use
`--test_iterations $(seq 1000 1000 40000)` (Linux shell). This list takes precedence
over the evaluation interval, and only the listed steps are evaluated; an unlisted
final step is not added. Repeated steps are evaluated once. Evaluation steps do
not set the total training length; use `--iterations 40000` for that.

The existing default configuration continues to save checkpoints every 1,000
steps, independently of evaluation. TensorBoard is an optional dependency.

Each test manifest entry must have `split: "test"`. With `--eval`, its `image`
is the GT unless a separate `sharp_image` path is present, which takes precedence.
GT must have the same resolution as the calibrated input. No extra GT flag is
needed. The earlier `--eval-test-images-as-sharp` option is retained only for
compatibility with existing commands. Using schedules without `--eval` still
requires explicit `sharp_image` references (or the legacy declaration).
Missing test views, missing GT and resolution mismatches fail before training starts.
Test views are never added to training or used to optimize test camera poses.

For the prepared blurfactory scene (29 train images, 5 confirmed sharp test images):

```bash
export CUDA_INC_PATH="$CONDA_PREFIX/targets/x86_64-linux/include"

python -u train.py \
    -s data/blurfactory_eval \
    -m outputs/blurfactory_40k_tb \
    --config configs/default.yaml \
    --iterations 40000 \
    --eval \
    --test_iterations $(seq 1000 1000 40000)
```

Use a new output directory for a new run. For a short validation, use
`--config configs/smoke.yaml --backend gsplat --device cuda --eval --test_iterations 12 24`
and omit `--iterations 40000` (the smoke configuration runs 24 steps).

In a second terminal, activate the same conda environment and run:

```bash
tensorboard --logdir outputs/blurfactory_40k_tb/tensorboard --host 127.0.0.1 --port 6006
```

When using VS Code Remote SSH, forward port 6006 from the compute node where
TensorBoard is running and open `http://localhost:6006`. A connection to a login
node alone does not forward a compute node's localhost.

If VS Code is connected to `login01` and TensorBoard runs on `compute02`, open
a separate terminal on **login01** and keep this tunnel running:

```bash
ssh -N -L 16006:127.0.0.1:6006 jwlee@compute02
```

Then forward **16006** in VS Code's Ports panel and open `http://localhost:16006`.
Do not run the tunnel on compute02 itself.

Scalars include `train/total`, `train/rgb`, `train/motion`, `train/gaussians`,
`test/psnr` and `test/ssim`. Training scalars use `log_every` (50 by default), plus
the first and last step. Test scalars are arithmetic means over all test views.
The 29 blurred training frames alone drive losses and trajectory optimization;
the 5 sharp held-out frames are read only for evaluation and visualization.
Images under `test/render_left_gt_right/*` show the prediction on the left and
sharp ground truth on the right. Per-image scores, GT paths and mean scores are
also saved to `evaluation.jsonl`, separate from the existing `metrics.jsonl`.
Predictions are clamped to [0,1]; PSNR uses an MSE floor of 1e-12. SSIM uses the
repository's existing implementation. These metrics do not include LPIPS or
test-pose fitting and should not be described as an identical baseline protocol.

Monitoring can be enabled when resuming an older checkpoint. Keep all optimization
settings the same and add the reporting flags plus, for example,
`--resume outputs/blurfactory_40k/checkpoint_001000.pt` with the original output
directory. TensorBoard uses the checkpoint step and hides events from abandoned
future steps. `evaluation.jsonl` is likewise truncated beyond the resumed step.
Old training losses are not backfilled into TensorBoard. Reporting does not
change model parameters, gradients or the PyTorch RNG used by refinement.
Already-running Python processes need to be restarted from a checkpoint to use
the new code; applying a patch does not modify a running process.
