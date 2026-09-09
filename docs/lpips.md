# LPIPS for completed training runs

Install `lpips==0.1.4` into the training environment. Existing compatible PyTorch
and torchvision installations can be reused. The first evaluation downloads the
pretrained AlexNet weights if they are not already in PyTorch's cache.

```bash
python -m pip install lpips==0.1.4
export CUDA_INC_PATH="$CONDA_PREFIX/targets/x86_64-linux/include"
python scripts/evaluate_lpips.py \
    -s data/blurfactory_eval \
    -m /path/to/outputs/blur-gs/ours/blurfactory \
    --eval --device cuda --backend gsplat
```

This evaluates the final `checkpoint.pt`. Add `--all-checkpoints` to evaluate all
available intermediate checkpoints and the final checkpoint in step order.
The final step is evaluated only once. The checkpoint's recorded evaluation
convention and manifest are reused when the corresponding arguments are omitted.

Only `split: test` frames are selected. With `--eval`, test input images serve as
GT unless the manifest supplies a separate `sharp_image`. Test frames appearing
in the checkpoint's training frame list are rejected. Test poses are fixed to
their provided calibration. No training, trajectory fitting, resizing, PNG
quantization or reblurring is performed.

The metric uses pretrained AlexNet LPIPS v0.1, native-resolution RGB images
clamped to [0,1] and mapped to [-1,1], with arithmetic averaging over test images.
This follows the AlexNet and input-normalization choices in
[CoMoGaussian's evaluation](https://github.com/Jho-Yonsei/CoMoGaussian/blob/main/train.py#L280)
and the [official LPIPS implementation](https://github.com/richzhang/PerceptualSimilarity).
Lower scores indicate smaller perceptual differences; this does not imply that
the complete BLUR-GS and CoMoGaussian evaluation protocols are identical.

Per-image and mean scores are saved to `OUTPUT/lpips.json`. When TensorBoard is
installed, `test/lpips` is added to `OUTPUT/tensorboard` at each checkpoint's saved
step. Existing training and PSNR/SSIM events are preserved. Repeating the command
replaces `lpips.json` and appends new LPIPS events for the evaluated steps.
This command evaluates saved models only; training-time PSNR/SSIM logging is unchanged.
