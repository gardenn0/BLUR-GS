# Implementation contract

## Preserved backbone

`scene/como_kernel.py`, `gaussian_renderer/__init__.py`, Gaussian representation,
and all CUDA sources are unchanged from the pinned CoMoGaussian commit. RGB
rendering, exposure weighting/masking, and original regularizers are retained.
`train.py` enables BLUR-GS by default and prepares flow before allocating scene
cameras/Gaussians on the GPU. `--baseline` explicitly disables flow supervision.
The loader passes its exact training CameraInfo paths/names to the preparation
callback, including its resolution suffix and dataset split. The estimator runs
in a subprocess using the current Python or `--iaai_python`, then exits before
reconstruction. Content-addressed automatic caches are validated and reused.

## Observed motion

The external IAAI model is evaluated with `supervise_pose=False`. The official
checkpoint is loaded strictly; its depth decoder remains instantiated for
checkpoint compatibility but is not used by BLUR-GS. No pose solver, velocity
conversion, or external metric depth supervision is used. Exposure duration is
not needed for displacement loss. The dummy intrinsics passed to `infer` are
irrelevant to its flow decoder with the pose head disabled.

The cache holds 2x224x320 displacement in network pixels, an explicit mask, and
center-crop/original resolution metadata. Mask means validity, not a learned
confidence estimate. Default masks only remove the network image border. Cache
names must match `camera.image_name`; duplicate and missing names fail loudly.

The original images must match the images used for CoMoGaussian reconstruction
(before its resolution downsampling). Undistort consistently before both paths.
Regenerate caches after changing source images, cropping, or checkpoint weights.

## Coordinates and flow

CoMo stores a transposed world-view matrix. All geometry functions use
`camera.world_view_transform.T` as column-vector world-to-camera W. Do not use
the kernel's internal concatenated `Rt` as a conventional extrinsic matrix.

The projection maps NDC to `(ndc+1)*size/2 - 0.5`, hence cx=(W-1)/2, cy=(H-1)/2.
Cropping/resizing uses half-pixel center conventions, matching align_corners=False.
The official IAAI crop offsets use Python round, as torchvision center_crop does.

Forward candidate:

    X0 = z0 * inverse(K) * [u,v,1]
    X1 = W1 * inverse(W0) * [X0,1]
    F01 = project(K, X1) - [u,v]

Backward candidate is computed independently from the END camera's depth and
W0 * inverse(W1), not by simply negating F01. Flow is converted from training
camera pixels into IAAI cache pixels before comparison. Only the endpoint field
is supervised. Intermediate poses remain constrained by the RGB blur loss and
CoMo's architecture/orthogonality regularizer.

Auto direction compares two image-global errors on a common valid support.
It never picks a sign separately per pixel. With known aligned direction use
`--flow_direction forward` or `backward`. Direction may remain temporally
ambiguous; this implementation does not claim recovery of true chronological
camera velocity from unordered images.

## Depth and gradient routing

An auxiliary render passes `[z_j, 1, 0]` per Gaussian via `override_color`, using
black background, where z_j=(W * [mu_j,1])_z. The first two output channels are
sum(w_j z_j) and sum(w_j); their quotient is expected z-depth. No RGB clamping,
normalization of features, or raw CUDA depth output is used in this pass.
This is a representative composited depth, not an exact ray/surface intersection.

Source/target accumulated alpha, positive depth, image bounds and optional
relative-depth consistency select valid points. Masks are detached; confidence
cannot be learned to switch off the loss. Loss is a weighted mean of vector
Charbonnier penalties in cache pixels. Low coverage skips the term and logs it;
frequent skips are an experiment failure signal, not a successful zero loss.

Auxiliary depth camera matrices are ALWAYS detached. In trajectory steps the
entire depth pass is no_grad; pose gradients come from explicit reprojection.
In geometry/joint steps, depth features and rasterizer weights remain connected
to Gaussian parameters. In joint mode this is an intentional block-gradient
surrogate, not the full derivative of depth w.r.t. the rendering camera.

The custom CoMo RGB/pose backward is inherited. The feature pass avoids the raw
distance-depth derivative but does not magically certify all CUDA derivatives.
Run the included actual-rasterizer finite-difference test on your GPU environment.

## Optimization schedule

| Phase | Gaussian RGB | Kernel RGB | Flow -> Gaussian | Flow -> kernel |
|---|---|---|---|---|
| baseline / warm-up | yes | upstream start_warp | no | no |
| trajectory_prior | yes | yes | no | yes |
| trajectory | frozen | yes | no | yes |
| geometry | yes | frozen (including mask/weight) | yes | no |
| joint | yes | yes | yes | yes through reprojection |

Default `flow_mode=como` preserves upstream optimizer step/zero-grad timing,
parameter trainability, learning-rate updates and densification/pruning schedule.
Both optimizer update gates stay enabled; train.py retains CoMo's start_warp
condition for kernel updates. After flow_start (default 4000), the phase is joint
immediately, with no trajectory-only stage and no alternating freezes. The flow
weight ramps for 2000 iterations. geometry_start/alternate_every and the separate
geometry_flow_weight are unused by this mode. Zero flow weight bypasses the
auxiliary render and flow computation. RGB terms remain unchanged.

Legacy modes remain explicit ablations: `trajectory` keeps trajectory_prior;
`joint` uses trajectory_prior until geometry_start then joint;
`alternating` uses trajectory_prior until geometry_start, then alternates
50 trajectory and 50 geometry steps by default. Legacy schedules manage
requires_grad and clear gradients at the beginning of each iteration;
geometry freezes suppress densification. These operations are bypassed in como
mode. All Gaussian parameters, not just centers, may respond via compositing;
translation-only geometry gradient routing is not claimed.

Flow weights are experimental, not paper-tuned hyperparameters. The default
isolates the additive loss at the schedule level, not computational cost or
resulting optimization trajectory. Extra gradients can change subsequent
geometry and pruning outcomes even when their schedules are unchanged.

## Checkpoint and evaluation

With BLUR-GS enabled, saving iterations write both ordinary scene PLY outputs
and `blur_chkpntITER.pth`: Gaussian state/filter/optimizer, complete CoMo kernel
and optimizer, phase configuration, camera identities, sampling stack and RNGs.
The phase is derived from the iteration on resume. Resume with the same dataset,
camera ordering, iteration budget and hyperparameters. BLUR-GS-specific flags
are checked; cache location may change. Original CoMo arguments are saved in
cfg_args but are not all checked automatically. For safety do not resume a run
with a different num_warp, model architecture or learning-rate schedule.

Gaussian-only upstream checkpoints are rejected for BLUR-GS resume because they
do not contain the learned exposure trajectories. Load only trusted checkpoints.

Sharp novel-view evaluation uses standard CoMo render.py/metrics.py and needs no
flow network. Use identical camera evaluation/refinement settings for all runs.

## Deliberately not asserted / not implemented

- No full-path motion observation or path-length loss from a two-component flow.
- No learned confidence, known time direction, or rotation/translation observed
  decomposition is fabricated from IAAI's available outputs.
- No metric scale recovery: jointly scaling depth and translation preserves flow.
- No guarantee that IAAI trained on its source distribution explains severe
  nonmonotonic blur, dynamic objects, or CoMo's spatial shutter mask.
- No replacement of CMR by strict SE(3), additional trajectory model, or extra
  pose/acceleration loss in this first implementation.
- No benchmark superiority or completed full-scene training is claimed.
