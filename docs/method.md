# Method-to-code mapping

Primary specification: the user-supplied `Blur_GS.pdf`, titled **BlurTraj-GS:
Geometry-Coupled Continuous Camera Trajectory Estimation for Motion-Blurred Gaussian
Splatting** (11 pages). The supplied ICCV 2025 Image-as-an-IMU and CoMoGaussian papers
are algorithm references. Implementation choices below are engineering choices, not
additional claims made by these papers.

| BLUR-GS specification | Implementation | Status |
| --- | --- | --- |
| Sec. 1.2 Gaussian centers, covariance, opacity, appearance | `scene.gaussian_model.GaussianScene` | Anisotropic Gaussian scene, DC appearance; configurable gradient-based split/prune |
| Eqs. 13-15 continuous SE(3) trajectory | `scene.trajectory.make_trajectory`, `utils.pose_utils.se3_exp` | CLI-selectable linear, clamped cubic spline, or Neural ODE in Lie algebra |
| Eqs. 17-20 exposure integration and L1/DSSIM | `gaussian_renderer.blur_renderer.render_blur`, `utils.loss_utils.rgb_loss` | Uniform temporal samples by default; optional weights in renderer API |
| Eqs. 21-23 fixed observed motion | `scene.motion_prior.ImageAsIMU`, `prepare_motion` | Official network adapter; locally defined confidence heuristic |
| Eqs. 24-28 Gaussian z-depth and backprojection | Both renderers, `utils.pose_utils.backproject` | Alpha-normalized expected z-depth, midpoint reference |
| Eqs. 29-34 predicted exposure motion | `utils.pose_utils.exposure_path` | Exact pinhole reprojection along the continuous pose samples |
| Eqs. 35-37 small-motion geometry | `utils.pose_utils.motion_jacobian`, `solve_camera_motion` | fx/fy-aware camera-motion Jacobian, weighted damped least squares |
| Eqs. 38-44 robust flow, magnitude, direction | `utils.motion_loss_utils.motion_loss` | Confidence-normalized Charbonnier; endpoint magnitude default |
| Eqs. 45-47 smoothness and anchor | Trajectory `regularizers()` | Zero linear acceleration, exact integrated spline acceleration, finite-difference ODE acceleration; midpoint twist anchor |
| Eqs. 49-50 depth warm-up | `train.mix_depth` | Optional registered initial depth; gradually fully GS-derived |
| Eqs. 52-56, Algorithm 1 alternating optimization | `train.Trainer.step` | Explicit parameter freezing, two optimizers, reduced-LR joint stage |
| Sec. 1.14 variable-specific translation/rotation routing | General Jacobian available | Specialized separate observed component losses not enabled |
| Eq. 63 path consistency | `utils.motion_loss_utils.path_consistency` | Standalone tested API; not enabled in trainer without measured paths |
| Eqs. 75-76 sharp novel-view rendering | `render.render_checkpoint` | Standard sharp GS rendering without blur estimator |

## Coordinate contract

All stored camera matrices are **world-to-camera** OpenCV transforms. Camera coordinates
are x right, y down, z forward. Twists are `[vx, vy, vz, wx, wy, wz]`; rotations are
radians and translations use scene units. Time is normalized to `[0,1]`, so the estimated
increment is displacement per exposure, not velocity per second. Known exposure seconds
are only needed when reporting physical velocities.

Pixel centers in the Python geometry are integers: `(0,0)` is the upper-left pixel center.
Depth is positive camera z, not inverse depth or ray length. `T(t) = T0 @ exp(xi(t))`
and `X_t = T(t) @ inverse(T_ref) @ X_ref` make Eq. 29 internally consistent. The Jacobian
instead maps camera body displacement to image displacement; its sign and coordinate
frame are converted with the SE(3) adjoint when initializing right-composed controls.

The default trajectory is linear in the shared reference Lie algebra:
`xi(t) = (1-t) xi_start + t xi_end`. There are two learned 6D endpoint controls per image.
SE(3) matrices are produced by the exponential, rather than elementwise matrix interpolation.
This is not a guarantee of a world-space straight camera-center path or constant body velocity
when the endpoint twists do not commute. Twist acceleration is exactly zero; its loss is
reported as zero and disabled by default. The midpoint remains trainable with a pose anchor.

Virtual render poses are sampled independently of the number of learned endpoints. Default
`exposure_samples=10` yields times `0, 1/9, ..., 1`, each weighted `1/10`; the smoke config also
uses 10 samples. Each objective additionally performs one midpoint depth render at t=0.5,
which is not among these ten exposure poses. `linear_exposure` controls radiometric
integration in linear light and is unrelated to the trajectory parameterization.

Linear checkpoints carry `format_version=2` and `trajectory_model=linear_se3`. Resume rejects
older four-control Bezier optimizer state with a clear error. Rendering can still recover
the exact midpoint of legacy version-1 trajectories using their cubic midpoint weights;
it does not reinterpret their saved trajectories as linear.

`--trajectory spline` selects a clamped single-segment cubic B-spline (four twist controls,
equivalent to a cubic Bezier basis). `--trajectory ode` selects a per-image Neural ODE with
trainable initial twist and velocity plus a 7-32-6 tanh MLP vector field. Fixed-step RK4
integrates from zero to each query independently; `--ode-steps` controls integration accuracy.
Both nonlinear models use checkpoint version 3, including model/config metadata for rendering.
Resume rejects model switches. These are BLUR-GS parameterizations, not CoMoGaussian reproduction.
The default acceleration weight remains zero; enable it explicitly for nonlinear experiments.

The [gsplat CUDA rasterizer](https://github.com/nerfstudio-project/gsplat/blob/v1.5.3/gsplat/cuda/csrc/RasterizeToPixels3DGSFwd.cu)
evaluates at half-integer centers. Its adapter adds 0.5 to principal points. COLMAP import
performs the opposite conversion after resolution scaling. The official
[COLMAP output format](https://colmap.github.io/format.html) defines w2c Hamilton quaternions
and little-endian binary records.

## Image-as-an-IMU alignment

IAAI Sec. 4.1 defines flow from the first virtual image to the last virtual image. It is
not a field attached to the midpoint image. Given a midpoint point with rendered z-depth,
we project it to `p0` and `p1`. We compare `p1-p0` to the observed flow sampled at `p0`.
This transports the fixed measurement to the same reconstructed scene point. For the
opposite time orientation we compare `p0-p1` to the same observed flow sampled at `p1`.
The lower-loss orientation is chosen **once per image**, preserving global consistency.
This does not establish the physical time direction from a single image. Temporal
disambiguation using adjacent video frames from IAAI Sec. 4.3 is not currently enabled.

For `reference=midpoint` caches, comparisons use the midpoint grid directly. This mode
is used by the analytic fixture and is not assigned to official IAAI outputs.

The official network has no learned confidence head. Our fixed confidence combines
valid crop coverage, finite depth/flow, texture, and nonsaturation. During optimization,
GS opacity and front-of-camera validity also mask unreliable reference points. Confidence
and discrete validity weights are detached. This is not a full visibility/occlusion
solution, and the effective supervision can be sparse; inspect `valid_fraction`.

## Depth, scale, and initialization

The motion solver operates on current GS/reference depth in the SfM coordinate scale.
For initialization only, it approximates IAAI's start-grid flow as a small-motion field
on the midpoint grid. Later training uses the reprojection-based alignment above. Invalid
or implausibly large initial estimates fall back to zero motion and are reported in
`summary.json`. The robust motion loss provides an initial nonzero direction signal.

The official network's predicted depth is cached for analysis but is not silently used
as GS warm-up depth. A user-supplied `initial_depth` must be midpoint-aligned z-depth in
the **same scene units**. Invalid external pixels fall back to GS depth. With no external
depth, GS depth is used from step zero. During trajectory steps the depth tensor is
detached; during geometry steps it remains differentiable.

## Loss conventions and choices

Pixel sums in the method document are implemented as confidence-normalized means for
resolution-stable hyperparameters. Charbonnier uses `sqrt(r^2 + eps^2) - eps`, with
`eps=1e-3`. DSSIM is `(1-SSIM)/2`, using a Gaussian window. Direction terms ignore
observed magnitudes below 0.1 pixels. Zero-confidence inputs produce finite zero loss.
An orientation with no valid observed pixels cannot win over one with valid measurements.

The default magnitude is endpoint displacement. Summed path length is implemented but
must only be selected if the observed magnitude represents that quantity. Equal endpoints
do not imply a short exposure path. Measured temporal paths and separate translation/
rotation observations are not provided by the IAAI checkpoint, so the optional corresponding
objectives in BLUR-GS are not silently fabricated.

By default, rendered sRGB colors are transformed to linear intensity before averaging
and converted back, following Image-as-an-IMU's image-formation discussion. Set
`linear_exposure: false` for direct intensity averaging as in a dataset's own convention.
The current appearance is bounded, view-independent RGB (equivalent to DC SH); no sensor
response or exposure gain is fitted.

## Relationship to CoMoGaussian

The supplied CoMoGaussian paper models latent trajectories with Neural ODEs and additionally
uses CMR transforms and learned pixel weighting. This implementation follows the supplied
BLUR-GS specification, which leaves the continuous parameterization open. Linear endpoint
interpolation remains the default; spline and a separate twist-space ODE are also available.
It does not claim numerical equivalence to the CoMoGaussian architecture or reported
metrics. The reference is useful for continuous exposure rendering and for future backbone
comparisons. No external repository source has been vendored here.

## Gaussian refinement and checkpoint continuity

Between refinement checks, geometry/joint updates accumulate the norms of position gradients
before clipping. Each Gaussian is scored by its mean over nonzero finite observations, so
the view present exactly at a refinement boundary does not determine the entire split set.
Zero gradient is used as an unobserved proxy; this is a world-space gradient heuristic, not
the original 3DGS screen-space visibility/radius statistics. Statistics start accumulating
with the first geometry update and reset after each eligible check, including checks with
no topology change. Scene scale therefore still affects the gradient threshold.

Split offsets are sampled in the parent's principal-axis frame and rotated by its quaternion
into world space. Unchanged surviving Gaussians retain their Adam moments after pruning or
splitting. Both children of a split have zero moments; the optimizer's shared per-parameter
step counters and group settings are retained. At least one Gaussian survives pruning.

New checkpoints additionally store pending refinement sums/counts, the PyTorch CPU RNG state,
and the RNG state of the selected CUDA device when training on CUDA. Resume restores them
after constructing and loading all models/optimizers. The existing linear version 2 and
nonlinear version 3 remain readable; older checkpoints without these fields initialize empty
statistics and warn if refinement can still occur. Exact future splits cannot be recovered
from a checkpoint that never stored the random state. Continuity checks apply to the same
device and runtime; switching CPU/CUDA or nondeterministic CUDA kernels is not a guarantee of
bitwise equality.

## Experimental limitations

This version has gradient-based Gaussian splitting and opacity pruning, but no high-order SH,
multi-scale training, rolling-shutter model, dynamic-object mask, or learned shutter weights.
The topology refinement does not reproduce the full 3DGS adaptive-density schedule.
PyTorch rendering is an O(NHW)
correctness backend with no tile culling; gsplat
is the intended GPU backend. Startup scale estimation is chunked O(N^2) nearest-neighbor
distance unless input scales are supplied. COLMAP import caps initial points at 10,000 by
default. There is no claimed real-scene benchmark reproduction in this release.
