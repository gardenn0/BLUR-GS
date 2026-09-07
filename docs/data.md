# Dataset contract

A dataset is a `scene.json` manifest, images, an initial point cloud, and fixed motion
caches. The import and synthetic commands create this layout:

```text
data/my-scene/
├── scene.json
├── scene.motion.json       # created by prepare-motion, use this for real training
├── points.npz
├── images/*.png
└── motion/*.npz
```

All paths in a manifest are relative to its parent directory. Example schema (numeric
values are illustrative):

```json
{
  "version": 1,
  "camera_convention": "opencv_w2c",
  "point_cloud": "points.npz",
  "frames": [
    {
      "name": "frame_001",
      "split": "train",
      "image": "images/001.png",
      "K": [[500, 0, 319.5], [0, 500, 223.5], [0, 0, 1]],
      "w2c": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
      "motion": "motion/001.npz",
      "exposure_seconds": 0.01
    }
  ]
}
```

`points.npz` contains `points: float32[N,3]` and `colors: float32[N,3]` in [0,1].
Optional positive `scales: float32[N,3]` avoids nearest-neighbor scale initialization.
Coordinates must match the camera poses. Gaussian rotations initialize to identity and
opacity to 0.2. Frames can have distinct resolutions and intrinsics. Names are unique.

Each motion NPZ contains:

| Key | Shape/type | Meaning |
| --- | --- | --- |
| `flow` | float32[H,W,2] | Exposure displacement `(du,dv)` in this image's pixel units |
| `confidence` | float32[H,W] | Fixed weights in [0,1], zero for invalid pixels |
| `reference` | scalar string | `start` for IAAI or `midpoint` for explicitly aligned measurements |
| `sign_ambiguous` | scalar bool | Whether image-level temporal reversal should be considered |
| `depth` | optional float32[H,W] | Estimator depth for analysis, not implicit GS warm-up depth |
| `source` | scalar string | E.g. official model or analytic synthetic ground truth |

The cache must exactly match the training image resolution. If images are resized after
cache extraction, regenerate the caches or rescale their vectors and camera intrinsics
consistently. Caches and arrays load with `allow_pickle=False`.

For optional warm-up, add `initial_depth: "depth/001.npy"` to a frame. This array must
be z-depth at the initial/midpoint camera in the SfM scene coordinate scale. Zero/NaN
depths fall back to current GS depth. Unaligned metric monocular depth is not a valid
warm-up input without registration and scale alignment.

For quantitative evaluation add `sharp_image: "sharp/001.png"`, a registered sharp ground
truth with matching resolution. Evaluation deliberately errors when no sharp ground truth
is present; it does not report blur reconstruction PSNR as deblurring PSNR. `--split test`
selects held-out frames. Test poses are not optimized. The built-in metrics are PSNR and
SSIM; LPIPS is not currently included.

`import-colmap` reads `cameras`, `images`, and `points3D` text or binary files. It checks image
dimensions and writes resized PNG copies with correspondingly scaled calibration. It
supports already-undistorted pinhole cameras; it does not run SfM or estimate calibration.
`--holdout-every 8` reserves every eighth registered image for testing. Such held-out images
are still blurred inputs unless separate sharp references are supplied.

All image, array, checkpoint, and run directories are excluded from version control.
