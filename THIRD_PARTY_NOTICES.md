# Attribution and license scope

## Standalone BAD-style path

`bad_blur_gs/trajectory.py` adapts interpolation equations/structure from
WU-CVGL/BAD-Gaussians, commit `bdd8b3e2ba068aa7be8e3ab6fb5277abd40909bc`
(Lingzhe Zhao, Peng Wang, Peidong Liu; ECCV 2024), licensed Apache-2.0.
`bad_blur_gs/gaussians.py` adapts initialization/refinement from Nerfstudio v1.0.3
`splatfacto.py`, copyright 2022 the Regents of the University of California,
Nerfstudio Team and contributors, licensed Apache-2.0. Both are modified to run
without Nerfstudio. The license is in `third_party/licenses/APACHE-2.0.txt`.
`bad_blur_gs/colmap_io.py` copies this repository's inherited Inria COLMAP reader
with its original research-use notice. Those terms continue to apply.
`gsplat` and `pypose` are installed dependencies, not vendored binaries. See
`docs/BAD_BLUR_GS.md` for implementation deviations and references. The standalone
port is not an official BAD-Gaussians release or a new-method novelty claim.

The v2 alignment adds `bad_blur_gs/coordinates.py`, adapted from Nerfstudio
v1.0.3 `camera_utils.py` and `colmap_dataparser.py`. Unmodified helper bodies
are included in `tests/reference/ns_camera_utils_v103.py` for comparison.
`tests/reference/bad_render.py` is an independent transcription of the
single-view rendering sequence from BAD commit bdd8b3e. These are Apache-2.0,
with the upstream copyright notices retained. The production renderer calls
gsplat 0.1.11 without modifying its CUDA kernels or approximate pose backward.

This is an experimental implementation of BLUR-GS, based on **CoMoGaussian** by
Jungho Lee et al. (ICCV 2025). It is not an official implementation by the authors
of CoMoGaussian or Image as an IMU, and no benchmark improvement is claimed.

## CoMoGaussian and inherited components

The repository preserves the upstream CoMoGaussian source snapshot at commit
`cd6d536a197a97d032849d49747be30e44ab5307`:
https://github.com/Jho-Yonsei/CoMoGaussian

Upstream demonstration assets, generated GLM documentation and precompiled binaries
are omitted; build the CUDA extensions from the included source.

Its root MIT license is retained in `LICENSE`. Files inherited from Inria/MPII
have their original notices and research-use terms; in particular see
`submodules/diff-gaussian-rasterization-pose-backprop/LICENSE.md`.
The root MIT notice does **not** override component-specific terms. The original
README and acknowledgements are preserved in `README_CoMoGaussian.md`.

## SplaTAM reference

The auxiliary z-depth feature pass in `blur_gs/depth.py` is adapted from the
approach in SplaTAM's `get_depth_and_silhouette`: render depth and silhouette as
precomputed color channels. The BLUR-GS version uses `[z, 1, 0]` with explicit
alpha normalization and detached camera matrices for the depth pass.

Source: https://github.com/spla-tam/SplaTAM/blob/main/utils/slam_helpers.py

BSD 3-Clause License

Copyright (c) 2023, Nikhil Varma Keetha

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

## Image as an IMU

Jerred Chen and Ronald Clark, ICCV 2025:
https://github.com/jerredchen/image-as-an-imu

Used as an external, frozen estimator. Its source, weights, and datasets are not
redistributed here. Install it separately and retain its applicable licenses.

## Other methodological references

- 2D Gaussian Splatting (Huang et al., SIGGRAPH 2024): depth-to-point geometry.
  https://github.com/hbb1/2d-gaussian-splatting
- Splat-based 3D Scene Reconstruction with Extreme Motion-blur (Jang et al.,
  ICCV 2025): depth backprojection/reprojection for inter-frame flow constraints.

No code from these two repositories is vendored by the BLUR-GS additions.
