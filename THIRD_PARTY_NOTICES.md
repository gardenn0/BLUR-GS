# Attribution and license scope

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
