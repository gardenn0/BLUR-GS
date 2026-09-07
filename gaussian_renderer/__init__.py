"""RGB and alpha-normalized z-depth with camera and geometry gradients."""

from dataclasses import dataclass

import torch
from torch import Tensor

from utils.pose_utils import pixel_grid
from scene.gaussian_model import GaussianScene


@dataclass
class RenderResult:
    rgb: Tensor
    depth: Tensor
    alpha: Tensor


class TorchRenderer:
    """Small-scene reference rasterizer: O(NHW), sorted front-to-back alpha compositing.

    Full anisotropic covariance projection; chunked pixel evaluation keeps memory bounded.
    Sorting and near-plane visibility are discrete, as in standard splatting.
    """

    def __init__(self, near: float = 0.01, eps2d: float = 0.3, pixel_chunk: int = 4096):
        self.near, self.eps2d, self.pixel_chunk = near, eps2d, pixel_chunk

    def __call__(
        self, scene: GaussianScene, w2c: Tensor, K: Tensor, height: int, width: int
    ) -> RenderResult:
        xyz = scene.means @ w2c[:3, :3].T + w2c[:3, 3]
        order = xyz[:, 2].argsort()
        xyz = xyz[order]
        z = xyz[:, 2].clamp_min(self.near)
        x, y = xyz[:, 0], xyz[:, 1]
        uv = torch.stack((K[0, 0] * x / z + K[0, 2], K[1, 1] * y / z + K[1, 2]), -1)
        zero = torch.zeros_like(z)
        J = torch.stack(
            (
                K[0, 0] / z,
                zero,
                -K[0, 0] * x / z.square(),
                zero,
                K[1, 1] / z,
                -K[1, 1] * y / z.square(),
            ),
            -1,
        ).reshape(-1, 2, 3)
        cov = scene.covariance()[order]
        R = w2c[:3, :3]
        cov2 = J @ R @ cov @ R.T @ J.transpose(-1, -2)
        cov2 = cov2 + self.eps2d * torch.eye(2, device=z.device, dtype=z.dtype)
        inverse = torch.linalg.inv(cov2)
        colors, opacity = scene.colors[order], scene.opacities[order]
        visible = (xyz[:, 2] > self.near).to(z.dtype)
        pixels = pixel_grid(height, width, K).reshape(-1, 2)
        rgbs, depths, alphas = [], [], []
        for grid in pixels.split(self.pixel_chunk):
            delta = grid[None] - uv[:, None]
            exponent = torch.einsum("npi,nij,npj->np", delta, inverse, delta)
            a = (opacity[:, None] * visible[:, None] * torch.exp(-0.5 * exponent)).clamp_max(0.999)
            prefix = torch.cat((torch.ones_like(a[:1]), 1 - a), 0)
            transmittance = torch.cumprod(prefix, 0)[:-1]
            weights = a * transmittance
            alpha = weights.sum(0)
            rgb = weights.T @ colors  # black background
            depth = (weights * z[:, None]).sum(0) / alpha.clamp_min(1e-8)
            rgbs.append(rgb)
            depths.append(depth)
            alphas.append(alpha)
        return RenderResult(
            torch.cat(rgbs).reshape(height, width, 3),
            torch.cat(depths).reshape(height, width),
            torch.cat(alphas).reshape(height, width),
        )


class GsplatRenderer:
    """gsplat 1.5.3 CUDA renderer. Ks are fixed; packed=False supports pose gradients."""

    def __init__(self, near: float = 0.01, eps2d: float = 0.3):
        try:
            from gsplat import rasterization
        except ImportError as exc:
            raise RuntimeError(
                "Install the cuda extra and a compatible CUDA toolkit for gsplat"
            ) from exc
        self.rasterization = rasterization
        self.near, self.eps2d = near, eps2d

    def __call__(
        self, scene: GaussianScene, w2c: Tensor, K: Tensor, height: int, width: int
    ) -> RenderResult:
        if not scene.means.is_cuda:
            raise ValueError("gsplat backend requires a CUDA device; use torch for CPU tests")
        # gsplat's CUDA kernels evaluate pixels at (j+.5, i+.5); our geometry uses (j,i).
        raster_K = K.clone()
        raster_K[0, 2] += 0.5
        raster_K[1, 2] += 0.5
        rendered, alpha, _ = self.rasterization(
            means=scene.means,
            quats=scene.quats,
            scales=scene.scales,
            opacities=scene.opacities,
            colors=scene.colors,
            viewmats=w2c[None],
            Ks=raster_K[None],
            width=width,
            height=height,
            near_plane=self.near,
            eps2d=self.eps2d,
            packed=False,
            render_mode="RGB+ED",
            rasterize_mode="classic",
        )
        return RenderResult(rendered[0, ..., :3], rendered[0, ..., 3], alpha[0, ..., 0])


def make_renderer(backend: str):
    if backend == "torch":
        return TorchRenderer()
    if backend == "gsplat":
        return GsplatRenderer()
    raise ValueError(f"Unknown renderer: {backend}")
