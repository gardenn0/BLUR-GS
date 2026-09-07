"""Conventional anisotropic Gaussians with degree-zero spherical-harmonic appearance."""

from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from .geometry import quaternion_matrix


class GaussianScene(nn.Module):
    def __init__(self, points: Tensor, colors: Tensor, scales: Tensor | None = None):
        super().__init__()
        if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
            raise ValueError("points must be a nonempty Nx3 array")
        if colors.shape != points.shape or not torch.isfinite(points).all():
            raise ValueError("points and RGB colors must be finite matching Nx3 arrays")
        if not torch.isfinite(colors).all() or (colors < 0).any() or (colors > 1).any():
            raise ValueError("colors must be finite RGB values in [0,1]")
        self.means = nn.Parameter(points.clone())
        if scales is None:
            # Chunked nearest neighbors avoids allocating the complete NxN matrix.
            if len(points) == 1:
                distance = points.new_full((1,), 0.05)
            else:
                distance = []
                for start in range(0, len(points), 512):
                    d = torch.cdist(points[start : start + 512], points)
                    rows = torch.arange(d.shape[0], device=d.device)
                    d[rows, start + rows] = float("inf")
                    distance.append(d.min(-1).values)
                distance = torch.cat(distance)
            scales = distance.clamp_min(1e-4)[:, None].repeat(1, 3) * 0.5
        if scales.shape != points.shape or not torch.isfinite(scales).all() or (scales <= 0).any():
            raise ValueError("scales must be finite positive Nx3")
        self.log_scales = nn.Parameter(scales.log())
        q = points.new_zeros(len(points), 4)
        q[:, 0] = 1
        self.quats = nn.Parameter(q)
        self.opacity_logits = nn.Parameter(points.new_full((len(points),), -1.38629436))
        self.color_logits = nn.Parameter(torch.logit(colors.clamp(1e-4, 1 - 1e-4)))

    @property
    def scales(self) -> Tensor:
        return self.log_scales.clamp(-12, 8).exp()

    @property
    def colors(self) -> Tensor:
        return self.color_logits.sigmoid()

    @property
    def opacities(self) -> Tensor:
        return self.opacity_logits.sigmoid()

    def covariance(self) -> Tensor:
        R = quaternion_matrix(self.quats)
        RS = R * self.scales[:, None, :]
        return RS @ RS.transpose(-1, -2)

    def export_ply(self, path: str | Path) -> None:
        """3DGS PLY convention: raw log-scales/logit opacity, wxyz, degree-zero SH."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        values = torch.cat(
            (
                self.means,
                torch.zeros_like(self.means),
                (self.colors - 0.5) / 0.28209479177387814,
                self.opacity_logits[:, None],
                self.log_scales,
                torch.nn.functional.normalize(self.quats, dim=-1),
            ),
            -1,
        )
        names = [
            "x",
            "y",
            "z",
            "nx",
            "ny",
            "nz",
            "f_dc_0",
            "f_dc_1",
            "f_dc_2",
            "opacity",
            "scale_0",
            "scale_1",
            "scale_2",
            "rot_0",
            "rot_1",
            "rot_2",
            "rot_3",
        ]
        with path.open("w", encoding="ascii") as stream:
            stream.write(f"ply\nformat ascii 1.0\nelement vertex {len(values)}\n")
            stream.writelines(f"property float {name}\n" for name in names)
            stream.write("end_header\n")
            np.savetxt(stream, values.detach().cpu().numpy(), fmt="%.8g")
