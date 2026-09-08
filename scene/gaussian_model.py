"""Anisotropic Gaussians with progressive real spherical-harmonic appearance."""

from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from utils.pose_utils import quaternion_matrix
from utils.sh_utils import sh_basis


class GaussianScene(nn.Module):
    def __init__(
        self, points: Tensor, colors: Tensor, scales: Tensor | None = None, sh_degree: int = 0
    ):
        super().__init__()
        if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
            raise ValueError("points must be a nonempty Nx3 array")
        if colors.shape != points.shape or not torch.isfinite(points).all():
            raise ValueError("points and RGB colors must be finite matching Nx3 arrays")
        if not torch.isfinite(colors).all() or (colors < 0).any() or (colors > 1).any():
            raise ValueError("colors must be finite RGB values in [0,1]")
        self.means = nn.Parameter(points.clone())
        if scales is None:
            # CPU KD-tree avoids quadratic startup work and GPU allocation.
            if len(points) == 1:
                distance = points.new_full((1,), 0.05)
            else:
                from scipy.spatial import cKDTree

                distances, _ = cKDTree(points.detach().cpu().numpy()).query(
                    points.detach().cpu().numpy(), k=2, workers=-1
                )
                distance = torch.as_tensor(
                    distances[:, 1], device=points.device, dtype=points.dtype
                )
            scales = distance.clamp_min(1e-4)[:, None].repeat(1, 3) * 0.5
        if scales.shape != points.shape or not torch.isfinite(scales).all() or (scales <= 0).any():
            raise ValueError("scales must be finite positive Nx3")
        self.log_scales = nn.Parameter(scales.log())
        q = points.new_zeros(len(points), 4)
        q[:, 0] = 1
        self.quats = nn.Parameter(q)
        self.opacity_logits = nn.Parameter(points.new_full((len(points),), -1.38629436))
        self.color_logits = nn.Parameter(torch.logit(colors.clamp(1e-4, 1 - 1e-4)))
        if sh_degree not in range(4):
            raise ValueError("sh_degree must be between zero and three")
        self.sh_rest = nn.Parameter(points.new_zeros(len(points), (sh_degree + 1) ** 2 - 1, 3))
        self.register_buffer("active_sh_degree", torch.tensor(0, device=points.device))

    def view_colors(self, w2c):
        degree = int(self.active_sh_degree)
        if degree == 0:
            return self.colors
        center = -w2c[:3, :3].T @ w2c[:3, 3]
        directions = torch.nn.functional.normalize(self.means - center, dim=-1)
        basis = sh_basis(directions, degree)[..., 1:]
        return (
            self.colors + (basis[..., None] * self.sh_rest[:, : basis.shape[-1]]).sum(1)
        ).clamp_min(0)

    @classmethod
    def from_state(cls, state):
        degree = int((state["sh_rest"].shape[1] + 1) ** 0.5 - 1) if "sh_rest" in state else 0
        scene = cls(
            state["means"], state["color_logits"].sigmoid(), state["log_scales"].exp(), degree
        )
        state = dict(state)
        state.setdefault("sh_rest", scene.sh_rest.detach())
        state.setdefault("active_sh_degree", scene.active_sh_degree)
        scene.load_state_dict(state)
        return scene

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
        """3DGS PLY convention: raw log-scales/logit opacity, wxyz, channel-major SH."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        values = torch.cat(
            (
                self.means,
                torch.zeros_like(self.means),
                (self.colors - 0.5) / 0.28209479177387814,
                self.sh_rest.transpose(1, 2).flatten(1),
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
        names[9:9] = [f"f_rest_{i}" for i in range(self.sh_rest.shape[1] * 3)]
        with path.open("w", encoding="ascii") as stream:
            stream.write(f"ply\nformat ascii 1.0\nelement vertex {len(values)}\n")
            stream.writelines(f"property float {name}\n" for name in names)
            stream.write("end_header\n")
            np.savetxt(stream, values.detach().cpu().numpy(), fmt="%.8g")
