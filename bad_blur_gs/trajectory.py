"""BAD-style translation interpolation and cumulative SO(3) splines.

Adapted from WU-CVGL/BAD-Gaussians (Apache-2.0); see THIRD_PARTY_NOTICES.md.
Camera adapters use OpenCV; controls use BAD's OpenGL local camera frame.
"""
from dataclasses import dataclass, replace
import torch
from torch import nn
import pypose as pp


@dataclass
class Camera:
    image_name: str
    image_path: str
    c2w: torch.Tensor
    K: torch.Tensor
    image_width: int
    image_height: int
    uid: int = -1

    @property
    def world_view_transform(self):
        r = self.c2w[:3, :3].T
        t = -r @ self.c2w[:3, 3:4]
        w2c = torch.cat((torch.cat((r, t), dim=1), self.c2w.new_tensor([[0, 0, 0, 1]])), dim=0)
        return w2c.T

    @property
    def focal_x(self):
        return self.K[0, 0]

    @property
    def focal_y(self):
        return self.K[1, 1]

    def scaled(self, width, height, factor=None):
        # Pixel-center coordinates used by the shared flow reprojection.
        scale = self.K.new_tensor([width / self.image_width, height / self.image_height]
                                 if factor is None else [factor, factor])
        k = self.K.clone()
        k[:2, :2] *= scale[:, None]
        k[:2, 2] = (k[:2, 2] + 0.5) * scale - 0.5
        return replace(self, K=k, image_width=width, image_height=height)


def interpolate(controls, times, mode):
    """Controls are PyPose SE3 (xyzw quaternion); returns SE3 at all times."""
    t = controls.translation()
    q = controls.rotation()
    u = times
    if mode == "linear":
        translation = (1 - u[:, None]) * t[0] + u[:, None] * t[1]
        rotation = q[0] @ pp.so3(u[:, None] * (q[0].Inv() @ q[1]).Log().tensor()).Exp()
    elif mode == "cubic":
        u2, u3 = u * u, u * u * u
        weights = torch.stack((1/6-u/2+u2/2-u3/6, 2/3-u2+u3/2,
                               1/6+u/2+u2/2-u3/2, u3/6), dim=-1)
        translation = weights @ t
        cumulative = torch.stack((5/6+u/2-u2/2+u3/6,
                                  1/6+u/2+u2/2-u3/3, u3/6), dim=-1)
        rotation = q[0]
        for j in range(3):
            delta = (q[j].Inv() @ q[j+1]).Log().tensor()
            rotation = rotation @ pp.so3(cumulative[:, j:j+1] * delta).Exp()
    else:
        raise ValueError(mode)
    return pp.SE3(torch.cat((translation, rotation.tensor()), dim=-1))


class ExposureTrajectory(nn.Module):
    def __init__(self, cameras, config, device):
        super().__init__()
        self.mode = config.trajectory
        self.num_virtual_views = config.num_virtual_views
        self.names = [c.image_name for c in cameras]
        if not self.names or len(set(self.names)) != len(self.names):
            raise ValueError("Training camera names must be nonempty and unique")
        self.indices = {name: i for i, name in enumerate(self.names)}
        self.register_buffer("base_c2w", torch.stack([c.c2w for c in cameras]).to(device))
        count = 2 if self.mode == "linear" else 4
        # Same nonzero tangent perturbation used by BAD; no learned view embedding.
        self.controls = pp.Parameter(pp.randn_se3(len(cameras), count,
                                                  sigma=config.initial_noise, device=device))

    def poses(self, name, times=None):
        index = self.indices[name]
        if times is None:
            times = torch.linspace(0, 1, self.num_virtual_views, device=self.controls.device)
        else:
            times = torch.as_tensor(times, dtype=self.controls.dtype, device=self.controls.device)
        delta = interpolate(self.controls[index].Exp(), times, self.mode)
        flip = torch.diag(self.base_c2w.new_tensor([1, -1, -1, 1]))
        # BAD composes base_gl @ delta_gl. Return OpenCV for renderer/flow.
        return (self.base_c2w[index] @ flip @ delta.matrix()) @ flip

    def cameras(self, camera, times=None):
        return [replace(camera, c2w=pose) for pose in self.poses(camera.image_name, times)]
