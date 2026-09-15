"""Standard 3DGS parameters with BAD/Splatfacto refinement rules.

Refinement and initialization adapted from Nerfstudio v1.0.3, copyright 2022
the Regents of the University of California, Nerfstudio Team and contributors,
Apache-2.0. See third_party/licenses/APACHE-2.0.txt and THIRD_PARTY_NOTICES.md.
"""
import math
from pathlib import Path
import torch
from torch import nn
from .config import exponential_lr


class Gaussians(nn.Module):
    def __init__(self, tensors, config):
        super().__init__()
        self.params = nn.ParameterDict({k: nn.Parameter(v) for k, v in tensors.items()})
        self.max_sh_degree = config.sh_degree
        self.active_sh_degree = 0
        lrs = dict(means=config.position_lr, features_dc=config.feature_lr,
                   features_rest=config.feature_lr/20, opacities=config.opacity_lr,
                   scales=config.scaling_lr, quats=config.rotation_lr)
        self.optimizer = torch.optim.Adam([
            {"params": [p], "lr": lrs[k], "name": k} for k, p in self.params.items()], eps=1e-15)
        self.grad_sum = self.vis_count = self.max_radii = None

    @classmethod
    def from_points(cls, xyz, rgb, config):
        from scipy.spatial import cKDTree
        distances, _ = cKDTree(xyz.detach().cpu().numpy()).query(xyz.detach().cpu().numpy(), k=4)
        scale = torch.as_tensor(distances[:, 1:].mean(-1), device=xyz.device, dtype=xyz.dtype).clamp_min(1e-7)
        n = len(xyz)
        u, v, w = torch.rand(3, n, device=xyz.device)
        quats = torch.stack((torch.sqrt(1-u)*torch.sin(2*math.pi*v),
                             torch.sqrt(1-u)*torch.cos(2*math.pi*v),
                             torch.sqrt(u)*torch.sin(2*math.pi*w),
                             torch.sqrt(u)*torch.cos(2*math.pi*w)), -1)
        dc = (rgb-0.5)/0.28209479177387814 if config.sh_degree else torch.logit(rgb.clamp(1e-6, 1-1e-6))
        return cls(dict(means=xyz, scales=scale.log()[:, None].repeat(1, 3), quats=quats,
                        features_dc=dc, features_rest=xyz.new_zeros(n, (config.sh_degree+1)**2-1, 3),
                        opacities=torch.logit(xyz.new_full((n, 1), 0.1))), config)

    @property
    def get_xyz(self):
        return self.params["means"]

    @property
    def get_scaling(self):
        return self.params["scales"].exp()

    @property
    def get_rotation(self):
        return torch.nn.functional.normalize(self.params["quats"], dim=-1)

    @property
    def get_opacity(self):
        return self.params["opacities"].sigmoid()

    @property
    def get_features(self):
        return torch.cat((self.params["features_dc"][:, None], self.params["features_rest"]), 1)

    def schedule(self, step, config):
        self.active_sh_degree = min(step//config.sh_interval, config.sh_degree)
        for group in self.optimizer.param_groups:
            if group["name"] == "means":
                group["lr"] = exponential_lr(config.position_lr, config.position_lr_final,
                                               max(step-1, 0), config.lr_max_steps)

    def scale_loss(self, step, config):
        if step % 10 or not config.scale_regularization:
            return self.get_xyz.new_zeros(())
        scales = self.get_scaling
        ratio = scales.amax(-1)/scales.amin(-1).clamp_min(1e-12)
        return config.scale_regularization * (ratio-config.max_scale_ratio).clamp_min(0).mean()

    @torch.no_grad()
    def collect_stats(self, info, config, step):
        if step >= config.stop_split:
            return
        # Faithful to BAD's last virtual view, not the auxiliary depth render.
        xy = info["means2d"]
        if xy.grad is None:
            return
        grad = xy.grad[0].norm(dim=-1)
        radii = info["radii"][0]
        if radii.ndim == 2:
            radii = radii.amax(-1)
        visible = radii > 0
        if self.grad_sum is None:
            self.grad_sum = grad.clone()
            self.vis_count = torch.ones_like(grad)
            self.max_radii = torch.zeros_like(grad)
        else:
            self.grad_sum[visible] += grad[visible]
            self.vis_count[visible] += 1
        self.max_radii[visible] = torch.maximum(self.max_radii[visible],
                                               radii[visible]/max(info["width"], info["height"]))

    @torch.no_grad()
    def replace_parameters(self, values, indices):
        """Migrate Adam moments; index -1 denotes a new Gaussian with zero moments."""
        for group in self.optimizer.param_groups:
            name = group["name"]
            old = self.params[name]
            new = nn.Parameter(values[name].detach())
            state = self.optimizer.state.pop(old, {})
            for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                if key in state:
                    out = torch.zeros_like(new)
                    valid = indices >= 0
                    out[valid] = state[key][indices[valid]]
                    state[key] = out
            self.params[name] = new
            group["params"] = [new]
            if state:
                self.optimizer.state[new] = state

    @torch.no_grad()
    def refine(self, step, config, train_count, width, height):
        if step <= config.warmup or step % config.refine_every:
            return
        do_densify = (step < config.stop_split and
                      step % config.opacity_reset_every > train_count+config.refine_every)
        if do_densify and self.grad_sum is not None:
            high = self.grad_sum/self.vis_count.clamp_min(1) * (0.5*max(width, height))
            high = high > config.densify_grad_threshold/config.num_virtual_views
            size = self.get_scaling.amax(-1)
            splits = size > config.densify_size_threshold
            if step < config.stop_screen_size:
                splits |= self.max_radii > config.split_screen_size
            splits &= high
            duplicates = (size <= config.densify_size_threshold) & high
            split_indices = torch.where(splits)[0].repeat(2)
            dup_indices = torch.where(duplicates)[0]
            new_indices = torch.cat((split_indices, dup_indices))
            values = {k: torch.cat((p.detach(), p.detach()[new_indices]), 0) for k, p in self.params.items()}
            n, ns = len(size), len(split_indices)
            if ns:
                import pypose as pp
                q = self.get_rotation[split_indices][:, [1, 2, 3, 0]]
                offsets = pp.SO3(q).Act(torch.randn_like(values["means"][n:n+ns]) * self.get_scaling[split_indices])
                values["means"][n:n+ns] += offsets
                values["scales"][n:n+ns] -= math.log(1.6)
            radii = torch.cat((self.max_radii, size.new_zeros(len(new_indices))))
            cull = values["opacities"].sigmoid().flatten() < config.cull_alpha
            cull[:n] |= splits
            if step > config.opacity_reset_every:
                cull |= values["scales"].exp().amax(-1) > config.cull_scale
                if step < config.stop_screen_size:
                    cull |= radii > config.cull_screen_size
            if cull.all():
                raise RuntimeError("Refinement would delete every Gaussian; check scene scale/data")
            indices = torch.cat((torch.arange(n, device=size.device),
                                 torch.full((len(new_indices),), -1, device=size.device, dtype=torch.long)))
            self.replace_parameters({k: v[~cull] for k, v in values.items()}, indices[~cull])
        if step < config.stop_split and step % config.opacity_reset_every == config.refine_every:
            opacity = self.params["opacities"]
            opacity.clamp_(max=math.log((2*config.cull_alpha)/(1-2*config.cull_alpha)))
            state = self.optimizer.state.get(opacity, {})
            for key in ("exp_avg", "exp_avg_sq"):
                if key in state:
                    state[key].zero_()
        self.grad_sum = self.vis_count = self.max_radii = None

    @torch.no_grad()
    def save_ply(self, path):
        import numpy as np
        from plyfile import PlyData, PlyElement
        xyz = self.get_xyz.cpu().numpy()
        features = self.get_features.transpose(1, 2).cpu().numpy()
        if self.max_sh_degree == 0:
            features = ((self.params["features_dc"].sigmoid().cpu().numpy()-0.5)
                        / 0.28209479177387814)[..., None]
        dc, rest = features[..., :1].reshape(len(xyz), -1), features[..., 1:].reshape(len(xyz), -1)
        names = ["x", "y", "z", "nx", "ny", "nz"] + [f"f_dc_{i}" for i in range(dc.shape[1])]
        names += [f"f_rest_{i}" for i in range(rest.shape[1])] + ["opacity"]
        names += [f"scale_{i}" for i in range(3)] + [f"rot_{i}" for i in range(4)]
        columns = np.concatenate((xyz, np.zeros_like(xyz), dc, rest,
                                  self.params["opacities"].cpu().numpy(), self.params["scales"].cpu().numpy(),
                                  self.params["quats"].cpu().numpy()), -1)
        array = np.empty(len(xyz), dtype=[(n, "f4") for n in names])
        for i, name in enumerate(names):
            array[name] = columns[:, i]
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        PlyData([PlyElement.describe(array, "vertex")]).write(str(path))
