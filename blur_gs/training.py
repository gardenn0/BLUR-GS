"""Training controller. train.py supplies the prepared cache unless --baseline is set."""
from dataclasses import dataclass, asdict
import torch
from .cache import FlowCache
from .depth import render_z_depth
from .loss import blur_flow_loss


@dataclass
class BlurConfig:
    flow_cache: str = ""
    flow_weight: float = 0.01
    geometry_flow_weight: float = 0.01
    flow_start: int = 4000
    geometry_start: int = 20000
    flow_ramp: int = 2000
    flow_mode: str = "alternating"
    alternate_every: int = 50
    flow_direction: str = "auto"
    flow_min_alpha: float = 0.5
    flow_occlusion_tolerance: float = 0.1
    flow_min_valid_fraction: float = 0.05

    @property
    def enabled(self):
        return bool(self.flow_cache)


def add_arguments(parser):
    group = parser.add_argument_group("BLUR-GS")
    for name, value in asdict(BlurConfig()).items():
        kwargs = dict(default=value, type=type(value))
        if name == "flow_mode":
            kwargs["choices"] = ("trajectory", "alternating", "joint")
        if name == "flow_direction":
            kwargs["choices"] = ("auto", "forward", "backward")
        group.add_argument("--" + name, **kwargs)


def config_from_args(args):
    return BlurConfig(**{k: getattr(args, k) for k in asdict(BlurConfig())})


def phase_at(iteration, config):
    if not config.enabled or iteration <= config.flow_start:
        return "baseline"
    if config.flow_mode == "trajectory" or iteration <= config.geometry_start:
        return "trajectory_prior"
    if config.flow_mode == "joint":
        return "joint"
    block = (iteration - config.geometry_start - 1) // config.alternate_every
    return "trajectory" if block % 2 == 0 else "geometry"


def set_optimizer_grad(optimizer, enabled):
    for group in optimizer.param_groups:
        for param in group["params"]:
            param.requires_grad_(enabled)


class BlurSupervisor:
    def __init__(self, config, cameras, start_warp):
        self.config = config
        if config.enabled:
            if config.flow_start < start_warp or config.geometry_start < config.flow_start:
                raise ValueError("Require start_warp <= flow_start <= geometry_start")
            if config.alternate_every < 1 or config.flow_ramp < 0:
                raise ValueError("Invalid alternation/ramp length")
            if min(config.flow_weight, config.geometry_flow_weight) < 0:
                raise ValueError("Flow loss weights must be nonnegative")
            if not 0 < config.flow_min_alpha <= 1 or not 0 < config.flow_min_valid_fraction <= 1:
                raise ValueError("Alpha and valid-fraction thresholds must be in (0,1]")
            self.cache = FlowCache(config.flow_cache, [c.image_name for c in cameras])
        self.phase = "baseline"

    def begin(self, iteration, gaussians, kernel):
        self.phase = phase_at(iteration, self.config)
        self.update_geometry = self.phase != "trajectory"
        self.update_kernel = self.phase != "geometry"
        if self.config.enabled:
            gaussians.optimizer.zero_grad(set_to_none=True)
            kernel.optimizer.zero_grad(set_to_none=True)
            set_optimizer_grad(gaussians.optimizer, self.update_geometry)
            set_optimizer_grad(kernel.optimizer, self.update_kernel)

    def loss(self, iteration, view, warped_cameras, gaussians, pipe, kernel_size):
        zero = gaussians.get_xyz.new_zeros(())
        if self.phase == "baseline":
            return zero, {}
        if len(warped_cameras) < 2:
            raise ValueError("BLUR-GS requires at least two exposure samples")
        observed, mask, crop = self.cache.get(view.image_name, gaussians.get_xyz.device)
        if abs(view.image_width / view.image_height - crop["original_width"] / crop["original_height"]) > 0.01:
            raise ValueError(f"Cache/source aspect ratio mismatch for {view.image_name}")
        geometry_grad = self.phase in ("geometry", "joint")
        depths = [render_z_depth(c, gaussians, pipe, kernel_size, geometry_grad=geometry_grad,
                                 min_alpha=self.config.flow_min_alpha)
                  for c in (warped_cameras[0], warped_cameras[-1])]
        value, stats = blur_flow_loss(warped_cameras, depths, observed, mask, crop,
                                     min_alpha=self.config.flow_min_alpha,
                                     occlusion_tolerance=self.config.flow_occlusion_tolerance,
                                     direction=self.config.flow_direction,
                                     min_valid_fraction=self.config.flow_min_valid_fraction)
        ramp = min(1., (iteration - self.config.flow_start) / max(1, self.config.flow_ramp))
        weight = self.config.geometry_flow_weight if self.phase == "geometry" else self.config.flow_weight
        stats.update(raw_loss=float(value.detach()), weight=weight * ramp)
        return weight * ramp * value, stats
