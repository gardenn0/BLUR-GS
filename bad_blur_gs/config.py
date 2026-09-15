"""Explicit BAD reproduction settings; upstream provenance is in docs/BAD_BLUR_GS.md."""
from dataclasses import asdict, dataclass
import math


@dataclass
class BadConfig:
    # Zero-based steps 0..30000, matching the upstream trainer's range(30001).
    iterations: int = 30001
    trajectory: str = "linear"
    num_virtual_views: int = 10
    initial_noise: float = 1e-5
    camera_accumulation: int = 25
    camera_lr: float = 1e-3
    camera_lr_final: float = 1e-5
    position_lr: float = 1.6e-4
    position_lr_final: float = 1.6e-6
    lr_max_steps: int = 30000
    feature_lr: float = 0.0025
    opacity_lr: float = 0.05
    scaling_lr: float = 0.005
    rotation_lr: float = 0.001
    sh_degree: int = 3
    sh_interval: int = 1000
    ssim_lambda: float = 0.2
    scale_regularization: float = 0.1
    max_scale_ratio: float = 10.0
    warmup: int = 500
    refine_every: int = 100
    stop_split: int = 15000
    opacity_reset_every: int = 3000
    densify_grad_threshold: float = 4e-4
    densify_size_threshold: float = 0.01
    cull_alpha: float = 0.005
    cull_scale: float = 0.5
    split_screen_size: float = 0.05
    cull_screen_size: float = 0.15
    stop_screen_size: int = 4000
    num_downscales: int = 0
    resolution_schedule: int = 250
    scene_scale: float = 0.25
    rasterize_mode: str = "antialiased"
    background: str = "random"
    eval_every: int = 500
    save_every: int = 2000

    def validate(self):
        positive = ("iterations", "camera_accumulation", "lr_max_steps", "sh_interval",
                    "refine_every", "opacity_reset_every", "resolution_schedule", "scene_scale",
                    "initial_noise", "max_scale_ratio", "densify_size_threshold", "cull_scale")
        for name in positive:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name, value in asdict(self).items():
            if isinstance(value, (float, int)) and (not math.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.num_virtual_views < 2 or not 0 <= self.sh_degree <= 3:
            raise ValueError("Require num_virtual_views >= 2 and sh_degree in [0,3]")
        if self.opacity_reset_every % self.refine_every:
            raise ValueError("opacity_reset_every must be divisible by refine_every")
        if not 0 < self.cull_alpha < 0.5 or not 0 <= self.ssim_lambda <= 1:
            raise ValueError("Invalid opacity/SSIM thresholds")
        if self.trajectory not in ("linear", "cubic"):
            raise ValueError("trajectory must be linear or cubic")
        for name in ("position_lr", "position_lr_final", "camera_lr", "camera_lr_final"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")


def add_arguments(parser):
    group = parser.add_argument_group("BAD-style reconstruction")
    choices = {"trajectory": ("linear", "cubic"), "rasterize_mode": ("classic", "antialiased"),
               "background": ("random", "black", "white")}
    for key, value in asdict(BadConfig()).items():
        options = dict(type=type(value), default=value)
        if key in choices:
            options["choices"] = choices[key]
        group.add_argument("--" + key, **options)


def from_args(args):
    result = BadConfig(**{k: getattr(args, k) for k in asdict(BadConfig())})
    result.validate()
    return result


def exponential_lr(initial, final, step, max_steps):
    t = min(max(step / max_steps, 0.0), 1.0)
    return math.exp(math.log(initial) * (1 - t) + math.log(final) * t)
