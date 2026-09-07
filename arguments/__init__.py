"""Training configuration and command-line conventions shared by the entry scripts."""

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import yaml


@dataclass
class TrainConfig:
    backend: str = "torch"
    device: str = "cpu"
    iterations: int = 30000
    exposure_samples: int = 9
    trajectory_steps: int = 1
    geometry_steps: int = 1
    joint_start: int = 27000
    joint_lr_scale: float = 0.1
    scene_lr: float = 0.001
    position_lr: float = 0.00016
    trajectory_lr: float = 0.001
    depth_warmup_steps: int = 2000
    motion_ramp_steps: int = 1000
    motion_weight: float = 0.1
    geometry_motion_weight: float = 0.05
    flow_weight: float = 1.0
    magnitude_weight: float = 0.1
    direction_weight: float = 0.01
    magnitude: str = "endpoint"
    acceleration_weight: float = 0.0
    pose_weight: float = 0.01
    dssim_weight: float = 0.2
    alpha_threshold: float = 0.1
    linear_exposure: bool = True
    initialize_motion: bool = True
    gradient_clip: float = 10.0
    checkpoint_every: int = 1000
    log_every: int = 50
    seed: int = 42

    def validate(self):
        if self.iterations <= 0 or self.exposure_samples < 3:
            raise ValueError("Positive iterations and at least three exposure samples are required")
        if self.trajectory_steps < 1 or self.geometry_steps < 1:
            raise ValueError("Both alternating phases need at least one step")
        if self.log_every < 1 or self.checkpoint_every < 1 or self.gradient_clip <= 0:
            raise ValueError("Logging/checkpoint intervals and gradient_clip must be positive")
        if self.magnitude not in {"endpoint", "path"} or not 0 <= self.dssim_weight <= 1:
            raise ValueError("Invalid magnitude mode or DSSIM weight")
        if not 0 <= self.alpha_threshold <= 1 or self.joint_start < 0:
            raise ValueError("Invalid alpha threshold or joint start")
        for key, value in asdict(self).items():
            if (
                key.endswith("weight") or key.endswith("lr") or key.endswith("_steps")
            ) and value < 0:
                raise ValueError(f"{key} must be nonnegative")
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; use configs/smoke.yaml for CPU verification")


def read_config(path: str | None, **overrides) -> TrainConfig:
    options = yaml.safe_load(Path(path).read_text(encoding="utf-8")) if path else {}
    options = dict(options or {})
    options.update({key: value for key, value in overrides.items() if value is not None})
    config = TrainConfig(**options)
    config.validate()
    return config


def resolve_source(source: str | Path) -> str:
    """Accept an explicit manifest or a prepared scene directory, never raw COLMAP."""
    path = Path(source).expanduser()
    if path.is_dir():
        for name in ("scene.motion.json", "scene.json"):
            candidate = path / name
            if candidate.is_file():
                return str(candidate.resolve())
        raise FileNotFoundError(
            f"No scene.motion.json or scene.json in {path}. "
            "Prepare the scene with scripts/import_colmap.py and scripts/prepare_motion.py."
        )
    if not path.is_file():
        raise FileNotFoundError(f"Scene manifest not found: {path}")
    return str(path.resolve())


def add_source_argument(parser, *, required=True):
    parser.add_argument(
        "-s",
        "--source_path",
        "--source-path",
        "--data",
        dest="source_path",
        required=required,
        help="Prepared scene directory or explicit scene JSON manifest",
    )


def configure_cpu_threads(device):
    if device == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))


def training_parser():
    parser = argparse.ArgumentParser(description="Train BLUR-GS geometry and linear trajectories")
    add_source_argument(parser)
    parser.add_argument(
        "-m",
        "--model_path",
        "--model-path",
        "--output",
        dest="model_path",
        required=True,
        help="Training output directory",
    )
    parser.add_argument(
        "--config",
        help="YAML configuration; use configs/default.yaml for CUDA or smoke.yaml for CPU",
    )
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--resume")
    parser.add_argument("--device", help="Override config device, e.g. cpu or cuda:0")
    parser.add_argument("--backend", choices=("torch", "gsplat"))
    parser.add_argument("--exposure-samples", "--exposure_samples", type=int)
    return parser


def render_parser(*, evaluate=False):
    parser = argparse.ArgumentParser(
        description="Evaluate BLUR-GS against sharp references"
        if evaluate
        else "Render sharp BLUR-GS views"
    )
    add_source_argument(parser, required=False)
    checkpoints = parser.add_mutually_exclusive_group(required=True)
    checkpoints.add_argument(
        "-m",
        "--model_path",
        "--model-path",
        dest="model_path",
        help="Training output directory containing checkpoint.pt",
    )
    checkpoints.add_argument(
        "--checkpoint", help="Explicit checkpoint, including intermediate steps"
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Render output directory; defaults to renders/ or evaluation/ beside checkpoint",
    )
    parser.add_argument("--backend", choices=("torch", "gsplat"), default="torch")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--split", choices=("train", "test", "all"), default="all")
    return parser
