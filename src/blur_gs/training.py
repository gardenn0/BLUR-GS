"""Alternating geometry/trajectory optimization from BLUR-GS Algorithm 1."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import yaml
from torch import nn

from .data import load_dataset, save_image
from .geometry import exposure_path, solve_camera_motion
from .losses import motion_loss, rgb_loss
from .rendering import make_renderer, render_blur
from .scene import GaussianScene
from .trajectory import (
    CHECKPOINT_VERSION,
    TRAJECTORY_MODEL,
    ExposureTrajectory,
    require_linear_checkpoint,
)


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


def read_config(path: str | None) -> TrainConfig:
    options = yaml.safe_load(Path(path).read_text(encoding="utf-8")) if path else {}
    config = TrainConfig(**(options or {}))
    config.validate()
    return config


def phase_at(step: int, config: TrainConfig) -> str:
    if step >= config.joint_start:
        return "joint"
    cycle = config.trajectory_steps + config.geometry_steps
    return "trajectory" if step % cycle < config.trajectory_steps else "geometry"


def mix_depth(gs_depth, initial, step: int, steps: int):
    if initial is None or steps <= 0:
        return gs_depth
    eta = min(1.0, step / steps)
    valid = torch.isfinite(initial) & (initial > 0)
    initial = torch.where(valid, initial, gs_depth)
    return (1 - eta) * initial + eta * gs_depth


class Trainer:
    def __init__(self, manifest: str, config: TrainConfig):
        config.validate()
        torch.manual_seed(config.seed)
        self.config = config
        self.manifest = str(Path(manifest).resolve())
        self.frames, cloud = load_dataset(manifest)
        self.scene = GaussianScene(**{k: v.to(config.device) for k, v in cloud.items()})
        self.trajectories = nn.ModuleList(
            [ExposureTrajectory(f.w2c.to(config.device)) for f in self.frames]
        )
        self.renderer = make_renderer(config.backend)
        self.scene_optimizer = torch.optim.Adam(
            [
                {"params": [self.scene.means], "lr": config.position_lr},
                {
                    "params": [p for name, p in self.scene.named_parameters() if name != "means"],
                    "lr": config.scene_lr,
                },
            ],
            eps=1e-15,
        )
        self.trajectory_optimizer = torch.optim.Adam(
            self.trajectories.parameters(), lr=config.trajectory_lr
        )
        self.times = torch.linspace(0, 1, config.exposure_samples, device=config.device)
        self.initialization_notes = []

    @torch.no_grad()
    def initialize_motion(self):
        for frame, trajectory in zip(self.frames, self.trajectories):
            frame = frame.to(self.config.device)
            rendered = self.renderer(
                self.scene, trajectory.at(0.5), frame.K, *frame.image.shape[:2]
            )
            depth = mix_depth(
                rendered.depth, frame.initial_depth, 0, self.config.depth_warmup_steps
            )
            confidence = frame.confidence * (rendered.alpha > self.config.alpha_threshold)
            try:
                motion = solve_camera_motion(frame.flow, depth, frame.K, confidence)
                # Approximate small-motion initialization; subsequent losses align the start grid exactly.
                if (
                    not torch.isfinite(motion).all()
                    or motion[:3].norm() > depth.mean()
                    or motion[3:].norm() > 0.5
                ):
                    raise ValueError("Motion initialization exceeds small-motion range")
                trajectory.initialize_from_camera_motion(motion)
            except ValueError as exc:
                self.initialization_notes.append(
                    f"{frame.name}: zero-motion initialization ({exc})"
                )

    def objective(self, frame_index: int, step: int, phase: str):
        config = self.config
        frame = self.frames[frame_index].to(config.device)
        trajectory = self.trajectories[frame_index]
        poses, reference = trajectory(self.times), trajectory.at(0.5)
        h, w = frame.image.shape[:2]
        ref = self.renderer(self.scene, reference, frame.K, h, w)
        depth = ref.depth.detach() if phase == "trajectory" else ref.depth
        depth = mix_depth(depth, frame.initial_depth, step, config.depth_warmup_steps)
        path, path_valid = exposure_path(depth, frame.K, reference, poses)
        m = motion_loss(
            path,
            path_valid,
            frame.flow,
            frame.confidence,
            (ref.alpha.detach() > config.alpha_threshold) & (depth.detach() > 0),
            reference=frame.flow_reference,
            ambiguous=frame.sign_ambiguous,
            flow_weight=config.flow_weight,
            magnitude_weight=config.magnitude_weight,
            direction_weight=config.direction_weight,
            magnitude=config.magnitude,
        )
        blurred = render_blur(
            self.renderer, self.scene, poses, frame.K, h, w, linear_exposure=config.linear_exposure
        )
        rgb = rgb_loss(blurred, frame.image, config.dssim_weight)
        acceleration, anchor = trajectory.regularizers()
        weight = config.geometry_motion_weight if phase == "geometry" else config.motion_weight
        weight *= min(1.0, (step + 1) / max(config.motion_ramp_steps, 1))
        total = rgb + weight * m["motion"]
        if phase != "geometry":
            total = total + config.acceleration_weight * acceleration + config.pose_weight * anchor
        return {"total": total, "rgb": rgb, **m, "acceleration": acceleration, "anchor": anchor}

    def step(self, frame_index: int, step: int) -> dict:
        phase = phase_at(step, self.config)
        self.scene.requires_grad_(phase != "trajectory")
        self.trajectories.requires_grad_(phase != "geometry")
        self.scene_optimizer.zero_grad(set_to_none=True)
        self.trajectory_optimizer.zero_grad(set_to_none=True)
        multiplier = self.config.joint_lr_scale if phase == "joint" else 1.0
        for group, lr in zip(
            self.scene_optimizer.param_groups, (self.config.position_lr, self.config.scene_lr)
        ):
            group["lr"] = multiplier * lr
        self.trajectory_optimizer.param_groups[0]["lr"] = multiplier * self.config.trajectory_lr
        losses = self.objective(frame_index, step, phase)
        if not torch.isfinite(losses["total"]):
            raise FloatingPointError(f"Non-finite loss at step {step}")
        losses["total"].backward()
        parameters = [
            p
            for p in list(self.scene.parameters()) + list(self.trajectories.parameters())
            if p.requires_grad
        ]
        nn.utils.clip_grad_norm_(parameters, self.config.gradient_clip, error_if_nonfinite=True)
        if phase != "trajectory":
            self.scene_optimizer.step()
        if phase != "geometry":
            self.trajectory_optimizer.step()
        return {
            "step": step + 1,
            "phase": phase,
            "frame": self.frames[frame_index].name,
            **{k: float(v.detach()) for k, v in losses.items()},
        }

    def save(self, path: Path, step: int):
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "format_version": CHECKPOINT_VERSION,
            "trajectory_model": TRAJECTORY_MODEL,
            "step": step,
            "config": asdict(self.config),
            "scene": self.scene.state_dict(),
            "trajectories": self.trajectories.state_dict(),
            "frame_names": [f.name for f in self.frames],
            "manifest": self.manifest,
            "scene_optimizer": self.scene_optimizer.state_dict(),
            "trajectory_optimizer": self.trajectory_optimizer.state_dict(),
        }
        torch.save(checkpoint, path)

    def resume(self, path: str) -> int:
        checkpoint = torch.load(path, map_location=self.config.device, weights_only=True)
        require_linear_checkpoint(checkpoint)
        if checkpoint["frame_names"] != [f.name for f in self.frames]:
            raise ValueError("Resume requires the same training frames in the same order")
        previous = checkpoint["config"]
        allowed = {"iterations", "log_every", "checkpoint_every", "device"}
        for key, value in asdict(self.config).items():
            if key not in allowed and previous[key] != value:
                raise ValueError(
                    f"Resume config changed {key}; keep optimization settings identical"
                )
        self.scene.load_state_dict(checkpoint["scene"])
        self.trajectories.load_state_dict(checkpoint["trajectories"])
        self.scene_optimizer.load_state_dict(checkpoint["scene_optimizer"])
        self.trajectory_optimizer.load_state_dict(checkpoint["trajectory_optimizer"])
        return int(checkpoint["step"])


def train(manifest: str, config: TrainConfig, output: str, resume: str | None = None) -> dict:
    directory = Path(output)
    if directory.exists() and any(directory.iterdir()) and resume is None:
        raise FileExistsError(
            f"Output directory is not empty: {directory}; use a new run or --resume"
        )
    trainer = Trainer(manifest, config)
    start = trainer.resume(resume) if resume else 0
    if start >= config.iterations:
        raise ValueError("iterations must exceed checkpoint step")
    if not resume and config.initialize_motion:
        trainer.initialize_motion()
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(
        json.dumps(asdict(config), indent=2) + "\n", encoding="utf-8"
    )
    with (directory / "metrics.jsonl").open("a" if resume else "w", encoding="utf-8") as stream:
        for step in range(start, config.iterations):
            # Every view receives both trajectory and geometry updates each cycle.
            cycle = config.trajectory_steps + config.geometry_steps
            index = ((step // cycle) if step < config.joint_start else step) % len(trainer.frames)
            metrics = trainer.step(index, step)
            stream.write(json.dumps(metrics) + "\n")
            if (step + 1) % config.log_every == 0 or step == start:
                print(json.dumps(metrics), flush=True)
                stream.flush()
            if (step + 1) % config.checkpoint_every == 0:
                trainer.save(directory / f"checkpoint_{step + 1:06d}.pt", step + 1)
    trainer.save(directory / "checkpoint.pt", config.iterations)
    trainer.scene.export_ply(directory / "scene.ply")
    with torch.no_grad():
        for i, frame in enumerate(trainer.frames):
            frame = frame.to(config.device)
            rendered = trainer.renderer(
                trainer.scene, trainer.trajectories[i].at(0.5), frame.K, *frame.image.shape[:2]
            )
            save_image(directory / "sharp" / f"{i:06d}.png", rendered.rgb)
    summary = {
        "trajectory_model": TRAJECTORY_MODEL,
        "exposure_samples": config.exposure_samples,
        "iterations": config.iterations,
        "final_step_metrics": metrics,
        "initialization_notes": trainer.initialization_notes,
        "backend": config.backend,
        "gaussians": len(trainer.scene.means),
    }
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
