"""Alternating geometry/trajectory optimization from BLUR-GS Algorithm 1."""

import json
from dataclasses import asdict
from pathlib import Path
import warnings

import torch
from torch import nn

from arguments import (
    TrainConfig,
    configure_cpu_threads,
    read_config,
    resolve_source,
    training_parser,
)
from scene import load_dataset
from utils.image_utils import save_image
from utils.pose_utils import exposure_path, solve_camera_motion
from utils.loss_utils import rgb_loss
from utils.motion_loss_utils import motion_loss
from utils.training_monitor import TrainingMonitor
from gaussian_renderer import make_renderer
from gaussian_renderer.blur_renderer import render_blur
from scene.gaussian_model import GaussianScene
from scene.trajectory import (
    CHECKPOINT_VERSION,
    make_trajectory,
    require_linear_checkpoint,
)


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
            [
                make_trajectory(f.w2c.to(config.device), config.trajectory, config.ode_steps)
                for f in self.frames
            ]
        )
        self.renderer = make_renderer(config.backend)
        self.scene_optimizer = self._make_scene_optimizer()
        self.trajectory_optimizer = torch.optim.Adam(
            self.trajectories.parameters(), lr=config.trajectory_lr
        )
        self.times = torch.linspace(0, 1, config.exposure_samples, device=config.device)
        self.initialization_notes = []
        self.refinement_events = []
        self._reset_refinement_stats()

    def _reset_refinement_stats(self):
        self.refinement_gradient_sum = self.scene.means.new_zeros(len(self.scene.means))
        self.refinement_gradient_count = torch.zeros(
            len(self.scene.means), device=self.scene.means.device, dtype=torch.long
        )

    @torch.no_grad()
    def accumulate_refinement_stats(self, step: int, phase: str):
        """Accumulate pre-clipping world-space gradient norms between refinement checks.

        Nonzero finite gradients are the observation proxy for this renderer API;
        invisible/zero-gradient views do not dilute a Gaussian's average score.
        """
        if (
            phase == "trajectory"
            or self.config.densify_every == 0
            or step + 1 > self.config.densify_until
        ):
            return
        gradient = self.scene.means.grad
        if gradient is not None:
            score = torch.linalg.vector_norm(gradient, dim=-1)
            observed = torch.isfinite(score) & (score > 0)
            self.refinement_gradient_sum += torch.where(observed, score, 0)
            self.refinement_gradient_count += observed.long()

    def _make_scene_optimizer(self):
        config = self.config
        return torch.optim.Adam(
            [
                {"params": [self.scene.means], "lr": config.position_lr},
                {
                    "params": [p for name, p in self.scene.named_parameters() if name != "means"],
                    "lr": config.scene_lr,
                },
            ],
            eps=1e-15,
        )

    @torch.no_grad()
    def maybe_refine_scene(self, step: int, phase: str) -> dict | None:
        config = self.config
        if (
            phase == "trajectory"
            or config.densify_every == 0
            or step + 1 < config.densify_from
            or step + 1 > config.densify_until
            or (step + 1) % config.densify_every
        ):
            return None
        score = self.refinement_gradient_sum / self.refinement_gradient_count.clamp_min(1)
        keep = self.scene.opacities >= config.prune_opacity_threshold
        if not keep.any():
            keep[self.scene.opacities.argmax()] = True
        capacity = max(0, config.max_gaussians - int(keep.sum()))
        candidates = (
            keep
            & (self.refinement_gradient_count > 0)
            & torch.isfinite(score)
            & (score >= config.densify_grad_threshold)
        )
        selected = candidates.nonzero(as_tuple=False).squeeze(1)
        if len(selected) > capacity:
            selected = (
                selected[torch.topk(score[selected], capacity).indices]
                if capacity
                else selected[:0]
            )
        clone = torch.zeros_like(keep)
        clone[selected] = True
        if keep.all() and not clone.any():
            self._reset_refinement_stats()
            return None
        event = self.scene.refine(
            clone, keep, config.densify_jitter, optimizer=self.scene_optimizer
        )
        event["step"] = step + 1
        self._reset_refinement_stats()
        self.refinement_events.append(event)
        return event

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
        self.accumulate_refinement_stats(step, phase)
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
        refinement = self.maybe_refine_scene(step, phase)
        return {
            "step": step + 1,
            "phase": phase,
            "frame": self.frames[frame_index].name,
            **{k: float(v.detach()) for k, v in losses.items()},
            "gaussians": len(self.scene.means),
            "refinement": refinement,
        }

    def save(self, path: Path, step: int):
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "format_version": CHECKPOINT_VERSION if self.config.trajectory == "linear" else 3,
            "trajectory_model": self.trajectories[0].model_name,
            "step": step,
            "config": asdict(self.config),
            "scene": self.scene.state_dict(),
            "trajectories": self.trajectories.state_dict(),
            "frame_names": [f.name for f in self.frames],
            "manifest": self.manifest,
            "scene_optimizer": self.scene_optimizer.state_dict(),
            "trajectory_optimizer": self.trajectory_optimizer.state_dict(),
            "refinement_events": self.refinement_events,
            "initialization_notes": self.initialization_notes,
            "refinement_stats": {
                "gradient_sum": self.refinement_gradient_sum,
                "gradient_count": self.refinement_gradient_count,
            },
            "rng_state": {
                "cpu": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state(self.config.device)
                if torch.device(self.config.device).type == "cuda"
                else None,
            },
        }
        torch.save(checkpoint, path)

    def resume(self, path: str) -> int:
        checkpoint = torch.load(path, map_location=self.config.device, weights_only=True)
        if checkpoint.get("format_version") != 3:
            require_linear_checkpoint(checkpoint)
        if checkpoint.get("trajectory_model") != self.trajectories[0].model_name:
            raise ValueError(
                "Resume requires the same trajectory model; start a new run to change it"
            )
        if checkpoint["frame_names"] != [f.name for f in self.frames]:
            raise ValueError("Resume requires the same training frames in the same order")
        previous = asdict(TrainConfig(**checkpoint["config"]))
        allowed = {
            "iterations",
            "log_every",
            "checkpoint_every",
            "device",
            "eval",
            "eval_every",
            "test_iterations",
            "tensorboard",
            "eval_test_images_as_sharp",
        }
        for key, value in asdict(self.config).items():
            if key not in allowed and previous[key] != value:
                raise ValueError(
                    f"Resume config changed {key}; keep optimization settings identical"
                )
        state = checkpoint["scene"]
        # A refined checkpoint can contain a different Gaussian count than the
        # original COLMAP cloud, so reconstruct parameters before restoring Adam.
        self.scene = GaussianScene(
            state["means"], state["color_logits"].sigmoid(), state["log_scales"].exp()
        ).to(self.config.device)
        self.scene.load_state_dict(state)
        self.scene_optimizer = self._make_scene_optimizer()
        self.trajectories.load_state_dict(checkpoint["trajectories"])
        self.scene_optimizer.load_state_dict(checkpoint["scene_optimizer"])
        self.trajectory_optimizer.load_state_dict(checkpoint["trajectory_optimizer"])
        self.refinement_events = checkpoint.get("refinement_events", [])
        self.initialization_notes = checkpoint.get("initialization_notes", [])
        self._reset_refinement_stats()
        stats = checkpoint.get("refinement_stats")
        if stats is not None:
            for key, destination in (
                ("gradient_sum", self.refinement_gradient_sum),
                ("gradient_count", self.refinement_gradient_count),
            ):
                if stats[key].shape != destination.shape:
                    raise ValueError("Checkpoint refinement statistics must match Gaussian count")
                destination.copy_(stats[key])
        rng = checkpoint.get("rng_state")
        if rng is not None:
            # map_location may have moved RNG ByteTensors to CUDA; these APIs expect CPU state.
            torch.set_rng_state(rng["cpu"].cpu())
            if torch.device(self.config.device).type == "cuda" and rng.get("cuda") is not None:
                torch.cuda.set_rng_state(rng["cuda"].cpu(), self.config.device)
        if (
            (rng is None or stats is None)
            and self.config.densify_every > 0
            and checkpoint["step"] < self.config.densify_until
        ):
            warnings.warn(
                "Legacy checkpoint lacks refinement/RNG state; loading succeeds, but future "
                "splits cannot exactly reproduce uninterrupted training.",
                RuntimeWarning,
                stacklevel=2,
            )
        return int(checkpoint["step"])


def train(manifest: str, config: TrainConfig, output: str, resume: str | None = None) -> dict:
    directory = Path(output)
    if directory.exists() and any(directory.iterdir()) and resume is None:
        raise FileExistsError(
            f"Output directory is not empty: {directory}; use a new run or --resume"
        )
    config.validate()
    monitor = TrainingMonitor(manifest, config)
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
    with (
        monitor.start(directory, start),
        (directory / "metrics.jsonl").open("a" if resume else "w", encoding="utf-8") as stream,
    ):
        for step in range(start, config.iterations):
            # Every view receives both trajectory and geometry updates each cycle.
            cycle = config.trajectory_steps + config.geometry_steps
            index = ((step // cycle) if step < config.joint_start else step) % len(trainer.frames)
            metrics = trainer.step(index, step)
            stream.write(json.dumps(metrics) + "\n")
            if (step + 1) % config.log_every == 0 or step == start or step + 1 == config.iterations:
                print(json.dumps(metrics), flush=True)
                stream.flush()
                monitor.log_train(metrics)
            if (step + 1) % config.checkpoint_every == 0:
                trainer.save(directory / f"checkpoint_{step + 1:06d}.pt", step + 1)
            if monitor.should_evaluate(step + 1):
                evaluation = monitor.evaluate(trainer, step + 1)
                print(json.dumps({"evaluation": evaluation}), flush=True)
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
        "trajectory_model": trainer.trajectories[0].model_name,
        "exposure_samples": config.exposure_samples,
        "iterations": config.iterations,
        "final_step_metrics": metrics,
        "initialization_notes": trainer.initialization_notes,
        "refinement_events": trainer.refinement_events,
        "backend": config.backend,
        "gaussians": len(trainer.scene.means),
    }
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main(argv=None):
    args = training_parser().parse_args(argv)
    config = read_config(
        args.config,
        iterations=args.iterations,
        device=args.device,
        backend=args.backend,
        exposure_samples=args.exposure_samples,
        trajectory=args.trajectory,
        ode_steps=args.ode_steps,
        acceleration_weight=args.acceleration_weight,
        eval=args.eval,
        eval_every=args.eval_every,
        test_iterations=args.test_iterations,
        tensorboard=args.tensorboard,
        eval_test_images_as_sharp=args.eval_test_images_as_sharp,
    )
    configure_cpu_threads(config.device)
    result = train(resolve_source(args.source_path), config, args.model_path, args.resume)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
