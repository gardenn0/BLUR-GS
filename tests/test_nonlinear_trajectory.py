import pytest
import torch

from arguments import TrainConfig, training_parser
from scene.trajectory import make_trajectory, checkpoint_midpoint
from scripts.make_synthetic import make_synthetic
from train import Trainer


@pytest.mark.parametrize("kind", ["linear", "spline", "ode"])
def test_initialization_queries_and_pose_gradients(kind):
    base = torch.eye(4, dtype=torch.float64)
    motion = base.new_tensor([0.02, -0.03, 0.01, 0.01, 0.02, -0.01])
    trajectory = make_trajectory(base, kind, 8)
    trajectory.initialize_from_camera_motion(motion)
    times = base.new_tensor([0.0, 0.23, 0.5, 1.0])
    expected = (0.5 - times[:, None]) * motion
    torch.testing.assert_close(trajectory.twists(times), expected)
    torch.testing.assert_close(trajectory(times)[2], trajectory.at(0.5))
    trajectory(times)[:, :3].square().sum().backward()
    grads = [p.grad for p in trajectory.parameters() if p.grad is not None]
    assert all(torch.isfinite(g).all() for g in grads)
    assert sum(g.abs().sum() for g in grads) > 0


def test_spline_curvature_and_exact_acceleration():
    trajectory = make_trajectory(torch.eye(4, dtype=torch.float64), "spline")
    with torch.no_grad():
        trajectory.controls[:, 0] = torch.tensor([0.0, 0.0, 1.0, 1.0])
    values = trajectory.twists(torch.tensor([0.0, 0.25, 0.5, 1.0], dtype=torch.float64))
    torch.testing.assert_close(
        values[:, 0], torch.tensor([0.0, 0.15625, 0.5, 1.0], dtype=torch.float64)
    )
    acceleration, _ = trajectory.regularizers()
    torch.testing.assert_close(acceleration, torch.tensor(2.0, dtype=torch.float64))
    acceleration.backward()
    assert trajectory.controls.grad.abs().sum() > 0


def test_neural_ode_matches_analytic_vector_field_and_converges():
    from torch import nn

    class ExponentialField(nn.Module):
        def forward(self, inputs):
            return inputs[:, 1:]

    errors = []
    for steps in (1, 8):
        trajectory = make_trajectory(torch.eye(4, dtype=torch.float64), "ode", steps)
        trajectory.field = ExponentialField()
        with torch.no_grad():
            trajectory.initial_twist.fill_(0.1)
        times = torch.tensor([1.0, 0.5, 0.0, 0.5], dtype=torch.float64)
        result = trajectory.twists(times)
        expected = 0.1 * times.exp()[:, None].expand(-1, 6)
        errors.append((result - expected).abs().max())
        torch.testing.assert_close(result[1], result[3])
        acceleration, _ = trajectory.regularizers()
        assert acceleration > 0
    assert errors[1] < errors[0] / 100
    assert errors[1] < 1e-6


@pytest.mark.parametrize("kind", ["spline", "ode"])
def test_nonlinear_training_freezing_checkpoint_and_resume(tmp_path, kind):
    manifest = make_synthetic(tmp_path / "data", size=16, views=1)
    config = TrainConfig(
        trajectory=kind,
        ode_steps=4,
        exposure_samples=3,
        depth_warmup_steps=0,
        motion_ramp_steps=1,
        acceleration_weight=0.01,
        densify_every=0,
        joint_start=2,
    )
    trainer = Trainer(str(manifest), config)
    trainer.initialize_motion()
    before = [p.detach().clone() for p in trainer.scene.parameters()]
    trainer.step(0, 0)
    assert all(torch.equal(a, b) for a, b in zip(before, trainer.scene.parameters()))
    before = [p.detach().clone() for p in trainer.trajectories.parameters()]
    trainer.step(0, 1)
    assert all(torch.equal(a, b) for a, b in zip(before, trainer.trajectories.parameters()))
    path = tmp_path / "checkpoint.pt"
    trainer.save(path, 2)
    saved = torch.load(path, weights_only=True)
    assert saved["format_version"] == 3
    with torch.no_grad():
        torch.testing.assert_close(checkpoint_midpoint(saved, 0), trainer.trajectories[0].at(0.5))
    from render import render_checkpoint

    rendered = render_checkpoint(str(path), str(manifest), str(tmp_path / "renders"))
    assert len(rendered["frames"]) == 1
    restored = Trainer(str(manifest), config)
    assert restored.resume(str(path)) == 2
    a, b = trainer.step(0, 2), restored.step(0, 2)
    assert abs(a["total"] - b["total"]) < 1e-7
    for a, b in zip(trainer.trajectories.parameters(), restored.trajectories.parameters()):
        torch.testing.assert_close(a, b)
    other = Trainer(str(manifest), TrainConfig())
    with pytest.raises(ValueError, match="same trajectory model"):
        other.resume(str(path))


def test_cli_trajectory_options():
    args = training_parser().parse_args(
        [
            "-s",
            "data",
            "-m",
            "run",
            "--trajectory",
            "ode",
            "--ode-steps",
            "8",
            "--acceleration-weight",
            "0.01",
        ]
    )
    assert (args.trajectory, args.ode_steps, args.acceleration_weight) == ("ode", 8, 0.01)
