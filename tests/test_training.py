import json

import torch

from render import render_checkpoint
from scripts.make_synthetic import make_synthetic
from arguments import TrainConfig
from train import Trainer, mix_depth, train


def test_alternation_freezes_correct_parameters_and_resume(tmp_path):
    manifest = make_synthetic(tmp_path / "data", size=16, views=1)
    config = TrainConfig(
        iterations=6, exposure_samples=3, joint_start=4, depth_warmup_steps=0, motion_ramp_steps=1
    )
    trainer = Trainer(str(manifest), config)
    trainer.initialize_motion()
    before_scene = {k: p.clone() for k, p in trainer.scene.named_parameters()}
    before_trajectory = trainer.trajectories[0].controls.clone()
    trainer.step(0, 0)
    for k, p in trainer.scene.named_parameters():
        assert torch.equal(before_scene[k], p)
    assert not torch.equal(before_trajectory, trainer.trajectories[0].controls)
    before_trajectory = trainer.trajectories[0].controls.clone()
    trainer.step(0, 1)
    assert torch.equal(before_trajectory, trainer.trajectories[0].controls)
    assert any(not torch.equal(before_scene[k], p) for k, p in trainer.scene.named_parameters())
    checkpoint = tmp_path / "saved.pt"
    trainer.save(checkpoint, 2)
    saved = torch.load(checkpoint, weights_only=True)
    assert saved["format_version"] == 2
    assert saved["trajectory_model"] == "linear_se3"
    assert saved["trajectories"]["0.controls"].shape == (2, 6)
    resumed = Trainer(str(manifest), config)
    assert resumed.resume(str(checkpoint)) == 2
    a, b = trainer.step(0, 2), resumed.step(0, 2)
    assert abs(a["total"] - b["total"]) < 1e-8
    torch.testing.assert_close(trainer.trajectories[0].controls, resumed.trajectories[0].controls)


def test_scene_and_trajectory_receive_motion_gradients(tmp_path):
    manifest = make_synthetic(tmp_path / "data", size=16, views=1)
    trainer = Trainer(str(manifest), TrainConfig(depth_warmup_steps=0))
    trainer.initialize_motion()
    loss = trainer.objective(0, 2001, "joint")["motion"]
    loss.backward()
    assert trainer.scene.means.grad.abs().sum() > 0
    assert trainer.trajectories[0].controls.grad.abs().sum() > 0


def test_training_loss_falls_and_outputs_render(tmp_path):
    manifest = make_synthetic(tmp_path / "data", size=16, views=1)
    config = TrainConfig(
        iterations=16,
        exposure_samples=3,
        joint_start=12,
        scene_lr=0.02,
        position_lr=0.004,
        motion_ramp_steps=1,
        log_every=16,
        checkpoint_every=8,
    )
    output = tmp_path / "run"
    train(str(manifest), config, str(output))
    rows = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    assert rows[-1]["rgb"] < rows[0]["rgb"]
    assert {r["phase"] for r in rows} == {"trajectory", "geometry", "joint"}
    result = render_checkpoint(
        str(output / "checkpoint.pt"), str(manifest), str(tmp_path / "renders"), evaluate=True
    )
    assert result["mean_psnr"] > 5 and -1 <= result["mean_ssim"] <= 1
    assert (output / "scene.ply").read_text().startswith("ply\n")


def test_depth_warmup_uses_scene_units_and_propagates_gradients():
    gs = torch.tensor([[2.0, 4.0]], requires_grad=True)
    initial = torch.tensor([[1.0, float("nan")]])
    depth = mix_depth(gs, initial, 5, 10)
    torch.testing.assert_close(depth, torch.tensor([[1.5, 4.0]]))
    depth.sum().backward()
    torch.testing.assert_close(gs.grad, torch.tensor([[0.5, 1.0]]))


def test_default_virtual_pose_count_is_independent_of_two_controls(tmp_path):
    manifest = make_synthetic(tmp_path / "data", size=16, views=1)
    trainer = Trainer(str(manifest), TrainConfig())
    trajectory = trainer.trajectories[0]
    trajectory.initialize_from_camera_motion(torch.tensor([0.01, 0.0, 0.0, 0.0, 0.0, 0.0]))
    assert trainer.config.exposure_samples == 10
    torch.testing.assert_close(trainer.times, torch.arange(10) / 9)
    assert trajectory.controls.shape == (2, 6)
    assert trajectory(trainer.times).shape == (10, 4, 4)
    calls = []
    renderer = trainer.renderer

    def recording_renderer(scene, pose, K, height, width):
        calls.append(pose.detach().clone())
        return renderer(scene, pose, K, height, width)

    trainer.renderer = recording_renderer
    trainer.objective(0, 0, "joint")
    # One midpoint depth pass plus ten exposure renders, with no change to sample count.
    assert len(calls) == 11
    torch.testing.assert_close(torch.stack(calls[1:]), trajectory(trainer.times))
