"""Regression coverage for rotated splits, multi-view scores, and resume continuity."""

import json
import math

import pytest
import torch

from arguments import TrainConfig
from scene.gaussian_model import GaussianScene
from scripts.make_synthetic import make_synthetic
from train import Trainer, phase_at
from utils.pose_utils import quaternion_matrix


def test_split_respects_rotated_anisotropic_covariance():
    points = torch.tensor([[0.1, -0.2, 2.0]])
    colors = torch.full_like(points, 0.5)
    scales = torch.tensor([[1.0, 0.01, 0.03]])
    plain = GaussianScene(points, colors, scales)
    rotated = GaussianScene(points, colors, scales)
    with torch.no_grad():
        rotated.quats[0] = torch.tensor([math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)])
    rotation = quaternion_matrix(rotated.quats)[0].detach()
    parent_covariance = rotated.covariance()[0].detach().clone()
    mask = torch.tensor([True])
    for scene in (plain, rotated):
        torch.manual_seed(42)
        scene.refine(mask, mask, jitter=0.5)
    offset = rotated.means - points
    torch.testing.assert_close(offset, (plain.means - points) @ rotation.T)
    torch.testing.assert_close(rotated.means.mean(0), points[0])
    # Each child lies on the parent's 0.5-sigma ellipsoid, regardless of orientation.
    distance_squared = (offset * torch.linalg.solve(parent_covariance, offset.T).T).sum(-1)
    torch.testing.assert_close(distance_squared, torch.full((2,), 0.25))


@pytest.mark.parametrize("split", [False, True])
def test_refinement_preserves_survivor_adam_state_and_group_options(split):
    points = torch.arange(12, dtype=torch.float32).reshape(4, 3) / 10
    scene = GaussianScene(points, torch.full_like(points, 0.4), torch.full_like(points, 0.1))
    optimizer = torch.optim.Adam(
        [
            {"params": [scene.means], "lr": 0.002},
            {"params": [p for n, p in scene.named_parameters() if n != "means"], "lr": 0.003},
        ],
        amsgrad=True,
    )
    sum((p + 0.3).square().sum() for p in scene.parameters()).backward()
    optimizer.step()
    before = {
        name: {key: value.clone() for key, value in optimizer.state[p].items()}
        for name, p in scene.named_parameters()
    }
    old_params = list(scene.parameters())
    result = scene.refine(
        torch.tensor([False, split, False, False]),
        torch.tensor([False, True, True, False]),
        optimizer=optimizer,
    )
    assert result == {"before": 4, "pruned": 2, "cloned": int(split), "after": 2 + int(split)}
    assert [group["lr"] for group in optimizer.param_groups] == [0.002, 0.003]
    assert {id(p) for p in optimizer.state}.isdisjoint(id(p) for p in old_params)
    assert {id(p) for group in optimizer.param_groups for p in group["params"]} == {
        id(p) for p in scene.parameters()
    }
    for name, parameter in scene.named_parameters():
        state = optimizer.state[parameter]
        torch.testing.assert_close(state["step"], before[name]["step"])
        for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            # The unchanged third input Gaussian becomes row 1 after pruning.
            torch.testing.assert_close(state[key][1], before[name][key][2])
            if split:
                assert torch.count_nonzero(state[key][[0, 2]]) == 0
            else:
                torch.testing.assert_close(state[key][0], before[name][key][1])
    before_step = scene.means.detach().clone()
    optimizer.zero_grad(set_to_none=True)
    sum((p + 0.3).square().sum() for p in scene.parameters()).backward()
    optimizer.step()
    assert not torch.equal(scene.means, before_step)


def test_default_schedule_uses_all_25_views_for_refinement(tmp_path):
    manifest = make_synthetic(tmp_path / "data", size=12, views=1)
    document = json.loads(manifest.read_text())
    document["frames"] = [dict(document["frames"][0], name=str(i)) for i in range(25)]
    manifest.write_text(json.dumps(document))
    config = TrainConfig(max_gaussians=50)
    trainer = Trainer(str(manifest), config)
    # Under the real default schedule, every refinement boundary is on view 24.
    # Simulate a Gaussian visible in each view, with no gradient in the other views.
    for step in range(500):
        phase = phase_at(step, config)
        frame_index = (step // 2) % len(trainer.frames)
        gradient = torch.zeros_like(trainer.scene.means)
        gradient[frame_index, 0] = 0.001
        trainer.scene.means.grad = gradient
        trainer.accumulate_refinement_stats(step, phase)
        event = trainer.maybe_refine_scene(step, phase)
        if step < 499:
            assert event is None
    assert frame_index == 24
    assert event["cloned"] == 25
    assert event["after"] == 50
    assert torch.count_nonzero(trainer.refinement_gradient_count) == 0


def test_no_split_check_resets_statistics_and_does_not_split_unobserved_points(tmp_path):
    manifest = make_synthetic(tmp_path / "data", size=12, views=1)
    config = TrainConfig(
        densify_from=1,
        densify_every=1,
        densify_grad_threshold=0,
        max_gaussians=25,
        prune_opacity_threshold=0,
    )
    trainer = Trainer(str(manifest), config)
    trainer.scene.means.grad = torch.ones_like(trainer.scene.means)
    trainer.accumulate_refinement_stats(0, "geometry")
    assert trainer.maybe_refine_scene(0, "geometry") is None
    assert torch.count_nonzero(trainer.refinement_gradient_count) == 0
    config.max_gaussians = 50
    # Neither old scores nor a threshold of zero may split unseen Gaussians.
    assert trainer.maybe_refine_scene(1, "geometry") is None
    assert len(trainer.scene.means) == 25


def test_all_pruned_fallback_keeps_optimizer_state_and_respects_capacity(tmp_path):
    manifest = make_synthetic(tmp_path / "data", size=12, views=1)
    config = TrainConfig(
        densify_from=1,
        densify_every=1,
        densify_grad_threshold=0,
        prune_opacity_threshold=1,
        max_gaussians=1,
    )
    trainer = Trainer(str(manifest), config)
    result = trainer.step(0, 1)
    assert result["gaussians"] == 1
    assert result["refinement"]["cloned"] == 0
    assert len(trainer.scene_optimizer.state) == 5


@pytest.mark.parametrize("kind", ["linear", "spline", "ode"])
def test_resume_preserves_pending_statistics_rng_and_next_split(tmp_path, kind):
    manifest = make_synthetic(tmp_path / "data", size=12, views=1)
    config = TrainConfig(
        trajectory=kind,
        ode_steps=2,
        exposure_samples=3,
        joint_start=0,
        depth_warmup_steps=0,
        motion_ramp_steps=1,
        densify_from=2,
        densify_until=4,
        densify_every=2,
        densify_grad_threshold=0,
        prune_opacity_threshold=0,
        max_gaussians=200,
    )
    original = Trainer(str(manifest), config)
    original.initialize_motion()
    for step in range(3):
        original.step(0, step)
    pending = original.refinement_gradient_sum.clone()
    counts = original.refinement_gradient_count.clone()
    assert (counts > 0).any()
    checkpoint = tmp_path / "checkpoint.pt"
    original.save(checkpoint, 3)
    rng_at_save = torch.get_rng_state().clone()
    expected = original.step(0, 3)
    expected_rng = torch.get_rng_state().clone()

    restored = Trainer(str(manifest), config)
    assert restored.resume(str(checkpoint)) == 3
    assert torch.equal(torch.get_rng_state(), rng_at_save)
    torch.testing.assert_close(restored.refinement_gradient_sum, pending, atol=0, rtol=0)
    torch.testing.assert_close(restored.refinement_gradient_count, counts, atol=0, rtol=0)
    actual = restored.step(0, 3)
    assert actual == expected
    assert actual["refinement"]["cloned"] > 0
    assert torch.equal(torch.get_rng_state(), expected_rng)
    for expected_module, actual_module in (
        (original.scene, restored.scene),
        (original.trajectories, restored.trajectories),
    ):
        for key, value in expected_module.state_dict().items():
            torch.testing.assert_close(actual_module.state_dict()[key], value, atol=0, rtol=0)
    for a, b in zip(original.scene.parameters(), restored.scene.parameters()):
        for key, value in original.scene_optimizer.state[a].items():
            torch.testing.assert_close(
                restored.scene_optimizer.state[b][key], value, atol=0, rtol=0
            )


def test_legacy_checkpoint_can_resume_with_explicit_continuity_warning(tmp_path):
    manifest = make_synthetic(tmp_path / "data", size=12, views=1)
    config = TrainConfig()
    trainer = Trainer(str(manifest), config)
    checkpoint = tmp_path / "legacy.pt"
    trainer.save(checkpoint, 1)
    saved = torch.load(checkpoint, weights_only=True)
    del saved["rng_state"], saved["refinement_stats"]
    torch.save(saved, checkpoint)
    restored = Trainer(str(manifest), config)
    with pytest.warns(RuntimeWarning, match="Legacy checkpoint lacks refinement/RNG state"):
        assert restored.resume(str(checkpoint)) == 1
    assert torch.count_nonzero(restored.refinement_gradient_count) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for CUDA RNG restoration")
def test_resume_restores_selected_cuda_generator(tmp_path):
    manifest = make_synthetic(tmp_path / "data", size=12, views=1)
    config = TrainConfig(device="cuda")
    trainer = Trainer(str(manifest), config)
    torch.rand(17, device="cuda")
    checkpoint = tmp_path / "cuda.pt"
    trainer.save(checkpoint, 1)
    expected_cpu = torch.rand(10)
    expected_cuda = torch.rand(10, device="cuda")
    restored = Trainer(str(manifest), config)
    restored.resume(str(checkpoint))
    torch.testing.assert_close(torch.rand(10), expected_cpu, atol=0, rtol=0)
    torch.testing.assert_close(torch.rand(10, device="cuda"), expected_cuda, atol=0, rtol=0)
