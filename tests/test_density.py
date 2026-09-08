import torch

from arguments import TrainConfig
from scene.density import DensityController
from scene.gaussian_model import GaussianScene
from scripts.make_synthetic import make_synthetic
from train import Trainer


def test_clone_split_prune_preserves_adam_and_cap():
    scene = GaussianScene(
        torch.tensor([[0.0, 0.0, 2.0], [0.1, 0.0, 2.0], [0.2, 0.0, 2.0]]),
        torch.full((3, 3), 0.5),
        torch.tensor([[0.001] * 3, [0.1] * 3, [0.1] * 3]),
        sh_degree=3,
    )
    optimizer = torch.optim.Adam(scene.parameters(), lr=0.001)
    sum(p.square().sum() for p in scene.parameters()).backward()
    optimizer.step()
    with torch.no_grad():
        scene.opacity_logits[2] = -20
    old_moment = optimizer.state[scene.means]["exp_avg"][0].clone()
    controller = DensityController(scene, 1.0)
    controller.gradient_sum.fill_(1)
    controller.count.fill_(1)
    controller.refine(scene, optimizer, TrainConfig(max_gaussians=4))
    assert len(scene.means) == 4  # retained small, cloned small, two split children
    torch.testing.assert_close(optimizer.state[scene.means]["exp_avg"][0], old_moment)
    assert optimizer.state[scene.means]["exp_avg"][1:].count_nonzero() == 0
    assert (scene.scales[-2:] < 0.1).all()
    assert not torch.equal(scene.means[-1], scene.means[-2])
    optimizer.zero_grad()
    sum(p.square().sum() for p in scene.parameters()).backward()
    optimizer.step()
    controller.reset_opacity(scene, optimizer)
    assert scene.opacities.max() <= 0.010001
    assert optimizer.state[scene.opacity_logits]["exp_avg"].count_nonzero() == 0


def test_density_resume_matches_uninterrupted_training(tmp_path):
    manifest = make_synthetic(tmp_path / "data", size=16, views=1)
    config = TrainConfig(
        exposure_samples=10,
        joint_start=0,
        sh_interval=1,
        densify_from=0,
        densify_until=20,
        densify_interval=2,
        densify_grad_threshold=0,
        max_gaussians=80,
        opacity_reset_interval=4,
    )
    trainer = Trainer(str(manifest), config)
    trainer.initialize_motion()
    initial_count = len(trainer.scene.means)
    for step in range(2):
        trainer.step(0, step)
    assert len(trainer.scene.means) > initial_count
    trainer.save(tmp_path / "checkpoint.pt", 2)
    resumed = Trainer(str(manifest), config)
    assert resumed.resume(str(tmp_path / "checkpoint.pt")) == 2
    assert resumed.orientations == trainer.orientations
    for step in range(2, 5):
        a, b = trainer.step(0, step), resumed.step(0, step)
        assert abs(a["total"] - b["total"]) < 1e-7
        for name, value in trainer.scene.state_dict().items():
            torch.testing.assert_close(value, resumed.scene.state_dict()[name])
    assert trainer.scene.sh_rest.abs().sum() > 0


def test_sh_view_dependence_and_export(tmp_path):
    scene = GaussianScene(torch.tensor([[0.0, 0.0, 2.0]]), torch.full((1, 3), 0.5), sh_degree=3)
    scene.active_sh_degree.fill_(3)
    with torch.no_grad():
        scene.sh_rest[0, 2, 0] = 0.1
    pose = torch.eye(4)
    other = pose.clone()
    other[0, 3] = 1
    assert not torch.equal(scene.view_colors(pose), scene.view_colors(other))
    scene.view_colors(other).sum().backward()
    assert scene.sh_rest.grad.abs().sum() > 0
    scene.export_ply(tmp_path / "scene.ply")
    text = (tmp_path / "scene.ply").read_text()
    assert "property float f_rest_44" in text
    assert len(text.splitlines()[-1].split()) == 62
    restored = GaussianScene.from_state(scene.state_dict())
    torch.testing.assert_close(scene.view_colors(other), restored.view_colors(other))


def test_orientation_does_not_reselect_after_trajectory_flip(tmp_path):
    manifest = make_synthetic(tmp_path / "data", size=16, views=1)
    trainer = Trainer(str(manifest), TrainConfig())
    trainer.frames[0].sign_ambiguous = True
    trainer.orientations[0] = None
    trainer.initialize_motion()
    trainer.objective(0, 0, "joint")
    direction = trainer.orientations[0]
    assert direction is not None
    with torch.no_grad():
        controls = trainer.trajectories[0].controls
        controls.copy_(controls.flip(0))
    loss = trainer.objective(0, 0, "joint")
    assert bool(loss["reversed"]) == direction
    assert trainer.orientations[0] == direction
