import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from arguments import TrainConfig, training_parser
from render import render_checkpoint
from scripts.make_synthetic import make_synthetic
from train import Trainer, train
from utils.training_monitor import TrainingMonitor


def held_out_scene(tmp_path):
    path = make_synthetic(tmp_path / "data", size=16, views=3)
    document = json.loads(path.read_text())
    for entry in document["frames"][1:]:
        entry["split"] = "test"
        # Held-out evaluation must not need motion caches.
        del entry["motion"]
    path.write_text(json.dumps(document))
    return path


def config(**kwargs):
    return TrainConfig(
        iterations=3,
        exposure_samples=3,
        densify_every=0,
        log_every=1,
        checkpoint_every=1,
        initialize_motion=False,
        **kwargs,
    )


def events(output):
    return EventAccumulator(str(output / "tensorboard"), size_guidance={"scalars": 0}).Reload()


def test_periodic_evaluation_records_real_metrics_and_images(tmp_path):
    manifest, output = held_out_scene(tmp_path), tmp_path / "run"
    train(str(manifest), config(eval_every=2), str(output))
    reports = [json.loads(row) for row in (output / "evaluation.jsonl").read_text().splitlines()]
    assert [r["step"] for r in reports] == [2, 3]  # Interval plus final step.
    assert all(r["test_frames"] == 2 and not r["test_pose_optimization"] for r in reports)
    assert {r["name"] for r in reports[0]["frames"]} == {"001", "002"}
    saved = torch.load(output / "checkpoint.pt", weights_only=True)
    assert saved["frame_names"] == ["000"]  # Test images never enter the training set.
    reference = render_checkpoint(
        str(output / "checkpoint_000002.pt"),
        str(manifest),
        str(tmp_path / "eval"),
        evaluate=True,
        split="test",
    )
    assert reports[0]["mean_psnr"] == pytest.approx(reference["mean_psnr"], abs=1e-6)
    assert reports[0]["mean_ssim"] == pytest.approx(reference["mean_ssim"], abs=1e-6)
    recorded = events(output)
    assert [e.step for e in recorded.Scalars("train/total")] == [1, 2, 3]
    assert [e.step for e in recorded.Scalars("test/psnr")] == [2, 3]
    assert recorded.Scalars("test/psnr")[0].value == pytest.approx(reports[0]["mean_psnr"])
    assert recorded.Scalars("test/ssim")[0].value == pytest.approx(reports[0]["mean_ssim"])
    comparison = recorded.Images("test/render_left_gt_right/000")[0]
    assert (comparison.width, comparison.height) == (32, 16)


def test_eval_uses_held_out_images_as_gt_without_an_extra_flag(tmp_path):
    manifest = held_out_scene(tmp_path)
    document = json.loads(manifest.read_text())
    for entry in document["frames"]:
        entry.pop("sharp_image")
    manifest.write_text(json.dumps(document))
    output = tmp_path / "run"
    with pytest.raises(ValueError, match="needs sharp_image"):
        train(str(manifest), config(eval_every=2), str(output))
    assert not output.exists()
    monitor = TrainingMonitor(manifest, config(eval=True))
    for frame, truth, path in monitor.views:
        torch.testing.assert_close(truth, frame.image)
        assert path.startswith("images/")
    assert monitor.should_evaluate(1000) and not monitor.should_evaluate(999)


def test_eval_prefers_separate_sharp_ground_truth_when_present(tmp_path):
    manifest = held_out_scene(tmp_path)
    monitor = TrainingMonitor(manifest, config(eval=True))
    document = json.loads(manifest.read_text())
    assert [path for _, _, path in monitor.views] == [
        entry["sharp_image"] for entry in document["frames"] if entry["split"] == "test"
    ]


def test_automatic_tensorboard_is_optional_and_can_be_disabled(tmp_path, monkeypatch, capsys):
    manifest = held_out_scene(tmp_path)
    assert TrainingMonitor(manifest, config()).writer_type is not None
    assert TrainingMonitor(manifest, config(tensorboard=False)).writer_type is None
    monkeypatch.setitem(sys.modules, "torch.utils.tensorboard", None)
    output = tmp_path / "without-tensorboard"
    train(str(manifest), config(eval=True), str(output))
    assert "TensorBoard not available" in capsys.readouterr().out
    assert (output / "checkpoint.pt").is_file()
    assert (output / "evaluation.jsonl").is_file()
    assert not (output / "tensorboard").exists()
    with pytest.raises(RuntimeError, match="TensorBoard requested but missing"):
        TrainingMonitor(manifest, config(tensorboard=True))


def test_evaluation_requires_test_split_and_matching_resolution(tmp_path):
    manifest = make_synthetic(tmp_path / "data", size=16, views=1)
    with pytest.raises(ValueError, match="held-out"):
        TrainingMonitor(manifest, config(eval_every=1))
    document = json.loads(manifest.read_text())
    document["frames"][0]["split"] = "test"
    manifest.write_text(json.dumps(document))
    from PIL import Image

    Image.new("RGB", (8, 8)).save(manifest.parent / document["frames"][0]["sharp_image"])
    with pytest.raises(ValueError, match="resolution"):
        TrainingMonitor(manifest, config(eval_every=1))


def test_evaluation_preserves_training_state_gradients_and_rng(tmp_path):
    manifest = held_out_scene(tmp_path)
    settings = config(eval_every=1)
    trainer = Trainer(str(manifest), settings)
    trainer.step(0, 0)
    parameters = list(trainer.scene.parameters()) + list(trainer.trajectories.parameters())
    snapshots = [
        (p.detach().clone(), p.requires_grad, None if p.grad is None else p.grad.clone())
        for p in parameters
    ]
    rng = torch.get_rng_state().clone()
    render = trainer.renderer

    def consuming_renderer(*args):
        assert not torch.is_grad_enabled()
        torch.rand(7)
        return render(*args)

    trainer.renderer = consuming_renderer
    with TrainingMonitor(manifest, settings).start(tmp_path, 0) as monitor:
        monitor.evaluate(trainer, 1)
    assert torch.equal(rng, torch.get_rng_state())
    for p, (value, requires_grad, gradient) in zip(parameters, snapshots):
        assert torch.equal(p, value) and p.requires_grad == requires_grad
        assert p.grad is None if gradient is None else torch.equal(p.grad, gradient)


def test_resume_legacy_checkpoint_purges_future_events_and_keeps_step_numbers(tmp_path):
    manifest, output = held_out_scene(tmp_path), tmp_path / "run"
    settings = replace(config(eval_every=2, tensorboard=True), iterations=4)
    train(str(manifest), settings, str(output))
    legacy = torch.load(output / "checkpoint_000002.pt", weights_only=True)
    for key in ("eval", "eval_every", "test_iterations", "tensorboard", "eval_test_images_as_sharp"):
        legacy["config"].pop(key)
    torch.save(legacy, output / "legacy.pt")
    train(
        str(manifest), replace(settings, iterations=5, eval=True), str(output),
        str(output / "legacy.pt"),
    )
    reports = [json.loads(row) for row in (output / "evaluation.jsonl").read_text().splitlines()]
    assert [r["step"] for r in reports] == [2, 4, 5]
    recorded = events(output)
    assert [e.step for e in recorded.Scalars("train/total")] == [1, 2, 3, 4, 5]
    assert [e.step for e in recorded.Scalars("test/psnr")] == [2, 4, 5]


def test_comogaussian_flags_enable_eval_and_default_tensorboard():
    parser = training_parser()
    args = parser.parse_args(["-s", "data", "-m", "run"])
    assert args.eval is None and args.eval_every is None and args.tensorboard is None
    assert TrainConfig().tensorboard is None
    args = parser.parse_args(["-s", "data", "-m", "run", "--eval"])
    assert args.eval and args.tensorboard is None and args.eval_test_images_as_sharp is None
    args = parser.parse_args(["-s", "data", "-m", "run", "--no-tensorboard"])
    assert args.tensorboard is False
    args = parser.parse_args(["-s", "data", "-m", "run", "--eval-every", "7"])
    assert args.eval_every == 7
    with pytest.raises(ValueError, match="eval_every"):
        config(eval_every=-1).validate()


@pytest.mark.parametrize("schedule, expected_steps", [([], [2]), (["1", "2"], [1, 2])])
def test_training_cli_enables_reporting_with_eval_only(tmp_path, schedule, expected_steps):
    manifest = held_out_scene(tmp_path)
    document = json.loads(manifest.read_text())
    for entry in document["frames"]:
        entry.pop("sharp_image")
    manifest.write_text(json.dumps(document))
    output = tmp_path / "cli-run"
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "train.py"),
            "-s",
            str(manifest),
            "-m",
            str(output),
            "--iterations",
            "2",
            "--exposure-samples",
            "3",
            "--eval",
        ] + (["--test_iterations", *schedule] if schedule else []),
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert '"evaluation"' in result.stdout
    assert [e.step for e in events(output).Scalars("test/psnr")] == expected_steps
    assert [e.step for e in events(output).Scalars("train/total")] == [1, 2]
    saved = torch.load(output / "checkpoint.pt", weights_only=True)
    assert saved["frame_names"] == ["000"]


def test_explicit_test_iterations_override_interval_without_extra_final_evaluation(tmp_path):
    manifest, output = held_out_scene(tmp_path), tmp_path / "run"
    settings = config(eval_every=1, test_iterations=[2, 2], tensorboard=True)
    train(str(manifest), settings, str(output))
    reports = [json.loads(row) for row in (output / "evaluation.jsonl").read_text().splitlines()]
    assert [row["step"] for row in reports] == [2]
    assert [event.step for event in events(output).Scalars("test/psnr")] == [2]
    args = training_parser().parse_args(
        ["-s", "data", "-m", "run", "--test_iterations"]
        + [str(step) for step in range(1000, 40001, 1000)]
    )
    assert args.test_iterations == list(range(1000, 40001, 1000))
    for steps in ([], [0], [-1], [1.5]):
        with pytest.raises(ValueError, match="test_iterations"):
            config(test_iterations=steps).validate()
