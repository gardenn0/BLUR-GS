import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter

from arguments import TrainConfig
from scripts import evaluate_lpips as evaluation
from scripts.make_synthetic import make_synthetic
from train import Trainer


class RecordingMetric(torch.nn.Module):
    """A deterministic test double for validating the image and reporting pipeline."""

    def __init__(self):
        super().__init__()
        self.pairs = []

    def forward(self, image, truth):
        assert not torch.is_grad_enabled()
        self.pairs.append((image.clone(), truth.clone()))
        return (image - truth).square().mean().reshape(1, 1, 1, 1)


def completed_run(tmp_path):
    manifest = make_synthetic(tmp_path / "data", size=16, views=3)
    document = json.loads(manifest.read_text())
    for entry in document["frames"][1:]:
        entry["split"] = "test"
        del entry["sharp_image"], entry["motion"]
    manifest.write_text(json.dumps(document))
    directory = tmp_path / "run"
    directory.mkdir()
    trainer = Trainer(str(manifest), TrainConfig(eval=True, tensorboard=False))
    trainer.save(directory / "checkpoint_000001.pt", 1)
    trainer.save(directory / "checkpoint_000002.pt", 2)
    trainer.save(directory / "checkpoint.pt", 2)
    return directory


def test_lpips_uses_clamped_native_rgb_nchw_minus_one_to_one():
    truth = torch.linspace(0, 1, 12 * 14 * 3).reshape(12, 14, 3)
    rendered = truth * 3 - 1
    metric = RecordingMetric()
    score = evaluation.image_lpips(metric, rendered.requires_grad_(True), truth)
    x, y = metric.pairs[0]
    assert x.shape == (1, 3, 12, 14)
    torch.testing.assert_close(x, rendered.detach().clamp(0, 1).permute(2, 0, 1)[None] * 2 - 1)
    torch.testing.assert_close(y, truth.permute(2, 0, 1)[None] * 2 - 1)
    assert score == pytest.approx(float(((x - y) ** 2).mean()))
    with pytest.raises(ValueError, match="matching HWC RGB"):
        evaluation.image_lpips(metric, rendered, truth[:4])
    with pytest.raises(FloatingPointError, match="Non-finite"):
        evaluation.image_lpips(lambda *args: torch.tensor(float("nan")), truth, truth)


def test_lpips_loads_alexnet_v01_in_eval_mode_and_reports_missing_dependency(monkeypatch):
    calls = []

    def constructor(**kwargs):
        calls.append(kwargs)
        return torch.nn.Conv2d(3, 3, 1)

    monkeypatch.setitem(sys.modules, "lpips", SimpleNamespace(LPIPS=constructor))
    model = evaluation.load_lpips("cpu")
    assert calls == [{"net": "alex", "version": "0.1", "verbose": False}]
    assert not model.training and not any(p.requires_grad for p in model.parameters())
    monkeypatch.setitem(sys.modules, "lpips", None)
    with pytest.raises(RuntimeError, match="pip install lpips==0.1.4"):
        evaluation.load_lpips("cpu")


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(),
                reason="CUDA required for checkpoint device placement",
            ),
        ),
    ],
)
def test_checkpoint_evaluation_preserves_run_files_and_existing_tensorboard_events(
    tmp_path, monkeypatch, device
):
    directory = completed_run(tmp_path)
    (directory / "summary.json").write_text('{"iterations":2}')
    (directory / "evaluation.jsonl").write_text('{"step":2,"mean_psnr":20}\n')
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir()}
    with SummaryWriter(str(directory / "tensorboard")) as writer:
        writer.add_scalar("train/total", 0.1, 2)
        writer.add_scalar("test/psnr", 20.0, 2)
    metric = RecordingMetric()
    monkeypatch.setattr(evaluation, "load_lpips", lambda device: metric)
    report = evaluation.evaluate_run(
        directory, device=device, backend="torch", all_checkpoints=True
    )
    assert [row["step"] for row in report["checkpoints"]] == [1, 2]
    assert report["checkpoints"][-1]["checkpoint"].endswith("checkpoint.pt")
    assert report["net"] == "alex" and not report["test_pose_optimization"]
    for row in report["checkpoints"]:
        assert row["test_frames"] == 2
        assert {f["name"] for f in row["frames"]} == {"001", "002"}
        assert row["mean_lpips"] == pytest.approx(sum(f["lpips"] for f in row["frames"]) / 2)
    assert len(metric.pairs) == 4
    assert json.loads((directory / "lpips.json").read_text()) == report
    recorded = EventAccumulator(str(directory / "tensorboard")).Reload()
    assert [e.step for e in recorded.Scalars("test/lpips")] == [1, 2]
    assert recorded.Scalars("test/psnr")[0].value == 20.0
    assert [e.step for e in recorded.Scalars("train/total")] == [2]
    assert all(
        hashlib.sha256((directory / name).read_bytes()).hexdigest() == digest
        for name, digest in hashes.items()
    )


def test_final_only_without_tensorboard_and_training_overlap_rejected(tmp_path, monkeypatch):
    directory = completed_run(tmp_path)
    monkeypatch.setattr(evaluation, "load_lpips", lambda device: RecordingMetric())
    monkeypatch.setitem(sys.modules, "torch.utils.tensorboard", None)
    report = evaluation.evaluate_run(directory, device="cpu", backend="torch")
    assert [row["step"] for row in report["checkpoints"]] == [2]
    assert not (directory / "tensorboard").exists()
    path = directory / "checkpoint.pt"
    saved = torch.load(path, weights_only=True)
    saved["frame_names"].append("001")
    torch.save(saved, path)
    with pytest.raises(ValueError, match="included in checkpoint training"):
        evaluation.evaluate_run(directory, device="cpu", backend="torch")


def test_lpips_cli_help_works_outside_the_repository(tmp_path):
    result = subprocess.run(
        [sys.executable, str(Path(evaluation.__file__).resolve()), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stderr
    assert "--all-checkpoints" in result.stdout and "--eval" in result.stdout
