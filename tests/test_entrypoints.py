"""Regression checks for the research-style layout and legacy command compatibility."""

import json
from pathlib import Path
import subprocess
import sys

import pytest

from arguments import read_config, resolve_source, training_parser

ROOT = Path(__file__).resolve().parents[1]


def run_cli(*args, cwd=ROOT):
    result = subprocess.run(
        [sys.executable, *map(str, args)],
        cwd=cwd,
        text=True,
        capture_output=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result


@pytest.mark.parametrize(
    "script",
    [
        "train.py",
        "render.py",
        "metrics.py",
        "scripts/make_synthetic.py",
        "scripts/import_colmap.py",
        "scripts/prepare_motion.py",
    ],
)
def test_direct_entrypoint_help_from_another_directory(script, tmp_path):
    # Absolute script paths must work without PYTHONPATH or cwd manipulation by the caller.
    assert "usage:" in run_cli(ROOT / script, "--help", cwd=tmp_path).stdout


def test_legacy_command_help_dispatches_to_training():
    result = run_cli("-m", "blur_gs", "train", "--help")
    assert "--source_path" in result.stdout and "--exposure-samples" in result.stdout


def test_source_resolution_and_option_aliases(tmp_path):
    with pytest.raises(FileNotFoundError, match="Prepare the scene"):
        resolve_source(tmp_path)
    (tmp_path / "scene.json").write_text("{}", encoding="utf-8")
    assert Path(resolve_source(tmp_path)).name == "scene.json"
    (tmp_path / "scene.motion.json").write_text("{}", encoding="utf-8")
    assert Path(resolve_source(tmp_path)).name == "scene.motion.json"
    assert Path(resolve_source(tmp_path / "scene.json")).name == "scene.json"
    modern = training_parser().parse_args(["-s", "data", "-m", "run"])
    legacy = training_parser().parse_args(["--data", "data", "--output", "run"])
    assert vars(modern) == vars(legacy)


def test_configuration_preserves_defaults_and_applies_overrides_before_validation():
    assert read_config(None).exposure_samples == 9
    config = read_config(
        str(ROOT / "configs/default.yaml"), device="cpu", backend="torch", iterations=2
    )
    assert config.exposure_samples == 9
    assert config.iterations == 2 and config.backend == "torch"
    with pytest.raises(ValueError, match="at least three"):
        read_config(None, exposure_samples=2)


def test_train_render_metrics_and_legacy_resume(tmp_path):
    data, output = tmp_path / "data", tmp_path / "run"
    run_cli("scripts/make_synthetic.py", "--output", data, "--size", 16, "--views", 1)
    run_cli("train.py", "-s", data, "-m", output, "--iterations", 2, "--exposure-samples", 3)
    summary = json.loads((output / "summary.json").read_text())
    assert summary["trajectory_model"] == "linear_se3"
    assert summary["exposure_samples"] == 3 and summary["iterations"] == 2
    run_cli("render.py", "-m", output)
    assert (output / "renders" / "000000.png").is_file()
    run_cli("metrics.py", "-m", output, "-s", data / "scene.json", "--split", "train")
    metrics = json.loads((output / "evaluation" / "renders.json").read_text())
    assert metrics["mean_psnr"] > 0 and metrics["test_pose_optimization"] is False
    run_cli(
        "-m",
        "blur_gs",
        "train",
        "--data",
        data / "scene.json",
        "--output",
        output,
        "--iterations",
        4,
        "--exposure-samples",
        3,
        "--resume",
        output / "checkpoint.pt",
    )
    summary = json.loads((output / "summary.json").read_text())
    assert summary["iterations"] == 4
