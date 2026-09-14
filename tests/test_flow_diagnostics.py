import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from PIL import Image
import pytest
from inspect_blur_flow import endpoint_error, source_image


def test_epe_uses_common_mask_and_preserves_direction():
    prediction = np.ones((2, 2, 3)) * np.array([3., 4.])[:, None, None]
    target = np.zeros_like(prediction)
    mask = np.ones((2, 3))
    target_mask = mask.copy()
    target_mask[0] = 0
    error, valid, mean = endpoint_error(prediction, target, mask, target_mask)
    assert mean == 5 and valid.sum() == 3
    assert np.all(error[valid] == 5)
    assert endpoint_error(prediction, -prediction, mask, mask)[2] == 10
    assert endpoint_error(prediction, target, mask, mask * 0)[2] is None
    with pytest.raises(ValueError, match="same flow grid"):
        endpoint_error(prediction, target[:, :, :1], mask, target_mask)


def test_source_relocation_checks_dimensions(tmp_path):
    Image.new("RGB", (40, 30)).save(tmp_path / "input.png")
    entry = dict(source="input.png", crop=dict(original_width=40, original_height=30))
    assert source_image(entry, tmp_path).size == (40, 30)
    entry["crop"]["original_width"] = 41
    with pytest.raises(ValueError, match="size differs"):
        source_image(entry, tmp_path)


def test_diagnostic_cli_with_and_without_gt(tmp_path):
    Image.new("RGB", (32, 24), (70, 80, 90)).save(tmp_path / "input.png")
    crop = dict(original_width=32, original_height=24, left=0, top=0, width=32, height=24)
    for name, offset in [("prediction", 3), ("gt", 0)]:
        directory = tmp_path / name
        directory.mkdir()
        flow = np.zeros((2, 24, 32), dtype=np.float32)
        flow[0] = offset
        np.savez(directory / "flow.npz", flow=flow, mask=np.ones((24, 32)))
        (directory / "manifest.json").write_text(json.dumps(dict(version=1, units="network_pixels",
            images={"frame": dict(file="flow.npz", crop=crop, source="input.png")})))
    script = Path(__file__).resolve().parents[1] / "inspect_blur_flow.py"
    for with_gt in [False, True]:
        output = tmp_path / str(with_gt)
        command = [sys.executable, str(script), "--flow_cache", str(tmp_path / "prediction"),
                   "--images", str(tmp_path), "--output", str(output)]
        if with_gt:
            command.extend(["--gt_cache", str(tmp_path / "gt")])
        result = subprocess.run(command, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        report = json.loads((output / "report.json").read_text())
        row = report["images"][0]
        assert row["gt_epe_px"] == (3 if with_gt else None)
        with Image.open(output / row["image"]) as image:
            assert image.size == (1200, 1032 if with_gt else 712)
