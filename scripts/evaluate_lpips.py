"""Evaluate saved checkpoints with AlexNet LPIPS v0.1, without further training."""

import argparse
import json
import math
from pathlib import Path
import re
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from arguments import TrainConfig, add_source_argument, configure_cpu_threads, resolve_source
from gaussian_renderer import make_renderer
from scene.gaussian_model import GaussianScene
from utils.training_monitor import TrainingMonitor


def load_lpips(device):
    try:
        import lpips
    except ImportError as exc:
        raise RuntimeError("Install LPIPS first: python -m pip install lpips==0.1.4") from exc
    # Match CoMoGaussian's LPIPS(net='alex') and [-1, 1] input convention.
    return (
        lpips.LPIPS(net="alex", version="0.1", verbose=False)
        .to(device)
        .eval()
        .requires_grad_(False)
    )


@torch.no_grad()
def image_lpips(model, rendered, truth):
    if rendered.shape != truth.shape or rendered.ndim != 3 or rendered.shape[-1] != 3:
        raise ValueError("LPIPS requires matching HWC RGB images")
    # Preserve native resolution; do not use rounded PNGs or training reblurred images.
    inputs = [
        image.clamp(0, 1).permute(2, 0, 1).unsqueeze(0) * 2 - 1 for image in (rendered, truth)
    ]
    value = model(*inputs)
    if value.numel() != 1:
        raise ValueError("LPIPS must return one scalar per image pair")
    score = float(value.item())
    if not math.isfinite(score):
        raise FloatingPointError("Non-finite LPIPS score")
    return score


def select_checkpoints(directory, all_checkpoints):
    final = directory / "checkpoint.pt"
    if not all_checkpoints:
        if not final.is_file():
            raise FileNotFoundError(final)
        return [final]
    named = {}
    for path in directory.glob("checkpoint_*.pt"):
        match = re.fullmatch(r"checkpoint_(\d+)\.pt", path.name)
        if match:
            named[int(match[1])] = path
    if final.is_file():
        saved = torch.load(final, map_location="cpu", weights_only=True)
        named[int(saved["step"])] = final  # Do not count the final step twice.
    if not named:
        raise FileNotFoundError(f"No checkpoints in {directory}")
    return [named[step] for step in sorted(named)]


@torch.no_grad()
def evaluate_checkpoint(path, views, model, device, backend):
    saved = torch.load(path, map_location=device, weights_only=True)
    test_names = {frame.name for frame, _, _ in views}
    overlap = test_names.intersection(saved["frame_names"])
    if overlap:
        raise ValueError(f"Test frames were included in checkpoint training: {sorted(overlap)}")
    state = saved["scene"]
    scene = GaussianScene(
        state["means"], state["color_logits"].sigmoid(), state["log_scales"].exp()
    ).to(device)
    scene.load_state_dict(state)
    scene.eval().requires_grad_(False)
    renderer = make_renderer(backend)
    rows = []
    for cpu_frame, cpu_truth, truth_path in views:
        frame, truth = cpu_frame.to(device), cpu_truth.to(device)
        # Held-out calibration only: never fit a pose or reuse a learned training trajectory.
        rendered = renderer(scene, frame.w2c, frame.K, *frame.image.shape[:2]).rgb
        rows.append(
            {
                "name": frame.name,
                "sharp_image": truth_path,
                "lpips": image_lpips(model, rendered, truth),
            }
        )
    if not rows:
        raise ValueError("LPIPS evaluation requires held-out test frames")
    return {
        "step": int(saved["step"]),
        "checkpoint": str(path.resolve()),
        "mean_lpips": sum(row["lpips"] for row in rows) / len(rows),
        "test_frames": len(rows),
        "frames": rows,
    }


def evaluate_run(
    directory,
    manifest=None,
    *,
    device="cuda",
    backend="gsplat",
    all_checkpoints=False,
    eval_inputs=False,
):
    directory = Path(directory).expanduser().resolve()
    paths = select_checkpoints(directory, all_checkpoints)
    # Reuse the saved evaluation convention and dataset unless explicitly supplied.
    saved = torch.load(paths[-1], map_location="cpu", weights_only=True)
    source = manifest or saved.get("manifest")
    if not source:
        raise ValueError("Checkpoint has no manifest; provide -s")
    source = resolve_source(source)
    previous = saved.get("config", {})
    settings = TrainConfig(
        eval=eval_inputs
        or previous.get("eval", False)
        or previous.get("eval_test_images_as_sharp", False),
        eval_every=1,
        tensorboard=False,
    )
    views = TrainingMonitor(source, settings).views
    del saved
    model = load_lpips(device)
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        print("TensorBoard not available: LPIPS will be saved to lpips.json", flush=True)
    else:
        # No purge_step: preserve the completed run's training and PSNR/SSIM history.
        writer = SummaryWriter(str(directory / "tensorboard"), filename_suffix=".lpips")
    report = {
        "metric": "LPIPS",
        "net": "alex",
        "version": "0.1",
        "backend": backend,
        "manifest": source,
        "split": "test",
        "test_pose_optimization": False,
        "checkpoints": [],
    }
    try:
        for path in paths:
            result = evaluate_checkpoint(path, views, model, device, backend)
            report["checkpoints"].append(result)
            if writer is not None:
                writer.add_scalar("test/lpips", result["mean_lpips"], result["step"])
                writer.flush()
            # A separate report leaves training summaries and evaluation.jsonl intact.
            (directory / "lpips.json").write_text(
                json.dumps(report, indent=2) + "\n", encoding="utf-8"
            )
            print(
                json.dumps({key: result[key] for key in ("step", "mean_lpips", "test_frames")}),
                flush=True,
            )
    finally:
        if writer is not None:
            writer.close()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_source_argument(parser, required=False)
    parser.add_argument(
        "-m", "--model_path", required=True, help="Completed training output directory"
    )
    parser.add_argument("--eval", action="store_true", help="Use held-out test images as GT")
    parser.add_argument(
        "--all-checkpoints", action="store_true", help="Also evaluate intermediate checkpoints"
    )
    parser.add_argument("--backend", choices=("torch", "gsplat"), default="gsplat")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    configure_cpu_threads(args.device)
    return evaluate_run(
        args.model_path,
        args.source_path,
        device=args.device,
        backend=args.backend,
        all_checkpoints=args.all_checkpoints,
        eval_inputs=args.eval,
    )


if __name__ == "__main__":
    main()
