"""Sharp rendering and explicit sharp-ground-truth evaluation, without test-pose fitting."""

import json
from pathlib import Path

import torch

from arguments import configure_cpu_threads, render_parser, resolve_source
from scene import load_manifest
from scene.dataset_readers import read_frame
from utils.image_utils import read_image, save_image
from utils.loss_utils import ssim
from gaussian_renderer import make_renderer
from scene.gaussian_model import GaussianScene
from scene.trajectory import checkpoint_midpoint


@torch.no_grad()
def render_checkpoint(
    checkpoint: str,
    manifest: str,
    output: str,
    device: str = "cpu",
    backend: str = "torch",
    split: str = "all",
    evaluate: bool = False,
) -> dict:
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    state = saved["scene"]
    scene = GaussianScene(
        state["means"], state["color_logits"].sigmoid(), state["log_scales"].exp()
    )
    scene.load_state_dict(state)
    renderer = make_renderer(backend)
    root, document = load_manifest(manifest)
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    by_name = {name: i for i, name in enumerate(saved["frame_names"])}
    rows = []
    for index, entry in enumerate(document["frames"]):
        if split != "all" and entry.get("split", "train") != split:
            continue
        frame = read_frame(root, {k: v for k, v in entry.items() if k != "motion"}, False).to(
            device
        )
        pose = frame.w2c
        if frame.name in by_name:
            i = by_name[frame.name]
            # Names are not a camera identity: only reuse a learned midpoint for the same calibration.
            initial = saved["trajectories"][f"{i}.initial_w2c"]
            if torch.allclose(initial, frame.w2c, atol=1e-6):
                pose = checkpoint_midpoint(saved, i)
        result = renderer(scene, pose, frame.K, *frame.image.shape[:2])
        save_image(destination / f"{index:06d}.png", result.rgb)
        row = {
            "name": frame.name,
            "image": f"{index:06d}.png",
            "split": entry.get("split", "train"),
        }
        if evaluate:
            if "sharp_image" not in entry:
                raise ValueError(
                    f"{frame.name}: evaluation requires explicit sharp_image ground truth"
                )
            truth = read_image(root / entry["sharp_image"]).to(device)
            if truth.shape != result.rgb.shape:
                raise ValueError("Sharp ground truth must match render resolution")
            mse = (result.rgb - truth).square().mean()
            row.update(
                psnr=float(-10 * torch.log10(mse.clamp_min(1e-12))),
                ssim=float(ssim(result.rgb, truth)),
            )
        rows.append(row)
    if not rows:
        raise ValueError(f"No frames selected for split {split!r}")
    summary = {"frames": rows, "backend": backend, "test_pose_optimization": False}
    if evaluate:
        summary["mean_psnr"] = sum(row["psnr"] for row in rows) / len(rows)
        summary["mean_ssim"] = sum(row["ssim"] for row in rows) / len(rows)
    (destination / "renders.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def run_render_cli(argv=None, *, evaluate=False):
    args = render_parser(evaluate=evaluate).parse_args(argv)
    configure_cpu_threads(args.device)
    checkpoint = (
        Path(args.checkpoint) if args.checkpoint else Path(args.model_path) / "checkpoint.pt"
    )
    if args.source_path:
        manifest = resolve_source(args.source_path)
    else:
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        source = saved.get("manifest")
        if not source:
            raise ValueError("Checkpoint has no dataset manifest; provide -s or --data")
        manifest = resolve_source(source)
    output = args.output or str(checkpoint.parent / ("evaluation" if evaluate else "renders"))
    result = render_checkpoint(
        str(checkpoint), manifest, output, args.device, args.backend, args.split, evaluate
    )
    print(json.dumps(result, indent=2))
    return result


def main(argv=None):
    return run_render_cli(argv)


if __name__ == "__main__":
    main()
