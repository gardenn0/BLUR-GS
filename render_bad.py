"""Render/evaluate a full BAD-BLUR-GS checkpoint, including exposure diagnostics."""
from argparse import ArgumentParser
from dataclasses import replace
from pathlib import Path
import json
import torch
from bad_blur_gs.config import BadConfig
from bad_blur_gs.data import load_scene, load_image
from bad_blur_gs.gaussians import Gaussians
from bad_blur_gs.trajectory import ExposureTrajectory
from bad_blur_gs.renderer import render, require_backend
from bad_blur_gs.training import evaluate, save_image, background_for


@torch.no_grad()
def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("-m", "--model_path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("-s", "--source_path", help="Optional relocated source root")
    parser.add_argument("--train_limit", type=int, default=5, help="Exposure diagnostic views, 0 means all")
    args = parser.parse_args()
    if args.train_limit < 0:
        parser.error("--train_limit must be nonnegative")
    require_backend()
    root = Path(args.model_path)
    metadata = json.loads((root/"bad_config.json").read_text())
    old = metadata["args"]
    # Full optimizer checkpoints are pickle-based: only load files you trust.
    state = torch.load(args.checkpoint, map_location="cuda", weights_only=False)
    if state.get("bad_blur_gs_version") != 1:
        raise ValueError("Not a BAD-BLUR-GS checkpoint")
    cfg = BadConfig(**state["config"])
    train, test, _, _, manifest = load_scene(args.source_path or old["source_path"], old["images"],
        old["resolution"], old["eval"], old["llffhold"], cfg.scene_scale)
    if manifest["fingerprint"] != state["data"]["fingerprint"]:
        raise ValueError("Dataset/calibration differs from checkpoint")
    model = Gaussians(state["gaussians"], cfg)
    model.active_sh_degree = state["active_sh_degree"]
    trajectory = ExposureTrajectory(train, cfg, "cuda")
    trajectory.load_state_dict(state["trajectory"])
    output = root/"rendered"/f"iteration_{state['step']}"
    import lpips
    metrics = evaluate(model, test, cfg, lpips.LPIPS(net="alex").eval().cuda(), output/"test")
    bg = background_for(cfg, "cuda", False)
    count = len(train) if args.train_limit == 0 else min(args.train_limit, len(train))
    for camera in train[:count]:
        name = camera.image_name.replace("/", "_")
        path = output/"train"/name
        gt, _ = load_image(camera, "cuda", background=bg)
        save_image(path/"input_blurry.png", gt)
        views = trajectory.cameras(camera)
        renders = [render(c, model, cfg, bg)["render"] for c in views]
        save_image(path/"predicted_blur.png", torch.stack(renders).mean(0))
        mid = trajectory.cameras(camera, [.5])[0]
        save_image(path/"sharp_mid.png", render(mid, model, cfg, bg)["render"])
        for i, image in enumerate(renders):
            save_image(path/f"exposure_{i:02d}.png", image)
    poses = {c.image_name: trajectory.poses(c.image_name).cpu().tolist() for c in train}
    output.mkdir(parents=True, exist_ok=True)
    (output/"trajectories.json").write_text(json.dumps(dict(
        convention="OpenCV c2w in normalized scene coordinates", normalization=state["data"],
        times=torch.linspace(0, 1, cfg.num_virtual_views).tolist(), poses=poses), indent=2))
    print(json.dumps(metrics, indent=2))
    print(f"Saved test metrics, renders and exposure diagnostics to {output}")


if __name__ == "__main__":
    main()
