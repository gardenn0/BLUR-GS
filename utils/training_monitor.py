"""Held-out sharp-image evaluation and optional TensorBoard training reports."""

import json
import math
from dataclasses import asdict

import torch

from scene.dataset_readers import load_manifest, read_frame
from utils.image_utils import read_image
from utils.loss_utils import ssim


class TrainingMonitor:
    def __init__(self, manifest, config):
        self.config = config
        self.test_steps = (
            set(config.test_iterations) if config.test_iterations is not None else None
        )
        self.eval_every = config.eval_every or (1000 if config.eval else 0)
        self.evaluation_enabled = bool(self.test_steps) or self.eval_every > 0
        self.writer = None
        self.writer_type = None
        self.views = []
        if config.tensorboard is not False:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError as exc:
                if config.tensorboard is True:
                    raise RuntimeError(
                        "TensorBoard requested but missing; run python -m pip install '.[logging]'"
                    ) from exc
                print("TensorBoard not available: continuing without TensorBoard logging", flush=True)
            else:
                self.writer_type = SummaryWriter
        if not self.evaluation_enabled:
            return
        root, document = load_manifest(manifest)
        for entry in document["frames"]:
            if entry.get("split", "train") != "test":
                continue
            truth_path = entry.get("sharp_image")
            if truth_path is None and (config.eval or config.eval_test_images_as_sharp):
                truth_path = entry["image"]
            if truth_path is None:
                raise ValueError(
                    f"Test frame {entry['name']} needs sharp_image ground truth. "
                    "Use --eval when the held-out test images are the ground truth."
                )
            frame = read_frame(root, {k: v for k, v in entry.items() if k != "motion"}, False)
            truth = read_image(root / truth_path)
            if truth.shape != frame.image.shape:
                raise ValueError(f"Test frame {frame.name}: sharp GT resolution must match image")
            self.views.append((frame, truth, str(truth_path)))
        if not self.views:
            raise ValueError("Evaluation requires held-out frames with split='test'")

    def start(self, directory, start):
        self.directory = directory
        self.report_path = directory / "evaluation.jsonl"
        # Resuming an earlier checkpoint discards later evaluations, matching purge_step below.
        if self.evaluation_enabled and self.report_path.exists():
            rows = [json.loads(line) for line in self.report_path.read_text().splitlines()]
            self.report_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows if row["step"] <= start),
                encoding="utf-8",
            )
        if self.writer_type is not None:
            self.writer = self.writer_type(
                log_dir=str(directory / "tensorboard"),
                purge_step=start + 1 if start else None,
            )
            self.writer.add_text("run/config", json.dumps(asdict(self.config), indent=2), start)
            self.writer.flush()
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if self.writer is not None:
            self.writer.close()

    def log_train(self, metrics):
        if self.writer is None:
            return
        for key, value in metrics.items():
            if key != "step" and isinstance(value, (int, float)):
                self.writer.add_scalar(f"train/{key}", value, metrics["step"])
        self.writer.flush()

    def should_evaluate(self, step):
        if self.test_steps is not None:
            return step in self.test_steps
        return self.eval_every > 0 and (
            step % self.eval_every == 0 or step == self.config.iterations
        )

    @torch.no_grad()
    def evaluate(self, trainer, step):
        device = torch.device(self.config.device)
        devices = (
            [device.index if device.index is not None else torch.cuda.current_device()]
            if device.type == "cuda"
            else []
        )
        # Reporting must not alter the randomness used by Gaussian refinement.
        with torch.random.fork_rng(devices=devices):
            rows = []
            for index, (cpu_frame, cpu_truth, truth_path) in enumerate(self.views):
                frame, truth = cpu_frame.to(device), cpu_truth.to(device)
                # Test cameras use their provided calibration; no training-pose lookup or fitting.
                rendered = trainer.renderer(
                    trainer.scene, frame.w2c, frame.K, *frame.image.shape[:2]
                ).rgb.clamp(0, 1)
                mse = (rendered - truth).square().mean()
                psnr = float(-10 * torch.log10(mse.clamp_min(1e-12)))
                similarity = float(ssim(rendered, truth))
                if not math.isfinite(psnr) or not math.isfinite(similarity):
                    raise FloatingPointError(
                        f"Non-finite test metrics at step {step}: {frame.name}"
                    )
                rows.append(
                    {
                        "name": frame.name,
                        "psnr": psnr,
                        "ssim": similarity,
                        "sharp_image": truth_path,
                    }
                )
                if self.writer is not None:
                    self.writer.add_image(
                        f"test/render_left_gt_right/{index:03d}",
                        torch.cat((rendered, truth), dim=1),
                        step,
                        dataformats="HWC",
                    )
            report = {
                "step": step,
                "mean_psnr": sum(row["psnr"] for row in rows) / len(rows),
                "mean_ssim": sum(row["ssim"] for row in rows) / len(rows),
                "test_frames": len(rows),
                "test_pose_optimization": False,
                "frames": rows,
            }
        with self.report_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(report) + "\n")
        if self.writer is not None:
            self.writer.add_scalar("test/psnr", report["mean_psnr"], step)
            self.writer.add_scalar("test/ssim", report["mean_ssim"], step)
            self.writer.flush()
        return report
