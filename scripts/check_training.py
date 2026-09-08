"""Exercise the selected renderer, data and both training phases before a long run."""

import json
import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arguments import add_source_argument, configure_cpu_threads, read_config, resolve_source
from train import Trainer


def main():
    parser = argparse.ArgumentParser(description="Check both training phases without saving")
    add_source_argument(parser)
    parser.add_argument("--config")
    parser.add_argument("--device")
    parser.add_argument("--backend", choices=("torch", "gsplat"))
    parser.add_argument("--exposure-samples", type=int)
    args = parser.parse_args()
    config = read_config(
        args.config,
        device=args.device,
        backend=args.backend,
        exposure_samples=args.exposure_samples,
        joint_start=1000000000,
    )
    configure_cpu_threads(config.device)
    trainer = Trainer(resolve_source(args.source_path), config)
    if config.initialize_motion:
        trainer.initialize_motion()
    results = []
    for step in [0, config.trajectory_steps]:
        results.append(trainer.step(0, step))
    print(
        json.dumps(
            {
                "backend": config.backend,
                "device": config.device,
                "exposure_samples": config.exposure_samples,
                "initialization_notes": trainer.initialization_notes,
                "checks": results,
            },
            indent=2,
        )
    )
    if any(row["valid_fraction"] == 0 for row in results):
        raise RuntimeError("No supported motion pixels in the first view; inspect depth and caches")


if __name__ == "__main__":
    main()
