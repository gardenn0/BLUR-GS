import argparse
import json


def main():
    parser = argparse.ArgumentParser(description="BLUR-GS geometry-coupled exposure reconstruction")
    commands = parser.add_subparsers(dest="command", required=True)
    synthetic = commands.add_parser("synthetic", help="Create an analytic CPU test fixture")
    synthetic.add_argument("--output", required=True)
    synthetic.add_argument("--size", type=int, default=32)
    synthetic.add_argument("--views", type=int, default=3)
    colmap = commands.add_parser("import-colmap", help="Import an undistorted COLMAP model")
    colmap.add_argument("--model", required=True)
    colmap.add_argument("--images", required=True)
    colmap.add_argument("--output", required=True)
    colmap.add_argument("--downscale", type=int, default=1)
    colmap.add_argument("--max-points", type=int, default=10000)
    colmap.add_argument("--holdout-every", type=int, default=0)
    motion = commands.add_parser(
        "prepare-motion", help="Cache official Image-as-an-IMU predictions"
    )
    motion.add_argument("--data", required=True)
    motion.add_argument("--checkpoint", required=True)
    motion.add_argument("--device", default="cuda")
    motion.add_argument("--output")
    train = commands.add_parser("train", help="Optimize geometry and trajectories")
    train.add_argument("--data", required=True)
    train.add_argument("--config")
    train.add_argument("--output", required=True)
    train.add_argument("--iterations", type=int)
    train.add_argument("--resume")
    for name in ("render", "evaluate"):
        command = commands.add_parser(
            name,
            help="Render sharp views" if name == "render" else "Evaluate against sharp references",
        )
        command.add_argument("--data", required=True)
        command.add_argument("--checkpoint", required=True)
        command.add_argument("--output", required=True)
        command.add_argument("--backend", choices=("torch", "gsplat"), default="torch")
        command.add_argument("--device", default="cpu")
        command.add_argument("--split", choices=("train", "test", "all"), default="all")
    args = parser.parse_args()
    # Limit CPU oversubscription for small research/test scenes.
    import torch

    if getattr(args, "device", "cpu") == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    if args.command == "synthetic":
        from .synthetic import make_synthetic

        print(make_synthetic(args.output, args.size, args.views))
    elif args.command == "import-colmap":
        from .colmap import import_colmap

        print(
            import_colmap(
                args.model,
                args.images,
                args.output,
                args.downscale,
                args.max_points,
                args.holdout_every,
            )
        )
    elif args.command == "prepare-motion":
        from .motion import prepare_motion

        print(prepare_motion(args.data, args.checkpoint, args.device, args.output))
    elif args.command == "train":
        from .training import read_config, train

        config = read_config(args.config)
        if args.iterations is not None:
            config.iterations = args.iterations
        print(json.dumps(train(args.data, config, args.output, args.resume), indent=2))
    else:
        from .evaluation import render_checkpoint

        print(
            json.dumps(
                render_checkpoint(
                    args.checkpoint,
                    args.data,
                    args.output,
                    args.device,
                    args.backend,
                    args.split,
                    args.command == "evaluate",
                ),
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
