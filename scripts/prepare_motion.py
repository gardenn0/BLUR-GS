"""Cache official Image-as-an-IMU predictions before BLUR-GS training."""

import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv=None):
    from arguments import add_source_argument, configure_cpu_threads, resolve_source
    from scene.motion_prior import prepare_motion

    parser = argparse.ArgumentParser(description=__doc__)
    add_source_argument(parser)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", help="Output manifest, in the same directory as the input")
    args = parser.parse_args(argv)
    configure_cpu_threads(args.device)
    result = prepare_motion(
        resolve_source(args.source_path), args.checkpoint, args.device, args.output
    )
    print(result)
    return result


if __name__ == "__main__":
    main()
