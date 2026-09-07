"""Import an already-undistorted COLMAP model into the BLUR-GS dataset format."""

import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv=None):
    from scene.colmap_loader import import_colmap

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--downscale", type=int, default=1)
    parser.add_argument("--max-points", type=int, default=10000)
    parser.add_argument("--holdout-every", type=int, default=0)
    args = parser.parse_args(argv)
    result = import_colmap(
        args.model, args.images, args.output, args.downscale, args.max_points, args.holdout_every
    )
    print(result)
    return result


if __name__ == "__main__":
    main()
