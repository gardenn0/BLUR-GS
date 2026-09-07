"""Compatibility dispatcher for the pre-restructure `python -m blur_gs` commands."""

import argparse
import importlib
import sys


def main(argv=None):
    commands = {
        "train": "train",
        "render": "render",
        "evaluate": "metrics",
        "synthetic": "scripts.make_synthetic",
        "import-colmap": "scripts.import_colmap",
        "prepare-motion": "scripts.prepare_motion",
    }
    parser = argparse.ArgumentParser(description="BLUR-GS legacy command dispatcher")
    parser.add_argument("command", choices=commands)
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(argv[:1])
    return importlib.import_module(commands[args.command]).main(argv[1:])


if __name__ == "__main__":
    main()
