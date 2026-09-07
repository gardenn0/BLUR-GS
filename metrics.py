"""Render and evaluate a checkpoint against registered sharp ground truth.

This preserves the existing PSNR/SSIM evaluation path; it does not compute LPIPS
or implicitly optimize test poses, and is not a rendered-folder-only evaluator.
"""

from render import run_render_cli


def main(argv=None):
    return run_render_cli(argv, evaluate=True)


if __name__ == "__main__":
    main()
