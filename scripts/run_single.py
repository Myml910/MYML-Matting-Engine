from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.alpha_utils import apply_alpha
from app.core.image_io import load_image, save_png
from app.core.model_router import available_models, get_backend


MATTING_MODES = ("full_foreground", "main_subject", "soft_matting", "hard_mask")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one matting backend on a local image.")
    parser.add_argument("--input", required=True, type=Path, help="Path to an input image.")
    parser.add_argument("--model", default="dummy", choices=available_models(), help="Backend name.")
    parser.add_argument(
        "--mode",
        default="full_foreground",
        choices=MATTING_MODES,
        help="Matting mode.",
    )
    parser.add_argument("--output", required=True, type=Path, help="Path for the RGBA PNG output.")
    parser.add_argument("--mask-output", type=Path, help="Optional path for the L-mode mask PNG output.")
    parser.add_argument("--edge-radius", type=int, default=8, help="Edge band radius for inspyrenet_ben2.")
    parser.add_argument("--blend", type=float, default=1.0, help="BEN2 blend amount for inspyrenet_ben2.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        input_path = args.input.expanduser().resolve()
        if not input_path.exists():
            raise ValueError(f"Input image does not exist: {input_path}")

        image = load_image(args.input)
        backend = get_backend(args.model)
        result = backend.predict(image, edge_radius=args.edge_radius, blend=args.blend)
        mask = result.mask.convert("L")
        rgba = result.rgba.convert("RGBA")
        if args.mode == "hard_mask":
            mask = mask.point(lambda value: 255 if value >= 128 else 0, mode="L")
            rgba = apply_alpha(image, mask)

        rgba_path = save_png(rgba, args.output)
        mask_path = None
        if args.mask_output:
            mask_path = save_png(mask, args.mask_output)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"RGBA: {rgba_path}")
    if mask_path:
        print(f"Mask: {mask_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
