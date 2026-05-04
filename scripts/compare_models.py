from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.compare_grid import make_compare_grid
from app.core.image_io import load_image, save_png
from app.core.model_router import get_backend


DEFAULT_COMPARE_MODELS = ("dummy", "inspyrenet", "inspyrenet_ben2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare phase-1 matting backends on one image.")
    parser.add_argument("input", type=Path, help="Path to an input image.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Directory for PNG outputs.")
    parser.add_argument("--edge-radius", type=int, default=8, help="Edge band radius for inspyrenet_ben2.")
    parser.add_argument("--blend", type=float, default=1.0, help="BEN2 blend amount for inspyrenet_ben2.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        image = load_image(args.input)
        stem = args.input.stem
        output_dir = args.output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        grid_items = [("input", image.convert("RGBA"))]
        model_reports: dict[str, dict[str, str]] = {}
        report: dict[str, object] = {
            "input": str(args.input.expanduser().resolve()),
            "output_dir": str(output_dir),
            "models": model_reports,
        }

        for model_name in DEFAULT_COMPARE_MODELS:
            try:
                backend = get_backend(model_name)
                result = backend.predict(image, edge_radius=args.edge_radius, blend=args.blend)
                rgba_path = save_png(result.rgba, output_dir / f"{stem}_{model_name}_rgba.png")
                mask_path = save_png(result.mask, output_dir / f"{stem}_{model_name}_mask.png")
                grid_items.append((model_name, result.rgba))
                model_reports[model_name] = {
                    "status": "ok",
                    "rgba": str(rgba_path),
                    "mask": str(mask_path),
                }
                print(f"{model_name}: {rgba_path} | {mask_path}")
            except Exception as exc:
                model_reports[model_name] = {
                    "status": "error",
                    "error": str(exc),
                }
                grid_items.append((f"{model_name} error", _error_tile(image.size, model_name, str(exc))))
                print(f"{model_name}: Error: {exc}", file=sys.stderr)
                continue

        grid_path = make_compare_grid(grid_items, output_dir / f"{stem}_compare_grid.png")
        report["compare_grid"] = str(grid_path)
        report_path = output_dir / "report.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Compare grid: {grid_path}")
    print(f"Report: {report_path}")
    return 0


def _error_tile(size: tuple[int, int], model_name: str, message: str) -> Image.Image:
    width, height = size
    tile = Image.new("RGBA", (width, height), (255, 245, 245, 255))
    draw = ImageDraw.Draw(tile)
    draw.rectangle((0, 0, width - 1, height - 1), outline=(180, 40, 40, 255), width=4)
    draw.text((16, 16), model_name, fill=(120, 20, 20, 255))
    draw.text((16, 48), "error", fill=(120, 20, 20, 255))
    draw.text((16, 80), _shorten(message), fill=(40, 40, 40, 255))
    return tile


def _shorten(message: str, limit: int = 120) -> str:
    text = " ".join(message.split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


if __name__ == "__main__":
    raise SystemExit(main())
