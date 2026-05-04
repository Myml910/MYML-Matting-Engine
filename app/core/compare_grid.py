from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from PIL import Image, ImageDraw

from app.core.image_io import save_png


def make_compare_grid(
    items: Sequence[tuple[str, Image.Image]],
    output_path: str | Path,
    thumb_width: int = 320,
    label_height: int = 34,
    padding: int = 12,
) -> Path:
    if not items:
        raise ValueError("Cannot create compare grid without images")

    thumbs: list[tuple[str, Image.Image]] = []
    for label, image in items:
        rgba = image.convert("RGBA")
        ratio = thumb_width / max(1, rgba.width)
        thumb_height = max(1, int(rgba.height * ratio))
        thumb = rgba.resize((thumb_width, thumb_height), Image.Resampling.LANCZOS)
        thumbs.append((label, thumb))

    cell_height = max(thumb.height for _, thumb in thumbs) + label_height
    width = padding + len(thumbs) * (thumb_width + padding)
    height = padding * 2 + cell_height
    grid = Image.new("RGBA", (width, height), (245, 245, 245, 255))
    draw = ImageDraw.Draw(grid)

    for index, (label, thumb) in enumerate(thumbs):
        x = padding + index * (thumb_width + padding)
        y = padding + label_height
        draw.text((x, padding + 8), label, fill=(20, 20, 20, 255))
        checker = _checkerboard(thumb.size)
        grid.alpha_composite(checker, (x, y))
        grid.alpha_composite(thumb, (x, y))

    return save_png(grid, output_path)


def _checkerboard(size: tuple[int, int], square: int = 16) -> Image.Image:
    width, height = size
    image = Image.new("RGBA", size, (255, 255, 255, 255))
    draw = ImageDraw.Draw(image)
    for y in range(0, height, square):
        for x in range(0, width, square):
            if (x // square + y // square) % 2:
                draw.rectangle((x, y, x + square - 1, y + square - 1), fill=(220, 220, 220, 255))
    return image

