from __future__ import annotations

from io import BytesIO
from pathlib import Path

from PIL import Image, UnidentifiedImageError


ImageLike = Image.Image


def load_image(path: str | Path) -> ImageLike:
    image_path = Path(path).expanduser().resolve()
    if not image_path.exists():
        raise ValueError(f"Input image does not exist: {image_path}")
    try:
        with Image.open(image_path) as image:
            return image.convert("RGB")
    except UnidentifiedImageError as exc:
        raise ValueError(f"Unsupported or invalid image file: {image_path}") from exc


def load_image_from_bytes(data: bytes) -> ImageLike:
    if not data:
        raise ValueError("Uploaded file is empty")
    try:
        with Image.open(BytesIO(data)) as image:
            return image.convert("RGB")
    except UnidentifiedImageError as exc:
        raise ValueError("Uploaded file is not a valid image") from exc


def save_png(image: ImageLike, path: str | Path) -> Path:
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG")
    return output_path


def encode_png(image: ImageLike) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()

