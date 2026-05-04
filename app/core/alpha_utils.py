from __future__ import annotations

import numpy as np
from PIL import Image


def ensure_mask_l(mask: Image.Image) -> Image.Image:
    return mask.convert("L")


def apply_alpha(image: Image.Image, mask: Image.Image) -> Image.Image:
    rgb = image.convert("RGB")
    alpha = ensure_mask_l(mask)
    rgba = rgb.convert("RGBA")
    rgba.putalpha(alpha)
    return rgba


def mask_to_float(mask: Image.Image) -> np.ndarray:
    return np.asarray(ensure_mask_l(mask), dtype=np.float32) / 255.0


def float_to_mask(alpha: np.ndarray) -> Image.Image:
    clipped = np.clip(alpha, 0.0, 1.0)
    return Image.fromarray((clipped * 255.0).astype(np.uint8), mode="L")

