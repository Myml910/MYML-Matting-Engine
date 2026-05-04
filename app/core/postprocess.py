from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def refine_mask(mask: Image.Image, blur_kernel: int = 9) -> Image.Image:
    """Smooth mask edges while preserving a standard 8-bit grayscale output."""

    gray = np.asarray(mask.convert("L"), dtype=np.uint8)
    kernel_size = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
    kernel_size = max(3, kernel_size)
    blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    return Image.fromarray(blurred, mode="L")

