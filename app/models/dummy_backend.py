from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from app.core.alpha_utils import apply_alpha
from app.core.postprocess import refine_mask
from app.models.base import MattingResult


class DummyBackend:
    name = "dummy"

    def predict(self, image: Image.Image, **kwargs: object) -> MattingResult:
        rgb = image.convert("RGB")
        mask = self._ellipse_mask(rgb.size)
        refined_mask = refine_mask(mask)
        rgba = apply_alpha(rgb, refined_mask)
        return MattingResult(model_name=self.name, rgba=rgba, mask=refined_mask)

    @staticmethod
    def _ellipse_mask(size: tuple[int, int]) -> Image.Image:
        width, height = size
        mask = np.zeros((height, width), dtype=np.uint8)
        center = (width // 2, height // 2)
        axes = (max(1, int(width * 0.34)), max(1, int(height * 0.42)))
        cv2.ellipse(mask, center, axes, 0, 0, 360, color=255, thickness=-1)
        return Image.fromarray(mask, mode="L")
