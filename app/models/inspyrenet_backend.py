from __future__ import annotations

from threading import Lock
from typing import Any

import numpy as np
from PIL import Image

from app.core.alpha_utils import apply_alpha
from app.models.base import BaseMattingBackend, MattingResult


class InspyrenetBackend(BaseMattingBackend):
    name = "inspyrenet"

    def __init__(self) -> None:
        self._remover: Any | None = None
        self._load_lock = Lock()

    def predict(self, image: Image.Image, **kwargs: object) -> MattingResult:
        rgb = image.convert("RGB")
        remover = self._get_remover()

        try:
            try:
                output = remover.process(rgb, type="rgba")
            except TypeError:
                output = remover.process(rgb)
        except Exception as exc:
            raise RuntimeError(
                "Inspyrenet inference failed. This may be caused by model download issues, "
                "missing dependencies, network access, or an unsupported CPU/GPU environment."
            ) from exc

        mask = self._extract_alpha_mask(output, rgb.size)
        rgba = apply_alpha(rgb, mask)
        return MattingResult(model_name=self.name, rgba=rgba, mask=mask)

    def _get_remover(self) -> Any:
        if self._remover is not None:
            return self._remover

        with self._load_lock:
            if self._remover is not None:
                return self._remover

            try:
                from transparent_background import Remover

                self._remover = Remover()
            except Exception as exc:
                raise RuntimeError(
                    "Failed to load Inspyrenet via transparent-background. This may be caused by "
                    "model download issues, missing dependencies, network access, or an unsupported "
                    "CPU/GPU environment."
                ) from exc

        return self._remover

    @staticmethod
    def _extract_alpha_mask(output: Any, size: tuple[int, int]) -> Image.Image:
        if isinstance(output, Image.Image):
            output_image = output
        elif isinstance(output, np.ndarray):
            array = InspyrenetBackend._to_uint8_array(output)
            if array.ndim == 2:
                output_image = Image.fromarray(array, mode="L")
            elif array.ndim == 3:
                output_image = Image.fromarray(array)
            else:
                raise RuntimeError(f"Unexpected Inspyrenet output shape: {array.shape}")
        else:
            raise RuntimeError(f"Unexpected Inspyrenet output type: {type(output).__name__}")

        if output_image.mode == "RGBA":
            mask = output_image.getchannel("A")
        elif output_image.mode == "LA":
            mask = output_image.getchannel("A")
        elif output_image.mode == "L":
            mask = output_image
        else:
            raise RuntimeError(f"Expected Inspyrenet RGBA or mask output, got mode {output_image.mode}")

        if mask.size != size:
            mask = mask.resize(size, Image.Resampling.BILINEAR)

        return mask.convert("L")

    @staticmethod
    def _to_uint8_array(array: np.ndarray) -> np.ndarray:
        if array.dtype == np.uint8:
            return array

        values = np.asarray(array)
        if np.issubdtype(values.dtype, np.floating) and values.size and values.max() <= 1.0:
            values = values * 255.0

        return np.clip(values, 0, 255).astype(np.uint8)
