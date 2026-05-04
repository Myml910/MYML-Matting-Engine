from __future__ import annotations

import os
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
from PIL import Image

from app.core.alpha_utils import apply_alpha
from app.models.base import BaseMattingBackend, MattingResult


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BEN2_MODEL_PATH = "models_cache/BEN2_Base.pth"


class BEN2Backend(BaseMattingBackend):
    name = "ben2"

    def __init__(self) -> None:
        self._model: Any | None = None
        self._device: Any | None = None
        self._load_lock = Lock()

    def predict(self, image: Image.Image, **kwargs: object) -> MattingResult:
        rgb = image.convert("RGB")
        model = self._get_model()

        try:
            output = model.inference(rgb, refine_foreground=False)
        except TypeError:
            output = model.inference(rgb)
        except Exception as exc:
            raise RuntimeError(
                "BEN2 inference failed. Check the BEN2 weights, dependencies, and CPU/GPU environment."
            ) from exc

        mask = self._extract_alpha_mask(output, rgb.size)
        rgba = apply_alpha(rgb, mask)
        return MattingResult(model_name=self.name, rgba=rgba, mask=mask)

    def model_path(self) -> Path:
        configured = os.getenv("BEN2_MODEL_PATH", DEFAULT_BEN2_MODEL_PATH)
        path = Path(configured).expanduser()
        if not path.is_absolute():
            path = ROOT / path
        return path.resolve()

    def _get_model(self) -> Any:
        if self._model is not None:
            return self._model

        with self._load_lock:
            if self._model is not None:
                return self._model

            model_path = self.model_path()
            if not model_path.exists():
                raise FileNotFoundError(
                    "Missing BEN2 weights. Set BEN2_MODEL_PATH to a valid BEN2_Base.pth file. "
                    f"Current lookup path: {model_path}"
                )

            try:
                import torch
            except Exception as exc:
                raise RuntimeError("Failed to import torch for BEN2 inference.") from exc

            try:
                model = self._build_official_ben2_model()
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = model.to(device).eval()
                model.loadcheckpoints(str(model_path))
            except Exception as exc:
                raise RuntimeError(
                    "Failed to load BEN2. Install the official BEN2 package or put the official BEN2 "
                    "repo on PYTHONPATH, then verify BEN2_MODEL_PATH points to BEN2_Base.pth. "
                    f"Current lookup path: {model_path}"
                ) from exc

            self._model = model
            self._device = device

        return self._model

    @staticmethod
    def _build_official_ben2_model() -> Any:
        try:
            import BEN2

            return BEN2.BEN_Base()
        except Exception:
            pass

        try:
            from ben2 import BEN_Base

            return BEN_Base()
        except Exception as exc:
            raise RuntimeError("Could not import BEN2.BEN_Base or ben2.BEN_Base.") from exc

    @staticmethod
    def _extract_alpha_mask(output: Any, size: tuple[int, int]) -> Image.Image:
        if isinstance(output, list):
            if not output:
                raise RuntimeError("BEN2 returned an empty output list")
            output = output[0]

        if isinstance(output, Image.Image):
            output_image = output
        elif isinstance(output, np.ndarray):
            array = BEN2Backend._to_uint8_array(output)
            if array.ndim == 2:
                output_image = Image.fromarray(array, mode="L")
            elif array.ndim == 3:
                output_image = Image.fromarray(array)
            else:
                raise RuntimeError(f"Unexpected BEN2 output shape: {array.shape}")
        else:
            raise RuntimeError(f"Unexpected BEN2 output type: {type(output).__name__}")

        if output_image.mode == "RGBA":
            mask = output_image.getchannel("A")
        elif output_image.mode == "LA":
            mask = output_image.getchannel("A")
        elif output_image.mode == "L":
            mask = output_image
        else:
            raise RuntimeError(f"Expected BEN2 RGBA or mask output, got mode {output_image.mode}")

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
