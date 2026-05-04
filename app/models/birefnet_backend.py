from __future__ import annotations

from PIL import Image

from app.models.base import BaseMattingBackend, MattingResult


class BiRefNetBackend(BaseMattingBackend):
    name = "birefnet"

    def predict(self, image: Image.Image, **kwargs: object) -> MattingResult:
        raise ValueError("model=birefnet is not supported in v1.5. Use dummy, inspyrenet, ben2, or inspyrenet_ben2.")
