from __future__ import annotations

from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Protocol

from PIL import Image


@dataclass(frozen=True)
class MattingResult:
    model_name: str
    rgba: Image.Image
    mask: Image.Image


class BaseMattingBackend(ABC):
    name: str

    @abstractmethod
    def predict(self, image: Image.Image, **kwargs: object) -> MattingResult:
        """Return an RGBA foreground image and an 8-bit grayscale alpha mask."""


class MattingBackend(Protocol):
    name: str

    def predict(self, image: Image.Image, **kwargs: object) -> MattingResult:
        """Return an RGBA foreground image and an 8-bit grayscale alpha mask."""
