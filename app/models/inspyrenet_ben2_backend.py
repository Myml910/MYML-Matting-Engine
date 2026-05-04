from __future__ import annotations

import numpy as np
from PIL import Image

from app.core.alpha_utils import apply_alpha
from app.core.fusion import fuse_inspyrenet_ben2_edge
from app.models.base import BaseMattingBackend, MattingResult
from app.models.ben2_backend import BEN2Backend
from app.models.inspyrenet_backend import InspyrenetBackend


class InspyrenetBEN2Backend(BaseMattingBackend):
    name = "inspyrenet_ben2"

    def __init__(self, inspyrenet: InspyrenetBackend, ben2: BEN2Backend) -> None:
        self._inspyrenet = inspyrenet
        self._ben2 = ben2

    def predict(self, image: Image.Image, **kwargs: object) -> MattingResult:
        edge_radius = int(kwargs.get("edge_radius", 8))
        blend = float(kwargs.get("blend", 1.0))

        rgb = image.convert("RGB")
        inspy_result = self._inspyrenet.predict(rgb)
        ben2_result = self._ben2.predict(rgb)

        fused_mask_array = fuse_inspyrenet_ben2_edge(
            np.asarray(inspy_result.mask.convert("L"), dtype=np.uint8),
            np.asarray(ben2_result.mask.convert("L"), dtype=np.uint8),
            edge_radius=edge_radius,
            blend=blend,
        )
        fused_mask = Image.fromarray(fused_mask_array, mode="L")
        rgba = apply_alpha(rgb, fused_mask)
        return MattingResult(model_name=self.name, rgba=rgba, mask=fused_mask)
