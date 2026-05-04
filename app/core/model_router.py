from __future__ import annotations

from app.models.base import MattingBackend
from app.models.ben2_backend import BEN2Backend
from app.models.birefnet_backend import BiRefNetBackend
from app.models.dummy_backend import DummyBackend
from app.models.inspyrenet_backend import InspyrenetBackend
from app.models.inspyrenet_ben2_backend import InspyrenetBEN2Backend


_DUMMY = DummyBackend()
_INSPYRENET = InspyrenetBackend()
_BEN2 = BEN2Backend()

_BACKENDS: dict[str, MattingBackend] = {
    "dummy": _DUMMY,
    "auto": _INSPYRENET,
    "inspyrenet": _INSPYRENET,
    "inspyrenet_ben2": InspyrenetBEN2Backend(_INSPYRENET, _BEN2),
    "birefnet": BiRefNetBackend(),
    "ben2": _BEN2,
}


def available_models() -> list[str]:
    return list(_BACKENDS.keys())


def get_backend(model_name: str) -> MattingBackend:
    try:
        return _BACKENDS[model_name]
    except KeyError as exc:
        names = ", ".join(available_models())
        raise ValueError(f"Unknown model '{model_name}'. Available models: {names}") from exc
