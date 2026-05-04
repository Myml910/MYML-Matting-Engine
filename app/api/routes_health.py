from __future__ import annotations

from fastapi import APIRouter

from app.config import get_settings
from app.core.model_router import available_models
from app.schemas.response import HealthResponse


router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    settings = get_settings()
    return HealthResponse(
        status="ok",
        service=settings.service_name,
        device=settings.device,
        available_models=available_models(),
    )

