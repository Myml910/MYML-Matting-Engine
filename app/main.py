from __future__ import annotations

from fastapi import FastAPI

from app.api.routes_health import router as health_router
from app.api.routes_matting import router as matting_router
from app.config import get_settings


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.service_name, version="0.1.0")
    app.include_router(health_router, prefix="/api")
    app.include_router(matting_router, prefix="/api/matting")
    return app


app = create_app()

