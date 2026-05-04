from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes_health import router as health_router
from app.api.routes_matting import router as matting_router
from app.config import get_settings


DEFAULT_CORS_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.service_name, version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=get_cors_origins(),
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(health_router, prefix="/api")
    app.include_router(matting_router, prefix="/api/matting")
    return app


def get_cors_origins() -> list[str]:
    configured = os.getenv("CORS_ORIGINS")
    if not configured:
        return DEFAULT_CORS_ORIGINS
    return [origin.strip() for origin in configured.split(",") if origin.strip()]


app = create_app()
