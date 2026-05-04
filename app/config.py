from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


DeviceName = Literal["cpu", "cuda", "mps"]


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    model_config = SettingsConfigDict(env_prefix="MYML_MATTING_", env_file=".env", extra="ignore")

    service_name: str = "MYML-Matting-Engine"
    device: DeviceName = Field(default="cpu")
    max_upload_mb: int = Field(default=25, ge=1)


@lru_cache
def get_settings() -> Settings:
    return Settings()

