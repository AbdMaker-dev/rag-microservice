"""Vivacité et disponibilité."""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends

from app.config import Settings, get_settings
from app.models.schemas import HealthResponse

router = APIRouter(tags=["health"])


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@router.get("/health/live", response_model=HealthResponse)
async def live(settings: Settings = Depends(get_settings)) -> HealthResponse:
    return HealthResponse(service=settings.service_name, status="ok", timestamp=_now())
