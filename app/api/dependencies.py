"""Dépendances partagées : authentification inter-services."""

from __future__ import annotations

import hmac
from typing import Optional

from fastapi import Depends, Header, HTTPException, status

from app.config import Settings, get_settings


async def require_service_token(
    x_service_token: Optional[str] = Header(default=None),
    settings: Settings = Depends(get_settings),
) -> None:
    """La plateforme est le seul appelant légitime.

    Comparaison à temps constant, et l'erreur ne renvoie jamais la valeur reçue.
    """

    if not x_service_token or not hmac.compare_digest(
        x_service_token, settings.service_shared_secret
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "SERVICE_AUTHENTICATION_REQUIRED"},
        )
