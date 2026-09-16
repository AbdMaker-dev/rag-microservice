"""POST /engines/test — la clé d'API du super admin marche-t-elle ?

Un aller-retour minuscule chez le fournisseur, SANS repli local : on veut
savoir si SA clé répond, pas si notre modèle le fait. Rien n'est gardé.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from app.api.dependencies import require_service_token
from app.core.engines import list_models, try_engine
from app.models.schemas import (
    EngineModelsRequest,
    EngineModelsResponse,
    EngineTestRequest,
    EngineTestResponse,
)

router = APIRouter(tags=["engines"], dependencies=[Depends(require_service_token)])


@router.post("/engines/test", response_model=EngineTestResponse)
async def engines_test(body: EngineTestRequest, request: Request) -> EngineTestResponse:
    result = await try_engine(body.engine, request.app.state.http)
    return EngineTestResponse(
        ok=result["ok"], latency_ms=result["latencyMs"], error=result["error"]
    )


@router.post("/engines/models", response_model=EngineModelsResponse)
async def engines_models(body: EngineModelsRequest, request: Request) -> EngineModelsResponse:
    """Les modèles que la clé du pays peut utiliser, pour la liste du super admin."""

    result = await list_models(
        body.provider, body.api_key.get_secret_value(), request.app.state.http
    )
    return EngineModelsResponse(**result)
