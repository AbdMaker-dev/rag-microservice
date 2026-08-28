"""Point d'entrée FastAPI.

Frontières du service :
  * il ne décide rien — le périmètre lui est transmis à chaque appel ;
  * il n'est pas public — seule la plateforme l'appelle, par secret partagé ;
  * il ne garde aucun état de conversation.

Extraire et Indexer sont deux appels distincts : la plateforme relit le texte
extrait avant de l'indexer, ce qui compte quand il vient d'un OCR.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.api import (
    routes_documents,
    routes_extract,
    routes_health,
    routes_index,
    routes_search,
)
from app.config import get_settings
from app.core.embeddings import build_embedding_provider
from app.core.logging import configure_logging
from app.db.engine import Database
from app.core.retrieval import Retriever
from app.core.repair import TextRepairer
from app.db.repository import IndexRepository

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings)

    database = Database(settings)
    await database.connect()
    await database.migrate()

    client = httpx.AsyncClient(timeout=settings.inference_timeout_s)

    app.state.settings = settings
    app.state.http = client
    app.state.database = database
    app.state.repository = IndexRepository(database.pool)
    embeddings = build_embedding_provider(settings, client)
    app.state.embeddings = embeddings
    app.state.repairer = TextRepairer(settings, client)
    app.state.retriever = Retriever(
        embeddings=embeddings, repository=app.state.repository
    )

    logger.info(
        "service démarré",
        extra={
            "environment": settings.environment,
            "embeddingModel": settings.embedding_model,
            "repairModel": settings.repair_model,
        },
    )
    try:
        yield
    finally:
        await client.aclose()
        await database.close()


def create_app() -> FastAPI:
    app = FastAPI(
        title="LawalSchool RAG microservice",
        version="0.1.0",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url="/openapi.json",
    )

    app.include_router(routes_health.router)
    app.include_router(routes_extract.router)
    app.include_router(routes_index.router)
    app.include_router(routes_documents.router)
    app.include_router(routes_search.router)

    @app.exception_handler(Exception)
    async def unhandled(request, exc):  # noqa: ANN001, ARG001
        # Ne jamais laisser fuir un message interne : il peut contenir un
        # fragment de document pédagogique.
        logger.exception("unhandled error")
        return JSONResponse(status_code=500, content={"detail": {"code": "INTERNAL_ERROR"}})

    return app


app = create_app()
