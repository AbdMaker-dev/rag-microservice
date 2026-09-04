"""POST /index — texte -> passages + vecteurs, rangés en base.

DELETE /index/{document_id} — retirer un document dépublié.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.api.dependencies import require_service_token
from app.config import Settings, get_settings
from app.core.chunking import chunk_document, default_token_counter
from app.core.embeddings import EmbeddingError, EmbeddingProvider
from app.db.repository import ChunkRow, IndexRepository
from app.models.schemas import DeleteResponse, IndexRequest, IndexResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["index"], dependencies=[Depends(require_service_token)])


def _repository(request: Request) -> IndexRepository:
    return request.app.state.repository


def _embeddings(request: Request) -> EmbeddingProvider:
    return request.app.state.embeddings


@router.post("/index", response_model=IndexResponse)
async def index(
    body: IndexRequest,
    request: Request,
    settings: Settings = Depends(get_settings),
) -> IndexResponse:
    # Un support appartient à un cours ; un programme officiel n'appartient à
    # aucun — il fait référence pour tout le périmètre. Les deux erreurs
    # inverses sont refusées : un support orphelin serait introuvable à la
    # génération, un programme rattaché à un cours cesserait d'être commun.
    if body.role in ("support-cours", "cours-publie") and not body.course_id:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"code": "COURSE_ID_REQUIRED_FOR_COURSE_MATERIAL"},
        )
    if body.role in ("programme-officiel", "annale") and body.course_id:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"code": "OFFICIAL_CURRICULUM_HAS_NO_COURSE"},
        )

    chunks = chunk_document(
        body.text,
        max_tokens=settings.chunk_max_tokens,
        overlap_tokens=settings.chunk_overlap_tokens,
        min_tokens=settings.chunk_min_tokens,
        count_tokens=default_token_counter(),
    )
    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"code": "NO_CHUNK_PRODUCED"},
        )

    embeddings = _embeddings(request)
    try:
        vectors = await embeddings.embed([chunk.content for chunk in chunks])
    except EmbeddingError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "EMBEDDING_BACKEND_UNAVAILABLE"},
        ) from error

    dimensions = {len(vector) for vector in vectors}
    if len(dimensions) != 1:
        # Des vecteurs de dimensions différentes rendraient la table
        # inutilisable : pgvector ne sait pas les comparer entre eux.
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={"code": "INCONSISTENT_EMBEDDING_DIMENSION"},
        )
    dimension = next(iter(dimensions))

    await _repository(request).replace_document(
        external_id=body.document_id,
        course_id=body.course_id,
        role=body.role,
        title=body.title,
        source_reference=body.source_reference,
        scope=body.scope,
        embedding_model=embeddings.model,
        embedding_dimension=dimension,
        content=body.text,
        characters=len(body.text),
        chunks=[
            ChunkRow(
                ordinal=chunk.ordinal,
                locator=chunk.locator,
                content=chunk.content,
                token_count=chunk.token_count,
                embedding=vector,
            )
            for chunk, vector in zip(chunks, vectors)
        ],
    )

    logger.info(
        "document indexé",
        extra={
            "requestId": body.request_id,
            "documentId": body.document_id,
            "chunks": len(chunks),
            "dimension": dimension,
        },
    )

    return IndexResponse(
        request_id=body.request_id,
        document_id=body.document_id,
        chunks=len(chunks),
        characters=len(body.text),
        embedding_model=embeddings.model,
        embedding_dimension=dimension,
    )


@router.delete("/index/{document_id}", response_model=DeleteResponse)
async def remove(document_id: str, request: Request) -> DeleteResponse:
    deleted = await _repository(request).delete_document(document_id)
    logger.info("document retiré de l'index", extra={"documentId": document_id, "deleted": deleted})
    return DeleteResponse(document_id=document_id, deleted=deleted)
