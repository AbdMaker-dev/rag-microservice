"""POST /search — question -> passages pertinents.

Brique partagée : Générer et Répondre s'appuient dessus. Exposée pour pouvoir
la déboguer sans passer par la génération.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.api.dependencies import require_service_token
from app.core.embeddings import EmbeddingError
from app.core.retrieval import Retriever
from app.models.schemas import SearchItem, SearchRequest, SearchResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["search"], dependencies=[Depends(require_service_token)])


def _retriever(request: Request) -> Retriever:
    return request.app.state.retriever


@router.post("/search", response_model=SearchResponse)
async def search(body: SearchRequest, request: Request) -> SearchResponse:
    try:
        passages = await _retriever(request).search(
            query=body.query,
            scope=body.scope,
            limit=body.limit,
            max_excerpt_characters=body.max_excerpt_characters,
            course_id=body.course_id,
            role=body.role,
            document_ids=body.document_ids or None,
        )
    except EmbeddingError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "EMBEDDING_BACKEND_UNAVAILABLE"},
        ) from error

    # La question n'est jamais journalisée : elle peut venir d'un élève
    # identifié et porter un contexte personnel.
    logger.info(
        "recherche effectuée",
        extra={"requestId": body.request_id, "results": len(passages)},
    )

    return SearchResponse(
        request_id=body.request_id,
        evidence="SUFFICIENT" if passages else "INSUFFICIENT_EVIDENCE",
        items=[
            SearchItem(
                chunk_id=passage.chunk_id,
                document_id=passage.document_id,
                title=passage.title,
                locator=passage.locator,
                excerpt=passage.content,
                language=passage.language,
                score=passage.score,
            )
            for passage in passages
        ],
    )
