"""GET /documents — ce que le prof voit dans l'écran de création.

Le texte original est stocké entier à l'indexation : les passages de `chunks`
sont découpés et se recouvrent, ils servent à chercher, pas à relire.
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from app.api.dependencies import require_service_token
from app.db.repository import IndexRepository
from app.models.schemas import (
    DocumentDetail,
    DocumentListResponse,
    DocumentSummary,
    Scope,
)

router = APIRouter(tags=["documents"], dependencies=[Depends(require_service_token)])


def _repository(request: Request) -> IndexRepository:
    return request.app.state.repository


def _summary(row: dict) -> DocumentSummary:
    return DocumentSummary(
        document_id=row["external_id"],
        course_id=row.get("course_id", ""),
        role=row.get("role", "support-cours"),
        title=row["title"],
        source_reference=row["source_reference"] or "",
        characters=row["characters"],
        chunks=row["chunk_count"],
        embedding_model=row["embedding_model"],
        indexed_at=row["indexed_at"].isoformat(),
    )


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(
    request: Request,
    country: str = Query(...),
    subject: str = Query(...),
    grade: str = Query(...),
    curriculum_version: str = Query(..., alias="curriculumVersion"),
    language: str = Query("fr"),
    # Optionnels : vides, ils ne filtrent pas — les documents indexés avant
    # l'introduction du niveau et de la série restent visibles.
    level: str = Query(""),
    track: str = Query(""),
    role: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> DocumentListResponse:
    scope = Scope(
        country=country,
        subject=subject,
        grade=grade,
        curriculum_version=curriculum_version,
        language=language,
        level=level,
        track=track,
    )
    repository = _repository(request)
    rows = await repository.list_documents(
        scope=scope, limit=limit, offset=offset, role=role
    )
    total = await repository.count_documents(scope=scope, role=role)
    return DocumentListResponse(
        total=total, documents=[_summary(row) for row in rows]
    )


@router.get("/courses/{course_id}/documents", response_model=DocumentListResponse)
async def list_course_documents(
    course_id: str, request: Request
) -> DocumentListResponse:
    """Les documents déjà validés d'un cours — l'écran de création les affiche.

    Un cours vide rend une liste vide, pas une erreur : c'est l'état normal
    d'un cours qui vient d'être créé.
    """

    rows = await _repository(request).list_course_documents(course_id=course_id)
    return DocumentListResponse(
        total=len(rows), documents=[_summary(row) for row in rows]
    )


@router.get("/documents/{document_id}", response_model=DocumentDetail)
async def get_document(document_id: str, request: Request) -> DocumentDetail:
    row = await _repository(request).get_document(document_id)
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "DOCUMENT_NOT_FOUND"},
        )
    summary = _summary(row)
    return DocumentDetail(
        **summary.model_dump(),
        scope=Scope(
            country=row["country"],
            subject=row["subject"],
            grade=row["grade"],
            curriculum_version=row["curriculum_version"],
            language=row["language"],
        ),
        content=row["content"],
    )
