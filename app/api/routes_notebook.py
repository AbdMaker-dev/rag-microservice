"""Le cahier de l'élève — réparer, ranger, supprimer.

Trois routes, et une règle qui les gouverne toutes : **le rag ne reçoit
jamais une photo**. Management lit les pages avec son fournisseur d'OCR ; le
rag travaille sur du texte, comme partout ailleurs.

L'ordre est celui du produit, pas celui du code : on répare D'ABORD, Awa
relit et valide, on indexe ENSUITE. Indexer avant sa validation rangerait
dans sa base un texte qu'elle n'a pas approuvé.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from app.api.dependencies import require_service_token
from app.models.schemas import (
    NotebookDeleteResponse,
    NotebookIndexRequest,
    NotebookIndexResponse,
    NotebookProof,
    NotebookRepairRequest,
    NotebookRepairResponse,
    NotebookSegment,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["notebook"], dependencies=[Depends(require_service_token)])


@router.post("/notebook/repair", response_model=NotebookRepairResponse)
async def repair(
    body: NotebookRepairRequest, request: Request
) -> NotebookRepairResponse:
    """Confronter la transcription au contenu validé du même périmètre.

    Synchrone : la réparation ne fait que des recherches vectorielles, pas
    de génération. Elle se compte en secondes, pas en minutes — la mettre en
    file ferait attendre Awa pour rien.
    """

    service = request.app.state.notebook
    report = await service.repair(
        text=body.text, scope=body.scope, chapter=body.chapter
    )
    logger.info(
        "cahier réparé",
        extra={
            "requestId": body.request_id,
            "corrected": report.corrected,
            "toCheck": report.to_check,
            "warnings": report.warnings,
        },
    )
    return NotebookRepairResponse(
        request_id=body.request_id,
        text=report.text,
        segments=[
            NotebookSegment(
                ordinal=item["ordinal"],
                original=item["original"],
                text=item["text"],
                status=item["status"],
                reason=item.get("reason", ""),
                proof=(
                    NotebookProof(**item["proof"]) if item.get("proof") else None
                ),
            )
            for item in report.segments
        ],
        corrected=report.corrected,
        to_check=report.to_check,
        proven_from=report.proven_from,
        warnings=report.warnings,
    )


@router.post("/notebook/index", response_model=NotebookIndexResponse)
async def index(
    body: NotebookIndexRequest, request: Request
) -> NotebookIndexResponse:
    """Ranger le cours d'un élève dans SA base, après qu'il a validé.

    Rejouable : réindexer le même `document_id` remplace ses passages. Awa
    peut corriger sa transcription et renvoyer sans créer de doublon.
    """

    service = request.app.state.notebook
    result = await service.index(
        external_id=body.document_id,
        student_account_id=body.student_account_id,
        title=body.title,
        chapter=body.chapter,
        scope=body.scope,
        text=body.text,
    )
    if result["document_id"] is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"code": (result["warnings"] or ["NOTEBOOK_NOT_INDEXED"])[0]},
        )
    return NotebookIndexResponse(
        request_id=body.request_id,
        document_id=body.document_id,
        chunks=result["chunks"],
        warnings=result["warnings"],
    )


@router.delete(
    "/notebook/{document_id}", response_model=NotebookDeleteResponse
)
async def delete(
    document_id: str,
    request: Request,
    student_account_id: str = Query(..., min_length=1, alias="studentAccountId"),
) -> NotebookDeleteResponse:
    """Awa supprime SON cours. Aucune purge automatique n'existe ici.

    Le propriétaire est obligatoire dans l'URL : connaître l'identifiant du
    cours d'un autre élève ne suffit pas à le supprimer.
    """

    deleted = await request.app.state.notebooks.delete_document(
        student_account_id=student_account_id, external_id=document_id
    )
    return NotebookDeleteResponse(document_id=document_id, deleted=deleted)
