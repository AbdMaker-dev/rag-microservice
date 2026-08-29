"""POST /extract — document -> texte structuré.

`pdfplumber` lit la couche texte et les tableaux avec leurs coordonnées.
Quand une page ressort illisible — polices mal déclarées, accents remplacés
par des caractères parasites — le modèle de langue la réécrit.

Sans état : rien n'est stocké. Le texte est rendu à l'appelant avec une note
de qualité, et c'est le prof qui relit avant d'indexer.
"""

from __future__ import annotations

import base64
import binascii
import logging

from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.api.dependencies import require_service_token
from app.config import Settings, get_settings
from app.core.extraction import (
    UnsupportedMediaType,
    assemble,
    clean_ocr_text,
    load,
    read_pdf_pages,
    to_sections,
)
from app.core.quality import assess, corrupted_pages
from app.core.repair import RepairUnavailable
from app.models.schemas import (
    ExtractedSection,
    ExtractionQuality,
    ExtractRequest,
    ExtractResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["extract"], dependencies=[Depends(require_service_token)])


def _decode(request: ExtractRequest, settings: Settings) -> bytes:
    try:
        payload = base64.b64decode(request.content_base64, validate=True)
    except (binascii.Error, ValueError) as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"code": "INVALID_BASE64_PAYLOAD"},
        ) from error
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"code": "EMPTY_DOCUMENT"},
        )
    if len(payload) > settings.max_document_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={"code": "DOCUMENT_TOO_LARGE", "maxBytes": settings.max_document_bytes},
        )
    return payload


async def _extract_pdf(
    payload: bytes, request: Request, settings: Settings
) -> tuple:
    """Lire un PDF, page par page, en réparant celles qui sont illisibles."""

    pages = read_pdf_pages(payload)
    warnings: list = []

    if not settings.repair_enabled:
        return clean_ocr_text(assemble(pages)), [], warnings

    # Une page vide n'a pas de couche texte du tout : rien à réparer, seule
    # une relecture de l'image la sauverait, et on y a renoncé.
    damaged = [
        number
        for number in corrupted_pages(
            pages, plausibility_floor=settings.repair_quality_threshold
        )
        if pages[number - 1].strip()
    ]

    if not damaged:
        return clean_ocr_text(assemble(pages)), [], warnings

    if len(damaged) > settings.repair_max_pages:
        warnings.append("TOO_MANY_PAGES_TO_REPAIR")
        logger.warning(
            "trop de pages à réparer, on garde le texte brut",
            extra={"pages": len(damaged), "limit": settings.repair_max_pages},
        )
        return clean_ocr_text(assemble(pages)), [], warnings

    repairer = request.app.state.repairer
    try:
        repaired = await repairer.repair_pages([pages[n - 1] for n in damaged])
    except RepairUnavailable:
        logger.warning("réparation indisponible, on garde le texte brut")
        warnings.append("REPAIR_UNAVAILABLE")
        return clean_ocr_text(assemble(pages)), [], warnings
    except Exception:  # noqa: BLE001
        logger.exception("échec de la réparation")
        warnings.append("REPAIR_FAILED")
        return clean_ocr_text(assemble(pages)), [], warnings

    for number, text in zip(damaged, repaired):
        pages[number - 1] = text

    return clean_ocr_text(assemble(pages)), damaged, warnings


@router.post("/extract", response_model=ExtractResponse)
async def extract(
    body: ExtractRequest,
    request: Request,
    settings: Settings = Depends(get_settings),
) -> ExtractResponse:
    payload = _decode(body, settings)
    repaired_pages: list = []
    warnings: list = []

    try:
        if body.media_type == "application/pdf":
            text, repaired_pages, warnings = await _extract_pdf(payload, request, settings)
        else:
            text = load(payload, body.media_type)
    except UnsupportedMediaType as error:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail={"code": "UNSUPPORTED_MEDIA_TYPE", "mediaType": body.media_type},
        ) from error
    except HTTPException:
        raise
    except Exception as error:  # noqa: BLE001
        logger.exception("extraction impossible")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"code": "DOCUMENT_UNREADABLE"},
        ) from error

    if not text.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"code": "DOCUMENT_HAS_NO_EXTRACTABLE_TEXT"},
        )

    measured = assess(text)
    if measured.score < settings.repair_quality_threshold:
        warnings.append("LOW_TEXT_QUALITY")

    # Le contenu extrait n'est jamais journalisé : c'est du matériel
    # pédagogique, parfois sous droits.
    logger.info(
        "document extrait",
        extra={
            "requestId": body.request_id,
            "mediaType": body.media_type,
            "characters": len(text),
            "quality": measured.score,
            "repairedPages": len(repaired_pages),
        },
    )

    return ExtractResponse(
        request_id=body.request_id,
        filename=body.filename,
        media_type=body.media_type,
        text=text,
        characters=len(text),
        sections=[ExtractedSection(**section) for section in to_sections(text)],
        quality=ExtractionQuality(
            score=measured.score,
            word_plausibility=measured.word_plausibility,
            cid_markers=measured.cid_markers,
            pages_repaired=repaired_pages,
        ),
        warnings=warnings,
    )
