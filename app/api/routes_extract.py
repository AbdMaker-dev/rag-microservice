"""POST /extract — document -> texte structuré.

Deux voies, choisies automatiquement :

  * `pdfplumber` lit la couche texte et les tableaux, avec leurs coordonnées.
    Rapide, exact, c'est le cas courant.
  * quand une page n'a pas de couche texte, ou qu'elle est illisible parce
    que ses polices sont mal déclarées, le modèle de vision relit l'image.

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
from app.core.vision import VisionUnavailable, render_pages
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
    """Lire un PDF, page par page, en relisant par vision celles qui le méritent."""

    pages = read_pdf_pages(payload)
    warnings: list = []

    if not settings.vision_enabled:
        return clean_ocr_text(assemble(pages)), [], warnings

    # Une page vide n'a pas de couche texte ; une page mal notée en a une mais
    # illisible. Les deux cas se traitent pareil : on regarde l'image.
    to_reread = sorted(
        set(index for index, page in enumerate(pages, start=1) if not page.strip())
        | set(corrupted_pages(pages, plausibility_floor=settings.vision_quality_threshold))
    )

    if not to_reread:
        return clean_ocr_text(assemble(pages)), [], warnings

    if len(to_reread) > settings.vision_max_pages:
        warnings.append("TOO_MANY_PAGES_FOR_VISION")
        logger.warning(
            "trop de pages à relire, vision abandonnée",
            extra={"pages": len(to_reread), "limit": settings.vision_max_pages},
        )
        return clean_ocr_text(assemble(pages)), [], warnings

    reader = request.app.state.vision
    try:
        images = render_pages(payload, to_reread, scale=settings.vision_render_scale)
        transcriptions = await reader.read_pages(images)
    except VisionUnavailable as error:
        logger.warning("vision indisponible, on garde la couche texte")
        warnings.append("VISION_UNAVAILABLE")
        return clean_ocr_text(assemble(pages)), [], warnings
    except Exception:  # noqa: BLE001
        logger.exception("échec de la relecture par vision")
        warnings.append("VISION_FAILED")
        return clean_ocr_text(assemble(pages)), [], warnings

    for number, transcription in zip(to_reread, transcriptions):
        pages[number - 1] = transcription

    return clean_ocr_text(assemble(pages)), to_reread, warnings


@router.post("/extract", response_model=ExtractResponse)
async def extract(
    body: ExtractRequest,
    request: Request,
    settings: Settings = Depends(get_settings),
) -> ExtractResponse:
    payload = _decode(body, settings)
    read_by_vision: list = []
    warnings: list = []

    try:
        if body.media_type == "application/pdf":
            text, read_by_vision, warnings = await _extract_pdf(payload, request, settings)
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
    if measured.score < settings.vision_quality_threshold:
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
            "visionPages": len(read_by_vision),
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
            pages_read_by_vision=read_by_vision,
        ),
        warnings=warnings,
    )
