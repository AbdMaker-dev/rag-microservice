"""POST /extract — document -> texte structuré.

`pdfplumber` lit la couche texte et les tableaux avec leurs coordonnées.
Quand un PDF déclare mal ses polices — accents remplacés par des caractères
parasites — l'encodage est rétabli de façon déterministe, police par police.

Aucun modèle de langue n'intervient ici. Un modèle réécrirait les passages
qu'il ne comprend pas, et sur un programme scolaire une formule inventée mais
crédible est plus dangereuse qu'un caractère manquant : le professeur repère
le second, jamais la première.

Sans état : rien n'est stocké. Le texte est rendu à l'appelant avec une note
de qualité, et c'est le prof qui relit avant d'indexer.
"""

from __future__ import annotations

import base64
import binascii
import logging

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.dependencies import require_service_token
from app.config import Settings, get_settings
from app.core.encoding import RepairPlan
from app.core.extraction import (
    UnsupportedMediaType,
    assemble,
    clean_ocr_text,
    load,
    read_pdf_document,
    to_sections,
)
from app.core.pdf_profile import profile
from app.core.quality import assess
from app.models.schemas import (
    DocumentAnalysis,
    ExtractedSection,
    ExtractionQuality,
    ExtractRequest,
    ExtractResponse,
    FontDiagnosis,
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


def _extract_pdf(payload: bytes, settings: Settings) -> tuple:
    """Lire un PDF et rétablir son encodage.

    Renvoie (texte, plan, analyse, avertissements).
    """

    described = profile(payload)
    pages, plan, fonts = read_pdf_document(
        payload, repair=settings.encoding_repair_enabled
    )
    warnings: list = []

    # Une page sans couche texte ne se rattrape pas à la lecture. On le dit,
    # au lieu de rendre une page vide sans explication.
    if described.needs_ocr:
        warnings.append("NEEDS_OCR")
        logger.info(
            "pages sans couche texte", extra={"pages": len(described.pages_needing_ocr)}
        )

    # Une police qu'aucune table connue ne rend lisible reste telle quelle. On
    # le dit plutôt que de livrer un texte à moitié faux sans le signaler.
    if plan.unreadable_fonts:
        warnings.append("UNREADABLE_FONTS")
        logger.warning(
            "polices non rétablies", extra={"fonts": len(plan.unreadable_fonts)}
        )

    analysis = DocumentAnalysis(
        tagged=described.tagged,
        text_coverage=described.text_coverage,
        pages_needing_ocr=described.pages_needing_ocr,
        fonts=[
            FontDiagnosis(
                font=font.fontname,
                table=font.encoding,
                confidence=font.confidence,
                samples=font.samples,
            )
            for font in fonts
            if font.changed
        ],
    )
    return clean_ocr_text(assemble(pages)), plan, analysis, warnings


@router.post("/extract", response_model=ExtractResponse)
async def extract(
    body: ExtractRequest,
    settings: Settings = Depends(get_settings),
) -> ExtractResponse:
    payload = _decode(body, settings)
    plan = RepairPlan(words={}, fonts={}, unreadable_fonts=[])
    analysis = None
    warnings: list = []

    try:
        if body.media_type == "application/pdf":
            text, plan, analysis, warnings = _extract_pdf(payload, settings)
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
    if measured.score < settings.text_quality_floor:
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
            "wordsRepaired": len(plan.words),
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
            words_repaired=len(plan.words),
            unreadable_fonts=plan.unreadable_fonts,
        ),
        analysis=analysis,
        warnings=warnings,
    )
