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
    is_legacy_doc,
    read_pdf_document,
    sniff,
    to_sections,
)
from app.core.routing import Route, classify
from app.core.confidence import inspect_section
from app.core.quality import assess
from app.models.schemas import (
    DocumentAnalysis,
    ExtractedSection,
    SectionIssue,
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

    reading = read_pdf_document(payload, repair=settings.encoding_repair_enabled)
    pages, plan, fonts = reading.pages, reading.plan, reading.fonts

    decision = classify(payload, reading.script, reading.profile, reading.ruled_pages)
    described = decision.document
    logger.info(
        "voie de lecture",
        extra={"route": decision.route.value, "reason": decision.reason},
    )

    # Chaque voie reste une voie. Aujourd'hui elles convergent toutes vers la
    # lecture géométrique : c'est le seul lecteur écrit. La branche balisée
    # existe pour recevoir la lecture de l'arbre logique, qui rendra les
    # tableaux et l'ordre de lecture sans aucune heuristique — mais tant
    # qu'elle n'est pas écrite, mieux vaut une voie nommée et vide qu'un
    # chemin unique qui se fige à la racine.
    if decision.route is Route.TAGGED:
        pass  # à venir : lecture de /StructTreeRoot

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
        route=decision.route.value,
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
    logger.info("étapes de lecture", extra={"timings": reading.timings})
    return clean_ocr_text(assemble(pages)), plan, analysis, warnings


def _describe(section: dict) -> ExtractedSection:
    """Rendre une section avec sa note et l'emplacement de ses doutes.

    C'est ce qui permet au professeur de relire dix passages désignés plutôt
    que cent pages : l'interface trie par confiance et surligne aux offsets.
    """

    confidence, issues = inspect_section(section["text"])
    return ExtractedSection(
        **section,
        confidence=confidence,
        issues=[
            SectionIssue(
                kind=issue.kind, start=issue.start, end=issue.end,
                excerpt=issue.excerpt[:120],
            )
            for issue in issues
        ],
    )


@router.post("/extract", response_model=ExtractResponse)
async def extract(
    body: ExtractRequest,
    settings: Settings = Depends(get_settings),
) -> ExtractResponse:
    payload = _decode(body, settings)
    plan = RepairPlan(words={}, fonts={}, unreadable_fonts=[])
    analysis = None
    warnings: list = []

    # Le type déclaré par l'appelant n'engage que lui : un navigateur devine
    # l'extension et se trompe régulièrement. On lit la signature du fichier et
    # on suit ce qu'elle dit — sans refuser pour autant, car un professeur dont
    # le dépôt échoue à cause d'une extension mal devinée ne comprendra pas
    # pourquoi. Cela ferme au passage une porte : un fichier arbitraire annoncé
    # comme PDF n'entre plus dans un analyseur qui ne l'attend pas.
    # Le Word 97-2003 se reconnaît à ses quatre premiers octets. On ne sait
    # pas le lire, mais « convertissez en .docx » vaut mieux qu'un « type non
    # supporté » générique.
    if is_legacy_doc(payload):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail={
                "code": "LEGACY_DOC_FORMAT",
                "hint": "Document Word 97-2003 : enregistrez-le au format .docx puis renvoyez-le.",
            },
        )

    media_type = body.media_type
    actual = sniff(payload)
    if actual is not None and actual != media_type:
        logger.info(
            "type déclaré démenti par le fichier",
            extra={"declared": media_type, "actual": actual},
        )
        warnings.append("MEDIA_TYPE_MISMATCH")
        media_type = actual

    try:
        if media_type == "application/pdf":
            text, plan, analysis, warnings = _extract_pdf(payload, settings)
        else:
            text = load(payload, media_type)
            # Le contrat ne change pas avec le format : l'interface prof lit
            # les mêmes clés pour un .docx que pour un PDF. Les champs sans
            # objet — polices, pages à océriser — restent simplement vides.
            analysis = DocumentAnalysis(
                route="docx" if media_type.endswith(".document") else "text",
                tagged=False,
            )
    except UnsupportedMediaType as error:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail={"code": "UNSUPPORTED_MEDIA_TYPE", "mediaType": media_type},
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
        # Un document scanné n'a pas « échoué » : il demande un autre outil.
        # Le dire évite à l'appelant de chercher une erreur là où il n'y en a
        # pas, et lui indique quelles pages passer à l'OCR.
        if analysis is not None and analysis.pages_needing_ocr:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "code": "DOCUMENT_NEEDS_OCR",
                    "pagesNeedingOcr": analysis.pages_needing_ocr,
                    "pages": len(analysis.pages_needing_ocr),
                },
            )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"code": "DOCUMENT_HAS_NO_EXTRACTABLE_TEXT"},
        )

    measured = assess(text)
    if measured.score < settings.text_quality_floor:
        warnings.append("LOW_TEXT_QUALITY")

    # Le contenu extrait n'est jamais journalisé : c'est du matériel
    # pédagogique, parfois sous droits. Ce qui l'est, c'est ce qu'on a décidé
    # à son sujet — la voie retenue, les tables d'encodage, le coût de chaque
    # étape. Les documents qui manquent au corpus arriveront d'eux-mêmes en
    # production : ce journal est ce qui les transformera en corpus, sans
    # jamais conserver leur contenu.
    logger.info(
        "document extrait",
        extra={
            "requestId": body.request_id,
            "mediaType": media_type,
            "declaredMediaType": body.media_type,
            "characters": len(text),
            "quality": measured.score,
            "wordPlausibility": measured.word_plausibility,
            "cidMarkers": measured.cid_markers,
            "wordsRepaired": len(plan.words),
            "route": analysis.route if analysis else None,
            "tagged": analysis.tagged if analysis else None,
            "textCoverage": analysis.text_coverage if analysis else None,
            "pagesNeedingOcr": len(analysis.pages_needing_ocr) if analysis else 0,
            "fontsRepaired": [
                {"table": font.table, "samples": font.samples}
                for font in (analysis.fonts if analysis else [])
            ],
            "warnings": warnings,
        },
    )

    sections = [_describe(section) for section in to_sections(text)]
    summary: dict = {}
    for section in sections:
        for issue in section.issues:
            summary[issue.kind] = summary.get(issue.kind, 0) + 1
    analysis.issue_summary = summary
    if any(section.confidence < 0.8 for section in sections):
        warnings.append("REVIEW_RECOMMENDED")

    return ExtractResponse(
        request_id=body.request_id,
        filename=body.filename,
        media_type=media_type,
        text=text,
        characters=len(text),
        sections=sections,
        quality=ExtractionQuality(
            score=measured.score,
            word_plausibility=measured.word_plausibility,
            cid_markers=measured.cid_markers,
            words_repaired=len(plan.words),
            characters_repaired=sum(font.samples for font in analysis.fonts),
            unreadable_fonts=plan.unreadable_fonts,
        ),
        analysis=analysis,
        warnings=warnings,
    )
