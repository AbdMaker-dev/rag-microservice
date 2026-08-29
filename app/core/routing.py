"""Choisir comment lire un document, avant de le lire.

Un PDF bien construit et un PDF dégradé ne se lisent pas de la même façon.
Appliquer à l'un les heuristiques faites pour l'autre, c'est perdre une
information que le fichier donnait, ou en inventer une qu'il ne donne pas.

Ce module ne lit rien : il classe, et nomme la voie. Chaque voie reste une
voie — aucune ne doit devenir le chemin par défaut à la racine du service.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import List

from app.core.pdf_profile import PdfProfile, profile

logger = logging.getLogger(__name__)

# Un filet plus court que la moitié de la page n'est pas une bordure de
# tableau. Mesuré : 0 page sur les 99 du programme national en a deux, contre
# 23 sur 43 dans un document à vrais tableaux et 10 sur 27 dans un autre.
_RULE_SPAN = 0.50
_RULES_PER_PAGE = 2
# Un trait « horizontal » n'est jamais parfaitement plat dans un PDF.
_RULE_FLATNESS = 1.5


class Route(str, Enum):
    """La voie de lecture retenue pour un document."""

    NON_LATIN = "non-latin"
    """Écriture que nos tables ne décrivent pas : on lit sans rien corriger."""

    NEEDS_OCR = "needs-ocr"
    """Pas de couche texte : aucune lecture ne rendra ce que l'image contient."""

    TAGGED = "tagged"
    """Le PDF porte son arbre logique. Aucune heuristique ne bat cette source."""

    UNTAGGED = "untagged"
    """Rien que la géométrie. C'est le cas dégradé, pas le cas normal."""


class PageShape(str, Enum):
    """Ce qu'une page donne à voir, dans la voie non balisée."""

    RULED = "ruled"
    """Des filets dessinent les tableaux : ils font autorité."""

    UNRULED = "unruled"
    """Aucun filet exploitable : il faut lire la disposition elle-même."""


@dataclass(frozen=True)
class Classification:
    """Ce qu'on a compris du document, et ce qu'on va en faire."""

    route: Route
    script: str
    document: PdfProfile
    ruled_pages: List[int] = field(default_factory=list)
    reason: str = ""

    @property
    def readable(self) -> bool:
        return self.route is not Route.NEEDS_OCR


def ruled_pages(payload: bytes) -> List[int]:
    """Pages dont les filets dessinent réellement des tableaux.

    On ne se fie pas à la présence de traits : un PDF de mathématiques en est
    plein, et ce sont des barres de fraction. On exige des filets assez longs
    pour être des bordures, et au moins deux, faute de quoi c'est un
    soulignement.
    """

    import pdfplumber

    found: List[int] = []
    try:
        with pdfplumber.open(io.BytesIO(payload)) as document:
            for number, page in enumerate(document.pages, start=1):
                width = float(page.width) or 1.0
                spans = [
                    abs(item["x1"] - item["x0"]) / width
                    for item in list(page.lines) + list(page.rects)
                    if abs(item["y1"] - item["y0"]) < _RULE_FLATNESS
                ]
                if sum(1 for span in spans if span >= _RULE_SPAN) >= _RULES_PER_PAGE:
                    found.append(number)
    except Exception:  # noqa: BLE001
        logger.warning("relevé des filets impossible")
    return found


def page_shape(number: int, ruled: List[int]) -> PageShape:
    return PageShape.RULED if number in ruled else PageShape.UNRULED


def classify(payload: bytes, script: str = "latin") -> Classification:
    """Déterminer la voie de lecture d'un PDF.

    L'ordre des questions est celui de leur gravité : un document sans couche
    texte ne se lit pas du tout, une écriture inconnue interdit de corriger,
    un balisage rend les heuristiques inutiles.
    """

    described = profile(payload)

    if described.pages and len(described.pages_needing_ocr) == described.pages:
        return Classification(
            Route.NEEDS_OCR, script, described,
            reason="aucune page ne porte de couche texte",
        )

    if script not in ("latin", "unknown", "mixed"):
        return Classification(
            Route.NON_LATIN, script, described,
            reason=f"écriture {script} : nos tables ne la décrivent pas",
        )

    if described.tagged:
        return Classification(
            Route.TAGGED, script, described,
            reason="le document porte son arbre logique",
        )

    ruled = ruled_pages(payload)
    return Classification(
        Route.UNTAGGED, script, described, ruled,
        reason=(
            f"{len(ruled)} page(s) à filets exploitables sur {described.pages}"
            if ruled
            else "aucun filet exploitable : la disposition seule fait foi"
        ),
    )
