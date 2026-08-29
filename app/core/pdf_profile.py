"""Ce qu'est un PDF, avant de décider comment le lire.

Le service reçoit des documents de toutes origines. Les traiter tous de la
même façon, c'est appliquer à un fichier bien construit des heuristiques
conçues pour un fichier dégradé. On commence donc par le regarder, puis on
route.

Deux constats décident du chemin :

* **Balisé** — le PDF porte son arbre logique (titres, paragraphes, tableaux,
  ordre de lecture). Aucune heuristique géométrique ne bat cette information,
  et les chaînes modernes en produisent. C'est le cas favorable.
* **Couche texte** — une page peut n'être qu'une image. Le signal n'est pas
  « peu de texte » : une page de titre en contient peu et n'a rien à faire à
  l'OCR. C'est « peu de texte **et** beaucoup d'image » qui trahit un scan.
"""

from __future__ import annotations

import io
import logging
import statistics
from dataclasses import dataclass, field
from typing import List

logger = logging.getLogger(__name__)

# --- Seuils, et le document qui a justifié chacun ---------------------------
#
# Ils sont calés sur un corpus de sept fichiers dont un seul cas dégradé. Quand
# de vrais documents arriveront en production, c'est ici qu'il faudra revenir :
# chaque valeur porte la mesure qui l'a fixée, pour éviter l'archéologie.
#
# Part de la page couverte par du texte, en dessous de laquelle la page est
# quasi vide.  Mesuré : 0 % (attestation d'assurance scannée), 0,8 % (page 4 du
# programme national, une page de titre), 11 à 17 % (page ordinaire de six
# documents d'origines différentes).
_TEXT_FLOOR = 0.01
#
# Part couverte par des images, au-delà de laquelle la page EST une image.
# Mesuré : 100 % (attestation scannée, « Scanner 9 juil. »), 0 à 8 % (les
# autres).  Les deux seuils sont exigés ensemble : une page de titre a peu de
# texte et aucune image, un scan a peu de texte et une grande image.
_IMAGE_CEILING = 0.40
#
# Un filet plus court que la moitié de la page n'est pas une bordure de
# tableau.  Mesuré : aucune des 99 pages du programme national n'en porte deux
# — ce que pdfplumber y prend pour des bordures sont des barres de fraction —
# contre 23 pages sur 43 (PHARES Dakar) et 10 sur 27 (socle KYC).
_RULE_SPAN = 0.50
_RULES_PER_PAGE = 2
# Un trait « horizontal » n'est jamais parfaitement plat dans un PDF.
_RULE_FLATNESS = 1.5


@dataclass(frozen=True)
class PdfProfile:
    """Le portrait d'un document, avant toute extraction."""

    pages: int
    tagged: bool
    text_coverage: float
    pages_needing_ocr: List[int] = field(default_factory=list)

    @property
    def needs_ocr(self) -> bool:
        return bool(self.pages_needing_ocr)


def is_tagged(payload: bytes) -> bool:
    """Le PDF porte-t-il son arbre logique ?"""

    try:
        from pypdf import PdfReader

        root = PdfReader(io.BytesIO(payload)).trailer.get("/Root")
        return root is not None and "/StructTreeRoot" in root.get_object()
    except Exception:  # noqa: BLE001
        return False


def profile(payload: bytes) -> PdfProfile:
    """Décrire un PDF : balisage, densité de texte, pages à passer à l'OCR."""

    import pdfplumber

    coverages: List[float] = []
    needing: List[int] = []

    try:
        with pdfplumber.open(io.BytesIO(payload)) as document:
            for number, page in enumerate(document.pages, start=1):
                area = float(page.width) * float(page.height)
                if area <= 0:
                    continue
                text = (
                    sum(
                        abs(word["x1"] - word["x0"]) * abs(word["bottom"] - word["top"])
                        for word in page.extract_words()
                    )
                    / area
                )
                image = (
                    sum(
                        abs(item["x1"] - item["x0"]) * abs(item["bottom"] - item["top"])
                        for item in page.images
                    )
                    / area
                )
                coverages.append(text)
                # Peu de texte ne suffit pas : une page de titre en a peu et se
                # lit très bien. C'est la conjonction avec une grande image qui
                # désigne un scan.
                if text < _TEXT_FLOOR and image > _IMAGE_CEILING:
                    needing.append(number)
    except Exception:  # noqa: BLE001
        logger.warning("profil du document indisponible")
        return PdfProfile(pages=0, tagged=False, text_coverage=0.0)

    return PdfProfile(
        pages=len(coverages),
        tagged=is_tagged(payload),
        text_coverage=round(statistics.median(coverages), 4) if coverages else 0.0,
        pages_needing_ocr=needing,
    )


def measure_page(
    number: int,
    page,
    words,
    coverages: List[float],
    needing_ocr: List[int],
    ruled: List[int],
) -> None:
    """Relever d'une page tout ce qui ne dépend pas du texte lui-même.

    Appelée depuis la passe de lecture, pour n'ouvrir le document qu'une fois.
    Elle nourrit trois décisions : la densité de texte, le besoin d'OCR, et la
    présence de filets assez longs pour être des bordures de tableau.
    """

    area = float(page.width) * float(page.height)
    if area <= 0:
        return

    text = sum(
        abs(word["x1"] - word["x0"]) * abs(word["bottom"] - word["top"])
        for word in words
    ) / area
    image = sum(
        abs(item["x1"] - item["x0"]) * abs(item["bottom"] - item["top"])
        for item in page.images
    ) / area
    coverages.append(text)

    # Peu de texte ne suffit pas : une page de titre en a peu et se lit très
    # bien. C'est la conjonction avec une grande image qui désigne un scan.
    if text < _TEXT_FLOOR and image > _IMAGE_CEILING:
        needing_ocr.append(number)

    width = float(page.width) or 1.0
    spans = [
        abs(item["x1"] - item["x0"]) / width
        for item in list(page.lines) + list(page.rects)
        if abs(item["y1"] - item["y0"]) < _RULE_FLATNESS
    ]
    if sum(1 for span in spans if span >= _RULE_SPAN) >= _RULES_PER_PAGE:
        ruled.append(number)
