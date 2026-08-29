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

# Sous cette part de la page couverte par du texte, la page est quasi vide.
# Mesuré : 0 % sur une attestation scannée, 0,8 % sur une page de titre,
# 11 à 17 % sur une page ordinaire de six documents différents.
_TEXT_FLOOR = 0.01
# Au-delà de cette part couverte par des images, la page est une image.
# Mesuré : 100 % sur les documents scannés, 0 à 8 % sur les autres.
_IMAGE_CEILING = 0.40


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


def _is_tagged(payload: bytes) -> bool:
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
        tagged=_is_tagged(payload),
        text_coverage=round(statistics.median(coverages), 4) if coverages else 0.0,
        pages_needing_ocr=needing,
    )
