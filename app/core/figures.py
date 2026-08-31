"""Capture des figures d'un PDF — ce qui est dessiné plutôt qu'écrit.

Les supports de maths, la géométrie surtout, portent des schémas que la
couche texte ignore : un triangle tracé, un cercle trigonométrique, mais
aussi — constaté sur nos documents réels — des formules entières posées en
image par l'export Word (`lim lnx/xᵅ` sur les progressions de l'IA Dakar).
Sans capture, tout cela disparaît du texte extrait en silence.

Le principe : repérer les zones dessinées pendant la lecture (les primitives
sont déjà parsées, ne rien rouvrir), puis rendre chaque zone en PNG avec
pypdfium2 — rendre une zone est l'opération d'un lecteur PDF, elle ne peut
pas inventer de contenu. Le service reste sans état : les images partent
dans la réponse, c'est l'appelant qui les stocke.

Ce qui fait qu'une zone est une figure et pas du décor :

- une image incorporée, quelle que soit sa taille une fois agrégée à ses
  voisines (les formules-images arrivent en morceaux de quelques points) ;
- ou un groupe d'au moins trois tracés vectoriels dont des courbes ou des
  traits obliques — les traits horizontaux/verticaux seuls sont des filets
  de tableau ou des soulignés, jamais une figure.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from typing import List

logger = logging.getLogger(__name__)

# Deux voisins à moins de 20 pt appartiennent au même dessin : c'est l'ordre
# de grandeur d'un interligne — plus loin, c'est un autre bloc.
_GAP = 20.0
# En dessous de 40 × 25 pt, une zone est un ornement (puce, icône), pas un
# contenu qu'un élève doit voir.
_MIN_WIDTH, _MIN_HEIGHT = 40.0, 25.0
# Au-delà de 80 % de la page, la « figure » est un fond ou un cadre de page.
_MAX_PAGE_SHARE = 0.8
# La relecture est humaine : au-delà de ces plafonds, on livre les premières
# figures et on le dit, plutôt que de noyer le professeur.
_MAX_PER_PAGE = 12
_MAX_PER_DOC = 60
# 150 dpi lisent confortablement une figure de manuel ; le plafond de pixels
# évite qu'une zone pleine page produise une image démesurée.
_SCALE = 150.0 / 72.0
_MAX_PIXELS = 1400
# Garde-fou sur la taille de la réponse JSON (les PNG y partent en base64).
_MAX_TOTAL_BYTES = 8_000_000


@dataclass(frozen=True)
class FigureRegion:
    """Une zone dessinée, en points PDF, origine en haut de page."""

    page: int
    x0: float
    top: float
    x1: float
    bottom: float


@dataclass(frozen=True)
class CapturedFigure:
    figure_id: str
    page: int
    width: int
    height: int
    png: bytes


def _diagonal(line: dict) -> bool:
    return abs(line["x1"] - line["x0"]) > 2 and abs(line["bottom"] - line["top"]) > 2


def collect_regions(number: int, page) -> List[FigureRegion]:
    """Les zones dessinées d'une page pdfplumber, agrégées et filtrées.

    Appelée pendant la passe de lecture unique : les listes de primitives y
    sont déjà construites, cette collecte ne rouvre rien.
    """

    boxes = []  # [x0, top, x1, bottom, has_image, count]
    for curve in page.curves:
        boxes.append([curve["x0"], curve["top"], curve["x1"], curve["bottom"], False, 1])
    for line in page.lines:
        if _diagonal(line):
            boxes.append([line["x0"], line["top"], line["x1"], line["bottom"], False, 1])
    for image in page.images:
        boxes.append([image["x0"], image["top"], image["x1"], image["bottom"], True, 1])
    if not boxes:
        return []

    merged = True
    while merged:
        merged = False
        clusters: list = []
        for box in boxes:
            for other in clusters:
                if (
                    box[0] - _GAP < other[2] and other[0] - _GAP < box[2]
                    and box[1] - _GAP < other[3] and other[1] - _GAP < box[3]
                ):
                    other[0] = min(other[0], box[0])
                    other[1] = min(other[1], box[1])
                    other[2] = max(other[2], box[2])
                    other[3] = max(other[3], box[3])
                    other[4] = other[4] or box[4]
                    other[5] += box[5]
                    merged = True
                    break
            else:
                clusters.append(list(box))
        boxes = clusters

    regions = []
    page_area = float(page.width) * float(page.height)
    for x0, top, x1, bottom, has_image, count in boxes:
        width, height = x1 - x0, bottom - top
        if width < _MIN_WIDTH or height < _MIN_HEIGHT:
            continue
        if not has_image and count < 3:
            continue
        if width * height > _MAX_PAGE_SHARE * page_area:
            continue
        regions.append(
            FigureRegion(
                page=number,
                x0=max(0.0, x0 - 6),
                top=max(0.0, top - 6),
                x1=min(float(page.width), x1 + 6),
                bottom=min(float(page.height), bottom + 6),
            )
        )
    regions.sort(key=lambda region: (region.top, region.x0))
    return regions[:_MAX_PER_PAGE]


def without_furniture(regions: List[FigureRegion]) -> List[FigureRegion]:
    """Écarter le décor : la même zone qui revient de page en page.

    Constaté sur les progressions de l'IA Dakar : l'en-tête du ministère
    (drapeau + logos) ressortait comme « figure » sur les 43 pages. Une vraie
    figure ne se répète pas aux mêmes coordonnées sur trois pages ou plus —
    ce qui le fait est un en-tête ou un pied de page. Même esprit que le
    filtre des lignes répétées du texte.
    """

    seen: dict = {}
    for region in regions:
        key = (
            round(region.x0 / 5), round(region.top / 5),
            round(region.x1 / 5), round(region.bottom / 5),
        )
        seen.setdefault(key, set()).add(region.page)
    return [
        region
        for region in regions
        if len(seen[(
            round(region.x0 / 5), round(region.top / 5),
            round(region.x1 / 5), round(region.bottom / 5),
        )]) < 3
    ]


def render_figures(payload: bytes, regions: List[FigureRegion]) -> List[CapturedFigure]:
    """Rendre chaque zone en PNG. Une zone qui échoue est passée, pas fatale."""

    if not regions:
        return []
    try:
        import pypdfium2 as pdfium
    except ImportError:  # pragma: no cover
        logger.warning("pypdfium2 absent : figures non capturées")
        return []

    captured: List[CapturedFigure] = []
    total_bytes = 0
    document = pdfium.PdfDocument(io.BytesIO(payload))
    try:
        by_page: dict = {}
        for region in regions[:_MAX_PER_DOC]:
            by_page.setdefault(region.page, []).append(region)

        index = 0
        for page_number in sorted(by_page):
            try:
                scale = _SCALE
                widest = max(
                    max(r.x1 - r.x0, r.bottom - r.top) for r in by_page[page_number]
                )
                if widest * scale > _MAX_PIXELS:
                    scale = _MAX_PIXELS / widest
                rendered = document[page_number - 1].render(scale=scale).to_pil()
            except Exception:  # noqa: BLE001
                logger.warning("page non rendue", extra={"page": page_number})
                continue
            for region in by_page[page_number]:
                index += 1
                crop = rendered.crop((
                    int(region.x0 * scale), int(region.top * scale),
                    int(region.x1 * scale), int(region.bottom * scale),
                ))
                buffer = io.BytesIO()
                crop.save(buffer, format="PNG")
                png = buffer.getvalue()
                total_bytes += len(png)
                if total_bytes > _MAX_TOTAL_BYTES:
                    logger.warning(
                        "plafond de figures atteint", extra={"captured": len(captured)}
                    )
                    return captured
                captured.append(
                    CapturedFigure(
                        figure_id=f"f{index}",
                        page=region.page,
                        width=crop.width,
                        height=crop.height,
                        png=png,
                    )
                )
    finally:
        document.close()
    return captured


def annotate(text: str, figures: List[CapturedFigure]) -> str:
    """Poser dans le texte un marqueur par figure, sous le repère de sa page.

    Le marqueur dit au professeur — et au texte indexé — qu'un dessin existe
    ici ; l'image elle-même voyage à part, dans `figures[]`. Une page sans
    repère (page au texte vide) voit ses marqueurs ajoutés en fin de document
    plutôt que perdus.
    """

    if not figures:
        return text
    by_page: dict = {}
    for figure in figures:
        by_page.setdefault(figure.page, []).append(figure)

    lines = text.split("\n")
    output = []
    for line in lines:
        output.append(line)
        if line.startswith("## p. "):
            try:
                number = int(line[6:].strip())
            except ValueError:
                continue
            for figure in by_page.pop(number, []):
                output.append("")
                output.append(f"[FIGURE {figure.figure_id} — p. {figure.page}]")

    for number in sorted(by_page):
        for figure in by_page[number]:
            output.append("")
            output.append(f"[FIGURE {figure.figure_id} — p. {figure.page}]")
    return "\n".join(output)
