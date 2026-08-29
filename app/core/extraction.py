"""Document parsers.

The caller resolves rights and passes bytes it is allowed to use. This module
never fetches a remote URL and never touches the filesystem outside the paths
given to it.
"""

from __future__ import annotations

import io
import logging
import re
import shutil
import subprocess
import tempfile
from typing import Callable, Dict, List

from app.core.encoding import RepairPlan, apply_plan, build_plan
from app.core.pdf_encoding import repair_pdf
from app.core.quality import assess

_CONTROL_CHARACTERS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_EXCESS_BLANK_LINES = re.compile(r"\n{3,}")
_TRAILING_SPACES = re.compile(r"[ \t]+\n")

# En dessous, le texte est assez mauvais pour justifier le filet de secours.
_RESCUE_FLOOR = 0.90


class UnsupportedMediaType(ValueError):
    """Le type de fichier envoyé n'est pas pris en charge."""


class OcrUnavailable(RuntimeError):
    """OCRmyPDF n'est pas disponible dans l'image."""


class OcrFailed(RuntimeError):
    """L'OCR a échoué sur ce document."""


def normalise(text: str) -> str:
    """Clean extracted text without destroying its structure.

    Blank lines are the signal the chunker uses to find block boundaries, so
    they are collapsed but never removed.
    """

    cleaned = _CONTROL_CHARACTERS.sub("", text.replace("\r\n", "\n").replace("\r", "\n"))
    cleaned = _TRAILING_SPACES.sub("\n", cleaned)
    cleaned = _EXCESS_BLANK_LINES.sub("\n\n", cleaned)
    return cleaned.strip()


def load_plain_text(payload: bytes) -> str:
    return normalise(payload.decode("utf-8", errors="replace"))


logger = logging.getLogger(__name__)


def _table_to_markdown(table) -> str:
    """Rendre un tableau en Markdown, une ligne par ligne du tableau.

    Le programme officiel EST un tableau : semaine, partie, leçon,
    compétences. Le rendre linéairement entrelacerait les colonnes et le
    rendrait illisible, pour un humain comme pour un modèle.
    """

    rows = []
    for row in table:
        cells = [
            " ".join((cell or "").split()) for cell in row
        ]
        if any(cells):
            rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def _clamp(box, page_box):
    """Ramener une boîte dans les limites de sa page."""

    left, top, right, bottom = box
    page_left, page_top, page_right, page_bottom = page_box
    return (
        max(left, page_left),
        max(top, page_top),
        min(right, page_right),
        min(bottom, page_bottom),
    )


def _read_pdf_text(payload: bytes) -> tuple:
    """Lire un PDF en respectant sa mise en page. Renvoie (texte, nb de pages).

    pdfplumber connaît la position de chaque mot : les tableaux sont extraits
    comme tableaux, et le texte hors tableau garde son ordre de lecture.
    """

    try:
        import pdfplumber
    except ImportError as error:  # pragma: no cover
        raise UnsupportedMediaType("pdfplumber n'est pas installé") from error

    pages = []
    with pdfplumber.open(io.BytesIO(payload)) as document:
        page_count = len(document.pages)
        for index, page in enumerate(document.pages, start=1):
            parts = []

            tables = page.find_tables()
            for table in tables:
                rendered = _table_to_markdown(table.extract())
                if rendered:
                    parts.append(rendered)

            # Le texte situé hors des tableaux, dans son ordre de lecture.
            outside = page
            for table in tables:
                # Un tableau peut déborder la page d'une fraction de point,
                # par arrondi. pdfplumber refuse alors la découpe et fait
                # échouer tout le document : on ramène la boîte dans la page.
                try:
                    outside = outside.outside_bbox(_clamp(table.bbox, page.bbox))
                except ValueError:
                    logger.warning("tableau hors page, découpe ignorée")
            remaining = outside.extract_text(layout=False) or ""
            if remaining.strip():
                parts.append(remaining.strip())

            # Le numéro de page devient un titre, donc un repère de
            # citation : un élève doit pouvoir être renvoyé à la bonne page.
            pages.append("\n\n".join(parts) if parts else "")

    rendered = "\n\n".join(
        f"## p. {number}\n\n{body}" for number, body in enumerate(pages, start=1) if body.strip()
    )
    return normalise(rendered), page_count


def read_pdf_document(payload: bytes, repair: bool = True) -> tuple:
    """Lire un PDF et corriger son encodage. Renvoie (pages, plan, polices).

    Deux passes sur une seule ouverture du fichier. La première relève le
    texte de chaque page et la police de chaque mot ; la seconde applique la
    correction, qui ne peut être décidée qu'une fois tout le document vu —
    une police se juge sur l'ensemble de ses mots, pas sur une page.
    """

    try:
        import pdfplumber
    except ImportError as error:  # pragma: no cover
        raise UnsupportedMediaType("pdfplumber n'est pas installé") from error

    fonts: list = []
    if repair:
        # On corrige d'abord le PDF lui-même : chaque police reçoit la table
        # qui décode réellement son texte. Le document sort alors juste dès la
        # première lecture, au lieu d'être rattrapé mot à mot ensuite.
        try:
            payload, fonts = repair_pdf(payload)
        except Exception:  # noqa: BLE001
            logger.warning("réécriture du PDF impossible, lecture telle quelle")

    pages: list = []
    pages_words: list = []

    with pdfplumber.open(io.BytesIO(payload)) as document:
        for page in document.pages:
            parts = []
            tables = page.find_tables()
            for table in tables:
                rendered = _table_to_markdown(table.extract())
                if rendered:
                    parts.append(rendered)
            outside = page
            for table in tables:
                # Un tableau peut déborder la page d'une fraction de point,
                # par arrondi. pdfplumber refuse alors la découpe et fait
                # échouer tout le document : on ramène la boîte dans la page.
                try:
                    outside = outside.outside_bbox(_clamp(table.bbox, page.bbox))
                except ValueError:
                    logger.warning("tableau hors page, découpe ignorée")
            remaining = outside.extract_text(layout=False) or ""
            if remaining.strip():
                parts.append(remaining.strip())
            pages.append("\n\n".join(parts))

            try:
                pages_words.append(page.extract_words(extra_attrs=["fontname"]))
            except Exception:  # noqa: BLE001
                # Un PDF peut refuser de livrer ses polices. On garde alors le
                # texte tel quel : sans police, aucune correction n'est sûre.
                logger.warning("polices illisibles sur une page, correction ignorée")
                pages_words.append([])

    # Rattrapage mot à mot : filet de secours, jamais routine. Il ignore les
    # polices, donc il confond un « ² » légitime avec un symbole Symbol et
    # écrit « ax≤ » à la place de « ax² ». On ne l'appelle que si la
    # correction des polices n'a rien pu faire et que le texte reste mauvais.
    plan = RepairPlan({}, {}, [])
    if repair and not any(font.changed for font in fonts):
        if assess(assemble(pages)).score < _RESCUE_FLOOR:
            plan = build_plan(pages_words)
            if not plan.is_empty:
                pages = [apply_plan(page, plan) for page in pages]
                logger.info("rattrapage mot à mot", extra={"words": len(plan.words)})
    return pages, plan, fonts


def read_pdf_pages(payload: bytes) -> list:
    """Texte de chaque page, dans l'ordre. Une page vide reste une chaîne vide."""

    pages, _, _ = read_pdf_document(payload)
    return pages


def assemble(pages: list) -> str:
    """Recomposer un document à partir de ses pages, avec les repères."""

    return normalise(
        "\n\n".join(
            f"## p. {number}\n\n{body.strip()}"
            for number, body in enumerate(pages, start=1)
            if body and body.strip()
        )
    )


def needs_ocr(text: str, page_count: int, minimum_per_page: int) -> bool:
    """Un PDF de scans n'a pas de couche texte, ou presque pas."""

    if page_count <= 0:
        return False
    return len(text.strip()) < minimum_per_page * page_count


def load_pdf(payload: bytes, **_ignored) -> str:
    """Lecture simple d'un PDF, sans réparation.

    La route `/extract` utilise `read_pdf_pages` puis répare les pages
    illisibles. Cette fonction reste pour les usages hors ligne : scripts,
    tests, exploration.
    """

    pages, _, _ = read_pdf_document(payload)
    return clean_ocr_text(assemble(pages))


def load_docx(payload: bytes) -> str:
    try:
        import docx
    except ImportError as error:  # pragma: no cover
        raise UnsupportedMediaType("python-docx is not installed") from error

    document = docx.Document(io.BytesIO(payload))
    blocks = []
    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if not text:
            continue
        style = (paragraph.style.name or "").lower()
        if style.startswith("heading"):
            level = "".join(character for character in style if character.isdigit()) or "1"
            blocks.append(f"{'#' * min(int(level), 6)} {text}")
        else:
            blocks.append(text)
    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            if any(cells):
                blocks.append(" | ".join(cells))
    return normalise("\n\n".join(blocks))


_LOADERS: Dict[str, Callable[[bytes], str]] = {
    "text/plain": load_plain_text,
    "text/markdown": load_plain_text,
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": load_docx,
}


def load(payload: bytes, media_type: str, **pdf_options) -> str:
    if media_type == "application/pdf":
        return load_pdf(payload, **pdf_options)
    loader = _LOADERS.get(media_type)
    if loader is None:
        raise UnsupportedMediaType(f"type non supporté : {media_type}")
    return loader(payload)


# --------------------------------------------------------------------------- sections

_SECTION_SPLIT = re.compile(r"\n\s*\n")
_HEADING_LINE = re.compile(r"^\s{0,3}#{1,6}\s+(?P<title>.*\S)\s*$")


def to_sections(text: str) -> list:
    """Découper le texte extrait en blocs, en gardant le chemin des titres.

    Le repère sert aux citations : un élève doit pouvoir être renvoyé à
    l'endroit exact du document source, pas à un fragment anonyme.
    """

    sections = []
    heading_path: list = []
    position = 0

    for raw in _SECTION_SPLIT.split(text):
        block = raw.strip()
        if not block:
            continue

        body_lines = []
        for line in block.split("\n"):
            heading = _HEADING_LINE.match(line)
            if heading:
                level = len(line) - len(line.lstrip("# \t"))
                level = max(1, line.strip().count("#", 0, 6) or 1)
                del heading_path[level - 1 :]
                heading_path.append(heading.group("title").strip())
                continue
            body_lines.append(line)

        body = "\n".join(body_lines).strip()
        if not body:
            continue

        locator = " > ".join(heading_path) if heading_path else f"§{position + 1}"
        sections.append(
            {
                "position": position,
                "locator": locator,
                "text": body,
                "characters": len(body),
            }
        )
        position += 1

    return sections


# --------------------------------------------------------------------- nettoyage

_PAGE_MARKER = re.compile(r"^## p\. \d+$", re.MULTILINE)
_HYPHENATED = re.compile(r"(\w)-\n(\w)", re.UNICODE)
_LIST_ITEM = re.compile(r"^\s*([-•*—]|\d+[.)])\s+")
_ENDS_SENTENCE = re.compile(r"[.!?…:;»\"]\s*$")


def _repeated_lines(pages: list, threshold: float = 0.2) -> set:
    """Repérer les en-têtes et pieds de page.

    Une ligne qui revient à l'identique sur beaucoup de pages est un
    habillage, pas du contenu. Le seuil est bas volontairement : un fichier
    peut contenir plusieurs progressions à la suite, chacune avec son propre
    pied de page, et aucune n'atteindrait alors la majorité des pages.
    On ne cherche qu'à partir de 3 pages, sinon une répétition est fortuite.
    """

    if len(pages) < 3:
        return set()

    counts: Dict[str, int] = {}
    for page in pages:
        lines = [line.strip() for line in page.split("\n") if line.strip()]
        for line in set(lines):
            # Les lignes de tableau sont du contenu, jamais de l'habillage :
            # les exclure évite de supprimer une ligne de programme répétée.
            if line.startswith("|"):
                continue
            if 3 <= len(line) <= 120:
                counts[line] = counts.get(line, 0) + 1

    minimum = max(3, int(len(pages) * threshold))
    return {line for line, count in counts.items() if count >= minimum}


def _join_broken_lines(text: str) -> str:
    """Recoller les lignes coupées au milieu d'une phrase.

    Un OCR retourne à la ligne à chaque ligne physique du scan. On ne recolle
    que lorsque la ligne suivante continue visiblement la phrase : titres,
    listes et fins de phrase sont laissés intacts.
    """

    lines = text.split("\n")
    output: List[str] = []

    for line in lines:
        stripped = line.strip()
        if not output:
            output.append(stripped)
            continue

        previous = output[-1]
        continues = (
            previous
            and stripped
            and not _ENDS_SENTENCE.search(previous)
            and not previous.startswith("#")
            and not stripped.startswith("#")
            and not _LIST_ITEM.match(stripped)
            and not _LIST_ITEM.match(previous)
            and stripped[0].islower()
        )
        if continues:
            output[-1] = f"{previous} {stripped}"
        else:
            output.append(stripped)

    return "\n".join(output)


def clean_ocr_text(text: str) -> str:
    """Nettoyage déterministe du texte extrait.

    Corrige ce qui est mécanique : césures, lignes coupées, habillage de page.
    Ne corrige pas les colonnes entrelacées, qui relèvent de l'analyse de mise
    en page — pas de règles.
    """

    if not text.strip():
        return text

    # Césures : "mathéma-\ntiques" -> "mathématiques"
    text = _HYPHENATED.sub(r"\1\2", text)

    # Découpage par page pour repérer l'habillage répété.
    markers = list(_PAGE_MARKER.finditer(text))
    if markers:
        pages = []
        for index, marker in enumerate(markers):
            start = marker.end()
            end = markers[index + 1].start() if index + 1 < len(markers) else len(text)
            pages.append(text[start:end])
        noise = _repeated_lines(pages)
        if noise:
            kept = []
            for line in text.split("\n"):
                if line.strip() in noise:
                    continue
                kept.append(line)
            text = "\n".join(kept)

    text = _join_broken_lines(text)
    return normalise(text)
