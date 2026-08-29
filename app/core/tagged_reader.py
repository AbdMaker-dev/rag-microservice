"""Lire un PDF à travers son arbre de structure.

Quand le document porte son propre plan — titres, paragraphes, tableaux,
ordre de lecture — aucune heuristique géométrique ne le vaut. On ne devine
plus les colonnes ni les lignes d'un tableau : le fichier les déclare.

Deux gains que la géométrie ne peut pas offrir. Les cellules d'un tableau sont
données comme telles, sans avoir à retrouver leurs frontières dans du blanc.
Et le décor — en-têtes, pieds de page, filigranes — est marqué « Artifact »
par le producteur : on le retire sans compter des répétitions.

Ce lecteur n'est appelé que si l'arbre a passé le contrôle de santé de
`struct_tree` : présent ne suffit pas, il doit couvrir le texte.
"""

from __future__ import annotations

import io
import logging
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Types qui ouvrent un bloc de texte à part entière.
_HEADINGS = {"H1": 1, "H2": 2, "H3": 3, "H4": 4, "H5": 5, "H6": 6, "H": 2}
_BLOCKS = {"P", "LI", "LBody", "Caption", "Figure", "TOCI", "Note"}
_TABLE = "Table"
_ROW = "TR"
_CELLS = {"TD", "TH"}


def _page_numbers(reader) -> Dict[int, int]:
    """Associer chaque objet page à son rang, pour relier l'arbre au texte."""

    numbers: Dict[int, int] = {}
    for index, page in enumerate(reader.pages, start=1):
        try:
            numbers[page.indirect_reference.idnum] = index
        except Exception:  # noqa: BLE001
            continue
    return numbers


def _text_by_mark(payload: bytes) -> Dict[Tuple[int, int], str]:
    """Le texte de chaque contenu marqué, repéré par (page, MCID).

    C'est le pont entre la structure et les caractères : l'arbre dit quel
    élément porte quel MCID, la page dit quel MCID porte quels caractères.
    """

    import pdfplumber

    pieces: Dict[Tuple[int, int], List[str]] = defaultdict(list)
    with pdfplumber.open(io.BytesIO(payload)) as document:
        for number, page in enumerate(document.pages, start=1):
            previous_bottom: Optional[float] = None
            for word in page.extract_words(extra_attrs=["mcid"]):
                mark = word.get("mcid")
                if mark is None:
                    continue  # décor, ou texte hors structure
                key = (number, int(mark))
                # Un même MCID peut couvrir plusieurs lignes : on rétablit
                # l'espace que le découpage en mots a mangé.
                pieces[key].append(word["text"])
                previous_bottom = word["bottom"]
    return {key: " ".join(parts) for key, parts in pieces.items()}


def _marks_of(node, page: Optional[int], numbers: Dict[int, int],
              seen: Set[int]) -> List[Tuple[int, int]]:
    """Tous les contenus marqués que porte un élément, dans l'ordre."""

    found: List[Tuple[int, int]] = []
    try:
        resolved = node.get_object()
    except Exception:  # noqa: BLE001
        return found
    if id(resolved) in seen:
        return found
    seen.add(id(resolved))

    if isinstance(resolved, list):
        for child in resolved:
            found.extend(_marks_of(child, page, numbers, seen))
        return found
    if not hasattr(resolved, "get"):
        return found

    holder = resolved.get("/Pg")
    if holder is not None:
        page = _number_of(holder, numbers)

    children = resolved.get("/K")
    if children is None:
        return found
    children = children.get_object()
    for child in children if isinstance(children, list) else [children]:
        target = child.get_object() if hasattr(child, "get_object") else child
        if isinstance(target, int):
            if page is not None:
                found.append((page, target))
        elif hasattr(target, "get") and str(target.get("/Type")) == "/MCR":
            own = target.get("/Pg")
            owner = _number_of(own, numbers) if own is not None else page
            if owner is not None:
                found.append((owner, int(target.get("/MCID", -1))))
        else:
            found.extend(_marks_of(target, page, numbers, seen))
    return found


def _number_of(reference, numbers: Dict[int, int]) -> Optional[int]:
    try:
        return numbers.get(reference.indirect_reference.idnum)
    except Exception:  # noqa: BLE001
        return None


def _gather(node, page: Optional[int], numbers: Dict[int, int],
            texts: Dict[Tuple[int, int], str], out: List[Tuple[int, str]],
            seen: Set[int]) -> None:
    """Parcourir l'arbre dans son ordre et rendre chaque bloc."""

    try:
        resolved = node.get_object()
    except Exception:  # noqa: BLE001
        return
    if id(resolved) in seen:
        return
    seen.add(id(resolved))

    if isinstance(resolved, list):
        for child in resolved:
            _gather(child, page, numbers, texts, out, seen)
        return
    if not hasattr(resolved, "get"):
        return

    holder = resolved.get("/Pg")
    if holder is not None:
        page = _number_of(holder, numbers) or page

    kind = str(resolved.get("/S", "")).lstrip("/")

    if kind == _TABLE:
        rendered, where = _render_table(resolved, page, numbers, texts)
        if rendered:
            out.append((where or page or 1, rendered))
        return

    if kind in _HEADINGS:
        marks = _marks_of(resolved, page, numbers, set())
        body = " ".join(
            texts[mark] for mark in marks if mark in texts and texts[mark].strip()
        ).strip()
        if body:
            out.append((marks[0][0] if marks else (page or 1),
                        f"{'#' * (_HEADINGS[kind] + 1)} {body}"))
        return

    children = resolved.get("/K")
    if children is None:
        return
    children = children.get_object()

    # Le texte accroché directement à cet élément, quel que soit son type. On
    # ne liste pas les types qui portent du contenu : un producteur peut
    # envelopper ses paragraphes dans n'importe quoi — « NonStruct » couvre à
    # lui seul la moitié de certains documents. Tout marqueur non émis ici
    # serait du texte perdu.
    direct: List[Tuple[int, int]] = []
    for child in children if isinstance(children, list) else [children]:
        target = child.get_object() if hasattr(child, "get_object") else child
        if isinstance(target, int):
            if page is not None:
                direct.append((page, target))
        elif hasattr(target, "get") and str(target.get("/Type")) == "/MCR":
            own = target.get("/Pg")
            owner = _number_of(own, numbers) if own is not None else page
            if owner is not None:
                direct.append((owner, int(target.get("/MCID", -1))))

    if direct:
        body = " ".join(
            texts[mark] for mark in direct if mark in texts and texts[mark].strip()
        ).strip()
        if body:
            out.append((direct[0][0], body))

    # Puis les éléments enfants, chacun émettant ce qui lui est propre : un
    # marqueur n'est rendu qu'une fois, au niveau où il est accroché.
    for child in children if isinstance(children, list) else [children]:
        target = child.get_object() if hasattr(child, "get_object") else child
        if hasattr(target, "get") and target.get("/S") is not None:
            _gather(target, page, numbers, texts, out, seen)


def _render_table(node, page, numbers, texts) -> Tuple[str, Optional[int]]:
    """Rendre un tableau déclaré, ligne par ligne, sans deviner ses frontières."""

    rows: List[str] = []
    where: Optional[int] = None

    def walk_rows(current, seen: Set[int]) -> None:
        nonlocal where
        try:
            resolved = current.get_object()
        except Exception:  # noqa: BLE001
            return
        if id(resolved) in seen:
            return
        seen.add(id(resolved))
        if isinstance(resolved, list):
            for child in resolved:
                walk_rows(child, seen)
            return
        if not hasattr(resolved, "get"):
            return
        if str(resolved.get("/S", "")).lstrip("/") == _ROW:
            cells = _render_row(resolved, page, numbers, texts)
            if cells:
                if where is None:
                    where = cells[1]
                rows.append("| " + " | ".join(cells[0]) + " |")
            return
        children = resolved.get("/K")
        if children is None:
            return
        children = children.get_object()
        for child in children if isinstance(children, list) else [children]:
            walk_rows(child, seen)

    walk_rows(node, set())
    return ("\n".join(rows), where) if rows else ("", None)


def _render_row(row, page, numbers, texts) -> Optional[Tuple[List[str], int]]:
    cells: List[str] = []
    first_page: Optional[int] = None
    children = row.get("/K")
    if children is None:
        return None
    children = children.get_object()
    for child in children if isinstance(children, list) else [children]:
        target = child.get_object() if hasattr(child, "get_object") else child
        if not hasattr(target, "get"):
            continue
        if str(target.get("/S", "")).lstrip("/") not in _CELLS:
            continue
        marks = _marks_of(target, page, numbers, set())
        if marks and first_page is None:
            first_page = marks[0][0]
        cells.append(
            " ".join(texts[mark] for mark in marks if mark in texts).strip()
        )
    return (cells, first_page or page or 1) if cells else None


def read(payload: bytes, texts: Optional[Dict[Tuple[int, int], str]] = None) -> List[str]:
    """Rendre le document page par page, en suivant son arbre de structure.

    `texts` évite de rouvrir le fichier : la passe de lecture relève déjà le
    texte de chaque contenu marqué, il n'y a aucune raison de le refaire.
    """

    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(payload))
    tree = reader.trailer["/Root"].get_object().get("/StructTreeRoot")
    if tree is None:
        return []

    numbers = _page_numbers(reader)
    texts = texts if texts is not None else _text_by_mark(payload)

    blocks: List[Tuple[int, str]] = []
    _gather(tree, None, numbers, texts, blocks, set())

    pages: Dict[int, List[str]] = defaultdict(list)
    for number, body in blocks:
        pages[number].append(body)

    total = len(reader.pages)
    return ["\n\n".join(pages.get(number, [])) for number in range(1, total + 1)]
