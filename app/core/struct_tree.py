"""Santé de l'arbre de structure d'un PDF.

Un `/ToUnicode` est une table : elle est juste ou fausse, et on peut la croire
sur parole. Un `/StructTreeRoot` non — il peut être **présent et vide de
sens** : arbre plat, tout en `<P>`, aucun `<Table>`, produit
automatiquement à l'export sans porter la moindre information.

Un arbre qui ne couvre qu'une partie du contenu est pire que pas d'arbre du
tout, parce qu'on lui fait confiance. On le mesure donc avant de s'en servir,
et la mesure décisive est la part du contenu marqué qui lui est réellement
rattachée — pas la simple présence de la racine.
"""

from __future__ import annotations

import io
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

_MCID = re.compile(rb"/MCID\s+(\d+)")

# Types qui portent du sens. Un arbre qui n'a que des « NonStruct » et des
# « Span » décrit la mise en page, pas le document.
_SEMANTIC = {"Table", "TR", "TD", "TH", "THead", "TBody",
             "H1", "H2", "H3", "H4", "H5", "H6", "H",
             "L", "LI", "LBody", "P", "Figure", "Caption"}
_TABLE_TYPES = {"Table", "TR", "TD", "TH"}

# En dessous, une partie du texte échappe à l'arbre : s'y fier reviendrait à
# perdre silencieusement ce qu'il ne couvre pas.
_MINIMUM_COVERAGE = 0.90
# Un arbre d'une seule strate ne dit rien de plus qu'une liste de paragraphes.
_MINIMUM_DEPTH = 3


@dataclass(frozen=True)
class TreeHealth:
    """Ce que vaut l'arbre de structure, avant de s'y fier."""

    present: bool = False
    coverage: float = 0.0
    depth: int = 0
    elements: int = 0
    types: Dict[str, int] = field(default_factory=dict)
    weak_pages: List[int] = field(default_factory=list)

    @property
    def has_tables(self) -> bool:
        return any(self.types.get(name) for name in _TABLE_TYPES)

    @property
    def semantic(self) -> bool:
        return any(self.types.get(name) for name in _SEMANTIC)

    @property
    def usable(self) -> bool:
        """L'arbre mérite-t-il qu'on lise le document à travers lui ?"""

        return (
            self.present
            and self.coverage >= _MINIMUM_COVERAGE
            and self.depth >= _MINIMUM_DEPTH
            and self.semantic
        )


def _walk(node, page: Optional[int], state: dict, seen: Set[int], depth: int = 0) -> None:
    """Parcourir l'arbre en retenant à quelle page chaque feuille appartient."""

    try:
        resolved = node.get_object()
    except Exception:  # noqa: BLE001
        return
    if id(resolved) in seen:
        return
    seen.add(id(resolved))
    state["depth"] = max(state["depth"], depth)

    if isinstance(resolved, list):
        for child in resolved:
            _walk(child, page, state, seen, depth + 1)
        return
    if not hasattr(resolved, "get"):
        return

    kind = resolved.get("/S")
    if kind is not None:
        state["types"][str(kind).lstrip("/")] += 1

    # Un élément peut désigner sa page ; ses descendants en héritent.
    holder = resolved.get("/Pg")
    if holder is not None:
        page = _page_id(holder)

    children = resolved.get("/K")
    if children is None:
        return
    children = children.get_object()
    for child in children if isinstance(children, list) else [children]:
        target = child.get_object() if hasattr(child, "get_object") else child
        if isinstance(target, int):
            # Feuille : le contenu marqué numéro N, sur la page courante.
            state["leaves"].add((page, target))
        elif hasattr(target, "get") and str(target.get("/Type")) == "/MCR":
            own = target.get("/Pg")
            state["leaves"].add(
                (_page_id(own) if own is not None else page, int(target.get("/MCID", -1)))
            )
        else:
            _walk(target, page, state, seen, depth + 1)


def _page_id(reference) -> Optional[int]:
    try:
        return reference.indirect_reference.idnum
    except Exception:  # noqa: BLE001
        try:
            return id(reference.get_object())
        except Exception:  # noqa: BLE001
            return None


def inspect(payload: bytes) -> TreeHealth:
    """Mesurer l'arbre de structure d'un PDF, sans encore s'en servir."""

    from pypdf import PdfReader

    try:
        reader = PdfReader(io.BytesIO(payload))
        root = reader.trailer["/Root"].get_object()
        tree = root.get("/StructTreeRoot")
    except Exception:  # noqa: BLE001
        return TreeHealth()
    if tree is None:
        return TreeHealth()

    state = {"types": Counter(), "leaves": set(), "depth": 0}
    _walk(tree, None, state, set())

    # Les MCID repartent de zéro à chaque page : la couverture ne se mesure
    # qu'en comparant page par page, jamais sur un ensemble global.
    total = attached = 0
    weak: List[int] = []
    for number, page in enumerate(reader.pages, start=1):
        identifier = _page_id(page)
        try:
            data = page.get_contents().get_data()
        except Exception:  # noqa: BLE001
            continue
        marks = {int(found) for found in _MCID.findall(data)}
        if not marks:
            continue
        covered = sum(1 for mark in marks if (identifier, mark) in state["leaves"])
        total += len(marks)
        attached += covered
        if covered / len(marks) < _MINIMUM_COVERAGE:
            weak.append(number)

    return TreeHealth(
        present=True,
        coverage=round(attached / total, 4) if total else 0.0,
        depth=state["depth"],
        elements=sum(state["types"].values()),
        types=dict(state["types"]),
        weak_pages=weak,
    )
