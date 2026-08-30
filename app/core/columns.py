"""Lire une page en colonnes, sans se fier aux filets.

Certains PDF dessinent leurs tableaux sans les tracer : les colonnes ne sont
séparées que par du blanc. `find_tables()` y échoue, et pire, il y voit des
tableaux là où il n'y a que des barres de fraction — sur les 99 pages du
programme national de mathématiques, aucun filet n'atteint la moitié de la
largeur de page.

On lit donc la disposition elle-même : les gouttières de blanc qui traversent
la page, puis les bandes horizontales qui coupent toutes les colonnes en même
temps.

**Tous les seuils sont des rapports**, calculés sur le document. Une constante
en points décrit une seule mise en page ; un rapport à l'interligne ou à la
largeur d'espace suit le document qu'on lui donne.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from statistics import median
from typing import Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# Une suite de chiffres, réduite à un marqueur : « page 9 » et « page 12 » sont
# le même habillage.
_DIGIT_RUN = re.compile(r"\d+")

# --- Rapports, et ce qu'ils valent sur le programme national ----------------
# Mesuré sur ce document : hauteur de ligne 10,8 pt, largeur d'espace 2,7 pt,
# interligne 12,5 pt. Les rapports ci-dessous y donnent 2,7 / 4,1 / 7,5 pt.
_LINE_MERGE = 0.25    # × hauteur de ligne : deux mots sur la même ligne visuelle
_GUTTER_MIN = 1.5     # × largeur d'espace : en deçà, c'est un espace, pas une colonne
_GUTTER_MERGE = 4.5   # × largeur d'espace : deux gouttières voisines n'en font qu'une
_ROW_BREAK = 0.6      # × interligne : un vide qui traverse tout coupe une ligne

# Une gouttière tolère quelques débordements — une formule qui dépasse ne doit
# pas faire disparaître la colonne pour toute la page.
_GUTTER_TOLERANCE = 0.10
# Et elle doit avoir du texte des deux côtés, sinon c'est une marge.
_GUTTER_USED = 0.12
# Une colonne confirmée sur assez de pages devient le gabarit du document.
_TEMPLATE_SHARE = 0.25


@dataclass(frozen=True)
class Metrics:
    """Les grandeurs du document, dont tous les seuils se déduisent."""

    line_height: float = 10.0
    space_width: float = 2.5
    line_gap: float = 12.0

    @property
    def merge(self) -> float:
        return self.line_height * _LINE_MERGE

    @property
    def gutter_min(self) -> float:
        return self.space_width * _GUTTER_MIN

    @property
    def gutter_merge(self) -> float:
        return self.space_width * _GUTTER_MERGE

    @property
    def row_break(self) -> float:
        return self.line_gap * _ROW_BREAK


def measure(pages_words: Sequence[Sequence[dict]]) -> Metrics:
    """Établir les grandeurs du document à partir de ses mots."""

    heights: List[float] = []
    spaces: List[float] = []
    gaps: List[float] = []

    for words in pages_words:
        if not words:
            continue
        for word in words:
            heights.append(word["bottom"] - word["top"])

        rows: Dict[int, List[dict]] = {}
        for word in words:
            rows.setdefault(round(word["top"] / 3), []).append(word)
        tops: List[float] = []
        for group in rows.values():
            group.sort(key=lambda item: item["x0"])
            for left, right in zip(group, group[1:]):
                gap = right["x0"] - left["x1"]
                if 0 < gap < 40:
                    spaces.append(gap)
            tops.append(min(item["top"] for item in group))
        tops.sort()
        for above, below in zip(tops, tops[1:]):
            if 0 < below - above < 60:
                gaps.append(below - above)

    return Metrics(
        line_height=median(heights) if heights else 10.0,
        space_width=median(spaces) if spaces else 2.5,
        line_gap=median(gaps) if gaps else 12.0,
    )


def visual_lines(words: Sequence[dict], metrics: Metrics) -> List[List[dict]]:
    """Regrouper les mots en lignes visuelles."""

    if not words:
        return []
    ordered = sorted(words, key=lambda item: (item["top"], item["x0"]))
    lines: List[List[dict]] = [[ordered[0]]]
    for word in ordered[1:]:
        if abs(word["top"] - lines[-1][0]["top"]) <= metrics.merge:
            lines[-1].append(word)
        else:
            lines.append([word])
    for line in lines:
        line.sort(key=lambda item: item["x0"])
    return lines


def gutters(lines: Sequence[Sequence[dict]], width: float, metrics: Metrics) -> List[Tuple[float, float]]:
    """Trouver les bandes de blanc qui traversent la page.

    On ne compte pas les lignes pleine largeur : un paragraphe de prose
    couvrirait toutes les abscisses et effacerait les gouttières.
    """

    if not lines:
        return []

    narrow = [line for line in lines if _span(line) < width * 0.85]
    if len(narrow) < 3:
        return []

    columns = int(width) + 1
    cover = [0] * columns
    for line in narrow:
        for word in line:
            for x in range(max(0, int(word["x0"])), min(columns, int(word["x1"]) + 1)):
                cover[x] += 1

    ceiling = max(1, int(len(narrow) * _GUTTER_TOLERANCE))
    bands: List[Tuple[float, float]] = []
    start: Optional[int] = None
    for x in range(columns):
        if cover[x] <= ceiling:
            start = x if start is None else start
        elif start is not None:
            bands.append((float(start), float(x)))
            start = None
    if start is not None:
        bands.append((float(start), float(columns)))

    # Les bandes trop étroites sont des espaces entre mots ; les bandes
    # voisines décrivent la même séparation vue à travers plusieurs lignes.
    wide = [band for band in bands if band[1] - band[0] >= metrics.gutter_min]
    merged: List[Tuple[float, float]] = []
    for band in wide:
        if merged and band[0] - merged[-1][1] <= metrics.gutter_merge:
            merged[-1] = (merged[-1][0], band[1])
        else:
            merged.append(band)

    # Une gouttière sans texte des deux côtés est une marge, pas une colonne.
    kept: List[Tuple[float, float]] = []
    for left, right in merged:
        both = sum(
            1
            for line in narrow
            if any(word["x1"] <= left for word in line)
            and any(word["x0"] >= right for word in line)
        )
        if both >= len(narrow) * _GUTTER_USED:
            kept.append((left, right))
    return kept


def _span(line: Sequence[dict]) -> float:
    return max(word["x1"] for word in line) - min(word["x0"] for word in line)


def template(page_gutters: Sequence[Sequence[Tuple[float, float]]], pages: int,
             metrics: Metrics) -> List[float]:
    """Le nombre de colonnes usuel du document, et leurs positions médianes.

    Les colonnes d'un même document ne tombent pas forcément aux mêmes
    abscisses : dans le programme national, chaque tableau a ses propres
    largeurs — 209 et 351 sur une page, 258 et 383 sur la suivante. Chercher
    des positions communes n'y donne rien.

    Le gabarit ne sert qu'à secourir une page qui a trouvé **moins** de
    colonnes que le document n'en a d'ordinaire — une page encombrée de
    formules, par exemple. Une page qui n'en trouve aucune n'est pas une page
    de tableau mal lue : c'est de la prose, et lui imposer des colonnes
    déchiquette ses phrases.
    """

    par_rang: Dict[int, List[float]] = {}
    comptes: Dict[int, int] = {}
    for gutters_ in page_gutters:
        if not gutters_:
            continue
        comptes[len(gutters_)] = comptes.get(len(gutters_), 0) + 1
        for rank, (left, right) in enumerate(gutters_):
            par_rang.setdefault(rank, []).append((left + right) / 2)

    if not comptes:
        return []
    usuel = max(comptes, key=lambda k: comptes[k])
    if comptes[usuel] < max(2, int(pages * _TEMPLATE_SHARE)):
        return []
    return [median(par_rang[rank]) for rank in range(usuel) if rank in par_rang]


def furniture(pages_lines: Sequence[Sequence[Sequence[dict]]]) -> set:
    """Les lignes d'habillage : en-têtes et pieds de page.

    Elles doivent être écartées **avant** le découpage en colonnes, sinon un
    pied de page se retrouve haché en cellules et devient une ligne de tableau
    de plus — impossible à reconnaître ensuite.

    Le discriminant n'est pas la position : dans le programme national,
    l'en-tête du tableau est à 5 % de la hauteur et le pied de page à 75 %.
    C'est le **numéro de page** qui trahit l'habillage — une ligne qui revient
    à l'identique sauf un chiffre est un titre courant, tandis que la ligne
    « Contenus Commentaires Compétences exigibles », qui revient elle aussi
    mais sans aucun chiffre, est un vrai en-tête de tableau.

    Limite connue : un pied de page sans numéro échappe à ce critère. Il sera
    alors rendu en prose et retiré plus tard par le nettoyage, qui repère les
    lignes répétées — à condition qu'il n'ait pas été haché en cellules avant.
    """

    from collections import Counter

    seen: Counter = Counter()
    pages = len(pages_lines)
    for lines in pages_lines:
        edges = list(lines[:2]) + list(lines[-3:])
        for line in edges:
            text = " ".join(word["text"] for word in line).strip()
            # Les chiffres varient d'une page à l'autre : « page 12 » et
            # « page 13 » sont le même habillage.
            normalised = _DIGIT_RUN.sub("#", text)
            # Sans chiffre, ce n'est pas un titre courant : c'est du contenu
            # qui se répète, typiquement l'en-tête d'un tableau reconduit de
            # page en page.
            if 8 <= len(normalised) <= 120 and any(c.isdigit() for c in text):
                seen[normalised] += 1

    # Seuil bas : un pied de page change souvent avec la partie du document,
    # et chaque variante ne couvre alors qu'une fraction des pages.
    minimum = max(3, int(pages * 0.04))
    return {text for text, count in seen.items() if count >= minimum}


def _is_furniture(line: Sequence[dict], known: set) -> bool:
    text = " ".join(word["text"] for word in line).strip()
    return _DIGIT_RUN.sub("#", text) in known


def _straddlers(line: Sequence[dict], boundaries: Sequence[float], slack: float) -> int:
    """Mots à cheval sur une frontière, débordant nettement des deux côtés."""

    count = 0
    for word in line:
        for boundary in boundaries:
            if word["x0"] < boundary - slack and word["x1"] > boundary + slack:
                count += 1
                break
    return count


def confirms(lines: Sequence[Sequence[dict]], boundaries: Sequence[float],
             width: float, metrics: Metrics) -> bool:
    """La page confirme-t-elle des frontières qu'on lui propose ?

    Le gabarit du document ne s'impose jamais : il se propose. Une page de
    prose justifiée le rejette — ses lignes chevauchent les frontières de part
    en part ; une page de tableau qui a raté sa détection locale le confirme.
    """

    if not boundaries or not lines:
        return False
    block = max((_span(line) for line in lines), default=width) or width
    slack = metrics.space_width * 2
    clean = 0
    narrow = 0
    for line in lines:
        if _span(line) >= block * 0.85:
            continue
        narrow += 1
        cells = {_column_of(word, [0.0, *boundaries, width]) for word in line}
        if len(cells) >= 2 and _straddlers(line, boundaries, slack) == 0:
            clean += 1
    return narrow > 0 and clean >= max(3, int(narrow * 0.25))


def clustered_boundaries(lines: Sequence[Sequence[dict]], width: float,
                         metrics: Metrics) -> List[float]:
    """Frontières de colonnes par le vote des débuts de colonnes.

    Voie de secours derrière `gutters` : l'histogramme échoue quand la page
    mêle prose et tableau, ou quand des cellules fusionnées traversent la
    gouttière. Ici chaque grand blanc interne vote — par son **bord droit**,
    le début de la colonne suivante. Le centre trompe : quand une cellule est
    vide, le blanc court de la colonne 0 à la colonne 2 et son centre tombe au
    milieu de la colonne 1. Le début de colonne, aligné à gauche, reste stable
    quelles que soient les cellules remplies.
    """

    narrow = [line for line in lines if _span(line) < width * 0.85]
    if len(narrow) < 3:
        return []

    threshold = metrics.gutter_min * 2
    starts = sorted(
        right["x0"]
        for line in narrow
        for left, right in zip(line, line[1:])
        if right["x0"] - left["x1"] >= threshold
    )
    if not starts:
        return []
    clusters: List[List[float]] = [[starts[0]]]
    for x in starts[1:]:
        if x - clusters[-1][-1] <= metrics.gutter_merge * 2:
            clusters[-1].append(x)
        else:
            clusters.append([x])
    candidates = [
        median(cluster) - metrics.gutter_min / 2
        for cluster in clusters
        if len(cluster) >= 2
    ]
    if not candidates:
        return []

    # Les lignes qui chevauchent un candidat sont prose ou cellule fusionnée :
    # elles seront rendues en prose, elles ne votent pas contre.
    slack = metrics.space_width * 2
    plain = [
        line
        for line in narrow
        if not any(
            word["x0"] < x - slack and word["x1"] > x + slack
            for word in line
            for x in candidates
        )
    ]
    if len(plain) < 3:
        return []

    usage: Dict[float, int] = {}
    for x in candidates:
        used = sum(
            1
            for line in plain
            if any(word["x1"] <= x for word in line)
            and any(word["x0"] >= x for word in line)
        )
        if used >= max(3, int(len(plain) * 0.12)):
            usage[x] = used

    # Deux frontières plus proches que la largeur minimale d'une colonne
    # décrivent le même bord ou un alignement interne : la moins utilisée
    # disparaît.
    kept = sorted(usage)
    minimum_column = width * 0.15
    changed = True
    while changed and len(kept) > 1:
        changed = False
        for index in range(len(kept) - 1):
            if kept[index + 1] - kept[index] < minimum_column:
                loser = (
                    kept[index]
                    if usage[kept[index]] <= usage[kept[index + 1]]
                    else kept[index + 1]
                )
                kept.remove(loser)
                changed = True
                break
    return kept


def _is_row(line: Sequence[dict], edges: Sequence[float], metrics: Metrics) -> bool:
    """Une ligne strictement cellulaire : ses mots tombent dans les colonnes.

    Deux exigences. Ses mots occupent au moins deux cellules, et chaque
    frontière franchie l'est par un vrai blanc — le dernier mot avant et le
    premier après laissent la largeur d'une gouttière. « la résolution de
    problèmes » n'y répond pas : son texte est continu à travers la frontière.
    """

    cells = sorted({_column_of(word, edges) for word in line})
    if len(cells) < 2:
        return False
    boundaries = list(edges[1:-1])
    for word in line:
        for boundary in boundaries:
            if word["x0"] < boundary and word["x1"] > boundary:
                return False
    for left_cell, right_cell in zip(cells, cells[1:]):
        boundary = edges[left_cell + 1]
        before = max(
            (word["x1"] for word in line if word["x1"] <= boundary), default=None
        )
        after = min(
            (word["x0"] for word in line if word["x0"] >= boundary), default=None
        )
        if before is None or after is None:
            continue
        if after - before < metrics.gutter_min:
            return False
    return True


def render(lines: Sequence[Sequence[dict]], boundaries: Sequence[float],
           width: float, metrics: Metrics, known_furniture: Optional[set] = None) -> str:
    """Rendre une page : prose et tableau chacun à sa place.

    La zone tabulaire va de la première à la dernière ligne strictement
    cellulaire — c'est le zonage par bandes. En dehors, tout est prose, même
    étroit : sans cette règle, les puces d'une page mixte ressortaient hachées
    en fausses cellules — « | la résolution | de problèmes | plane | ». À
    l'intérieur, une ligne non cellulaire — une formule qui déborde, une
    cellule fusionnée — rejoint la ligne de tableau en cours au lieu d'en
    ouvrir une fausse.
    """

    if not lines:
        return ""
    block = max((_span(line) for line in lines), default=width) or width
    if not boundaries:
        return "\n".join(" ".join(word["text"] for word in line) for line in lines)

    edges = [0.0] + list(boundaries) + [width]
    known_furniture = known_furniture or set()

    kept = [line for line in lines if not _is_furniture(line, known_furniture)]
    row_flags = [_is_row(line, edges, metrics) for line in kept]
    first = row_flags.index(True) if True in row_flags else None
    last = len(row_flags) - 1 - row_flags[::-1].index(True) if True in row_flags else None

    out: List[str] = []
    buffer: List[List[List[str]]] = []

    def flush() -> None:
        for row in buffer:
            if any(cell for cell in row):
                out.append("| " + " | ".join(" ".join(cell) for cell in row) + " |")
        buffer.clear()

    previous_bottom: Optional[float] = None
    for index, line in enumerate(kept):
        in_zone = first is not None and first <= index <= last
        top = min(word["top"] for word in line)
        wide = _span(line) >= block * 0.85

        if not in_zone or (wide and not row_flags[index]):
            flush()
            out.append(" ".join(word["text"] for word in line))
            previous_bottom = max(word["bottom"] for word in line)
            continue

        if row_flags[index]:
            if previous_bottom is not None and top - previous_bottom > metrics.row_break:
                flush()
            if not buffer:
                buffer.append([[] for _ in range(len(edges) - 1)])
        elif not buffer:
            # Ligne non cellulaire en tête de zone : prose à sa place.
            out.append(" ".join(word["text"] for word in line))
            previous_bottom = max(word["bottom"] for word in line)
            continue

        for word in line:
            buffer[-1][_column_of(word, edges)].append(word["text"])
        previous_bottom = max(word["bottom"] for word in line)

    flush()
    return "\n".join(out)


def _column_of(word: dict, edges: Sequence[float]) -> int:
    """La colonne d'un mot, par recouvrement — jamais par son centre.

    Un mot court posé au bord d'une gouttière basculerait sinon dans la
    colonne voisine.
    """

    best, score = 0, -1.0
    for index in range(len(edges) - 1):
        overlap = min(word["x1"], edges[index + 1]) - max(word["x0"], edges[index])
        if overlap > score:
            best, score = index, overlap
    return best
