"""Le planning du ministère : un tableau de PDF devient des données.

Le Sénégal publie DEUX documents qu'il ne faut pas confondre. Le
*programme* dit QUOI enseigner ; le *planning* — « outil d'harmonisation
des progressions », publié par les Inspections d'Académie — dit QUAND :
mois par mois, semaine par semaine, chapitre par chapitre.

Aucun modèle de langue ici, et c'est délibéré. Un planning officiel se
recopie, il ne se paraphrase pas : une semaine inventée mais crédible
décalerait toute une année scolaire sans que personne ne s'en aperçoive.
Une ligne qu'on ne sait pas lire part donc en `warnings`, jamais dans une
supposition.

Le tableau change de forme d'un document à l'autre (le moyen a une colonne
« Parties » que le lycée n'a pas), d'où la lecture des rôles depuis la
ligne d'en-tête plutôt que par position de colonne.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import List, Optional

_MONTHS = (
    "janvier",
    "février",
    "mars",
    "avril",
    "mai",
    "juin",
    "juillet",
    "août",
    "septembre",
    "octobre",
    "novembre",
    "décembre",
)

# Les rôles de colonnes, reconnus par mot-clé : le moyen écrit « Leçons /
# Contenus » là où le lycée écrit « Thème/Chapitre », et « Compétences
# exigibles » là où l'autre écrit « Objectifs spécifiques ».
_ROLES = (
    ("weeks", ("periode", "periodes")),
    ("part", ("partie", "parties")),
    ("chapter", ("lecon", "lecons", "contenu", "contenus", "theme", "chapitre")),
    ("objectives", ("objectif", "objectifs", "competence", "competences")),
    ("assessments", ("acquis", "evaluer", "evaluation", "evaluations")),
)

_WEEK = re.compile(r"semaine\s*(\d{1,2})", re.IGNORECASE)
_CHAPTER_PREFIX = re.compile(r"^\s*chapitre\s*\d+\s*:?\s*", re.IGNORECASE)
_CONTINUATION = re.compile(r"\(\s*suite", re.IGNORECASE)
# « 1. Déterminer … 2. Utiliser … » — on coupe DEVANT le numéro, pas dessus,
# pour ne pas perdre le texte qui suit.
_NUMBERED = re.compile(r"(?<!\d)(?=\d{1,2}\.\s)")
_YEAR = re.compile(r"(\d{4})\s*[–—-]\s*(\d{4})")
_HOURS = re.compile(
    r"\(?\s*(\d{1,2})\s*(?:h|heures?)\s*/\s*semaine\s*\)?", re.IGNORECASE
)
_FIELD = re.compile(
    r"(discipline|niveau)\s*:\s*(.+?)(?=\s{2,}|\s*(?:discipline|niveau|annee|année)\s*:|$)",
    re.IGNORECASE | re.MULTILINE,
)


def _fold(value: str) -> str:
    """Sans accents ni casse — « Périodes » et « PERIODES » sont un seul mot."""

    # L'apostrophe typographique des PDF (U+2019) devient l'apostrophe
    # ordinaire : sans ça, « Mois d’octobre » n'est pas reconnu comme un mois
    # et toute la colonne des dates part en silence.
    stripped = unicodedata.normalize("NFKD", value.replace("\u2019", "'"))
    return "".join(c for c in stripped if not unicodedata.combining(c)).lower().strip()


@dataclass
class Period:
    month: Optional[str]
    weeks: List[int]


@dataclass
class PlanningEntry:
    position: int
    chapter: str
    details: str = ""
    part: str = ""
    objectives: List[str] = field(default_factory=list)
    assessments: List[str] = field(default_factory=list)
    periods: List[Period] = field(default_factory=list)


@dataclass
class Planning:
    school_year: Optional[str] = None
    subject: Optional[str] = None
    grade_label: Optional[str] = None
    weekly_hours: Optional[int] = None
    entries: List[PlanningEntry] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def _cells(line: str) -> List[str]:
    """Une ligne « | a | b | » devient ['a', 'b'] — bords compris."""

    parts = line.split("|")
    if parts and not parts[0].strip():
        parts = parts[1:]
    if parts and not parts[-1].strip():
        parts = parts[:-1]
    return [p.strip() for p in parts]


def _month_of(cells: List[str]) -> Optional[str]:
    """« Mois de novembre » ou « OCTOBRE » seul sur sa ligne."""

    filled = [c for c in cells if c]
    if len(filled) != 1:
        return None
    folded = _fold(filled[0])
    folded = re.sub(r"^mois\s+d[eu']?\s*", "", folded).strip()
    for month in _MONTHS:
        if folded == _fold(month):
            return month
    return None


def _weeks_of(cell: str) -> List[int]:
    """« Semaine 2 Semaine 3 Semaine 4 » → [2, 3, 4]."""

    return [int(m.group(1)) for m in _WEEK.finditer(cell)]


def _items(cell: str) -> List[str]:
    """Une liste numérotée dans une cellule devient des éléments distincts."""

    if not cell.strip():
        return []
    pieces = [p.strip(" .;") for p in _NUMBERED.split(cell)]
    items = [re.sub(r"^\d{1,2}\.\s*", "", p).strip() for p in pieces if p.strip()]
    return [i for i in items if i]


def _header_roles(cells: List[str]) -> Optional[dict]:
    """Associe chaque colonne à son rôle, d'après les mots de l'en-tête."""

    roles: dict = {}
    for index, cell in enumerate(cells):
        folded = _fold(cell)
        if not folded:
            continue
        for role, keywords in _ROLES:
            if role in roles.values():
                continue
            if any(word in folded for word in keywords):
                roles[index] = role
                break
    # Sans colonne de périodes, ce n'est pas l'en-tête d'un planning.
    return roles if "weeks" in roles.values() else None


_PLAN = re.compile(r"\s*\d{1,2}\)\s")


def _split_chapter(raw: str) -> tuple:
    """Sépare le titre du chapitre de ce qu'il contient.

    Deux formes selon le document : le lycée met le détail entre parenthèses
    (« Fonctions numériques (limites, dérivées…) »), le moyen enchaîne le
    plan de la leçon dans la même cellule (« LES NOMBRES DECIMAUX 1) Entiers
    naturels 2) … »). Dans les deux cas le titre seul sert de nom de cours,
    et le détail reste disponible pour le professeur.
    """

    # Le point final d'une phrase ne doit pas empêcher de reconnaître la
    # parenthèse fermante : « Fonctions numériques (limites…). »
    text = _CHAPTER_PREFIX.sub("", raw).strip().rstrip(" .;")
    opening = text.find("(")
    if opening > 0 and text.endswith(")"):
        return text[:opening].strip(" :–-"), text[opening + 1 : text.rfind(")")].strip()
    plan = _PLAN.search(text)
    if plan and plan.start() > 0:
        return text[: plan.start()].strip(" :–-"), text[plan.start() :].strip()
    return text.strip(" :–-"), ""


def _read_header_fields(text: str, planning: Planning) -> None:
    """L'année, la discipline, le niveau et l'horaire, lus dans l'en-tête."""

    year = _YEAR.search(text)
    if year:
        planning.school_year = f"{year.group(1)}-{year.group(2)}"
    hours = _HOURS.search(text)
    if hours:
        planning.weekly_hours = int(hours.group(1))
    for match in _FIELD.finditer(text):
        value = match.group(2).strip(" |")
        # Le niveau traîne souvent son horaire : « TS2 (5h/semaine) ».
        value = _HOURS.sub("", value).strip(" ()|")
        if not value:
            continue
        if _fold(match.group(1)) == "discipline" and planning.subject is None:
            planning.subject = value
        elif _fold(match.group(1)) == "niveau" and planning.grade_label is None:
            planning.grade_label = value


def parse_planning(text: str) -> Planning:
    """Le texte extrait d'un planning devient des chapitres ordonnés.

    L'ordre (`position`) est ce qui compte pour la plateforme ; les mois et
    les semaines sont un repère à afficher. Un chapitre qui déborde sur le
    mois suivant (« Chapitre 1 (suite et fin) ») ne crée pas une seconde
    entrée : il gagne une période.
    """

    planning = Planning()
    # L'en-tête tient sur les premières lignes, avant le tableau.
    _read_header_fields("\n".join(text.splitlines()[:40]), planning)

    roles: Optional[dict] = None
    month: Optional[str] = None
    current: Optional[PlanningEntry] = None

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = _cells(stripped)
        if not cells:
            continue

        found = _month_of(cells)
        if found:
            month = found
            continue

        if roles is None:
            roles = _header_roles(cells)
            if roles is not None:
                continue

        def value(role: str) -> str:
            if not roles:
                return ""
            for index, name in roles.items():
                if name == role and index < len(cells):
                    return cells[index]
            return ""

        # Sans en-tête reconnu, on lit par position : périodes, chapitre,
        # puis tout le reste en objectifs. C'est dégradé, on le dit.
        if roles is None:
            weeks_cell = cells[0]
            chapter_cell = cells[1] if len(cells) > 1 else ""
            objectives_cell = " ".join(cells[2:])
            assessments_cell = ""
            # On ne signale l'en-tête manquant que si une VRAIE ligne de
            # planning arrive sans lui : le titre et le premier mois passent
            # avant l'en-tête, ils ne sont pas des lignes dégradées.
            if _WEEK.search(weeks_cell) and "PLANNING_HEADER_NOT_FOUND" not in planning.warnings:
                planning.warnings.append("PLANNING_HEADER_NOT_FOUND")
        else:
            weeks_cell = value("weeks")
            chapter_cell = value("chapter")
            objectives_cell = value("objectives")
            assessments_cell = value("assessments")

        weeks = _weeks_of(weeks_cell)

        if weeks and chapter_cell:
            if _CONTINUATION.search(chapter_cell) and current is not None:
                # « Chapitre 1 (suite et fin) » : même chapitre, mois suivant.
                current.periods.append(Period(month=month, weeks=weeks))
                current.objectives.extend(_items(objectives_cell))
                continue
            title, details = _split_chapter(chapter_cell)
            current = PlanningEntry(
                position=len(planning.entries) + 1,
                chapter=title,
                details=details,
                part=value("part") if roles else "",
                objectives=_items(objectives_cell),
                assessments=_items(assessments_cell),
                periods=[Period(month=month, weeks=weeks)],
            )
            planning.entries.append(current)
            continue

        # Une ligne sans semaine ni chapitre prolonge la précédente : le
        # tableau a été coupé par un saut de page, pas par un nouveau sujet.
        if current is not None and (objectives_cell or assessments_cell):
            current.objectives.extend(_items(objectives_cell))
            current.assessments.extend(_items(assessments_cell))
            continue

        # Une semaine vide n'est pas une anomalie : le calendrier officiel en
        # contient (rentrée, semaine sans leçon). On ne la signale que si elle
        # portait quelque chose qu'on n'a pas su rattacher.
        if weeks and not chapter_cell and (objectives_cell or assessments_cell):
            planning.warnings.append("PLANNING_ROW_WITHOUT_CHAPTER")

    _finish(planning)
    return planning


def _finish(planning: Planning) -> None:
    """Dédoublonne, et dit ce qui manque plutôt que de le deviner."""

    for entry in planning.entries:
        entry.objectives = _unique(entry.objectives)
        entry.assessments = _unique(entry.assessments)

    if not planning.entries:
        planning.warnings.append("PLANNING_NOT_RECOGNISED")
    if planning.school_year is None:
        planning.warnings.append("PLANNING_YEAR_MISSING")
    if planning.subject is None:
        planning.warnings.append("PLANNING_SUBJECT_MISSING")
    if planning.grade_label is None:
        planning.warnings.append("PLANNING_GRADE_MISSING")
    if any(p.month is None for e in planning.entries for p in e.periods):
        planning.warnings.append("PLANNING_MONTH_MISSING")


def _unique(values: List[str]) -> List[str]:
    """Les PDF répètent la même colonne d'une page à l'autre — une seule fois."""

    seen = set()
    kept = []
    for value in values:
        key = _fold(value)
        if key and key not in seen:
            seen.add(key)
            kept.append(value)
    return kept
