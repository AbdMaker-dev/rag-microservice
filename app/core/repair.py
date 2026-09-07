"""Réparer une transcription de cahier — par PREUVE, jamais par invention.

Awa photographie son cahier ; l'OCR lit son écriture et se trompe parfois.
La tentation serait de demander au modèle de « corriger » : il produirait
une formule plausible et fausse, et Awa réviserait dessus sans le savoir.
Sur un cours de terminale, c'est le pire résultat possible — pire que de
laisser le passage abîmé, qu'elle aurait au moins repéré.

Alors on ne corrige que ce qu'on peut PROUVER. Un passage mal lu est
remplacé uniquement si on retrouve, dans la base des professeurs et dans
SON périmètre, un passage validé qui dit la même chose. La correction est
alors une citation, pas une reformulation : le texte substitué est
littéralement celui du cours validé, avec sa source.

Trois issues, et aucune n'est silencieuse :

  - `corrige`      preuve trouvée — le passage validé remplace, source à l'appui ;
  - `a-verifier`   le passage semble abîmé mais aucune preuve ne le couvre ;
  - `inchange`     rien à signaler.

Aucun modèle de langue n'intervient ici, et un test le garde.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

# En dessous de cette distance cosinus, deux passages disent la même chose.
# Au-dessus, on ne prouve rien — et on préfère ne rien faire.
_PROOF_DISTANCE = 0.18

# Un passage identique au caractère près n'a rien à corriger.
_IDENTICAL = 0.02

_WORD = re.compile(r"[^\W\d_]{2,}", re.UNICODE)
# Ce qu'on s'attend à lire dans un cahier, toutes matières : des lettres,
# des chiffres, de la ponctuation, et les symboles des sciences.
_EXPECTED = re.compile(
    r"[\w\s.,;:!?'\"«»()\[\]{}\-–—+*/=<>%°$€£§&@#^~|\\"
    r"√∑∏∫∞≈≠≤≥±×÷→←↔⇒⇔∈∉⊂∪∩∅αβγδεθλμπρσφψωΩΔΣΦΨ]",
    re.UNICODE,
)
_PAIRS = (("(", ")"), ("[", "]"), ("{", "}"))


@dataclass(frozen=True)
class Proof:
    """Le passage validé qui sert de preuve, et d'où il vient."""

    text: str
    title: str
    locator: str
    distance: float


@dataclass
class Segment:
    """Un morceau de la transcription, et ce qu'on en a fait."""

    ordinal: int
    original: str
    status: str = "inchange"  # inchange | corrige | a-verifier
    text: str = ""
    proof: Optional[Proof] = None
    reason: str = ""

    def __post_init__(self) -> None:
        if not self.text:
            self.text = self.original


@dataclass
class RepairResult:
    segments: List[Segment] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def corrected(self) -> int:
        return sum(1 for s in self.segments if s.status == "corrige")

    @property
    def to_check(self) -> int:
        return sum(1 for s in self.segments if s.status == "a-verifier")

    def text(self) -> str:
        """La transcription telle qu'Awa la relira."""

        return "\n".join(s.text for s in self.segments if s.text.strip())


def split_segments(text: str) -> List[str]:
    """Découper la transcription en morceaux réparables.

    On coupe aux lignes, pas aux phrases : un cahier est écrit ligne à
    ligne, une formule occupe sa ligne, et un titre aussi. Les lignes vides
    séparent les blocs et disparaissent.
    """

    return [line.strip() for line in text.splitlines() if line.strip()]


def looks_broken(segment: str) -> str:
    """Dit POURQUOI un passage semble mal lu — ou rien s'il va bien.

    Des signes mécaniques, pas un jugement de sens : on ne cherche pas à
    savoir si le contenu est juste, seulement si la lecture a déraillé.
    """

    stripped = segment.strip()
    if not stripped:
        return ""

    for opening, closing in _PAIRS:
        if stripped.count(opening) != stripped.count(closing):
            return "PARENTHESE_NON_FERMEE"

    # On cherche du BRUIT, pas des mathématiques. « f(x) = 2x + 1 » n'a que
    # trois lettres sur treize caractères : compter les lettres signalerait
    # toute formule, dans toutes les matières où il y en a. On compte donc
    # les caractères qui n'ont leur place NI dans une phrase NI dans une
    # formule — c'est ça, une lecture qui a déraillé.
    noise = sum(1 for c in stripped if not _EXPECTED.match(c))
    if len(stripped) >= 10 and noise / len(stripped) > 0.2:
        return "CARACTERES_INATTENDUS"

    # Le caractère de remplacement Unicode : l'OCR a rendu un signe qu'il
    # n'a pas su nommer. Aucune ambiguïté, celui-là.
    if "�" in stripped:
        return "CARACTERE_ILLISIBLE"

    words = _WORD.findall(stripped)
    if words and sum(len(w) for w in words) / len(words) > 18:
        return "MOTS_AGGLUTINES"

    return ""


def _fold(value: str) -> str:
    stripped = unicodedata.normalize("NFKD", value.replace("’", "'"))
    return "".join(
        c for c in stripped if not unicodedata.combining(c)
    ).lower().strip()


def _same_text(left: str, right: str) -> bool:
    """Identiques une fois la casse, les accents et les espaces mis de côté."""

    def normalise(value: str) -> str:
        # La ponctuation de bord seulement : celle du milieu peut porter du
        # sens, on ne l'efface pas pour faire coïncider deux textes.
        return re.sub(r"\s+", " ", _fold(value)).strip(" .;:,!?")

    return normalise(left) == normalise(right)


def repair_by_proof(
    text: str,
    find_proof: Callable[[str], Optional[Proof]],
) -> RepairResult:
    """Réparer une transcription depuis la base validée du même périmètre.

    `find_proof` reçoit un passage et rend le passage validé le plus proche,
    ou `None`. Le seuil de preuve est appliqué ICI, pas chez l'appelant :
    c'est la règle métier, elle ne doit pas dépendre de qui appelle.
    """

    result = RepairResult()
    for ordinal, original in enumerate(split_segments(text), start=1):
        segment = Segment(ordinal=ordinal, original=original)
        broken = looks_broken(original)

        proof = find_proof(original)
        if proof is not None and proof.distance <= _PROOF_DISTANCE:
            if proof.distance <= _IDENTICAL or _same_text(original, proof.text):
                # Le cours validé dit déjà exactement ça : rien à corriger,
                # et surtout pas de « correction » cosmétique qui ferait
                # douter Awa de sa propre copie.
                segment.status = "inchange"
            else:
                segment.status = "corrige"
                segment.text = proof.text
                segment.proof = proof
        elif broken:
            segment.status = "a-verifier"
            segment.reason = broken

        result.segments.append(segment)

    if result.to_check:
        result.warnings.append("NOTEBOOK_SEGMENTS_TO_CHECK")
    if result.corrected:
        result.warnings.append("NOTEBOOK_SEGMENTS_CORRECTED")
    if not result.segments:
        result.warnings.append("NOTEBOOK_EMPTY_TRANSCRIPTION")
    return result


def proof_from_hits(hits: Sequence[dict]) -> Optional[Proof]:
    """Le meilleur passage validé d'une recherche, en preuve utilisable.

    Rendre `None` plutôt qu'une preuve faible est délibéré : sans couverture
    validée, on signale, on n'invente pas.
    """

    if not hits:
        return None
    best = min(hits, key=lambda hit: hit.get("distance", 1.0))
    content = (best.get("content") or "").strip()
    if not content:
        return None
    return Proof(
        text=content,
        title=best.get("title") or "",
        locator=best.get("locator") or "",
        distance=float(best.get("distance", 1.0)),
    )
