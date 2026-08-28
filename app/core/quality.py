"""Mesure de la qualité d'un texte extrait.

Le principe : on ne cherche pas les caractères cassés — une liste noire
vieillit et rate la prochaine façon de casser. On vérifie au contraire que le
texte ressemble à de la langue écrite, ce qui reste vrai quelle que soit
l'origine du problème.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import List

# Marqueur d'échec standard de l'extracteur PDF : « je ne sais pas quel
# glyphe c'est ». Universel, pas une variante d'encodage.
_CID_MARKER = re.compile(r"\(cid:\d+\)")

_TOKEN = re.compile(r"[^\W\d_]+", re.UNICODE)

# Alphabet réellement utilisé en français. C'est une liste blanche de la
# langue, pas une liste noire des corruptions : elle ne vieillit pas.
# Pour une autre langue, on remplace ce jeu, rien d'autre.
# Répertoire attendu hors lettres : ponctuation et symboles mathématiques
# usuels d'un document pédagogique. Là encore, une liste blanche.
_ALLOWED_PUNCTUATION = set(
    " \t\n\r.,;:!?…'\"«»‘’“”()[]{}<>/\\|-–—+±×÷=≠≤≥%‰°#&*@$€£§©®™_^~`•·◦"
)
_ALLOWED_MATH = set("πΠΣσΔδθλμαβγΩω∈∉⊂⊄∩∪∅∀∃∞√∫≈≡→←↔⇒⇔⊥∥∠")

_FRENCH_ALPHABET = set("abcdefghijklmnopqrstuvwxyzàâäçéèêëîïñôöùûüÿœæ")
_VOWELS = set("aeiouyàâäéèêëîïôöùûÿœæ")


@dataclass(frozen=True)
class Quality:
    score: float
    cid_markers: int
    out_of_repertoire: float
    word_plausibility: float
    characters: int

    @property
    def is_corrupted(self) -> bool:
        return self.score < 0.7


def _is_plausible_word(token: str) -> bool:
    """Un mot écrit dans la langue attendue.

    Trois critères, tous positifs — on décrit ce qu'est un mot correct, jamais
    ce à quoi ressemble une corruption :

      * ses lettres appartiennent à l'alphabet de la langue ;
      * il contient une voyelle ;
      * il n'a pas de majuscule en plein milieu.

    Les mots d'une ou deux lettres sont acceptés sans condition : ce sont des
    articles, des initiales ou des variables mathématiques.
    """

    if not token or len(token) > 30:
        return False
    if len(token) <= 2:
        return all(character.lower() in _FRENCH_ALPHABET for character in token)

    lowered = token.lower()
    if not all(character in _FRENCH_ALPHABET for character in lowered):
        return False
    if not any(character in _VOWELS for character in lowered):
        return False
    # `compŽtences`, `dŽmontrer` : une capitale surgit au milieu d'un mot en
    # minuscules. Aucun mot réel ne fait ça.
    interior = token[1:]
    if any(character.isupper() for character in interior) and not token.isupper():
        return False
    return True


def _out_of_repertoire_ratio(text: str) -> float:
    if not text:
        return 0.0
    unexpected = 0
    for character in text:
        if character in _ALLOWED_PUNCTUATION or character in _ALLOWED_MATH:
            continue
        category = unicodedata.category(character)
        if category.startswith("L") and "LATIN" in unicodedata.name(character, ""):
            continue
        if category.startswith("N"):
            continue
        unexpected += 1
    return unexpected / len(text)


def assess(text: str) -> Quality:
    """Noter un texte de 0 (illisible) à 1 (propre)."""

    characters = len(text)
    if characters == 0:
        return Quality(0.0, 0, 0.0, 0.0, 0)

    cid = len(_CID_MARKER.findall(text))
    stripped = _CID_MARKER.sub(" ", text)

    tokens = _TOKEN.findall(stripped)
    plausible = sum(1 for token in tokens if _is_plausible_word(token))
    plausibility = plausible / len(tokens) if tokens else 0.0

    out_of_repertoire = _out_of_repertoire_ratio(stripped)

    # Le score combine les trois signaux. La plausibilité des mots pèse le
    # plus : c'est le seul qui juge la langue plutôt que les octets.
    cid_penalty = min(1.0, cid / max(1, len(tokens)) * 10)
    score = max(
        0.0,
        min(
            1.0,
            plausibility * 0.7
            + (1.0 - min(1.0, out_of_repertoire * 20)) * 0.3
            - cid_penalty * 0.3,
        ),
    )

    return Quality(
        score=round(score, 4),
        cid_markers=cid,
        out_of_repertoire=round(out_of_repertoire, 5),
        word_plausibility=round(plausibility, 4),
        characters=characters,
    )


def corrupted_pages(
    pages: List[str],
    *,
    plausibility_floor: float = 0.80,
) -> List[int]:
    """Numéros (à partir de 1) des pages à repasser en OCR.

    Deux signaux complémentaires, mesurés sur des documents réels :

      * un marqueur `(cid:)` suffit à condamner la page — c'est l'extracteur
        lui-même qui déclare ne pas savoir lire un glyphe ;
      * en dessous d'un plancher de plausibilité, la page est illisible même
        sans marqueur, ce qui arrive quand la police mappe vers de mauvaises
        lettres au lieu de rien.

    Le plancher est volontairement bas : mieux vaut manquer une page douteuse
    que réOCRiser un document entier. Une page saine score au-dessus de 0,95,
    une page de maths dense reste au-dessus de 0,90.
    """

    flagged = []
    for index, page in enumerate(pages, start=1):
        if not page.strip():
            continue
        quality = assess(page)
        if quality.cid_markers > 0 or quality.word_plausibility < plausibility_floor:
            flagged.append(index)
    return flagged
