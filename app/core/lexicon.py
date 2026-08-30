"""Recoller les mots que l'extraction a coupés.

« théor ème », « déc embre », « INTRODUCTION GENERAL E » : l'espace parasite
vient du PDF lui-même — deux segments de texte distincts pour un même mot.
Pour la recherche, c'est une perte sèche : « théor ème » ne répondra jamais à
une requête « théorème ».

Le correcteur n'a pas besoin de dictionnaire externe : **le vocabulaire du
document suffit**. Un mot coupé apparaît presque toujours ailleurs dans le
même fichier sous sa forme entière — « théorème » figure dix fois dans un
programme de mathématiques. On ne recolle donc que sur preuve interne, et
jamais quand les deux morceaux sont eux-mêmes des mots du document : « définiti
on » se recolle parce que « définiti » n'existe pas, « rapport à » reste
intact parce que les deux existent.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Set

_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)
# Une élision française ne porte jamais d'espace après l'apostrophe :
# « d' octobre » est toujours « d'octobre ».
_ELISION = re.compile(r"\b([cdjlmnst]|qu)([’']) (?=[^\W\d_])", re.UNICODE | re.IGNORECASE)


# La preuve interne a une limite : dans une progression annuelle réelle,
# « décembre » et « mars » n'apparaissent QUE coupés — aucune forme entière
# ailleurs. Pour un ensemble fermé et universel comme les mois et les jours,
# l'amorçage relève de la langue, pas du document. On s'interdit d'aller plus
# loin : un vrai dictionnaire fusionnerait des mots légitimement séparés.
# Les mots grammaticaux courts : trop brefs pour entrer au lexique — qui
# commence à trois lettres — mais ce sont des mots à part entière. Sans eux,
# « la porte » fusionnait en « laporte » dès qu'un nom propre l'attestait :
# « la » ne comptant pas comme mot, la garde ne bloquait jamais.
_GRAMMAR = {
    "a", "à", "y", "au", "ai", "as", "ce", "de", "du", "en", "es", "et", "eu",
    "il", "je", "la", "le", "ma", "me", "ne", "ni", "nos", "on", "ou", "où",
    "sa", "se", "si", "ta", "te", "tu", "un", "va", "vu", "vos", "par", "sur",
    "les", "des", "aux", "est", "son", "ses", "mes", "tes", "que", "qui",
}

_SEED = {
    "janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août",
    "septembre", "octobre", "novembre", "décembre",
    "lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche",
}


def build(text: str) -> Counter:
    """Le vocabulaire du document : chaque mot d'au moins trois lettres."""

    words = Counter(
        word.lower() for word in _WORD.findall(text) if len(word) >= 3
    )
    for word in _SEED:
        words[word] += 1
    return words


def mend(text: str, words: Counter | None = None) -> str:
    """Recoller les mots coupés, sur preuve interne au document.

    Le balayage est manuel, pas un `re.sub` : une substitution consomme la
    paire qu'elle examine même quand elle la laisse intacte, si bien que dans
    « En déc embre », l'échec de « En déc » emportait « déc » et la vraie
    coupure n'était jamais essayée. Ici, l'échec d'une paire fait simplement
    avancer d'un jeton.
    """

    if words is None:
        words = build(text)

    mended_lines = []
    for line in text.split("\n"):
        tokens = [(m.group(), m.start(), m.end()) for m in _WORD.finditer(line)]
        gaps = []
        index = 0
        while index < len(tokens) - 1:
            left, _, left_end = tokens[index]
            right, right_start, _ = tokens[index + 1]
            if right_start == left_end + 1 and line[left_end] == " ":
                whole = (left + right).lower()
                def _is_word(token: str) -> bool:
                    lowered = token.lower()
                    return lowered in _GRAMMAR or words.get(lowered, 0) >= 2

                if (
                    words.get(whole, 0) >= 1
                    and len(left + right) >= 4
                    and not (_is_word(left) and _is_word(right))
                ):
                    gaps.append(left_end)
                    index += 2
                    continue
            index += 1
        for gap in reversed(gaps):
            line = line[:gap] + line[gap + 1 :]
        mended_lines.append(line)

    return _ELISION.sub(lambda m: f"{m.group(1)}{m.group(2)}", "\n".join(mended_lines))
