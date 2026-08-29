"""Situer les passages douteux d'un texte extrait.

Une note globale dit qu'un document est bon ou mauvais ; elle ne dit pas *où*
regarder. Un professeur qui doit relire cent pages a besoin qu'on lui désigne
les dix endroits qui posent question, pas qu'on lui donne un pourcentage.

Chaque signal est repéré par ses positions exactes dans le texte de la
section, pour que l'interface puisse surligner. Aucun modèle de langue n'est
requis : ces défauts se voient à la forme du texte.

Les formules pèsent double. Un cours de mathématiques dont une équation est
fausse est plus dangereux qu'un cours mal découpé : le second se voit, la
première s'enseigne.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple

# Caractères qui signent une méprise d'encodage non résolue.
#
# « È » et « Ê » en sont exclus : ce sont des majuscules françaises
# parfaitement légitimes — ALGÈBRE, FENÊTRE — et les y laisser faisait
# passer un titre de chapitre pour du texte abîmé. « Ë » et « Í » restent,
# leur emploi en français étant assez rare pour que leur présence intrigue.
_MOJIBAKE = set("ŽÕƒÏ™šÐ¥ˆ‰‹›¡¢¤¦¨ª¬¯¶¸ºËÍ")

# Symboles mathématiques : leur présence rend la section sensible.
_MATH = set("≤≥≠≈±×÷√∑∏∫∈∉⊂⊆∩∪∧∨⇔⇒→←↔∞∀∃⊥∥∠°αβγδθλμπσωΩΔΣΠ")

_TOKEN = re.compile(r"[^\W\d_]+", re.UNICODE)
_FRENCH = set("abcdefghijklmnopqrstuvwxyzàâäçéèêëîïôöùûüÿœæ")
_VOWELS = set("aeiouyàâäéèêëîïôöùûÿœæ")

# Les mots français d'une ou deux lettres. Sans cette liste, « I et si »
# passe pour du texte haché alors que c'est du français ordinaire.
_SHORT_WORDS = {
    "a", "à", "y", "au", "ai", "as", "ce", "de", "du", "en", "es", "et",
    "eu", "il", "je", "la", "le", "ma", "me", "ne", "ni", "on", "ou", "où",
    "sa", "se", "si", "ta", "te", "tu", "un", "va", "vu",
}

# Un fragment : une ou deux LETTRES seules. Ni chiffre, ni ponctuation, ni
# barre de tableau — « ax + b », « | | | » et « S2 et S4 » sont corrects et
# ne doivent pas passer pour du texte haché.
_FRAGMENT = re.compile(r"^[^\W\d_]{1,2}$", re.UNICODE)

# « a) », « 2) », « IV) » sont des puces d'énumération, pas des parenthèses.
_ENUMERATION = re.compile(r"[0-9]+\)|\b[a-zA-Z]{1,3}\)")

# En dessous, une section est trop pauvre pour avoir été correctement extraite.
_THIN = 25


@dataclass(frozen=True)
class Issue:
    """Un passage à vérifier, et où le trouver."""

    kind: str
    start: int
    end: int
    excerpt: str


# Ce que chaque défaut coûte à la confiance de la section.
_WEIGHTS = {
    "FORMULA": 0.20,      # une équation fausse s'enseigne : elle pèse double
    "ENCODING": 0.10,
    "UNREADABLE": 0.10,
    "OCR_NOISE": 0.08,
    "THIN": 0.30,
}


def _plausible(token: str) -> bool:
    lowered = token.lower()
    return (
        len(lowered) <= 2
        or (set(lowered) <= _FRENCH and bool(_VOWELS & set(lowered)))
    )


# Une capitale accentuée au milieu d'un mot, suivie d'une minuscule :
# « RÈpublique », « SÈnÈgal ». C'est la signature d'un accent perdu, et elle se
# passe d'énumérer les caractères suspects.
#
# Un mot tout en capitales n'y répond pas — « ALGÈBRE », « THÉORÈME » — parce
# que la lettre suivante est elle aussi une capitale. C'est ce qui permet de
# distinguer un titre de chapitre d'un texte abîmé, là où une simple liste de
# caractères interdits confondait les deux.
_BROKEN_CASE = re.compile(r"(?<=[^\W\d_])([ÀÁÂÃÄÅÇÈÉÊËÌÍÎÏÑÒÓÔÕÖÙÚÛÜÝ])(?=[a-zà-öø-ÿ])")


def _encoding_issues(text: str) -> List[Issue]:
    found = [
        Issue("ENCODING", index, index + 1, text[max(0, index - 20): index + 20])
        for index, character in enumerate(text)
        if character in _MOJIBAKE
    ]
    # Une même position peut répondre aux deux règles — « Ë » est à la fois un
    # caractère de signature et une capitale intruse. On ne la signale qu'une.
    seen = {issue.start for issue in found}
    found.extend(
        Issue("ENCODING", match.start(1), match.end(1),
              text[max(0, match.start() - 20): match.end() + 20])
        for match in _BROKEN_CASE.finditer(text)
        if match.start(1) not in seen
    )
    return found


def _ocr_issues(text: str) -> List[Issue]:
    """Suites de fragments qui ne sont pas des mots : « Mo o x »."""

    found: List[Issue] = []
    run: List[Tuple[int, int]] = []
    for match in re.finditer(r"\S+", text):
        token = match.group()
        if _FRAGMENT.match(token) and token.lower() not in _SHORT_WORDS:
            run.append((match.start(), match.end()))
            continue
        if len(run) >= 3:
            found.append(Issue("OCR_NOISE", run[0][0], run[-1][1],
                               text[run[0][0]:run[-1][1]]))
        run = []
    if len(run) >= 3:
        found.append(Issue("OCR_NOISE", run[0][0], run[-1][1],
                           text[run[0][0]:run[-1][1]]))
    return found


def _line_issues(text: str) -> List[Issue]:
    """Lignes dont presque aucun mot n'appartient à la langue attendue."""

    found: List[Issue] = []
    offset = 0
    for line in text.split("\n"):
        stripped = line.strip()
        tokens = [token for token in _TOKEN.findall(stripped) if len(token) > 2]
        if len(tokens) >= 4:
            good = sum(1 for token in tokens if _plausible(token))
            if good / len(tokens) < 0.5:
                found.append(Issue("UNREADABLE", offset, offset + len(line), stripped[:80]))
        offset += len(line) + 1
    return found


def _formula_issues(text: str) -> List[Issue]:
    """Formules visiblement incomplètes.

    Deux cas se voient sans comprendre les mathématiques : un symbole isolé
    entre deux espaces, qui a perdu ses opérandes, et des parenthèses
    déséquilibrées sur une ligne qui porte des symboles.
    """

    found: List[Issue] = []
    for match in re.finditer(r"(?:(?<=\s)|^)([" + "".join(_MATH) + r"])(?=\s|$)", text):
        before = text[max(0, match.start() - 3): match.start()].strip()
        after = text[match.end(): match.end() + 3].strip()
        if not before or not after:
            found.append(
                Issue("FORMULA", match.start(), match.end(),
                      text[max(0, match.start() - 30): match.end() + 30])
            )

    offset = 0
    for line in text.split("\n"):
        cleaned = _ENUMERATION.sub("", line)
        if _MATH & set(cleaned) and cleaned.count("(") != cleaned.count(")"):
            found.append(Issue("FORMULA", offset, offset + len(line), line.strip()[:80]))
        offset += len(line) + 1
    return found


def inspect_section(text: str) -> Tuple[float, List[Issue]]:
    """Noter une section et situer ce qui y pose question."""

    issues: List[Issue] = []
    # Une section courte reste examinée : « RÈpublique du SÈnÈgal » fait
    # vingt et un caractères, et dire au professeur qu'elle est « trop
    # courte » lui cache que son vrai défaut est un accent perdu.
    if len(text.strip()) < _THIN:
        issues.append(Issue("THIN", 0, len(text), text.strip()[:60]))
    issues.extend(_encoding_issues(text))
    issues.extend(_ocr_issues(text))
    issues.extend(_line_issues(text))
    issues.extend(_formula_issues(text))

    # La pénalité est rapportée à la longueur : deux défauts sur trois lignes
    # sont graves, les mêmes sur trois pages ne le sont pas.
    scale = max(1.0, len(text) / 1000)
    penalty = sum(_WEIGHTS.get(issue.kind, 0.05) for issue in issues) / scale
    confidence = max(0.0, min(1.0, 1.0 - penalty))

    issues.sort(key=lambda issue: issue.start)
    return round(confidence, 3), issues
