"""Verbaliser un cours : traduire l'écrit en français parlé, avant Piper.

Un moteur de synthèse lit ce qu'on lui donne : « x² + 3x ≥ 0 » lu tel quel
devient du charabia, et nos marqueurs « ## p. 12 » deviendraient « dièse
dièse p point douze ». Cette couche traduit tout en mots français par des
règles déterministes — la méthode des moteurs d'accessibilité (MathSpeak,
Speech Rule Engine), jamais un modèle qui improvise une lecture.

Un symbole que les règles ne couvrent pas reste tel quel : Piper l'ignore ou
l'épelle, et l'ajout d'une règle est un commit d'une ligne. On n'invente
rien à la place du professeur.
"""

from __future__ import annotations

import re

# Ordre d'application : les motifs à plusieurs caractères d'abord, sinon
# « ≥ » serait mangé par une règle sur « = » et « +∞ » par celle sur « + ».
_SEQUENCES = [
    ("<=", " est inférieur ou égal à "),
    (">=", " est supérieur ou égal à "),
    ("+∞", " plus l'infini"),
    ("-∞", " moins l'infini"),
    ("−∞", " moins l'infini"),
    ("m/s²", " mètres par seconde au carré"),
    ("m/s", " mètres par seconde"),
    ("km/h", " kilomètres par heure"),
    ("^2", " au carré"),
    ("^3", " au cube"),
]

_SYMBOLS = {
    "²": " au carré",
    "³": " au cube",
    "√": " racine carrée de ",
    "≤": " est inférieur ou égal à ",
    "≥": " est supérieur ou égal à ",
    "≠": " est différent de ",
    "≈": " est environ égal à ",
    "±": " plus ou moins ",
    "×": " multiplié par ",
    "÷": " divisé par ",
    "·": " scalaire ",
    "∞": " l'infini",
    "→": " tend vers ",
    "⇒": " implique ",
    "⇔": " équivaut à ",
    "∈": " appartient à ",
    "∉": " n'appartient pas à ",
    "⊂": " est inclus dans ",
    "∪": " union ",
    "∩": " inter ",
    "∑": " la somme de ",
    "∫": " l'intégrale de ",
    "π": " pi ",
    "α": " alpha ",
    "β": " bêta ",
    "γ": " gamma ",
    "δ": " delta ",
    "Δ": " delta ",
    "θ": " thêta ",
    "λ": " lambda ",
    "μ": " mu ",
    "µ": " mu ",
    "ω": " oméga ",
    "Ω": " oméga ",
    "°": " degrés ",
    "%": " pour cent ",
    "½": " un demi ",
    "¼": " un quart ",
    "¾": " trois quarts ",
}

# Indices : H₂O se lit « H 2 O ».
_SUBSCRIPTS = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
# Exposants au-delà du carré et du cube : x⁴ → « x puissance 4 ».
_SUPERSCRIPTS = {"⁰": "0", "¹": "1", "⁴": "4", "⁵": "5", "⁶": "6",
                 "⁷": "7", "⁸": "8", "⁹": "9"}

_PAGE_MARK = re.compile(r"^##+ *p\. *\d+ *$", re.MULTILINE)
_FIGURE_MARK = re.compile(r"\[FIGURE [^\]]+\]")
_LACUNE_MARK = re.compile(r"\[LACUNE[^\]]*\]")
_HEADING = re.compile(r"^#+ *", re.MULTILINE)
_EMPHASIS = re.compile(r"\*{1,2}([^*]+)\*{1,2}")
_VECTOR = re.compile(r"([A-Z]{1,3})⃗")
_TABLE_ROW = re.compile(r"^\|.*\|\s*$", re.MULTILINE)
_EQUALS = re.compile(r"(?<=[\w)\s]) ?= ?(?=[\w(\s])")
_PLUS = re.compile(r" \+ ")


def verbalize(text: str) -> str:
    """Le texte tel qu'un professeur le lirait à voix haute.

    Déterministe : le même texte donne toujours la même lecture, et chaque
    règle se voit ci-dessus. La sortie ne contient plus ni marqueurs ni
    syntaxe d'écran — seulement des phrases.
    """

    # Les repères d'écran ne se lisent pas ; les figures et tableaux
    # s'annoncent, puisque l'élève qui écoute ne les voit pas forcément.
    text = _PAGE_MARK.sub("", text)
    text = _FIGURE_MARK.sub("Une figure accompagne ce passage : regardez l'écran.", text)
    text = _LACUNE_MARK.sub("", text)
    tables = len(_TABLE_ROW.findall(text))
    text = _TABLE_ROW.sub("", text)
    if tables:
        text += "\nUn tableau accompagne ce passage : regardez l'écran."

    # La syntaxe d'affichage disparaît, le contenu reste : un titre se lit
    # comme une phrase, le gras ne s'entend pas.
    text = _HEADING.sub("", text)
    text = _EMPHASIS.sub(r"\1", text)

    # Les vecteurs avant tout : la flèche combinante colle à la lettre.
    text = _VECTOR.sub(r"le vecteur \1", text)

    for sequence, spoken in _SEQUENCES:
        text = text.replace(sequence, spoken)
    for symbol, spoken in _SYMBOLS.items():
        text = text.replace(symbol, spoken)
    for exponent, digit in _SUPERSCRIPTS.items():
        text = text.replace(exponent, f" puissance {digit}")
    text = text.translate(_SUBSCRIPTS)

    text = _EQUALS.sub(" égale ", text)
    text = _PLUS.sub(" plus ", text)

    # Piper marque les pauses sur la ponctuation : un paragraphe sans point
    # final en reçoit un. Par PARAGRAPHE, pas par ligne — un retour à la
    # ligne au milieu d'une phrase (« Leur produit\nscalaire ») n'est qu'un
    # repli d'affichage, pas une fin de phrase.
    paragraphs = []
    for block in re.split(r"\n\s*\n", text):
        joined = " ".join(line.strip() for line in block.split("\n") if line.strip())
        if not joined:
            continue
        if joined[-1] not in ".!?:;":
            joined += "."
        paragraphs.append(joined)
    return re.sub(r"[ \t]+", " ", " ".join(paragraphs)).strip()
