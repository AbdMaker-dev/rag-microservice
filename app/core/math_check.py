"""Vérifier les CALCULS d'une réponse avec SymPy, sans modèle.

Note d'évaluation du 16/09/2026 : Lawal écrivait 2e^{iπ/3} = 1 + i√3, juste,
puis « les coordonnées du point sont (1, 3) » — faux. Faire relire le modèle
par lui-même a été essayé le 15/09 et a rendu fausse une réponse juste : un
modèle ne peut pas être l'auteur et le juge. SymPy, lui, calcule.

Ce module ne juge QUE ce qui est un calcul entièrement chiffré :
- une chaîne d'égalités dans une formule (`\\sqrt{9+16} = \\sqrt{25} = 5`) :
  chaque paire voisine dont les deux côtés sont des nombres est comparée ;
- les coordonnées annoncées du point d'un nombre complexe chiffré.

Tout le reste est ignoré sans bruit : une formule à variables (z' = az + b),
une notation qu'on ne sait pas lire, une inégalité. **Ne rien dire vaut
mieux qu'un faux signalement** : une réponse juste ne doit jamais être
renvoyée en correction à cause d'une lecture ratée.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)

# Au-delà, une « formule » est un paragraphe : on ne s'y risque pas.
_MAX_PART = 160
_TOLERANCE = 1e-9

_FORMULA = re.compile(r"\\\((.+?)\\\)|\\\[(.+?)\\\]|\$\$(.+?)\$\$|\$(.+?)\$", re.DOTALL)
# Ce qui n'est pas une égalité à vérifier : on laisse la formule entière.
_NOT_EQUALITY = re.compile(r"\\neq|\\ne\b|\\leq?|\\geq?|\\approx|\\simeq|\\equiv|≠|≤|≥|≈|<|>|\\pmod|\\mod|\\in\b")
_PAIR = re.compile(r"\(\s*([^()]{1,40}?)\s*[;,]\s*([^()]{1,40}?)\s*\)")


@dataclass(frozen=True)
class Finding:
    """Une erreur de calcul, dite pour le modèle qui doit la corriger."""

    message: str


def _braced(text: str, start: int):
    """Le contenu de l'accolade qui s'ouvre en `start`, et l'indice qui suit."""

    if start >= len(text) or text[start] != "{":
        return None, start
    depth = 0
    for index in range(start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : index], index + 1
    return None, start


def _to_python(latex: str) -> Optional[str]:
    """Une expression LaTeX d'élève, en syntaxe SymPy — ou None si on ne
    sait pas la lire avec certitude."""

    s = latex.strip()
    for noise in ("\\left", "\\right", "\\,", "\\;", "\\!", "\\:", "\\displaystyle"):
        s = s.replace(noise, "")
    s = s.replace("\\cdot", "*").replace("\\times", "*").replace("×", "*").replace("·", "*")
    s = s.replace("−", "-").replace("\\pi", " pi ").replace("π", " pi ")
    s = s.replace("\\theta", " theta ").replace("θ", " theta ")
    out: List[str] = []
    index = 0
    while index < len(s):
        if s.startswith("\\frac", index) or s.startswith("\\dfrac", index) or s.startswith("\\tfrac", index):
            index = s.index("frac", index) + 4
            top, index = _braced(s, index)
            bottom, index = _braced(s, index)
            if top is None or bottom is None:
                return None
            top_py, bottom_py = _to_python(top), _to_python(bottom)
            if top_py is None or bottom_py is None:
                return None
            out.append(f" (({top_py})/({bottom_py}))")
            continue
        if s.startswith("\\sqrt", index):
            index += 5
            if index < len(s) and s[index] == "{":
                inner, index = _braced(s, index)
                inner_py = _to_python(inner) if inner is not None else None
                if inner_py is None:
                    return None
                out.append(f" sqrt({inner_py})")
            else:
                digits = re.match(r"\s*(\d+)", s[index:])
                if not digits:
                    return None
                out.append(f" sqrt({digits.group(1)})")
                index += digits.end()
            continue
        if s[index] == "√":
            index += 1
            if index < len(s) and s[index] in "({":
                close = ")" if s[index] == "(" else "}"
                end = s.find(close, index)
                if end < 0:
                    return None
                inner_py = _to_python(s[index + 1 : end])
                if inner_py is None:
                    return None
                out.append(f" sqrt({inner_py})")
                index = end + 1
            else:
                digits = re.match(r"\s*(\d+)", s[index:])
                if not digits:
                    return None
                out.append(f" sqrt({digits.group(1)})")
                index += digits.end()
            continue
        if s[index] == "^":
            index += 1
            if index < len(s) and s[index] == "{":
                inner, index = _braced(s, index)
                inner_py = _to_python(inner) if inner is not None else None
                if inner_py is None:
                    return None
                out.append(f"**({inner_py})")
            elif index < len(s):
                out.append(f"**({s[index]})")
                index += 1
            continue
        if s.startswith("\\cos", index) or s.startswith("\\sin", index) or s.startswith("\\tan", index):
            out.append(" " + s[index + 1 : index + 4] + " ")
            index += 4
            continue
        if s[index] == "\\":
            # Une commande qu'on ne connaît pas : on ne devine pas.
            return None
        if s[index] in "{}":
            out.append("(" if s[index] == "{" else ")")
            index += 1
            continue
        if s[index] == "²":
            out.append("**2")
            index += 1
            continue
        out.append(s[index])
        index += 1
    python = "".join(out)
    # Une virgule décimale (1,5) ou un couple : pas un nombre qu'on lit.
    if "," in python or ";" in python or "|" in python:
        return None
    return python


def _value(latex: str):
    """Le nombre complexe que vaut cette expression, ou None."""

    if not latex.strip() or len(latex) > _MAX_PART:
        return None
    python = _to_python(latex)
    if python is None:
        return None
    # Des exposants démesurés feraient calculer SymPy sans fin.
    if re.search(r"\*\*\(?\s*\d{4,}", python):
        return None
    try:
        import sympy
        from sympy.parsing.sympy_parser import (
            convert_xor,
            implicit_multiplication_application,
            parse_expr,
            standard_transformations,
        )

        local = {"i": sympy.I, "pi": sympy.pi, "sqrt": sympy.sqrt, "exp": sympy.exp,
                 "cos": sympy.cos, "sin": sympy.sin, "tan": sympy.tan,
                 "e": sympy.E}
        expr = parse_expr(
            python,
            local_dict=local,
            transformations=standard_transformations
            + (implicit_multiplication_application, convert_xor),
            evaluate=True,
        )
        if getattr(expr, "free_symbols", None):
            return None
        value = complex(sympy.N(expr, 15))
    except Exception:  # noqa: BLE001 — une lecture ratée ne signale rien
        return None
    return value


def _close(a: complex, b: complex) -> bool:
    return abs(a - b) <= _TOLERANCE * max(1.0, abs(a), abs(b))


def _formulas(text: str) -> List[str]:
    return [next(group for group in match.groups() if group is not None)
            for match in _FORMULA.finditer(text)]


def _exact(number: float) -> str:
    """√3 plutôt que 1.7321 : le modèle doit recopier la valeur du cours."""

    try:
        import sympy

        guess = sympy.nsimplify(number, [sympy.sqrt(2), sympy.sqrt(3), sympy.sqrt(5), sympy.pi],
                                tolerance=1e-9)
        if abs(float(guess) - number) < 1e-9:
            # En LaTeX : l'extrait et l'explication sont rendus par l'écran,
            # et « √3 » y resterait du texte brut au milieu d'une formule.
            text = re.sub(r"sqrt\((\d+)\)", r"\\sqrt{\1}", str(guess)).replace("*", " ")
            return text if text == f"{number:g}" else f"{text} \\approx {number:.4g}"
    except Exception:  # noqa: BLE001
        pass
    return f"{number:.4g}"


def _show(value: complex) -> str:
    real = round(value.real, 4)
    imag = round(value.imag, 4)
    if abs(imag) < 1e-12:
        return f"{real:g}"
    return f"{real:g} {'+' if imag >= 0 else '-'} {abs(imag):g} i"


def _equalities(text: str) -> List[Finding]:
    findings = []
    for formula in _formulas(text):
        if "=" not in formula or _NOT_EQUALITY.search(formula):
            continue
        parts = formula.split("=")
        values = [_value(part) for part in parts]
        for index in range(len(parts) - 1):
            left, right = values[index], values[index + 1]
            if left is None or right is None or _close(left, right):
                continue
            findings.append(Finding(
                f"L'égalité \\( {parts[index].strip()} = {parts[index + 1].strip()} \\) "
                f"est fausse : le côté gauche vaut \\( {_show(left)} \\), "
                f"le côté droit \\( {_show(right)} \\)."
            ))
    return findings


def _coordinates(text: str) -> List[Finding]:
    """« 1 + i√3 … coordonnées (1, 3) » : la partie imaginaire est-elle
    recopiée telle quelle ?"""

    findings = []
    sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
    for position, sentence in enumerate(sentences):
        if "coordonn" not in sentence.lower():
            continue
        # Le nombre est souvent calculé dans la phrase d'avant : « On a
        # … = 1 + i√3. Les coordonnées du point sont donc (1, 3). »
        window = (sentences[position - 1] + " " if position else "") + sentence
        complexes = []
        for formula in _formulas(window):
            if _PAIR.search(formula):
                continue
            last = formula.split("=")[-1]
            value = _value(last)
            if value is not None and abs(value.imag) > 1e-12:
                complexes.append((last.strip(), value))
        plain = _FORMULA.sub(lambda m: m.group(0) if _PAIR.search(m.group(0)) else " ", sentence)
        pairs = []
        for match in _PAIR.finditer(plain.replace("\\(", " ").replace("\\)", " ")):
            x, y = _value(match.group(1)), _value(match.group(2))
            if x is not None and y is not None:
                pairs.append((match.group(0), x, y))
        # Un seul nombre et un seul point dans la phrase : sinon on ne sait
        # pas qui correspond à qui, et on se tait.
        if len(complexes) != 1 or len(pairs) != 1:
            continue
        (expression, value), (shown, x, y) = complexes[0], pairs[0]
        if _close(complex(value.real), x) and _close(complex(value.imag), y):
            continue
        findings.append(Finding(
            f"Le point associé à \\( {expression} \\) a pour coordonnées "
            f"\\( ({_exact(value.real)} ; {_exact(value.imag)}) \\), pas \\( {shown} \\) : "
            "la partie réelle est l'abscisse, la partie imaginaire (sans le i) l'ordonnée."
        ))
    return findings


def check(text: str) -> List[Finding]:
    """Les erreurs de calcul CERTAINES de ce texte. Vide si rien de sûr."""

    try:
        return _equalities(text) + _coordinates(text)
    except Exception:  # noqa: BLE001 — le contrôle ne coûte jamais la réponse
        logger.warning("vérification des calculs impossible", exc_info=True)
        return []
