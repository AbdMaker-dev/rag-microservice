"""Réparation d'encodage, déterministe et par police.

Un PDF ancien déclare souvent ses polices avec une table de caractères et
encode son texte avec une autre. Le lecteur applique alors la mauvaise table
et « compétences » ressort en « compŽtences ». Ce n'est pas du bruit : c'est
une permutation, donc elle est réversible exactement.

On ne devine pas la permutation sur le document entier — un même PDF mélange
des polices saines et des polices décalées, et parfois deux décalages
opposés. On la détermine **police par police**, en essayant les candidats et
en gardant celui qui produit le plus de mots français plausibles.

Aucun modèle de langue n'intervient : un modèle comblerait les trous en
inventant, ce qui est le pire résultat possible sur du contenu pédagogique.
Ici, ce qui n'est pas reconnu reste visible, et le professeur le voit.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple

from app.core.quality import assess

logger = logging.getLogger(__name__)

_CID = re.compile(r"\(cid:(\d+)\)")
_TOKEN = re.compile(r"[^\W\d_]", re.UNICODE)
_WORDS = re.compile(r"[^\W\d_]+", re.UNICODE)

# Un candidat décrit une méprise : les octets étaient dans `reelle`, ils ont
# été lus comme `supposee`. Corriger, c'est refaire le chemin inverse.
_CANDIDATES: Tuple[Tuple[str, str, str], ...] = (
    ("aucune", "", ""),
    ("mac_roman lu en cp1252", "cp1252", "mac_roman"),
    ("cp1252 lu en mac_roman", "mac_roman", "cp1252"),
    ("latin-1 lu en cp1252", "cp1252", "latin-1"),
    ("cp1252 lu en latin-1", "latin-1", "cp1252"),
)

# Police Symbol (codage Adobe). Elle porte les mathématiques : la traiter
# comme du texte transformerait « ≤ » en « £ », donc une inégalité en une
# devise. Table explicite, jamais devinée.
_SYMBOL: Dict[int, str] = {
    0x22: "∀", 0x24: "∃", 0x2D: "−", 0x40: "≅",
    0x61: "α", 0x62: "β", 0x63: "χ", 0x64: "δ", 0x65: "ε", 0x66: "φ",
    0x67: "γ", 0x68: "η", 0x69: "ι", 0x6A: "ϕ", 0x6B: "κ", 0x6C: "λ",
    0x6D: "μ", 0x6E: "ν", 0x6F: "ο", 0x70: "π", 0x71: "θ", 0x72: "ρ",
    0x73: "σ", 0x74: "τ", 0x75: "υ", 0x76: "ϖ", 0x77: "ω", 0x78: "ξ",
    0x79: "ψ", 0x7A: "ζ",
    0x41: "Α", 0x42: "Β", 0x44: "Δ", 0x46: "Φ", 0x47: "Γ", 0x4C: "Λ",
    0x50: "Π", 0x51: "Θ", 0x53: "Σ", 0x57: "Ω", 0x58: "Ξ", 0x59: "Ψ",
    0xA3: "≤", 0xA5: "∞", 0xAB: "↔", 0xAC: "←", 0xAD: "↑", 0xAE: "→",
    0xAF: "↓", 0xB1: "±", 0xB3: "≥", 0xB4: "×", 0xB7: "•", 0xB8: "÷",
    0xB9: "≠", 0xBA: "≡", 0xBB: "≈", 0xBD: "|", 0xBE: "↵",
    0xC2: "ℜ", 0xC4: "⊗", 0xC5: "⊕", 0xC7: "∩", 0xC8: "∪", 0xC9: "⊃",
    0xCA: "⊇", 0xCB: "⊄", 0xCC: "⊂", 0xCD: "⊆", 0xCE: "∈", 0xCF: "∉",
    0xD0: "∠", 0xD1: "∇", 0xD5: "∏", 0xD6: "√", 0xD7: "⋅", 0xD8: "¬",
    0xD9: "∧", 0xDA: "∨", 0xDB: "⇔", 0xDC: "⇐", 0xDD: "⇑", 0xDE: "⇒",
    0xDF: "⇓", 0xE5: "∑", 0xF2: "∫", 0xBC: "…",
    0xE6: "⎛", 0xE7: "⎜", 0xE8: "⎝", 0xE9: "⎡", 0xEA: "⎢", 0xEB: "⎣",
    0xEC: "⎧", 0xED: "⎨", 0xEE: "⎩", 0xEF: "⎪",
    0xF6: "⎞", 0xF7: "⎟", 0xF8: "⎠", 0xF9: "⎤", 0xFA: "⎥", 0xFB: "⎦",
    0xFC: "⎫", 0xFD: "⎬", 0xFE: "⎭",
}

# En deçà, on considère qu'aucun candidat n'a rendu la police lisible et on
# préfère livrer le texte tel quel plutôt qu'un texte faux.
_ACCEPTANCE_FLOOR = 0.80

# Sous ce nombre de mots, un taux de plausibilité ne veut rien dire : on
# laisse la police tranquille au lieu de la « corriger » sur un coup de dé.
_MINIMUM_SAMPLE = 20


def _translation(supposee: str, reelle: str) -> Dict[int, str]:
    """Table de substitution d'un octet mal interprété vers son vrai caractère."""

    table: Dict[int, str] = {}
    for byte in range(0x80, 0x100):
        try:
            lu = bytes([byte]).decode(supposee)
            vrai = bytes([byte]).decode(reelle)
        except UnicodeDecodeError:
            continue
        # Une substitution vers un caractère de contrôle efface le problème au
        # lieu de le résoudre : le parasite disparaît à l'affichage et toute
        # mesure de qualité s'en trouve flattée. On refuse ces tables. L'espace
        # insécable, lui, est légitime — le français en met avant « : ».
        if lu != vrai and unicodedata.category(vrai) not in {"Cc", "Cf"}:
            table[ord(lu)] = vrai
    return table


# Toutes les tables candidates, préconstruites : le repêchage mot à mot les
# essaie une à une et n'accepte qu'un verdict unanime.
_ALL_TABLES: Tuple[Tuple[str, Dict[int, str], str], ...] = tuple(
    (label, _translation(supposee, reelle), reelle)
    for label, supposee, reelle in _CANDIDATES[1:]
    if _translation(supposee, reelle)
)


def _reads_as_french(word: str) -> bool:
    """Le mot est-il écrit dans la langue attendue, sans trou ?"""

    measured = assess(word)
    return measured.cid_markers == 0 and measured.word_plausibility >= 1.0


def _rescue(word: str) -> str:
    """Repêcher un mot resté illisible après la table de sa police.

    Une police peut être saine dans l'ensemble et porter quelques mots
    encodés autrement — un passage collé depuis un autre document, par
    exemple. La décision majoritaire ne les voit pas.

    On n'accepte la correction que si **un seul** candidat produit un mot
    français valide. Si deux tables donnent chacune un mot plausible, on ne
    peut pas trancher, et on laisse le mot abîmé : visible vaut mieux que
    faux.
    """

    if not _TOKEN.search(word) or _reads_as_french(word):
        return word

    # Une table qui remplace une lettre par une ponctuation coupe le mot en
    # deux fragments, chacun plausible pris isolément : « DÈfinition » devient
    # « D»finition », soit « D » et « finition ». Exiger que le nombre de mots
    # ne grandisse pas écarte ces fausses réussites.
    before = len(_WORDS.findall(word))

    found = set()
    for _label, table, reelle in _ALL_TABLES:
        candidate = _CID.sub(
            lambda match: _byte_as(int(match.group(1)), reelle, match.group(0)), word
        ).translate(table)
        if candidate == word or len(_WORDS.findall(candidate)) > before:
            continue
        if _reads_as_french(candidate):
            found.add(candidate)
    return found.pop() if len(found) == 1 else word


@dataclass(frozen=True)
class FontRepair:
    """Ce qu'il faut faire au texte d'une police donnée."""

    label: str
    table: Dict[int, str] = field(default_factory=dict)
    source_encoding: str = ""
    is_symbol: bool = False

    def apply(self, word: str) -> str:
        if self.is_symbol:
            return _repair_symbol(word)
        if self.source_encoding:
            word = _CID.sub(
                lambda match: _byte_as(int(match.group(1)), self.source_encoding, match.group(0)),
                word,
            )
        return word.translate(self.table) if self.table else word


@dataclass(frozen=True)
class RepairPlan:
    """Le plan complet : un mot abîmé, sa version réparée."""

    words: Dict[str, str]
    fonts: Dict[str, str]
    unreadable_fonts: List[str]

    @property
    def is_empty(self) -> bool:
        return not self.words


def _byte_as(code: int, encoding: str, fallback: str) -> str:
    """Un marqueur (cid:N) porte le code d'octet brut : on le relit."""

    if not 0 <= code <= 0xFF:
        return fallback
    try:
        return bytes([code]).decode(encoding)
    except (UnicodeDecodeError, LookupError):
        return fallback


def _repair_symbol(word: str) -> str:
    word = _CID.sub(lambda m: _SYMBOL.get(int(m.group(1)), m.group(0)), word)
    out = []
    for character in word:
        if character.isspace() or character.isdigit():
            out.append(character)
            continue
        try:
            code = character.encode("cp1252")[0]
        except (UnicodeEncodeError, IndexError):
            out.append(character)
            continue
        out.append(_SYMBOL.get(code, character))
    return "".join(out)


def _plausibility(words: Iterable[str]) -> float:
    """Proportion de mots qui ressemblent à du français écrit.

    On réutilise la mesure de `quality` : la même définition sert à choisir
    un candidat ici et à noter le document ensuite, donc les deux ne peuvent
    pas diverger.
    """

    sample = " ".join(words)
    return assess(sample).word_plausibility if sample.strip() else 0.0


def _fitness(original: List[str], translated: List[str]) -> float:
    """Juger un candidat sans se laisser abuser par la fragmentation.

    Une table qui remplace une lettre accentuée par une ponctuation coupe
    chaque mot en deux morceaux plausibles — « DÈfinition » devient
    « D»finition », soit « D » et « finition » — et la seule plausibilité
    grimpe alors à 1. On divise donc par le facteur de découpage : un candidat
    qui double le nombre de mots voit sa note réduite de moitié.
    """

    avant = sum(len(_WORDS.findall(word)) for word in original)
    apres = sum(len(_WORDS.findall(word)) for word in translated)
    fragmentation = max(1.0, apres / avant) if avant else 1.0
    return _plausibility(translated) / fragmentation


def _choose(words: List[str]) -> FontRepair:
    """Trouver la méprise qui rend cette police la plus lisible."""

    best = FontRepair(label="aucune")
    best_score = _fitness(words, words)

    for label, supposee, reelle in _CANDIDATES[1:]:
        table = _translation(supposee, reelle)
        if not table:
            continue
        translated = [word.translate(table) for word in words]
        score = _fitness(words, translated)
        if score > best_score:
            best, best_score = (
                FontRepair(label=label, table=table, source_encoding=reelle),
                score,
            )
    return best


def build_plan(pages_words: List[List[dict]]) -> RepairPlan:
    """Construire le plan de réparation d'un document.

    `pages_words` vient de `page.extract_words(extra_attrs=["fontname"])` :
    chaque mot connaît sa police, ce qui permet de décider police par police
    plutôt que d'appliquer une table unique à tout le document.
    """

    par_police: Dict[str, List[str]] = {}
    for words in pages_words:
        for word in words:
            par_police.setdefault(word["fontname"], []).append(word["text"])

    fonts: Dict[str, str] = {}
    unreadable: List[str] = []
    repairs: Dict[str, FontRepair] = {}

    for fontname, words in par_police.items():
        if "Symbol" in fontname:
            repairs[fontname] = FontRepair(label="symbol", is_symbol=True)
            fonts[fontname] = "symbol"
            continue
        if len(words) < _MINIMUM_SAMPLE:
            repairs[fontname] = FontRepair(label="échantillon insuffisant")
            fonts[fontname] = "ignorée"
            continue

        repair = _choose(words)
        repairs[fontname] = repair
        fonts[fontname] = repair.label
        if _fitness(words, [repair.apply(w) for w in words]) < _ACCEPTANCE_FLOOR:
            unreadable.append(fontname)

    # Un mot ne se répare que si toutes ses occurrences donnent le même
    # résultat. Sinon on n'y touche pas : mieux vaut un mot abîmé visible
    # qu'un mot corrigé au hasard dans un sens ou dans l'autre.
    candidates: Dict[str, set] = {}
    for words in pages_words:
        for word in words:
            original = word["text"]
            repaired = _rescue(repairs[word["fontname"]].apply(original))
            if repaired != original:
                candidates.setdefault(original, set()).add(repaired)

    mapping = {
        original: next(iter(variants))
        for original, variants in candidates.items()
        if len(variants) == 1
    }
    ambiguous = len(candidates) - len(mapping)
    if ambiguous:
        logger.info("mots ambigus laissés intacts", extra={"count": ambiguous})

    return RepairPlan(words=mapping, fonts=fonts, unreadable_fonts=sorted(unreadable))


def apply_plan(text: str, plan: RepairPlan) -> str:
    """Réécrire un texte extrait en substituant mot à mot.

    On remplace des mots entiers plutôt que des caractères : le même
    caractère peut être sain dans une police et abîmé dans une autre, alors
    qu'un mot porte sans ambiguïté la police qui l'a produit.
    """

    if plan.is_empty or not text:
        return text

    mapping = plan.words
    out: List[str] = []
    for line in text.split("\n"):
        out.append(
            " ".join(mapping.get(token, token) for token in line.split(" "))
        )
    return "\n".join(out)
