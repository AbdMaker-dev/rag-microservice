"""Réparation de l'encodage d'un PDF, à la source.

Un PDF ancien déclare ses polices avec une table de caractères et encode son
texte avec une autre. Corriger le texte après coup ne suffit pas : une fois
décodé, le même caractère peut venir de deux polices différentes et vouloir
dire deux choses — « ¥ » vaut une puce en police de texte et l'infini en
Symbol. Sur la chaîne, l'ambiguïté est insoluble.

On la tranche donc **avant** la lecture. On relève les codes d'octets réels de
chaque police, on détermine sa vraie table, et on inscrit dans le PDF une CMap
`/ToUnicode` qui dit au lecteur comment décoder. Le texte sort alors juste du
premier coup, et aucune police n'en contredit une autre.
"""

from __future__ import annotations

import contextlib
import io
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Zone à usage privé d'Unicode : on y projette temporairement les codes bruts,
# le temps de les relever. Aucun document réel n'utilise ces points de code.
_SENTINEL = 0xE000
_SENTINEL_SPAN = 0x2000

# Tables candidates pour une police de texte. L'ordre n'a pas d'importance :
# on garde la mieux notée, pas la première.
_TEXT_ENCODINGS: Tuple[str, ...] = (
    "cp1252", "mac_roman", "latin-1", "cp1250", "cp1251", "cp1253",
)

# Police Symbol (codage Adobe). Elle porte les mathématiques : la traiter comme
# du texte transformerait « ≤ » en « £ », donc une inégalité en une devise.
SYMBOL_ENCODING: Dict[int, str] = {
    0x22: "∀", 0x24: "∃", 0x27: "∍", 0x2D: "−", 0x40: "≅",
    0x41: "Α", 0x42: "Β", 0x43: "Χ", 0x44: "Δ", 0x45: "Ε", 0x46: "Φ",
    0x47: "Γ", 0x48: "Η", 0x49: "Ι", 0x4A: "ϑ", 0x4B: "Κ", 0x4C: "Λ",
    0x4D: "Μ", 0x4E: "Ν", 0x4F: "Ο", 0x50: "Π", 0x51: "Θ", 0x52: "Ρ",
    0x53: "Σ", 0x54: "Τ", 0x55: "Υ", 0x56: "ς", 0x57: "Ω", 0x58: "Ξ",
    0x59: "Ψ", 0x5A: "Ζ", 0x5C: "∴", 0x5E: "⊥",
    0x61: "α", 0x62: "β", 0x63: "χ", 0x64: "δ", 0x65: "ε", 0x66: "φ",
    0x67: "γ", 0x68: "η", 0x69: "ι", 0x6A: "ϕ", 0x6B: "κ", 0x6C: "λ",
    0x6D: "μ", 0x6E: "ν", 0x6F: "ο", 0x70: "π", 0x71: "θ", 0x72: "ρ",
    0x73: "σ", 0x74: "τ", 0x75: "υ", 0x76: "ϖ", 0x77: "ω", 0x78: "ξ",
    0x79: "ψ", 0x7A: "ζ",
    0xA0: "€", 0xA3: "≤", 0xA4: "⁄", 0xA5: "∞", 0xA6: "ƒ", 0xA7: "♣",
    0xA8: "♦", 0xA9: "♥", 0xAA: "♠", 0xAB: "↔", 0xAC: "←", 0xAD: "↑",
    0xAE: "→", 0xAF: "↓", 0xB0: "°", 0xB1: "±", 0xB2: "″", 0xB3: "≥",
    0xB4: "×", 0xB5: "∝", 0xB6: "∂", 0xB7: "•", 0xB8: "÷", 0xB9: "≠",
    0xBA: "≡", 0xBB: "≈", 0xBC: "…", 0xBD: "⏐", 0xBE: "⎯", 0xBF: "↵",
    0xC0: "ℵ", 0xC1: "ℑ", 0xC2: "ℜ", 0xC3: "℘", 0xC4: "⊗", 0xC5: "⊕",
    0xC6: "∅", 0xC7: "∩", 0xC8: "∪", 0xC9: "⊃", 0xCA: "⊇", 0xCB: "⊄",
    0xCC: "⊂", 0xCD: "⊆", 0xCE: "∈", 0xCF: "∉", 0xD0: "∠", 0xD1: "∇",
    0xD5: "∏", 0xD6: "√", 0xD7: "⋅", 0xD8: "¬", 0xD9: "∧", 0xDA: "∨",
    0xDB: "⇔", 0xDC: "⇐", 0xDD: "⇑", 0xDE: "⇒", 0xDF: "⇓",
    0xE5: "∑", 0xF2: "∫",
    0xE6: "⎛", 0xE7: "⎜", 0xE8: "⎝", 0xE9: "⎡", 0xEA: "⎢", 0xEB: "⎣",
    0xEC: "⎧", 0xED: "⎨", 0xEE: "⎩", 0xEF: "⎪",
    0xF6: "⎞", 0xF7: "⎟", 0xF8: "⎠", 0xF9: "⎤", 0xFA: "⎥", 0xFB: "⎦",
    0xFC: "⎫", 0xFD: "⎬", 0xFE: "⎭",
}

# Ces polices ne portent pas de langue : leur codage n'est pas une table de
# texte et les mettre en concurrence avec cp1252 n'a aucun sens.
_SYMBOLIC_NAMES = ("Symbol", "Wingdings", "ZapfDingbats", "Dingbats", "MT-Extra")

# Caractères qui ne se rencontrent pratiquement jamais dans un texte français
# et signent donc une méprise d'encodage. Liste volontairement courte : la
# typographie légitime — puces, points de suspension, guillemets, tirets — n'y
# a pas sa place, sinon une table qui décode correctement se voit pénalisée
# pour avoir bien travaillé.
_MOJIBAKE_SIGNATURE = set("ŽÕƒÏ™šÐ¥ˆ‰‹›¡¢¤¦¨ª¬¯¶¸º")

# Répertoire attendu d'un texte français.
_FRENCH = set("abcdefghijklmnopqrstuvwxyzàâäçéèêëîïôöùûüÿœæ")
_FRENCH |= {c.upper() for c in _FRENCH}
_NEUTRAL = set(" \t\n\r\xa0.,;:!?'\"()[]{}-–—/\\%&*+=<>0123456789")
# Typographie française légitime : elle ne doit jamais compter comme un défaut.
_NEUTRAL |= set("•…‘’“”«»°±×÷§©®")

# Sous ce nombre de caractères non-ASCII, une police ne dit rien de fiable.
_MINIMUM_SAMPLE = 8
# On n'écarte la table déclarée que si le gain est net...
_MINIMUM_GAIN = 0.15
# ...et si la deuxième hypothèse est nettement derrière.
_MINIMUM_MARGIN = 0.10


@dataclass(frozen=True)
class FontDecision:
    """Ce qu'on a conclu pour une police, et avec quelle assurance."""

    fontname: str
    encoding: Optional[str]
    confidence: float
    samples: int
    is_symbolic: bool = False

    @property
    def changed(self) -> bool:
        return self.encoding is not None


@contextlib.contextmanager
def _raw_codes():
    """Neutraliser la conversion code → Unicode de pdfminer.

    Chaque caractère rend alors son code d'origine, projeté dans la zone à
    usage privé. C'est le seul moyen de voir ce que le PDF contient vraiment,
    avant que la mauvaise table ne s'applique.
    """

    from pdfminer.pdffont import PDFCIDFont, PDFFont, PDFSimpleFont

    originals = {}
    for klass in (PDFFont, PDFSimpleFont, PDFCIDFont):
        if "to_unichr" in klass.__dict__:
            originals[klass] = klass.to_unichr
            klass.to_unichr = lambda self, cid: chr(
                _SENTINEL + (cid % _SENTINEL_SPAN)
            )
    try:
        yield
    finally:
        for klass, original in originals.items():
            klass.to_unichr = original


def scan_codes(payload: bytes) -> Dict[str, Counter]:
    """Relever, pour chaque police, les codes d'octets réellement utilisés."""

    import pdfplumber

    counts: Dict[str, Counter] = defaultdict(Counter)
    with _raw_codes():
        with pdfplumber.open(io.BytesIO(payload)) as document:
            for page in document.pages:
                for char in page.chars:
                    text = char.get("text") or ""
                    if not text:
                        continue
                    code = ord(text[0])
                    if _SENTINEL <= code < _SENTINEL + _SENTINEL_SPAN:
                        counts[char["fontname"]][code - _SENTINEL] += 1
    return dict(counts)


def fonts_with_tounicode(payload: bytes) -> set:
    """Polices qui déclarent déjà leur table de caractères.

    Une CMap /ToUnicode présente est la parole du producteur du document :
    elle fait autorité. La contredire revient à corrompre un texte correct —
    c'est exactement ce qui arrivait sur un document sain, où des polices
    sous-ensemblées voyaient leurs index de glyphes relus comme du MacRoman.
    """

    from pypdf import PdfReader

    with_map: set = set()
    without_map: set = set()
    try:
        reader = PdfReader(io.BytesIO(payload))
        seen: set = set()
        for page in reader.pages:
            resources = page.get("/Resources")
            if resources is None:
                continue
            fonts = resources.get_object().get("/Font")
            if fonts is None:
                continue
            for reference in fonts.get_object().values():
                font = reference.get_object()
                if id(font) in seen:
                    continue
                seen.add(id(font))
                name = str(font.get("/BaseFont", "")).lstrip("/")
                (with_map if "/ToUnicode" in font else without_map).add(name)
    except Exception:  # noqa: BLE001
        logger.warning("lecture des polices impossible, aucune correction tentée")
        return set()

    # Un même nom peut désigner plusieurs objets police, les uns déclarant leur
    # table et les autres non — c'est le cas dans les deux documents de
    # référence. On n'accorde sa confiance à un nom que si **tous** ses objets
    # la déclarent ; sinon le diagnostic reste nécessaire pour les autres.
    return with_map - without_map


# Fenêtres Unicode des écritures que l'on sait reconnaître. Le service reçoit
# des documents du monde entier : noter des candidates « part de caractères
# latins » sur un texte arabe ou cyrillique choisirait n'importe quoi, et
# pourrait basculer une police saine vers une table fausse.
_SCRIPTS = {
    "latin": ((0x0041, 0x024F),),
    "greek": ((0x0370, 0x03FF), (0x1F00, 0x1FFF)),
    "cyrillic": ((0x0400, 0x04FF),),
    "arabic": ((0x0600, 0x06FF), (0x0750, 0x077F)),
    "hebrew": ((0x0590, 0x05FF),),
    "cjk": ((0x3040, 0x30FF), (0x4E00, 0x9FFF), (0xAC00, 0xD7AF)),
    "devanagari": ((0x0900, 0x097F),),
}


def dominant_script(payload: bytes, trusted: set, pages: int = 20) -> str:
    """Deviner l'écriture du document à partir de sa portion déjà fiable.

    On ne lit que les polices qui déclarent leur table : celles-là sortent un
    texte correct, et disent donc la vérité sur la langue du document.
    """

    import pdfplumber

    tally: Counter = Counter()
    try:
        with pdfplumber.open(io.BytesIO(payload)) as document:
            for page in document.pages[:pages]:
                for char in page.chars:
                    if trusted and char.get("fontname") not in trusted:
                        continue
                    text = char.get("text") or ""
                    if not text or not text[0].isalpha():
                        continue
                    point = ord(text[0])
                    for name, spans in _SCRIPTS.items():
                        if any(low <= point <= high for low, high in spans):
                            tally[name] += 1
                            break
    except Exception:  # noqa: BLE001
        return "unknown"

    if not tally:
        return "unknown"
    name, count = tally.most_common(1)[0]
    return name if count / sum(tally.values()) >= 0.6 else "mixed"


def _character_value(character: str) -> float:
    if character in _MOJIBAKE_SIGNATURE:
        return -1.0
    if character in _FRENCH or character in _NEUTRAL:
        return 1.0
    return 0.0


def _score(codes: Counter, encoding: str) -> float:
    """Noter une table sur la distribution réelle des codes d'une police."""

    total = sum(count for code, count in codes.items() if code > 0x7F)
    if not total:
        return 0.0
    earned = 0.0
    for code, count in codes.items():
        if code <= 0x7F or code > 0xFF:
            continue
        try:
            character = bytes([code]).decode(encoding)
        except (UnicodeDecodeError, LookupError):
            earned -= count
            continue
        earned += count * _character_value(character)
    return earned / total


def is_symbolic(fontname: str) -> bool:
    return any(name in fontname for name in _SYMBOLIC_NAMES)


def decide(
    counts: Dict[str, Counter],
    current: Dict[str, str],
    trusted: Optional[set] = None,
    script: str = "latin",
) -> List[FontDecision]:
    """Déterminer la vraie table de chaque police.

    `current` donne la table que le PDF déclare, pour ne la contredire que sur
    un écart net : mieux vaut laisser une police douteuse en l'état que
    corrompre une police déjà correcte.
    """

    trusted = trusted or set()
    decisions: List[FontDecision] = []

    # Nos tables candidates décrivent des écritures latines. Sur un document
    # arabe, russe ou chinois, les noter n'a aucun sens : on s'abstient.
    if script not in ("latin", "unknown"):
        return [
            FontDecision(name, None, 0.0, sum(counts[name].values()))
            for name in counts
        ]

    for fontname, codes in counts.items():
        samples = sum(count for code, count in codes.items() if 0x7F < code <= 0xFF)

        # Une police qui déclare déjà sa table fait autorité.
        if fontname in trusted:
            decisions.append(FontDecision(fontname, None, 0.0, samples))
            continue

        if samples < _MINIMUM_SAMPLE:
            decisions.append(FontDecision(fontname, None, 0.0, samples))
            continue

        if is_symbolic(fontname):
            decisions.append(
                FontDecision(fontname, "symbol", 1.0, samples, is_symbolic=True)
            )
            continue

        declared = current.get(fontname, "cp1252")
        base = _score(codes, declared)
        ranked = sorted(
            ((_score(codes, name), name) for name in _TEXT_ENCODINGS),
            reverse=True,
        )
        best_score, best_name = ranked[0]
        runner_up = ranked[1][0] if len(ranked) > 1 else -1.0

        gain = best_score - base
        margin = best_score - runner_up
        if best_name != declared and gain >= _MINIMUM_GAIN and margin >= _MINIMUM_MARGIN:
            decisions.append(
                FontDecision(fontname, best_name, round(min(gain, 1.0), 3), samples)
            )
        else:
            decisions.append(FontDecision(fontname, None, round(gain, 3), samples))
    return decisions


def unicode_map(decision: FontDecision, codes: Iterable[int]) -> Dict[int, str]:
    """La correspondance code → caractère à inscrire dans le PDF."""

    if decision.is_symbolic:
        return {code: SYMBOL_ENCODING[code] for code in codes if code in SYMBOL_ENCODING}
    if not decision.encoding:
        return {}
    mapping: Dict[int, str] = {}
    for code in codes:
        if not 0 <= code <= 0xFF:
            continue
        try:
            mapping[code] = bytes([code]).decode(decision.encoding)
        except UnicodeDecodeError:
            continue
    return mapping


# Ce que le PDF déclare, traduit en codec Python. Une police sans déclaration
# est présumée en cp1252, l'hypothèse la plus courante des producteurs de PDF.
_DECLARED = {
    "/WinAnsiEncoding": "cp1252",
    "/MacRomanEncoding": "mac_roman",
    "/MacExpertEncoding": "cp1252",
    "/StandardEncoding": "cp1252",
}


def declared_encodings(payload: bytes) -> Dict[str, str]:
    """Lire la table que le PDF déclare pour chaque police."""

    from pypdf import PdfReader

    found: Dict[str, str] = {}
    reader = PdfReader(io.BytesIO(payload))
    for page in reader.pages:
        resources = page.get("/Resources")
        if resources is None:
            continue
        fonts = resources.get_object().get("/Font")
        if fonts is None:
            continue
        for reference in fonts.get_object().values():
            font = reference.get_object()
            base = str(font.get("/BaseFont", "")).lstrip("/")
            if not base:
                continue
            encoding = font.get("/Encoding")
            if encoding is None:
                continue
            encoding = encoding.get_object()
            name = (
                str(encoding)
                if not hasattr(encoding, "get")
                else str(encoding.get("/BaseEncoding", ""))
            )
            if name in _DECLARED:
                found[base] = _DECLARED[name]
    return found


def differences_map(font) -> Dict[int, str]:
    """Traduire le tableau /Differences déclaré par une police.

    C'est la parole du document : il énonce lui-même quel code porte quel
    glyphe. Quand elle existe sans /ToUnicode, on la matérialise plutôt que de
    laisser chaque lecteur la réinterpréter à sa façon.
    """

    from pdfminer.encodingdb import name2unicode

    encoding = font.get("/Encoding")
    if encoding is None:
        return {}
    encoding = encoding.get_object()
    if not hasattr(encoding, "get"):
        return {}
    differences = encoding.get("/Differences")
    if not differences:
        return {}

    mapping: Dict[int, str] = {}
    code = 0
    for item in differences.get_object():
        raw = item.get_object() if hasattr(item, "get_object") else item
        if isinstance(raw, (int, float)):
            code = int(raw)
            continue
        name = str(raw).lstrip("/")
        try:
            mapping[code] = name2unicode(name)
        except Exception:  # noqa: BLE001
            pass  # glyphe hors répertoire connu : on le laisse tel quel
        code += 1
    return mapping


def _cmap(mapping: Dict[int, str]) -> bytes:
    """Composer une CMap /ToUnicode pour une police.

    C'est le tableau que le lecteur consultera pour traduire chaque code en
    caractère. En l'inscrivant dans le PDF, on corrige la cause au lieu de
    rattraper les conséquences.
    """

    entries = sorted(mapping.items())
    lines = [
        "/CIDInit /ProcSet findresource begin",
        "12 dict begin",
        "begincmap",
        "/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def",
        "/CMapName /Adobe-Identity-UCS def",
        "/CMapType 2 def",
        "1 begincodespacerange",
        "<00> <FF>",
        "endcodespacerange",
    ]
    # Le format impose des blocs d'au plus 100 correspondances.
    for start in range(0, len(entries), 100):
        block = entries[start : start + 100]
        lines.append(f"{len(block)} beginbfchar")
        for code, character in block:
            target = "".join(f"{unit:04X}" for unit in _utf16_units(character))
            lines.append(f"<{code:02X}> <{target}>")
        lines.append("endbfchar")
    lines += [
        "endcmap",
        "CMapName currentdict /CMap defineresource pop",
        "end",
        "end",
    ]
    return "\n".join(lines).encode("ascii", "replace")


def _utf16_units(character: str) -> List[int]:
    encoded = character.encode("utf-16-be")
    return [
        (encoded[index] << 8) | encoded[index + 1]
        for index in range(0, len(encoded), 2)
    ]


def rewrite(payload: bytes, decisions: List[FontDecision], counts: Dict[str, Counter]) -> bytes:
    """Réécrire le PDF avec la bonne table pour chaque police.

    Le document sort identique à l'original, à ceci près que chaque police
    corrigée porte désormais une CMap /ToUnicode. Aucune n'en avait : on
    n'écrase donc rien, on complète.
    """

    from pypdf import PdfReader, PdfWriter
    from pypdf.generic import DecodedStreamObject, NameObject

    plan = {d.fontname: d for d in decisions if d.changed}
    if not plan:
        return payload

    reader = PdfReader(io.BytesIO(payload))
    writer = PdfWriter(clone_from=reader)

    patched: set = set()
    applied = 0
    for page in writer.pages:
        resources = page.get("/Resources")
        if resources is None:
            continue
        fonts = resources.get_object().get("/Font")
        if fonts is None:
            continue
        for reference in fonts.get_object().values():
            font = reference.get_object()
            identity = id(font)
            if identity in patched:
                continue
            patched.add(identity)

            # Ne jamais écraser la table d'un objet qui en déclare une :
            # elle fait autorité, même si d'autres objets du même nom n'en ont
            # pas et ont besoin, eux, d'être corrigés.
            if "/ToUnicode" in font:
                continue

            base = str(font.get("/BaseFont", "")).lstrip("/")
            decision = plan.get(base)
            if decision is None:
                # Pas de diagnostic pour cette police : si elle déclare un
                # /Differences, on le matérialise. Le document se décrit alors
                # lui-même, et n'importe quel lecteur en tire le même texte.
                declared_map = differences_map(font)
                if not declared_map:
                    continue
                mapping = declared_map
            else:
                mapping = unicode_map(decision, counts.get(base, {}))
            # Les codes ASCII se traduisent par eux-mêmes. Les inscrire évite
            # qu'un lecteur ne trouve rien pour eux une fois la CMap présente.
            for code in counts.get(base, {}):
                if 0x20 <= code <= 0x7E:
                    mapping.setdefault(code, chr(code))
            if not mapping:
                continue

            stream = DecodedStreamObject()
            stream.set_data(_cmap(mapping))
            font[NameObject("/ToUnicode")] = writer._add_object(stream)
            applied += 1

    buffer = io.BytesIO()
    writer.write(buffer)
    logger.info("polices corrigées dans le PDF", extra={"fonts": applied})
    return buffer.getvalue()


def repair_pdf(payload: bytes) -> Tuple[bytes, List[FontDecision]]:
    """Point d'entrée : rendre un PDF lisible avant toute extraction."""

    counts = scan_codes(payload)
    if not counts:
        return payload, []
    trusted = fonts_with_tounicode(payload)
    script = dominant_script(payload, trusted)
    decisions = decide(counts, declared_encodings(payload), trusted, script)
    if not any(d.changed for d in decisions):
        # Rien à corriger : on rend le document tel quel, sans copie. La
        # plupart des PDF sont correctement encodés et n'ont rien à gagner
        # à passer par une réécriture.
        return payload, decisions
    return rewrite(payload, decisions, counts), decisions
