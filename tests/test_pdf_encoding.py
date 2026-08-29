"""La correction se décide police par police, et n'ose que sur un écart net."""

from __future__ import annotations

from collections import Counter

from app.core.pdf_encoding import (
    SYMBOL_ENCODING,
    FontDecision,
    _cmap,
    decide,
    is_symbolic,
    unicode_map,
)


def _codes_mac_roman():
    """Distribution typique d'une police MacRoman lue comme du cp1252.

    0x8E est « é », 0x8F « è », 0xC9 « … », 0xA5 « • ».
    """

    return Counter({0x8E: 400, 0x8F: 50, 0xC9: 480, 0xA5: 70, 0x41: 900})


def test_reconnait_une_police_mac_roman_declaree_en_cp1252():
    decisions = decide({"Times": _codes_mac_roman()}, {"Times": "cp1252"})

    assert decisions[0].encoding == "mac_roman"
    assert decisions[0].samples == 1000


def test_ne_touche_pas_une_police_sur_trop_peu_de_caracteres():
    """Sous le seuil d'échantillon, une note ne veut rien dire.

    Corriger sur un coup de dé abîmerait une police peut-être saine.
    """

    decisions = decide({"Rare": Counter({0x8E: 3})}, {})

    assert decisions[0].encoding is None
    assert decisions[0].samples == 3


def test_ecarte_les_polices_symboliques_du_concours_des_tables():
    assert is_symbolic("Symbol")
    assert is_symbolic("ENNMNA+Wingdings")
    assert is_symbolic("BOKNFC+MT-Extra")
    assert not is_symbolic("ENNLHC+TimesNewRomanPSMT")

    decisions = decide({"Symbol": Counter({0xA3: 30})}, {})

    assert decisions[0].encoding == "symbol"
    assert decisions[0].is_symbolic


def test_le_meme_code_ne_veut_pas_dire_la_meme_chose_selon_la_police():
    """C'est la raison d'être du module.

    En Symbol, 0xA5 est l'infini ; en police de texte MacRoman, c'est une puce.
    Corriger après décodage rend ces deux cas indiscernables.
    """

    symbole = FontDecision("Symbol", "symbol", 1.0, 40, is_symbolic=True)
    texte = FontDecision("Times", "mac_roman", 1.0, 40)

    assert unicode_map(symbole, [0xA5])[0xA5] == "∞"
    assert unicode_map(texte, [0xA5])[0xA5] == "•"


def test_la_table_symbol_restitue_les_mathematiques():
    assert SYMBOL_ENCODING[0xA3] == "≤"
    assert SYMBOL_ENCODING[0x6C] == "λ"
    assert SYMBOL_ENCODING[0xDB] == "⇔"
    assert SYMBOL_ENCODING[0xD6] == "√"


def test_la_cmap_produite_est_bien_formee():
    rendu = _cmap({0x41: "A", 0x8E: "é"}).decode("ascii")

    assert "begincmap" in rendu and "endcmap" in rendu
    assert "2 beginbfchar" in rendu
    assert "<8E> <00E9>" in rendu
    assert rendu.count("beginbfchar") == rendu.count("endbfchar")


def test_la_cmap_decoupe_en_blocs_de_cent_au_plus():
    """Le format PDF n'accepte pas plus de 100 correspondances par bloc."""

    rendu = _cmap({code: chr(code) for code in range(0x20, 0x20 + 250)}).decode("ascii")

    assert rendu.count("beginbfchar") == 3
    assert "100 beginbfchar" in rendu
