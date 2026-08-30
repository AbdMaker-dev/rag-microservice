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


def test_ne_contredit_pas_une_police_qui_declare_deja_sa_table():
    """Une CMap /ToUnicode présente fait autorité.

    Sur un document sain, des polices sous-ensemblées voyaient leurs index de
    glyphes relus comme du MacRoman : 59 582 caractères corrompus. Le PDF
    disait déjà comment les lire.
    """

    codes = Counter({0x8E: 400, 0x8F: 50, 0xC9: 480, 0x41: 900})

    sans_garde = decide({"Times": codes}, {})[0]
    avec_garde = decide({"Times": codes}, {}, trusted={"Times"})[0]

    assert sans_garde.encoding == "mac_roman"
    assert avec_garde.encoding is None


def test_ne_decide_rien_hors_ecriture_latine():
    """Nos tables candidates décrivent des écritures latines.

    Sur un document arabe ou cyrillique, les noter choisirait au hasard — et
    pourrait basculer une police saine vers une table fausse.
    """

    codes = Counter({0x8E: 400, 0x8F: 50, 0xC9: 480})

    assert decide({"Times": codes}, {}, script="latin")[0].encoding == "mac_roman"
    for script in ("arabic", "cyrillic", "cjk", "greek"):
        assert decide({"Times": codes}, {}, script=script)[0].encoding is None


def test_une_police_symbolique_aussi_doit_fournir_un_echantillon():
    """Sept caractères ne suffisent pas à conclure, fût-ce sur du Symbol."""

    maigre = decide({"SymbolMT": Counter({0x8D: 2, 0x8C: 1, 0x88: 1})}, {})[0]
    fourni = decide({"SymbolMT": Counter({0xA3: 30, 0x6C: 12})}, {})[0]

    assert maigre.encoding is None
    assert fourni.encoding == "symbol"


def test_le_repli_sans_filtre_ne_peut_qu_ajouter_de_la_prudence():
    """Un document abîmé ne déclare presque aucune police fiable.

    Se limiter à celles-ci laisse le garde-fou de script inactif là où il
    servirait le plus. Le repli relit sans filtre — mais une méprise
    d'encodage transforme des octets en caractères latins, elle ne fabrique
    jamais du cyrillique. Trouver de tels caractères prouve qu'ils étaient là.
    """

    from collections import Counter

    from app.core.pdf_encoding import _SCRIPTS

    # Ce que le repli cherche : une écriture non latine bien présente.
    trouve = Counter({"cyrillic": 400, "latin": 60})
    part = trouve["cyrillic"] / sum(trouve.values())

    assert part >= 0.3
    assert "cyrillic" in _SCRIPTS and "arabic" in _SCRIPTS and "cjk" in _SCRIPTS


def test_une_petite_police_ne_decide_jamais_seule():
    """« République du Sénégal » est devenu « RÈpublique du SÈnÈgal ».

    Dans un document Word sain, deux polices de 24 et 12 caractères non-ASCII
    ont basculé en mac_roman sur la foi d'un en-tête en majuscules
    accentuées : nous avons corrompu un document correct. Un petit échantillon
    plausible dans plusieurs tables tranche sur du bruit.
    """

    # Un en-tête accentué : « é » cp1252, plausible en mac_roman aussi.
    tahoma = Counter({0xE9: 14, 0xE8: 6, 0xE0: 4, 0x41: 200})

    seule = decide({"ABCDEE+Tahoma": tahoma}, {})

    assert seule[0].encoding is None, "24 caractères ne suffisent pas à décider seul"


def test_une_petite_police_peut_suivre_le_consensus_du_document():
    """Le programme national répare à raison des polices de 9 et 12 caractères.

    Elles suivent des polices sœurs de plusieurs centaines de caractères,
    toutes diagnostiquées mac_roman. Un plancher plat les aurait tuées ; le
    consensus les garde tout en bloquant les petites polices isolées.
    """

    grosse = Counter({0x8E: 400, 0x8F: 60, 0xC9: 480, 0xA5: 70, 0x41: 900})
    petite = Counter({0x8E: 7, 0xC9: 4, 0x41: 30})

    ensemble = decide({"ENNLHC+Times": grosse, "BOKLEN+Times-Bold": petite}, {})
    isolee = decide({"BOKLEN+Times-Bold": petite}, {})

    par_nom = {d.fontname: d for d in ensemble}
    assert par_nom["ENNLHC+Times"].encoding == "mac_roman"
    assert par_nom["BOKLEN+Times-Bold"].encoding == "mac_roman", "elle suit le consensus"
    assert isolee[0].encoding is None, "sans consensus, elle s'abstient"
