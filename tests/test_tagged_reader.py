"""Le lecteur d'arbre de structure, sur des arbres simulés.

Le cas qui a motivé ces tests : un PDF imprimé par Chrome (13/09/2026)
enveloppe chaque changement de police — un mot en gras, un exposant — dans
un « NonStruct » à part. Le lecteur rendait chacun comme un bloc, et
« Le module de z est la distance OM » sortait en cinq lignes.
"""

from app.core.tagged_reader import _gather


class Node(dict):
    """Un élément de structure tel que pypdf le rend : un dict qui sait se
    résoudre lui-même."""

    def get_object(self):
        return self


def elem(kind, *kids):
    return Node({"/S": "/" + kind, "/K": Node({}) if not kids else _kids(kids)})


class Kids(list):
    def get_object(self):
        return self


def _kids(kids):
    return Kids(kids)


def rendre(tree, texts):
    out = []
    _gather(tree, 1, {}, texts, out, set())
    return out


def test_les_fragments_en_ligne_rejoignent_leur_paragraphe():
    # <p>Le <b>module</b> de z est la distance OM.</p> tel que Chrome le tague.
    p = elem("P", elem("NonStruct", 0), elem("NonStruct", 1), elem("NonStruct", 2))
    texts = {(1, 0): "Le", (1, 1): "module", (1, 2): "de z est la distance OM."}

    assert rendre(elem("Document", p), texts) == [
        (1, "Le module de z est la distance OM.")
    ]


def test_deux_paragraphes_restent_deux_blocs():
    tree = elem(
        "Document",
        elem("P", elem("Span", 0), 1),
        elem("P", 2),
    )
    texts = {(1, 0): "Définition.", (1, 1): "Soit z non nul.", (1, 2): "Exemple."}

    assert rendre(tree, texts) == [
        (1, "Définition. Soit z non nul."),
        (1, "Exemple."),
    ]


def test_une_enveloppe_qui_contient_des_blocs_est_parcourue_bloc_par_bloc():
    # « NonStruct » couvre la moitié de certains documents : du texte direct,
    # puis des paragraphes, puis encore du texte. Chaque morceau garde sa
    # place ; rien ne se retrouve aplati ni déplacé avant les paragraphes.
    tree = elem(
        "Document",
        elem("NonStruct", 0, elem("P", 1), 2),
    )
    texts = {(1, 0): "avant", (1, 1): "paragraphe", (1, 2): "après"}

    assert rendre(tree, texts) == [
        (1, "avant"),
        (1, "paragraphe"),
        (1, "après"),
    ]


def test_l_etiquette_d_un_item_rejoint_son_corps():
    # <ol><li>Écrire z sous forme trigonométrique.</li></ol>
    tree = elem(
        "Document",
        elem("L", elem("LI", elem("Lbl", 0), elem("LBody", elem("NonStruct", 1)))),
    )
    texts = {(1, 0): "1.", (1, 1): "Écrire z sous forme trigonométrique."}

    assert rendre(tree, texts) == [(1, "1. Écrire z sous forme trigonométrique.")]


def test_un_corps_d_item_en_paragraphes_garde_ses_paragraphes():
    tree = elem(
        "Document",
        elem("L", elem("LI", elem("Lbl", 0), elem("LBody", elem("P", 1), elem("P", 2)))),
    )
    texts = {(1, 0): "1.", (1, 1): "Premier alinéa.", (1, 2): "Second alinéa."}

    assert rendre(tree, texts) == [
        (1, "1."),
        (1, "Premier alinéa."),
        (1, "Second alinéa."),
    ]


def test_un_titre_reste_un_titre():
    tree = elem("Document", elem("H2", elem("NonStruct", 0)), elem("P", 1))
    texts = {(1, 0): "1. Affixe d'un point", (1, 1): "Le plan est muni d'un repère."}

    assert rendre(tree, texts) == [
        (1, "### 1. Affixe d'un point"),
        (1, "Le plan est muni d'un repère."),
    ]


def test_la_jointure_ne_met_pas_d_espace_apres_l_apostrophe_ni_avant_la_virgule():
    # <p>On dit que z est l’<b>affixe</b> du point (<b>u</b>, <b>v</b>).</p>
    p = elem("P", 0, elem("Span", 1), 2, elem("Span", 3), 4, elem("Span", 5), 6)
    texts = {(1, 0): "On dit que z est l’", (1, 1): "affixe", (1, 2): "du point (",
             (1, 3): "u", (1, 4): ",", (1, 5): "v", (1, 6): ")."}

    assert rendre(elem("Document", p), texts) == [
        (1, "On dit que z est l’affixe du point (u, v).")
    ]


def test_aucun_marqueur_n_est_perdu_dans_un_type_inconnu():
    # Un producteur peut inventer n'importe quel type : le texte qu'il porte
    # finit dans le paragraphe qui l'entoure, jamais à la poubelle.
    tree = elem("Document", elem("P", 0, elem("Machin", elem("Truc", 1)), 2))
    texts = {(1, 0): "a", (1, 1): "b", (1, 2): "c"}

    assert rendre(tree, texts) == [(1, "a b c")]
