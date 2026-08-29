"""Lire les colonnes d'un tableau que rien ne trace."""

from __future__ import annotations

from app.core.columns import Metrics, furniture, measure, render, visual_lines


def _word(text, x0, top, width=20.0, height=10.0):
    return {"text": text, "x0": x0, "x1": x0 + width,
            "top": top, "bottom": top + height}


def test_les_seuils_sont_des_rapports_pas_des_points():
    """Une constante en points décrit une seule mise en page.

    Mesuré sur le programme national : hauteur 10,8 pt, espace 2,7 pt,
    interligne 12,5 pt. Les rapports y donnent 2,7 / 4,1 / 7,5 pt.
    """

    petit = Metrics(line_height=10.8, space_width=2.7, line_gap=12.5)
    grand = Metrics(line_height=21.6, space_width=5.4, line_gap=25.0)

    assert round(petit.merge, 1) == 2.7
    assert round(petit.gutter_min, 1) == 4.1
    assert round(petit.row_break, 1) == 7.5
    # Doublez le corps du texte, les seuils suivent.
    assert grand.merge == petit.merge * 2
    assert grand.gutter_min == petit.gutter_min * 2


def test_les_grandeurs_se_lisent_sur_le_document():
    mots = [_word("un", 0, 100), _word("deux", 30, 100), _word("trois", 0, 120)]

    grandeurs = measure([mots])

    assert grandeurs.line_height == 10.0
    assert grandeurs.space_width == 10.0


def test_les_mots_proches_forment_une_ligne_visuelle():
    m = Metrics(line_height=10.0)
    mots = [_word("a", 0, 100), _word("b", 40, 101), _word("c", 0, 130)]

    lignes = visual_lines(mots, m)

    assert len(lignes) == 2
    assert [w["text"] for w in lignes[0]] == ["a", "b"]


def test_un_habillage_se_reconnait_a_son_numero_de_page():
    """La position ne discrimine pas : dans le programme national l'en-tête du
    tableau est à 5 % de la hauteur et le pied de page à 75 %. C'est le
    numéro qui trahit le titre courant.
    """

    pages = [
        [[_word(f"{n}", 0, 700), _word("Programmes", 20, 700),
          _word("de", 60, 700), _word("mathematiques", 90, 700)]]
        for n in range(1, 9)
    ]
    entetes = [[[_word("Contenus", 0, 40), _word("Commentaires", 60, 40),
                 _word("Competences", 200, 40)]] for _ in range(8)]

    habillage = furniture(pages)
    entete = furniture(entetes)

    assert habillage, "un pied de page numéroté doit être reconnu"
    assert not entete, "un en-tête de tableau sans chiffre n'est pas de l'habillage"


def test_une_ligne_pleine_largeur_reste_de_la_prose():
    """La découper aux frontières de colonnes la hacherait."""

    m = Metrics(line_height=10.0, space_width=2.5, line_gap=12.0)
    lignes = [[_word("phrase", 0, 100, width=500)]]

    rendu = render(lignes, [200.0], 500.0, m)

    assert not rendu.startswith("|")
    assert "phrase" in rendu


def test_un_mot_au_bord_reste_dans_sa_colonne():
    """L'affectation se fait par recouvrement, jamais par le centre du mot."""

    m = Metrics(line_height=10.0, space_width=2.5, line_gap=12.0)
    # Le mot déborde la frontière, mais l'essentiel est à gauche.
    lignes = [[_word("gauche", 150, 100, width=60), _word("droite", 260, 100)]]

    rendu = render(lignes, [205.0], 400.0, m)

    assert rendu.startswith("| gauche | droite |")
