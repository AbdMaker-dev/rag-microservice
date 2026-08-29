"""Un arbre de structure ne se croit pas sur parole."""

from __future__ import annotations

from app.core.struct_tree import TreeHealth


def test_un_arbre_absent_n_est_pas_exploitable():
    assert not TreeHealth().usable


def test_un_arbre_plat_n_est_pas_exploitable():
    """Tout en <P>, une seule strate : il décrit une liste, pas un document."""

    plat = TreeHealth(present=True, coverage=1.0, depth=1, elements=200,
                      types={"P": 200})

    assert not plat.usable


def test_un_arbre_sans_type_porteur_de_sens_n_est_pas_exploitable():
    """« NonStruct » et « Span » décrivent la mise en page, pas la structure."""

    creux = TreeHealth(present=True, coverage=1.0, depth=5, elements=300,
                       types={"NonStruct": 264, "Span": 36})

    assert not creux.usable


def test_un_arbre_qui_couvre_mal_est_pire_que_pas_d_arbre():
    """S'y fier ferait perdre silencieusement ce qu'il ne rattache pas."""

    partiel = TreeHealth(present=True, coverage=0.30, depth=7, elements=900,
                         types={"P": 500, "Table": 12}, weak_pages=[2, 3, 4])

    assert not partiel.usable
    assert partiel.has_tables  # il a bien des tableaux, et reste inutilisable


def test_un_arbre_riche_et_couvrant_est_exploitable():
    """Mesuré sur quatre documents : couverture 100 %, profondeur 6 à 9."""

    sain = TreeHealth(present=True, coverage=1.0, depth=9, elements=3476,
                      types={"P": 1385, "TD": 440, "TR": 227, "Table": 43})

    assert sain.usable
    assert sain.has_tables
