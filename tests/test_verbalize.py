"""La verbalisation : ce que Piper reçoit ne contient plus que des mots.

Chaque règle vient d'un cas réel de nos documents (produit scalaire,
limites, PHARES). Un symbole mal lu à l'oral induit l'élève en erreur —
ces tests sont la liste de ce qu'on garantit.
"""

from app.core.verbalize import verbalize


def test_les_marqueurs_ne_se_lisent_pas():
    spoken = verbalize("## p. 12\n\nLe produit scalaire.\n\n[FIGURE f3 — p. 12]")
    assert "p. 12" not in spoken
    assert "##" not in spoken
    assert "FIGURE" not in spoken
    assert "Une figure accompagne ce passage : regardez l'écran." in spoken


def test_les_maths_se_disent_en_francais():
    spoken = verbalize("x² + 3x ≥ 0 et √2 ∈ ℝ")
    assert "x au carré" in spoken
    assert "plus 3x" in spoken
    assert "est supérieur ou égal à 0" in spoken
    assert "racine carrée de 2" in spoken
    assert "appartient à" in spoken
    assert "²" not in spoken and "√" not in spoken and "≥" not in spoken


def test_les_limites_et_l_infini():
    spoken = verbalize("lim x → +∞")
    assert "tend vers plus l'infini" in spoken
    assert "∞" not in spoken


def test_les_vecteurs_et_le_produit_scalaire():
    spoken = verbalize("AB⃗ · AC⃗ = 0")
    assert "le vecteur AB" in spoken
    assert "scalaire" in spoken
    assert "égale" in spoken


def test_la_chimie_se_dit_en_chiffres():
    spoken = verbalize("H₂O et CO₂ ; g = 9,8 m/s²")
    assert "H2O" in spoken
    assert "CO2" in spoken
    assert "mètres par seconde au carré" in spoken
    assert "₂" not in spoken


def test_le_gras_et_les_titres_ne_s_entendent_pas():
    spoken = verbalize("# Chapitre 2\n\nLe **produit scalaire** est *essentiel*")
    assert "#" not in spoken and "*" not in spoken
    assert "Chapitre 2." in spoken
    assert "produit scalaire" in spoken


def test_un_tableau_s_annonce_sans_se_lire():
    spoken = verbalize("Voici les valeurs :\n\n| x | 1 | 2 |\n| f(x) | 3 | 5 |")
    assert "|" not in spoken
    assert "Un tableau accompagne ce passage" in spoken


def test_chaque_paragraphe_finit_en_phrase():
    # Sans point final, Piper enchaîne sans pause : un paragraphe devient
    # une phrase, jamais deux points de suite.
    spoken = verbalize("Définition\n\nUn vecteur a une norme")
    assert "Définition. Un vecteur a une norme." == spoken


def test_un_repli_de_ligne_ne_coupe_pas_la_phrase():
    # « Leur produit\nscalaire » : le retour à la ligne est un repli
    # d'affichage — entendu « Leur produit. scalaire » avant la règle.
    spoken = verbalize("Leur produit\nscalaire est nul")
    assert spoken == "Leur produit scalaire est nul."


def test_un_symbole_inconnu_ne_fait_pas_echouer():
    # La règle absente s'ajoutera ; en attendant le texte sort quand même.
    spoken = verbalize("l'opérateur ⊕ reste tel quel")
    assert "⊕" in spoken
