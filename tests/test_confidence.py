"""Situer les doutes, pas seulement les compter."""

from __future__ import annotations

from app.core.confidence import inspect_section


def _kinds(text):
    return {issue.kind for issue in inspect_section(text)[1]}


def test_un_texte_propre_ne_declenche_rien():
    propre = (
        "Le produit scalaire est un outil pour démontrer des propriétés, "
        "calculer des distances et des mesures d'angles, et résoudre des "
        "problèmes d'orthogonalité dans le plan comme dans l'espace."
    )

    confidence, issues = inspect_section(propre)

    assert confidence == 1.0
    assert issues == []


def test_une_ocr_hachee_est_reperee_et_situee():
    """« Mo o x » : trois fragments à la suite qui ne sont pas des mots."""

    texte = "DECLARATION Mo o x CONSTITUTION DE PERSONNE MORALE ETRANGERE ici."

    confidence, issues = inspect_section(texte)
    haches = [issue for issue in issues if issue.kind == "OCR_NOISE"]

    assert haches, "trois fragments consécutifs doivent être signalés"
    assert texte[haches[0].start:haches[0].end] == "Mo o x"
    assert confidence < 1.0


def test_les_mots_francais_courts_ne_sont_pas_du_hachis():
    """« I et si » est du français ordinaire, pas de l'OCR ratée.

    Sans la liste des mots courts, ce détecteur signalait 1 326 passages sur
    un seul document et la confiance médiane tombait à 0,47 : tout ressortait
    en rouge, donc plus rien n'était lisible.
    """

    assert "OCR_NOISE" not in _kinds(
        "Si f est dérivable sur I et si sa dérivée est positive sur I alors f croît."
    )


def test_les_mathematiques_ordinaires_ne_sont_pas_du_hachis():
    """« ax + b » et « f ( x ) » sont corrects : chiffres et ponctuation."""

    assert "OCR_NOISE" not in _kinds("La fonction f ( x ) = ax + b est affine.")


def test_les_barres_de_tableau_ne_sont_pas_du_hachis():
    assert "OCR_NOISE" not in _kinds("| Contenus |  |  | Commentaires |")


def test_un_reste_d_encodage_est_signale_a_sa_position():
    texte = "Les compŽtences exigibles de la classe de seconde sont listées ici."

    _, issues = inspect_section(texte)
    encodage = [issue for issue in issues if issue.kind == "ENCODING"]

    assert encodage
    assert texte[encodage[0].start] == "Ž"


def test_une_puce_d_enumeration_n_est_pas_une_parenthese_ouverte():
    """« 2) Barycentre » ne doit pas passer pour une formule déséquilibrée."""

    assert "FORMULA" not in _kinds(
        "2) Barycentre de trois points pondérés, avec λ réel et somme non nulle."
    )


def test_une_section_trop_maigre_est_signalee():
    confidence, issues = inspect_section("Suite.")

    assert confidence < 1.0
    assert [issue.kind for issue in issues] == ["THIN"]


def test_les_defauts_sont_rendus_dans_l_ordre_du_texte():
    """L'interface surligne de haut en bas : l'ordre doit suivre le texte."""

    _, issues = inspect_section(
        "DŽbut du texte avec assez de mots pour ne pas être maigre, "
        "puis Mo o x plus loin dans la même section de contrôle."
    )

    assert [issue.start for issue in issues] == sorted(issue.start for issue in issues)


def test_les_majuscules_accentuees_francaises_ne_sont_pas_du_mojibake():
    """« ALGÈBRE » et « FENÊTRE » sont du français, pas du texte abîmé.

    Trouvé sur une vraie réponse de l'API : le titre de chapitre « ALGÈBRE »
    était signalé comme reste d'encodage et faisait chuter la confiance de sa
    section.
    """

    assert "ENCODING" not in _kinds(
        "ALGÈBRE. Composition des applications et factorisation des polynômes "
        "par la méthode de Hörner, avec étude du signe sur un intervalle."
    )
