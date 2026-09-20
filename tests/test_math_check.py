"""La vérification des calculs par SymPy (note d'évaluation du 16/09/2026).

Deux exigences, testées à égalité : attraper une erreur CERTAINE, et ne
JAMAIS signaler une réponse juste — un faux signalement renverrait en
correction ce qui était bon.
"""

import pytest

from app.core.math_check import check

NOTE = (r"On a \( 2e^{i\pi/3} = 2\left(\frac{1}{2} + i\frac{\sqrt{3}}{2}\right) = 1 + i\sqrt{3} \). "
        "Les coordonnées du nouveau point sont donc (1, 3).")


def test_l_erreur_de_la_note_est_attrapee_avec_la_valeur_exacte():
    findings = check(NOTE)
    assert len(findings) == 1
    # Rendu en LaTeX : l'écran du prof affiche la formule, pas du texte brut.
    assert "\\sqrt{3}" in findings[0].message and "pas \\( (1, 3) \\)" in findings[0].message


def test_une_egalite_chiffree_fausse_est_attrapee():
    findings = check(r"\[ \sqrt{9 + 16} = \sqrt{24} = 5 \]")
    assert findings and "4.899" in findings[0].message
    assert check(r"\[ u_1 = 2 \cdot 3 = 7 \]")


@pytest.mark.parametrize("juste", [
    r"On a \( 2e^{i\pi/3} = 1 + i\sqrt{3} \). Les coordonnées du point sont \( (1 ; \sqrt{3}) \).",
    r"Le point d'affixe \( 1 + i\sqrt{3} \) a pour coordonnées \( (1, \sqrt{3}) \).",
    r"Le point A d'affixe \( 2 - i \) a pour coordonnées (2 ; -1).",
    r"\[ |3 + 4i| = \sqrt{3^2 + 4^2} = \sqrt{9 + 16} = \sqrt{25} = 5 \]",
    r"\[ \omega = \frac{3-2i}{1-(1+i)} = 2+3i \]",
    r"\[ i^2 = \cos \pi + i \sin \pi = -1 \]",
    r"\[ c^2 = 3^2 + 4^2 = 25 \]",
    r"\[ u_1 = 2 \cdot 3 = 6, \quad u_2 = 2 \cdot 6 = 12 \]",
])
def test_une_reponse_juste_n_est_jamais_signalee(juste):
    assert check(juste) == []


@pytest.mark.parametrize("illisible", [
    r"\( z' = az + b \) et \( \omega = \frac{b}{1-a} \)",   # variables
    r"\( \sqrt{2} \approx 1{,}41 \) et \( a \neq 1 \)",      # pas des égalités
    r"\( \int_0^1 x \, dx = 3 \)",                           # notation inconnue
    r"\( 10^{99999} = 1 \)",                                 # calcul démesuré
    "Pas de formule du tout.",
])
def test_ce_qu_on_ne_sait_pas_lire_ne_signale_rien(illisible):
    assert check(illisible) == []


def test_deux_nombres_dans_la_phrase_on_ne_devine_pas_lequel():
    texte = r"Avec \( 1 + i\sqrt{3} \) et \( 2 + i \), les coordonnées sont (1, 3)."
    assert check(texte) == []
