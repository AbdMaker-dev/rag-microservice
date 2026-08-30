"""Recoller les mots coupés, sur preuve — jamais sur intuition."""

from __future__ import annotations

from app.core.lexicon import build, mend


def test_les_huit_cas_reels_se_recollent():
    """Les coupures relevées sur les trois documents du corpus."""

    document = (
        "Le théorème et la définition des triangles : leurs relatives sont "
        "connues, INTRODUCTION GENERALE comprise.\n"
        "En déc embre puis en mar s, au Mois d’ octobre : le théor ème, la "
        "définiti on, les relati ves, un trian gle, INTRODUCTION GENERAL E.\n"
        "Le triangle est une figure générale."
    )

    mended = mend(document)

    for whole in ("décembre", "mars", "d’octobre", "théorème", "définition",
                  "relatives", "triangle", "GENERALE"):
        assert whole in mended, whole


def test_deux_vrais_mots_ne_fusionnent_jamais():
    """« rapport à » et « la porte » sont des suites de mots légitimes."""

    document = (
        "Par rapport à la porte : le rapport est clair, la porte est là, "
        "et à chaque rapport la porte s'ouvre sur un vrai laporte rapporta."
    )

    mended = mend(document)

    assert "rapport à" in mended
    assert "la porte" in mended


def test_l_echec_d_une_paire_n_emporte_pas_la_suivante():
    """« En déc embre » : l'échec de « En déc » ne doit pas consommer « déc ».

    C'était le défaut du premier jet en `re.sub` : la substitution avançait
    au-delà de la paire examinée même sans la fusionner, et la vraie coupure
    n'était jamais essayée.
    """

    document = "Nous sommes en décembre. En déc embre, il neige souvent ici."

    assert "En décembre, il neige" in mend(document)


def test_sans_preuve_interne_on_ne_touche_pas():
    """« bri colage » : rien dans le document n'atteste « bricolage »."""

    document = "Un bri colage est visible dans le texte de cette page-là."

    assert "bri colage" in mend(document)
