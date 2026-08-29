from __future__ import annotations

import pytest

from app.core.extraction import UnsupportedMediaType, load, normalise, to_sections


def test_les_lignes_vides_survivent():
    # Elles servent de frontière de bloc en aval.
    assert normalise("a\n\n\n\n\nb") == "a\n\nb"


def test_retours_chariot_et_caracteres_de_controle_retires():
    assert normalise("a\r\nb\x00c") == "a\nbc"


def test_type_non_supporte_refuse():
    with pytest.raises(UnsupportedMediaType):
        load(b"...", "application/zip")


def test_markdown_traite_comme_du_texte():
    assert load("# Titre\n\ncorps".encode("utf-8"), "text/markdown") == "# Titre\n\ncorps"


def test_sections_portent_le_chemin_des_titres():
    sections = to_sections("# Maths\n\n## Algèbre\n\nune équation\n\n# Français\n\nun texte")
    assert [s["locator"] for s in sections] == ["Maths > Algèbre", "Français"]


def test_sections_numerotees_sans_titre():
    sections = to_sections("premier bloc\n\ndeuxième bloc")
    assert [s["position"] for s in sections] == [0, 1]
    assert sections[0]["locator"] == "§1"


def test_texte_vide_ne_produit_aucune_section():
    assert to_sections("   \n\n  ") == []


# ----------------------------------------------------------------- qualité

from app.core.quality import assess, corrupted_pages  # noqa: E402


def test_texte_francais_correct_bien_note():
    q = assess("Les compétences exigibles portent sur la géométrie plane.")
    assert q.score > 0.95


def test_texte_a_encodage_casse_mal_note():
    q = assess("Les compŽtences portent sur la gŽomŽtrie et les probl(cid:143)mes.")
    assert q.score < 0.8
    assert q.cid_markers == 1


def test_formules_mathematiques_ne_penalisent_pas():
    # Les variables d'une lettre et les symboles usuels sont attendus.
    q = assess("Soit f dérivable sur I ; si f'(x) ≥ 0 alors f croît. π ≈ 3,14")
    assert q.score > 0.85


def test_tableau_markdown_bien_note():
    q = assess("| Semaine 2 | Activité Numérique | Les nombres décimaux |")
    assert q.score > 0.9


def test_texte_vide_note_a_zero():
    assert assess("").score == 0.0


def test_pages_corrompues_reperees():
    pages = [
        "Une page parfaitement lisible en français correct.",
        "Une page avec (cid:143) des marqueurs (cid:12) d'échec.",
        "Encore une page saine et bien écrite en français.",
    ]
    assert corrupted_pages(pages) == [2]


def test_page_vide_ignoree():
    assert corrupted_pages(["texte correct en français", "   ", ""]) == []


# ------------------------------------------------------------------ index

from app.db.repository import to_vector_literal  # noqa: E402


def test_vecteur_au_format_pgvector():
    assert to_vector_literal([1.0, -0.5]) == "[1.0,-0.5]"


def test_vecteur_vide_refuse():
    with pytest.raises(ValueError):
        to_vector_literal([])


# ------------------------------------------------- contrat de la réponse

from app.models.schemas import ExtractionQuality, ExtractResponse  # noqa: E402


def test_reponse_extraction_se_construit():
    """Le modèle refuse tout champ inconnu ; un nom mal orthographié dans la
    route ne se voyait qu'au premier appel réel."""

    reponse = ExtractResponse(
        request_id="r1",
        filename="doc.pdf",
        media_type="application/pdf",
        text="Un texte correct en français.",
        characters=29,
        sections=[],
        quality=ExtractionQuality(
            score=0.95, word_plausibility=1.0, cid_markers=0, words_repaired=12
        ),
    )
    assert reponse.quality.words_repaired == 12
    assert reponse.model_dump(by_alias=True)["quality"]["wordsRepaired"] == 12


def test_la_reponse_porte_l_analyse_du_document_sans_casser_le_contrat():
    """`analysis` est additive : un appelant qui l'ignore voit la même réponse."""

    from app.models.schemas import (
        DocumentAnalysis,
        ExtractionQuality,
        ExtractResponse,
        FontDiagnosis,
    )

    reponse = ExtractResponse(
        request_id="r-1",
        filename="programme.pdf",
        media_type="application/pdf",
        text="Compétences exigibles",
        characters=21,
        sections=[],
        quality=ExtractionQuality(
            score=0.99, word_plausibility=0.99, cid_markers=0, words_repaired=0
        ),
        analysis=DocumentAnalysis(
            tagged=False,
            text_coverage=0.149,
            pages_needing_ocr=[3],
            fonts=[FontDiagnosis(font="Times-Roman", table="mac_roman",
                                 confidence=1.0, samples=1436)],
        ),
    )

    rendu = reponse.model_dump(by_alias=True)
    assert rendu["analysis"]["pagesNeedingOcr"] == [3]
    assert rendu["analysis"]["fonts"][0]["table"] == "mac_roman"
    assert rendu["analysis"]["textCoverage"] == 0.149
    # Les clés historiques restent en place et au même endroit.
    assert rendu["contractVersion"] == "1.0"
    assert rendu["quality"]["score"] == 0.99


def test_le_type_reel_du_fichier_prime_sur_le_type_declare():
    """Un navigateur devine l'extension et se trompe régulièrement.

    Suivre la signature ferme deux problèmes : le dépôt légitime mal étiqueté,
    et le fichier arbitraire annoncé comme PDF qui entrerait dans un analyseur
    qui ne l'attend pas.
    """

    from app.core.extraction import sniff

    assert sniff(b"%PDF-1.4\n%...") == "application/pdf"
    assert sniff(b"   %PDF-1.7") == "application/pdf"
    assert sniff(b"PK\x03\x04\x14\x00\x06\x00").endswith("wordprocessingml.document")
    # Pas de signature reconnue : on ne refuse pas pour autant, un fichier
    # texte n'en a pas.
    assert sniff(b"Bonjour, ceci est du texte.") is None
    assert sniff(b"") is None
