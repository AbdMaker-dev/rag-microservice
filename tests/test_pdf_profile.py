"""Un document se regarde avant de se lire."""

from __future__ import annotations

from app.core.pdf_profile import _IMAGE_CEILING, _TEXT_FLOOR, PdfProfile


def test_une_page_de_titre_ne_demande_pas_l_ocr():
    """Peu de texte ne suffit pas à conclure.

    La page 4 du programme national ne porte qu'un titre : 0,8 % de texte et
    aucune image. La signaler à l'OCR serait un faux positif — c'est ce que
    fait un test qui regarde le texte seul.
    """

    texte, image = 0.008, 0.0

    assert texte < _TEXT_FLOOR
    assert not (texte < _TEXT_FLOOR and image > _IMAGE_CEILING)


def test_une_page_scannee_demande_l_ocr():
    """Aucun texte et toute la page couverte d'image : c'est un scan."""

    texte, image = 0.0, 1.0

    assert texte < _TEXT_FLOOR and image > _IMAGE_CEILING


def test_une_page_ordinaire_ne_declenche_rien():
    """Mesuré entre 11 % et 17 % sur six documents d'origines différentes."""

    for texte in (0.109, 0.147, 0.171):
        assert texte > _TEXT_FLOOR


def test_le_profil_expose_le_besoin_d_ocr():
    sain = PdfProfile(pages=10, tagged=True, text_coverage=0.15)
    scanne = PdfProfile(pages=2, tagged=False, text_coverage=0.0, pages_needing_ocr=[1, 2])

    assert not sain.needs_ocr
    assert scanne.needs_ocr
    assert scanne.pages_needing_ocr == [1, 2]
