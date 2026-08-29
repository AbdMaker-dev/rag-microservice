"""Chaque voie reste une voie : aucune ne doit devenir le chemin par défaut."""

from __future__ import annotations

from app.core.pdf_profile import PdfProfile
from app.core.routing import Classification, PageShape, Route, page_shape


def test_un_document_sans_couche_texte_ne_se_lit_pas():
    vide = PdfProfile(pages=2, tagged=False, text_coverage=0.0, pages_needing_ocr=[1, 2])
    decision = Classification(Route.NEEDS_OCR, "latin", vide)

    assert not decision.readable


def test_les_filets_courts_ne_font_pas_un_tableau():
    """Un PDF de mathématiques est plein de traits : ce sont des fractions.

    Mesuré : aucune des 99 pages du programme national ne porte deux filets
    couvrant la moitié de la page, contre 23 sur 43 dans un document à vrais
    tableaux.
    """

    sans_filet: list = []
    avec_filets = [3, 7, 11]

    assert page_shape(5, sans_filet) is PageShape.UNRULED
    assert page_shape(7, avec_filets) is PageShape.RULED
    assert page_shape(5, avec_filets) is PageShape.UNRULED


def test_les_voies_sont_nommees_et_distinctes():
    """Le squelette doit exister avant les lecteurs qu'il accueillera."""

    assert {r.value for r in Route} == {"non-latin", "needs-ocr", "tagged", "untagged"}
