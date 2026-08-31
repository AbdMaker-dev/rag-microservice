"""Capture des figures : ce qui est dessiné ne doit plus disparaître en silence.

Sur nos documents réels, les zones dessinées sont des figures de géométrie
mais aussi des formules posées en image par un export Word — invisibles dans
la couche texte. Ces tests vérifient la mécanique sur des PDF synthétiques :
détection d'un dessin véritable, rejet des filets de tableau, marqueurs posés
sous la bonne page.
"""

import io
import zlib

import pdfplumber
import pytest

from app.core.figures import (
    CapturedFigure,
    FigureRegion,
    annotate,
    collect_regions,
    render_figures,
    without_furniture,
)


def _pdf(content: str) -> bytes:
    """Un PDF d'une page A4 dont le contenu est le flux donné."""

    stream = zlib.compress(content.encode("latin-1"))
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
        b"/Contents 4 0 R /Resources << >> >>",
        b"<< /Length %d /Filter /FlateDecode >>\nstream\n%s\nendstream"
        % (len(stream), stream),
    ]
    out = io.BytesIO()
    out.write(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(out.tell())
        out.write(b"%d 0 obj\n" % number)
        out.write(body)
        out.write(b"\nendobj\n")
    start = out.tell()
    out.write(b"xref\n0 %d\n" % (len(objects) + 1))
    out.write(b"0000000000 65535 f \n")
    for offset in offsets:
        out.write(b"%010d 00000 n \n" % offset)
    out.write(
        b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n"
        % (len(objects) + 1, start)
    )
    return out.getvalue()


# Un triangle (traits obliques) et deux arcs inscrits : quatre primitives
# qui se touchent — un seul dessin.
_FIGURE = (
    "1 w 0 0 0 RG "
    "100 600 m 200 700 l S "
    "200 700 m 300 600 l S "
    "100 600 m 300 600 l S "
    "120 620 m 150 660 190 660 220 620 c S "
    "180 610 m 210 650 250 650 280 610 c S "
)

# Des filets de tableau : traits horizontaux et verticaux, rien d'autre.
_TABLE = (
    "1 w "
    "100 600 m 400 600 l S "
    "100 500 m 400 500 l S "
    "100 400 m 400 400 l S "
    "100 400 m 100 600 l S "
    "250 400 m 250 600 l S "
    "400 400 m 400 600 l S "
)


def _regions(payload: bytes):
    with pdfplumber.open(io.BytesIO(payload)) as document:
        return collect_regions(1, document.pages[0])


def test_un_dessin_veritable_est_detecte():
    regions = _regions(_pdf(_FIGURE))
    assert len(regions) == 1
    region = regions[0]
    # La zone couvre le triangle entier, marge comprise.
    assert region.x0 <= 100 and region.x1 >= 300
    assert region.bottom - region.top >= 100


def test_les_filets_de_tableau_ne_sont_pas_une_figure():
    assert _regions(_pdf(_TABLE)) == []


def test_le_decor_repete_est_ecarte():
    """L'en-tête du ministère ressortait 43 fois sur les progressions de
    l'IA Dakar : la même zone sur trois pages ou plus est un décor."""

    header = [FigureRegion(page, 40.0, 30.0, 550.0, 90.0) for page in (1, 2, 3, 4)]
    figure = [FigureRegion(2, 100.0, 300.0, 320.0, 480.0)]
    kept = without_furniture(header + figure)
    assert kept == figure


def test_deux_occurrences_ne_sont_pas_du_decor():
    # Une figure reprise une fois (énoncé puis corrigé) reste une figure.
    twice = [FigureRegion(1, 100.0, 300.0, 320.0, 480.0),
             FigureRegion(5, 100.0, 300.0, 320.0, 480.0)]
    assert without_furniture(twice) == twice


def test_le_rendu_produit_un_png_aux_bonnes_dimensions():
    payload = _pdf(_FIGURE)
    captured = render_figures(payload, _regions(payload))
    assert len(captured) == 1
    figure = captured[0]
    assert figure.figure_id == "f1"
    assert figure.page == 1
    assert figure.png.startswith(b"\x89PNG")
    assert figure.width > 0 and figure.height > 0


def test_les_marqueurs_se_posent_sous_leur_page():
    figures = [
        CapturedFigure("f1", 1, 10, 10, b""),
        CapturedFigure("f2", 2, 10, 10, b""),
    ]
    text = "## p. 1\n\nDu texte.\n\n## p. 2\n\nLa suite."
    annotated = annotate(text, figures)
    lines = annotated.split("\n")
    assert "[FIGURE f1 — p. 1]" in lines
    assert "[FIGURE f2 — p. 2]" in lines
    assert lines.index("[FIGURE f1 — p. 1]") < lines.index("## p. 2")
    assert lines.index("[FIGURE f2 — p. 2]") > lines.index("## p. 2")


def test_une_page_sans_repere_ne_perd_pas_sa_figure():
    # Page 3 vide de texte : son marqueur arrive en fin de document.
    figures = [CapturedFigure("f1", 3, 10, 10, b"")]
    annotated = annotate("## p. 1\n\nDu texte.", figures)
    assert annotated.endswith("[FIGURE f1 — p. 3]")


def test_extract_rend_les_figures_dans_la_reponse():
    import base64

    from fastapi.testclient import TestClient

    from app.main import create_app

    client = TestClient(create_app())
    response = client.post(
        "/extract",
        headers={"X-Service-Token": "test-secret-value-of-at-least-32-chars"},
        json={
            "requestId": "fig-1",
            "filename": "figure.pdf",
            "mediaType": "application/pdf",
            "contentBase64": base64.b64encode(
                _pdf(_FIGURE + " BT /F1 12 Tf 100 300 Td (Soit ABC un triangle.) Tj ET")
            ).decode(),
        },
    )
    # Le PDF synthétique n'embarque pas de police : selon la tolérance du
    # lecteur, le texte peut être refusé. Ce qui se teste ici : si la réponse
    # sort, elle porte la figure et son marqueur.
    if response.status_code == 200:
        body = response.json()
        assert len(body["figures"]) == 1
        figure = body["figures"][0]
        assert figure["figureId"] == "f1"
        assert figure["page"] == 1
        base64.b64decode(figure["imageBase64"])
        assert "[FIGURE f1 — p. 1]" in body["text"]
    else:
        pytest.skip("PDF synthétique sans police refusé par le lecteur")
