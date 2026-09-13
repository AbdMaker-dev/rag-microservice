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

# Un tableau : des filets ET son contenu. Le contenu n'est pas un détail de
# mise en scène — c'est LUI qui distingue un tableau d'un schéma en boîtes
# (13/09/2026). Une grille vide et un diagramme sont géométriquement
# identiques ; voir `test_une_grille_vide_est_capturee`.
_TABLE = (
    "1 w "
    "100 600 m 400 600 l S "
    "100 500 m 400 500 l S "
    "100 400 m 400 400 l S "
    "100 400 m 100 600 l S "
    "250 400 m 250 600 l S "
    "400 400 m 400 600 l S "
    "BT /F1 9 Tf 110 570 Td (Periode une colonne une) Tj ET "
    "BT /F1 9 Tf 260 570 Td (Periode une colonne deux) Tj ET "
    "BT /F1 9 Tf 110 470 Td (Periode deux colonne une) Tj ET "
    "BT /F1 9 Tf 260 470 Td (Periode deux colonne deux) Tj ET "
    "BT /F1 9 Tf 110 430 Td (Periode trois colonne une) Tj ET "
    "BT /F1 9 Tf 260 430 Td (Periode trois colonne deux) Tj ET "
)


def _regions(payload: bytes):
    with pdfplumber.open(io.BytesIO(payload)) as document:
        page = document.pages[0]
        # Les mots comme dans la vraie lecture : ils séparent une figure
        # étiquetée d'un tableau rempli.
        return collect_regions(1, page, page.extract_words())


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


# ─────────── ce que la règle d'origine laissait passer (13/09/2026) ───────────
#
# Mesuré sur le serveur avant correction : une étoile à cinq traits obliques
# était capturée, un triangle à deux côtés obliques ne l'était pas, un
# rectangle avec des axes non plus. Seuls les obliques, les courbes et les
# images comptaient. Or ce sont les figures les plus ordinaires d'un support
# scolaire — et elles disparaissaient sans un mot.

_TRIANGLE_DEUX_OBLIQUES = """2 w
120 560 m 320 560 l S
120 560 m 220 700 l S
220 700 m 320 560 l S
220 700 m 220 560 l S
"""

_CADRE_ET_AXES = """2 w
120 560 m 320 560 l S
320 560 m 320 700 l S
320 700 m 120 700 l S
120 700 m 120 560 l S
120 540 m 340 540 l S
120 540 m 120 720 l S
"""


def _tableau_borde_rempli() -> str:
    """Quatre horizontales, quatre verticales, et du texte dans chaque case."""

    traits = [f"120 {y} m 420 {y} l S" for y in (700, 660, 620, 580)]
    traits += [f"{x} 580 m {x} 700 l S" for x in (120, 220, 320, 420)]
    mots = [
        f"BT /F1 9 Tf {x} {y} Td (Periode {r} colonne {c}) Tj ET"
        for r, y in enumerate((675, 635, 595), start=1)
        for c, x in enumerate((130, 230, 330), start=1)
    ]
    return "2 w\n" + "\n".join(traits + mots) + "\n"


def test_un_triangle_a_deux_cotes_obliques_est_capture():
    """Le cas qui manquait : deux obliges seulement, et pourtant un triangle."""

    assert len(_regions(_pdf(_TRIANGLE_DEUX_OBLIQUES))) == 1


def test_un_cadre_et_des_axes_sont_captures():
    """Zéro oblique : rectangle, axes, tableau de variations, schéma en boîtes."""

    assert len(_regions(_pdf(_CADRE_ET_AXES))) == 1


def test_un_tableau_borde_et_rempli_n_est_toujours_pas_une_figure():
    """Le garde-fou qui rend la nouvelle règle tenable.

    Un tableau bordé porte lui aussi de l'horizontal ET du vertical. Ce qui
    l'en distingue se mesure : une figure est ÉTIQUETÉE (A, O, x, 30°), un
    tableau est REMPLI.
    """

    assert _regions(_pdf(_tableau_borde_rempli())) == []


def test_une_grille_vide_est_capturee():
    """Décision assumée du 13/09/2026, pas un défaut.

    Une grille SANS contenu et un schéma en boîtes sont géométriquement
    identiques : mêmes horizontales, mêmes verticales, aucun mot. Rien ne
    permet de les séparer, et il faut donc choisir lequel des deux risques
    on prend.

    On capture. La règle produit est écrite depuis le 31/08 — « il peut
    rester des captures inutiles, c'est le prof qui tranche » — et les deux
    erreurs ne coûtent pas le même prix : une capture en trop se jette d'un
    clic, une figure manquée disparaît sans que personne ne l'apprenne.
    """

    grille = (
        "1 w "
        "100 600 m 400 600 l S "
        "100 500 m 400 500 l S "
        "100 400 m 400 400 l S "
        "100 400 m 100 600 l S "
        "250 400 m 250 600 l S "
        "400 400 m 400 600 l S "
    )
    assert len(_regions(_pdf(grille))) == 1
