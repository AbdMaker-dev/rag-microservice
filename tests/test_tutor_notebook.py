"""Lawal sur le cahier de l'élève.

Deux exigences, et la seconde compte autant que la première : il doit
pouvoir répondre sur le cours qu'Awa a ajouté elle-même, ET dire d'où
vient chaque extrait. Ses propres notes peuvent être fausses ; le cours
validé par son professeur, non. Confondre les deux serait lui faire
réviser une erreur avec l'autorité de l'école.
"""

import asyncio

from app.core.retrieval import Passage
from app.core.tutor import Tutor, _citations
from app.models.schemas import Scope


class _FakeEmbeddings:
    async def embed(self, texts):
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]


class _FakeNotebooks:
    def __init__(self, rows=None) -> None:
        self.rows = rows or []
        self.calls = []

    async def search(self, **kwargs):
        self.calls.append(kwargs)
        return list(self.rows)


def _row(document_id: str, content: str) -> dict:
    return {
        "chunk_id": f"c-{document_id}",
        "document_id": document_id,
        "title": "Mon cours de similitudes",
        "chapter": "Similitudes directes",
        "locator": "§1",
        "content": content,
        "language": "fr",
        "distance": 0.1,
    }


def _tutor(notebooks) -> Tutor:
    return Tutor(
        llm=None,
        retriever=None,
        settings=None,
        notebooks=notebooks,
        embeddings=_FakeEmbeddings(),
    )


def test_la_recherche_du_cahier_passe_toujours_le_proprietaire():
    """Le dépôt refuse un propriétaire vide ; le tuteur ne doit jamais le
    mettre dans cette situation, ni chercher « pour tout le monde »."""

    notebooks = _FakeNotebooks([_row("doc-1", "Une similitude conserve les angles.")])
    passages = asyncio.run(
        _tutor(notebooks)._search_notebook(
            question="c'est quoi une similitude ?",
            student_account_id="awa",
            document_id="doc-1",
        )
    )
    assert notebooks.calls[0]["student_account_id"] == "awa"
    assert len(passages) == 1
    assert passages[0].title == "Mon cours de similitudes"


def test_un_passage_d_un_autre_cours_du_meme_eleve_est_ecarte():
    """Awa pose une question SUR un cours précis. Un autre de ses cours,
    même à elle, n'est pas le sujet — et mélanger deux chapitres brouille
    la réponse plus qu'il ne l'enrichit."""

    notebooks = _FakeNotebooks([_row("doc-1", "bon"), _row("doc-2", "autre chapitre")])
    passages = asyncio.run(
        _tutor(notebooks)._search_notebook(
            question="?", student_account_id="awa", document_id="doc-1"
        )
    )
    assert [p.document_id for p in passages] == ["doc-1"]


def test_sans_cahier_le_tuteur_continue_de_fonctionner():
    """Un élève qui n'a jamais rien scanné doit avoir un tuteur qui marche :
    le cahier est un ajout, pas une dépendance."""

    tutor = Tutor(
        llm=None, retriever=None, settings=None, notebooks=None, embeddings=None
    )
    assert tutor._notebooks is None
    passages = asyncio.run(
        _tutor(_FakeNotebooks([]))._search_notebook(
            question="?", student_account_id="awa", document_id="doc-1"
        )
    )
    assert passages == []


def test_les_citations_distinguent_le_cahier_du_contenu_valide():
    """C'est l'exigence d'honnêteté : Awa doit voir que la réponse s'appuie
    sur SES notes, pas sur le cours de son professeur."""

    mine = Passage("c-1", "doc-1", "Mon cours", "§1", "…", "fr", 0.9)
    validated = Passage("c-2", "doc-2", "Cours du prof", "§4", "…", "fr", 0.9)
    citations = _citations([mine, validated], notebook_ids={"c-1"})
    assert citations[0]["source"] == "cahier"
    assert citations[1]["source"] == "valide"


def test_sans_cahier_toutes_les_citations_restent_validees():
    """Le champ existe toujours : le front n'a pas à deviner son absence."""

    passage = Passage("c-1", "doc-1", "Cours du prof", "§1", "…", "fr", 0.9)
    assert _citations([passage])[0]["source"] == "valide"
