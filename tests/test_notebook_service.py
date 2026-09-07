"""Le service du cahier : il CONSOMME la base prof, il ne l'alimente jamais."""

import asyncio

from app.core.notebook import NotebookService
from app.models.schemas import Scope


class _FakeEmbeddings:
    model = "bge-m3"

    def __init__(self) -> None:
        self.calls = 0

    async def embed(self, texts):
        self.calls += 1
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]


class _FakeDocuments:
    """La base PROF. Elle ne doit recevoir QUE des lectures."""

    def __init__(self, hits=None) -> None:
        self.hits = hits or []
        self.searches = 0

    async def search(self, **kwargs):
        self.searches += 1
        return list(self.hits)

    async def replace_document(self, **kwargs):  # pragma: no cover
        raise AssertionError("le cahier n'écrit JAMAIS dans la base des profs")


class _FakeNotebooks:
    def __init__(self) -> None:
        self.saved = []

    async def replace_document(self, **kwargs):
        self.saved.append(kwargs)
        return "doc-1"


class _Settings:
    chunk_max_tokens = 400
    chunk_overlap_tokens = 40
    chunk_min_tokens = 20


def _scope() -> Scope:
    return Scope(
        country="SN", subject="maths", grade="terminale", level="lycee",
        track="S2", curriculum_version="2006", language="fr",
    )


def _service(documents, notebooks=None):
    return NotebookService(
        embeddings=_FakeEmbeddings(),
        documents=documents,
        notebooks=notebooks or _FakeNotebooks(),
        settings=_Settings(),
    )


def test_une_preuve_du_cours_valide_corrige_et_cite_sa_source():
    """La correction n'est pas une reformulation : c'est le passage validé
    lui-même, et Awa voit d'où il vient."""

    documents = _FakeDocuments([
        {
            "content": "Une similitude directe conserve les angles orientés.",
            "title": "Similitudes directes",
            "locator": "§3",
            "distance": 0.05,
        }
    ])
    report = asyncio.run(
        _service(documents).repair(
            text="Une similitde dirccte conserve les angls orientés",
            scope=_scope(),
        )
    )
    assert report.corrected == 1
    assert report.segments[0]["status"] == "corrige"
    assert report.segments[0]["proof"]["title"] == "Similitudes directes"
    assert report.proven_from == ["Similitudes directes"]


def test_sans_couverture_validee_rien_n_est_invente():
    """Aucune preuve → le texte d'Awa reste le sien. C'est la règle qui
    interdit la formule plausible et fausse."""

    report = asyncio.run(
        _service(_FakeDocuments([])).repair(
            text="Le théorème de la médiane s'écrit MA² + MB² = ...",
            scope=_scope(),
        )
    )
    assert report.corrected == 0
    assert report.text.startswith("Le théorème de la médiane")


def test_les_embeddings_sont_calcules_en_UNE_passe():
    """L'appel coûteux ne doit pas se faire ligne à ligne : un cahier de
    trente lignes ferait trente allers-retours."""

    embeddings = _FakeEmbeddings()
    service = NotebookService(
        embeddings=embeddings,
        documents=_FakeDocuments([]),
        notebooks=_FakeNotebooks(),
        settings=_Settings(),
    )
    asyncio.run(service.repair(text="a\nb\nc\nd\ne", scope=_scope()))
    assert embeddings.calls == 1


def test_la_preuve_se_cherche_dans_les_roles_valides_seulement():
    """Le cahier d'un camarade qui a mal recopié ne prouve rien : on ne
    cherche que dans ce qu'un professeur ou le ministère a validé."""

    documents = _FakeDocuments([])
    asyncio.run(_service(documents).repair(text="une ligne", scope=_scope()))
    # trois rôles de preuve × une ligne
    assert documents.searches == 3


def test_indexer_range_dans_la_base_cahiers_avec_son_proprietaire():
    notebooks = _FakeNotebooks()
    result = asyncio.run(
        _service(_FakeDocuments([]), notebooks).index(
            external_id="c-1",
            student_account_id="awa",
            title="Similitudes",
            chapter="Similitudes directes",
            scope=_scope(),
            text="Une similitude directe conserve les angles orientés. " * 20,
        )
    )
    assert result["chunks"] >= 1
    assert notebooks.saved[0]["student_account_id"] == "awa"
    assert notebooks.saved[0]["chapter"] == "Similitudes directes"


def test_indexer_n_ecrit_jamais_dans_la_base_des_professeurs():
    """Garde-fou : `_FakeDocuments.replace_document` lève. Si un jour
    quelqu'un range un cahier du mauvais côté, ce test tombe."""

    documents = _FakeDocuments([])
    asyncio.run(
        _service(documents).index(
            external_id="c-1",
            student_account_id="awa",
            title="Similitudes",
            chapter="",
            scope=_scope(),
            text="Un texte de cahier assez long pour produire un passage. " * 20,
        )
    )
    assert documents.searches == 0  # indexer ne cherche même pas
