"""La base cahiers est cloisonnée — par le schéma, pas par la discipline.

Ce qu'on protège ici n'est pas une préférence d'architecture : le cahier
d'un élève porte son écriture, ses erreurs, parfois son nom. Qu'un passage
remonte dans le cours d'un autre, ou dans la génération d'un professeur,
serait une fuite — pas un bug d'affichage.
"""

import asyncio
import inspect

import pytest

from app.db.notebook_repository import NotebookRepository
from app.db.repository import ChunkRow
from app.models.schemas import Scope


class _RecordingPool:
    """Un faux pool qui retient les requêtes au lieu de les exécuter."""

    def __init__(self) -> None:
        self.queries: list = []

    async def fetch(self, sql, *args):
        self.queries.append((sql, args))
        return []

    async def fetchval(self, sql, *args):
        self.queries.append((sql, args))
        return None


def _scope() -> Scope:
    return Scope(
        country="SN",
        subject="maths",
        grade="terminale",
        level="lycee",
        track="S2",
        curriculum_version="2006",
        language="fr",
    )


def test_la_recherche_exige_un_proprietaire():
    """Sans élève, pas de recherche. Le cloisonnement ne peut pas s'oublier
    en écrivant une requête trop vite : il n'y a aucun chemin sans lui."""

    repo = NotebookRepository(_RecordingPool())
    with pytest.raises(ValueError):
        asyncio.run(repo.search(student_account_id="", embedding=[0.0] * 4, limit=5))


def test_l_eleve_est_le_premier_filtre_de_la_recherche():
    """Le propriétaire filtre en SQL, AVANT le tri par similarité — jamais
    après, sinon un passage d'un autre élève aurait déjà été lu."""

    pool = _RecordingPool()
    repo = NotebookRepository(pool)
    asyncio.run(repo.search(student_account_id="awa", embedding=[0.0] * 4, limit=5))
    sql, args = pool.queries[0]
    assert "c.student_account_id = $1" in sql
    assert "d.student_account_id = $1" in sql
    assert args[0] == "awa"


def test_la_recherche_ne_touche_jamais_les_tables_du_prof():
    """Deux bases, jamais mélangées : aucune requête de cahier ne doit citer
    `documents` ou `chunks`, sous peine de rendre un contenu partagé."""

    pool = _RecordingPool()
    repo = NotebookRepository(pool)
    asyncio.run(repo.search(student_account_id="awa", embedding=[0.0] * 4, limit=5))
    sql, _ = pool.queries[0]
    assert "notebook_chunks" in sql and "notebook_documents" in sql
    assert "FROM chunks" not in sql
    assert "JOIN documents" not in sql


def test_la_suppression_exige_le_proprietaire_et_le_verifie():
    """Awa supprime SON cours. Connaître l'identifiant d'un autre cours ne
    suffit pas : le propriétaire est dans la clause WHERE."""

    pool = _RecordingPool()
    repo = NotebookRepository(pool)
    asyncio.run(repo.delete_document(student_account_id="awa", external_id="c-1"))
    sql, args = pool.queries[0]
    assert "student_account_id = $2" in sql
    assert args == ("c-1", "awa")

    with pytest.raises(ValueError):
        asyncio.run(repo.delete_document(student_account_id="", external_id="c-1"))


def test_lister_un_cahier_exige_son_proprietaire():
    repo = NotebookRepository(_RecordingPool())
    with pytest.raises(ValueError):
        asyncio.run(repo.list_documents(student_account_id=""))


def test_indexer_sans_proprietaire_est_refuse():
    """Un passage de cahier sans propriétaire ne peut pas exister — la base
    l'interdit (NOT NULL), le dépôt le dit plus tôt et plus clairement."""

    repo = NotebookRepository(_RecordingPool())
    with pytest.raises(ValueError):
        asyncio.run(
            repo.replace_document(
                external_id="c-1",
                student_account_id="",
                title="Similitudes",
                chapter="Similitudes directes",
                scope=_scope(),
                embedding_model="bge-m3",
                embedding_dimension=1024,
                characters=10,
                chunks=[],
            )
        )


def test_le_proprietaire_est_recopie_sur_chaque_passage():
    """La colonne est dupliquée sur les passages exprès : une recherche
    cloisonnée ne doit pas dépendre d'une jointure qu'on pourrait oublier."""

    source = inspect.getsource(NotebookRepository.replace_document)
    assert "INSERT INTO notebook_chunks" in source
    assert "student_account_id" in source


def test_la_migration_cree_bien_deux_tables_a_part():
    """Garde-fou de schéma : si un jour quelqu'un remplace les tables par une
    colonne `role` sur la base prof, ce test tombe — et c'est le but."""

    sql = open("migrations/005_notebook.sql", encoding="utf-8").read()
    assert "CREATE TABLE IF NOT EXISTS notebook_documents" in sql
    assert "CREATE TABLE IF NOT EXISTS notebook_chunks" in sql
    assert "student_account_id  text NOT NULL" in sql
    # Le cahier ne s'ajoute pas à la base prof.
    assert "ALTER TABLE documents" not in sql
