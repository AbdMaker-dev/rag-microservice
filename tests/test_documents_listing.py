"""Le listing des documents filtre par niveau, série et rôle — quand ils sont donnés.

L'écran d'administration des programmes liste par périmètre fin (niveau,
série) ; un champ vide ne doit PAS filtrer, sinon les documents indexés
avant l'introduction de ces colonnes deviendraient invisibles.
"""

import asyncio

from app.db.repository import IndexRepository
from app.models.schemas import Scope


class _RecordingPool:
    def __init__(self):
        self.queries = []

    async def fetch(self, sql, *args):
        self.queries.append((sql, args))
        return []

    async def fetchval(self, sql, *args):
        self.queries.append((sql, args))
        return 0


def _scope(**extra) -> Scope:
    return Scope(
        country="SN", subject="mathematiques", grade="terminale",
        curriculum_version="2024", language="fr", **extra,
    )


def test_listing_transmet_niveau_serie_et_role():
    pool = _RecordingPool()
    repo = IndexRepository(pool)
    asyncio.run(repo.list_documents(
        scope=_scope(level="secondaire", track="S2"),
        limit=50, offset=0, role="programme-officiel",
    ))
    sql, args = pool.queries[0]
    assert "level = $6" in sql and "track = $7" in sql and "role = $8" in sql
    assert args[5] == "secondaire"
    assert args[6] == "S2"
    assert args[7] == "programme-officiel"


def test_champs_vides_ne_filtrent_pas():
    # Le SQL neutralise le filtre quand la valeur est vide ou absente :
    # la garde est écrite dans la requête elle-même ($n = '' OR ...).
    pool = _RecordingPool()
    repo = IndexRepository(pool)
    asyncio.run(repo.list_documents(scope=_scope(), limit=50, offset=0))
    sql, args = pool.queries[0]
    assert "($6::text = '' OR level = $6)" in sql
    assert "($7::text = '' OR track = $7)" in sql
    assert "($8::text IS NULL OR role = $8)" in sql
    assert args[5] == "" and args[6] == "" and args[7] is None


def test_count_applique_les_memes_filtres():
    pool = _RecordingPool()
    repo = IndexRepository(pool)
    asyncio.run(repo.count_documents(
        scope=_scope(level="secondaire", track="S2"), role="support-cours"
    ))
    sql, args = pool.queries[0]
    assert "level = $6" in sql and "track = $7" in sql and "role = $8" in sql
    assert args[5:8] == ("secondaire", "S2", "support-cours")
