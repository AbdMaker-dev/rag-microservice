"""L'embedding a son propre délai, distinct de celui de l'inférence.

Mesuré le 13/09/2026 sur le serveur, pendant qu'une section se rédigeait :
38 s pour vectoriser une phrase. Trois essais à 30 s, et la publication
d'un cours échouait avec « service d'indexation indisponible ».
"""

import asyncio

from app.config import Settings
from app.core.embeddings import OllamaEmbeddingProvider


class FakeResponse:
    def __init__(self, n):
        self._n = n

    def raise_for_status(self):
        return None

    def json(self):
        return {"embeddings": [[0.1, 0.2]] * self._n}


class FakeClient:
    def __init__(self):
        self.timeouts = []

    async def post(self, url, *, json, timeout):
        self.timeouts.append(timeout)
        return FakeResponse(len(json["input"]))


def test_l_embedding_attend_plus_longtemps_que_l_inference():
    settings = Settings(inference_timeout_s=30.0, embedding_timeout_s=120.0)
    client = FakeClient()

    vectors = asyncio.run(
        OllamaEmbeddingProvider(settings, client).embed(["un", "deux"])
    )

    assert len(vectors) == 2
    assert client.timeouts == [120.0]


def test_le_delai_par_defaut_couvre_trois_fois_la_mesure_sous_charge():
    assert Settings().embedding_timeout_s >= 3 * 38
