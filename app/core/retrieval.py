"""Recherche vectorielle scopée.

Le périmètre est filtré en SQL avant le tri par similarité : le service ne
cherche jamais en dehors de ce que la plateforme lui a transmis.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence

from app.core.embeddings import EmbeddingProvider
from app.db.repository import IndexRepository
from app.models.schemas import Scope

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Passage:
    chunk_id: str
    document_id: str
    title: str
    locator: str
    content: str
    language: str
    score: float


def _excerpt(content: str, maximum: int) -> str:
    text = content.strip()
    if len(text) <= maximum:
        return text
    # Couper sur un mot entier : un extrait ne doit jamais finir au milieu.
    return text[:maximum].rsplit(" ", 1)[0].rstrip() + "…"


class Retriever:
    def __init__(
        self,
        *,
        embeddings: EmbeddingProvider,
        repository: IndexRepository,
        minimum_score: float = 0.0,
    ) -> None:
        self._embeddings = embeddings
        self._repository = repository
        self._minimum_score = minimum_score

    async def search(
        self,
        *,
        query: str,
        scope: Scope,
        limit: int,
        max_excerpt_characters: int,
        document_ids: Optional[Sequence[str]] = None,
    ) -> List[Passage]:
        vectors = await self._embeddings.embed([query])
        rows = await self._repository.search(
            embedding=vectors[0],
            scope=scope,
            limit=limit,
            document_ids=document_ids,
        )

        passages = []
        for row in rows:
            # pgvector rend une distance cosinus dans [0, 2] ; on la ramène en
            # similarité pour que le score soit lisible.
            score = max(0.0, min(1.0, 1.0 - float(row["distance"])))
            if score < self._minimum_score:
                continue
            passages.append(
                Passage(
                    chunk_id=row["chunk_id"],
                    document_id=row["document_id"],
                    title=row["title"],
                    locator=row["locator"],
                    content=_excerpt(row["content"], max_excerpt_characters),
                    language=row["language"],
                    score=round(score, 6),
                )
            )
        return passages
