"""Lecture et écriture de l'index."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import asyncpg

from app.models.schemas import Scope


def to_vector_literal(embedding: Sequence[float]) -> str:
    """Rendre un vecteur au format que pgvector sait lire."""

    if not embedding:
        raise ValueError("vecteur vide")
    return "[" + ",".join(repr(float(value)) for value in embedding) + "]"


@dataclass(frozen=True)
class ChunkRow:
    ordinal: int
    locator: str
    content: str
    token_count: int
    embedding: Sequence[float]


class IndexRepository:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def replace_document(
        self,
        *,
        external_id: str,
        title: str,
        source_reference: str,
        scope: Scope,
        embedding_model: str,
        embedding_dimension: int,
        content: str,
        characters: int,
        chunks: List[ChunkRow],
    ) -> str:
        """Indexer un document, en remplaçant intégralement sa version précédente.

        Réindexer est donc idempotent : la plateforme peut rejouer l'appel sans
        créer de doublons ni laisser d'anciens passages orphelins.
        """

        async with self._pool.acquire() as connection:
            async with connection.transaction():
                document_id = await connection.fetchval(
                    """
                    INSERT INTO documents (
                        external_id, title, source_reference,
                        country, subject, grade, curriculum_version, language,
                        embedding_model, embedding_dimension,
                        content, characters, chunk_count, indexed_at
                    )
                    VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13, now())
                    ON CONFLICT (external_id) DO UPDATE SET
                        title = EXCLUDED.title,
                        source_reference = EXCLUDED.source_reference,
                        country = EXCLUDED.country,
                        subject = EXCLUDED.subject,
                        grade = EXCLUDED.grade,
                        curriculum_version = EXCLUDED.curriculum_version,
                        language = EXCLUDED.language,
                        embedding_model = EXCLUDED.embedding_model,
                        embedding_dimension = EXCLUDED.embedding_dimension,
                        content = EXCLUDED.content,
                        characters = EXCLUDED.characters,
                        chunk_count = EXCLUDED.chunk_count,
                        indexed_at = now()
                    RETURNING id
                    """,
                    external_id, title, source_reference,
                    scope.country, scope.subject, scope.grade,
                    scope.curriculum_version, scope.language,
                    embedding_model, embedding_dimension,
                    content, characters, len(chunks),
                )

                # Remplacement complet : plus simple et plus sûr qu'une
                # réconciliation passage par passage.
                await connection.execute(
                    "DELETE FROM chunks WHERE document_id = $1", document_id
                )
                await connection.executemany(
                    """
                    INSERT INTO chunks
                        (document_id, ordinal, locator, content, token_count, embedding)
                    VALUES ($1, $2, $3, $4, $5, $6::vector)
                    """,
                    [
                        (
                            document_id,
                            chunk.ordinal,
                            chunk.locator,
                            chunk.content,
                            chunk.token_count,
                            to_vector_literal(chunk.embedding),
                        )
                        for chunk in chunks
                    ],
                )
                return str(document_id)

    async def delete_document(self, external_id: str) -> bool:
        """Retirer un document de l'index.

        Appelé quand la plateforme dépublie un contenu : il doit cesser d'être
        trouvable immédiatement.
        """

        async with self._pool.acquire() as connection:
            result = await connection.execute(
                "DELETE FROM documents WHERE external_id = $1", external_id
            )
            return result.endswith(" 1")

    async def list_documents(
        self, *, scope: Scope, limit: int, offset: int
    ) -> List[dict]:
        """Lister les documents d'un périmètre, sans leur contenu.

        L'écran de création affiche des titres et des tailles, pas des pavés
        de texte : renvoyer le contenu ici serait lourd pour rien.
        """

        rows = await self._pool.fetch(
            """
            SELECT external_id, title, source_reference, characters,
                   chunk_count, embedding_model, indexed_at
            FROM documents
            WHERE country = $1 AND subject = $2 AND grade = $3
              AND curriculum_version = $4 AND language = $5
            ORDER BY indexed_at DESC
            LIMIT $6 OFFSET $7
            """,
            scope.country, scope.subject, scope.grade,
            scope.curriculum_version, scope.language, limit, offset,
        )
        return [dict(row) for row in rows]

    async def count_documents(self, *, scope: Scope) -> int:
        return await self._pool.fetchval(
            """
            SELECT count(*) FROM documents
            WHERE country = $1 AND subject = $2 AND grade = $3
              AND curriculum_version = $4 AND language = $5
            """,
            scope.country, scope.subject, scope.grade,
            scope.curriculum_version, scope.language,
        )

    async def get_document(self, external_id: str) -> Optional[dict]:
        """Un document avec son texte complet, pour relecture par le prof."""

        row = await self._pool.fetchrow(
            """
            SELECT external_id, title, source_reference, content, characters,
                   chunk_count, embedding_model, indexed_at,
                   country, subject, grade, curriculum_version, language
            FROM documents WHERE external_id = $1
            """,
            external_id,
        )
        return dict(row) if row else None

    async def search(
        self,
        *,
        embedding: Sequence[float],
        scope: Scope,
        limit: int,
        document_ids: Optional[Sequence[str]] = None,
    ) -> List[dict]:
        """Passages les plus proches, dans le périmètre demandé.

        Le filtre de périmètre est appliqué en SQL, avant le tri : le service
        ne cherche jamais en dehors de ce que la plateforme lui a autorisé.

        `document_ids` restreint encore la recherche à certains documents —
        utile quand un prof génère un cours à partir d'une sélection précise.
        """

        return [
            dict(row)
            for row in await self._pool.fetch(
                """
                SELECT
                    c.id::text        AS chunk_id,
                    d.external_id     AS document_id,
                    d.title           AS title,
                    c.locator         AS locator,
                    c.content         AS content,
                    d.language        AS language,
                    (c.embedding <=> $1::vector) AS distance
                FROM chunks AS c
                INNER JOIN documents AS d ON d.id = c.document_id
                WHERE d.country = $2
                  AND d.subject = $3
                  AND d.grade = $4
                  AND d.curriculum_version = $5
                  AND d.language = $6
                  AND ($7::text[] IS NULL OR d.external_id = ANY($7::text[]))
                ORDER BY c.embedding <=> $1::vector
                LIMIT $8
                """,
                to_vector_literal(embedding),
                scope.country, scope.subject, scope.grade,
                scope.curriculum_version, scope.language,
                list(document_ids) if document_ids else None,
                limit,
            )
        ]
