"""La base CAHIERS — lecture et écriture, toujours pour UN élève.

Séparée de `IndexRepository` comme les tables le sont du reste : la base
prof et la base cahiers ne se mélangent jamais (décision du 01/09/2026).

Chaque méthode exige `student_account_id`. Ce n'est pas une politesse de
signature : un passage de cahier sans propriétaire n'existe pas, et une
recherche qui « oublierait » l'élève ne compile pas ici — il n'y a aucun
chemin pour lire un cahier sans dire lequel.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import asyncpg

from app.db.repository import ChunkRow, to_vector_literal
from app.models.schemas import Scope


class NotebookRepository:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def replace_document(
        self,
        *,
        external_id: str,
        student_account_id: str,
        title: str,
        chapter: str,
        scope: Scope,
        embedding_model: str,
        embedding_dimension: int,
        characters: int,
        chunks: List[ChunkRow],
    ) -> str:
        """Indexer le cours d'un élève, en remplaçant sa version précédente.

        Idempotent comme la base prof : Awa peut corriger sa transcription
        et renvoyer, sans laisser d'anciens passages orphelins.
        """

        if not student_account_id:
            raise ValueError("un cours de cahier a toujours un propriétaire")

        async with self._pool.acquire() as connection:
            async with connection.transaction():
                document_id = await connection.fetchval(
                    """
                    INSERT INTO notebook_documents (
                        external_id, student_account_id, title, chapter,
                        country, subject, grade, track, curriculum_version,
                        language, embedding_model, embedding_dimension,
                        characters, chunk_count
                    )
                    VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14)
                    ON CONFLICT (external_id) DO UPDATE SET
                        student_account_id = EXCLUDED.student_account_id,
                        title = EXCLUDED.title,
                        chapter = EXCLUDED.chapter,
                        country = EXCLUDED.country,
                        subject = EXCLUDED.subject,
                        grade = EXCLUDED.grade,
                        track = EXCLUDED.track,
                        curriculum_version = EXCLUDED.curriculum_version,
                        language = EXCLUDED.language,
                        embedding_model = EXCLUDED.embedding_model,
                        embedding_dimension = EXCLUDED.embedding_dimension,
                        characters = EXCLUDED.characters,
                        chunk_count = EXCLUDED.chunk_count,
                        indexed_at = now()
                    RETURNING id
                    """,
                    external_id, student_account_id, title, chapter,
                    scope.country, scope.subject, scope.grade, scope.track,
                    scope.curriculum_version, scope.language,
                    embedding_model, embedding_dimension,
                    characters, len(chunks),
                )
                await connection.execute(
                    "DELETE FROM notebook_chunks WHERE document_id = $1",
                    document_id,
                )
                for chunk in chunks:
                    await connection.execute(
                        """
                        INSERT INTO notebook_chunks (
                            document_id, student_account_id, ordinal,
                            locator, content, token_count, embedding
                        )
                        VALUES ($1,$2,$3,$4,$5,$6,$7::vector)
                        """,
                        document_id, student_account_id, chunk.ordinal,
                        chunk.locator, chunk.content, chunk.token_count,
                        to_vector_literal(chunk.embedding),
                    )
                return str(document_id)

    async def search(
        self,
        *,
        student_account_id: str,
        embedding: Sequence[float],
        limit: int,
        subject: Optional[str] = None,
        chapter: Optional[str] = None,
    ) -> List[dict]:
        """Les passages du cahier de CET élève, et de personne d'autre.

        Le propriétaire est le premier filtre, appliqué en SQL avant le tri.
        Il n'est pas optionnel : sans lui, la requête ne peut pas s'écrire.
        """

        if not student_account_id:
            raise ValueError("une recherche de cahier exige son propriétaire")

        return [
            dict(row)
            for row in await self._pool.fetch(
                """
                SELECT
                    c.id::text    AS chunk_id,
                    d.external_id AS document_id,
                    d.title       AS title,
                    d.chapter     AS chapter,
                    c.locator     AS locator,
                    c.content     AS content,
                    d.language    AS language,
                    (c.embedding <=> $2::vector) AS distance
                FROM notebook_chunks AS c
                INNER JOIN notebook_documents AS d ON d.id = c.document_id
                WHERE c.student_account_id = $1
                  AND d.student_account_id = $1
                  AND ($3::text IS NULL OR d.subject = $3::text)
                  AND ($4::text IS NULL OR d.chapter = $4::text)
                ORDER BY c.embedding <=> $2::vector
                LIMIT $5
                """,
                student_account_id,
                to_vector_literal(embedding),
                subject,
                chapter,
                limit,
            )
        ]

    async def delete_document(
        self, *, student_account_id: str, external_id: str
    ) -> bool:
        """Awa supprime SON cours. Personne ne supprime celui d'un autre.

        Aucune suppression automatique n'existe ici : le scan EST son cours,
        il vit tant qu'elle le garde (décision d'Alioune, 07/09/2026).
        """

        if not student_account_id:
            raise ValueError("une suppression de cahier exige son propriétaire")

        deleted = await self._pool.fetchval(
            """
            DELETE FROM notebook_documents
            WHERE external_id = $1 AND student_account_id = $2
            RETURNING id
            """,
            external_id, student_account_id,
        )
        return deleted is not None

    async def list_documents(
        self, *, student_account_id: str, subject: Optional[str] = None
    ) -> List[dict]:
        """Le cahier d'Awa, du plus récent au plus ancien."""

        if not student_account_id:
            raise ValueError("lister un cahier exige son propriétaire")

        return [
            dict(row)
            for row in await self._pool.fetch(
                """
                SELECT external_id AS document_id, title, chapter, subject,
                       grade, track, characters, chunk_count, indexed_at
                FROM notebook_documents
                WHERE student_account_id = $1
                  AND ($2::text IS NULL OR subject = $2::text)
                ORDER BY indexed_at DESC
                """,
                student_account_id, subject,
            )
        ]
