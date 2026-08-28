"""Connexion à la base de l'index."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import asyncpg

from app.config import Settings

logger = logging.getLogger(__name__)

MIGRATIONS_DIRECTORY = Path(__file__).resolve().parents[2] / "migrations"


class Database:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._pool: Optional[asyncpg.Pool] = None

    @property
    def pool(self) -> asyncpg.Pool:
        if self._pool is None:
            raise RuntimeError("la base n'est pas connectée")
        return self._pool

    async def connect(self) -> None:
        if self._pool is not None:
            return
        self._pool = await asyncpg.create_pool(
            dsn=self._settings.database_url,
            min_size=self._settings.database_pool_min,
            max_size=self._settings.database_pool_max,
            server_settings={
                "statement_timeout": str(self._settings.database_statement_timeout_ms),
                "application_name": self._settings.service_name,
            },
        )

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def healthy(self) -> bool:
        if self._pool is None:
            return False
        try:
            async with self._pool.acquire() as connection:
                return await connection.fetchval("SELECT 1") == 1
        except Exception:  # noqa: BLE001
            return False

    async def migrate(self) -> None:
        """Appliquer les fichiers SQL dans l'ordre, une seule fois chacun.

        Volontairement minimal : pas d'Alembic tant que le schéma tient en
        quelques tables. Chaque fichier est enregistré une fois appliqué.
        """

        async with self.pool.acquire() as connection:
            await connection.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    filename   text PRIMARY KEY,
                    applied_at timestamptz NOT NULL DEFAULT now()
                )
                """
            )
            applied = {
                row["filename"]
                for row in await connection.fetch("SELECT filename FROM schema_migrations")
            }

            for path in sorted(MIGRATIONS_DIRECTORY.glob("*.sql")):
                if path.name in applied:
                    continue
                async with connection.transaction():
                    await connection.execute(path.read_text(encoding="utf-8"))
                    await connection.execute(
                        "INSERT INTO schema_migrations (filename) VALUES ($1)", path.name
                    )
                logger.info("migration appliquée", extra={"migration": path.name})
