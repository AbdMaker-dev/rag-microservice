#!/usr/bin/env python3
"""Read-only report on the vector index.

Replaces the `reset_db.py` of the original sketch: this service owns no
database, so it has nothing to reset. What it can usefully do is verify the
index it reads from — in particular that a single embedding configuration is
in use per scope, since pgvector cannot compare vectors of different
dimensions and a mixed scope makes retrieval fail at query time.

    python scripts/inspect_index.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import asyncpg  # noqa: E402

from app.config import get_settings  # noqa: E402

_REPORT = """
SELECT
    d.country_id::text                          AS country,
    d.subject_id::text                          AS subject,
    d.language_code                             AS language,
    c.embedding_provider_configuration_id::text AS configuration,
    count(*)                                    AS chunks,
    min(vector_dims(c.embedding))               AS min_dimension,
    max(vector_dims(c.embedding))               AS max_dimension
FROM document_chunks AS c
INNER JOIN pedagogical_documents AS d ON d.id = c.document_id
GROUP BY 1, 2, 3, 4
ORDER BY chunks DESC
"""


async def main() -> int:
    settings = get_settings()
    connection = await asyncpg.connect(settings.database_url)
    try:
        rows = await connection.fetch(_REPORT)
    finally:
        await connection.close()

    if not rows:
        print("aucun chunk indexé")
        return 0

    print(f"{'pays':<38} {'matière':<38} {'lg':<4} {'chunks':>7} {'dim':>6}")
    for row in rows:
        print(
            f"{row['country']:<38} {row['subject']:<38} {row['language']:<4} "
            f"{row['chunks']:>7} {row['max_dimension']:>6}"
        )
        if row["min_dimension"] != row["max_dimension"]:
            print("    ⚠ dimensions mixtes dans ce périmètre : les requêtes échoueront")

    scopes = {}
    for row in rows:
        key = (row["country"], row["subject"], row["language"])
        scopes.setdefault(key, set()).add(row["configuration"])
    mixed = [key for key, configurations in scopes.items() if len(configurations) > 1]
    if mixed:
        print(f"\n⚠ {len(mixed)} périmètre(s) avec plusieurs configurations d'embedding")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
