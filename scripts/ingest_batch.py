#!/usr/bin/env python3
"""Chunk local documents and report the result, without any backend.

Purpose: tune the chunking window on real curriculum material before wiring an
embedding model. It calls no API and writes nothing.

    python scripts/ingest_batch.py data/test_documents --max-tokens 384
"""

from __future__ import annotations

import argparse
import mimetypes
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.chunking import chunk_document, default_token_counter  # noqa: E402
from app.utils.loaders import UnsupportedMediaType, load  # noqa: E402

_EXTENSION_MEDIA_TYPES = {
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--overlap-tokens", type=int, default=64)
    parser.add_argument("--show", type=int, default=0, help="print the first N chunks")
    arguments = parser.parse_args()

    if not arguments.directory.is_dir():
        print(f"not a directory: {arguments.directory}", file=sys.stderr)
        return 2

    count_tokens = default_token_counter()
    total_chunks = 0
    documents = 0

    for path in sorted(arguments.directory.rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue
        media_type = _EXTENSION_MEDIA_TYPES.get(
            path.suffix.lower()
        ) or mimetypes.guess_type(path.name)[0]
        if media_type is None:
            continue

        try:
            text = load(path.read_bytes(), media_type)
        except UnsupportedMediaType:
            continue

        chunks = chunk_document(
            text,
            max_tokens=arguments.max_tokens,
            overlap_tokens=arguments.overlap_tokens,
            count_tokens=count_tokens,
        )
        documents += 1
        total_chunks += len(chunks)
        sizes = [chunk.token_count for chunk in chunks] or [0]
        print(
            f"{path.name}: {len(chunks)} chunks · "
            f"médiane {statistics.median(sizes):.0f} tk · max {max(sizes)} tk"
        )
        for chunk in chunks[: arguments.show]:
            print(f"    [{chunk.ordinal}] {chunk.locator}")
            print(f"        {chunk.content[:160].replace(chr(10), ' ')}…")

    print(f"\n{documents} document(s), {total_chunks} chunk(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
