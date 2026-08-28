"""Chunking is the highest-leverage piece here: a bad split makes every
downstream answer worse, and it is testable without any backend."""

from __future__ import annotations

import pytest

from app.core.chunking import chunk_document, split_blocks

# Deterministic counter so the tests do not depend on tiktoken being installed.
def words(text: str) -> int:
    return max(1, len(text.split()))


def test_headings_become_the_locator_path():
    document = "# Mathématiques\n\n## Algèbre\n\nUne équation du premier degré.\n"
    chunks = chunk_document(document, max_tokens=64, overlap_tokens=0, count_tokens=words)
    assert len(chunks) == 1
    assert chunks[0].locator.startswith("Mathématiques > Algèbre")


def test_blocks_track_nested_headings():
    document = "# A\n\ntexte a\n\n## B\n\ntexte b\n\n# C\n\ntexte c"
    blocks = split_blocks(document)
    assert [tuple(block.heading_path) for block in blocks] == [("A",), ("A", "B"), ("C",)]


def test_window_is_respected():
    document = "\n\n".join(f"Paragraphe numéro {index} avec du contenu." for index in range(40))
    chunks = chunk_document(document, max_tokens=30, overlap_tokens=0, count_tokens=words)
    assert len(chunks) > 1
    assert all(chunk.token_count <= 30 for chunk in chunks)


def test_overlap_carries_context_forward():
    document = "\n\n".join(f"Bloc {index} " + "mot " * 20 for index in range(6))
    with_overlap = chunk_document(document, max_tokens=40, overlap_tokens=10, count_tokens=words)
    without = chunk_document(document, max_tokens=40, overlap_tokens=0, count_tokens=words)
    assert len(with_overlap) >= len(without)


def test_a_long_sentence_is_never_split_mid_word():
    sentence = " ".join(f"mot{index}" for index in range(200))
    chunks = chunk_document(sentence, max_tokens=20, overlap_tokens=0, count_tokens=words)
    assert len(chunks) > 1
    for chunk in chunks:
        for token in chunk.content.split():
            assert token.startswith("mot")


def test_sentence_boundaries_are_preferred():
    document = ". ".join(f"Phrase numéro {index} sur le sujet" for index in range(20)) + "."
    chunks = chunk_document(document, max_tokens=25, overlap_tokens=0, count_tokens=words)
    # No chunk should start with a lowercase fragment of a cut sentence.
    assert all(chunk.content.lstrip()[0].isupper() for chunk in chunks)


def test_a_trailing_fragment_is_folded_back():
    document = "\n\n".join(["Bloc " + "mot " * 30 for _ in range(3)] + ["fin"])
    chunks = chunk_document(
        document, max_tokens=40, overlap_tokens=0, min_tokens=10, count_tokens=words
    )
    assert chunks[-1].token_count >= 10


def test_empty_document_produces_no_chunk():
    assert chunk_document("   \n\n  ", count_tokens=words) == []


def test_overlap_must_stay_below_the_window():
    with pytest.raises(ValueError):
        chunk_document("texte", max_tokens=10, overlap_tokens=10, count_tokens=words)


def test_ordinals_are_contiguous():
    document = "\n\n".join(f"Bloc {index} " + "mot " * 15 for index in range(10))
    chunks = chunk_document(document, max_tokens=40, overlap_tokens=5, count_tokens=words)
    assert [chunk.ordinal for chunk in chunks] == list(range(len(chunks)))
