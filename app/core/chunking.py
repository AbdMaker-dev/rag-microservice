"""Token-aware chunking.

The monorepo placeholder split on blank lines and then sliced every 800
characters. That cuts through sentences, formulas and table rows, which is the
fastest way to make retrieval return unusable excerpts.

This implementation:
  * keeps structural blocks (headings, paragraphs, list items, table rows)
    intact whenever they fit the window;
  * packs consecutive blocks up to `max_tokens`;
  * splits an oversized block on sentence boundaries, then on whitespace, never
    mid-word;
  * carries `overlap_tokens` of trailing context into the next chunk so a
    definition split across a boundary stays retrievable from both sides;
  * emits a human-readable `locator` so a citation can point a student back to
    the exact place in the source.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence

# Sentence boundary: punctuation followed by whitespace and an opening
# character. Keeps decimal numbers and common French abbreviations intact.
_SENTENCE_BOUNDARY = re.compile(
    r"(?<=[.!?…])\s+(?=[«\"'(\[]?[A-ZÀ-ÖØ-Þ0-9])",
    re.UNICODE,
)
_HEADING = re.compile(r"^\s{0,3}(#{1,6})\s+(.*\S)\s*$")
_BLANK_LINE = re.compile(r"\n\s*\n")
_WHITESPACE = re.compile(r"\s+", re.UNICODE)


def default_token_counter() -> Callable[[str], int]:
    """Return a token counter.

    Uses `tiktoken` when available so the window matches what a model actually
    sees. Falls back to a whitespace heuristic calibrated for French, where a
    word averages slightly more than one token.
    """

    try:  # pragma: no cover - depends on the optional dependency
        import tiktoken

        encoding = tiktoken.get_encoding("cl100k_base")
        return lambda text: len(encoding.encode(text))
    except Exception:  # pragma: no cover - deterministic fallback
        return lambda text: max(1, int(len(_WHITESPACE.split(text.strip())) * 1.3))


@dataclass(frozen=True)
class Block:
    """A structural unit of the source document."""

    text: str
    heading_path: Sequence[str]
    position: int


@dataclass(frozen=True)
class TextChunk:
    ordinal: int
    locator: str
    content: str
    token_count: int


def split_blocks(document: str) -> List[Block]:
    """Split raw text into structural blocks, tracking the heading path."""

    blocks: List[Block] = []
    heading_path: List[str] = []
    position = 0

    for raw_section in _BLANK_LINE.split(document.replace("\r\n", "\n")):
        section = raw_section.strip()
        if not section:
            continue

        lines = section.split("\n")
        buffer: List[str] = []

        def flush() -> None:
            nonlocal buffer, position
            if not buffer:
                return
            text = "\n".join(buffer).strip()
            buffer = []
            if not text:
                return
            blocks.append(Block(text=text, heading_path=tuple(heading_path), position=position))
            position += 1

        for line in lines:
            heading = _HEADING.match(line)
            if heading:
                flush()
                level = len(heading.group(1))
                title = heading.group(2).strip()
                del heading_path[level - 1 :]
                heading_path.append(title)
                continue
            buffer.append(line)

        flush()

    return blocks


def _split_oversized(text: str, max_tokens: int, count: Callable[[str], int]) -> List[str]:
    """Break a block that exceeds the window, preferring sentence boundaries."""

    if count(text) <= max_tokens:
        return [text]

    pieces: List[str] = []
    for sentence in _SENTENCE_BOUNDARY.split(text):
        sentence = sentence.strip()
        if not sentence:
            continue
        if count(sentence) <= max_tokens:
            pieces.append(sentence)
            continue
        # A single sentence longer than the window: pack whole words.
        words = _WHITESPACE.split(sentence)
        current: List[str] = []
        for word in words:
            candidate = " ".join(current + [word])
            if current and count(candidate) > max_tokens:
                pieces.append(" ".join(current))
                current = [word]
            else:
                current.append(word)
        if current:
            pieces.append(" ".join(current))
    return pieces or [text]


def _tail_for_overlap(text: str, overlap_tokens: int, count: Callable[[str], int]) -> str:
    """Return the trailing fragment carrying at most `overlap_tokens`."""

    if overlap_tokens <= 0:
        return ""
    words = _WHITESPACE.split(text.strip())
    tail: List[str] = []
    for word in reversed(words):
        candidate = [word] + tail
        if count(" ".join(candidate)) > overlap_tokens:
            break
        tail = candidate
    return " ".join(tail)


def _locator(block: Optional[Block], ordinal: int) -> str:
    if block is None or not block.heading_path:
        return f"§{ordinal + 1}"
    return " > ".join(block.heading_path) + f" · §{ordinal + 1}"


def chunk_document(
    document: str,
    *,
    max_tokens: int = 384,
    overlap_tokens: int = 64,
    min_tokens: int = 32,
    count_tokens: Optional[Callable[[str], int]] = None,
) -> List[TextChunk]:
    """Chunk a document into overlapping, token-bounded passages."""

    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    if overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens must stay below max_tokens")

    count = count_tokens or default_token_counter()
    blocks = split_blocks(document)
    if not blocks:
        return []

    chunks: List[TextChunk] = []
    buffer: List[str] = []
    buffer_block: Optional[Block] = None
    carry = ""

    def emit() -> None:
        nonlocal buffer, carry, buffer_block
        body = "\n\n".join(part for part in buffer if part).strip()
        if not body:
            buffer = []
            return
        content = f"{carry}\n\n{body}".strip() if carry else body
        chunks.append(
            TextChunk(
                ordinal=len(chunks),
                locator=_locator(buffer_block, len(chunks)),
                content=content,
                token_count=count(content),
            )
        )
        carry = _tail_for_overlap(body, overlap_tokens, count)
        buffer = []

    for block in blocks:
        for piece in _split_oversized(block.text, max_tokens, count):
            candidate = "\n\n".join(buffer + [piece])
            projected = count(f"{carry}\n\n{candidate}" if carry else candidate)
            if buffer and projected > max_tokens:
                emit()
                buffer_block = block
            if not buffer:
                buffer_block = block
                # Le report de recouvrement ne doit pas faire déborder le
                # premier chunk qui suit une émission : le morceau seul tient
                # déjà dans la fenêtre, on sacrifie le report.
                if carry and count(f"{carry}\n\n{piece}") > max_tokens:
                    carry = ""
            buffer.append(piece)

    emit()

    # A trailing fragment shorter than `min_tokens` carries no standalone
    # meaning; fold it back into the previous chunk rather than indexing noise.
    if len(chunks) > 1 and chunks[-1].token_count < min_tokens:
        merged = f"{chunks[-2].content}\n\n{chunks[-1].content}".strip()
        # Ne fusionner que si le résultat tient encore dans la fenêtre :
        # un fragment court vaut mieux qu'un chunk hors gabarit.
        if count(merged) <= max_tokens:
            last = chunks.pop()
            previous = chunks.pop()
            chunks.append(
                TextChunk(
                    ordinal=previous.ordinal,
                    locator=previous.locator,
                    content=merged,
                    token_count=count(merged),
                )
            )

    return chunks
