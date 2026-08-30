"""Embedding backends.

One protocol, two implementations. The monorepo already abstracts providers
behind ports with circuit breakers; this service mirrors that discipline so a
backend swap never reaches calling code.
"""

from __future__ import annotations

import asyncio
import logging
from typing import List, Protocol, Sequence

import httpx

from app.config import Settings

logger = logging.getLogger(__name__)


class EmbeddingError(RuntimeError):
    """Raised when no attempt produced a usable vector."""


class EmbeddingProvider(Protocol):
    model: str

    async def embed(self, texts: Sequence[str]) -> List[List[float]]:
        """Vectoriser par lots.

        Un document entier envoyé d'un bloc — 180 chunks du programme
        national — dépassait le délai d'inférence sur CPU et ressortait en
        503. Le lot borne le coût de chaque appel ; le délai redevient une
        garantie par lot, pas une loterie par document.
        """

        vectors: List[List[float]] = []
        size = max(1, self._settings.embedding_batch_size)
        for start in range(0, len(texts), size):
            vectors.extend(await self._embed_batch(texts[start : start + size]))
        return vectors

    async def _embed_batch(self, texts: Sequence[str]) -> List[List[float]]: ...

    async def healthy(self) -> bool: ...


class OllamaEmbeddingProvider:
    """Ollama `/api/embed`, which accepts a batch and returns one vector each."""

    def __init__(self, settings: Settings, client: httpx.AsyncClient) -> None:
        self._settings = settings
        self._client = client
        self.model = settings.embedding_model
        self._base_url = settings.ollama_base_url.rstrip("/")

    async def embed(self, texts: Sequence[str]) -> List[List[float]]:
        """Vectoriser par lots.

        Un document entier envoyé d'un bloc — 180 chunks du programme
        national — dépassait le délai d'inférence sur CPU et ressortait en
        503. Le lot borne le coût de chaque appel ; le délai redevient une
        garantie par lot, pas une loterie par document.
        """

        vectors: List[List[float]] = []
        size = max(1, self._settings.embedding_batch_size)
        for start in range(0, len(texts), size):
            vectors.extend(await self._embed_batch(texts[start : start + size]))
        return vectors

    async def _embed_batch(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        payload = {"model": self.model, "input": list(texts)}
        last_error: Exception | None = None

        for attempt in range(1, self._settings.inference_max_attempts + 1):
            try:
                response = await self._client.post(
                    f"{self._base_url}/api/embed",
                    json=payload,
                    timeout=self._settings.inference_timeout_s,
                )
                response.raise_for_status()
                body = response.json()
                vectors = body.get("embeddings")
                if not isinstance(vectors, list) or len(vectors) != len(texts):
                    raise EmbeddingError("embedding backend returned an unexpected shape")
                return [[float(value) for value in vector] for vector in vectors]
            except Exception as error:  # noqa: BLE001 - retried below
                last_error = error
                # Never log the texts: they are pedagogical content tied to a student.
                logger.warning(
                    "embedding attempt failed",
                    extra={"attempt": attempt, "model": self.model},
                )
                if attempt < self._settings.inference_max_attempts:
                    await asyncio.sleep(min(2 ** (attempt - 1) * 0.25, 4.0))

        raise EmbeddingError("embedding backend unavailable") from last_error

    async def healthy(self) -> bool:
        try:
            response = await self._client.get(f"{self._base_url}/api/tags", timeout=3.0)
            return response.status_code == 200
        except Exception:  # noqa: BLE001
            return False


class VllmEmbeddingProvider:
    """OpenAI-compatible `/v1/embeddings`, as served by vLLM."""

    def __init__(self, settings: Settings, client: httpx.AsyncClient) -> None:
        if not settings.vllm_base_url:
            raise ValueError("VLLM_BASE_URL is required for the vLLM embedding provider")
        self._settings = settings
        self._client = client
        self.model = settings.embedding_model
        self._base_url = settings.vllm_base_url.rstrip("/")

    async def embed(self, texts: Sequence[str]) -> List[List[float]]:
        """Vectoriser par lots.

        Un document entier envoyé d'un bloc — 180 chunks du programme
        national — dépassait le délai d'inférence sur CPU et ressortait en
        503. Le lot borne le coût de chaque appel ; le délai redevient une
        garantie par lot, pas une loterie par document.
        """

        vectors: List[List[float]] = []
        size = max(1, self._settings.embedding_batch_size)
        for start in range(0, len(texts), size):
            vectors.extend(await self._embed_batch(texts[start : start + size]))
        return vectors

    async def _embed_batch(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        response = await self._client.post(
            f"{self._base_url}/v1/embeddings",
            json={"model": self.model, "input": list(texts)},
            timeout=self._settings.inference_timeout_s,
        )
        response.raise_for_status()
        data = response.json().get("data", [])
        if len(data) != len(texts):
            raise EmbeddingError("embedding backend returned an unexpected shape")
        ordered = sorted(data, key=lambda item: item.get("index", 0))
        return [[float(value) for value in item["embedding"]] for item in ordered]

    async def healthy(self) -> bool:
        try:
            response = await self._client.get(f"{self._base_url}/v1/models", timeout=3.0)
            return response.status_code == 200
        except Exception:  # noqa: BLE001
            return False


def build_embedding_provider(
    settings: Settings, client: httpx.AsyncClient
) -> EmbeddingProvider:
    if settings.embedding_provider == "vllm":
        return VllmEmbeddingProvider(settings, client)
    return OllamaEmbeddingProvider(settings, client)
