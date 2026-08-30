"""LLM backends.

Generation is optional here. The NestJS `AiGateway` owns policy, quotas,
lineage and structured-output validation; when it delegates a completion it
passes an already-approved prompt and expects nothing but the text back.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Protocol

import httpx

from app.config import Settings

logger = logging.getLogger(__name__)


class GenerationError(RuntimeError):
    """Raised when the backend could not produce a completion."""


class LlmProvider(Protocol):
    model: str

    async def complete(self, system: str, user: str) -> str: ...

    async def chat(self, messages: List[dict], *, timeout: float, num_ctx: int) -> str: ...

    async def healthy(self) -> bool: ...


class OllamaLlmProvider:
    def __init__(self, settings: Settings, client: httpx.AsyncClient) -> None:
        self._settings = settings
        self._client = client
        self.model = settings.llm_model
        self._base_url = settings.ollama_base_url.rstrip("/")

    async def complete(self, system: str, user: str) -> str:
        response = await self._client.post(
            f"{self._base_url}/api/chat",
            json={
                "model": self.model,
                "stream": False,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
            timeout=self._settings.inference_timeout_s,
        )
        response.raise_for_status()
        message = response.json().get("message", {})
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise GenerationError("completion backend returned an empty message")
        return content

    async def chat(self, messages: List[dict], *, timeout: float, num_ctx: int) -> str:
        """Une conversation complète — c'est elle que la génération utilise.

        L'historique porte les extraits déjà fournis et les recherches déjà
        faites : le modèle demande la suite en connaissant ce qu'il a reçu.
        """

        response = await self._client.post(
            f"{self._base_url}/api/chat",
            json={
                "model": self.model,
                "stream": False,
                "messages": messages,
                "options": {"num_ctx": num_ctx, "temperature": 0.3},
            },
            timeout=timeout,
        )
        response.raise_for_status()
        content = response.json().get("message", {}).get("content")
        if not isinstance(content, str) or not content.strip():
            raise GenerationError("completion backend returned an empty message")
        return content

    async def healthy(self) -> bool:
        try:
            response = await self._client.get(f"{self._base_url}/api/tags", timeout=3.0)
            return response.status_code == 200
        except Exception:  # noqa: BLE001
            return False


class DisabledLlmProvider:
    """Used when LLM_PROVIDER=none: retrieval works, generation refuses."""

    model = "disabled"

    async def complete(self, system: str, user: str) -> str:
        raise GenerationError("generation is disabled on this deployment")

    async def chat(self, messages: List[dict], *, timeout: float, num_ctx: int) -> str:
        raise GenerationError("generation is disabled on this deployment")

    async def healthy(self) -> bool:
        return True


def build_llm_provider(settings: Settings, client: httpx.AsyncClient) -> LlmProvider:
    if settings.llm_provider == "none":
        return DisabledLlmProvider()
    if settings.llm_provider == "vllm":
        # vLLM exposes an OpenAI-compatible chat endpoint; wire it when a real
        # deployment exists rather than guessing its shape now.
        raise NotImplementedError("the vLLM completion backend is not wired yet")
    return OllamaLlmProvider(settings, client)
