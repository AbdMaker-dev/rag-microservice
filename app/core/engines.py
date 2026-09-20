"""Le moteur IA choisi par le super admin, PAR PAYS et PAR USAGE.

Décision d'Alioune (15/09/2026) : tant qu'on n'a pas de GPU, un pays peut
faire écrire Lawal, la génération ou la relecture par une API en ligne
(Claude, GPT, Gemini) ; le jour du GPU, il repasse en local d'un clic —
sans code ni redéploiement. Le choix et la clé vivent dans management
(chiffrée, par pays) et arrivent avec CHAQUE demande : le rag reste sans
état et ne garde aucune clé.

Trois règles :
1. **Les adresses sont écrites ici**, jamais reçues : une clé ne part que
   chez le fournisseur nommé, et personne ne peut faire appeler au rag une
   adresse de son choix.
2. **La clé ne s'écrit nulle part** : ni journal, ni message d'erreur, ni
   représentation d'objet.
3. **Une panne du fournisseur ne coûte pas la réponse** : le modèle local
   prend le relais, et le repli est signalé (`ENGINE_FALLBACK_LOCAL`).
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional

import httpx

from app.core.llm import GenerationError, LlmProvider

logger = logging.getLogger(__name__)

_CLAUDE_URL = "https://api.anthropic.com/v1/messages"
_OPENAI_URL = "https://api.openai.com/v1/chat/completions"
# Gemini expose une interface compatible OpenAI : même forme de requête.
_GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"

# Un grand modèle en ligne n'a pas la limite de sortie d'un 7B sur CPU, et
# un modèle qui raisonne consomme des jetons avant d'écrire : la limite
# locale (700 pour Lawal) le couperait avant sa réponse.
_EXTERNAL_MIN_OUTPUT_TOKENS = 8_192
_EXTERNAL_TIMEOUT_S = 180.0


class EngineError(GenerationError):
    """Le fournisseur en ligne n'a pas rendu de texte. Message sans secret."""

    def __init__(self, message: str, status: int = 0) -> None:
        super().__init__(message)
        self.status = status


# Une surcharge passagère n'est pas une panne. Constaté le 16/09/2026 :
# gemini-3.8-flash répondait 503 « high demand » par vagues, et chaque vague
# envoyait la question sur le modèle local — deux minutes au lieu de deux
# secondes. On réessaie d'abord, brièvement ; une clé refusée (401, 403) ou
# une demande invalide (400) ne se réessaie pas.
_RETRY_STATUSES = {429, 500, 502, 503, 504}
_RETRY_DELAYS_S = (2.0, 5.0)
_sleep = asyncio.sleep


def _retryable(error: Exception) -> bool:
    if isinstance(error, EngineError):
        return error.status in _RETRY_STATUSES
    return isinstance(error, httpx.TransportError)


def _raise_for(provider: str, response: httpx.Response) -> None:
    if response.status_code < 400:
        return
    # Le corps d'erreur des fournisseurs ne contient pas la clé ; on n'en
    # garde qu'un extrait, pour dire au super admin CE qui ne va pas.
    detail = response.text[:200].replace("\n", " ")
    raise EngineError(
        f"{provider} a répondu {response.status_code} : {detail}", response.status_code
    )


def _split_system(messages: List[dict]):
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    rest: List[dict] = []
    for message in messages:
        if message["role"] == "system":
            continue
        # Claude refuse deux messages de même rôle à la suite : on les joint.
        if rest and rest[-1]["role"] == message["role"]:
            rest[-1] = {
                "role": message["role"],
                "content": rest[-1]["content"] + "\n\n" + message["content"],
            }
        else:
            rest.append({"role": message["role"], "content": message["content"]})
    return system, rest


class ClaudeProvider:
    provider = "claude"

    def __init__(self, *, model: str, api_key: str, client: httpx.AsyncClient) -> None:
        self.model = model
        self.__key = api_key
        self._client = client

    def __repr__(self) -> str:  # la clé n'apparaît jamais
        return f"ClaudeProvider(model={self.model!r})"

    async def complete(self, system: str, user: str) -> str:
        return await self.chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            timeout=_EXTERNAL_TIMEOUT_S, num_ctx=0, num_predict=0,
        )

    async def chat(self, messages, *, timeout, num_ctx, num_predict, schema=None) -> str:
        system, rest = _split_system(messages)
        response = await self._client.post(
            _CLAUDE_URL,
            headers={"x-api-key": self.__key, "anthropic-version": "2023-06-01"},
            json={
                "model": self.model,
                "max_tokens": max(num_predict, _EXTERNAL_MIN_OUTPUT_TOKENS),
                **({"system": system} if system else {}),
                "messages": rest,
            },
            timeout=max(timeout, _EXTERNAL_TIMEOUT_S),
        )
        _raise_for("Claude", response)
        text = "".join(
            block.get("text", "")
            for block in response.json().get("content", [])
            if block.get("type") == "text"
        )
        if not text.strip():
            raise EngineError("Claude a rendu une réponse vide")
        return text

    async def healthy(self) -> bool:
        return True


class OpenAiCompatibleProvider:
    """GPT, et Gemini par son interface compatible."""

    def __init__(
        self, *, provider: str, url: str, model: str, api_key: str,
        client: httpx.AsyncClient, output_field: str,
    ) -> None:
        self.provider = provider
        self.model = model
        self._url = url
        self.__key = api_key
        self._client = client
        self._output_field = output_field

    def __repr__(self) -> str:
        return f"OpenAiCompatibleProvider(provider={self.provider!r}, model={self.model!r})"

    async def complete(self, system: str, user: str) -> str:
        return await self.chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            timeout=_EXTERNAL_TIMEOUT_S, num_ctx=0, num_predict=0,
        )

    async def chat(self, messages, *, timeout, num_ctx, num_predict, schema=None) -> str:
        # Pas de température : les modèles qui raisonnent refusent toute
        # valeur autre que celle par défaut.
        response = await self._client.post(
            self._url,
            headers={"Authorization": f"Bearer {self.__key}"},
            json={
                "model": self.model,
                "messages": [{"role": m["role"], "content": m["content"]} for m in messages],
                self._output_field: max(num_predict, _EXTERNAL_MIN_OUTPUT_TOKENS),
            },
            timeout=max(timeout, _EXTERNAL_TIMEOUT_S),
        )
        _raise_for(self.provider.upper(), response)
        choices = response.json().get("choices") or []
        text = (choices[0].get("message", {}).get("content") if choices else "") or ""
        if not text.strip():
            raise EngineError(f"{self.provider.upper()} a rendu une réponse vide")
        return text

    async def healthy(self) -> bool:
        return True


def external_provider(engine, client: httpx.AsyncClient) -> Optional[LlmProvider]:
    """Le fournisseur en ligne demandé, ou None pour le modèle local."""

    if engine is None or engine.provider == "local":
        return None
    key = engine.api_key.get_secret_value()
    if engine.provider == "claude":
        return ClaudeProvider(model=engine.model, api_key=key, client=client)
    if engine.provider == "gpt":
        return OpenAiCompatibleProvider(
            provider="gpt", url=_OPENAI_URL, model=engine.model, api_key=key,
            client=client, output_field="max_completion_tokens",
        )
    return OpenAiCompatibleProvider(
        provider="gemini", url=_GEMINI_URL, model=engine.model, api_key=key,
        client=client, output_field="max_tokens",
    )


@dataclass
class EngineTrace:
    """Ce qui a réellement écrit — rendu avec le résultat, pour la traçabilité."""

    provider: str
    model: str
    fell_back: bool = False
    retries: int = 0
    errors: List[str] = field(default_factory=list)

    def warnings(self) -> List[str]:
        return ["ENGINE_FALLBACK_LOCAL"] if self.fell_back else []


class FallbackLlm:
    """Le fournisseur en ligne, et le modèle local s'il tombe.

    Le repli se décide APPEL PAR APPEL : une génération longue qui a perdu
    le réseau au troisième échange finit en local plutôt qu'en échec.
    """

    def __init__(self, primary: LlmProvider, local: LlmProvider, trace: EngineTrace) -> None:
        self._primary = primary
        self._local = local
        self.trace = trace
        self.model = primary.model

    async def complete(self, system: str, user: str) -> str:
        try:
            return await self._with_retries(lambda: self._primary.complete(system, user))
        except (httpx.HTTPError, GenerationError, ValueError) as error:
            self._record(error)
            return await self._local.complete(system, user)

    async def chat(self, messages, *, timeout, num_ctx, num_predict, schema=None) -> str:
        try:
            return await self._with_retries(
                lambda: self._primary.chat(
                    messages, timeout=timeout, num_ctx=num_ctx, num_predict=num_predict,
                    schema=schema,
                )
            )
        except (httpx.HTTPError, GenerationError, ValueError) as error:
            self._record(error)
            kwargs = {"schema": schema} if schema else {}
            return await self._local.chat(
                messages, timeout=timeout, num_ctx=num_ctx, num_predict=num_predict,
                **kwargs,
            )

    async def _with_retries(self, call):
        for delay in (*_RETRY_DELAYS_S, None):
            try:
                return await call()
            except (httpx.HTTPError, GenerationError, ValueError) as error:
                if delay is None or not _retryable(error):
                    raise
                self.trace.retries += 1
                logger.info(
                    "moteur en ligne surchargé, nouvel essai",
                    extra={"provider": self.trace.provider, "model": self.trace.model,
                           "dans": delay},
                )
                await _sleep(delay)

    def _record(self, error: Exception) -> None:
        # httpx.HTTPError ne porte que l'adresse, jamais les en-têtes : la
        # clé ne peut pas fuiter par ce message.
        message = f"{type(error).__name__}: {error}"[:300]
        self.trace.fell_back = True
        self.trace.errors.append(message)
        logger.warning(
            "moteur en ligne indisponible, repli local",
            extra={"provider": self.trace.provider, "model": self.trace.model, "erreur": message},
        )

    async def healthy(self) -> bool:
        return True


def resolve_llm(engine, local: LlmProvider, client: httpx.AsyncClient, *, allow_fallback: bool = True):
    """(llm à utiliser, trace). Sans moteur en ligne : le local, tracé tel.

    `allow_fallback=False` pour la RELECTURE d'un cours : un modèle local ne
    vérifie pas des maths, et management marque de toute façon la relecture
    « non concluante ». Constaté le 20/09/2026 : Claude refusait en une
    seconde (crédit épuisé), le repli local tournait quinze minutes, et le
    professeur lisait « non vérifié » au bout. Mieux vaut échouer tout de
    suite en disant pourquoi.
    """

    primary = external_provider(engine, client)
    if primary is None:
        return local, EngineTrace(provider="local", model=getattr(local, "model", ""))
    trace = EngineTrace(provider=engine.provider, model=engine.model)
    if not allow_fallback:
        return primary, trace
    return FallbackLlm(primary, local, trace), trace


async def try_engine(engine, client: httpx.AsyncClient) -> dict:
    """Un aller-retour minuscule, SANS repli : le super admin veut savoir si
    SA clé marche, pas si le local répond."""

    provider = external_provider(engine, client)
    if provider is None:
        return {"ok": True, "latencyMs": 0, "error": None}
    started = time.monotonic()
    try:
        await provider.chat(
            [{"role": "user", "content": "Réponds seulement : OK"}],
            timeout=30.0, num_ctx=0, num_predict=16,
        )
    except (httpx.HTTPError, GenerationError) as error:
        return {
            "ok": False,
            "latencyMs": int((time.monotonic() - started) * 1000),
            "error": f"{type(error).__name__}: {error}"[:300],
        }
    return {"ok": True, "latencyMs": int((time.monotonic() - started) * 1000), "error": None}


_CLAUDE_MODELS_URL = "https://api.anthropic.com/v1/models"
_OPENAI_MODELS_URL = "https://api.openai.com/v1/models"
_GEMINI_MODELS_URL = "https://generativelanguage.googleapis.com/v1beta/openai/models"

# Ce que la liste d'un fournisseur contient et qui n'écrit pas de texte :
# embeddings, voix, images, recherche. Proposer ces modèles au super admin
# serait lui tendre un piège — la génération échouerait au premier appel.
_NOT_TEXT = ("embedding", "tts", "transcribe", "audio", "realtime", "image",
             "dall-e", "whisper", "moderation", "search", "aqa", "imagen", "veo")
_OPENAI_TEXT_PREFIXES = ("gpt-", "o1", "o3", "o4", "chatgpt-")


async def list_models(provider: str, api_key: str, client: httpx.AsyncClient) -> dict:
    """Les modèles qui écrivent du texte, tels que CETTE clé les voit.

    Rien en dur : le jour où un fournisseur sort un modèle, il apparaît dans
    l'écran du super admin sans qu'on touche au code.
    """

    try:
        if provider == "claude":
            response = await client.get(
                _CLAUDE_MODELS_URL,
                params={"limit": 1000},
                headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"},
                timeout=30.0,
            )
            _raise_for("Claude", response)
            models = [
                {"id": item["id"], "name": item.get("display_name") or item["id"]}
                for item in response.json().get("data", [])
            ]
        else:
            url = _OPENAI_MODELS_URL if provider == "gpt" else _GEMINI_MODELS_URL
            response = await client.get(
                url, headers={"Authorization": f"Bearer {api_key}"}, timeout=30.0
            )
            _raise_for(provider.upper(), response)
            models = []
            for item in response.json().get("data", []):
                model_id = str(item.get("id", "")).removeprefix("models/")
                if provider == "gpt" and not model_id.startswith(_OPENAI_TEXT_PREFIXES):
                    continue
                if provider == "gemini" and not model_id.startswith("gemini"):
                    continue
                models.append({"id": model_id, "name": model_id})
    except (httpx.HTTPError, GenerationError, ValueError) as error:
        return {"models": [], "error": f"{type(error).__name__}: {error}"[:300]}
    models = [m for m in models if not any(word in m["id"].lower() for word in _NOT_TEXT)]
    unique = {m["id"]: m for m in models}
    return {"models": sorted(unique.values(), key=lambda m: m["id"]), "error": None}
