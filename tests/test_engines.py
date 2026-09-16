"""Le moteur IA en ligne, choisi par pays et par usage (15/09/2026).

Tout passe par un transport httpx simulé : aucun appel réel, aucune clé
réelle. Ce qu'on vérifie : la forme envoyée à chaque fournisseur, le repli
local quand il tombe, la trace rendue avec le résultat, et que la clé ne
s'écrit nulle part.
"""

import asyncio
import json
import logging
import time

import httpx
import pytest
from pydantic import ValidationError

from app.core.engines import resolve_llm
from app.core.engines import try_engine as essayer_la_cle
from app.models.schemas import EngineChoice
from tests.test_generation import FakeRetriever, ScriptedLlm

CLE = "sk-cle-secrete-de-test-123456"
TOKEN = {"X-Service-Token": "test-secret-value-of-at-least-32-chars"}
SCOPE = {"country": "SN", "subject": "maths", "level": "secondaire", "track": "S2",
         "grade": "terminale", "curriculumVersion": "2006"}
QUIZ = ("### QUESTION\nQ\n- A) a\n- B) b\n- C) c\n- D) d\n### RÉPONSE B\n### EXPLICATION\ne")


def _client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _engine(provider="claude", model="claude-sonnet-5"):
    return EngineChoice(provider=provider, model=model, api_key=CLE)


class _Local:
    model = "qwen2.5:14b"

    def __init__(self, reply="réponse locale"):
        self.reply = reply
        self.calls = 0

    async def chat(self, messages, **kwargs):
        self.calls += 1
        return self.reply

    async def complete(self, system, user):
        self.calls += 1
        return self.reply


def test_claude_recoit_la_cle_le_systeme_a_part_et_des_roles_alternes():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["key"] = request.headers.get("x-api-key")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"content": [{"type": "text", "text": "Bonjour"}]})

    llm, trace = resolve_llm(_engine(), _Local(), _client(handler))
    text = asyncio.run(llm.chat(
        [{"role": "system", "content": "S"}, {"role": "user", "content": "a"},
         {"role": "user", "content": "b"}],
        timeout=10, num_ctx=8192, num_predict=700,
    ))
    assert text == "Bonjour"
    assert seen["url"] == "https://api.anthropic.com/v1/messages"
    assert seen["key"] == CLE
    assert seen["body"]["system"] == "S"
    assert seen["body"]["messages"] == [{"role": "user", "content": "a\n\nb"}]
    assert seen["body"]["max_tokens"] >= 8192
    assert trace.provider == "claude" and not trace.fell_back


@pytest.mark.parametrize("provider,url,field", [
    ("gpt", "https://api.openai.com/v1/chat/completions", "max_completion_tokens"),
    ("gemini", "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions", "max_tokens"),
])
def test_gpt_et_gemini_par_l_interface_openai(provider, url, field):
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "Salut"}}]})

    llm, _ = resolve_llm(_engine(provider, "modele-x"), _Local(), _client(handler))
    text = asyncio.run(llm.chat([{"role": "user", "content": "q"}],
                                timeout=10, num_ctx=0, num_predict=700))
    assert text == "Salut"
    assert seen["url"] == url
    assert seen["auth"] == f"Bearer {CLE}"
    assert seen["body"][field] >= 8192
    assert "temperature" not in seen["body"]


def test_sans_moteur_ou_en_local_c_est_le_modele_local():
    local = _Local()
    for engine in (None, EngineChoice(provider="local")):
        llm, trace = resolve_llm(engine, local, None)
        assert llm is local
        assert trace.provider == "local" and trace.model == "qwen2.5:14b"


def test_une_panne_du_fournisseur_bascule_sur_le_local_et_le_trace(caplog):
    local = _Local("réponse locale")
    llm, trace = resolve_llm(
        _engine(), local,
        _client(lambda request: httpx.Response(529, text='{"error":"overloaded"}')),
    )
    with caplog.at_level(logging.DEBUG):
        text = asyncio.run(llm.chat([{"role": "user", "content": "q"}],
                                    timeout=10, num_ctx=0, num_predict=10))
    assert text == "réponse locale"
    assert trace.fell_back and trace.warnings() == ["ENGINE_FALLBACK_LOCAL"]
    assert "529" in trace.errors[0]
    assert CLE not in caplog.text


def test_la_cle_ne_s_affiche_ni_dans_l_objet_ni_dans_une_erreur_de_validation():
    engine = _engine()
    assert CLE not in repr(engine)
    assert CLE not in str(engine.model_dump())
    llm, _ = resolve_llm(engine, _Local(), _client(lambda r: httpx.Response(200)))
    assert CLE not in repr(llm._primary)
    with pytest.raises(ValidationError) as error:
        EngineChoice(provider="claude", model="", api_key=CLE)
    assert CLE not in str(error.value)


def test_un_moteur_en_ligne_exige_modele_et_cle():
    with pytest.raises(ValidationError):
        EngineChoice(provider="gpt", model="x", api_key="")
    with pytest.raises(ValidationError):
        EngineChoice(provider="gemini", model="", api_key=CLE)


def test_tester_une_cle_ne_se_replie_jamais():
    ok = asyncio.run(essayer_la_cle(
        _engine(), _client(lambda r: httpx.Response(200, json={"content": [{"type": "text", "text": "OK"}]}))))
    assert ok["ok"] is True
    ko = asyncio.run(essayer_la_cle(
        _engine(), _client(lambda r: httpx.Response(401, text='{"error":"invalid x-api-key"}'))))
    assert ko["ok"] is False and "401" in ko["error"] and CLE not in ko["error"]


def _app(handler, llm_local):
    from fastapi.testclient import TestClient

    from app.core.jobs import JobStore
    from app.main import create_app

    app = create_app()
    app.state.jobs = JobStore()
    app.state.llm = llm_local
    app.state.retriever = FakeRetriever()
    app.state.http = _client(handler)
    return TestClient(app)


def _wait(client, path):
    for _ in range(100):
        body = client.get(path, headers=TOKEN).json()
        if body["status"] in ("done", "failed"):
            return body
        time.sleep(0.05)
    raise AssertionError(body)


def _blocks_body(engine=None):
    body = {"requestId": "b-1", "courseId": "cours-7", "kind": "quiz", "count": 1,
            "scope": SCOPE, "text": "Une similitude directe a pour écriture z' = az + b. " * 5}
    if engine:
        body["engine"] = engine
    return body


def test_la_route_rend_le_moteur_qui_a_ecrit_meme_en_local():
    client = _app(lambda r: httpx.Response(500), ScriptedLlm([QUIZ]))
    job = client.post("/generate/blocks", headers=TOKEN, json=_blocks_body()).json()["jobId"]
    body = _wait(client, f"/generate/{job}")
    assert body["status"] == "done"
    assert body["engine"]["provider"] == "local"
    assert body["engine"]["fallback"] is False


def test_la_route_en_ligne_trace_claude_puis_le_repli_local():
    client = _app(lambda r: httpx.Response(503, text="indisponible"), ScriptedLlm([QUIZ]))
    engine = {"provider": "claude", "model": "claude-sonnet-5", "apiKey": CLE}
    job = client.post("/generate/blocks", headers=TOKEN, json=_blocks_body(engine)).json()["jobId"]
    body = _wait(client, f"/generate/{job}")
    assert body["status"] == "done"
    assert body["engine"] == {"provider": "claude", "model": "claude-sonnet-5", "fallback": True}
    assert "ENGINE_FALLBACK_LOCAL" in body["warnings"]
    assert CLE not in json.dumps(body)


def test_une_tache_en_ligne_n_attend_pas_derriere_la_file_locale():
    """Une réponse de Claude ne doit pas patienter derrière une génération
    locale de cinq minutes : elle part hors file."""

    class _Lent(ScriptedLlm):
        async def chat(self, messages, **kwargs):
            await asyncio.sleep(3)
            return QUIZ

    def claude(request):
        return httpx.Response(200, json={"content": [{"type": "text", "text": QUIZ}]})

    client = _app(claude, _Lent([]))
    lent = client.post("/generate/blocks", headers=TOKEN, json=_blocks_body()).json()["jobId"]
    engine = {"provider": "claude", "model": "claude-sonnet-5", "apiKey": CLE}
    rapide = client.post("/generate/blocks", headers=TOKEN, json=_blocks_body(engine)).json()["jobId"]
    started = time.monotonic()
    body = _wait(client, f"/generate/{rapide}")
    assert body["status"] == "done" and body["engine"]["provider"] == "claude"
    assert time.monotonic() - started < 2.5
    assert client.get(f"/generate/{lent}", headers=TOKEN).json()["status"] != "done"


def test_la_route_de_test_de_cle():
    client = _app(lambda r: httpx.Response(401, text="invalid key"), ScriptedLlm([]))
    body = client.post("/engines/test", headers=TOKEN, json={
        "engine": {"provider": "gpt", "model": "modele-x", "apiKey": CLE}}).json()
    assert body["ok"] is False and "401" in body["error"]


def test_la_liste_claude_rend_les_noms_affiches():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["key"] = request.headers.get("x-api-key")
        return httpx.Response(200, json={"data": [
            {"id": "claude-sonnet-5", "display_name": "Claude Sonnet 5"},
            {"id": "claude-fable-5-1", "display_name": "Claude Fable 5.1"},
        ]})

    client = _app(handler, ScriptedLlm([]))
    body = client.post("/engines/models", headers=TOKEN,
                       json={"provider": "claude", "apiKey": CLE}).json()
    assert seen["url"].startswith("https://api.anthropic.com/v1/models")
    assert seen["key"] == CLE
    assert body == {"models": [{"id": "claude-fable-5-1", "name": "Claude Fable 5.1"},
                               {"id": "claude-sonnet-5", "name": "Claude Sonnet 5"}],
                    "error": None}


def test_la_liste_gpt_ne_garde_que_les_modeles_qui_ecrivent():
    def handler(request):
        return httpx.Response(200, json={"data": [
            {"id": "gpt-texte"}, {"id": "o3-raisonne"}, {"id": "text-embedding-3-large"},
            {"id": "gpt-audio-preview"}, {"id": "whisper-1"}, {"id": "dall-e-3"},
        ]})

    client = _app(handler, ScriptedLlm([]))
    body = client.post("/engines/models", headers=TOKEN, json={"provider": "gpt", "apiKey": CLE}).json()
    assert [m["id"] for m in body["models"]] == ["gpt-texte", "o3-raisonne"]


def test_la_liste_gemini_retire_le_prefixe_et_les_embeddings():
    def handler(request):
        assert str(request.url).startswith("https://generativelanguage.googleapis.com/")
        return httpx.Response(200, json={"data": [
            {"id": "models/gemini-flash-x"}, {"id": "models/gemini-embedding-001"},
            {"id": "models/imagen-4"},
        ]})

    client = _app(handler, ScriptedLlm([]))
    body = client.post("/engines/models", headers=TOKEN, json={"provider": "gemini", "apiKey": CLE}).json()
    assert body["models"] == [{"id": "gemini-flash-x", "name": "gemini-flash-x"}]


def test_une_cle_refusee_rend_une_liste_vide_et_l_erreur_sans_la_cle():
    client = _app(lambda r: httpx.Response(401, text="invalid x-api-key"), ScriptedLlm([]))
    body = client.post("/engines/models", headers=TOKEN, json={"provider": "claude", "apiKey": CLE}).json()
    assert body["models"] == [] and "401" in body["error"] and CLE not in body["error"]
