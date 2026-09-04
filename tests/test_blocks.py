"""Les trois blocs d'un cours : résumé, exercices, quiz — depuis le validé.

Une question de quiz dont la réponse n'est pas dans le cours est un piège,
pas de l'entraînement ; une question sans 4 propositions ou avec une réponse
hors plage est inutilisable et se retire en le disant.
"""

import asyncio
import json

import pytest

from app.config import get_settings
from app.core.generation import CourseGenerator, GenerationFailed
from tests.test_generation import SCOPE, FakeRetriever, ScriptedLlm

COURS = "Une similitude directe a pour écriture complexe z' = az + b. " * 5


def _gen(replies):
    return CourseGenerator(llm=ScriptedLlm(replies), retriever=FakeRetriever(),
                           settings=get_settings())


def test_le_quiz_sort_structure_avec_quatre_choix():
    reply = json.dumps({"questions": [
        {"question": "Forme de l'écriture complexe ?", "choix": ["z'=az+b", "z'=a", "z'=b", "z'=z"],
         "reponse": 0, "explication": "C'est la définition du cours."},
    ]})
    draft = asyncio.run(_gen([reply]).generate_blocks(
        kind="quiz", text=COURS, scope=SCOPE, count=5))
    assert draft.kind == "quiz"
    assert draft.quiz[0]["choices"] == ["z'=az+b", "z'=a", "z'=b", "z'=z"]
    assert draft.quiz[0]["answer"] == 0
    assert draft.warnings == []


def test_une_question_inutilisable_est_retiree_et_signalee():
    reply = json.dumps({"questions": [
        {"question": "Bonne", "choix": ["a", "b", "c", "d"], "reponse": 2},
        {"question": "Trois choix seulement", "choix": ["a", "b", "c"], "reponse": 0},
        {"question": "Réponse hors plage", "choix": ["a", "b", "c", "d"], "reponse": 7},
    ]})
    draft = asyncio.run(_gen([reply]).generate_blocks(
        kind="quiz", text=COURS, scope=SCOPE))
    assert [q["question"] for q in draft.quiz] == ["Bonne"]
    assert "BLOCK_ITEMS_DROPPED" in draft.warnings


def test_les_exercices_portent_leur_corrige():
    reply = json.dumps({"exercices": [
        {"enonce": "Déterminer S(A).", "corrige": "On calcule…", "difficulte": "facile"},
        {"enonce": "Sans corrigé", "corrige": ""},
    ]})
    draft = asyncio.run(_gen([reply]).generate_blocks(
        kind="exercices", text=COURS, scope=SCOPE))
    assert len(draft.exercises) == 1
    assert draft.exercises[0]["difficulty"] == "facile"
    assert "BLOCK_ITEMS_DROPPED" in draft.warnings


def test_le_resume_est_un_texte():
    draft = asyncio.run(_gen([json.dumps({"resume": "Les idées clés…"})]).generate_blocks(
        kind="resume", text=COURS, scope=SCOPE))
    assert draft.summary == "Les idées clés…"


def test_un_modele_hors_format_est_relance_puis_echoue_clairement():
    with pytest.raises(GenerationFailed):
        asyncio.run(_gen(["du texte", "encore du texte"]).generate_blocks(
            kind="quiz", text=COURS, scope=SCOPE))


def test_le_bloc_ne_cherche_jamais_dans_la_base():
    # La matière première est le cours validé : aucune recherche.
    retriever = FakeRetriever()
    gen = CourseGenerator(llm=ScriptedLlm([json.dumps({"resume": "ok"})]),
                          retriever=retriever, settings=get_settings())
    asyncio.run(gen.generate_blocks(kind="resume", text=COURS, scope=SCOPE))
    assert retriever.calls == []


def test_la_consigne_du_prof_et_le_cours_arrivent_au_modele():
    llm = ScriptedLlm([json.dumps({"resume": "ok"})])
    gen = CourseGenerator(llm=llm, retriever=FakeRetriever(), settings=get_settings())
    asyncio.run(gen.generate_blocks(kind="resume", text=COURS, scope=SCOPE,
                                    instruction="insiste sur le centre"))
    system, user = llm.exchanges[0][0]["content"], llm.exchanges[0][1]["content"]
    assert "insiste sur le centre" in system and "professionnel" in system
    assert "z' = az + b" in user


def test_la_route_blocks_rend_un_ticket_puis_le_quiz():
    import time

    from fastapi.testclient import TestClient

    from app.core.jobs import JobStore
    from app.main import create_app

    app = create_app()
    app.state.jobs = JobStore()
    app.state.llm = ScriptedLlm([json.dumps({"questions": [
        {"question": "Q", "choix": ["a", "b", "c", "d"], "reponse": 1, "explication": "e"}]})])
    app.state.retriever = FakeRetriever()
    client = TestClient(app)
    token = {"X-Service-Token": "test-secret-value-of-at-least-32-chars"}
    accepted = client.post("/generate/blocks", headers=token, json={
        "requestId": "b-1", "courseId": "cours-7", "kind": "quiz", "count": 3,
        "scope": {"country": "SN", "subject": "maths", "level": "secondaire",
                  "track": "S2", "grade": "terminale", "curriculumVersion": "2006"},
        "text": COURS,
    })
    assert accepted.status_code == 202
    job = accepted.json()["jobId"]
    for _ in range(50):
        body = client.get(f"/generate/{job}", headers=token).json()
        if body["status"] != "running":
            break
        time.sleep(0.1)
    assert body["status"] == "done"
    assert body["kind"] == "quiz"
    assert body["quiz"][0]["answer"] == 1


def test_les_formules_a_backslash_ne_cassent_plus_le_json():
    """Constaté en production : « \\( z' \\) » dans une question rendait le
    quiz entier illisible — échappement JSON invalide."""

    from app.core.generation import _parse_json_block

    raw = 'Voici : {"questions": [{"question": "Que vaut \\( z\' \\) si \\frac{1}{2} ?", ' \
          '"choix": ["a", "b", "c", "d"], "reponse": 0, "explication": "voir \\(\\omega\\)"}]} merci'
    parsed = _parse_json_block(raw)
    assert parsed is not None
    assert parsed["questions"][0]["question"].startswith("Que vaut")


def test_deux_objets_cote_a_cote_fusionnent():
    from app.core.generation import _parse_json_block

    parsed = _parse_json_block('{"reponse": "A"}\n{"verification": "B"}')
    assert parsed == {"reponse": "A", "verification": "B"}


def test_frac_et_theta_survivent_a_la_lecture_json():
    """« \\frac » devenait « rac » et « \\theta » une tabulation : \\f et \\t
    sont des échappements JSON valides. Les commandes LaTeX se protègent
    avant lecture ; un vrai « \\n » de retour à la ligne, lui, reste un
    retour à la ligne."""

    from app.core.generation import _parse_json_block

    raw = '{"q": "\\( \\Omega = \\frac{b}{1-a} \\), \\( \\theta = \\arg(a) \\), ligne1\\nligne2"}'
    parsed = _parse_json_block(raw)
    assert parsed is not None
    assert "\\frac{b}{1-a}" in parsed["q"]
    assert "\\theta" in parsed["q"]
    assert "ligne1\nligne2" in parsed["q"]


def test_les_exercices_ont_un_plafond_de_sortie_plus_haut():
    """3 exercices corrigés dépassaient 1 200 tokens : JSON tronqué."""

    class Recording(ScriptedLlm):
        def __init__(self, replies):
            super().__init__(replies); self.predicts = []
        async def chat(self, messages, *, timeout, num_ctx, num_predict):
            self.predicts.append(num_predict)
            return await super().chat(messages, timeout=timeout, num_ctx=num_ctx, num_predict=num_predict)

    llm = Recording([json.dumps({"exercices": [{"enonce": "e", "corrige": "c"}]})])
    gen = CourseGenerator(llm=llm, retriever=FakeRetriever(), settings=get_settings())
    asyncio.run(gen.generate_blocks(kind="exercices", text=COURS, scope=SCOPE))
    assert llm.predicts[0] >= 3000
