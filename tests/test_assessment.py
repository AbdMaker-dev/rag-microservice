"""Devoirs, compositions et examens blancs : plusieurs cours, un barème juste.

Un devoir porte sur les cours que le prof désigne, une composition sur un
semestre. Deux exigences non négociables : le barème tombe sur le total
annoncé, et aucun cours désigné n'est laissé de côté sans le dire.
"""

import asyncio
import json

import pytest

from app.config import get_settings
from app.core.generation import CourseGenerator, GenerationFailed
from tests.test_generation import SCOPE, FakeRetriever, ScriptedLlm

SOURCES = [
    {"heading": "Similitudes directes", "text": "z' = az + b, centre, rapport, angle. " * 3},
    {"heading": "Nombres complexes", "text": "Module, argument, forme exponentielle. " * 3},
]


def _gen(replies):
    return CourseGenerator(llm=ScriptedLlm(replies), retriever=FakeRetriever(),
                           settings=get_settings())


def _epreuve(exercices, titre="Devoir de maths"):
    return json.dumps({"titre": titre, "consignes": "Calculatrice interdite.",
                       "exercices": exercices})


def test_le_devoir_sort_avec_ses_exercices_corriges_et_son_bareme():
    reply = _epreuve([
        {"enonce": "Déterminer le centre.", "corrige": "Ω = b/(1−a)…", "points": 12,
         "couvre": ["Similitudes directes"]},
        {"enonce": "Calculer le module.", "corrige": "|z| = …", "points": 8,
         "couvre": ["Nombres complexes"]},
    ])
    draft = asyncio.run(_gen([reply]).compose_assessment(
        kind="devoir", sources=SOURCES, scope=SCOPE, duration_minutes=60, total_points=20))

    assert draft.kind == "devoir"
    assert draft.title == "Devoir de maths"
    assert draft.instructions.startswith("Calculatrice")
    assert [e["points"] for e in draft.exercises] == [12, 8]
    assert draft.exercises[0]["covers"] == ["Similitudes directes"]
    assert draft.warnings == []


def test_un_bareme_qui_ne_tombe_pas_juste_est_rectifie():
    """« sur 20 » avec 23 points distribués : le prof corrigerait à la main
    à chaque fois."""

    reply = _epreuve([
        {"enonce": "A", "corrige": "a", "points": 15, "couvre": ["Similitudes directes"]},
        {"enonce": "B", "corrige": "b", "points": 8, "couvre": ["Nombres complexes"]},
    ])
    draft = asyncio.run(_gen([reply]).compose_assessment(
        kind="devoir", sources=SOURCES, scope=SCOPE, total_points=20))

    assert sum(e["points"] for e in draft.exercises) == 20
    assert "ASSESSMENT_POINTS_ADJUSTED" in draft.warnings


def test_un_cours_non_couvert_est_signale_au_professeur():
    reply = _epreuve([
        {"enonce": "A", "corrige": "a", "points": 20, "couvre": ["Similitudes directes"]},
    ])
    draft = asyncio.run(_gen([reply]).compose_assessment(
        kind="composition", sources=SOURCES, scope=SCOPE, total_points=20))

    assert "ASSESSMENT_COURSE_NOT_COVERED" in draft.warnings


def test_un_exercice_sans_corrige_ou_sans_points_est_retire():
    reply = _epreuve([
        {"enonce": "Bon", "corrige": "c", "points": 20, "couvre": ["Similitudes directes"]},
        {"enonce": "Sans corrigé", "corrige": "", "points": 5},
        {"enonce": "Sans points", "corrige": "c", "points": 0},
    ])
    draft = asyncio.run(_gen([reply]).compose_assessment(
        kind="devoir", sources=SOURCES, scope=SCOPE, total_points=20))

    assert [e["statement"] for e in draft.exercises] == ["Bon"]
    assert "ASSESSMENT_ITEMS_DROPPED" in draft.warnings


def test_un_cours_inconnu_dans_couvre_est_ignore():
    # Le modèle invente parfois un titre : seuls les cours fournis comptent.
    reply = _epreuve([
        {"enonce": "A", "corrige": "a", "points": 10, "couvre": ["Chapitre inventé"]},
        {"enonce": "B", "corrige": "b", "points": 10,
         "couvre": ["Similitudes directes", "Nombres complexes"]},
    ])
    draft = asyncio.run(_gen([reply]).compose_assessment(
        kind="devoir", sources=SOURCES, scope=SCOPE, total_points=20))

    assert draft.exercises[0]["covers"] == []
    assert draft.exercises[1]["covers"] == ["Similitudes directes", "Nombres complexes"]


def test_le_modele_recoit_tous_les_cours_la_duree_et_le_bareme():
    llm = ScriptedLlm([_epreuve([{"enonce": "A", "corrige": "a", "points": 30,
                                  "couvre": ["Similitudes directes", "Nombres complexes"]}])])
    gen = CourseGenerator(llm=llm, retriever=FakeRetriever(), settings=get_settings())
    asyncio.run(gen.compose_assessment(
        kind="composition", sources=SOURCES, scope=SCOPE,
        duration_minutes=120, total_points=30, exercise_count=4,
        instruction="insiste sur les démonstrations"))

    system, user = llm.exchanges[0][0]["content"], llm.exchanges[0][1]["content"]
    assert "COMPOSITION" in system and "120 minutes" in system and "30 points" in system
    assert "4 exercices" in system and "insiste sur les démonstrations" in system
    assert "Similitudes directes" in user and "Nombres complexes" in user
    # Le schéma contraint la sortie, comme pour les blocs.
    assert llm.schemas[0]["required"] == ["titre", "exercices"]


def test_une_epreuve_illisible_echoue_apres_une_relance():
    with pytest.raises(GenerationFailed):
        asyncio.run(_gen(["du texte", "encore du texte"]).compose_assessment(
            kind="devoir", sources=SOURCES, scope=SCOPE))


def test_la_route_assessment_rend_un_ticket_puis_l_epreuve():
    import time

    from fastapi.testclient import TestClient

    from app.core.jobs import JobStore
    from app.main import create_app

    app = create_app()
    app.state.jobs = JobStore()
    app.state.llm = ScriptedLlm([_epreuve([
        {"enonce": "Ex 1", "corrige": "corrigé", "points": 20,
         "couvre": ["Similitudes directes", "Nombres complexes"]}])])
    app.state.retriever = FakeRetriever()
    client = TestClient(app)
    token = {"X-Service-Token": "test-secret-value-of-at-least-32-chars"}
    accepted = client.post("/generate/assessment", headers=token, json={
        "requestId": "a-1", "kind": "devoir", "durationMinutes": 55, "totalPoints": 20,
        "scope": {"country": "SN", "subject": "maths", "level": "secondaire",
                  "track": "S2", "grade": "terminale", "curriculumVersion": "2006"},
        "sources": [{"heading": s["heading"], "text": s["text"]} for s in SOURCES],
    })
    assert accepted.status_code == 202
    job = accepted.json()["jobId"]
    for _ in range(50):
        body = client.get(f"/generate/{job}", headers=token).json()
        if body["status"] != "running":
            break
        time.sleep(0.1)
    assert body["status"] == "done"
    assert body["kind"] == "devoir"
    assert body["assessment"]["durationMinutes"] == 55
    assert body["assessment"]["exercises"][0]["points"] == 20
