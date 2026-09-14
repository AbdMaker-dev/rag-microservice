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


def test_des_cours_entiers_ne_font_plus_echouer_la_composition():
    """Le premier devoir jamais composé sur le serveur a été refusé à la
    porte : un cours validé de 28 000 caractères, pour une limite de 20 000.

    Le contrat demande les RÉSUMÉS, qui tiennent — mais un appelant qui
    envoie les cours entiers ne doit pas voir la composition échouer. Chaque
    cours reçoit la même part de la fenêtre, et on le dit.
    """

    reponse = json.dumps({"titre": "Devoir", "consignes": "Traitez tout.",
                          "exercices": [{"enonce": "E1", "corrige": "C1", "points": 20,
                                         "couvre": ["Suites"]}]})
    llm = ScriptedLlm([reponse])
    generateur = CourseGenerator(llm=llm, retriever=FakeRetriever(),
                                 settings=get_settings())
    long = "Une phrase de cours qui ne dit pas grand-chose. " * 700  # ~33 000 car

    epreuve = asyncio.run(generateur.compose_assessment(
        kind="devoir", scope=SCOPE, total_points=20, exercise_count=1,
        sources=[{"heading": "Suites", "text": long},
                 {"heading": "Complexes", "text": long}]))

    assert epreuve.exercises[0]["points"] == 20
    assert "SOURCES_TRUNCATED" in epreuve.warnings
    # L'appel tient dans la fenêtre : c'est tout l'objet de la coupe.
    envoye = sum(len(m["content"]) for m in llm.exchanges[0]) // 3
    assert envoye <= get_settings().generation_context_tokens
    # Les deux cours sont présents, aucun n'est sacrifié pour l'autre.
    corpus = llm.exchanges[0][-1]["content"]
    assert "COURS 1 — Suites" in corpus and "COURS 2 — Complexes" in corpus


def test_des_resumes_courts_ne_sont_pas_tronques():
    reponse = json.dumps({"titre": "Devoir", "consignes": "",
                          "exercices": [{"enonce": "E1", "corrige": "C1", "points": 20,
                                         "couvre": ["Suites"]}]})
    llm = ScriptedLlm([reponse])
    generateur = CourseGenerator(llm=llm, retriever=FakeRetriever(),
                                 settings=get_settings())

    epreuve = asyncio.run(generateur.compose_assessment(
        kind="devoir", scope=SCOPE, total_points=20, exercise_count=1,
        sources=[{"heading": "Suites", "text": "Le résumé validé du cours. " * 10}]))

    assert epreuve.warnings == []


def test_un_accent_egare_dans_une_epreuve_est_signale():
    """« d\u2019\u0308finie » pour « définie » : relevé sur le premier devoir
    composé sur le serveur (14/09/2026), l'accent posé sur une apostrophe.

    On ne répare pas — un tréma égaré peut venir de « é », de « ë » ou d'un
    mot tout autre. On le DIT, et le professeur relit avant de publier.
    """

    abime = json.dumps({"titre": "Devoir", "consignes": "",
                        "exercices": [{"enonce": "Soit la suite d\u2019\u0308finie par u_0 = 2",
                                       "corrige": "C1", "points": 20, "couvre": ["Suites"]}]})
    epreuve = asyncio.run(_gen([abime]).compose_assessment(
        kind="devoir", scope=SCOPE, total_points=20, exercise_count=1,
        sources=[{"heading": "Suites", "text": "Le résumé du cours. " * 10}]))

    assert "DAMAGED_ACCENTS" in epreuve.warnings
    # Le texte est rendu tel quel : visible, pas deviné.
    assert "d\u2019\u0308finie" in epreuve.exercises[0]["statement"]


def test_les_accents_normaux_ne_sont_jamais_signales():
    # « majorée », « école », « l'élève » en forme décomposée : l'accent suit
    # sa lettre, tout va bien. Un garde qui crie sur du texte juste serait
    # aussi inutile qu'un garde muet.
    sain = json.dumps({"titre": "Devoir", "consignes": "",
                       "exercices": [{"enonce": "La suite majore\u0301e converge, l\u2019e\u0301le\u0300ve conclut.",
                                      "corrige": "C1", "points": 20, "couvre": ["Suites"]}]})
    epreuve = asyncio.run(_gen([sain]).compose_assessment(
        kind="devoir", scope=SCOPE, total_points=20, exercise_count=1,
        sources=[{"heading": "Suites", "text": "Le résumé du cours. " * 10}]))

    assert "DAMAGED_ACCENTS" not in epreuve.warnings


class RetrieverAvecAnnales(FakeRetriever):
    """Le périmètre porte deux annales indexées."""

    async def search(self, *, query, scope, limit, max_excerpt_characters,
                     course_id=None, role=None, document_ids=None):
        self.calls.append({"query": query, "role": role, "course_id": course_id})
        if role == "annale":
            from app.core.retrieval import Passage
            return [
                Passage(chunk_id="a1", document_id="bac-2024", title="Bac S2 2024",
                        locator="p. 2", content="Exercice 1 (5 points) — Le plan complexe…",
                        language="fr", score=0.9),
                Passage(chunk_id="a2", document_id="bac-2023", title="Bac S2 2023",
                        locator="p. 1", content="Exercice 2 (4 points) — Suites et limites…",
                        language="fr", score=0.8),
            ]
        return []


def test_une_epreuve_puise_dans_les_annales_du_perimetre():
    """Demande d'Alioune (14/09/2026) : « lors de la proposition des devoirs
    et exercices, il faut que l'IA puise dans les annales ».

    Le rôle « annale » existait dans l'index depuis le début — mais rien ne
    l'y cherchait, ni la rédaction ni la composition. Une annale déposée
    n'aurait servi à personne.
    """

    reponse = json.dumps({"titre": "Devoir", "consignes": "",
                          "exercices": [{"enonce": "E1", "corrige": "C1", "points": 20,
                                         "couvre": ["Suites"]}]})
    llm = ScriptedLlm([reponse])
    retriever = RetrieverAvecAnnales()
    generateur = CourseGenerator(llm=llm, retriever=retriever, settings=get_settings())

    asyncio.run(generateur.compose_assessment(
        kind="devoir", scope=SCOPE, total_points=20, exercise_count=1,
        sources=[{"heading": "Suites", "text": "Le résumé validé. " * 10}]))

    # La recherche est faite sur le PÉRIMÈTRE, pas sur un cours : une annale
    # est commune à la classe, comme un programme officiel.
    appel = [c for c in retriever.calls if c["role"] == "annale"][0]
    assert appel["course_id"] is None
    assert "Suites" in appel["query"]
    # Les annales accompagnent les cours dans la demande, étiquetées.
    demande = llm.exchanges[0][-1]["content"]
    assert "### ANNALE 1 — Bac S2 2024 (p. 2)" in demande
    assert "Le plan complexe" in demande
    # Et le modèle sait quoi en faire : le ton et le niveau, pas le contenu.
    consigne = llm.exchanges[0][0]["content"]
    assert "ne recopie aucun de leurs exercices" in consigne.lower()


def test_sans_annale_la_composition_se_fait_comme_avant():
    # Le cas d'aujourd'hui : aucune annale déposée. Rien ne change, et
    # surtout rien n'échoue.
    reponse = json.dumps({"titre": "Devoir", "consignes": "",
                          "exercices": [{"enonce": "E1", "corrige": "C1", "points": 20,
                                         "couvre": ["Suites"]}]})
    llm = ScriptedLlm([reponse])
    generateur = CourseGenerator(llm=llm, retriever=FakeRetriever(),
                                 settings=get_settings())

    epreuve = asyncio.run(generateur.compose_assessment(
        kind="devoir", scope=SCOPE, total_points=20, exercise_count=1,
        sources=[{"heading": "Suites", "text": "Le résumé validé. " * 10}]))

    assert epreuve.exercises[0]["points"] == 20
    assert "ANNALE" not in llm.exchanges[0][-1]["content"]
    assert "annales accompagnent" not in llm.exchanges[0][0]["content"]
