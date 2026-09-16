"""L'audit d'un cours avant publication (16/09/2026)."""

import asyncio
import time

from app.core.course_audit import audit_course
from tests.test_generation import FakeRetriever, ScriptedLlm

COURS = (
    "Calculons le module : \\( |z_1| = \\sqrt{(-1)^2 + 1^2} = \\sqrt{2} \\).\n"
    "Calculons l'argument : \\( \\arg(z_1) = \\frac{3\\pi}{4} \\) (puisque \\( z_1 \\) est dans "
    "le premier quadrant du plan complexe).\n\n"
    "Vérifions : \\[ \\sqrt{9 + 16} = \\sqrt{24} = 5 \\]"
)
QUADRANT = (
    "### PROBLÈME\ngravité : probable\n"
    "extrait : (puisque \\( z_1 \\) est dans le premier quadrant du plan complexe)\n"
    "explication : −1 + i a une partie réelle négative et imaginaire positive : deuxième quadrant.\n"
    "correction : dans le deuxième quadrant\n"
)
INVENTE = (
    "### PROBLÈME\ngravité : probable\nextrait : le triangle est équilatéral en B\n"
    "explication : faux\ncorrection : rien\n"
)


def _run(replies):
    return asyncio.run(audit_course(text=COURS, llm=ScriptedLlm(replies), timeout=10, num_ctx=8192))


def test_le_calcul_faux_est_certain_et_le_quadrant_probable():
    result = _run([QUADRANT])
    calcul = [f for f in result.findings if f.source == "calcul"]
    relecture = [f for f in result.findings if f.source == "relecture"]
    assert calcul and all(f.severity == "certaine" for f in calcul)
    assert len(relecture) == 1 and relecture[0].severity == "probable"
    assert "deuxième quadrant" in relecture[0].explanation


def test_un_signalement_qui_cite_une_phrase_absente_est_ecarte():
    result = _run([INVENTE])
    assert not [f for f in result.findings if f.source == "relecture"]
    assert "AUDIT_QUOTE_NOT_FOUND" in result.warnings


def test_un_cours_sans_probleme_ne_rend_que_les_calculs():
    result = _run(["### AUCUN"])
    assert all(f.source == "calcul" for f in result.findings)


def test_la_route_rend_l_audit_au_sondage():
    from fastapi.testclient import TestClient

    from app.core.jobs import JobStore
    from app.main import create_app

    app = create_app()
    app.state.jobs = JobStore()
    app.state.llm = ScriptedLlm([QUADRANT])
    app.state.retriever = FakeRetriever()
    client = TestClient(app)
    token = {"X-Service-Token": "test-secret-value-of-at-least-32-chars"}
    accepted = client.post("/audit/course", headers=token, json={
        "requestId": "a-1", "courseId": "cours-7", "text": COURS,
        "scope": {"country": "SN", "subject": "maths", "level": "secondaire",
                  "track": "S2", "grade": "terminale", "curriculumVersion": "2006"},
    })
    assert accepted.status_code == 202
    job = accepted.json()["jobId"]
    for _ in range(50):
        body = client.get(f"/generate/{job}", headers=token).json()
        if body["status"] != "running" and body["status"] != "queued":
            break
        time.sleep(0.1)
    assert body["status"] == "done"
    severities = {f["severity"] for f in body["audit"]["findings"]}
    assert severities == {"certaine", "probable"}
    assert body["engine"]["provider"] == "local"


def test_un_titre_suivant_ne_se_colle_pas_dans_la_correction():
    """Vu le 16/09/2026 : « ### AUCUN » finissait dans la correction."""

    result = _run([QUADRANT + "\n### AUCUN\nPour les autres sections, rien à signaler."])
    relecture = [f for f in result.findings if f.source == "relecture"]
    assert relecture[0].correction == "dans le deuxième quadrant"


def test_un_quiz_qui_contredit_son_explication_est_une_erreur_certaine():
    quiz = {"question": "Forme de -1 + i ?", "choices": ["a", "b", "c", "d"], "answer": 0,
            "explanation": "La réponse correcte est C) √2 e^{i3π/4}."}
    result = asyncio.run(audit_course(text="Un cours.", llm=ScriptedLlm(["### AUCUN"]),
                                      timeout=10, num_ctx=8192, quizzes=[quiz]))
    certain = [f for f in result.findings if f.severity == "certaine"]
    assert len(certain) == 1
    assert "enregistrée est A" in certain[0].explanation and "annonce C" in certain[0].explanation
