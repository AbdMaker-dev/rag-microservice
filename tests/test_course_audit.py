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


SECTIONS = [
    {"id": "sec-1", "heading": "Module et argument", "content": COURS},
    {"id": "sec-2", "heading": "Autre section", "content": "Rien à signaler ici."},
]


def test_un_signalement_dit_quelle_section_corriger():
    result = asyncio.run(audit_course(text=COURS, llm=ScriptedLlm([QUADRANT]), timeout=10,
                                      num_ctx=8192, sections=SECTIONS))
    relecture = [f for f in result.findings if f.source == "relecture"][0]
    assert relecture.target == {"kind": "section", "id": "sec-1", "heading": "Module et argument"}


def test_un_passage_present_dans_deux_sections_ne_designe_rien():
    """Désigner la mauvaise section ferait corriger au prof ce qui était juste."""

    doubles = SECTIONS[:1] + [{"id": "sec-3", "heading": "Copie", "content": COURS}]
    result = asyncio.run(audit_course(text=COURS, llm=ScriptedLlm([QUADRANT]), timeout=10,
                                      num_ctx=8192, sections=doubles))
    assert [f for f in result.findings if f.source == "relecture"][0].target is None


def test_un_quiz_contradictoire_propose_la_reponse_annoncee():
    quiz = {"id": "q-1", "question": "Forme de -1 + i ?", "choices": ["a", "b", "c", "d"],
            "answer": 0, "explanation": "La réponse correcte est C) √2 e^{i3π/4}."}
    result = asyncio.run(audit_course(text="Un cours.", llm=ScriptedLlm(["### AUCUN"]),
                                      timeout=10, num_ctx=8192, quizzes=[quiz]))
    trouve = [f for f in result.findings if f.severity == "certaine"][0]
    assert trouve.suggested_answer == 2
    assert trouve.target == {"kind": "quiz", "id": "q-1", "index": 0}


def test_le_titre_du_cours_est_donne_au_correcteur():
    """Management l'envoie, et il aide : « premier quadrant » se juge mieux
    en sachant qu'on lit un cours sur les nombres complexes."""

    llm = ScriptedLlm(["### AUCUN"])
    asyncio.run(audit_course(text=COURS, llm=llm, timeout=10, num_ctx=8192,
                             title="Nombres complexes"))
    assert "Cours : Nombres complexes" in llm.exchanges[0][-1]["content"]


def test_le_statut_d_une_tache_d_un_autre_genre_ne_plante_pas():
    """20/09/2026 : management sondait ici l'audio d'une section ; la route
    répondait 500 et le professeur voyait « en cours » pour toujours."""

    from types import SimpleNamespace

    from app.api.routes_generate import _generation_status
    from app.core.jobs import JobStore

    async def scenario():
        jobs = JobStore()

        async def audio():
            return object()  # ni cours, ni bloc, ni audit

        job = jobs.submit(audio, lane="prof")
        for _ in range(50):
            if job.status in ("done", "failed"):
                break
            await asyncio.sleep(0.02)
        requete = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(jobs=jobs)))
        return await _generation_status(job.id, requete)

    reponse = asyncio.run(scenario())
    assert reponse.status == "done"
    assert reponse.warnings == ["RESULT_KIND_UNEXPECTED"]
    assert reponse.title is None
