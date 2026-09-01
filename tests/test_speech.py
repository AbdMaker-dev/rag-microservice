"""/speech : le contrat asynchrone, sans dépendre du modèle vocal.

La voix (.onnx, 60 Mo) n'est pas dans le dépôt : ces tests remplacent le
moteur par un factice et vérifient le contrat — ticket, statut, base64,
503 quand la voix manque. La synthèse réelle s'est mesurée à part (trois
échantillons écoutés, voix siwis retenue le 31/08/2026).
"""

import base64

from fastapi.testclient import TestClient

from app.core.jobs import JobStore
from app.core.speech import SpeechEngine, SpeechResult
from app.main import create_app

_TOKEN = {"X-Service-Token": "test-secret-value-of-at-least-32-chars"}


class _FakeEngine:
    available = True

    def synthesize(self, text: str) -> SpeechResult:
        # Le texte reçu doit être verbalisé : c'est le contrat de la route.
        assert "²" not in text and "##" not in text
        return SpeechResult(b"AUDIO", "mp3", 12.5, len(text))


def _client(engine) -> TestClient:
    # Pas de lifespan (il ouvrirait la base) : l'état utile se pose à la main.
    app = create_app()
    app.state.jobs = JobStore()
    app.state.speech = engine
    return TestClient(app)


def test_le_cycle_complet_ticket_puis_audio():
    client = _client(_FakeEngine())
    accepted = client.post(
        "/speech", headers=_TOKEN,
        json={"requestId": "s-1", "courseId": "cours-7",
              "text": "## p. 1\n\nx² est positif."},
    )
    assert accepted.status_code == 202
    job_id = accepted.json()["jobId"]

    # La tâche est asynchrone : on interroge comme le fera la plateforme.
    import time

    done = {}
    for _ in range(50):
        done = client.get(f"/speech/{job_id}", headers=_TOKEN).json()
        if done["status"] != "running":
            break
        time.sleep(0.1)
    assert done["status"] == "done"
    assert done["format"] == "mp3"
    assert base64.b64decode(done["audioBase64"]) == b"AUDIO"
    assert done["seconds"] == 12.5


def test_sans_voix_le_service_le_dit():
    class Missing:
        available = False

    client = _client(Missing())
    response = client.post(
        "/speech", headers=_TOKEN,
        json={"requestId": "s-2", "text": "Bonjour."},
    )
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "SPEECH_BACKEND_UNAVAILABLE"


def test_un_texte_vide_apres_verbalisation_est_refuse():
    client = _client(_FakeEngine())
    response = client.post(
        "/speech", headers=_TOKEN,
        json={"requestId": "s-3", "text": "## p. 1"},
    )
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "NOTHING_TO_SPEAK"


def test_un_job_inconnu_rend_404():
    client = _client(_FakeEngine())
    assert client.get("/speech/inconnu", headers=_TOKEN).status_code == 404


def test_le_moteur_reel_est_branche_dans_l_application():
    """app.state.llm oublié = 500 au premier appel réel — jamais deux fois."""

    import inspect

    from app import main

    source = inspect.getsource(main)
    assert "app.state.speech = SpeechEngine(" in source


def test_une_voix_absente_se_declare_indisponible():
    engine = SpeechEngine("/nulle/part/voix.onnx")
    assert engine.available is False
