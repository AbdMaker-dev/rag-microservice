"""Le démarrage complet de l'application, joué — plus jamais relu.

Le câblage du lifespan a déjà cassé la production deux fois : app.state.llm
jamais instancié, puis le tuteur construit AVANT le retriever qu'il
consomme (AttributeError au démarrage, conteneur en Exited). Les tests
d'inspection de source vérifiaient la présence des lignes, pas leur ordre.
Celui-ci exécute le lifespan entier, base de données simulée : une erreur
de câblage échoue ici, pas au déploiement.
"""

from fastapi.testclient import TestClient

from app import main


class _FakeDatabase:
    def __init__(self, settings) -> None:
        self.pool = None

    async def connect(self) -> None:
        pass

    async def migrate(self) -> None:
        pass

    async def close(self) -> None:
        pass


def test_le_lifespan_demarre_et_cable_chaque_dependance(monkeypatch):
    monkeypatch.setattr(main, "Database", _FakeDatabase)
    app = main.create_app()
    with TestClient(app):
        for name in (
            "settings", "http", "database", "repository", "embeddings",
            "jobs", "llm", "speech", "retriever", "tutor",
        ):
            assert hasattr(app.state, name), f"app.state.{name} manquant"
        # Le tuteur consomme LE retriever câblé — pas un autre.
        assert app.state.tutor._retriever is app.state.retriever
