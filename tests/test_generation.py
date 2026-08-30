"""L'IA a la main sur ses recherches — jamais sur le périmètre."""

from __future__ import annotations

import asyncio
import json
from typing import List

from app.config import get_settings
from app.core.generation import CourseGenerator, GenerationFailed
from app.core.retrieval import Passage
from app.models.schemas import Scope

SCOPE = Scope(country="SN", subject="maths", level="lycee", track="S",
              grade="seconde", curriculum_version="2006")


def _passage(label: str, content: str) -> Passage:
    return Passage(chunk_id=label, document_id=f"doc-{label}", title=f"Titre {label}",
                   locator=f"p. {label}", content=content, language="fr", score=0.8)


class FakeRetriever:
    def __init__(self) -> None:
        self.calls: List[dict] = []

    async def search(self, *, query, scope, limit, max_excerpt_characters,
                     course_id=None, role=None, document_ids=None):
        self.calls.append({"query": query, "course_id": course_id, "role": role})
        if role == "programme-officiel":
            return [_passage("prog", "Compétences exigibles : produit scalaire.")]
        return [_passage("supp", "Définition du produit scalaire, projeté orthogonal.")]


class ScriptedLlm:
    """Rejoue des réponses prévues ; la conversation réelle est vérifiée."""

    model = "fake"

    def __init__(self, replies: List[str]) -> None:
        self._replies = list(replies)
        self.exchanges: List[List[dict]] = []

    async def complete(self, system: str, user: str) -> str:
        raise AssertionError("la génération passe par chat()")

    async def chat(self, messages, *, timeout, num_ctx) -> str:
        self.exchanges.append(list(messages))
        return self._replies.pop(0)

    async def healthy(self) -> bool:
        return True


def _generator(llm, retriever=None):
    return CourseGenerator(llm=llm, retriever=retriever or FakeRetriever(),
                           settings=get_settings())


def test_le_cours_sort_avec_plan_citations_et_journal_des_recherches():
    plan = json.dumps({"titre": "Le produit scalaire", "sections": ["Définition"]})
    retriever = FakeRetriever()
    llm = ScriptedLlm([plan, "Le produit scalaire est défini par [S1] et cadré par [P1]."])

    course = asyncio.run(_generator(llm, retriever).generate(
        instruction="cours sur le produit scalaire", scope=SCOPE,
        course_id="cours-7", strictness="grounded"))

    assert course.title == "Le produit scalaire"
    assert course.sections[0].citations == [
        {"documentId": "doc-prog", "locator": "p. prog", "title": "Titre prog"},
        {"documentId": "doc-supp", "locator": "p. supp", "title": "Titre supp"},
    ]
    # Le journal dit comment le cours a été construit : cadre + section.
    assert [c["role"] for c in retriever.calls] == ["programme-officiel", "support-cours"]


def test_le_modele_peut_demander_une_recherche_et_le_perimetre_reste_verrouille():
    plan = json.dumps({"titre": "T", "sections": ["Propriétés"]})
    demande = json.dumps({"chercher": {"question": "identités remarquables vecteurs",
                                       "nature": "support-cours"}})
    retriever = FakeRetriever()
    llm = ScriptedLlm([plan, demande, "Les identités [S2] complètent la définition [S1]."])

    course = asyncio.run(_generator(llm, retriever).generate(
        instruction="cours", scope=SCOPE, course_id="cours-7", strictness="grounded"))

    extra = retriever.calls[-1]
    assert extra["query"] == "identités remarquables vecteurs"
    # La recherche demandée par le modèle reste verrouillée sur SON cours.
    assert extra["course_id"] == "cours-7"
    assert course.queries[-1]["demandeParLeModele"] is True
    assert "Les identités" in course.sections[0].text


def test_un_ajout_en_mode_grounded_est_signale_jamais_fondu():
    plan = json.dumps({"titre": "T", "sections": ["Définition"]})
    texte = "Définition [S1]. ⟦AJOUT⟧Une anecdote inventée.⟦/AJOUT⟧"
    llm = ScriptedLlm([plan, texte])

    course = asyncio.run(_generator(llm).generate(
        instruction="cours", scope=SCOPE, course_id="c", strictness="grounded"))

    assert course.sections[0].has_additions
    assert "ADDITIONS_IN_GROUNDED_MODE" in course.warnings
    assert "⟦AJOUT⟧" in course.sections[0].text  # visible pour le professeur


def test_un_plan_illisible_echoue_clairement():
    llm = ScriptedLlm(["Je propose plutôt un poème."])

    try:
        asyncio.run(_generator(llm).generate(
            instruction="cours", scope=SCOPE, course_id="c", strictness="grounded"))
    except GenerationFailed as error:
        assert "plan" in str(error)
    else:
        raise AssertionError("un plan illisible doit échouer, pas improviser")


def test_le_plafond_de_recherches_borne_le_modele():
    plan = json.dumps({"titre": "T", "sections": ["A"]})
    demande = json.dumps({"chercher": {"question": "encore", "nature": "support-cours"}})
    retriever = FakeRetriever()
    llm = ScriptedLlm([plan, demande, demande, demande, "Fini [S1]."])

    settings = get_settings().model_copy(update={"generation_max_queries": 1})
    generator = CourseGenerator(llm=llm, retriever=retriever, settings=settings)
    course = asyncio.run(generator.generate(
        instruction="cours", scope=SCOPE, course_id="c", strictness="grounded"))

    solicited = [c for c in course.queries if c["demandeParLeModele"]]
    assert len(solicited) == 1, "au-delà du plafond, la recherche est refusée"
    assert course.sections[0].text.startswith("Fini")
