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

    async def chat(self, messages, *, timeout, num_ctx, num_predict) -> str:
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


def test_le_modele_sait_pour_qui_il_ecrit():
    """Le périmètre filtrait les recherches mais n'était jamais énoncé.

    Sans cette ligne, l'IA ignore qu'elle écrit pour une seconde S — et le
    pays était écrit en dur « sénégalais », faux dès le premier cours malien.
    """

    plan = json.dumps({"titre": "T", "sections": ["Définition"]})
    llm = ScriptedLlm([plan, "Texte [S1]."])

    asyncio.run(_generator(llm).generate(
        instruction="cours", scope=SCOPE, course_id="c", strictness="grounded"))

    for exchange in llm.exchanges:
        system = exchange[0]["content"]
        assert "seconde" in system and "série S" in system and "SN" in system
        assert "sénégalais" not in system


def _course_sections():
    return [
        {"heading": "Définition", "text": "La définition originale [S1].",
         "citations": [{"documentId": "doc-a", "locator": "p. 3"}]},
        {"heading": "Propriétés", "text": "Les propriétés actuelles [S2]."},
        {"heading": "Anecdote", "text": "Une longue anecdote."},
    ]


def test_la_revision_ne_touche_que_ce_que_le_prof_demande():
    """« Revois les propriétés et retire l'anecdote » : la définition ne doit
    pas bouger d'un caractère — sur CPU, chaque section réécrite coûte des
    minutes."""

    plan = json.dumps({"operations": [
        {"action": "reecrire", "section": "Propriétés", "consigne": "plus d'exemples"},
        {"action": "supprimer", "section": "Anecdote"},
    ]})
    llm = ScriptedLlm([plan, "Les propriétés révisées avec exemples [S1]."])

    course = asyncio.run(_generator(llm).adjust(
        title="Le produit scalaire", sections=_course_sections(),
        request="revois les propriétés avec des exemples, retire l'anecdote",
        instruction="cours produit scalaire", scope=SCOPE,
        course_id="cours-7", strictness="grounded"))

    headings = [s.heading for s in course.sections]
    assert headings == ["Définition", "Propriétés"]
    # Conservée mot pour mot, citations comprises.
    assert course.sections[0].text == "La définition originale [S1]."
    assert course.sections[0].citations == [{"documentId": "doc-a", "locator": "p. 3"}]
    assert "révisées" in course.sections[1].text


def test_la_revision_peut_ajouter_une_section():
    plan = json.dumps({"operations": [
        {"action": "ajouter", "section": "Exercices", "consigne": "trois exercices"},
    ]})
    llm = ScriptedLlm([plan, "Exercice 1 fondé sur [S1]."])

    course = asyncio.run(_generator(llm).adjust(
        title="T", sections=_course_sections(), request="ajoute des exercices",
        instruction="cours", scope=SCOPE, course_id="c", strictness="grounded"))

    assert [s.heading for s in course.sections][-1] == "Exercices"
    assert len(course.sections) == 4


def test_une_revision_qui_ne_vise_rien_echoue_clairement():
    llm = ScriptedLlm([json.dumps({"operations": []})])

    try:
        asyncio.run(_generator(llm).adjust(
            title="T", sections=_course_sections(), request="euh",
            instruction="cours", scope=SCOPE, course_id="c", strictness="grounded"))
    except GenerationFailed as error:
        assert "révision" in str(error)
    else:
        raise AssertionError("un plan de révision vide doit échouer")


def test_les_consignes_precedentes_suivent_la_conversation():
    """« Comme je t'ai dit, garde un ton simple » doit encore compter.

    L'historique vient de la table de discussion côté plateforme ; le service
    le lit à chaque tour et ne le stocke jamais.
    """

    plan = json.dumps({"operations": [
        {"action": "reecrire", "section": "Définition", "consigne": "simplifier"}]})
    llm = ScriptedLlm([plan, "Définition simple [S1]."])

    asyncio.run(_generator(llm).adjust(
        title="T", sections=_course_sections(), request="simplifie encore",
        instruction="cours", scope=SCOPE, course_id="c", strictness="grounded",
        history=[{"author": "prof", "message": "garde un ton simple"},
                 {"author": "prof", "message": "pas de jargon"}]))

    premier_echange = llm.exchanges[0][1]["content"]
    assert "garde un ton simple" in premier_echange
    assert "pas de jargon" in premier_echange


def test_l_application_branche_le_redacteur():
    """/generate a levé un 500 au premier appel réel : app.state.llm
    n'existait pas. Le module llm.py existait, rien ne l'instanciait — et les
    tests exerçaient le générateur en direct, jamais la route dans l'app."""

    import inspect

    from app import main

    source = inspect.getsource(main)
    assert "app.state.llm = build_llm_provider" in source


def test_le_plan_se_propose_avec_resumes_sans_rediger_une_ligne():
    """Étape 1 du progressif : le prof juge le plan avant de payer le contenu."""

    plan = json.dumps({"titre": "Le produit scalaire",
        "description": "Définition, propriétés et applications du produit scalaire en seconde S.",
        "sections": [
        {"titre": "Définition", "description": "J'expliquerai le projeté orthogonal et la notation du produit scalaire."},
        {"titre": "Propriétés", "description": "Je démontrerai la symétrie et la bilinéarité."}]})
    retriever = FakeRetriever()
    llm = ScriptedLlm([plan])

    draft = asyncio.run(_generator(llm, retriever).draft_plan(
        instruction="cours produit scalaire", scope=SCOPE, course_id="c"))

    assert draft.title == "Le produit scalaire"
    # La description du plan — colonne de la table Plan côté plateforme —
    # est rédigée par l'IA en même temps que le plan.
    assert draft.description.startswith("Définition, propriétés")
    assert [s.heading for s in draft.sections] == ["Définition", "Propriétés"]
    assert draft.sections[0].description.startswith("J'expliquerai le projeté")
    # Un seul appel modèle, une seule recherche : le plan ne coûte presque rien.
    assert len(llm.exchanges) == 1
    assert retriever.calls[0]["role"] == "programme-officiel"


def test_le_plan_se_revise_en_conversation():
    revise = json.dumps({"titre": "T", "sections": [
        {"titre": "Définition", "description": ""},
        {"titre": "Exercices", "description": "Je proposerai trois applications."}]})
    llm = ScriptedLlm([revise])

    draft = asyncio.run(_generator(llm).draft_plan(
        instruction="cours", scope=SCOPE, course_id="c",
        current_plan={"title": "T", "sections": [{"heading": "Définition"}]},
        request="ajoute une partie exercices",
        history=[{"author": "prof", "message": "reste simple"}]))

    contenu = llm.exchanges[0][1]["content"]
    assert "Plan actuel" in contenu and "ajoute une partie exercices" in contenu
    assert "reste simple" in contenu
    assert [s.heading for s in draft.sections][-1] == "Exercices"


def test_une_section_seule_recoit_le_plan_et_les_resumes_valides():
    """Des sections rédigées séparément doivent rester UN cours."""

    llm = ScriptedLlm(["Contenu de la section [S1], sans répéter la définition."])
    retriever = FakeRetriever()

    result = asyncio.run(_generator(llm, retriever).write_one_section(
        heading="Propriétés", instruction="cours produit scalaire",
        scope=SCOPE, course_id="cours-7", strictness="grounded",
        plan_headings=["Définition", "Propriétés", "Exercices"],
        previous_summaries=[{"heading": "Définition",
                             "description": "Le projeté orthogonal est posé."}]))

    contenu = llm.exchanges[0][1]["content"]
    assert "Définition | Propriétés | Exercices" in contenu
    assert "Le projeté orthogonal est posé." in contenu
    assert result.sections[0].heading == "Propriétés"
    # Les recherches de la section restent verrouillées sur le cours.
    support = [c for c in retriever.calls if c["role"] == "support-cours"][0]
    assert support["course_id"] == "cours-7"


def test_une_section_se_revise_avec_sa_version_actuelle():
    llm = ScriptedLlm(["Version révisée, plus courte [S1]."])

    result = asyncio.run(_generator(llm).write_one_section(
        heading="Propriétés", instruction="cours", scope=SCOPE,
        course_id="c", strictness="grounded", plan_headings=["Propriétés"],
        current_text="Une version actuelle beaucoup trop longue.",
        request="raccourcis de moitié",
        history=[{"author": "prof", "message": "ton simple"}]))

    contenu = llm.exchanges[0][1]["content"]
    assert "beaucoup trop longue" in contenu
    assert "raccourcis de moitié" in contenu and "ton simple" in contenu
    assert "révisée" in result.sections[0].text
