"""Le « chat » des blocs : réviser un exercice, une question, un résumé.

Le document, le plan et les sections ont le leur depuis le début. Les trois
blocs n'en avaient pas : un quiz dont une réponse est fausse arrivait intact
jusqu'à l'élève, et le professeur ne pouvait que tout refaire ou corriger à
la main. Demandé par Alioune le 14/09/2026 — « on fait la même chose que
les autres » — et exercice par exercice, comme la relecture d'un document.
"""

import asyncio

import pytest

from app.config import get_settings
from app.core.generation import CourseGenerator, GenerationFailed
from tests.test_generation import SCOPE, FakeRetriever, ScriptedLlm

COURS = "La suite est croissante et majorée, donc elle converge. " * 5

QUIZ = [
    {
        "question": "Forme trigonométrique de z = -1 + i ?",
        "choices": ["2e^{i3pi/4}", "racine(2)e^{-ipi/4}", "racine(2)e^{i5pi/4}", "2"],
        "answer": 0,
        "explanation": "Le module vaut 2.",
    },
    {
        "question": "Que vaut la limite ?",
        "choices": ["0", "2", "4", "1"],
        "answer": 2,
        "explanation": "Le point fixe.",
    },
]

EXERCICES = [
    {"statement": "Calculer u_1.", "solution": "On applique.", "difficulty": "facile"},
    {"statement": "Montrer la convergence.", "solution": "Croissante et majorée.",
     "difficulty": "difficile"},
]


def _gen(replies):
    return CourseGenerator(llm=ScriptedLlm(replies), retriever=FakeRetriever(),
                           settings=get_settings())


def test_une_question_fausse_se_corrige_sans_toucher_aux_autres():
    # Le cas réel : « module = 2 » au lieu de racine de 2. Le professeur le
    # dit, l'IA corrige CETTE question — et la seconde ne bouge pas d'un
    # caractère, ce qui est la promesse faite au professeur.
    corrigee = ("### QUESTION\nForme trigonométrique de z = -1 + i ?\n"
                "- A) racine(2)e^{i3pi/4}\n- B) 2e^{i3pi/4}\n"
                "- C) racine(2)e^{-ipi/4}\n- D) 2\n"
                "### RÉPONSE A\n### EXPLICATION\nLe module vaut racine de 2.")
    llm = ScriptedLlm([corrigee])

    draft = asyncio.run(CourseGenerator(
        llm=llm, retriever=FakeRetriever(), settings=get_settings()
    ).discuss_block(
        kind="quiz", text=COURS, scope=SCOPE, current_items=QUIZ, target_index=0,
        request="La question 1 est fausse : le module de -1+i vaut racine de 2."))

    assert len(draft.quiz) == 2
    assert draft.quiz[0]["choices"][0] == "racine(2)e^{i3pi/4}"
    assert draft.quiz[0]["explanation"] == "Le module vaut racine de 2."
    assert draft.quiz[1] == QUIZ[1]  # intacte, mot pour mot
    # Le modèle a relu SA question, pas tout le quiz.
    demande = llm.exchanges[0][-1]["content"]
    assert "Forme trigonométrique" in demande
    assert "Que vaut la limite" not in demande
    assert "cette QUESTION" in demande


def test_un_exercice_se_revise_de_la_meme_facon():
    revise = ("### EXERCICE (moyen)\nCalculer $u_1$ puis $u_2$.\n"
              "### CORRIGÉ\nOn applique deux fois la relation.")
    llm = ScriptedLlm([revise])

    draft = asyncio.run(CourseGenerator(
        llm=llm, retriever=FakeRetriever(), settings=get_settings()
    ).discuss_block(
        kind="exercices", text=COURS, scope=SCOPE, current_items=EXERCICES,
        target_index=0, request="Demande aussi u_2."))

    assert draft.exercises[0]["statement"] == "Calculer $u_1$ puis $u_2$."
    assert draft.exercises[0]["difficulty"] == "moyen"
    assert draft.exercises[1] == EXERCICES[1]
    assert "cet EXERCICE" in llm.exchanges[0][-1]["content"]


def test_sans_item_vise_la_consigne_porte_sur_tout_le_bloc():
    # « Ils sont tous trop faciles » ne vise aucun exercice en particulier.
    deux = ("### EXERCICE (moyen)\nA\n### CORRIGÉ\nca\n"
            "### EXERCICE (difficile)\nB\n### CORRIGÉ\ncb")
    llm = ScriptedLlm([deux])

    draft = asyncio.run(CourseGenerator(
        llm=llm, retriever=FakeRetriever(), settings=get_settings()
    ).discuss_block(
        kind="exercices", text=COURS, scope=SCOPE, current_items=EXERCICES,
        request="Ils sont tous trop faciles, relève le niveau."))

    assert [e["difficulty"] for e in draft.exercises] == ["moyen", "difficile"]
    demande = llm.exchanges[0][-1]["content"]
    assert "ces EXERCICES" in demande
    # Tout le bloc lui est montré, puisque la consigne porte sur tout.
    assert "Calculer u_1." in demande and "Montrer la convergence." in demande


def test_le_resume_se_revise_aussi():
    llm = ScriptedLlm(["La suite converge vers 4, par convergence monotone."])

    draft = asyncio.run(CourseGenerator(
        llm=llm, retriever=FakeRetriever(), settings=get_settings()
    ).discuss_block(
        kind="resume", text=COURS, scope=SCOPE,
        current_summary="La suite converge.",
        request="Précise vers quoi et par quel théorème."))

    assert draft.summary == "La suite converge vers 4, par convergence monotone."
    assert "ce RÉSUMÉ" in llm.exchanges[0][-1]["content"]


def test_l_historique_de_la_conversation_accompagne_la_consigne():
    # Sans lui, chaque tour serait amnésique : « comme je t'ai dit, garde un
    # énoncé court » ne voudrait rien dire au deuxième tour.
    llm = ScriptedLlm(["### EXERCICE (facile)\nA\n### CORRIGÉ\nc"])

    asyncio.run(CourseGenerator(
        llm=llm, retriever=FakeRetriever(), settings=get_settings()
    ).discuss_block(
        kind="exercices", text=COURS, scope=SCOPE, current_items=EXERCICES,
        target_index=0, request="Encore plus court.",
        history=[{"author": "prof", "message": "garde un énoncé court"},
                 {"author": "ia", "message": "voici une version raccourcie"}]))

    demande = llm.exchanges[0][-1]["content"]
    assert "garde un énoncé court" in demande


def test_un_item_qui_n_existe_pas_echoue_clairement():
    llm = ScriptedLlm(["### EXERCICE (facile)\nA\n### CORRIGÉ\nc"])

    with pytest.raises(GenerationFailed) as erreur:
        asyncio.run(CourseGenerator(
            llm=llm, retriever=FakeRetriever(), settings=get_settings()
        ).discuss_block(
            kind="exercices", text=COURS, scope=SCOPE, current_items=EXERCICES,
            target_index=7, request="Corrige."))

    assert "n'existe pas" in str(erreur.value)
    assert llm.exchanges == []  # on n'a pas payé le modèle pour rien


def test_la_revision_parle_le_meme_format_que_la_redaction():
    from app.core.generation import format_du_bloc

    llm = ScriptedLlm(["### EXERCICE (facile)\nA\n### CORRIGÉ\nc"])
    asyncio.run(CourseGenerator(
        llm=llm, retriever=FakeRetriever(), settings=get_settings()
    ).discuss_block(
        kind="exercices", text=COURS, scope=SCOPE, current_items=EXERCICES,
        target_index=0, request="Corrige."))

    # Une seule description du format, pour la rédaction comme pour la
    # révision : deux copies finiraient par diverger, et le lecteur balisé
    # ne reconnaîtrait plus ce que la révision rend.
    assert format_du_bloc("exercices") in llm.exchanges[0][0]["content"]
