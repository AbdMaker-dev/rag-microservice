"""Lawal : les règles qui protègent l'élève, vérifiées une à une.

Le tuteur parle à des enfants : pas de réponse inventée (fallback honnête
sans modèle quand rien n'est trouvé), priorité au cours publié, recherches
traçées, périmètre jamais discutable par le modèle.
"""

import asyncio
import json

import pytest

from app.config import Settings
from app.core.retrieval import Passage
from app.core.tutor import Tutor, TutorFailed
from app.models.schemas import Scope


def _settings() -> Settings:
    return Settings(
        service_shared_secret="test-secret-value-of-at-least-32-chars",
        database_url="postgresql://x:y@localhost/z",
    )


def _scope() -> Scope:
    return Scope(
        country="SN", subject="mathematiques", level="secondaire",
        track="S2", grade="terminale", curriculum_version="2024",
    )


def _passage(chunk="c1", doc="doc-1", title="Cours — produit scalaire"):
    return Passage(
        chunk_id=chunk, document_id=doc, title=title,
        locator="p. 3 · §2", content="Le produit scalaire est nul si...",
        language="fr", score=0.8,
    )


class _Retriever:
    def __init__(self, by_role):
        self.by_role = by_role
        self.calls = []

    async def search(self, *, query, scope, limit, max_excerpt_characters,
                     course_id=None, role=None, document_ids=None):
        self.calls.append({"query": query, "role": role, "courseId": course_id})
        return self.by_role.get(role, [])


class _Llm:
    def __init__(self, replies):
        self.replies = list(replies)
        self.messages_seen = []

    async def chat(self, messages, **kwargs):
        self.messages_seen.append(messages)
        return self.replies.pop(0)


def test_le_cours_publie_est_consulte_en_premier():
    retriever = _Retriever({"cours-publie": [_passage()]})
    reply = json.dumps({"reponse": "L'idée [S1]...", "verification": "Et toi ?",
                        "conceptes": ["produit scalaire"]})
    tutor = Tutor(llm=_Llm([reply]), retriever=retriever, settings=_settings())
    answer = asyncio.run(tutor.answer(
        question="C'est quoi le produit scalaire ?",
        scope=_scope(), course_id="cours-7",
    ))
    assert retriever.calls[0]["role"] == "cours-publie"
    assert retriever.calls[0]["courseId"] == "cours-7"
    assert answer.text.startswith("L'idée")
    assert answer.check == "Et toi ?"
    assert answer.concepts == ["produit scalaire"]
    assert answer.citations[0]["label"] == "S1"
    assert answer.queries[0]["nature"] == "cours-publie"


def test_cours_muet_les_supports_prennent_le_relais():
    # Le cours est un résumé court : quand il ne couvre pas la question,
    # les supports validés du professeur sont consultés d'office.
    retriever = _Retriever({"support-cours": [_passage(title="Support ch.2")]})
    reply = json.dumps({"reponse": "Dans le support [S1]...", "verification": "?"})
    tutor = Tutor(llm=_Llm([reply]), retriever=retriever, settings=_settings())
    answer = asyncio.run(tutor.answer(
        question="Explique la norme", scope=_scope(), course_id="cours-7",
    ))
    assert [c["role"] for c in retriever.calls] == ["cours-publie", "support-cours"]
    assert "NO_PUBLISHED_COURSE_CONTENT" in answer.warnings


def test_sans_aucune_source_lawal_est_honnete_sans_modele():
    llm = _Llm([])  # toute sollicitation du modèle ferait échouer le pop
    tutor = Tutor(llm=llm, retriever=_Retriever({}), settings=_settings())
    answer = asyncio.run(tutor.answer(
        question="Question hors programme", scope=_scope(), course_id="cours-7",
    ))
    assert "INSUFFICIENT_EVIDENCE" in answer.warnings
    assert "professeur" in answer.text
    assert llm.messages_seen == []


def test_le_modele_peut_demander_une_recherche_de_plus():
    retriever = _Retriever({
        "cours-publie": [_passage()],
        "programme-officiel": [_passage(chunk="c2", doc="prog", title="Programme")],
    })
    ask = json.dumps({"chercher": {"question": "orthogonalité",
                                   "nature": "programme-officiel"}})
    final = json.dumps({"reponse": "Avec [S2]...", "verification": "?"})
    tutor = Tutor(llm=_Llm([ask, final]), retriever=retriever, settings=_settings())
    answer = asyncio.run(tutor.answer(
        question="Pourquoi orthogonal ?", scope=_scope(), course_id="cours-7",
    ))
    natures = [q["nature"] for q in answer.queries]
    assert natures == ["cours-publie", "programme-officiel"]
    assert answer.queries[1]["demandeParLeModele"] is True
    # Le programme officiel se cherche sans courseId : il est commun.
    assert retriever.calls[1]["courseId"] is None
    assert len(answer.citations) == 2


def test_l_historique_du_fil_revient_dans_la_conversation():
    retriever = _Retriever({"cours-publie": [_passage()]})
    llm = _Llm([json.dumps({"reponse": "Suite [S1].", "verification": "?"})])
    tutor = Tutor(llm=llm, retriever=retriever, settings=_settings())
    asyncio.run(tutor.answer(
        question="Et ensuite ?", scope=_scope(), course_id="cours-7",
        history=[{"role": "eleve", "content": "C'est quoi un vecteur ?"},
                 {"role": "lawal", "content": "Une flèche qui..."}],
    ))
    conversation = llm.messages_seen[0]
    assert conversation[1] == {"role": "user", "content": "C'est quoi un vecteur ?"}
    assert conversation[2] == {"role": "assistant", "content": "Une flèche qui..."}


def test_reponse_hors_format_repetee_finit_en_echec_clair():
    retriever = _Retriever({"cours-publie": [_passage()]})
    llm = _Llm(["du texte libre"] * 10)
    tutor = Tutor(llm=llm, retriever=retriever, settings=_settings())
    with pytest.raises(TutorFailed):
        asyncio.run(tutor.answer(
            question="?", scope=_scope(), course_id="cours-7",
        ))


def test_les_regles_pedagogiques_sont_dans_le_prompt():
    from app.core.tutor import _SYSTEM

    assert "JAMAIS la solution" in _SYSTEM
    assert "demander au professeur" in _SYSTEM
    assert "question qui vérifie" in _SYSTEM
    assert "Adapte ton langage à la classe" in _SYSTEM


def test_le_tuteur_est_branche_dans_l_application():
    import inspect

    from app import main

    assert "app.state.tutor = Tutor(" in inspect.getsource(main)
