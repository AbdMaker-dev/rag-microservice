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


def _settings(**overrides) -> Settings:
    return Settings(
        service_shared_secret="test-secret-value-of-at-least-32-chars",
        database_url="postgresql://x:y@localhost/z",
        **overrides,
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
        # Une fois les réponses scriptées épuisées, c'est la RELECTURE qui
        # parle — et par défaut elle ne trouve rien à redire.
        return self.replies.pop(0) if self.replies else "### VERDICT\nOK"


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


def test_sans_aucune_source_lawal_repond_de_memoire_et_le_dit():
    """Décision d'Alioune (15/09/2026) : si le cours ne contient pas la
    réponse, Lawal répond avec ses connaissances — mais le dit à l'élève."""

    reply = json.dumps({"reponse": "Ce n'est pas dans ton cours, mais voici ce que je sais : "
                                   "dans un triangle rectangle, a² + b² = c².",
                        "verification": "?"})
    llm = _Llm([reply])
    tutor = Tutor(llm=llm, retriever=_Retriever({}), settings=_settings())
    answer = asyncio.run(tutor.answer(
        question="C'est quoi Pythagore ?", scope=_scope(), course_id="cours-7",
    ))
    assert "INSUFFICIENT_EVIDENCE" in answer.warnings
    assert "ANSWER_OUTSIDE_COURSE" in answer.warnings
    assert answer.citations == []
    assert "Aucun extrait du cours ne couvre" in llm.messages_seen[0][-1]["content"]


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


def test_une_question_de_section_se_cherche_dans_son_contexte():
    # « La norme » ne veut pas dire la même chose selon le chapitre : la
    # section d'où parle l'élève guide la recherche ET la réponse.
    retriever = _Retriever({"cours-publie": [_passage()]})
    llm = _Llm([json.dumps({"reponse": "Ici [S1].", "verification": "?"})])
    tutor = Tutor(llm=llm, retriever=retriever, settings=_settings())
    asyncio.run(tutor.answer(
        question="C'est quoi la norme ?", scope=_scope(), course_id="cours-7",
        section_heading="II-2) Produit scalaire et orthogonalité",
    ))
    assert retriever.calls[0]["query"] == (
        "II-2) Produit scalaire et orthogonalité — C'est quoi la norme ?"
    )
    assert "L'élève lit la section « II-2) Produit scalaire" in (
        llm.messages_seen[0][-1]["content"]
    )


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
    # Un rappel de contexte dans le message de l'élève, pas des tours rejoués :
    # rejoués, ils faisaient répondre qwen à la question d'avant (15/09/2026).
    assert [m["role"] for m in conversation] == ["system", "user"]
    last = conversation[-1]["content"]
    assert "L'élève a demandé : C'est quoi un vecteur ?" in last
    assert "Tu as déjà expliqué : Une flèche qui..." in last
    assert last.index("Rappel de la conversation") < last.index("Et ensuite ?")


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
    assert "question qui vérifie" in _SYSTEM
    assert "Adapte ton langage à la classe" in _SYSTEM
    assert "DERNIÈRE question" in _SYSTEM
    assert "Le cours est ta source principale" in _SYSTEM
    assert "Ce n'est pas dans ton cours, mais voici ce que je sais" in _SYSTEM
    assert "[S1] indique que" in _SYSTEM  # nommé pour être interdit


def test_le_tuteur_est_branche_dans_l_application():
    import inspect

    from app import main

    assert "app.state.tutor = Tutor(" in inspect.getsource(main)


def test_une_bonne_reponse_en_prose_est_acceptee_apres_une_relance():
    """Constaté au premier test réel : qwen expliquait bien, mais en prose —
    et l'élève recevait un échec après 4 minutes. Une relance pour le
    format, puis le texte libre est accepté tel quel."""

    retriever = _Retriever({"cours-publie": [_passage()]})
    prose = ("Le centre d'une similitude directe est l'unique point invariant. "
             "Pour le trouver, on résout z = az + b, ce qui donne ω = b/(1−a).")
    llm = _Llm([prose, prose])
    tutor = Tutor(llm=llm, retriever=retriever, settings=_settings())
    answer = asyncio.run(tutor.answer(
        question="Comment trouver le centre ?", scope=_scope(), course_id="cours-7",
    ))
    assert "point invariant" in answer.text
    assert "TUTOR_PLAIN_TEXT" in answer.warnings
    assert answer.check == ""
    # la relance a bien eu lieu : deux appels au modèle
    assert len(llm.messages_seen) == 2


def test_deux_objets_json_cote_a_cote_sont_fusionnes():
    """Vu au premier test réel : {"reponse": …} PUIS {"verification": …} —
    l'élève recevait les accolades brutes. Les blocs se fusionnent."""

    retriever = _Retriever({"cours-publie": [_passage()]})
    raw = ('{"reponse": "Le centre est le point invariant [S1]."}\n\n'
           '{"verification": "Sauras-tu le retrouver ?"}')
    tutor = Tutor(llm=_Llm([raw]), retriever=retriever, settings=_settings())
    answer = asyncio.run(tutor.answer(
        question="Le centre ?", scope=_scope(), course_id="cours-7",
    ))
    assert answer.text == "Le centre est le point invariant [S1]."
    assert answer.check == "Sauras-tu le retrouver ?"
    assert "TUTOR_PLAIN_TEXT" not in answer.warnings


def test_la_route_rend_la_reponse_avec_la_source_de_chaque_citation():
    """Constaté en production le 15/09/2026 : chaque citation porte
    « source » (cahier ou contenu validé), mais le contrat de sortie ne le
    déclarait pas — GET /answer/{job} répondait 500 sur TOUTE réponse finie,
    et l'élève attendait Lawal indéfiniment. On passe par la vraie route."""

    import time

    from fastapi.testclient import TestClient

    from app.core.jobs import JobStore
    from app.core.tutor import TutorAnswer, _citations
    from app.main import create_app

    class _Tuteur:
        async def answer(self, **kwargs):
            return TutorAnswer(
                text="Le module de a [S1].", check="Et l'argument ?",
                concepts=["similitude"],
                citations=_citations([_passage(chunk="c1"), _passage(chunk="n1")], {"n1"}),
                queries=[], warnings=[],
            )

    app = create_app()
    app.state.jobs = JobStore()
    app.state.tutor = _Tuteur()
    client = TestClient(app)
    token = {"X-Service-Token": "test-secret-value-of-at-least-32-chars"}
    accepted = client.post("/answer", headers=token, json={
        "requestId": "a-1", "courseId": "cours-7", "question": "Pourquoi ?",
        "scope": {"country": "SN", "subject": "maths", "level": "secondaire",
                  "track": "S2", "grade": "terminale", "curriculumVersion": "2006"},
    })
    assert accepted.status_code == 202
    job = accepted.json()["jobId"]
    for _ in range(50):
        response = client.get(f"/answer/{job}", headers=token)
        assert response.status_code == 200, response.text
        if response.json()["status"] == "done":
            break
        time.sleep(0.1)
    body = response.json()
    assert body["status"] == "done"
    assert [c["source"] for c in body["citations"]] == ["valide", "cahier"]


CENTRE = ("Le centre est le point invariant : si \\( a \\neq 1 \\), "
          "\\[ \\omega = \\frac{b}{1-a} \\] [S1]")


def test_une_reponse_recopiee_du_fil_est_relancee_sur_la_nouvelle_question():
    """Constaté le 15/09/2026 : à « comment reconnaît-on une similitude
    directe ? », Lawal a recopié sa réponse précédente sur le centre."""

    retriever = _Retriever({"cours-publie": [_passage()]})
    copie = json.dumps({"reponse": CENTRE + " Par exemple z' = (1+i)z + 3.", "verification": "?"})
    nouvelle = json.dumps({"reponse": "On la reconnaît à la forme z' = az + b, avec a non nul [S1].",
                           "verification": "?"})
    llm = _Llm([copie, nouvelle])
    tutor = Tutor(llm=llm, retriever=retriever, settings=_settings())
    answer = asyncio.run(tutor.answer(
        question="Comment reconnaît-on une similitude directe ?", scope=_scope(),
        course_id="cours-7",
        history=[{"role": "eleve", "content": "Le centre ?"}, {"role": "lawal", "content": CENTRE}],
    ))
    assert answer.text.startswith("On la reconnaît")
    assert "TUTOR_REPETITION_RETRIED" in answer.warnings
    assert "AUTRE question" in llm.messages_seen[1][-1]["content"]


def test_une_longue_reponse_passee_n_est_rappelee_que_par_son_debut():
    from app.core.tutor import _rappel_du_fil

    rappel = _rappel_du_fil([
        {"role": "lawal", "content": "x" * 1000},
        {"role": "eleve", "content": "Et ensuite ?"},
        {"role": "lawal", "content": "y" * 1000},
    ])
    assert "x" * 300 + "…" in rappel
    assert "x" * 301 not in rappel
    # la DERNIÈRE réponse reste entière : c'est elle que l'élève conteste
    assert "y" * 1000 in rappel


def test_l_erreur_contestee_reste_visible_dans_le_rappel():
    """Note d'évaluation du 16/09/2026 : « (1 ; 3) » pour 1 + i√3, écrit
    en fin de réponse — coupé à 300 caractères, Lawal ne pouvait pas le voir
    quand l'élève le contestait."""

    from app.core.tutor import _rappel_du_fil

    longue = "On calcule pas à pas. " * 20 + "Les coordonnées du point sont donc (1, 3)."
    rappel = _rappel_du_fil([{"role": "eleve", "content": "Calcule 2e^{iπ/3}"},
                             {"role": "lawal", "content": longue}])
    assert "(1, 3)" in rappel
    assert "corrige-le si l'élève y signale une erreur" in rappel


@pytest.mark.parametrize("question", [
    "Tu t'es trompé, c'est pas (1,3)",
    "C'est faux ton résultat",
    "Vérifie ta réponse stp",
])
def test_une_erreur_signalee_est_nommee_au_modele(question):
    retriever = _Retriever({"cours-publie": [_passage()]})
    llm = _Llm(["### RÉPONSE\nJ'ai fait une erreur : (1 ; √3).\n### VÉRIFICATION\n?"])
    tutor = Tutor(llm=llm, retriever=retriever, settings=_settings())
    asyncio.run(tutor.answer(question=question, scope=_scope(), course_id="cours-7"))
    assert "L'élève signale une erreur : recalcule" in llm.messages_seen[0][-1]["content"]


@pytest.mark.parametrize("question", [
    "Je comprends toujours pas",
    "Explique autrement",
    "je suis perdu",
])
def test_une_incomprehension_demande_un_autre_angle(question):
    retriever = _Retriever({"cours-publie": [_passage()]})
    llm = _Llm(["### RÉPONSE\nPrenons z = 1.\n### VÉRIFICATION\n?"])
    tutor = Tutor(llm=llm, retriever=retriever, settings=_settings())
    asyncio.run(tutor.answer(question=question, scope=_scope(), course_id="cours-7"))
    last = llm.messages_seen[0][-1]["content"]
    assert "ne répète pas ta définition" in last
    assert "L'élève signale une erreur" not in last


def test_une_question_ordinaire_ne_porte_aucune_consigne_de_situation():
    retriever = _Retriever({"cours-publie": [_passage()]})
    llm = _Llm(["### RÉPONSE\nLe module.\n### VÉRIFICATION\n?"])
    tutor = Tutor(llm=llm, retriever=retriever, settings=_settings())
    asyncio.run(tutor.answer(question="Comment calculer le module de 3+4i ?",
                             scope=_scope(), course_id="cours-7"))
    last = llm.messages_seen[0][-1]["content"]
    assert "signale une erreur" not in last and "ne répète pas ta définition" not in last


def test_les_regles_de_correction_et_d_incomprehension_sont_dans_le_prompt():
    from app.core.tutor import _SYSTEM

    assert "J'ai fait une erreur :" in _SYSTEM
    assert "ne redis PAS la même définition" in _SYSTEM
    assert "1 + i√3 donne (1 ; √3)" in _SYSTEM


def test_une_reponse_differente_n_est_pas_prise_pour_une_repetition():
    retriever = _Retriever({"cours-publie": [_passage()]})
    llm = _Llm([json.dumps({"reponse": "L'angle est l'argument de a [S1].", "verification": "?"})])
    tutor = Tutor(llm=llm, retriever=retriever, settings=_settings())
    answer = asyncio.run(tutor.answer(
        question="Et l'angle ?", scope=_scope(), course_id="cours-7",
        history=[{"role": "eleve", "content": "Le centre ?"}, {"role": "lawal", "content": CENTRE}],
    ))
    assert "TUTOR_REPETITION_RETRIED" not in answer.warnings
    assert len(llm.messages_seen) == 1


def test_la_question_arrive_apres_les_extraits():
    retriever = _Retriever({"cours-publie": [_passage()]})
    llm = _Llm([json.dumps({"reponse": "Ici [S1].", "verification": "?"})])
    tutor = Tutor(llm=llm, retriever=retriever, settings=_settings())
    asyncio.run(tutor.answer(question="C'est quoi la norme ?", scope=_scope(), course_id="cours-7"))
    last = llm.messages_seen[0][-1]["content"]
    assert last.index("Extraits validés") < last.index("C'est quoi la norme ?")
    assert last.rstrip().endswith("C'est quoi la norme ?")


FAUX = "Si elle n'est pas une translation (c'est-à-dire |a| = 1), son centre est b/(1-a) [S1]."
JUSTE = "Si elle n'est pas une translation (c'est-à-dire a ≠ 1), son centre est b/(1-a) [S1]."


def test_la_relecture_corrige_une_erreur_qu_elle_nomme():
    """Constaté le 15/09/2026 : « |a| = 1 » là où le cours dit a ≠ 1."""

    retriever = _Retriever({"cours-publie": [_passage()]})
    relecture = ("### VERDICT\nERREUR\n### ERREUR\n« |a| = 1 » est faux : [S1] dit a ≠ 1.\n"
                 "### RÉPONSE CORRIGÉE\n" + JUSTE)
    llm = _Llm([json.dumps({"reponse": FAUX, "verification": "?"}), relecture])
    tutor = Tutor(llm=llm, retriever=retriever, settings=_settings(answer_review=True))
    answer = asyncio.run(tutor.answer(question="Le centre ?", scope=_scope(), course_id="cours-7"))
    assert answer.text == JUSTE
    assert "TUTOR_ANSWER_REVISED" in answer.warnings
    assert FAUX in llm.messages_seen[1][-1]["content"]


def test_une_correction_sans_erreur_nommee_est_ignoree():
    # Un petit modèle peut « corriger » ce qui était juste : sans erreur
    # argumentée, la réponse d'origine reste.
    retriever = _Retriever({"cours-publie": [_passage()]})
    relecture = "### VERDICT\nERREUR\n### ERREUR\n\n### RÉPONSE CORRIGÉE\nAutre chose, bien plus longue que prévu."
    llm = _Llm([json.dumps({"reponse": JUSTE, "verification": "?"}), relecture])
    tutor = Tutor(llm=llm, retriever=retriever, settings=_settings(answer_review=True))
    answer = asyncio.run(tutor.answer(question="Le centre ?", scope=_scope(), course_id="cours-7"))
    assert answer.text == JUSTE
    assert "TUTOR_ANSWER_REVISED" not in answer.warnings


def test_une_relecture_en_panne_laisse_la_reponse():
    retriever = _Retriever({"cours-publie": [_passage()]})

    class _Panne(_Llm):
        async def chat(self, messages, **kwargs):
            self.messages_seen.append(messages)
            if len(self.messages_seen) == 2:
                raise RuntimeError("ollama injoignable")
            return self.replies.pop(0)

    llm = _Panne([json.dumps({"reponse": JUSTE, "verification": "?"})])
    tutor = Tutor(llm=llm, retriever=retriever, settings=_settings(answer_review=True))
    answer = asyncio.run(tutor.answer(question="Le centre ?", scope=_scope(), course_id="cours-7"))
    assert answer.text == JUSTE
    assert "TUTOR_REVIEW_FAILED" in answer.warnings


def test_le_compris_final_ne_doublonne_pas_la_verification():
    retriever = _Retriever({"cours-publie": [_passage()]})
    llm = _Llm([json.dumps({"reponse": JUSTE + "\n\nCompris ?", "verification": "Et si a = 1 ?"})])
    tutor = Tutor(llm=llm, retriever=retriever, settings=_settings())
    answer = asyncio.run(tutor.answer(question="Le centre ?", scope=_scope(), course_id="cours-7"))
    assert answer.text == JUSTE


def test_la_reponse_balisee_est_lue_avec_ses_formules():
    """Le format balisé remplace le JSON : les formules y passent telles quelles."""

    retriever = _Retriever({"cours-publie": [_passage()]})
    raw = ("### RÉPONSE\nLe module de \\( 3+4i \\) vaut \\[ \\sqrt{3^2+4^2} = 5 \\] [S1]\n"
           "### VÉRIFICATION\nEt celui de \\( 12-5i \\) ?\n### NOTIONS\nmodule, affixe")
    tutor = Tutor(llm=_Llm([raw]), retriever=retriever, settings=_settings())
    answer = asyncio.run(tutor.answer(question="Le module ?", scope=_scope(), course_id="cours-7"))
    assert answer.text == "Le module de \\( 3+4i \\) vaut \\[ \\sqrt{3^2+4^2} = 5 \\] [S1]"
    assert answer.check == "Et celui de \\( 12-5i \\) ?"
    assert answer.concepts == ["module", "affixe"]
    assert "TUTOR_PLAIN_TEXT" not in answer.warnings


def test_un_json_casse_n_arrive_jamais_brut_chez_l_eleve():
    """Vu au banc du 15/09/2026 (« Pourquoi i² = -1 ? ») : deux objets aux
    échappements invalides, et l'élève recevait {"reponse": …} tel quel."""

    retriever = _Retriever({"cours-publie": [_passage()]})
    casse = ('{"reponse": "Comme \\(i = \\cos \\frac{\\pi}{2}\\), on a \\(i^2 = -1\\).\n\nQuelle est '
             'la signification ?"}\n\n{"verification": "Et -1 ?", "conceptes": ["argument"]}')
    tutor = Tutor(llm=_Llm([casse, casse]), retriever=retriever, settings=_settings())
    answer = asyncio.run(tutor.answer(question="Pourquoi i² = -1 ?", scope=_scope(), course_id="cours-7"))
    assert not answer.text.lstrip().startswith("{")
    assert '"reponse"' not in answer.text
    assert "i^2 = -1" in answer.text


def test_un_json_coupe_par_la_limite_n_arrive_jamais_brut():
    from app.core.tutor import _sauver_json

    coupe = '{"reponse": "Pour montrer que la suite est croissante, on étudie le signe de u_{n+1} - u_n.", "verification": "Quelle est l\'étape'
    assert _sauver_json(coupe) == "Pour montrer que la suite est croissante, on étudie le signe de u_{n+1} - u_n."
    tronque = '{"reponse": "Pour montrer que la suite est croissante, on étudie le signe'
    assert _sauver_json(tronque) == "Pour montrer que la suite est croissante, on étudie le signe"


def test_la_relecture_est_eteinte_par_defaut():
    """Au banc du 15/09/2026, la relecture par qwen2.5:7b a rendu fausse une
    réponse juste : elle ne tourne que si on l'allume."""

    retriever = _Retriever({"cours-publie": [_passage()]})
    relecture = ("### VERDICT\nERREUR\n### ERREUR\nprétendue erreur\n"
                 "### RÉPONSE CORRIGÉE\nUne version fausse mais bien plus longue que l'originale.")
    llm = _Llm([json.dumps({"reponse": JUSTE, "verification": "?"}), relecture])
    tutor = Tutor(llm=llm, retriever=retriever, settings=_settings())
    answer = asyncio.run(tutor.answer(question="Le centre ?", scope=_scope(), course_id="cours-7"))
    assert answer.text == JUSTE
    assert len(llm.messages_seen) == 1


FAUX_COORD = ("### RÉPONSE\nOn a \\( 2e^{i\\pi/3} = 1 + i\\sqrt{3} \\). "
              "Les coordonnées du point sont donc (1, 3).\n### VÉRIFICATION\n?")
JUSTE_COORD = ("### RÉPONSE\nOn a \\( 2e^{i\\pi/3} = 1 + i\\sqrt{3} \\). "
               "Les coordonnées du point sont donc \\( (1 ; \\sqrt{3}) \\).\n### VÉRIFICATION\n?")


def test_un_calcul_faux_est_renvoye_en_correction_avec_la_valeur_sure():
    """Note d'évaluation du 16/09/2026 : SymPy calcule, le modèle réécrit."""

    retriever = _Retriever({"cours-publie": [_passage()]})
    llm = _Llm([FAUX_COORD, JUSTE_COORD])
    tutor = Tutor(llm=llm, retriever=retriever, settings=_settings())
    answer = asyncio.run(tutor.answer(question="Calcule 2e^{iπ/3}", scope=_scope(), course_id="cours-7"))
    assert "(1 ; \\sqrt{3})" in answer.text
    assert "MATH_CHECK_RETRIED" in answer.warnings
    assert "MATH_CHECK_STILL_WRONG" not in answer.warnings
    relance = llm.messages_seen[1][-1]["content"]
    assert "logiciel de calcul" in relance and "\\sqrt{3}" in relance


def test_un_calcul_toujours_faux_est_signale_sans_boucler():
    retriever = _Retriever({"cours-publie": [_passage()]})
    llm = _Llm([FAUX_COORD, FAUX_COORD])
    tutor = Tutor(llm=llm, retriever=retriever, settings=_settings())
    answer = asyncio.run(tutor.answer(question="Calcule 2e^{iπ/3}", scope=_scope(), course_id="cours-7"))
    assert "MATH_CHECK_STILL_WRONG" in answer.warnings
    assert len(llm.messages_seen) == 2


def test_un_calcul_juste_ne_coute_aucun_appel_de_plus():
    retriever = _Retriever({"cours-publie": [_passage()]})
    llm = _Llm([JUSTE_COORD])
    tutor = Tutor(llm=llm, retriever=retriever, settings=_settings())
    answer = asyncio.run(tutor.answer(question="Calcule 2e^{iπ/3}", scope=_scope(), course_id="cours-7"))
    assert "MATH_CHECK_RETRIED" not in answer.warnings
    assert len(llm.messages_seen) == 1


def test_une_erreur_du_cours_est_signalee_et_non_recopiee():
    """16/09/2026 : le cours publié écrit « z₁ = −1 + i est dans le premier
    quadrant ». Lawal doit distinguer ce que dit le cours de ce qui est juste."""

    from app.core.tutor import _SYSTEM

    assert "un extrait n'est pas une preuve" in _SYSTEM
    assert "ne le corrige pas en silence" in _SYSTEM
    assert "Le cours indique … ; en fait … car …" in _SYSTEM
