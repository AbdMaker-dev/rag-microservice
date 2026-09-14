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

    async def chat(self, messages, *, timeout, num_ctx, num_predict, schema=None) -> str:
        self.exchanges.append(list(messages))
        self.schemas = getattr(self, "schemas", [])
        self.schemas.append(schema)
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
        "parties": [
        {"titre": "Introduction",
         "description": "Je rappellerai les acquis sur les vecteurs.",
         "sousParties": []},
        {"titre": "Le produit scalaire",
         "description": "Je définirai le produit scalaire, je démontrerai les propriétés, je proposerai des exercices.",
         "sousParties": ["Définition par le projeté orthogonal",
                          "Propriétés de calcul", "Exercices"]}]})
    retriever = FakeRetriever()
    llm = ScriptedLlm([plan])

    draft = asyncio.run(_generator(llm, retriever).draft_plan(
        instruction="cours produit scalaire", scope=SCOPE, course_id="c"))

    assert draft.title == "Le produit scalaire"
    assert draft.description.startswith("Définition, propriétés")
    assert [i.heading for i in draft.items] == ["Introduction", "Le produit scalaire"]
    # Une partie simple est une feuille ; les sous-parties sont des titres
    # seuls — leur annonce, c'est la description du parent.
    assert draft.items[0].children == []
    assert draft.items[1].children == [
        "Définition par le projeté orthogonal", "Propriétés de calcul", "Exercices"]
    # Un seul appel modèle ; deux recherches — le programme donne le cadre,
    # les supports déposés disent ce que le plan doit réellement couvrir.
    assert len(llm.exchanges) == 1
    assert retriever.calls[0]["role"] == "programme-officiel"
    assert retriever.calls[1]["role"] == "support-cours"
    assert retriever.calls[1]["course_id"] == "c"


def test_le_plan_cherche_avec_le_sujet_pas_avec_la_consigne():
    # Mesuré le 13/09/2026 : la consigne « n'invente aucune formule » servait
    # de question de recherche au programme officiel et aux supports.
    plan = json.dumps({"titre": "T", "parties": [
        {"titre": "Définition", "description": "", "sousParties": []}]})
    retriever = FakeRetriever()
    llm = ScriptedLlm([plan])

    asyncio.run(_generator(llm, retriever).draft_plan(
        instruction="Reste fidèle au document : n'invente aucune formule.",
        title="Nombres complexes et transformations du plan",
        scope=SCOPE, course_id="c"))

    assert [c["query"] for c in retriever.calls] == [
        "Nombres complexes et transformations du plan",
        "Nombres complexes et transformations du plan"]
    # Le modèle voit les deux : le sujet, et la consigne qui reste une consigne.
    user = llm.exchanges[0][-1]["content"]
    assert user.startswith("Cours : Nombres complexes et transformations du plan\n")
    assert "Demande du professeur : Reste fidèle au document" in user


def test_sans_titre_le_plan_cherche_avec_l_instruction_comme_avant():
    plan = json.dumps({"titre": "T", "parties": [
        {"titre": "Définition", "description": "", "sousParties": []}]})
    retriever = FakeRetriever()

    asyncio.run(_generator(ScriptedLlm([plan]), retriever).draft_plan(
        instruction="cours produit scalaire", scope=SCOPE, course_id="c"))

    assert retriever.calls[0]["query"] == "cours produit scalaire"


def test_le_plan_se_revise_en_conversation():
    revise = json.dumps({"titre": "T", "parties": [
        {"titre": "Définition", "description": "", "sousParties": []},
        {"titre": "Exercices", "description": "Je proposerai trois applications.",
         "sousParties": []}]})
    llm = ScriptedLlm([revise])

    draft = asyncio.run(_generator(llm).draft_plan(
        instruction="cours", scope=SCOPE, course_id="c",
        current_plan={"title": "T", "items": [
            {"heading": "Définition", "description": "…",
             "children": [{"heading": "Notation"}]}]},
        request="ajoute une partie exercices",
        history=[{"author": "prof", "message": "reste simple"}]))

    contenu = llm.exchanges[0][1]["content"]
    assert "Plan actuel" in contenu and "ajoute une partie exercices" in contenu
    assert "reste simple" in contenu
    # la hiérarchie du plan courant est montrée au modèle, enfants compris
    assert "    - Notation" in contenu
    assert [i.heading for i in draft.items][-1] == "Exercices"


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


class RetrieverAvecFigures(FakeRetriever):
    """Le premier support porte une figure ancrée, le second aucune."""

    async def search(self, *, query, scope, limit, max_excerpt_characters,
                     course_id=None, role=None):
        self.calls.append({"query": query, "course_id": course_id, "role": role})
        if role == "support-cours":
            return [
                Passage(chunk_id="s1", document_id="doc", title="Cours", locator="p. 3",
                        content="Le triangle ABC… [FIGURE dded9a9f — p. 3]",
                        language="fr", score=0.9, figures=["dded9a9f"]),
                Passage(chunk_id="s2", document_id="doc", title="Cours", locator="p. 4",
                        content="L'aire vaut (base × hauteur) / 2.",
                        language="fr", score=0.8),
            ]
        return [_passage("1", "programme")]


def test_la_figure_d_un_passage_cite_se_pose_apres_le_paragraphe_qui_le_cite():
    # Décision « 1 + 2 » du 13/09/2026 : l'IA place, le prof corrige.
    llm = ScriptedLlm([
        "Une hauteur est une droite issue d'un sommet [S2].\n\n"
        "Considérons le triangle ABC et sa hauteur AD [S1].\n\n"
        "On retrouve la même hauteur plus loin [S1]."
    ])

    result = asyncio.run(_generator(llm, RetrieverAvecFigures()).write_one_section(
        heading="Hauteurs", instruction="cours", scope=SCOPE, course_id="c",
        strictness="grounded", plan_headings=["Hauteurs"]))

    assert result.sections[0].text == (
        "Une hauteur est une droite issue d'un sommet [S2].\n\n"
        "Considérons le triangle ABC et sa hauteur AD [S1].\n\n"
        "[FIGURE dded9a9f]\n\n"
        "On retrouve la même hauteur plus loin [S1]."
    )


def test_une_figure_deja_placee_par_le_modele_ne_se_pose_pas_deux_fois():
    llm = ScriptedLlm(["Le triangle [S1].\n\n[FIGURE dded9a9f]\n\nSuite [S1]."])

    result = asyncio.run(_generator(llm, RetrieverAvecFigures()).write_one_section(
        heading="Hauteurs", instruction="cours", scope=SCOPE, course_id="c",
        strictness="grounded", plan_headings=["Hauteurs"]))

    assert result.sections[0].text.count("[FIGURE dded9a9f]") == 1


def test_une_figure_d_un_passage_non_cite_reste_ou_elle_est():
    llm = ScriptedLlm(["L'aire vaut base fois hauteur sur deux [S2]."])

    result = asyncio.run(_generator(llm, RetrieverAvecFigures()).write_one_section(
        heading="Aire", instruction="cours", scope=SCOPE, course_id="c",
        strictness="grounded", plan_headings=["Aire"]))

    assert "[FIGURE" not in result.sections[0].text


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


def test_la_section_tient_l_engagement_valide_par_le_prof():
    """La description validée — éventuellement corrigée à la main par le
    professeur — est le contrat de contenu : elle doit être sous les yeux du
    modèle au moment de rédiger."""

    llm = ScriptedLlm(["Contenu conforme à l'engagement [S1]."])

    asyncio.run(_generator(llm).write_one_section(
        heading="Propriétés", instruction="cours", scope=SCOPE,
        course_id="c", strictness="grounded", plan_headings=["Propriétés"],
        description=("Je démontrerai la symétrie et la bilinéarité, puis "
                     "deux exemples guidés — SANS l'inégalité de Schwarz.")))

    contenu = llm.exchanges[0][1]["content"]
    assert "validé par le professeur" in contenu
    assert "SANS l'inégalité de Schwarz" in contenu


def test_la_description_guide_aussi_la_recherche_d_extraits():
    """« Je démontrerai les identités remarquables » cherche les identités
    remarquables dans les documents — pas seulement le titre de la section."""

    llm = ScriptedLlm(["Contenu [S1]."])
    retriever = FakeRetriever()

    asyncio.run(_generator(llm, retriever).write_one_section(
        heading="Propriétés", instruction="cours", scope=SCOPE,
        course_id="c", strictness="grounded", plan_headings=["Propriétés"],
        description="Je démontrerai les identités remarquables (U+V)²."))

    for call in retriever.calls:
        assert "identités remarquables" in call["query"]


def test_le_prompt_du_plan_impose_le_ton_impersonnel():
    """La description ÉNONCE ce qui sera fait — « Définition de… ;
    démonstration de… » — sans personne : ni « je » qui approprierait le
    cours à l'IA, ni « nous » qui le personnaliserait. Le style des
    programmes officiels."""

    plan = json.dumps({"titre": "T", "parties": [
        {"titre": "A", "description": "Définition du produit scalaire.",
         "sousParties": []}]})
    llm = ScriptedLlm([plan])

    asyncio.run(_generator(llm).draft_plan(
        instruction="cours", scope=SCOPE, course_id="c"))

    consigne = llm.exchanges[0][0]["content"]
    assert "IMPERSONNELLE" in consigne
    assert "Définition du produit scalaire" in consigne


def test_le_ton_professionnel_est_impose_partout():
    """Exigence d'Alioune (01/09/2026) : plan, sections et révisions parlent
    comme un enseignant expérimenté, et la rédaction est incitée à chercher
    autant que nécessaire avant d'écrire."""

    from app.core.generation import _TONE

    plan = json.dumps({"titre": "T", "parties": [
        {"titre": "A", "description": "Définition.", "sousParties": []}]})
    llm = ScriptedLlm([plan])
    asyncio.run(_generator(llm).draft_plan(
        instruction="cours", scope=SCOPE, course_id="c"))
    assert "professionnel et mature" in _TONE
    assert _TONE.strip(" ") in llm.exchanges[0][0]["content"]


def test_un_texte_grounded_sans_citation_est_signale():
    """Constaté au premier test serveur : un texte plausible sans une seule
    étiquette [S]/[P] — invérifiable, en silence. Le prof doit le savoir."""

    plan = json.dumps({"titre": "T", "sections": ["Définition"]})
    # Deux réponses sans étiquette : la relance ferme échoue aussi.
    llm = ScriptedLlm([plan, "Un texte plausible sans aucune étiquette.",
                       "Encore sans étiquette."])

    course = asyncio.run(_generator(llm).generate(
        instruction="cours", scope=SCOPE, course_id="c", strictness="grounded"))

    assert "GROUNDED_TEXT_WITHOUT_CITATIONS" in course.warnings


def test_un_texte_grounded_sans_etiquette_est_relance_une_fois():
    """7 sections sur 8 sorties sans étiquette en production : une relance
    ferme, et le texte relancé (étiqueté) remplace le premier."""

    plan = json.dumps({"titre": "T", "sections": ["Définition"]})
    llm = ScriptedLlm([plan, "Un texte plausible sans étiquette.",
                       "La définition [S1], cadrée par [P1]."])

    course = asyncio.run(_generator(llm).generate(
        instruction="cours", scope=SCOPE, course_id="c", strictness="grounded"))

    assert course.sections[0].text.startswith("La définition [S1]")
    assert len(course.sections[0].citations) == 2
    assert "GROUNDED_TEXT_WITHOUT_CITATIONS" not in course.warnings
    relance = llm.exchanges[-1][-1]["content"]
    assert "AUCUNE étiquette" in relance


def test_la_relance_qui_echoue_garde_le_texte_et_le_warning():
    plan = json.dumps({"titre": "T", "sections": ["Définition"]})
    llm = ScriptedLlm([plan, "Sans étiquette.", "Toujours sans étiquette."])

    course = asyncio.run(_generator(llm).generate(
        instruction="cours", scope=SCOPE, course_id="c", strictness="grounded"))

    assert course.sections[0].text == "Sans étiquette."
    assert "GROUNDED_TEXT_WITHOUT_CITATIONS" in course.warnings


def test_une_demande_de_recherche_melee_au_texte_est_bien_vue():
    """Vécu le 01/09 : le modèle a émis deux {"chercher"} PUIS la section.
    Le parseur d'alors butait sur les accolades LaTeX et le cours du prof
    s'est retrouvé avec du JSON en tête."""

    from app.core.generation import _wants_search

    raw = (
        '{"chercher": {"question": "relation module argument ?", "nature": "support-cours"}}\n\n'
        '{"chercher": {"question": "interprétation géométrique ?", "nature": "support-cours"}}\n\n'
        "Sur la base des informations fournies, voici la section :\n\n"
        "Le centre est \\(z' - \\omega = ke^{i\\theta}(z - \\omega)\\) — noter les accolades."
    )
    wanted = _wants_search(raw)
    assert wanted is not None, "la demande de recherche doit être vue"
    assert "interprétation" in wanted[0] or "relation" in wanted[0]


def _generator_fenetre(llm, context_tokens):
    settings = get_settings().model_copy(update={"generation_context_tokens": context_tokens})
    return CourseGenerator(llm=llm, retriever=FakeRetriever(), settings=settings)


def _cours_long(sections=10, taille=1_500):
    return "\n\n".join(
        f"## Section {i}\n\n" + ("Le produit scalaire de deux vecteurs. " * (taille // 38))
        for i in range(1, sections + 1)
    )


def test_un_cours_qui_tient_produit_son_bloc_en_un_seul_appel():
    llm = ScriptedLlm([json.dumps({"resume": "Résumé direct."})])

    draft = asyncio.run(_generator_fenetre(llm, 32_768).generate_blocks(
        kind="resume", text=_cours_long(), scope=SCOPE))

    assert draft.summary == "Résumé direct."
    assert len(llm.exchanges) == 1
    assert draft.warnings == []


def test_un_cours_trop_long_est_decoupe_et_son_resume_fusionne():
    # Constaté le 13/09/2026 : deux cours de dix sections, 8 600 et 9 800
    # tokens pour une fenêtre de 8 192 — les trois blocs échouaient.
    llm = ScriptedLlm([
        json.dumps({"resume": "Partie A."}),
        json.dumps({"resume": "Partie B."}),
        json.dumps({"resume": "A puis B, fusionnés."}),
    ])

    draft = asyncio.run(_generator_fenetre(llm, 8_192).generate_blocks(
        kind="resume", text=_cours_long(), scope=SCOPE))

    assert draft.summary == "A puis B, fusionnés."
    assert draft.warnings == ["COURSE_SPLIT_FOR_BLOCKS:2"]
    assert len(llm.exchanges) == 3
    # Aucun appel ne dépasse la fenêtre : c'est tout l'objet du découpage.
    for messages in llm.exchanges:
        assert sum(len(m["content"]) for m in messages) // 3 <= 8_192
    # La fusion reçoit les résumés partiels, dans l'ordre, et la consigne le dit.
    fusion = llm.exchanges[2]
    assert "Partie A.\n\nPartie B." in fusion[1]["content"]
    assert "résumés partiels" in fusion[0]["content"]


def test_un_quiz_sur_un_cours_decoupe_prend_sa_part_dans_chaque_partie():
    def q(n):
        return {"question": f"Q{n} ?", "choix": ["a", "b", "c", "d"], "reponse": 1,
                "explication": "…"}
    llm = ScriptedLlm([
        json.dumps({"questions": [q(1), q(2), q(3)]}),
        json.dumps({"questions": [q(4), q(5), q(6)]}),
    ])

    draft = asyncio.run(_generator_fenetre(llm, 8_192).generate_blocks(
        kind="quiz", text=_cours_long(), scope=SCOPE, count=5))

    assert [x["question"] for x in draft.quiz] == ["Q1 ?", "Q2 ?", "Q3 ?", "Q4 ?", "Q5 ?"]
    # Chaque partie s'est vu demander sa part : ⌈5/2⌉ = 3.
    assert "QUIZ de 3 questions" in llm.exchanges[0][0]["content"]


def test_le_decoupage_respecte_les_paragraphes():
    from app.core.generation import _split_course

    texte = "\n\n".join(["a" * 400, "b" * 400, "c" * 400])
    parts = _split_course(texte, 900)

    assert parts == ["a" * 400 + "\n\n" + "b" * 400, "c" * 400]
    assert _split_course(texte, 5_000) == [texte]


def test_une_queue_repetitive_est_coupee_et_le_resume_sauve():
    # Mesuré le 14/09/2026 : le modèle a glissé en LaTeX puis répété
    # « \boldsymbol{ » sur 3 000 caractères, épuisant son budget de sortie —
    # JSON jamais refermé, bloc perdu. Deux tentatives, même réponse au
    # caractère près.
    raw = ('{\n  "resume": "Une suite croissante et majorée converge. '
           'Le théorème des gendarmes encadre la limite.' + "\\\\boldsymbol{" * 300)
    llm = ScriptedLlm([raw])

    draft = asyncio.run(_generator(llm).generate_blocks(
        kind="resume", text="Cours court.", scope=SCOPE))

    assert draft.summary == ("Une suite croissante et majorée converge. "
                             "Le théorème des gendarmes encadre la limite.")
    assert "MODEL_REPETITION_TRIMMED" in draft.warnings
    assert "BLOCK_JSON_TRUNCATED" in draft.warnings
    # Une seule tentative : le texte sauvé suffit, on ne repaie pas le modèle.
    assert len(llm.exchanges) == 1


def test_la_relance_change_la_demande_au_lieu_de_la_repeter():
    llm = ScriptedLlm(["pas du json", json.dumps({"resume": "Enfin."})])

    draft = asyncio.run(_generator(llm).generate_blocks(
        kind="resume", text="Cours court.", scope=SCOPE))

    assert draft.summary == "Enfin."
    relance = llm.exchanges[1][-1]["content"]
    assert "pas de LaTeX" in relance and "plus court" in relance
    assert relance != llm.exchanges[0][-1]["content"]


def test_les_trois_blocs_interdisent_le_latex():
    for kind, reply in [
        ("resume", json.dumps({"resume": "R."})),
        ("quiz", json.dumps({"questions": [
            {"question": "Q ?", "choix": ["a", "b", "c", "d"], "reponse": 0,
             "explication": "…"}]})),
        ("exercices", json.dumps({"exercices": [
            {"enonce": "E", "corrige": "C", "difficulte": "moyen"}]})),
    ]:
        llm = ScriptedLlm([reply])
        asyncio.run(_generator(llm).generate_blocks(
            kind=kind, text="Cours court.", scope=SCOPE))
        assert "ni LaTeX ni commande à contre-oblique" in llm.exchanges[0][0]["content"], kind


def test_un_texte_sain_n_est_jamais_coupe():
    from app.core.generation import _couper_repetition

    # Une énumération répète sa forme sans jamais répéter le même motif.
    texte = " ".join(f"Exercice {n} : calculer u_{n}." for n in range(1, 40))
    assert _couper_repetition(texte) == texte
    # Un texte court reste intact même s'il bégaie.
    assert _couper_repetition("ha" * 20) == "ha" * 20


def test_un_bloc_de_controle_ne_finit_jamais_dans_le_cours():
    """Le filet : quoi qu'il arrive en amont, la section rendue au
    professeur ne porte pas de JSON de contrôle. Le cours du 01/09 en
    portait deux en tête."""

    from app.core.generation import _strip_control_blocks

    pollue = (
        '{"chercher": {"question": "relation module argument ?", "nature": "support-cours"}}\n\n'
        '{"chercher": {"question": "interprétation géométrique ?", "nature": "support-cours"}}\n\n'
        "### Relation entre les éléments\n\n"
        "Le centre vérifie \\(z' - \\omega = ke^{i\\theta}(z - \\omega)\\)."
    )
    propre = _strip_control_blocks(pollue)
    assert "chercher" not in propre
    assert propre.startswith("### Relation")
    # Les formules, elles, ne sont pas touchées.
    assert "ke^{i\\theta}" in propre


def test_un_texte_sans_bloc_de_controle_reste_intact():
    from app.core.generation import _strip_control_blocks

    texte = "Une section avec des accolades \\(\\frac{a}{b}\\) et rien d'autre."
    assert _strip_control_blocks(texte) == texte
