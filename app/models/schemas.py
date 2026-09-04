"""Contrats d'entrée et de sortie de l'API.

Convention : camelCase sur le fil, snake_case en Python.
Chaque réponse porte `contractVersion` pour qu'une rupture soit visible.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

CONTRACT_VERSION = "1.0"


def _camel(name: str) -> str:
    head, *rest = name.split("_")
    return head + "".join(part.capitalize() for part in rest)


class Wire(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True, alias_generator=_camel, extra="forbid"
    )


class Scope(Wire):
    """Périmètre pédagogique.

    Résolu par la plateforme et transmis à chaque appel. Le service ne
    l'élargit jamais et ne décide jamais qui a le droit de lire quoi.
    """

    country: str
    subject: str
    # Le cycle, dans les appellations du pays : « primaire », « cem »,
    # « lycee » au Sénégal. D'autres pays nommeront autrement — le service
    # stocke, il n'interprète pas.
    level: str = ""
    # La série (L, S, S1…). Vide pour le primaire et le CEM, qui n'en ont pas.
    track: str = ""
    # La classe : seconde, première, terminale, CM2…
    grade: str
    curriculum_version: str
    language: str = "fr"


# --------------------------------------------------------------------- extraire

MediaType = Literal[
    "text/plain",
    "text/markdown",
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    # Le type générique de qui ne sait pas nommer son fichier : un dépôt
    # multipart relayé par la plateforme arrive ainsi. On ne le croit pas
    # davantage que les autres — la SIGNATURE du fichier décide, comme le
    # promet cette route. Le refuser d'entrée contredisait cette promesse et
    # a cassé le premier appel réel de management (04/09/2026).
    "application/octet-stream",
]


class ExtractRequest(Wire):
    request_id: str
    filename: str = Field(min_length=1, max_length=255)
    media_type: MediaType
    # Contenu du fichier encodé en base64. Le service ne télécharge jamais
    # une URL : la plateforme envoie les octets qu'elle a le droit d'utiliser.
    content_base64: str = Field(min_length=1)


class SectionIssue(Wire):
    """Un passage à vérifier, situé par ses positions dans le texte.

    Les offsets sont des indices de caractères dans `text` de la section, pour
    que l'interface puisse surligner sans avoir à rechercher la chaîne.
    """

    kind: str
    start: int
    end: int
    excerpt: str = ""


class ExtractedSection(Wire):
    """Un bloc de texte et l'endroit d'où il vient."""

    position: int
    locator: str
    text: str
    characters: int
    # Ce que vaut cette section, et où regarder. Une note globale dit qu'un
    # document est bon ou mauvais ; elle ne dit pas où un professeur doit
    # porter les yeux sur cent pages.
    confidence: float = 1.0
    issues: List[SectionIssue] = []


class ExtractionQuality(Wire):
    """Ce que vaut le texte rendu, pour que le prof sache quoi relire."""

    score: float
    word_plausibility: float
    cid_markers: int
    # Ce que la passe de rattrapage mot à mot a corrigé. La correction des
    # polices, elle, se mesure dans charactersRepaired : un compteur qui
    # restait à zéro pendant que neuf polices étaient réécrites n'était pas un
    # compteur.
    words_repaired: int = 0
    # Caractères non-ASCII dont la table a été rétablie par la réécriture des
    # polices — borne basse, comptée sur les pages échantillonnées.
    characters_repaired: int = 0
    unreadable_fonts: List[str] = []


class FontDiagnosis(Wire):
    """Ce qu'on a conclu pour une police du document."""

    font: str
    table: Optional[str] = None
    confidence: float = 0.0
    samples: int = 0


class DocumentAnalysis(Wire):
    """Ce qu'est le document, avant même ce qu'il dit.

    Additive et ignorable : un appelant qui ne la lit pas voit la même réponse
    qu'avant. Elle sert à router — un PDF balisé porte son propre plan, une
    page sans couche texte demande l'OCR.
    """

    route: str = "untagged"
    tagged: bool = False
    # Le décompte des passages douteux par type, toutes sections confondues —
    # pour l'appelant qui ne lit pas sections[].
    issue_summary: Dict[str, int] = {}
    text_coverage: float = 0.0
    pages_needing_ocr: List[int] = []
    fonts: List[FontDiagnosis] = []


class CapturedFigure(Wire):
    """Une zone dessinée du document, capturée en image.

    Figures de géométrie, mais aussi formules posées en image par un export
    Word : tout ce que la couche texte ne porte pas. Le texte extrait signale
    chaque capture par un marqueur `[FIGURE fN — p. P]` ; l'image part ici,
    en PNG base64 — le service ne stocke rien, c'est l'appelant qui la garde
    et qui remplace le marqueur par son URL à l'affichage.
    """

    figure_id: str
    page: int
    # Dimensions en pixels du PNG rendu (150 dpi, plafonné).
    width: int
    height: int
    image_base64: str


class ExtractResponse(Wire):
    contract_version: Literal["1.0"] = CONTRACT_VERSION
    request_id: str
    filename: str
    media_type: str
    text: str
    characters: int
    sections: List[ExtractedSection]
    quality: ExtractionQuality
    analysis: Optional[DocumentAnalysis] = None
    figures: List[CapturedFigure] = []
    warnings: List[str] = []


# --------------------------------------------------------------------- indexer


# « cours-publie » : le contenu assemblé d'un cours publié, indexé par la
# plateforme à la publication — c'est ce que le tuteur élève cite en premier.
# « annale » : un sujet d'examen passé, commun au périmètre comme un programme
# officiel — la matière première du futur coach examen.
DocumentRole = Literal["support-cours", "programme-officiel", "cours-publie", "annale"]


class IndexRequest(Wire):
    request_id: str
    # Identifiant du document côté plateforme. Réindexer le même identifiant
    # remplace ses passages : l'appel est donc rejouable sans risque.
    document_id: str = Field(min_length=1, max_length=255)
    # Le cours auquel ce document appartient, créé côté plateforme AVANT tout
    # dépôt. C'est cette référence qui permettra à la génération de retrouver
    # les documents d'un cours précis, pas seulement ceux d'un périmètre.
    # Vide UNIQUEMENT pour un programme officiel, qui fait référence pour tout
    # le périmètre et n'appartient à aucun cours.
    course_id: str = Field(default="", max_length=255)
    # La nature du document : support déposé par un professeur dans son cours,
    # ou programme officiel déposé par un administrateur pour le périmètre.
    role: DocumentRole = "support-cours"
    title: str = Field(min_length=1, max_length=500)
    source_reference: str = ""
    scope: Scope
    text: str = Field(min_length=1)


class IndexResponse(Wire):
    contract_version: Literal["1.0"] = CONTRACT_VERSION
    request_id: str
    document_id: str
    chunks: int
    characters: int
    embedding_model: str
    embedding_dimension: int


class DeleteResponse(Wire):
    contract_version: Literal["1.0"] = CONTRACT_VERSION
    document_id: str
    deleted: bool


# -------------------------------------------------------------------- chercher


class SearchRequest(Wire):
    request_id: str
    scope: Scope
    # Restreindre la recherche aux documents d'un cours : c'est le mode de la
    # génération. Absent, on cherche dans tout le périmètre.
    course_id: Optional[str] = None
    # Restreindre à une nature de document. La génération distingue les deux :
    # le programme officiel donne le cadre et les compétences exigibles, les
    # supports du professeur donnent la matière du cours.
    role: Optional[DocumentRole] = None
    query: str = Field(min_length=1, max_length=4_000)
    limit: int = Field(default=5, ge=1, le=50)
    max_excerpt_characters: int = Field(default=1_200, ge=100, le=10_000)
    # Restreindre à une sélection de documents. Vide = tout le périmètre.
    document_ids: List[str] = []


class SearchItem(Wire):
    chunk_id: str
    document_id: str
    title: str
    locator: str
    excerpt: str
    language: str
    score: float


class SearchResponse(Wire):
    contract_version: Literal["1.0"] = CONTRACT_VERSION
    request_id: str
    # Explicite plutôt que silencieux : une liste vide doit se lire comme
    # "aucune preuve", pour que le modèle en aval puisse dire "je ne sais pas"
    # au lieu d'inventer.
    evidence: Literal["SUFFICIENT", "INSUFFICIENT_EVIDENCE"]
    items: List[SearchItem]


# -------------------------------------------------------------------- générer


class GenerateRequest(Wire):
    request_id: str
    # Le cours dont on rédige le contenu : les recherches du modèle sur les
    # supports y sont verrouillées.
    course_id: str = Field(min_length=1, max_length=255)
    scope: Scope
    # La note du professeur : ce qu'il veut comme cours.
    instruction: str = Field(min_length=3, max_length=4000)
    # grounded : rien hors des extraits, les manques sont signalés.
    # enriched : le modèle peut compléter, chaque ajout encadré ⟦AJOUT⟧…⟦/AJOUT⟧.
    strictness: Literal["grounded", "enriched"] = "grounded"


class AdjustSection(Wire):
    heading: str
    text: str
    citations: List[Dict[str, Any]] = []


class AdjustRequest(Wire):
    """La conversation : le cours actuel + la consigne du professeur.

    Le brouillon vit côté plateforme ; ce service reste sans état. Les
    sections non visées par la consigne sont conservées mot pour mot — sur
    CPU, chaque section réécrite coûte des minutes.
    """

    request_id: str
    course_id: str = Field(min_length=1, max_length=255)
    scope: Scope
    instruction: str = Field(min_length=3, max_length=4000)
    strictness: Literal["grounded", "enriched"] = "grounded"
    title: str = Field(min_length=1, max_length=500)
    sections: List[AdjustSection] = Field(min_length=1)
    # Ce que le professeur demande : « revois la partie 2, ajoute des
    # exercices, retire l'anecdote »…
    request: str = Field(min_length=3, max_length=4000)
    # Les dernières consignes de la conversation, tirées de la table de
    # discussion côté plateforme. Sans elles, chaque tour serait amnésique :
    # « comme je t'ai dit, garde un ton simple » ne marcherait pas. Le service
    # les lit et ne les stocke jamais.
    history: List[Dict[str, str]] = Field(default_factory=list, max_length=20)


class PlanChild(Wire):
    """Une sous-partie proposée : un titre seul.

    Son annonce, c'est la description de son parent — chaque sous-partie
    correspond à un élément qui y est annoncé.
    """

    heading: str


class PlanItem(Wire):
    """Une partie du plan proposé. Sans enfants, c'est une feuille : elle
    recevra un contenu. Avec enfants, son contenu EST ses enfants."""

    heading: str
    description: str = ""
    children: List[PlanChild] = []


class PlanRequest(Wire):
    """Proposer — ou réviser — le plan, sans rédiger une ligne.

    Le professeur discute le plan et le valide AVANT de payer le moindre
    contenu. Pour une révision : joindre currentPlan, request et history.
    """

    request_id: str
    course_id: str = Field(min_length=1, max_length=255)
    scope: Scope
    instruction: str = Field(min_length=3, max_length=4000)
    strictness: Literal["grounded", "enriched"] = "grounded"
    current_plan: Optional[Dict[str, Any]] = None
    request: Optional[str] = Field(default=None, max_length=4000)
    history: List[Dict[str, str]] = Field(default_factory=list, max_length=20)


class SectionRequest(Wire):
    """Rédiger — ou réviser — UNE section du plan validé.

    Le plan complet et les résumés des sections déjà validées accompagnent
    l'appel : des sections rédigées séparément doivent rester un seul cours.
    Pour une révision : joindre currentText, request et history.
    """

    request_id: str
    course_id: str = Field(min_length=1, max_length=255)
    scope: Scope
    instruction: str = Field(min_length=3, max_length=4000)
    strictness: Literal["grounded", "enriched"] = "grounded"
    heading: str = Field(min_length=1, max_length=500)
    # La description VALIDÉE de la section — telle que sauvée par la
    # plateforme, y compris si le professeur l'a corrigée à la main. C'est le
    # contrat de contenu que la rédaction doit tenir.
    description: str = ""
    plan_headings: List[str] = Field(min_length=1, max_length=20)
    previous_summaries: List[Dict[str, str]] = Field(default_factory=list, max_length=20)
    current_text: Optional[str] = None
    request: Optional[str] = Field(default=None, max_length=4000)
    history: List[Dict[str, str]] = Field(default_factory=list, max_length=20)


BlockKind = Literal["resume", "exercices", "quiz"]


class BlocksRequest(Wire):
    """Les trois blocs d'un cours — résumé, exercices, quiz — depuis son
    contenu VALIDÉ. C'est le prof qui décide lesquels existent et lesquels
    sont « demandés » à l'élève ; ici on ne fait que les produire, pour
    qu'il les relise et les valide comme une section."""

    request_id: str
    course_id: str = Field(min_length=1, max_length=255)
    scope: Scope
    kind: BlockKind
    # Le contenu validé du cours : une section, ou les sections assemblées.
    text: str = Field(min_length=50, max_length=60_000)
    # Nombre de questions / d'exercices souhaité.
    count: int = Field(default=5, ge=1, le=20)
    # Consigne du prof (« insiste sur les similitudes de rapport 1 »).
    instruction: str = Field(default="", max_length=2000)


class QuizQuestion(Wire):
    question: str
    # Toujours quatre propositions, une seule juste.
    choices: List[str]
    # Index (0-3) de la bonne réponse.
    answer: int
    # Pourquoi c'est la bonne — l'élève apprend aussi en se trompant.
    explanation: str = ""


class Exercise(Wire):
    statement: str
    # Corrigé pas à pas.
    solution: str
    difficulty: Literal["facile", "moyen", "difficile"] = "moyen"


AssessmentKind = Literal["devoir", "composition", "examen-blanc"]


class AssessmentSource(Wire):
    """Un cours couvert par l'épreuve — son titre et sa matière première.

    Envoyer le RÉSUMÉ du cours plutôt que son texte entier : plusieurs cours
    complets ne tiennent pas dans la fenêtre du modèle, et le résumé a été
    produit puis validé pour exactement cet usage.
    """

    heading: str = Field(min_length=1, max_length=500)
    text: str = Field(min_length=20, max_length=20_000)


class AssessmentRequest(Wire):
    """Un devoir, une composition ou un examen blanc sur PLUSIEURS cours.

    Le prof compose : quels cours, combien de temps, sur combien de points.
    Il relit et valide ensuite, comme tout le reste.
    """

    request_id: str
    course_id: str = Field(default="", max_length=255)
    scope: Scope
    kind: AssessmentKind
    title: str = Field(default="", max_length=300)
    sources: List[AssessmentSource] = Field(min_length=1, max_length=12)
    duration_minutes: int = Field(default=60, ge=10, le=300)
    total_points: int = Field(default=20, ge=5, le=100)
    exercise_count: int = Field(default=3, ge=1, le=10)
    instruction: str = Field(default="", max_length=2000)


class AssessmentExercise(Wire):
    statement: str
    solution: str
    points: int
    # Les cours d'où sort l'exercice — le prof voit la couverture.
    covers: List[str] = []


class AssessmentDraft(Wire):
    title: str
    instructions: str = ""
    duration_minutes: int = 60
    total_points: int = 20
    exercises: List[AssessmentExercise] = []


class GenerateAccepted(Wire):
    contract_version: Literal["1.0"] = CONTRACT_VERSION
    request_id: str
    job_id: str


class GeneratedSection(Wire):
    heading: str
    text: str
    # Les sources réellement citées par la section : documentId + locator,
    # la page du document d'origine.
    citations: List[Dict[str, Any]] = []
    has_additions: bool = False


class GenerateStatus(Wire):
    contract_version: Literal["1.0"] = CONTRACT_VERSION
    job_id: str
    status: Literal["running", "done", "failed"]
    title: Optional[str] = None
    sections: List[GeneratedSection] = []
    # Rendu par /generate/plan : le plan proposé — sa description (le bref
    # résumé de ce que le cours couvrira) et ses parties hiérarchiques,
    # exactement la forme que la plateforme copiera à la validation.
    description: str = ""
    items: List[PlanItem] = []
    # Les recherches que l'IA a faites pour construire le cours — on sait
    # toujours comment un cours a été construit.
    queries: List[Dict[str, Any]] = []
    # Rendus par /generate/blocks selon le bloc demandé.
    kind: Optional[str] = None
    summary: Optional[str] = None
    # Rendu par /generate/assessment.
    assessment: Optional[AssessmentDraft] = None
    quiz: List[QuizQuestion] = []
    exercises: List[Exercise] = []
    warnings: List[str] = []
    error: Optional[str] = None


# ---------------------------------------------------------------------- answer


class TutorTurn(Wire):
    """Un tour du fil élève ↔ Lawal — l'historique vit côté plateforme."""

    role: Literal["eleve", "lawal"]
    content: str = Field(min_length=1, max_length=4_000)


class AnswerRequest(Wire):
    request_id: str
    # Le cours depuis lequel l'élève pose sa question — jamais choisi par le
    # client final : la plateforme le déduit de la page où il se trouve.
    course_id: str = Field(min_length=1)
    # Le périmètre vient du COMPTE de l'élève et de son inscription : c'est
    # lui qui adapte le niveau de langue (classe) et verrouille le pays.
    scope: Scope
    question: str = Field(min_length=1, max_length=2_000)
    # L'élève peut interroger le cours globalement ou UNE section : quand la
    # question part d'une section, son titre arrive ici — Lawal cherche et
    # répond dans ce contexte. Vide = question sur le cours entier.
    section_heading: str = Field(default="", max_length=300)
    history: List[TutorTurn] = Field(default_factory=list, max_length=20)


class AnswerAccepted(Wire):
    contract_version: Literal["1.0"] = CONTRACT_VERSION
    request_id: str
    job_id: str
    status: Literal["running"] = "running"


class TutorCitation(Wire):
    label: str
    document_id: str
    title: str
    locator: str


class AnswerStatus(Wire):
    contract_version: Literal["1.0"] = CONTRACT_VERSION
    job_id: str
    status: Literal["running", "done", "failed"]
    answer: str = ""
    # La petite question finale : vérifier que le concept est compris.
    check: str = ""
    # Les notions abordées — la prise pour accrocher un jour une vidéo de
    # démonstration par concept.
    concepts: List[str] = []
    citations: List[TutorCitation] = []
    queries: List[Dict[str, Any]] = []
    warnings: List[str] = []
    error: Optional[str] = None


# ---------------------------------------------------------------------- speech


class SpeechRequest(Wire):
    request_id: str
    # Pour les journaux et le suivi — le rag ne vérifie pas le statut du
    # cours : la garde « cours publié seulement » vit côté plateforme.
    course_id: str = ""
    # Le texte du cours tel que validé (marqueurs compris) : la verbalisation
    # — formules dites en français, marqueurs retirés — se fait ici.
    text: str = Field(min_length=1, max_length=400_000)


class SpeechAccepted(Wire):
    contract_version: Literal["1.0"] = CONTRACT_VERSION
    request_id: str
    job_id: str
    status: Literal["running"] = "running"


class SpeechStatus(Wire):
    contract_version: Literal["1.0"] = CONTRACT_VERSION
    job_id: str
    status: Literal["running", "done", "failed"]
    # mp3 (mono 64 kbit/s) en temps normal ; wav si ffmpeg manque — le champ
    # format fait foi, l'appelant stocke tel quel.
    format: Optional[str] = None
    audio_base64: Optional[str] = None
    seconds: Optional[float] = None
    characters: Optional[int] = None
    error: Optional[str] = None


# ------------------------------------------------------------------- documents


class DocumentSummary(Wire):
    """Ce qu'affiche l'écran de création : un titre, une taille, une date."""

    document_id: str
    course_id: str = ""
    role: str = "support-cours"
    title: str
    source_reference: str = ""
    characters: int
    chunks: int
    embedding_model: str
    indexed_at: str


class DocumentListResponse(Wire):
    contract_version: Literal["1.0"] = CONTRACT_VERSION
    total: int
    documents: List[DocumentSummary]


class DocumentDetail(DocumentSummary):
    """Le document avec son texte complet, pour relecture."""

    scope: Scope
    content: str


# ----------------------------------------------------------------------- santé


class HealthResponse(Wire):
    service: str
    status: Literal["ok", "degraded"]
    timestamp: str
    dependencies: Optional[Dict[str, Literal["up", "down"]]] = None
