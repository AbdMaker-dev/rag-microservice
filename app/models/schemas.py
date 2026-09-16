"""Contrats d'entrée et de sortie de l'API.

Convention : camelCase sur le fil, snake_case en Python.
Chaque réponse porte `contractVersion` pour qu'une rupture soit visible.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator

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


class EngineChoice(Wire):
    """Le moteur IA choisi par le super admin pour ce pays et cet usage.

    Envoyé par management avec la demande ; absent = modèle local. La clé
    est un `SecretStr` : elle ne s'affiche ni dans un journal ni dans une
    erreur de validation.
    """

    provider: Literal["local", "claude", "gpt", "gemini"] = "local"
    model: str = Field(default="", max_length=100)
    api_key: SecretStr = Field(default=SecretStr(""), max_length=500)

    @model_validator(mode="after")
    def _online_needs_model_and_key(self):
        if self.provider != "local":
            if not self.model.strip():
                raise ValueError("un moteur en ligne demande un modèle")
            if len(self.api_key.get_secret_value()) < 10:
                raise ValueError("un moteur en ligne demande une clé d'API")
        return self


class EngineUsed(Wire):
    """Ce qui a réellement écrit. `fallback` : le fournisseur en ligne est
    tombé et le modèle local a pris le relais."""

    provider: str
    model: str
    fallback: bool = False
    # Pourquoi le fournisseur est tombé (extrait de sa réponse, sans la clé) :
    # le 16/09/2026, un modèle retiré (404) retombait en local sans que le
    # super admin le voie.
    fallback_reason: Optional[str] = None


class EngineTestRequest(Wire):
    engine: EngineChoice


class EngineModelsRequest(Wire):
    provider: Literal["claude", "gpt", "gemini"]
    api_key: SecretStr = Field(max_length=500)


class EngineModel(Wire):
    id: str
    name: str


class EngineModelsResponse(Wire):
    models: List[EngineModel] = []
    error: Optional[str] = None


class EngineTestResponse(Wire):
    ok: bool
    latency_ms: int
    error: Optional[str] = None


class ExtractRequest(Wire):
    request_id: str
    filename: str = Field(min_length=1, max_length=255)
    media_type: MediaType
    # Contenu du fichier encodé en base64. Le service ne télécharge jamais
    # une URL : la plateforme envoie les octets qu'elle a le droit d'utiliser.
    content_base64: str = Field(min_length=1)


# ─────────────────────────── LE CAHIER DE L'ÉLÈVE ───────────────────────────


class NotebookRepairRequest(Wire):
    """Le texte lu dans le cahier, à confronter au contenu validé.

    Le rag ne reçoit JAMAIS de photo : management lit les pages, le rag
    travaille sur du texte. Le périmètre vient du compte de l'élève.
    """

    request_id: str
    student_account_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    scope: Scope
    chapter: str = ""


class NotebookProof(Wire):
    """D'où vient une correction — Awa doit pouvoir le lire."""

    text: str
    title: str
    locator: str


class NotebookSegment(Wire):
    ordinal: int
    original: str
    text: str
    # inchange | corrige | a-verifier
    status: str
    reason: str = ""
    proof: Optional[NotebookProof] = None


class NotebookRepairResponse(Wire):
    """Le résultat de la confrontation au contenu validé.

    `segments` fait foi. `text` n'est qu'un APERÇU — la transcription si
    TOUTES les corrections étaient acceptées. L'indexer reviendrait à
    accepter à la place de l'élève, ce que ce contrat interdit : c'est à
    l'appelant de reconstruire le texte depuis les décisions de l'élève,
    passage par passage.
    """

    contract_version: Literal["1.0"] = CONTRACT_VERSION
    request_id: str
    text: str
    segments: List[NotebookSegment] = []
    corrected: int = 0
    to_check: int = 0
    proven_from: List[str] = []
    warnings: List[str] = []


class NotebookGenerateRequest(Wire):
    """Produire de quoi réviser SUR le cours que l'élève a validé.

    Le texte envoyé est celui qu'il a approuvé passage par passage — pas la
    transcription brute, pas l'aperçu tout corrigé.
    """

    request_id: str
    student_account_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    kind: Literal["resume", "exercices", "quiz"]
    text: str = Field(min_length=1)
    scope: Scope
    count: int = Field(default=5, ge=1, le=20)
    # Moteur IA du pays pour cet usage — absent : modèle local.
    engine: Optional[EngineChoice] = None


class NotebookIndexRequest(Wire):
    """Ranger le cours d'un élève dans SA base, après qu'il a validé."""

    request_id: str
    document_id: str = Field(min_length=1)
    student_account_id: str = Field(min_length=1)
    title: str = Field(min_length=1, max_length=300)
    chapter: str = ""
    scope: Scope
    text: str = Field(min_length=1)


class NotebookIndexResponse(Wire):
    contract_version: Literal["1.0"] = CONTRACT_VERSION
    request_id: str
    document_id: Optional[str] = None
    chunks: int = 0
    warnings: List[str] = []


class NotebookDeleteResponse(Wire):
    contract_version: Literal["1.0"] = CONTRACT_VERSION
    document_id: str
    deleted: bool


class PlanningPeriod(Wire):
    """Quand un chapitre se traite. Un chapitre peut déborder sur le mois
    suivant : il a alors deux périodes, pas deux entrées."""

    month: Optional[str] = None
    weeks: List[int] = []


class PlanningEntryOut(Wire):
    position: int
    chapter: str
    details: str = ""
    part: str = ""
    objectives: List[str] = []
    assessments: List[str] = []
    periods: List[PlanningPeriod] = []


class PlanningResponse(Wire):
    """Le planning officiel, lu sans qu'aucun modèle n'intervienne.

    L'en-tête (année, matière, niveau) peut manquer : il part alors en
    `warnings` et c'est l'admin qui complète. On ne devine pas l'année
    scolaire d'un document officiel.
    """

    contract_version: Literal["1.0"] = CONTRACT_VERSION
    request_id: str
    filename: str
    school_year: Optional[str] = None
    subject: Optional[str] = None
    grade_label: Optional[str] = None
    weekly_hours: Optional[int] = None
    entries: List[PlanningEntryOut] = []
    warnings: List[str] = []


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


# ------------------------------------------------- proposer, sans appliquer


class ProposalRequest(Wire):
    """Un seul passage à la fois — jamais le document entier."""

    request_id: str
    passage: str = Field(min_length=1, max_length=4000)
    # Ce que l'extraction a signalé sur ce passage (`THIN`, `FORMULA`…).
    # Transmis au modèle comme indice, jamais comme instruction.
    issues: List[str] = []
    # Moteur IA du pays pour cet usage — absent : modèle local.
    engine: Optional[EngineChoice] = None


class ProposalResponse(Wire):
    """La proposition ET l'original. Le service ne remplace jamais.

    `changedSymbols` est le champ qui compte : une réparation de
    transcription ne touche ni un chiffre ni un opérateur. S'il n'est pas
    vide, la proposition change le SENS, et l'écran doit le dire avant que
    le professeur accepte.
    """

    contract_version: Literal["1.0"] = CONTRACT_VERSION
    passage: str
    proposal: Optional[str] = None
    changed: bool = False
    uncertain: bool = False
    changed_symbols: List[str] = []
    warning: Optional[str] = None
    engine: Optional[EngineUsed] = None


class ProposalTurn(Wire):
    """Un tour de la discussion, tel que management le conserve."""

    role: Literal["prof", "ia"]
    content: str


class ProposalChatRequest(Wire):
    """Le texte ENTIER, pour que « cette partie-là » veuille dire quelque chose.

    Le modèle lit tout et ne rend que des remplacements ciblés : c'est ce qui
    permet de lui parler sans lui laisser réécrire le document.
    """

    request_id: str
    text: str = Field(min_length=1, max_length=120_000)
    instruction: str = Field(min_length=1, max_length=2_000)
    history: List[ProposalTurn] = []
    # Moteur IA du pays pour cet usage — absent : modèle local.
    engine: Optional[EngineChoice] = None


class ProposedEdit(Wire):
    before: str
    after: str
    # Où `before` commence dans le texte — l'écran n'a pas à le rechercher,
    # et il ne PEUT pas : le service a déjà vérifié qu'il s'y trouve une
    # seule fois.
    position: int
    changed_symbols: List[str] = []
    warning: Optional[str] = None


class RejectedEdit(Wire):
    before: str
    after: str
    reason: str


class ProposalChatResponse(Wire):
    contract_version: Literal["1.0"] = CONTRACT_VERSION
    # Ce que l'IA dit au professeur, en français.
    reply: str
    edits: List[ProposedEdit] = []
    # Ce qu'elle a proposé et qu'on a refusé, avec la raison. Montré : un
    # refus silencieux ferait croire qu'elle n'a rien trouvé.
    rejected: List[RejectedEdit] = []


class ProposalChatAccepted(Wire):
    """Le tour est parti en file. L'état se lit sur `GET /generate/{jobId}`.

    Asynchrone depuis le 13/09/2026 : le texte entier repart au modèle à
    chaque tour, et 3 000 caractères ont pris 53 secondes en production.
    Un chapitre de quarante pages dépasserait n'importe quel délai HTTP
    raisonnable. Même file, même sondage, même `lost` que le reste — le
    professeur voit sa place et peut quitter la page.
    """

    contract_version: Literal["1.0"] = CONTRACT_VERSION
    job_id: str


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
    # Moteur IA du pays pour cet usage — absent : modèle local.
    engine: Optional[EngineChoice] = None


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
    # Moteur IA du pays pour cet usage — absent : modèle local.
    engine: Optional[EngineChoice] = None


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
    # Le SUJET du cours — son titre. C'est avec lui que l'on cherche le
    # programme officiel et les supports ; l'instruction, elle, est une
    # consigne au rédacteur (« n'invente aucune formule ») et ne décrit rien
    # qu'on puisse chercher. Absent = on cherche avec l'instruction, comme
    # avant le 13/09/2026.
    title: Optional[str] = Field(default=None, min_length=1, max_length=500)
    strictness: Literal["grounded", "enriched"] = "grounded"
    current_plan: Optional[Dict[str, Any]] = None
    request: Optional[str] = Field(default=None, max_length=4000)
    history: List[Dict[str, str]] = Field(default_factory=list, max_length=20)
    # Moteur IA du pays pour cet usage — absent : modèle local.
    engine: Optional[EngineChoice] = None


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
    # Moteur IA du pays pour cet usage — absent : modèle local.
    engine: Optional[EngineChoice] = None


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
    # Moteur IA du pays pour cet usage — absent : modèle local.
    engine: Optional[EngineChoice] = None


class BlocksDiscussRequest(BlocksRequest):
    """Réviser un bloc sur consigne du professeur — le « chat » des blocs.

    Le document, le plan et les sections ont chacun le leur depuis le début ;
    les trois blocs n'en avaient pas, et un quiz dont une réponse est fausse
    arrivait donc intact jusqu'à l'élève. Demandé par Alioune le 14/09/2026 :
    « on fait la même chose que les autres ».

    Comme partout ailleurs, le brouillon vit côté plateforme et ce service
    reste sans état : il reçoit le bloc actuel, la consigne, l'historique,
    et rend le bloc révisé.
    """

    # Le bloc tel qu'il est aujourd'hui. Un résumé porte son texte ; un quiz
    # ou des exercices portent leurs items, dans l'ordre où le prof les voit.
    current_summary: str = Field(default="", max_length=20_000)
    current_items: List[Dict[str, Any]] = Field(default_factory=list, max_length=30)
    # L'item visé, par sa place dans la liste. C'est le cas courant, et c'est
    # une exigence d'Alioune (14/09/2026) : « c'est exo par exo, question par
    # question ». Le même principe que la relecture d'un document — l'IA
    # propose, le professeur accepte, passage par passage. Un seul item
    # révisé, c'est aussi trente secondes d'attente au lieu de cinq minutes,
    # et les autres qui ne bougent pas d'un caractère.
    # Omis : la consigne porte sur le bloc entier (« ajoute un exercice
    # difficile », « ils sont tous trop faciles »).
    target_index: Optional[int] = Field(default=None, ge=0, le=29)
    # Ce que le professeur demande : « la question 1 est fausse, le module
    # vaut racine de 2 », « remplace l'exercice 3 par un plus facile ».
    request: str = Field(min_length=3, max_length=4000)
    # Les derniers tours, tirés de la table de discussion de la plateforme.
    history: List[Dict[str, str]] = Field(default_factory=list, max_length=20)


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
    # Le plafond est large exprès : un service qui refuse une entrée un peu
    # trop longue oblige son appelant à couper au hasard, et le premier
    # devoir jamais composé sur le serveur a échoué là-dessus (14/09/2026,
    # un cours validé de 28 000 caractères). Ce qui ne tient pas dans la
    # fenêtre du modèle est borné à la composition, équitablement entre les
    # cours — voir `compose_assessment`.
    text: str = Field(min_length=20, max_length=60_000)


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
    # Moteur IA du pays pour cet usage — absent : modèle local.
    engine: Optional[EngineChoice] = None


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


class AuditRequest(Wire):
    """Auditer un cours avant publication (16/09/2026) : SymPy + correcteur."""

    request_id: str
    course_id: str = Field(min_length=1, max_length=255)
    scope: Scope
    text: str = Field(min_length=1, max_length=120_000)
    # Moteur IA du pays pour l'usage RELECTURE — absent : modèle local.
    engine: Optional[EngineChoice] = None


class AuditFindingOut(Wire):
    # certaine (SymPy) | probable | ambiguite (correcteur)
    severity: Literal["certaine", "probable", "ambiguite"]
    source: Literal["calcul", "relecture"]
    excerpt: str
    explanation: str
    correction: str = ""


class CourseAudit(Wire):
    findings: List[AuditFindingOut] = []


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
    status: Literal["queued", "running", "done", "failed"]
    # Quand la tâche attend son tour : sa place (1 = le prochain)
    # et l'attente annoncée. L'écran dit « 2e — environ 1 min 20 »
    # au lieu d'un « en cours » muet.
    queue_position: Optional[int] = None
    wait_seconds: Optional[int] = None
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
    # Rendus par /proposal/chat — un tour de relecture. `reply` est ce que
    # l'IA dit au professeur ; `edits` ce qu'elle propose de remplacer, situé
    # et vérifié ; `rejected` ce qu'on a refusé de lui montrer, avec la
    # raison. Trois champs vides sur tout autre job.
    reply: Optional[str] = None
    edits: List[ProposedEdit] = []
    rejected: List[RejectedEdit] = []
    warnings: List[str] = []
    error: Optional[str] = None
    engine: Optional[EngineUsed] = None
    audit: Optional[CourseAudit] = None


# ---------------------------------------------------------------------- answer


class TutorTurn(Wire):
    """Un tour du fil élève ↔ Lawal — l'historique vit côté plateforme."""

    role: Literal["eleve", "lawal"]
    content: str = Field(min_length=1, max_length=4_000)


class AnswerRequest(Wire):
    request_id: str
    # Le cours depuis lequel l'élève pose sa question — jamais choisi par le
    # client final : la plateforme le déduit de la page où il se trouve.
    #
    # VIDE quand la question porte sur un cours du CAHIER : c'est le cas
    # nominal du flux 2 — le professeur a fait cours en classe et rien n'est
    # publié sur la plateforme, c'est précisément pour ça que l'élève a
    # scanné. Exiger un courseId l'empêcherait de parler de ses propres
    # notes tant que son prof n'a pas publié : ce serait une limite produit
    # inventée par le contrat, pas par le besoin.
    course_id: str = ""
    # Le périmètre vient du COMPTE de l'élève et de son inscription : c'est
    # lui qui adapte le niveau de langue (classe) et verrouille le pays.
    scope: Scope
    question: str = Field(min_length=1, max_length=2_000)
    # L'élève peut interroger le cours globalement ou UNE section : quand la
    # question part d'une section, son titre arrive ici — Lawal cherche et
    # répond dans ce contexte. Vide = question sur le cours entier.
    section_heading: str = Field(default="", max_length=300)
    history: List[TutorTurn] = Field(default_factory=list, max_length=20)
    # Le cahier de l'élève, quand la question porte sur un cours qu'il a
    # lui-même ajouté. Les deux vont ensemble : sans propriétaire, aucun
    # cahier ne s'ouvre.
    student_account_id: str = ""
    notebook_document_id: str = ""
    # Moteur IA du pays pour cet usage — absent : modèle local.
    engine: Optional[EngineChoice] = None


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
    # D'où vient l'extrait : le cahier de l'élève ou le contenu validé.
    # Absent du contrat, il faisait répondre 500 à toute réponse finie.
    source: Literal["cahier", "valide"] = "valide"


class AnswerStatus(Wire):
    contract_version: Literal["1.0"] = CONTRACT_VERSION
    job_id: str
    status: Literal["queued", "running", "done", "failed"]
    # Quand la tâche attend son tour : sa place (1 = le prochain)
    # et l'attente annoncée. L'écran dit « 2e — environ 1 min 20 »
    # au lieu d'un « en cours » muet.
    queue_position: Optional[int] = None
    wait_seconds: Optional[int] = None
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
    engine: Optional[EngineUsed] = None


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
    status: Literal["queued", "running", "done", "failed"]
    # Quand la tâche attend son tour : sa place (1 = le prochain)
    # et l'attente annoncée. L'écran dit « 2e — environ 1 min 20 »
    # au lieu d'un « en cours » muet.
    queue_position: Optional[int] = None
    wait_seconds: Optional[int] = None
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
