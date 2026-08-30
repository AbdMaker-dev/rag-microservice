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
    warnings: List[str] = []


# --------------------------------------------------------------------- indexer


DocumentRole = Literal["support-cours", "programme-officiel"]


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
