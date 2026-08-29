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


class ExtractedSection(Wire):
    """Un bloc de texte et l'endroit d'où il vient."""

    position: int
    locator: str
    text: str
    characters: int


class ExtractionQuality(Wire):
    """Ce que vaut le texte rendu, pour que le prof sache quoi relire."""

    score: float
    word_plausibility: float
    cid_markers: int
    # Nombre de mots dont l'encodage a été rétabli, et polices qu'aucune table
    # connue n'a rendues lisibles : le prof sait alors où porter son attention.
    words_repaired: int = 0
    unreadable_fonts: List[str] = []


class ExtractResponse(Wire):
    contract_version: Literal["1.0"] = CONTRACT_VERSION
    request_id: str
    filename: str
    media_type: str
    text: str
    characters: int
    sections: List[ExtractedSection]
    quality: ExtractionQuality
    warnings: List[str] = []


# --------------------------------------------------------------------- indexer


class IndexRequest(Wire):
    request_id: str
    # Identifiant du document côté plateforme. Réindexer le même identifiant
    # remplace ses passages : l'appel est donc rejouable sans risque.
    document_id: str = Field(min_length=1, max_length=255)
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
