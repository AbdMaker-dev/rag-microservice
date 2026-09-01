"""Runtime configuration.

Every value comes from the environment. Nothing here is read from the
LawalSchool monorepo: this service is deployed on its own and must never
assume a shared filesystem with the API.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List, Literal, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- Identity and environment -------------------------------------------------
    service_name: str = "rag-microservice"
    environment: Literal["development", "test", "staging", "production"] = "development"

    # Mirrors DATA_CLASSIFICATION in the monorepo. The service refuses to start
    # outside SYNTHETIC_ONLY until a human decision widens it, so that real
    # minors' data cannot reach an inference backend by configuration drift.
    data_classification: Literal["SYNTHETIC_ONLY", "PERSONAL_DATA_APPROVED"] = (
        "SYNTHETIC_ONLY"
    )

    # --- Service-to-service authentication ----------------------------------------
    # The NestJS API is the only legitimate caller. This service must never be
    # exposed to the public internet.
    service_shared_secret: str = Field(min_length=32)

    # --- Database (read-only) -----------------------------------------------------
    # Base propre au service : chunks, vecteurs et périmètres. Elle ne contient
    # ni cours publiés, ni élèves, ni droits — ceux-là restent chez LawalSchool.
    database_url: str
    database_pool_min: int = Field(default=1, ge=1, le=32)
    database_pool_max: int = Field(default=8, ge=1, le=64)
    database_statement_timeout_ms: int = Field(default=5_000, ge=100, le=60_000)

    # --- Inference backends -------------------------------------------------------
    llm_provider: Literal["ollama", "vllm", "none"] = "ollama"
    llm_model: str = "qwen2.5:7b"
    embedding_provider: Literal["ollama", "vllm"] = "ollama"
    embedding_model: str = "bge-m3"
    # Chunks vectorisés par appel au backend. Mesuré : un document entier
    # d'un bloc dépasse le délai sur CPU ; par lots, chaque appel reste court.
    embedding_batch_size: int = Field(default=16, ge=1, le=256)
    ollama_base_url: str = "http://localhost:11434"
    vllm_base_url: Optional[str] = None
    inference_timeout_s: float = Field(default=30.0, ge=1.0, le=300.0)

    # --- Génération de cours ------------------------------------------------------
    # Rédiger prend plusieurs minutes sur CPU : le délai est par appel au
    # modèle, et la génération est asynchrone (tâche + statut).
    generation_timeout_s: float = Field(default=600.0, ge=30.0, le=3600.0)
    # L'IA interroge la base autant qu'elle veut — dans cette limite : chaque
    # aller-retour coûte un appel au modèle, un plafond évite la boucle folle.
    generation_max_queries: int = Field(default=20, ge=1, le=100)
    generation_max_sections: int = Field(default=8, ge=1, le=20)
    generation_context_tokens: int = Field(default=8_192, ge=2_048, le=32_768)
    # Plafond de sortie par appel : sans lui, une section peut divaguer sur
    # des milliers de tokens — des minutes de plus sur CPU, pour du remplissage.
    generation_output_tokens: int = Field(default=1_200, ge=128, le=8_192)
    inference_max_attempts: int = Field(default=3, ge=1, le=10)

    # --- Synthèse vocale ----------------------------------------------------------
    # La voix choisie à l'oreille par Alioune sur trois échantillons
    # (31/08/2026). Le fichier .onnx est téléchargé au build de l'image ;
    # s'il manque, /speech répond 503 SPEECH_BACKEND_UNAVAILABLE — le reste
    # du service n'est pas concerné.
    piper_voice_path: str = "voices/fr_FR-siwis-medium.onnx"

    # --- Chunking -----------------------------------------------------------------
    chunk_max_tokens: int = Field(default=384, ge=64, le=2048)
    chunk_overlap_tokens: int = Field(default=64, ge=0, le=512)
    chunk_min_tokens: int = Field(default=32, ge=1, le=512)

    # --- Retrieval ----------------------------------------------------------------
    retrieval_candidates: int = Field(default=24, ge=1, le=200)
    retrieval_top_k: int = Field(default=4, ge=1, le=50)
    retrieval_max_excerpt_characters: int = Field(default=1_200, ge=100, le=10_000)
    reranker_enabled: bool = False
    reranker_model: Optional[str] = None

    # --- Ingestion limits ---------------------------------------------------------
    max_document_bytes: int = Field(default=25_000_000, ge=1_024)
    allowed_media_types: List[str] = [
        "text/plain",
        "text/markdown",
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ]

    # --- Correction de l'encodage du texte extrait ----------------------------------
    # Certains PDF déclarent une table de caractères et en encodent une autre :
    # « compétences » ressort en « compŽtences ». C'est une permutation, donc
    # elle s'inverse exactement. La correction est décidée police par police et
    # ne fait intervenir aucun modèle de langue — un modèle comblerait les trous
    # en inventant, ce qui est le pire résultat sur du contenu pédagogique.
    encoding_repair_enabled: bool = True
    # En dessous de ce score, le texte rendu est signalé comme douteux au prof.
    text_quality_floor: float = Field(default=0.90, ge=0.0, le=1.0)

    # --- Observability ------------------------------------------------------------
    # Prompts, questions and chunk contents are pedagogical material tied to
    # identified students. They are never written to logs.
    log_level: str = "INFO"
    log_redaction_enabled: bool = True

    @field_validator("chunk_overlap_tokens")
    @classmethod
    def _overlap_below_window(cls, value: int, info) -> int:
        maximum = info.data.get("chunk_max_tokens")
        if maximum is not None and value >= maximum:
            raise ValueError("chunk_overlap_tokens must stay below chunk_max_tokens")
        return value

    @field_validator("database_pool_max")
    @classmethod
    def _pool_range(cls, value: int, info) -> int:
        minimum = info.data.get("database_pool_min")
        if minimum is not None and value < minimum:
            raise ValueError("database_pool_max must be >= database_pool_min")
        return value

    @field_validator("log_redaction_enabled")
    @classmethod
    def _redaction_required_outside_development(cls, value: bool, info) -> bool:
        environment = info.data.get("environment")
        if not value and environment in {"staging", "production"}:
            raise ValueError("log redaction cannot be disabled outside development")
        return value

    @property
    def inference_base_url(self) -> str:
        if self.llm_provider == "vllm" or self.embedding_provider == "vllm":
            if not self.vllm_base_url:
                raise ValueError("VLLM_BASE_URL is required when a vLLM backend is selected")
        return self.ollama_base_url


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
