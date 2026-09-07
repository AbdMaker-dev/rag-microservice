"""Le cahier d'Awa : réparer, ranger, puis produire de quoi réviser.

Ce module fait le lien entre trois choses qui existent déjà — la base
validée des professeurs, la réparation par preuve, et la génération de
blocs — pour un cas nouveau : un cours que l'élève a photographié.

La règle qui gouverne tout : **le cahier consomme la base prof, il ne la
nourrit jamais**. On y cherche des preuves, on n'y écrit rien.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

from app.core.chunking import chunk_document, default_token_counter
from app.core.repair import Proof, RepairResult, proof_from_hits, repair_by_proof, split_segments
from app.db.notebook_repository import NotebookRepository
from app.db.repository import ChunkRow, IndexRepository
from app.models.schemas import Scope

logger = logging.getLogger(__name__)

# Les rôles où une preuve peut se trouver : ce qu'un professeur a validé et
# ce que le ministère a publié. Jamais le cahier d'un autre élève — un
# camarade qui a mal recopié ne prouve rien.
_PROOF_ROLES = ("cours-publie", "support-cours", "programme-officiel")

# Combien de passages validés on regarde par ligne du cahier. Au-delà, on
# paie des recherches pour des candidats que le seuil rejettera de toute façon.
_PROOF_CANDIDATES = 3


@dataclass
class NotebookRepairReport:
    segments: List[dict] = field(default_factory=list)
    text: str = ""
    corrected: int = 0
    to_check: int = 0
    proven_from: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class NotebookService:
    """Réparation et indexation du cahier — sans jamais toucher la base prof."""

    def __init__(
        self,
        *,
        embeddings,
        documents: IndexRepository,
        notebooks: NotebookRepository,
        settings,
    ) -> None:
        self._embeddings = embeddings
        # La base PROF, en LECTURE seule : on y cherche des preuves, on n'y
        # écrit jamais. C'est le dépôt lui-même, pas le Retriever — celui-ci
        # filtre par score pour une recherche d'élève, alors qu'une preuve se
        # juge sur la distance et sur elle seule.
        self._documents = documents
        self._notebooks = notebooks
        self._settings = settings

    async def repair(
        self, *, text: str, scope: Scope, chapter: str = ""
    ) -> NotebookRepairReport:
        """Confronter la transcription au contenu validé du même périmètre.

        Chaque ligne est cherchée séparément : une preuve vaut pour UNE
        ligne, jamais pour un paragraphe entier — sinon on remplacerait du
        texte qu'on n'a pas vérifié.
        """

        segments = split_segments(text)
        proofs: dict = {}
        if segments:
            # Une seule passe d'embeddings pour toutes les lignes : c'est
            # l'appel coûteux, il ne doit pas se faire ligne à ligne.
            vectors = await self._embeddings.embed(segments)
            for segment, vector in zip(segments, vectors):
                hits = await self._search_proof(vector, scope)
                proofs[segment] = proof_from_hits(hits)

        result: RepairResult = repair_by_proof(text, lambda s: proofs.get(s))
        report = NotebookRepairReport(
            text=result.text(),
            corrected=result.corrected,
            to_check=result.to_check,
            warnings=list(result.warnings),
        )
        sources = []
        for segment in result.segments:
            entry = {
                "ordinal": segment.ordinal,
                "original": segment.original,
                "text": segment.text,
                "status": segment.status,
            }
            if segment.reason:
                entry["reason"] = segment.reason
            if segment.proof is not None:
                entry["proof"] = {
                    "text": segment.proof.text,
                    "title": segment.proof.title,
                    "locator": segment.proof.locator,
                }
                sources.append(segment.proof.title)
            report.segments.append(entry)

        # Les sources citées, sans doublon et dans l'ordre où elles servent :
        # Awa doit voir D'OÙ viennent les corrections qu'on lui propose.
        seen = set()
        report.proven_from = [
            title for title in sources
            if title and not (title in seen or seen.add(title))
        ]
        if not report.corrected and not report.to_check and segments:
            report.warnings.append("NOTEBOOK_NOTHING_TO_REPAIR")
        return report

    async def _search_proof(self, vector, scope: Scope) -> List[dict]:
        """Les passages validés les plus proches, tous rôles de preuve."""

        hits: List[dict] = []
        for role in _PROOF_ROLES:
            found = await self._documents.search(
                embedding=vector,
                scope=scope,
                limit=_PROOF_CANDIDATES,
                role=role,
            )
            hits.extend(found)
        return hits

    async def index(
        self,
        *,
        external_id: str,
        student_account_id: str,
        title: str,
        chapter: str,
        scope: Scope,
        text: str,
    ) -> dict:
        """Ranger le cours d'Awa dans SA base — jamais dans celle des profs."""

        pieces = chunk_document(
            text,
            max_tokens=self._settings.chunk_max_tokens,
            overlap_tokens=self._settings.chunk_overlap_tokens,
            min_tokens=self._settings.chunk_min_tokens,
            count_tokens=default_token_counter(),
        )
        if not pieces:
            return {"document_id": None, "chunks": 0, "warnings": ["NOTEBOOK_TEXT_TOO_SHORT"]}

        vectors = await self._embeddings.embed([piece.content for piece in pieces])
        dimensions = {len(vector) for vector in vectors}
        if len(dimensions) != 1:
            # pgvector ne sait pas comparer des vecteurs de tailles
            # différentes : mieux vaut refuser que ranger l'illisible.
            return {
                "document_id": None,
                "chunks": 0,
                "warnings": ["INCONSISTENT_EMBEDDING_DIMENSION"],
            }
        rows = [
            ChunkRow(
                ordinal=piece.ordinal,
                locator=piece.locator,
                content=piece.content,
                token_count=piece.token_count,
                embedding=vector,
            )
            for piece, vector in zip(pieces, vectors)
        ]
        document_id = await self._notebooks.replace_document(
            external_id=external_id,
            student_account_id=student_account_id,
            title=title,
            chapter=chapter,
            scope=scope,
            embedding_model=self._embeddings.model,
            embedding_dimension=next(iter(dimensions)),
            characters=len(text),
            chunks=rows,
        )
        logger.info(
            "cours de cahier indexé",
            extra={"documentId": external_id, "chunks": len(rows)},
        )
        return {"document_id": document_id, "chunks": len(rows), "warnings": []}
