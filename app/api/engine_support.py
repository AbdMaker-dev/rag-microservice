"""Le moteur IA de CHAQUE demande, et sa trace jusqu'au résultat.

Les routes n'utilisent plus `app.state.llm` directement : elles demandent
ici le modèle qui correspond au moteur reçu (local par défaut), et la
tâche garde la trace de ce qui a réellement écrit — rendue au sondage,
jobs locaux compris, pour que management n'ait jamais à deviner.
"""

from __future__ import annotations

from typing import Optional

from fastapi import Request

from app.core.engines import EngineTrace, resolve_llm
from app.models.schemas import EngineUsed


def engine_for(body, request: Request):
    """(llm, trace) pour cette demande."""

    return resolve_llm(
        getattr(body, "engine", None),
        # Absent quand le tuteur est instancié sans lui (tests) : le local
        # reste alors celui du composant.
        getattr(request.app.state, "llm", None),
        getattr(request.app.state, "http", None),
    )


def submit_traced(request: Request, trace: EngineTrace, work, *, lane: str):
    """Mettre la tâche en file — ou HORS file si elle part en ligne.

    Une réponse de Claude prend quelques secondes : l'aligner derrière une
    génération locale de cinq minutes annulerait tout l'intérêt du moteur
    en ligne. Seules les tâches locales se partagent le processeur.
    """

    async def traced():
        result = await work()
        warnings = getattr(result, "warnings", None)
        if trace.fell_back and isinstance(warnings, list):
            for warning in trace.warnings():
                if warning not in warnings:
                    warnings.append(warning)
        return result

    job = request.app.state.jobs.submit(
        traced, lane=lane, exclusive=trace.provider == "local"
    )
    job.engine = trace
    return job


def engine_used(trace: Optional[EngineTrace]) -> Optional[EngineUsed]:
    if trace is None:
        return None
    return EngineUsed(
        provider=trace.provider,
        model=trace.model,
        fallback=trace.fell_back,
        fallback_reason=trace.errors[-1][:200] if trace.fell_back and trace.errors else None,
    )
