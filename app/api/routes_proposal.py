"""POST /proposal — une correction PROPOSÉE sur un passage relu.

Volontairement hors de `routes_extract.py` : l'extraction se déclare sans
modèle de langue et le reste. Ici, un modèle intervient — c'est une étape
distincte, demandée explicitement par le professeur sur un passage qu'il a
choisi, et dont la sortie n'entre nulle part sans son accord.

Synchrone, contrairement à la génération d'un cours : un passage fait
quelques lignes, pas un chapitre. Le professeur attend devant son écran,
sur le passage qu'il est en train de lire.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Request

from app.api.dependencies import require_service_token
from app.core.proposal import proposer
from app.models.schemas import ProposalRequest, ProposalResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["extract"], dependencies=[Depends(require_service_token)])


@router.post("/proposal", response_model=ProposalResponse)
async def proposal(body: ProposalRequest, request: Request) -> ProposalResponse:
    """Propose une réparation. N'applique rien, ne stocke rien."""

    resultat = await proposer(
        request.app.state.llm,
        body.passage,
        signalements=body.issues,
    )
    logger.info(
        "proposition rendue",
        extra={
            "requestId": body.request_id,
            "changee": resultat.changee,
            "incertaine": resultat.incertaine,
            # On journalise les symboles déplacés : le jour où une formule
            # fausse passe, c'est ici qu'on verra qu'elle avait été signalée.
            "symbolesModifies": resultat.symboles_modifies,
        },
    )
    return ProposalResponse(
        passage=resultat.passage,
        proposal=resultat.proposition,
        changed=resultat.changee,
        uncertain=resultat.incertaine,
        changed_symbols=resultat.symboles_modifies,
        warning=resultat.avertissement,
    )
