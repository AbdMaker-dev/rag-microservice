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
from app.config import Settings, get_settings
from app.core.proposal import discuter, proposer
from app.models.schemas import (
    ProposalChatRequest,
    ProposalChatResponse,
    ProposalRequest,
    ProposalResponse,
    ProposedEdit,
    RejectedEdit,
)

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


@router.post("/proposal/chat", response_model=ProposalChatResponse)
async def proposal_chat(
    body: ProposalChatRequest,
    request: Request,
    settings: Settings = Depends(get_settings),
) -> ProposalChatResponse:
    """Un tour de discussion sur le texte extrait. N'applique rien.

    Le professeur DIT ce qu'il veut changer plutôt que de sélectionner un
    passage. Le modèle voit le texte entier pour comprendre de quoi il
    parle, et ne rend que des remplacements situés — vérifiés un par un
    contre le texte avant d'être rendus.

    L'historique arrive de management, qui le conserve : le rag reste sans
    état, il ne se souvient d'aucun tour.
    """

    resultat = await discuter(
        request.app.state.llm,
        body.text,
        body.instruction,
        historique=[tour.model_dump() for tour in body.history],
        timeout=settings.generation_timeout_s,
        num_ctx=settings.generation_context_tokens,
        num_predict=settings.generation_output_tokens,
    )
    logger.info(
        "tour de relecture rendu",
        extra={
            "requestId": body.request_id,
            "corrections": len(resultat.corrections),
            "refusees": len(resultat.refusees),
            "symbolesModifies": [
                s for c in resultat.corrections for s in c.symboles_modifies
            ],
        },
    )
    return ProposalChatResponse(
        reply=resultat.reponse,
        edits=[
            ProposedEdit(
                before=c.avant,
                after=c.apres,
                position=c.position,
                changed_symbols=c.symboles_modifies,
                warning=c.avertissement,
            )
            for c in resultat.corrections
        ],
        rejected=[
            RejectedEdit(before=r["avant"], after=r["apres"], reason=r["raison"])
            for r in resultat.refusees
        ],
    )
