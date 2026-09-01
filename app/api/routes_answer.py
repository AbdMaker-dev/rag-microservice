"""POST /answer — Lawal répond à un élève, depuis le contenu validé.

Asynchrone comme la génération : une réponse coûte des dizaines de secondes
sur CPU. La plateforme relaie la question (le périmètre vient du COMPTE de
l'élève, jamais du client), conserve le fil, et renvoie l'historique à
chaque tour — le rag reste sans état.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.api.dependencies import require_service_token
from app.models.schemas import (
    AnswerAccepted,
    AnswerRequest,
    AnswerStatus,
    TutorCitation,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["answer"], dependencies=[Depends(require_service_token)])


@router.post("/answer", response_model=AnswerAccepted, status_code=status.HTTP_202_ACCEPTED)
async def answer(body: AnswerRequest, request: Request) -> AnswerAccepted:
    tutor = request.app.state.tutor

    async def work():
        result = await tutor.answer(
            question=body.question,
            scope=body.scope,
            course_id=body.course_id,
            section_heading=body.section_heading,
            history=[turn.model_dump() for turn in body.history],
        )
        logger.info(
            "réponse du tuteur",
            extra={
                "requestId": body.request_id,
                "courseId": body.course_id,
                "queries": len(result.queries),
                "warnings": result.warnings,
            },
        )
        return result

    job = request.app.state.jobs.submit(work)
    return AnswerAccepted(request_id=body.request_id, job_id=job.id)


@router.get("/answer/{job_id}", response_model=AnswerStatus)
async def answer_status(job_id: str, request: Request) -> AnswerStatus:
    job = request.app.state.jobs.get(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "JOB_NOT_FOUND"},
        )
    if job.status == "failed":
        return AnswerStatus(job_id=job.id, status="failed", error=job.error)
    if job.status != "done":
        return AnswerStatus(job_id=job.id, status="running")
    result = job.result
    return AnswerStatus(
        job_id=job.id,
        status="done",
        answer=result.text,
        check=result.check,
        concepts=result.concepts,
        citations=[TutorCitation(**c) for c in result.citations],
        queries=result.queries,
        warnings=result.warnings,
    )
