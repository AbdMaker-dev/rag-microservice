"""POST /generate — la note du professeur devient un cours, en asynchrone.

La rédaction prend plusieurs minutes sur CPU : l'appel rend immédiatement un
identifiant de tâche, et GET /generate/{jobId} rend le statut puis le cours.

Pendant la rédaction, le modèle interroge la base autant qu'il veut — chaque
recherche reste verrouillée sur le périmètre et le cours reçus ici. Les
questions qu'il s'est posées sont rendues avec le cours : on sait toujours
comment un cours a été construit.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.api.dependencies import require_service_token
from app.config import Settings, get_settings
from app.core.generation import CourseGenerator, GeneratedCourse
from app.models.schemas import (
    AdjustRequest,
    GenerateAccepted,
    GenerateRequest,
    GenerateStatus,
    GeneratedSection,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["generate"], dependencies=[Depends(require_service_token)])


@router.post(
    "/generate", response_model=GenerateAccepted, status_code=status.HTTP_202_ACCEPTED
)
async def generate(
    body: GenerateRequest,
    request: Request,
    settings: Settings = Depends(get_settings),
) -> GenerateAccepted:
    generator = CourseGenerator(
        llm=request.app.state.llm,
        retriever=request.app.state.retriever,
        settings=settings,
    )

    job = request.app.state.jobs.submit(
        lambda: generator.generate(
            instruction=body.instruction,
            scope=body.scope,
            course_id=body.course_id,
            strictness=body.strictness,
        )
    )
    logger.info(
        "génération lancée",
        extra={"requestId": body.request_id, "job": job.id, "mode": body.strictness},
    )
    return GenerateAccepted(request_id=body.request_id, job_id=job.id)


@router.get("/generate/{job_id}", response_model=GenerateStatus)
async def generation_status(job_id: str, request: Request) -> GenerateStatus:
    job = request.app.state.jobs.get(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail={"code": "JOB_NOT_FOUND"}
        )
    if job.status != "done":
        return GenerateStatus(job_id=job.id, status=job.status, error=job.error)

    course: GeneratedCourse = job.result  # type: ignore[assignment]
    return GenerateStatus(
        job_id=job.id,
        status="done",
        title=course.title,
        sections=[
            GeneratedSection(
                heading=section.heading,
                text=section.text,
                citations=section.citations,
                has_additions=section.has_additions,
            )
            for section in course.sections
        ],
        queries=course.queries,
        warnings=course.warnings,
    )


@router.post(
    "/generate/adjust",
    response_model=GenerateAccepted,
    status_code=status.HTTP_202_ACCEPTED,
)
async def adjust(
    body: AdjustRequest,
    request: Request,
    settings: Settings = Depends(get_settings),
) -> GenerateAccepted:
    """Réviser un cours généré, sur consigne du professeur — le « chat ».

    Autant d'allers-retours que nécessaire jusqu'au bon cours final ; chaque
    appel rend le cours complet révisé, sections intactes comprises.
    """

    generator = CourseGenerator(
        llm=request.app.state.llm,
        retriever=request.app.state.retriever,
        settings=settings,
    )
    job = request.app.state.jobs.submit(
        lambda: generator.adjust(
            title=body.title,
            sections=[section.model_dump() for section in body.sections],
            request=body.request,
            instruction=body.instruction,
            scope=body.scope,
            course_id=body.course_id,
            strictness=body.strictness,
        )
    )
    logger.info(
        "révision lancée",
        extra={"requestId": body.request_id, "job": job.id},
    )
    return GenerateAccepted(request_id=body.request_id, job_id=job.id)
