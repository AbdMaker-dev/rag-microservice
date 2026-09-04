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
from app.core.generation import BlocksDraft, CourseGenerator, GeneratedCourse, PlanDraft
from app.models.schemas import (
    AdjustRequest,
    BlocksRequest,
    Exercise,
    QuizQuestion,
    PlanChild,
    PlanItem,
    PlanRequest,
    SectionRequest,
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

    if isinstance(job.result, PlanDraft):
        plan: PlanDraft = job.result
        return GenerateStatus(
            job_id=job.id,
            status="done",
            title=plan.title,
            description=plan.description,
            items=[
                PlanItem(
                    heading=item.heading,
                    description=item.description,
                    children=[PlanChild(heading=child) for child in item.children],
                )
                for item in plan.items
            ],
            queries=plan.queries,
            warnings=plan.warnings,
        )

    if isinstance(job.result, BlocksDraft):
        blocks: BlocksDraft = job.result
        return GenerateStatus(
            job_id=job.id,
            status="done",
            kind=blocks.kind,  # type: ignore[arg-type]
            summary=blocks.summary or None,
            quiz=[QuizQuestion(**q) for q in blocks.quiz],
            exercises=[Exercise(**e) for e in blocks.exercises],
            warnings=blocks.warnings,
        )

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
            history=body.history,
        )
    )
    logger.info(
        "révision lancée",
        extra={"requestId": body.request_id, "job": job.id},
    )
    return GenerateAccepted(request_id=body.request_id, job_id=job.id)


@router.post(
    "/generate/plan",
    response_model=GenerateAccepted,
    status_code=status.HTTP_202_ACCEPTED,
)
async def plan(
    body: PlanRequest,
    request: Request,
    settings: Settings = Depends(get_settings),
) -> GenerateAccepted:
    """Étape 1 du mode progressif : le plan, discutable, avant tout contenu."""

    generator = CourseGenerator(
        llm=request.app.state.llm,
        retriever=request.app.state.retriever,
        settings=settings,
    )
    job = request.app.state.jobs.submit(
        lambda: generator.draft_plan(
            instruction=body.instruction,
            scope=body.scope,
            course_id=body.course_id,
            current_plan=body.current_plan,
            request=body.request,
            history=body.history,
        )
    )
    logger.info("plan lancé", extra={"requestId": body.request_id, "job": job.id})
    return GenerateAccepted(request_id=body.request_id, job_id=job.id)


@router.post(
    "/generate/section",
    response_model=GenerateAccepted,
    status_code=status.HTTP_202_ACCEPTED,
)
async def section(
    body: SectionRequest,
    request: Request,
    settings: Settings = Depends(get_settings),
) -> GenerateAccepted:
    """Étape 2 : le contenu d'UN item du plan validé, à la demande.

    Le professeur avance section par section — génération, chat de révision,
    validation — jusqu'à la conclusion. Une à deux minutes par section, au
    moment où il la demande.
    """

    generator = CourseGenerator(
        llm=request.app.state.llm,
        retriever=request.app.state.retriever,
        settings=settings,
    )
    job = request.app.state.jobs.submit(
        lambda: generator.write_one_section(
            heading=body.heading,
            description=body.description,
            instruction=body.instruction,
            scope=body.scope,
            course_id=body.course_id,
            strictness=body.strictness,
            plan_headings=body.plan_headings,
            previous_summaries=body.previous_summaries,
            current_text=body.current_text,
            request=body.request,
            history=body.history,
        )
    )
    logger.info(
        "section lancée",
        extra={"requestId": body.request_id, "job": job.id, "heading": body.heading},
    )
    return GenerateAccepted(request_id=body.request_id, job_id=job.id)


@router.post(
    "/generate/blocks",
    response_model=GenerateAccepted,
    status_code=status.HTTP_202_ACCEPTED,
)
async def blocks(
    body: BlocksRequest,
    request: Request,
    settings: Settings = Depends(get_settings),
) -> GenerateAccepted:
    """Les trois blocs d'un cours — résumé, exercices, quiz — depuis son
    contenu validé. Un appel par bloc ; le prof relit et valide, comme une
    section. Le statut se lit sur GET /generate/{jobId}."""

    generator = CourseGenerator(
        llm=request.app.state.llm,
        retriever=request.app.state.retriever,
        settings=settings,
    )
    job = request.app.state.jobs.submit(
        lambda: generator.generate_blocks(
            kind=body.kind,
            text=body.text,
            scope=body.scope,
            count=body.count,
            instruction=body.instruction,
        )
    )
    logger.info(
        "bloc lancé",
        extra={"requestId": body.request_id, "job": job.id, "kind": body.kind},
    )
    return GenerateAccepted(request_id=body.request_id, job_id=job.id)
