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
from app.api.engine_support import engine_for, engine_used, submit_traced
from app.config import Settings, get_settings
from app.core.course_audit import AuditResult, audit_course
from app.core.proposal import Discussion
from app.core.generation import (
    AssessmentResult,
    BlocksDraft,
    CourseGenerator,
    GeneratedCourse,
    PlanDraft,
)
from app.models.schemas import (
    AuditFindingOut,
    AuditRequest,
    AuditTarget,
    CourseAudit,
    AdjustRequest,
    AssessmentDraft,
    AssessmentExercise,
    AssessmentRequest,
    BlocksDiscussRequest,
    BlocksRequest,
    Exercise,
    QuizQuestion,
    PlanChild,
    PlanItem,
    ProposedEdit,
    RejectedEdit,
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
    engine_llm, trace = engine_for(body, request)
    generator = CourseGenerator(
        llm=engine_llm,
        retriever=request.app.state.retriever,
        settings=settings,
    )

    job = submit_traced(
        request,
        trace,
        lambda: generator.generate(
            instruction=body.instruction,
            scope=body.scope,
            course_id=body.course_id,
            strictness=body.strictness,
        ),
        lane="prof",
    )
    logger.info(
        "génération lancée",
        extra={"requestId": body.request_id, "job": job.id, "mode": body.strictness},
    )
    return GenerateAccepted(request_id=body.request_id, job_id=job.id)


@router.get("/generate/{job_id}", response_model=GenerateStatus)
async def generation_status(job_id: str, request: Request) -> GenerateStatus:
    response = await _generation_status(job_id, request)
    job = request.app.state.jobs.get(job_id)
    # Le moteur qui a écrit, sur TOUT résultat — jobs locaux compris.
    response.engine = engine_used(getattr(job, "engine", None))
    return response


async def _generation_status(job_id: str, request: Request) -> GenerateStatus:
    job = request.app.state.jobs.get(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail={"code": "JOB_NOT_FOUND"}
        )
    if job.status != "done":
        # En file d'attente, on dit la place et le temps — jamais un « en
        # cours » muet pendant que le professeur regarde son écran.
        store = request.app.state.jobs
        return GenerateStatus(
            job_id=job.id,
            status=job.status,
            error=job.error,
            queue_position=store.position(job) or None,
            wait_seconds=store.wait_seconds(job),
        )

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

    if isinstance(job.result, Discussion):
        # Un tour de relecture d'un document. Les trois champs partent tels
        # quels jusqu'au front : `rejected` surtout — un refus caché ferait
        # croire au professeur que l'IA n'a rien trouvé.
        tour: Discussion = job.result
        return GenerateStatus(
            job_id=job.id,
            status="done",
            reply=tour.reponse,
            edits=[
                ProposedEdit(
                    before=c.avant,
                    after=c.apres,
                    position=c.position,
                    changed_symbols=c.symboles_modifies,
                    warning=c.avertissement,
                )
                for c in tour.corrections
            ],
            rejected=[
                RejectedEdit(before=r["avant"], after=r["apres"], reason=r["raison"])
                for r in tour.refusees
            ],
        )

    if isinstance(job.result, AuditResult):
        audit: AuditResult = job.result
        return GenerateStatus(
            job_id=job.id,
            status="done",
            audit=CourseAudit(
                findings=[
                    AuditFindingOut(
                        severity=f.severity, source=f.source, excerpt=f.excerpt,
                        explanation=f.explanation, correction=f.correction,
                        target=AuditTarget(**f.target) if f.target else None,
                        suggested_answer=f.suggested_answer,
                    )
                    for f in audit.findings
                ]
            ),
            warnings=audit.warnings,
        )

    if isinstance(job.result, AssessmentResult):
        epreuve: AssessmentResult = job.result
        return GenerateStatus(
            job_id=job.id,
            status="done",
            kind=epreuve.kind,
            assessment=AssessmentDraft(
                title=epreuve.title,
                instructions=epreuve.instructions,
                duration_minutes=epreuve.duration_minutes,
                total_points=epreuve.total_points,
                exercises=[AssessmentExercise(**e) for e in epreuve.exercises],
            ),
            warnings=epreuve.warnings,
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

    engine_llm, trace = engine_for(body, request)
    generator = CourseGenerator(
        llm=engine_llm,
        retriever=request.app.state.retriever,
        settings=settings,
    )
    job = submit_traced(
        request,
        trace,
        lambda: generator.adjust(
            title=body.title,
            sections=[section.model_dump() for section in body.sections],
            request=body.request,
            instruction=body.instruction,
            scope=body.scope,
            course_id=body.course_id,
            strictness=body.strictness,
            history=body.history,
        ),
        lane="prof",
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

    engine_llm, trace = engine_for(body, request)
    generator = CourseGenerator(
        llm=engine_llm,
        retriever=request.app.state.retriever,
        settings=settings,
    )
    job = submit_traced(
        request,
        trace,
        lambda: generator.draft_plan(
            instruction=body.instruction,
            title=body.title,
            scope=body.scope,
            course_id=body.course_id,
            current_plan=body.current_plan,
            request=body.request,
            history=body.history,
        ),
        lane="prof",
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

    engine_llm, trace = engine_for(body, request)
    generator = CourseGenerator(
        llm=engine_llm,
        retriever=request.app.state.retriever,
        settings=settings,
    )
    job = submit_traced(
        request,
        trace,
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
        ),
        lane="prof",
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

    engine_llm, trace = engine_for(body, request)
    generator = CourseGenerator(
        llm=engine_llm,
        retriever=request.app.state.retriever,
        settings=settings,
    )
    job = submit_traced(
        request,
        trace,
        lambda: generator.generate_blocks(
            kind=body.kind,
            text=body.text,
            scope=body.scope,
            count=body.count,
            instruction=body.instruction,
        ),
        lane="prof",
    )
    logger.info(
        "bloc lancé",
        extra={"requestId": body.request_id, "job": job.id, "kind": body.kind},
    )
    return GenerateAccepted(request_id=body.request_id, job_id=job.id)


@router.post(
    "/generate/blocks/discuss",
    response_model=GenerateAccepted,
    status_code=status.HTTP_202_ACCEPTED,
)
async def blocks_discuss(
    body: BlocksDiscussRequest,
    request: Request,
    settings: Settings = Depends(get_settings),
) -> GenerateAccepted:
    """Réviser un bloc sur consigne du professeur — le « chat » des blocs.

    Le document, le plan et les sections ont le leur depuis le début ; les
    trois blocs n'en avaient pas, et un quiz dont une réponse est fausse
    arrivait intact jusqu'à l'élève. Avec `targetIndex`, la révision porte
    sur UNE question ou UN exercice : les autres sont rendus mot pour mot.
    """

    engine_llm, trace = engine_for(body, request)
    generator = CourseGenerator(
        llm=engine_llm,
        retriever=request.app.state.retriever,
        settings=settings,
    )
    job = submit_traced(
        request,
        trace,
        lambda: generator.discuss_block(
            kind=body.kind,
            text=body.text,
            scope=body.scope,
            request=body.request,
            current_summary=body.current_summary,
            current_items=body.current_items,
            target_index=body.target_index,
            count=body.count,
            instruction=body.instruction,
            history=body.history,
        ),
        lane="prof",
    )
    logger.info(
        "révision de bloc lancée",
        extra={
            "requestId": body.request_id,
            "job": job.id,
            "kind": body.kind,
            "item": body.target_index,
        },
    )
    return GenerateAccepted(request_id=body.request_id, job_id=job.id)


@router.post(
    "/generate/assessment",
    response_model=GenerateAccepted,
    status_code=status.HTTP_202_ACCEPTED,
)
async def assessment(
    body: AssessmentRequest,
    request: Request,
    settings: Settings = Depends(get_settings),
) -> GenerateAccepted:
    """Un devoir, une composition ou un examen blanc sur PLUSIEURS cours.

    Le professeur désigne les cours couverts, la durée et le barème ; il
    relit et valide l'épreuve comme le reste. Statut sur GET /generate/{jobId}.
    """

    engine_llm, trace = engine_for(body, request)
    generator = CourseGenerator(
        llm=engine_llm,
        retriever=request.app.state.retriever,
        settings=settings,
    )
    job = submit_traced(
        request,
        trace,
        lambda: generator.compose_assessment(
            kind=body.kind,
            sources=[source.model_dump() for source in body.sources],
            scope=body.scope,
            title=body.title,
            duration_minutes=body.duration_minutes,
            total_points=body.total_points,
            exercise_count=body.exercise_count,
            instruction=body.instruction,
        ),
        lane="prof",
    )
    logger.info(
        "épreuve lancée",
        extra={
            "requestId": body.request_id,
            "job": job.id,
            "kind": body.kind,
            "cours": len(body.sources),
        },
    )
    return GenerateAccepted(request_id=body.request_id, job_id=job.id)


@router.post(
    "/audit/course",
    response_model=GenerateAccepted,
    status_code=status.HTTP_202_ACCEPTED,
)
async def course_audit(
    body: AuditRequest,
    request: Request,
    settings: Settings = Depends(get_settings),
) -> GenerateAccepted:
    """Auditer un cours avant qu'un élève le lise : les calculs par SymPy
    (certains), le reste par un correcteur qui doit citer le cours (probable,
    ambiguïté). Rien n'est corrigé ; le statut se lit sur GET /generate/{jobId}.
    """

    engine_llm, trace = engine_for(body, request)
    job = submit_traced(
        request,
        trace,
        lambda: audit_course(
            text=body.text,
            llm=engine_llm,
            timeout=settings.generation_timeout_s,
            num_ctx=settings.generation_context_tokens,
            quizzes=body.quizzes,
            exercises=body.exercises,
            sections=body.sections,
        ),
        lane="prof",
    )
    logger.info("audit de cours lancé", extra={"requestId": body.request_id, "job": job.id})
    return GenerateAccepted(request_id=body.request_id, job_id=job.id)
