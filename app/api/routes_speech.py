"""POST /speech — le texte validé d'un cours devient son audio.

Asynchrone comme la génération : la synthèse d'un chapitre prend des
dizaines de secondes sur CPU. La plateforme reçoit un ticket, interroge le
statut, puis stocke l'audio chez elle (MinIO) — le rag ne garde rien.

La garde « cours publié seulement » vit côté plateforme : le rag ne connaît
pas le statut d'un cours, il synthétise le texte qu'on lui confie.
"""

from __future__ import annotations

import asyncio
import base64
import logging

from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.api.dependencies import require_service_token
from app.core.verbalize import verbalize
from app.models.schemas import SpeechAccepted, SpeechRequest, SpeechStatus

logger = logging.getLogger(__name__)

router = APIRouter(tags=["speech"], dependencies=[Depends(require_service_token)])


@router.post("/speech", response_model=SpeechAccepted, status_code=status.HTTP_202_ACCEPTED)
async def speak(body: SpeechRequest, request: Request) -> SpeechAccepted:
    engine = request.app.state.speech
    if not engine.available:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "SPEECH_BACKEND_UNAVAILABLE"},
        )

    spoken = verbalize(body.text)
    if not spoken:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"code": "NOTHING_TO_SPEAK"},
        )

    async def work():
        # La synthèse est du CPU pur : un thread, pas l'event loop — sinon
        # tout le service (extract, search…) attendrait derrière la voix.
        result = await asyncio.to_thread(engine.synthesize, spoken)
        logger.info(
            "audio synthétisé",
            extra={
                "requestId": body.request_id,
                "courseId": body.course_id,
                "seconds": result.seconds,
                "format": result.format,
                "characters": result.characters,
            },
        )
        return result

    job = request.app.state.jobs.submit(work)
    return SpeechAccepted(request_id=body.request_id, job_id=job.id)


@router.get("/speech/{job_id}", response_model=SpeechStatus)
async def speech_status(job_id: str, request: Request) -> SpeechStatus:
    job = request.app.state.jobs.get(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "JOB_NOT_FOUND"},
        )
    if job.status == "failed":
        return SpeechStatus(job_id=job.id, status="failed", error=job.error)
    if job.status != "done":
        return SpeechStatus(job_id=job.id, status="running")
    result = job.result
    return SpeechStatus(
        job_id=job.id,
        status="done",
        format=result.format,
        audio_base64=base64.b64encode(result.audio).decode("ascii"),
        seconds=result.seconds,
        characters=result.characters,
    )
