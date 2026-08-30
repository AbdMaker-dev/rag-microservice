"""Tâches de génération, asynchrones.

Rédiger un cours prend plusieurs minutes sur CPU : l'appel HTTP répond tout
de suite avec un identifiant, la plateforme interroge le statut. Les tâches
vivent en mémoire du processus — un redémarrage les perd, et c'est assumé :
un cours en cours de rédaction se relance, il n'y a rien d'irremplaçable.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Awaitable, Callable, Dict, Optional

logger = logging.getLogger(__name__)

_KEEP = 50


@dataclass
class Job:
    id: str
    status: str = "running"  # running | done | failed
    result: Optional[object] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class JobStore:
    def __init__(self) -> None:
        self._jobs: Dict[str, Job] = {}

    def submit(self, work: Callable[[], Awaitable[object]]) -> Job:
        job = Job(id=uuid.uuid4().hex)
        self._jobs[job.id] = job
        self._prune()

        async def _run() -> None:
            try:
                job.result = await work()
                job.status = "done"
            except Exception as error:  # noqa: BLE001
                # Le message est montré au professeur : jamais de trace brute.
                logger.exception("génération échouée", extra={"job": job.id})
                job.error = str(error)
                job.status = "failed"

        asyncio.get_running_loop().create_task(_run())
        return job

    def get(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    def _prune(self) -> None:
        finished = [j for j in self._jobs.values() if j.status != "running"]
        for stale in sorted(finished, key=lambda j: j.created_at)[: max(0, len(self._jobs) - _KEEP)]:
            self._jobs.pop(stale.id, None)
