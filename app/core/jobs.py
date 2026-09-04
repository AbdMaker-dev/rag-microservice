"""Tâches asynchrones : deux files, un seul travail d'IA à la fois.

Rédiger un cours prend deux à quatre minutes sur notre processeur. Sans
file, trois professeurs qui génèrent en même temps se partagent la machine
et chacun avance trois fois moins vite : personne n'est servi plus tôt, et
tout le monde regarde un écran muet. Mesuré pendant l'E2E du 04/09/2026 —
une section passait de 78 s seule à 240 s quand des traitements se
chevauchaient.

Donc : **un seul travail d'IA s'exécute à la fois**, les autres attendent
dans une file, et chacun sait où il en est (« 2e — environ 1 min 20 »).

Deux files, parce que deux usages :

  * **élèves** — une question à Lawal, ~30 s, l'élève est DEVANT son écran ;
  * **professeurs** — plan, section, blocs, épreuve, audio : long, différé,
    le prof peut aller faire autre chose.

À chaque place qui se libère, un élève passe avant un professeur : sans
cela, une question de 30 s attendrait derrière un cours de 4 minutes et
l'élève abandonnerait.

Les tâches vivent en mémoire du processus. Un redémarrage les perd : la
tâche interrompue est alors marquée « interrompue », avec un message qui
invite à relancer — jamais un écran qui tourne dans le vide.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Awaitable, Callable, Deque, Dict, Optional

logger = logging.getLogger(__name__)

_KEEP = 50
# Sans mesure encore, l'attente annoncée part de ces ordres de grandeur ;
# ils sont ensuite remplacés par la moyenne réellement observée.
_DEFAULT_SECONDS = {"eleve": 35.0, "prof": 150.0}
# La moyenne suit les dernières exécutions : le jour où le modèle change,
# l'estimation suit sans qu'on touche au code.
_HISTORY = 8


@dataclass
class Job:
    id: str
    lane: str = "prof"  # prof | eleve
    status: str = "queued"  # queued | running | done | failed
    result: Optional[object] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None


class JobStore:
    """Deux files, un exécutant. Les élèves passent devant."""

    def __init__(self) -> None:
        self._jobs: Dict[str, Job] = {}
        self._queues: Dict[str, Deque[tuple]] = {"eleve": deque(), "prof": deque()}
        self._durations: Dict[str, Deque[float]] = {
            "eleve": deque(maxlen=_HISTORY),
            "prof": deque(maxlen=_HISTORY),
        }
        self._running: Optional[Job] = None
        self._worker: Optional[asyncio.Task] = None

    # ── soumettre ────────────────────────────────────────────────────────
    def submit(
        self, work: Callable[[], Awaitable[object]], *, lane: str = "prof"
    ) -> Job:
        if lane not in self._queues:
            lane = "prof"
        job = Job(id=uuid.uuid4().hex, lane=lane)
        self._jobs[job.id] = job
        self._queues[lane].append((job, work))
        self._prune()
        self._ensure_worker()
        logger.info(
            "tâche en file",
            extra={"job": job.id, "file": lane, "devant": self.position(job) - 1},
        )
        return job

    def get(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    # ── ce que l'appelant montre à l'utilisateur ─────────────────────────
    def position(self, job: Job) -> int:
        """1 = le prochain servi. 0 quand la tâche n'attend plus."""

        if job.status != "queued":
            return 0
        # Un élève passe devant les professeurs : sa position ne compte que
        # les élèves déjà en file.
        ahead = [j for j, _ in self._queues["eleve"] if j.created_at < job.created_at]
        if job.lane == "prof":
            ahead += [j for j, _ in self._queues["eleve"]]
            ahead += [
                j for j, _ in self._queues["prof"] if j.created_at < job.created_at
            ]
        return len(ahead) + 1

    def wait_seconds(self, job: Job) -> Optional[int]:
        """L'attente annoncée, en secondes. None si la tâche n'attend plus."""

        if job.status != "queued":
            return None
        ahead = self.position(job) - 1
        wait = ahead * self._average("prof")
        if self._running is not None and self._running.started_at is not None:
            elapsed = (
                datetime.now(timezone.utc) - self._running.started_at
            ).total_seconds()
            wait += max(0.0, self._average(self._running.lane) - elapsed)
        return int(wait + self._average(job.lane))

    def _average(self, lane: str) -> float:
        history = self._durations[lane]
        if not history:
            return _DEFAULT_SECONDS.get(lane, 120.0)
        return sum(history) / len(history)

    # ── l'exécutant ─────────────────────────────────────────────────────
    def _ensure_worker(self) -> None:
        if self._worker is None or self._worker.done():
            self._worker = asyncio.get_running_loop().create_task(self._drain())

    def _next(self) -> Optional[tuple]:
        for lane in ("eleve", "prof"):  # les élèves d'abord, toujours
            if self._queues[lane]:
                return self._queues[lane].popleft()
        return None

    async def _drain(self) -> None:
        while True:
            entry = self._next()
            if entry is None:
                return
            job, work = entry
            job.status = "running"
            job.started_at = datetime.now(timezone.utc)
            self._running = job
            try:
                job.result = await work()
                job.status = "done"
            except Exception as error:  # noqa: BLE001
                # Le message est montré au professeur : jamais de trace brute.
                logger.exception("tâche échouée", extra={"job": job.id})
                job.error = str(error)
                job.status = "failed"
            finally:
                job.finished_at = datetime.now(timezone.utc)
                if job.started_at is not None:
                    self._durations[job.lane].append(
                        (job.finished_at - job.started_at).total_seconds()
                    )
                self._running = None
                logger.info(
                    "tâche terminée",
                    extra={
                        "job": job.id,
                        "file": job.lane,
                        "statut": job.status,
                        "secondes": int(
                            (job.finished_at - job.started_at).total_seconds()
                        )
                        if job.started_at
                        else None,
                    },
                )

    def _prune(self) -> None:
        finished = [j for j in self._jobs.values() if j.status in ("done", "failed")]
        for stale in sorted(finished, key=lambda j: j.created_at)[
            : max(0, len(self._jobs) - _KEEP)
        ]:
            self._jobs.pop(stale.id, None)
