"""Synthèse vocale Piper : texte verbalisé → WAV → MP3.

Piper tourne sur notre CPU, sans réseau : la voix des cours ne dépend
d'aucun service extérieur et ne coûte rien à l'usage. La voix retenue —
choisie à l'oreille par Alioune sur trois échantillons (31/08/2026) — est
fr_FR-siwis-medium.

Mesuré sur le poste de développement : ~50 s d'audio en 3,5 s de calcul,
chargement de la voix en 1 s. La voix se charge une fois et reste en
mémoire ; la synthèse est bloquante, l'appelant la met dans un thread.

Le WAV sort de Piper ; ffmpeg le compresse en MP3 mono 64 kbit/s — largement
suffisant pour de la parole, dix fois plus léger. Sans ffmpeg, le WAV part
tel quel : un audio lourd vaut mieux que pas d'audio, et la réponse dit son
format.
"""

from __future__ import annotations

import io
import logging
import shutil
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class SpeechUnavailable(RuntimeError):
    """La voix n'est pas installée : l'appelant rend un 503 explicite."""


@dataclass(frozen=True)
class SpeechResult:
    audio: bytes
    format: str  # "mp3" | "wav"
    seconds: float
    characters: int


class SpeechEngine:
    """Une voix chargée une fois, des synthèses à la demande."""

    def __init__(self, voice_path: str) -> None:
        self._voice_path = Path(voice_path)
        self._voice = None

    @property
    def available(self) -> bool:
        return self._voice_path.exists()

    def _load(self):
        if self._voice is None:
            if not self.available:
                raise SpeechUnavailable(
                    f"voix absente : {self._voice_path} — "
                    "python -m piper.download_voices fr_FR-siwis-medium"
                )
            try:
                from piper import PiperVoice
            except ImportError as error:  # pragma: no cover
                raise SpeechUnavailable("piper-tts n'est pas installé") from error
            self._voice = PiperVoice.load(str(self._voice_path))
            logger.info("voix chargée", extra={"voice": self._voice_path.name})
        return self._voice

    def synthesize(self, text: str) -> SpeechResult:
        """Bloquant (CPU) : à appeler depuis un thread, jamais l'event loop."""

        voice = self._load()
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as output:
            voice.synthesize_wav(text, output)
        payload = buffer.getvalue()
        with wave.open(io.BytesIO(payload)) as reader:
            seconds = reader.getnframes() / reader.getframerate()

        encoded = _to_mp3(payload)
        if encoded is not None:
            return SpeechResult(encoded, "mp3", round(seconds, 1), len(text))
        return SpeechResult(payload, "wav", round(seconds, 1), len(text))


def _to_mp3(wav_payload: bytes) -> Optional[bytes]:
    """MP3 mono 64 kbit/s via ffmpeg — None si ffmpeg manque ou échoue."""

    if shutil.which("ffmpeg") is None:
        logger.warning("ffmpeg absent : l'audio part en WAV")
        return None
    try:
        completed = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error",
             "-i", "pipe:0", "-ac", "1", "-b:a", "64k", "-f", "mp3", "pipe:1"],
            input=wav_payload, capture_output=True, timeout=300, check=True,
        )
        return completed.stdout
    except Exception:  # noqa: BLE001
        logger.exception("compression MP3 impossible, l'audio part en WAV")
        return None
