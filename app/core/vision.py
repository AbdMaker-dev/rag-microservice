"""Lecture de pages par modèle de vision.

Quand la couche texte d'un PDF est absente ou corrompue, on ne tente pas de
la réparer : on fait relire la page telle qu'elle s'affiche. Le modèle voit
l'image, pas l'encodage — les polices mal déclarées et les colonnes
entrelacées cessent d'être un problème.
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
from typing import List, Optional, Sequence

import httpx

from app.config import Settings

logger = logging.getLogger(__name__)


class VisionUnavailable(RuntimeError):
    """Le modèle de vision n'a pas pu lire la page."""


# Consigne volontairement étroite : transcrire, pas interpréter. Le modèle ne
# doit ni résumer, ni compléter, ni corriger le fond — sinon on réintroduit
# par l'IA le risque qu'on cherchait à éviter avec l'OCR.
_TRANSCRIPTION_PROMPT = """Transcris intégralement le texte de cette page.

Règles strictes :
- Reproduis le texte exactement, y compris les accents français.
- Rends les tableaux au format Markdown, une ligne par ligne du tableau.
- Conserve les titres et la numérotation.
- N'ajoute aucun commentaire, aucune explication, aucun résumé.
- Si une zone est illisible, écris [illisible] plutôt que de deviner.

Réponds uniquement par la transcription."""


def render_pages(payload: bytes, pages: Sequence[int], scale: float = 2.0) -> List[bytes]:
    """Rendre certaines pages d'un PDF en PNG. `pages` est indexé à partir de 1."""

    try:
        import pypdfium2
    except ImportError as error:  # pragma: no cover
        raise VisionUnavailable("pypdfium2 n'est pas installé") from error

    document = pypdfium2.PdfDocument(payload)
    images: List[bytes] = []
    try:
        for number in pages:
            image = document[number - 1].render(scale=scale).to_pil()
            buffer = io.BytesIO()
            # PNG plutôt que JPEG : pas d'artefact de compression sur du texte
            # dense, et le surcoût de taille est sans importance en local.
            image.save(buffer, format="PNG")
            images.append(buffer.getvalue())
    finally:
        document.close()
    return images


class VisionReader:
    def __init__(self, settings: Settings, client: httpx.AsyncClient) -> None:
        self._settings = settings
        self._client = client
        self.model = settings.vision_model
        self._base_url = settings.ollama_base_url.rstrip("/")

    async def read_page(self, image: bytes) -> str:
        payload = {
            "model": self.model,
            "stream": False,
            "options": {
                "temperature": 0,
                # Une page rendue en image produit beaucoup de jetons visuels.
                # Avec la fenêtre par défaut d'Ollama, le moteur natif dépasse
                # sa mémoire de travail et s'arrête brutalement.
                "num_ctx": self._settings.vision_context_tokens,
                "num_predict": self._settings.vision_max_output_tokens,
            },
            "messages": [
                {
                    "role": "user",
                    "content": _TRANSCRIPTION_PROMPT,
                    "images": [base64.b64encode(image).decode("ascii")],
                }
            ],
        }
        response = await self._client.post(
            f"{self._base_url}/api/chat",
            json=payload,
            timeout=self._settings.vision_timeout_s,
        )
        response.raise_for_status()
        content = response.json().get("message", {}).get("content")
        if not isinstance(content, str) or not content.strip():
            raise VisionUnavailable("le modèle de vision a renvoyé une page vide")
        return content.strip()

    async def read_pages(self, images: Sequence[bytes]) -> List[str]:
        """Lire plusieurs pages, en série.

        En série volontairement : le modèle occupe déjà tous les cœurs pour
        une page. Paralléliser ne ferait que les mettre en concurrence.
        """

        results: List[str] = []
        for index, image in enumerate(images, start=1):
            logger.info("lecture vision", extra={"page": index, "total": len(images)})
            results.append(await self.read_page(image))
        return results

    async def healthy(self) -> bool:
        try:
            response = await self._client.get(f"{self._base_url}/api/tags", timeout=3.0)
            return response.status_code == 200
        except Exception:  # noqa: BLE001
            return False
