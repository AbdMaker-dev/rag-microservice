"""Réparation d'un texte extrait dont l'encodage est cassé.

Certains PDF déclarent mal leurs polices : « compétences » ressort en
« compŽtences ». Le texte est présent mais illisible. Plutôt que de rendre la
page en image et de la faire relire — trop lent sur CPU — on soumet le texte
au modèle de langue, qui rétablit les accents à partir du contexte.

Deux limites assumées, mesurées sur des documents réels :

  * les séquences `(cid:NNN)` désignent un caractère perdu par le PDF ;
    aucun modèle ne peut le deviner, il est marqué `[?]` ;
  * la structure d'un tableau dont les colonnes ont fusionné à l'extraction
    ne se reconstitue pas : l'information a disparu avant le modèle.
"""

from __future__ import annotations

import logging
from typing import List, Sequence

import httpx

from app.config import Settings

logger = logging.getLogger(__name__)


class RepairUnavailable(RuntimeError):
    """Le modèle n'a pas pu réparer le texte."""


# Consigne étroite : restaurer l'encodage, jamais reformuler. Sans cette
# contrainte le modèle résume, corrige le fond ou complète — et on
# réintroduirait par l'IA le risque qu'on cherche justement à éviter.
_REPAIR_PROMPT = """Le texte ci-dessous provient d'un PDF dont les polices sont mal encodées.
Les lettres accentuées ont été remplacées par des caractères parasites.

Restaure le texte français correct.

Règles strictes :
- Corrige uniquement l'encodage : accents, ponctuation, symboles.
- N'ajoute rien, ne résume pas, ne reformule pas, ne complète pas.
- Si un caractère est irrécupérable, écris [?] plutôt que d'inventer.
- Conserve la mise en forme telle quelle, y compris les tableaux Markdown.

Réponds uniquement par le texte restauré, sans commentaire.

Texte à restaurer :

"""


class TextRepairer:
    def __init__(self, settings: Settings, client: httpx.AsyncClient) -> None:
        self._settings = settings
        self._client = client
        self.model = settings.repair_model
        self._base_url = settings.ollama_base_url.rstrip("/")

    async def repair(self, text: str) -> str:
        payload = {
            "model": self.model,
            "stream": False,
            "options": {
                "temperature": 0,
                "num_ctx": self._settings.repair_context_tokens,
                "num_predict": self._settings.repair_max_output_tokens,
            },
            "messages": [{"role": "user", "content": _REPAIR_PROMPT + text}],
        }
        response = await self._client.post(
            f"{self._base_url}/api/chat",
            json=payload,
            timeout=self._settings.repair_timeout_s,
        )
        response.raise_for_status()
        content = response.json().get("message", {}).get("content")
        if not isinstance(content, str) or not content.strip():
            raise RepairUnavailable("le modèle a renvoyé un texte vide")
        return content.strip()

    async def repair_pages(self, pages: Sequence[str]) -> List[str]:
        """Réparer plusieurs pages, en série.

        En série volontairement : le modèle occupe déjà tous les cœurs pour
        une page. Paralléliser ne ferait que les mettre en concurrence.
        """

        repaired: List[str] = []
        for index, page in enumerate(pages, start=1):
            logger.info("réparation", extra={"page": index, "total": len(pages)})
            repaired.append(await self.repair(page))
        return repaired

    async def healthy(self) -> bool:
        try:
            response = await self._client.get(f"{self._base_url}/api/tags", timeout=3.0)
            return response.status_code == 200
        except Exception:  # noqa: BLE001
            return False
