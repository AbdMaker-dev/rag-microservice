"""Lawal, le tuteur des élèves : une question, une explication comprise.

Lawal parle à des élèves — parfois des enfants. Ses règles ne sont pas
celles du rédacteur de cours :

1. **Progression pédagogique** : expliquer, donner un exemple, vérifier que
   c'est compris. Jamais la réponse toute faite à un exercice — un indice
   et la méthode, c'est l'élève qui trouve.
2. **Ancré dans le validé** : le cours publié d'abord (c'est ce que l'élève
   a sous les yeux), les supports du professeur ensuite (le cours est un
   résumé volontairement court — l'approfondissement est dans les
   supports), le programme officiel enfin. Tout a été validé par un humain.
3. **Honnête** : ce que les sources ne couvrent pas, Lawal le dit et
   renvoie vers le professeur. Il n'invente rien.
4. **Tracé** : chaque recherche faite pour répondre est rendue à la
   plateforme, comme pour la génération de cours.

Sans état : l'historique du fil arrive à chaque appel, la plateforme le
conserve.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

from app.config import Settings
from app.core.generation import (
    _context_line,
    _parse_json_block,
    _render_passages,
)
from app.core.llm import LlmProvider
from app.core.retrieval import Passage, Retriever
from app.models.schemas import Scope

logger = logging.getLogger(__name__)

# Lawal cherche partout où un humain a validé : le cours publié, les
# supports relus par le professeur, le programme officiel.
_TUTOR_ROLES = ("cours-publie", "support-cours", "programme-officiel")

_SOURCE_LABELS = {
    "cours-publie": "ton cours",
    "support-cours": "le support du professeur",
    "programme-officiel": "le programme officiel",
}


class TutorFailed(RuntimeError):
    """La réponse n'a pas pu être produite ; le message est montrable."""


@dataclass(frozen=True)
class TutorAnswer:
    text: str
    # La petite question finale qui vérifie que le concept est compris.
    check: str
    # Les notions abordées — la prise pour accrocher plus tard une vidéo de
    # démonstration du concept.
    concepts: List[str]
    citations: List[dict]
    queries: List[dict]
    warnings: List[str] = field(default_factory=list)


_SYSTEM = """Tu es Lawal, le tuteur de la plateforme LawalSchool. Tu aides un élève à COMPRENDRE, en français simple et clair.

{contexte}

Règles absolues :
- Sois professionnel et mature, comme un enseignant expérimenté : bienveillant mais rigoureux, jamais familier, jamais approximatif.
- Avant de répondre, assure-toi d'avoir assez d'éléments : demande autant de recherches que nécessaire plutôt que de répondre avec des extraits insuffisants.
- Adapte ton langage à la classe de l'élève : phrases courtes pour les petits, vocabulaire précis pour les grands.
- Structure : l'idée en une phrase, puis l'explication pas à pas, puis UN exemple concret. Reste court.
- Si la question est un exercice à résoudre : n'en donne JAMAIS la solution. Explique la méthode, donne un indice, laisse l'élève faire.
- Appuie-toi UNIQUEMENT sur les extraits fournis. Cite-les par leur étiquette [S1], [S2]…
- Si les extraits ne suffisent pas pour répondre : dis-le simplement et conseille de demander au professeur. N'invente rien.
- Termine par une petite question qui vérifie que l'élève a compris.

Pour chercher dans les documents, réponds SEULEMENT :
{{"chercher": {{"question": "...", "nature": "cours-publie|support-cours|programme-officiel"}}}}

Pour répondre à l'élève, réponds SEULEMENT ce JSON :
{{"reponse": "l'explication, avec les étiquettes [S1]…", "verification": "la petite question finale", "conceptes": ["notion1", "notion2"]}}"""


class Tutor:
    """L'orchestrateur : Lawal choisit quoi chercher, nous où il a le droit."""

    def __init__(
        self, *, llm: LlmProvider, retriever: Retriever, settings: Settings
    ) -> None:
        self._llm = llm
        self._retriever = retriever
        self._settings = settings

    async def answer(
        self,
        *,
        question: str,
        scope: Scope,
        course_id: str,
        section_heading: str = "",
        history: Optional[List[dict]] = None,
    ) -> TutorAnswer:
        queries: List[dict] = []
        warnings: List[str] = []
        passages: List[Passage] = []
        seen: set = set()
        budget = self._settings.answer_max_queries

        async def search(query: str, role: str, from_model: bool) -> int:
            nonlocal budget
            if from_model:
                if budget <= 0:
                    return 0
                budget -= 1
            # L'élève et le modèle choisissent la question ; le périmètre et
            # le cours viennent de la plateforme et ne se discutent pas.
            found = await self._retriever.search(
                query=query,
                scope=scope,
                limit=4,
                max_excerpt_characters=700,
                course_id=course_id if role != "programme-officiel" else None,
                role=role,
            )
            fresh = [p for p in found if p.chunk_id not in seen]
            seen.update(p.chunk_id for p in fresh)
            passages.extend(fresh)
            queries.append(
                {
                    "question": query,
                    "nature": role,
                    "demandeParLeModele": from_model,
                    "resultats": len(fresh),
                }
            )
            return len(fresh)

        # Le premier réflexe : le cours publié — c'est ce que l'élève lit.
        # Une question posée depuis une section se cherche dans son contexte :
        # « la norme » ne veut pas dire la même chose selon le chapitre.
        probe = f"{section_heading} — {question}" if section_heading else question
        found = await search(probe, "cours-publie", from_model=False)
        if found == 0:
            warnings.append("NO_PUBLISHED_COURSE_CONTENT")
            # Le cours est un résumé court : l'approfondissement vit dans
            # les supports. On les consulte d'office si le cours se tait.
            await search(probe, "support-cours", from_model=False)

        if not passages:
            # Rien de validé ne couvre la question : réponse honnête, sans
            # modèle — un modèle sans extraits inventerait.
            return TutorAnswer(
                text=(
                    "Je n'ai pas trouvé de quoi répondre dans ton cours ni "
                    "dans les documents de ton professeur. Pose-lui la "
                    "question en classe — et si c'est un mot précis, essaie "
                    "de me le demander autrement."
                ),
                check="",
                concepts=[],
                citations=[],
                queries=queries,
                warnings=warnings + ["INSUFFICIENT_EVIDENCE"],
            )

        messages = [
            {
                "role": "system",
                "content": _SYSTEM.format(contexte=_context_line(scope)),
            }
        ]
        for turn in history or []:
            role = "assistant" if turn.get("role") == "lawal" else "user"
            messages.append({"role": role, "content": str(turn.get("content", ""))})
        situation = (
            f"L'élève lit la section « {section_heading} » du cours.\n"
            if section_heading
            else ""
        )
        messages.append(
            {
                "role": "user",
                "content": (
                    situation
                    + f"Question de l'élève : {question}\n\n"
                    "Extraits validés :\n\n"
                    + _render_passages(passages, "S", 1)
                ),
            }
        )

        for _ in range(self._settings.answer_max_queries + 1):
            raw = await self._chat(messages)
            wanted = _wants_tutor_search(raw)
            if wanted is not None:
                asked, role = wanted
                before = len(passages)
                await search(asked, role, from_model=True)
                messages.append({"role": "assistant", "content": raw})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Nouveaux extraits :\n\n"
                            + _render_passages(passages[before:], "S", before + 1)
                            if len(passages) > before
                            else "Aucun nouvel extrait. Réponds avec ce que tu as, honnêtement."
                        ),
                    }
                )
                continue
            parsed = _parse_json_block(raw)
            if parsed and str(parsed.get("reponse", "")).strip():
                return TutorAnswer(
                    text=str(parsed["reponse"]).strip(),
                    check=str(parsed.get("verification", "")).strip(),
                    concepts=[
                        str(c).strip()
                        for c in parsed.get("conceptes", [])
                        if str(c).strip()
                    ][:6],
                    citations=_citations(passages),
                    queries=queries,
                    warnings=warnings,
                )
            # Réponse hors format : on la refuse une fois, puis on échoue.
            messages.append({"role": "assistant", "content": raw})
            messages.append(
                {
                    "role": "user",
                    "content": "Réponds uniquement avec le JSON demandé.",
                }
            )

        raise TutorFailed(
            "Lawal n'a pas réussi à formuler une réponse : réessaie, ou "
            "pose la question au professeur."
        )

    async def _chat(self, messages: List[dict]) -> str:
        estimated = sum(len(m["content"]) for m in messages) // 3
        if estimated > self._settings.generation_context_tokens:
            raise TutorFailed(
                "la conversation est devenue trop longue : ouvre un nouveau "
                "fil avec Lawal"
            )
        return await self._llm.chat(
            messages,
            timeout=self._settings.generation_timeout_s,
            num_ctx=self._settings.generation_context_tokens,
            num_predict=self._settings.answer_output_tokens,
        )


def _wants_tutor_search(raw: str):
    parsed = _parse_json_block(raw)
    if not parsed or "chercher" not in parsed:
        return None
    request = parsed["chercher"]
    if not isinstance(request, dict):
        return None
    question = str(request.get("question", "")).strip()
    role = str(request.get("nature", "cours-publie")).strip()
    if not question:
        return None
    return question, role if role in _TUTOR_ROLES else "cours-publie"


def _citations(passages: List[Passage]) -> List[dict]:
    return [
        {
            "label": f"S{index}",
            "documentId": passage.document_id,
            "title": passage.title,
            "locator": passage.locator,
        }
        for index, passage in enumerate(passages, start=1)
    ]
