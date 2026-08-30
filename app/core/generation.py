"""Rédiger un cours à partir de la base de connaissance.

Le principe, décidé avec la plateforme : **l'IA a la main sur ses
recherches, jamais sur le périmètre**. Pendant la rédaction, le modèle peut
interroger la base autant de fois qu'il en a besoin, avec ses propres mots —
mais chaque recherche est exécutée par nous, verrouillée sur le cours et le
périmètre reçus de la plateforme. Il choisit *quoi* chercher, nous décidons
*où* il a le droit de chercher.

Le déroulé :

1. **Le cadre.** Le programme officiel du périmètre est interrogé d'office :
   il dit ce qui doit figurer au cours et ce qui est hors programme.
2. **Le plan.** Le modèle propose les sections, en JSON.
3. **La rédaction, section par section.** Chaque section reçoit des extraits
   trouvés pour elle ; si le modèle les juge insuffisants, il répond
   `{"chercher": {...}}` au lieu de rédiger, reçoit les nouveaux extraits, et
   reprend — dans la limite d'un plafond global de recherches.

Chaque affirmation du cours doit pouvoir être citée : le modèle référence les
extraits par leurs étiquettes ([S1], [P2]…), et la réponse porte pour chaque
section la liste des sources réellement citées, avec leur `locator` — la page
du document d'origine.

En mode `grounded`, rien ne doit sortir des extraits ; ce qui manque est
signalé, pas inventé. En mode `enriched`, le modèle peut compléter de
lui-même, à condition d'encadrer chaque ajout entre ⟦AJOUT⟧ et ⟦/AJOUT⟧ pour
que le professeur le voie — c'est lui qui valide.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from app.config import Settings
from app.core.llm import LlmProvider
from app.core.retrieval import Passage, Retriever
from app.models.schemas import Scope

logger = logging.getLogger(__name__)

_LABEL = re.compile(r"\[(?:S|P)\d+\]")
_ADDITION = "⟦AJOUT⟧"
_ROLES = {"support-cours", "programme-officiel"}


class GenerationFailed(RuntimeError):
    """La génération n'a pas pu aboutir ; le message est montrable au prof."""


@dataclass(frozen=True)
class SectionDraft:
    heading: str
    text: str
    citations: List[dict]
    has_additions: bool


@dataclass(frozen=True)
class GeneratedCourse:
    title: str
    sections: List[SectionDraft]
    queries: List[dict]
    warnings: List[str] = field(default_factory=list)


def _context_line(scope: Scope) -> str:
    """Dire au modèle pour qui il écrit.

    Le périmètre filtrait déjà ses recherches, mais ne lui était jamais
    énoncé : sans cette ligne, l'IA ignore qu'elle écrit pour une seconde S
    et ne peut calibrer ni le niveau de langue ni la difficulté. Et le pays
    était écrit en dur « sénégalais » — faux dès le premier cours malien.
    """

    serie = f", série {scope.track}" if scope.track else ""
    return (
        f"Matière : {scope.subject}. Classe : {scope.grade}{serie} "
        f"({scope.level}), pays {scope.country}, programme officiel "
        f"{scope.curriculum_version}, langue d'enseignement {scope.language}."
    )


def _parse_json_block(raw: str) -> Optional[dict]:
    """Le modèle répond parfois avec du texte autour du JSON : on l'isole."""

    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        parsed = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _wants_search(raw: str) -> Optional[Tuple[str, str]]:
    """La réponse est-elle une demande de recherche plutôt qu'une rédaction ?"""

    parsed = _parse_json_block(raw)
    if not parsed or "chercher" not in parsed:
        return None
    request = parsed["chercher"]
    if not isinstance(request, dict):
        return None
    question = str(request.get("question", "")).strip()
    nature = str(request.get("nature", "support-cours")).strip()
    if not question:
        return None
    return question, nature if nature in _ROLES else "support-cours"


def _render_passages(passages: List[Passage], prefix: str, start: int) -> str:
    lines = []
    for offset, passage in enumerate(passages):
        label = f"[{prefix}{start + offset}]"
        lines.append(
            f"{label} ({passage.title} — {passage.locator})\n{passage.content}"
        )
    return "\n\n".join(lines)


class CourseGenerator:
    """L'orchestrateur : le modèle rédige, nous exécutons ses recherches."""

    def __init__(
        self, *, llm: LlmProvider, retriever: Retriever, settings: Settings
    ) -> None:
        self._llm = llm
        self._retriever = retriever
        self._settings = settings

    async def generate(
        self,
        *,
        instruction: str,
        scope: Scope,
        course_id: str,
        strictness: str,
    ) -> GeneratedCourse:
        queries: List[dict] = []
        warnings: List[str] = []
        budget = self._settings.generation_max_queries

        async def search(
            question: str, nature: Optional[str], from_model: bool
        ) -> List[Passage]:
            nonlocal budget
            if from_model:
                if budget <= 0:
                    return []
                budget -= 1
            # Le modèle choisit la question et la nature ; le périmètre et le
            # cours, eux, viennent de la plateforme et ne se discutent pas.
            passages = await self._retriever.search(
                query=question,
                scope=scope,
                limit=5,
                max_excerpt_characters=900,
                course_id=course_id if nature == "support-cours" else None,
                role=nature,
            )
            queries.append(
                {
                    "question": question,
                    "nature": nature,
                    "demandeParLeModele": from_model,
                    "resultats": len(passages),
                }
            )
            return passages

        # 1. Le cadre : le programme officiel du périmètre, toujours consulté.
        frame = await search(instruction, "programme-officiel", from_model=False)
        if not frame:
            warnings.append("NO_OFFICIAL_CURRICULUM_IN_SCOPE")

        # 2. Le plan.
        title, headings = await self._plan(instruction, scope, frame)

        # 3. La rédaction, section par section.
        sections: List[SectionDraft] = []
        for heading in headings[: self._settings.generation_max_sections]:
            sections.append(
                await self._write_section(
                    heading=heading,
                    instruction=instruction,
                    scope=scope,
                    strictness=strictness,
                    frame=frame,
                    search=search,
                )
            )

        if strictness == "grounded" and any(s.has_additions for s in sections):
            # Le modèle a encadré des ajouts alors qu'aucun n'était permis :
            # on les laisse visibles et on prévient, plutôt que de les fondre
            # silencieusement dans le cours.
            warnings.append("ADDITIONS_IN_GROUNDED_MODE")

        return GeneratedCourse(
            title=title, sections=sections, queries=queries, warnings=warnings
        )


    async def adjust(
        self,
        *,
        title: str,
        sections: List[dict],
        request: str,
        instruction: str,
        scope: Scope,
        course_id: str,
        strictness: str,
    ) -> GeneratedCourse:
        """Réviser un cours existant sur consigne du professeur.

        C'est la conversation : « revois la partie 2, ajoute des exercices,
        retire l'anecdote »… jusqu'au bon cours final. Le brouillon vit côté
        plateforme — nous recevons le cours actuel et la consigne, nous
        rendons le cours révisé, sans état conservé ici.

        Sur CPU, chaque section coûte des minutes : le modèle décide d'abord
        QUOI toucher, et **les sections non concernées passent telles
        quelles** — pas une seconde de rédaction gaspillée.
        """

        queries: List[dict] = []
        warnings: List[str] = []
        budget = self._settings.generation_max_queries

        async def search(
            question: str, nature: Optional[str], from_model: bool
        ) -> List[Passage]:
            nonlocal budget
            if from_model:
                if budget <= 0:
                    return []
                budget -= 1
            passages = await self._retriever.search(
                query=question,
                scope=scope,
                limit=5,
                max_excerpt_characters=900,
                course_id=course_id if nature == "support-cours" else None,
                role=nature,
            )
            queries.append(
                {
                    "question": question,
                    "nature": nature,
                    "demandeParLeModele": from_model,
                    "resultats": len(passages),
                }
            )
            return passages

        plan = await self._adjustment_plan(title, sections, request, scope)
        frame = await search(
            f"{instruction} — {request}", "programme-officiel", from_model=False
        )

        by_heading = {section["heading"]: section for section in sections}
        kept: List[SectionDraft] = []
        operations = {op["section"]: op for op in plan["operations"]}

        for section in sections:
            heading = section["heading"]
            operation = operations.get(heading)
            if operation is None:
                # Non concernée par la consigne : conservée mot pour mot.
                kept.append(
                    SectionDraft(
                        heading=heading,
                        text=section["text"],
                        citations=list(section.get("citations", [])),
                        has_additions=_ADDITION in section["text"],
                    )
                )
                continue
            if operation["action"] == "supprimer":
                continue
            kept.append(
                await self._write_section(
                    heading=heading,
                    instruction=(
                        f"{instruction}\nConsigne de révision : "
                        f"{operation.get('consigne') or request}\n"
                        f"Version actuelle de la section :\n{section['text']}"
                    ),
                    scope=scope,
                    strictness=strictness,
                    frame=frame,
                    search=search,
                )
            )

        for operation in plan["operations"]:
            if operation["action"] == "ajouter" and operation["section"] not in by_heading:
                kept.append(
                    await self._write_section(
                        heading=operation["section"],
                        instruction=(
                            f"{instruction}\nNouvelle section demandée : "
                            f"{operation.get('consigne') or request}"
                        ),
                        scope=scope,
                        strictness=strictness,
                        frame=frame,
                        search=search,
                    )
                )

        if strictness == "grounded" and any(s.has_additions for s in kept):
            warnings.append("ADDITIONS_IN_GROUNDED_MODE")

        return GeneratedCourse(
            title=str(plan.get("titre") or title),
            sections=kept,
            queries=queries,
            warnings=warnings,
        )

    async def _adjustment_plan(
        self, title: str, sections: List[dict], request: str, scope: Scope
    ) -> dict:
        """Demander au modèle quoi toucher — et seulement quoi toucher."""

        summary = "\n".join(
            f"- {section['heading']} ({len(section['text'])} caractères)"
            for section in sections
        )
        raw = await self._chat(
            [
                {
                    "role": "system",
                    "content": (
                        "Un professeur demande une révision de son cours. "
                        + _context_line(scope)
                        + " Réponds UNIQUEMENT en JSON : "
                        '{"titre": "...", "operations": [{"action": '
                        '"reecrire"|"ajouter"|"supprimer", "section": '
                        '"<titre de section>", "consigne": "..."}]}. '
                        "Ne liste QUE les sections réellement concernées par "
                        "la demande — les autres seront conservées telles "
                        "quelles."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Cours : {title}\nSections :\n{summary}\n\n"
                        f"Demande du professeur : {request}"
                    ),
                },
            ]
        )
        parsed = _parse_json_block(raw)
        if not parsed or not isinstance(parsed.get("operations"), list):
            raise GenerationFailed(
                "le modèle n'a pas produit de plan de révision exploitable"
            )
        operations = []
        for op in parsed["operations"]:
            if not isinstance(op, dict):
                continue
            action = str(op.get("action", "")).strip()
            section = str(op.get("section", "")).strip()
            if action in {"reecrire", "ajouter", "supprimer"} and section:
                operations.append(
                    {"action": action, "section": section,
                     "consigne": str(op.get("consigne", "")).strip()}
                )
        if not operations:
            raise GenerationFailed("la demande de révision ne vise aucune section")
        return {"titre": parsed.get("titre"), "operations": operations}

    # ------------------------------------------------------------------ étapes

    async def _plan(
        self, instruction: str, scope: Scope, frame: List[Passage]
    ) -> Tuple[str, List[str]]:
        raw = await self._chat(
            [
                {
                    "role": "system",
                    "content": (
                        "Tu prépares le plan d'un cours pour un professeur. "
                        + _context_line(scope)
                        + " Réponds UNIQUEMENT en JSON : "
                        '{"titre": "...", "sections": ["...", "..."]}. '
                        "De 3 à "
                        f"{self._settings.generation_max_sections} sections, "
                        "fidèles au programme officiel fourni."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Demande du professeur : {instruction}\n\n"
                        "Extraits du programme officiel :\n\n"
                        + _render_passages(frame, "P", 1)
                    ),
                },
            ]
        )
        parsed = _parse_json_block(raw)
        if not parsed or not isinstance(parsed.get("sections"), list):
            raise GenerationFailed("le modèle n'a pas produit de plan exploitable")
        headings = [str(h).strip() for h in parsed["sections"] if str(h).strip()]
        if not headings:
            raise GenerationFailed("le plan ne contient aucune section")
        return str(parsed.get("titre") or instruction).strip(), headings

    async def _write_section(
        self,
        *,
        heading: str,
        instruction: str,
        scope: Scope,
        strictness: str,
        frame: List[Passage],
        search,
    ) -> SectionDraft:
        supports = await search(
            f"{heading} — {instruction}", "support-cours", from_model=False
        )

        registry: Dict[str, Passage] = {}
        for index, passage in enumerate(frame, start=1):
            registry[f"P{index}"] = passage
        for index, passage in enumerate(supports, start=1):
            registry[f"S{index}"] = passage

        rules = (
            "Rédige UNIQUEMENT à partir des extraits fournis. Chaque "
            "affirmation porte l'étiquette de sa source, ex. [S1] ou [P2]. "
            "Si une notion manque dans les extraits, écris "
            "[LACUNE: ce qui manque] au lieu de l'inventer."
            if strictness == "grounded"
            else (
                "Appuie-toi d'abord sur les extraits fournis, étiquettes "
                "[S1]/[P2] à l'appui. Tu peux compléter de tes connaissances, "
                "mais chaque ajout personnel DOIT être encadré entre "
                "⟦AJOUT⟧ et ⟦/AJOUT⟧ pour que le professeur le repère."
            )
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "Tu rédiges une section d'un cours pour des élèves. "
                    + _context_line(scope)
                    + f" {rules}\n"
                    "Si les extraits ne suffisent pas, tu peux demander une "
                    "recherche en répondant UNIQUEMENT : "
                    '{"chercher": {"question": "...", "nature": '
                    '"support-cours"}} (ou "programme-officiel"). '
                    "Sinon, rédige la section — texte seulement, sans JSON."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Cours demandé : {instruction}\n"
                    f"Section à rédiger : {heading}\n\n"
                    "Extraits du programme officiel :\n\n"
                    + _render_passages(frame, "P", 1)
                    + "\n\nExtraits des documents du cours :\n\n"
                    + (_render_passages(supports, "S", 1) or "(aucun)")
                ),
            },
        ]

        text = ""
        for _round in range(4):
            raw = await self._chat(messages)
            wanted = _wants_search(raw)
            if wanted is None:
                text = raw.strip()
                break
            question, nature = wanted
            extra = await search(question, nature, from_model=True)
            start = sum(1 for k in registry if k.startswith("S")) + 1
            for index, passage in enumerate(extra):
                registry[f"S{start + index}"] = passage
            messages.append({"role": "assistant", "content": raw})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Extraits supplémentaires :\n\n"
                        + (_render_passages(extra, "S", start) or "(rien trouvé)")
                        + "\n\nRédige la section maintenant."
                    ),
                }
            )
        else:
            raise GenerationFailed(
                f"la section « {heading} » n'a pas été rédigée après 4 tours"
            )

        cited = sorted(
            {label.strip("[]") for label in _LABEL.findall(text)}
        )
        citations = [
            {
                "documentId": registry[label].document_id,
                "locator": registry[label].locator,
                "title": registry[label].title,
            }
            for label in cited
            if label in registry
        ]
        return SectionDraft(
            heading=heading,
            text=text,
            citations=citations,
            has_additions=_ADDITION in text,
        )

    async def _chat(self, messages: List[dict]) -> str:
        return await self._llm.chat(
            messages,
            timeout=self._settings.generation_timeout_s,
            num_ctx=self._settings.generation_context_tokens,
        )
