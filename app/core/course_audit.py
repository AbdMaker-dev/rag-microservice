"""Auditer un cours avant qu'un élève le lise.

16/09/2026 : le cours publié « Nombres complexes » écrivait que −1 + i est
dans le premier quadrant, et « montrait » rectangle isocèle un triangle qui
ne l'est pas, en prenant un produit de complexes pour un produit scalaire.
Le prof l'avait validé ; aucune machine ne l'avait relu.

Deux couches, et elles ne jouent pas le même rôle :

1. **SymPy** — les égalités chiffrées et les coordonnées. Déterministe :
   ce qu'il signale est CERTAIN (`certaine`).
2. **Un modèle correcteur** — quadrants, conditions, méthodes, conclusions
   que les données ne prouvent pas. Il se trompe parfois : ce qu'il signale
   est `probable` ou `ambiguite`, jamais `certaine`, et chaque signalement
   doit CITER le cours mot pour mot. Une citation introuvable dans le texte
   est écartée : un correcteur qui invente l'erreur qu'il dénonce ferait
   corriger au prof ce qui était juste.

L'audit ne corrige rien. Il rend une liste ; le professeur décide.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import List

from app.core.llm import GenerationError, LlmProvider
from app.core.math_check import check as check_calculations

logger = logging.getLogger(__name__)

# Une part du cours par appel : le correcteur lit mieux court, et un cours
# long dépasserait la fenêtre du modèle local.
_PART_CHARACTERS = 6_000
_OUTPUT_TOKENS = 2_000

_SYSTEM = """Tu es correcteur de mathématiques pour un cours de lycée, AVANT sa publication aux élèves. Le texte n'est PAS nécessairement correct.

Vérifie chaque formule, chaque calcul, chaque position géométrique (quadrants, signes), chaque condition d'application, chaque méthode et chaque conclusion : une conclusion doit être prouvée par les données de l'exemple.

Les formules sont en LaTeX (\\( … \\), \\[ … \\]) : lis-les comme telles. Ne signale ni le style, ni la mise en page, ni une notation LaTeX correcte.

Pour CHAQUE problème, écris exactement :
### PROBLÈME
gravité : probable | ambiguite
extrait : la phrase du cours, recopiée mot pour mot
explication : ce qui est faux, et pourquoi, calcul à l'appui
correction : ce qu'il faudrait écrire

« probable » : une erreur mathématique. « ambiguite » : une formulation qui peut induire l'élève en erreur.
S'il n'y a aucun problème, écris seulement : ### AUCUN"""


@dataclass(frozen=True)
class AuditFinding:
    severity: str  # certaine | probable | ambiguite
    source: str  # calcul | relecture
    excerpt: str
    explanation: str
    correction: str = ""


@dataclass
class AuditResult:
    findings: List[AuditFinding]
    warnings: List[str] = field(default_factory=list)


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _parts(text: str) -> List[str]:
    """Des parts d'au plus _PART_CHARACTERS, coupées entre paragraphes."""

    parts: List[str] = []
    current = ""
    for paragraph in re.split(r"\n\s*\n", text):
        if current and len(current) + len(paragraph) > _PART_CHARACTERS:
            parts.append(current)
            current = ""
        current = f"{current}\n\n{paragraph}" if current else paragraph
    if current.strip():
        parts.append(current)
    return parts


_BLOCK = re.compile(r"^\s*#{2,4}\s*PROBL[ÈE]ME\s*$", re.IGNORECASE | re.MULTILINE)
_FIELD = re.compile(
    r"^\s*(gravit[ée]|extrait|explication|correction)\s*:\s*", re.IGNORECASE | re.MULTILINE
)


def _read(raw: str) -> List[dict]:
    blocks = _BLOCK.split(raw)[1:]
    found = []
    for block in blocks:
        # Un bloc s'arrête au titre suivant, quel qu'il soit : vu le
        # 16/09/2026, « ### AUCUN » (pour une autre part) se collait dans la
        # correction du dernier problème.
        block = re.split(r"^\s*#{2,4}\s*\S", block, maxsplit=1, flags=re.MULTILINE)[0]
        marks = list(_FIELD.finditer(block))
        values = {}
        for index, mark in enumerate(marks):
            end = marks[index + 1].start() if index + 1 < len(marks) else len(block)
            key = mark.group(1).lower().replace("é", "e")
            values[key] = block[mark.end():end].strip()
        if values.get("extrait") and values.get("explication"):
            found.append(values)
    return found


def _quote_in(excerpt: str, text: str) -> bool:
    quote = _normalise(excerpt.strip(" «»\"'"))
    return len(quote) >= 12 and quote in _normalise(text)


def _blocks_text(quizzes: List[dict], exercises: List[dict]) -> str:
    """Les blocs, écrits pour le correcteur, avec la réponse ANNONCÉE."""

    lines: List[str] = []
    for number, quiz in enumerate(quizzes, start=1):
        choices = quiz.get("choices") or []
        answer = quiz.get("answer")
        lines.append(f"Quiz {number} : {quiz.get('question', '')}")
        lines += [f"{'ABCD'[i]}) {c}" for i, c in enumerate(choices[:4])]
        if isinstance(answer, int) and 0 <= answer < 4:
            lines.append(f"Bonne réponse enregistrée : {'ABCD'[answer]}")
        lines.append(f"Explication : {quiz.get('explanation', '')}\n")
    for number, exercise in enumerate(exercises, start=1):
        lines.append(f"Exercice {number} : {exercise.get('statement', '')}")
        lines.append(f"Corrigé : {exercise.get('solution', '')}\n")
    return "\n".join(lines)


async def audit_course(
    *, text: str, llm: LlmProvider, timeout: float, num_ctx: int,
    quizzes: List[dict] = (), exercises: List[dict] = (),
) -> AuditResult:
    result = AuditResult(findings=[])

    # Certain, sans modèle : la réponse enregistrée contredit l'explication.
    from app.core.generation import _quiz_contredit, reponse_annoncee

    for number, quiz in enumerate(quizzes, start=1):
        if _quiz_contredit(quiz):
            annoncee = "ABCD"[reponse_annoncee(str(quiz.get("explanation") or ""))]
            enregistree = quiz.get("answer")
            result.findings.append(AuditFinding(
                severity="certaine", source="calcul",
                excerpt=str(quiz.get("question", "")),
                explanation=(
                    f"Quiz {number} : la bonne réponse enregistrée est "
                    f"{'ABCD'[enregistree] if isinstance(enregistree, int) and 0 <= enregistree < 4 else enregistree}, "
                    f"mais l'explication annonce {annoncee}. Un élève qui répond juste est compté faux."
                ),
            ))
    blocks = _blocks_text(list(quizzes), list(exercises))
    if blocks:
        text = f"{text}\n\n{blocks}"

    for finding in check_calculations(text):
        result.findings.append(AuditFinding(
            severity="certaine", source="calcul",
            excerpt=finding.message.split(" » ")[0].lstrip("« "),
            explanation=finding.message,
        ))

    for part in _parts(text):
        try:
            raw = await llm.chat(
                [{"role": "system", "content": _SYSTEM},
                 {"role": "user", "content": f"Cours à corriger :\n\n{part}"}],
                timeout=timeout, num_ctx=num_ctx, num_predict=_OUTPUT_TOKENS,
            )
        except GenerationError:
            logger.warning("relecture d'une part du cours impossible", exc_info=True)
            result.warnings.append("AUDIT_PART_FAILED")
            continue
        for values in _read(raw):
            if not _quote_in(values["extrait"], part):
                # Le correcteur cite une phrase qui n'est pas dans le cours :
                # son signalement ne peut pas être vérifié, on l'écarte.
                result.warnings.append("AUDIT_QUOTE_NOT_FOUND")
                continue
            severity = values.get("gravite", "").strip().lower()
            result.findings.append(AuditFinding(
                severity="ambiguite" if severity.startswith("ambig") else "probable",
                source="relecture",
                excerpt=values["extrait"].strip(" «»\"'"),
                explanation=values["explication"],
                correction=values.get("correction", ""),
            ))
    return result
