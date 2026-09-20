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
from typing import Dict, List, Optional

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


class AuditFailed(RuntimeError):
    """Le relecteur n'a pas pu se prononcer ; le message est montrable.

    `code` dit la CAUSE sans qu'on ait à reconnaître une phrase : un écran
    qui lit une phrase casse à la première reformulation.
    """

    def __init__(self, message: str, code: str = "relecteur_muet") -> None:
        super().__init__(message)
        self.code = code


def _cause(erreur: Exception) -> str:
    """Le code de cause, lu sur la réponse du fournisseur."""

    texte = str(erreur).lower()
    statut = getattr(erreur, "status", 0)
    if "credit balance" in texte or "quota" in texte or statut == 429:
        return "credit_epuise"
    if statut in (401, 403) or "api key" in texte or "unauthorized" in texte:
        return "cle_refusee"
    if statut in (404,) or "no longer available" in texte:
        return "modele_indisponible"
    return "relecteur_muet"


@dataclass(frozen=True)
class AuditFinding:
    severity: str  # certaine | probable | ambiguite
    source: str  # calcul | relecture
    excerpt: str
    explanation: str
    correction: str = ""
    # OÙ corriger : { kind: "section"|"quiz"|"exercice", id?, index? }. Sans
    # cible, le professeur devrait chercher lui-même le passage.
    target: Optional[dict] = None
    # Pour un quiz dont la réponse contredit son explication : la réponse
    # que l'explication annonce (0-3), applicable en un clic.
    suggested_answer: Optional[int] = None


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


_OUVRANTS = (r"\(", r"\[")
_FERMANTS = (r"\)", r"\]")


def _delimiteurs_orphelins(sections: List[dict]) -> List[AuditFinding]:
    """Une formule jamais refermée : l'élève lit des antislashs.

    Constaté le 20/09/2026 dans « Nombres complexes » : 44 « \\( » pour 43
    « \\) » dans une section. L'écran peut l'afficher proprement, il ne peut
    pas la réparer — c'est le texte qui est faux, et c'est certain : ça se
    compte, ça ne s'interprète pas.
    """

    trouves = []
    for section in sections:
        contenu = str(section.get("content") or "")
        for ouvrant, fermant in zip(_OUVRANTS, _FERMANTS):
            manquants = contenu.count(ouvrant) - contenu.count(fermant)
            if manquants == 0:
                continue
            trop = "ouverte" if manquants > 0 else "fermée"
            trouves.append(AuditFinding(
                severity="certaine", source="calcul",
                excerpt=f"{section.get('heading') or 'Section'}",
                explanation=(
                    f"{abs(manquants)} formule(s) {trop}(s) sans leur paire : "
                    f"« {ouvrant} » apparaît {contenu.count(ouvrant)} fois et "
                    f"« {fermant} » {contenu.count(fermant)} fois. L'élève voit "
                    "les antislashs au milieu du texte."
                ),
                target={"kind": "section", "id": section.get("id"),
                        "heading": section.get("heading")},
            ))
    return trouves


def _target_of(excerpt: str, sections: List[dict]) -> Optional[dict]:
    """La section qui contient CE passage, une seule fois.

    Deux sections qui le contiennent : on ne désigne rien plutôt que de
    faire corriger la mauvaise.
    """

    quote = _normalise(excerpt.strip(" «»\"'"))
    if len(quote) < 12:
        return None
    trouves = [s for s in sections if _normalise(str(s.get("content") or "")).count(quote) == 1]
    if len(trouves) != 1:
        return None
    return {"kind": "section", "id": trouves[0].get("id"), "heading": trouves[0].get("heading")}


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
    sections: List[dict] = (), title: str = "",
) -> AuditResult:
    result = AuditResult(findings=[])

    # Certain, sans modèle : la réponse enregistrée contredit l'explication.
    from app.core.generation import _quiz_contredit, reponse_annoncee

    for number, quiz in enumerate(quizzes, start=1):
        if _quiz_contredit(quiz):
            annoncee_index = reponse_annoncee(str(quiz.get("explanation") or ""))
            annoncee = "ABCD"[annoncee_index]
            enregistree = quiz.get("answer")
            result.findings.append(AuditFinding(
                severity="certaine", source="calcul",
                excerpt=str(quiz.get("question", "")),
                explanation=(
                    f"Quiz {number} : la bonne réponse enregistrée est "
                    f"{'ABCD'[enregistree] if isinstance(enregistree, int) and 0 <= enregistree < 4 else enregistree}, "
                    f"mais l'explication annonce {annoncee}. Un élève qui répond juste est compté faux."
                ),
                correction=f"Bonne réponse : {annoncee}",
                target={"kind": "quiz", "id": quiz.get("id"), "index": number - 1},
                suggested_answer=annoncee_index,
            ))
    result.findings.extend(_delimiteurs_orphelins(list(sections)))

    blocks = _blocks_text(list(quizzes), list(exercises))
    if blocks:
        text = f"{text}\n\n{blocks}"

    for finding in check_calculations(text):
        # L'extrait est la formule, RENDUE : « \( … \) ». Sans délimiteurs,
        # l'écran du professeur affiche du LaTeX brut, illisible (20/09/2026).
        entre = re.search(r"\\\((.+?)\\\)", finding.message, re.DOTALL)
        extrait = f"\\( {entre.group(1).strip()} \\)" if entre else finding.message
        result.findings.append(AuditFinding(
            severity="certaine", source="calcul",
            excerpt=extrait,
            explanation=finding.message,
            target=_target_of(extrait, list(sections)),
        ))

    for part in _parts(text):
        try:
            raw = await llm.chat(
                [{"role": "system", "content": _SYSTEM},
                 {"role": "user", "content": (
                     (f"Cours : {title}\n\n" if title else "")
                     + f"Passage à corriger :\n\n{part}"
                 )}],
                timeout=timeout, num_ctx=num_ctx, num_predict=_OUTPUT_TOKENS,
            )
        except GenerationError as erreur:
            # Le relecteur ne répond pas (clé refusée, crédit épuisé, panne) :
            # on ARRÊTE. Continuer sur les parts suivantes ferait attendre le
            # professeur de longues minutes pour rendre un résultat partiel
            # qu'on ne croira pas (20/09/2026).
            logger.warning("relecture impossible : le relecteur ne répond pas", exc_info=True)
            raise AuditFailed(str(erreur), _cause(erreur)) from erreur
        for values in _read(raw):
            if not _quote_in(values["extrait"], part):
                # Le correcteur cite une phrase qui n'est pas dans le cours :
                # son signalement ne peut pas être vérifié, on l'écarte.
                result.warnings.append("AUDIT_QUOTE_NOT_FOUND")
                continue
            severity = values.get("gravite", "").strip().lower()
            extrait = values["extrait"].strip(" «»\"'")
            result.findings.append(AuditFinding(
                severity="ambiguite" if severity.startswith("ambig") else "probable",
                source="relecture",
                excerpt=extrait,
                explanation=values["explication"],
                correction=values.get("correction", ""),
                target=_target_of(extrait, list(sections)),
            ))
    return result
