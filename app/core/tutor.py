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

import json
import logging
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import List, Optional

from app.config import Settings
from app.core.generation import (
    _FORMULES,
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
- Réponds à la DERNIÈRE question de l'élève, précisément. Les échanges précédents ne servent que de contexte : ne répète jamais une explication que tu as déjà donnée.
- Sois professionnel et mature, comme un enseignant expérimenté : bienveillant mais rigoureux, jamais familier, jamais approximatif.
- Avant de répondre, assure-toi d'avoir assez d'éléments : demande autant de recherches que nécessaire plutôt que de répondre avec des extraits insuffisants.
- Adapte ton langage à la classe de l'élève : phrases courtes pour les petits, vocabulaire précis pour les grands.
- Structure : l'idée en une phrase, puis l'explication pas à pas, puis UN exemple concret. Reste court.
- Si la question est un exercice à résoudre : n'en donne JAMAIS la solution. Explique la méthode, donne un indice, laisse l'élève faire.
- Le cours fait foi : les définitions, formules, notations et méthodes viennent des extraits fournis, jamais de ta mémoire. Relis chaque formule contre l'extrait avant de l'écrire.
- Pour EXPLIQUER (reformuler, donner une image, un exemple), tu peux t'appuyer sur tes propres connaissances, sans contredire le cours.
- Si les extraits ne contiennent pas la réponse : réponds avec tes connaissances, en commençant EXACTEMENT par « Ce n'est pas dans ton cours, mais voici ce que je sais : ».
- Explique directement, comme un professeur. N'écris jamais « [S1] indique que », « selon l'extrait » ou « le document dit » : place l'étiquette [S1] en fin de phrase, comme une référence.
- Si l'élève demande POURQUOI : explique d'où ça vient (la raison, la démonstration courte, l'intuition). Ne te contente pas de répéter l'énoncé.
- Ne mets une étiquette [S1] que si CET extrait dit vraiment ce que tu écris. Ce qui vient de tes connaissances ne porte aucune étiquette.
- Termine par une petite question qui vérifie que l'élève a compris : elle va sous « ### VÉRIFICATION », pas dans l'explication.
-{formules}

Pour chercher dans les documents, réponds SEULEMENT :
{{"chercher": {{"question": "...", "nature": "cours-publie|support-cours|programme-officiel"}}}}

Pour répondre à l'élève, écris SEULEMENT, dans ce format (jamais de JSON pour la réponse) :
### RÉPONSE
l'explication, avec les étiquettes [S1]…
### VÉRIFICATION
la petite question finale
### NOTIONS
notion1, notion2"""


class Tutor:
    """L'orchestrateur : Lawal choisit quoi chercher, nous où il a le droit."""

    def __init__(
        self,
        *,
        llm: LlmProvider,
        retriever: Retriever,
        settings: Settings,
        notebooks=None,
        embeddings=None,
    ) -> None:
        self._llm = llm
        self._retriever = retriever
        self._settings = settings
        # Le cahier de l'élève, quand il en a un. Optionnel : le tuteur doit
        # continuer de fonctionner pour un élève qui n'a jamais rien scanné.
        self._notebooks = notebooks
        self._embeddings = embeddings

    async def _search_notebook(
        self, *, question: str, student_account_id: str, document_id: str
    ) -> List[Passage]:
        """Les passages du cahier de CET élève — jamais de celui d'un autre.

        Le propriétaire est passé au dépôt, qui le refuse vide. Rien ici ne
        peut ouvrir le cahier de quelqu'un d'autre, même par erreur.
        """

        if self._embeddings is None:
            return []
        vectors = await self._embeddings.embed([question])
        rows = await self._notebooks.search(
            student_account_id=student_account_id,
            embedding=vectors[0],
            limit=4,
        )
        return [
            Passage(
                chunk_id=row["chunk_id"],
                document_id=row["document_id"],
                title=row["title"],
                locator=row["locator"],
                content=row["content"][:700],
                language=row.get("language", "fr"),
                score=1.0 - float(row.get("distance", 1.0)),
            )
            for row in rows
            if row["document_id"] == document_id
        ]

    async def answer(
        self,
        *,
        question: str,
        scope: Scope,
        course_id: str = "",
        section_heading: str = "",
        history: Optional[List[dict]] = None,
        student_account_id: str = "",
        notebook_document_id: str = "",
    ) -> TutorAnswer:
        queries: List[dict] = []
        warnings: List[str] = []
        passages: List[Passage] = []
        # Les passages venant du CAHIER de l'élève, retenus à part : ils
        # n'ont pas la même autorité qu'un cours validé, et il doit le voir.
        notebook_ids: set = set()
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
                # Chaîne vide = pas de cours (question sur le cahier) :
                # il faut None, sinon le filtre SQL cherche un cours dont
                # l'identifiant est vide et ne rend jamais rien.
                course_id=(
                    course_id
                    if course_id and role != "programme-officiel"
                    else None
                ),
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

        # Le cahier D'ABORD quand la question porte sur un cours qu'il a
        # lui-même ajouté : c'est SON cours, celui qu'il a sous les yeux.
        # Le contenu validé vient ensuite, en appui — et les deux se
        # distinguent dans les citations, parce qu'ils n'ont pas la même
        # autorité : ses notes peuvent être fausses, le cours du prof non.
        if notebook_document_id and student_account_id and self._notebooks:
            found_notebook = await self._search_notebook(
                question=probe,
                student_account_id=student_account_id,
                document_id=notebook_document_id,
            )
            fresh = [p for p in found_notebook if p.chunk_id not in seen]
            seen.update(p.chunk_id for p in fresh)
            notebook_ids.update(p.chunk_id for p in fresh)
            passages.extend(fresh)
            queries.append(
                {"query": probe, "role": "cahier", "results": len(fresh),
                 "fromModel": False}
            )
            if not fresh:
                warnings.append("NOTEBOOK_NO_CONTENT")

        found = await search(probe, "cours-publie", from_model=False)
        if found == 0:
            warnings.append("NO_PUBLISHED_COURSE_CONTENT")
            # Le cours est un résumé court : l'approfondissement vit dans
            # les supports. On les consulte d'office si le cours se tait.
            await search(probe, "support-cours", from_model=False)

        if not passages and notebook_document_id:
            # Question sur SES notes, et ses notes n'en parlent pas : lui
            # répondre de mémoire lui ferait croire que c'est dans son cahier.
            return TutorAnswer(
                text=(
                    "Je n'ai pas trouvé de quoi répondre dans les notes que "
                    "tu as ajoutées. Vérifie que la page que tu cherches est "
                    "bien dans ce cours — et si c'est un mot précis, essaie "
                    "de me le demander autrement."
                ),
                check="",
                concepts=[],
                citations=[],
                queries=queries,
                warnings=warnings + ["INSUFFICIENT_EVIDENCE"],
            )
        if not passages:
            # Rien de validé ne couvre la question. Décision d'Alioune
            # (15/09/2026) : Lawal répond quand même avec ses connaissances,
            # mais le DIT à l'élève — la règle est dans le prompt.
            warnings.append("INSUFFICIENT_EVIDENCE")

        history = history or []
        messages = [
            {
                "role": "system",
                "content": _SYSTEM.format(
                    contexte=_context_line(scope), formules=_FORMULES
                ),
            }
        ]
        for turn in history:
            role = "assistant" if turn.get("role") == "lawal" else "user"
            messages.append({"role": role, "content": str(turn.get("content", ""))})
        situation = (
            f"L'élève lit la section « {section_heading} » du cours.\n\n"
            if section_heading
            else ""
        )
        # La question vient APRÈS les extraits, au plus près de ce que le
        # modèle écrit : constaté le 15/09/2026, noyée entre l'historique et
        # les extraits, elle laissait qwen recopier sa réponse précédente.
        messages.append(
            {
                "role": "user",
                "content": (
                    situation
                    + (
                        "Extraits validés :\n\n" + _render_passages(passages, "S", 1)
                        if passages
                        else "Aucun extrait du cours ne couvre cette question."
                    )
                    + f"\n\nQuestion de l'élève, celle à laquelle tu réponds : {question}"
                ),
            }
        )
        previous_answers = [
            str(turn.get("content", "")) for turn in history if turn.get("role") == "lawal"
        ]
        repetition_reminded = False

        format_reminded = False
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
            # Le format balisé d'abord : constaté au banc du 15/09/2026, le
            # JSON cassait sur les réponses longues à formules (2 sur 8) et
            # l'élève voyait les accolades. Le JSON reste lu, par tolérance.
            parsed = _lire_reponse(raw) or _parse_json_block(raw)
            if not (parsed and str(parsed.get("reponse", "")).strip()):
                # qwen émet parfois {"reponse":…} PUIS {"verification":…} :
                # deux objets côte à côte — on les fusionne avant d'abandonner.
                parsed = _merge_json_blocks(raw)
            if parsed and str(parsed.get("reponse", "")).strip():
                text = _sans_compris(str(parsed["reponse"]).strip())
                if not repetition_reminded and _repete(text, previous_answers):
                    # Constaté le 15/09/2026 : à « comment reconnaît-on une
                    # similitude ? », qwen a recopié sa réponse sur le centre.
                    # Une relance ciblée ; la seconde réponse est gardée.
                    repetition_reminded = True
                    warnings.append("TUTOR_REPETITION_RETRIED")
                    messages.append({"role": "assistant", "content": raw})
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Tu viens de répéter ta réponse précédente. "
                                "L'élève pose une AUTRE question : « "
                                + question
                                + " ». Réponds à celle-ci, avec une explication "
                                "nouvelle, dans le même format."
                            ),
                        }
                    )
                    continue
                text = await self._review(
                    question=question,
                    text=text,
                    passages=passages,
                    warnings=warnings,
                )
                if text.startswith("Ce n'est pas dans ton cours"):
                    warnings.append("ANSWER_OUTSIDE_COURSE")
                return TutorAnswer(
                    text=text,
                    check=str(parsed.get("verification", "")).strip(),
                    concepts=[
                        str(c).strip()
                        for c in parsed.get("conceptes", [])
                        if str(c).strip()
                    ][:6],
                    citations=_citations(passages, notebook_ids),
                    queries=queries,
                    warnings=warnings,
                )
            # Réponse hors format. Une seule relance pour le format ; ensuite
            # le texte libre est ACCEPTÉ tel quel — constaté au premier test
            # réel (02/09/2026) : qwen répondait une bonne explication en
            # prose, et l'élève recevait un échec après 4 minutes. Une bonne
            # réponse sans question de vérification vaut mieux que pas de
            # réponse du tout.
            plain = _sauver_json(raw) or raw.strip().strip("`").strip()
            if format_reminded and len(plain) > 80:
                warnings.append("TUTOR_PLAIN_TEXT")
                plain = await self._review(
                    question=question, text=plain, passages=passages, warnings=warnings
                )
                return TutorAnswer(
                    text=plain,
                    check="",
                    concepts=[],
                    citations=_citations(passages, notebook_ids),
                    queries=queries,
                    warnings=warnings,
                )
            format_reminded = True
            messages.append({"role": "assistant", "content": raw})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Réponds uniquement dans le format demandé : "
                        "### RÉPONSE, ### VÉRIFICATION, ### NOTIONS."
                    ),
                }
            )

        raise TutorFailed(
            "Lawal n'a pas réussi à formuler une réponse : réessaie, ou "
            "pose la question au professeur."
        )

    async def _review(
        self,
        *,
        question: str,
        text: str,
        passages: List[Passage],
        warnings: List[str],
    ) -> str:
        """Relire la réponse AVANT que l'élève la voie : répond-elle à la
        question, chaque formule est-elle celle du cours ?

        Constaté le 15/09/2026 : « si elle n'est pas une translation
        (c'est-à-dire que |a| = 1) » — faux, et contraire au cours. Un petit
        modèle se relit mieux qu'il n'écrit. Mais il peut aussi « corriger »
        ce qui était juste : la correction n'est retenue que s'il NOMME
        l'erreur, et qu'il rend une réponse complète et différente.
        Toute panne de la relecture laisse la réponse d'origine.
        """

        if not self._settings.answer_review:
            return text

        sources = (
            _render_passages(passages, "S", 1)
            if passages
            else "(aucun extrait : la réponse vient des connaissances du tuteur)"
        )
        messages = [
            {"role": "system", "content": _REVIEW_SYSTEM + _FORMULES},
            {
                "role": "user",
                "content": (
                    f"Extraits du cours :\n\n{sources}\n\n"
                    f"Question de l'élève : {question}\n\n"
                    f"Réponse du tuteur :\n{text}"
                ),
            },
        ]
        try:
            raw = await self._chat(messages)
        except Exception:  # noqa: BLE001 — la relecture ne doit jamais coûter la réponse
            logger.warning("relecture de Lawal impossible", exc_info=True)
            warnings.append("TUTOR_REVIEW_FAILED")
            return text
        verdict = _lire_relecture(raw)
        if verdict is None:
            return text
        error, corrected = verdict
        if (
            not error
            or len(corrected) < 0.4 * len(text)
            or _normalise(corrected) == _normalise(text)
        ):
            return text
        logger.info("réponse de Lawal corrigée à la relecture", extra={"erreur": error[:300]})
        warnings.append("TUTOR_ANSWER_REVISED")
        return _sans_compris(corrected)

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


_REVIEW_SYSTEM = """Tu relis la réponse d'un tuteur à un élève, AVANT qu'il la voie. Tu es exigeant sur deux points seulement :
1. Répond-elle à LA question posée (et pas à une autre) ?
2. Chaque définition, formule et condition est-elle exacte et conforme aux extraits du cours ? Vérifie les conditions (≠, =, <), les signes, les modules, les arguments.

Ne touche ni au style, ni à la longueur, ni aux étiquettes [S1]. Si tout est juste, ne change rien.

Réponds dans ce format, sans rien d'autre :
### VERDICT
OK ou ERREUR
### ERREUR
si ERREUR : la phrase fausse, puis pourquoi, en citant l'extrait qui la contredit
### RÉPONSE CORRIGÉE
si ERREUR : la réponse entière, corrigée, à la place de l'ancienne
"""

_COMPRIS_FINAL = re.compile(r"\s*(?:\n|^)?\s*(?:Compris|C'est clair|D'accord)\s*\?\s*$", re.IGNORECASE)


_BALISE = re.compile(r"^\s*#{2,4}\s*(R[ÉE]PONSE|V[ÉE]RIFICATION|NOTIONS)\s*:?\s*$", re.IGNORECASE | re.MULTILINE)


def _lire_reponse(raw: str) -> Optional[dict]:
    """La réponse balisée, sous la même forme que l'ancien JSON — ou None."""

    marks = list(_BALISE.finditer(raw))
    if not marks:
        return None
    parts = {}
    for index, mark in enumerate(marks):
        end = marks[index + 1].start() if index + 1 < len(marks) else len(raw)
        key = mark.group(1).upper().replace("É", "E")
        parts.setdefault(key, raw[mark.end():end].strip())
    if not parts.get("REPONSE"):
        return None
    notions = [n.strip(" -•\t") for n in re.split(r"[,\n]", parts.get("NOTIONS", ""))]
    return {
        "reponse": parts["REPONSE"],
        "verification": parts.get("VERIFICATION", ""),
        "conceptes": [n for n in notions if n],
    }


def _sauver_json(raw: str) -> str:
    """Un JSON cassé ne s'affiche JAMAIS tel quel : on en retire le texte de
    « reponse ». Vu au banc du 15/09/2026 : l'élève recevait {"reponse": …}."""

    text = raw.strip()
    if not text.startswith("{"):
        return ""
    found = re.search(r'"reponse"\s*:\s*"(.*?)"\s*(?:,\s*"(?:verification|conceptes)"|\}\s*$|\}\s*\{)', text, re.DOTALL)
    if found:
        body = found.group(1)
    else:
        # Réponse coupée par la limite de sortie : pas de guillemet final.
        start = re.search(r'"reponse"\s*:\s*"', text)
        if not start:
            return ""
        body = text[start.end():].rstrip('"} \n')
    return body.replace("\\n", "\n").replace('\\"', '"').strip()


def _sans_compris(text: str) -> str:
    """Le « Compris ? » que qwen ajoute en fin d'explication doublonne la
    question de vérification, affichée à part."""

    return _COMPRIS_FINAL.sub("", text).strip()


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _repete(text: str, previous: List[str]) -> bool:
    """La réponse recopie-t-elle une réponse déjà donnée dans le fil ?"""

    current = _normalise(text)
    return any(
        SequenceMatcher(None, current, _normalise(old)).ratio() >= _REPETITION_RATIO
        for old in previous
        if old.strip()
    )


# Mesuré sur les vraies réponses du 15/09/2026, même cours : la réponse
# « reconnaître » recopiée sur la réponse « centre » → 0,65 ; les réponses
# « angle » et « centre », réellement différentes → 0,05 et 0,16.
_REPETITION_RATIO = 0.5


def _lire_relecture(raw: str):
    """(erreur, réponse corrigée) si la relecture signale une erreur
    argumentée ; ("", "") si elle dit OK ; None si elle est illisible."""

    sections = {}
    current = None
    for line in raw.splitlines():
        heading = re.match(r"^\s*#{2,4}\s*(VERDICT|ERREUR|R[ÉE]PONSE CORRIG[ÉE]E)\s*$", line, re.IGNORECASE)
        if heading:
            current = heading.group(1).upper().replace("É", "E")
            sections[current] = []
            continue
        if current:
            sections[current].append(line)
    if "VERDICT" not in sections:
        return None
    verdict = " ".join(sections["VERDICT"]).strip().upper()
    if verdict.startswith("OK"):
        return "", ""
    if not verdict.startswith("ERREUR"):
        return None
    error = "\n".join(sections.get("ERREUR", [])).strip()
    corrected = "\n".join(sections.get("REPONSE CORRIGEE", [])).strip()
    return error, corrected


def _merge_json_blocks(raw: str) -> dict:
    """Fusionner tous les objets JSON équilibrés d'une réponse.

    Vu au premier test réel : qwen émet parfois {"reponse": …} PUIS
    {"verification": …} — deux objets côte à côte. Le découpage naïf
    premier-{ / dernier-} produisait un JSON invalide et l'élève recevait
    les accolades brutes. Ici chaque bloc équilibré est lu, les champs se
    cumulent.
    """

    merged: dict = {}
    depth, start = 0, None
    in_string, escape = False, False
    for index, char in enumerate(raw):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    block = json.loads(raw[start : index + 1])
                    if isinstance(block, dict):
                        merged.update(block)
                except json.JSONDecodeError:
                    pass
                start = None
    return merged


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


def _citations(
    passages: List[Passage], notebook_ids: Optional[set] = None
) -> List[dict]:
    """Chaque extrait dit d'où il vient : du cahier de l'élève, ou du contenu
    validé. Les deux n'ont pas la même autorité — ses propres notes peuvent
    être fausses, et il doit pouvoir en tenir compte."""

    notebook_ids = notebook_ids or set()
    return [
        {
            "label": f"S{index}",
            "documentId": passage.document_id,
            "title": passage.title,
            "locator": passage.locator,
            "source": "cahier" if passage.chunk_id in notebook_ids else "valide",
        }
        for index, passage in enumerate(passages, start=1)
    ]
