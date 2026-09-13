"""Proposer une correction sur UN passage relu — jamais l'appliquer.

Décision d'Alioune du 13/09/2026, prise en connaissance du risque : on a
MESURÉ cette aide se tromper. Le 29/08/2026, sur le programme national de
maths, le modèle a fabriqué des formules plausibles et fausses — « U+V ≤
√(U²+V²) » à la place de l'inégalité triangulaire — et le score de qualité
valait 0,98 sur ce texte faux. **Un score ne voit pas une erreur de sens.**

Alors on ne rend jamais un texte corrigé : on rend une PROPOSITION, à côté
de l'original, avec ce qu'elle change. Quatre garde-fous, et chacun répond
à une façon précise de se tromper observée à ce moment-là :

1. **Un passage à la fois, jamais le document.** Ce qui se relit ligne à
   ligne se vérifie ; ce qui se réécrit en bloc se survole.
2. **Le modèle a le droit de ne rien proposer.** « Je ne sais pas » est une
   réponse acceptée et rendue telle quelle — c'est ce qui sépare une aide
   d'une invention.
3. **Une réécriture est refusée, pas montrée.** Si la proposition s'écarte
   trop en longueur, le modèle n'a pas réparé une transcription : il a
   rédigé autre chose.
4. **Les symboles sont comparés un par un.** Une réparation de
   transcription remet des espaces et recolle des mots coupés ; elle ne
   change ni un chiffre, ni un opérateur. Tout symbole ajouté, retiré ou
   remplacé est NOMMÉ dans l'avertissement — c'est exactement ce qui aurait
   attrapé le `√` apparu et le `‖` disparu en août.

Ce quatrième point ne protège pas que les maths. Une date, un pourcentage,
une formule chimique, une année en histoire : partout, le symbole porte le
sens et la prose l'habille.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import List, Optional, Protocol, Sequence


class _Llm(Protocol):
    async def complete(self, system: str, user: str) -> str: ...

    async def chat(
        self,
        messages: List[dict],
        *,
        timeout: float,
        num_ctx: int,
        num_predict: int,
        schema: Optional[dict] = None,
    ) -> str: ...


# Ce que le modèle répond quand il ne sait pas. Court, littéral, et
# impossible à confondre avec un passage réparé.
INCERTAIN = "[?]"

# Au-delà de cet écart de longueur, ce n'est plus une réparation mais une
# rédaction. Deux seuils, parce qu'un seul ne marche pas : en proportion
# pour les longs passages, ET un plancher en caractères pour les courts.
# Sans le plancher, « U+V ≤ U+V » → « U+V ≤ √(U²+V²) » est écarté comme une
# réécriture (+55 %) alors que c'est cinq caractères — et on perdrait
# justement l'avertissement sur le « √ » ajouté, qui est tout l'intérêt.
_ECART_MAX = 0.40
_ECART_PLANCHER = 12

# Ce qui PORTE le sens : chiffres, opérateurs, lettres grecques, symboles
# scientifiques. La prose autour peut bouger, pas ceux-là.
_PORTEURS = re.compile(
    r"[0-9"
    r"+\-*/=<>%°±×÷≈≠≤≥∞√∑∏∫∈∉⊂⊃∪∩∅→←↔⇒⇔∀∃∂∇‖"
    r"αβγδεζηθικλμνξοπρστυφχψω"
    r"ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ]"
)

_CONSIGNE = """Tu répares une TRANSCRIPTION de document, tu ne rédiges pas.

Le passage vient d'un PDF mal lu. Tu peux seulement :
- remettre des espaces manquants,
- recoller un mot coupé,
- retirer un caractère parasite d'extraction.

Tu ne dois JAMAIS :
- changer un chiffre, un symbole, une formule, une unité, une date ;
- reformuler, résumer, compléter, expliquer ;
- ajouter ce qui n'est pas là, même si le passage semble incomplet.

Si tu n'es pas certain, réponds exactement {incertain} et rien d'autre.
Un passage laissé abîmé est préférable à un passage faux.

Réponds uniquement par le passage réparé, sans guillemets ni commentaire."""


@dataclass(frozen=True)
class Proposition:
    """Ce qu'on rend au professeur : les deux textes, et ce qui a bougé."""

    passage: str
    proposition: Optional[str]
    changee: bool
    incertaine: bool
    # Les symboles ajoutés, retirés ou remplacés. Vide = la proposition ne
    # touche qu'à la mise en forme, c'est le cas sûr.
    symboles_modifies: List[str]
    avertissement: Optional[str]


def porteurs(texte: str) -> Counter:
    """Les symboles porteurs de sens, comptés.

    On normalise en NFKC d'abord : « ² » et « 2 » en exposant ne doivent pas
    passer pour deux symboles différents selon l'humeur de l'extracteur.
    """

    return Counter(_PORTEURS.findall(unicodedata.normalize("NFKC", texte)))


def _difference(avant: str, apres: str) -> List[str]:
    """Ce qui a changé parmi les symboles, dans les deux sens."""

    a, b = porteurs(avant), porteurs(apres)
    ajoutes = sorted((b - a).elements())
    retires = sorted((a - b).elements())
    return [f"+{s}" for s in ajoutes] + [f"-{s}" for s in retires]


def _ecart_excessif(avant: str, apres: str) -> bool:
    if not avant:
        return True
    ecart = abs(len(apres) - len(avant))
    return ecart > _ECART_PLANCHER and ecart / len(avant) > _ECART_MAX


async def proposer(
    llm: _Llm,
    passage: str,
    *,
    signalements: Sequence[str] = (),
) -> Proposition:
    """Une proposition pour UN passage. N'applique rien, ne décide rien."""

    passage = passage.strip()
    if not passage:
        return Proposition(passage, None, False, True, [], "Passage vide.")

    indices = (
        f"\n\nL'extraction a signalé : {', '.join(signalements)}."
        if signalements
        else ""
    )
    brut = await llm.complete(
        _CONSIGNE.format(incertain=INCERTAIN),
        f"Passage à réparer :\n{passage}{indices}",
    )
    reponse = (brut or "").strip().strip('"').strip()

    if not reponse or reponse == INCERTAIN or INCERTAIN in reponse:
        return Proposition(
            passage,
            None,
            False,
            True,
            [],
            "Le modèle n'a pas su réparer ce passage — il le dit lui-même. "
            "À corriger à la main, ou à laisser tel quel.",
        )

    if reponse == passage:
        return Proposition(passage, None, False, False, [], None)

    if _ecart_excessif(passage, reponse):
        # Le modèle a rédigé au lieu de réparer. On ne le montre même pas :
        # une reformulation crédible est ce qu'on cherche à éviter.
        return Proposition(
            passage,
            None,
            False,
            True,
            [],
            "Le modèle a réécrit le passage au lieu de le réparer — "
            "proposition écartée.",
        )

    bouges = _difference(passage, reponse)
    avertissement = (
        "⚠️ Cette proposition CHANGE des symboles porteurs de sens : "
        + ", ".join(bouges)
        + ". Vérifiez sur le document d'origine avant d'accepter — "
        "c'est exactement ainsi qu'une formule fausse passe pour vraie."
        if bouges
        else None
    )
    return Proposition(passage, reponse, True, False, bouges, avertissement)


# ─────────────────────────── LE CHAT DE RELECTURE ───────────────────────────
#
# Décision d'Alioune du 13/09/2026 : le professeur doit pouvoir DIRE ce qu'il
# veut changer — « à la partie sur le centre, remplace θ par l'angle » — au
# lieu de sélectionner un passage à la souris. Pour comprendre « cette
# partie », le modèle a besoin du texte ENTIER.
#
# Mais lui donner le texte entier et lui demander de le corriger, c'est
# exactement ce qui a produit les formules fausses du 29/08. Alors il voit
# tout et ne rend que des REMPLACEMENTS CIBLÉS : `avant` / `après`. Trois
# vérifications déterministes suivent, et aucune ne fait confiance au modèle :
#
#   - `avant` doit se trouver TEL QUEL dans le texte, sinon la correction
#     désigne un endroit qui n'existe pas — le modèle a cité de mémoire ;
#   - `avant` doit s'y trouver UNE SEULE fois, sinon on ne sait pas laquelle
#     il vise, et appliquer au hasard serait pire que refuser ;
#   - les symboles porteurs sont comparés, comme pour une proposition simple.
#
# Le texte n'est jamais modifié ici. On rend au professeur ce qui changerait.

_CORRECTIONS_SCHEMA = {
    "type": "object",
    "properties": {
        "reponse": {"type": "string"},
        "corrections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "avant": {"type": "string"},
                    "apres": {"type": "string"},
                },
                "required": ["avant", "apres"],
            },
        },
    },
    "required": ["reponse", "corrections"],
}

_CONSIGNE_CHAT = """Tu aides un professeur à relire un texte extrait d'un PDF.

Tu reçois le texte entier pour COMPRENDRE de quoi il parle. Tu ne le
réécris jamais. Tu réponds par des remplacements ciblés.

Pour chaque correction :
- `avant` : le passage EXACT tel qu'il apparaît dans le texte, copié
  caractère pour caractère. Ne le cite pas de mémoire, relis-le.
- `apres` : ce par quoi le remplacer.

Règles :
- n'invente aucun contenu, ne complète pas ce qui manque ;
- ne change un chiffre, un symbole ou une formule QUE si le professeur te
  le demande explicitement ;
- si tu n'as pas compris ce qu'il veut, rends `corrections` vide et
  demande-lui de préciser dans `reponse` ;
- `reponse` s'adresse au professeur, en français, brièvement.

Un texte laissé tel quel est préférable à un texte faux."""


@dataclass(frozen=True)
class Correction:
    avant: str
    apres: str
    # Où le passage commence dans le texte — pour que l'écran le montre.
    position: int
    symboles_modifies: List[str]
    avertissement: Optional[str]


@dataclass(frozen=True)
class Discussion:
    """Ce que le tour de chat produit. Rien n'est appliqué."""

    reponse: str
    corrections: List[Correction]
    # Ce que le modèle a proposé et qu'on a refusé, avec la raison. Montré
    # au professeur : un refus silencieux lui ferait croire que l'IA n'a
    # rien trouvé.
    refusees: List[dict]


def _situer(texte: str, avant: str) -> tuple[Optional[int], Optional[str]]:
    """Où se trouve ce passage — et seulement s'il s'y trouve une seule fois."""

    if not avant:
        return None, "Passage vide."
    occurrences = texte.count(avant)
    if occurrences == 0:
        return None, (
            "Ce passage ne se trouve pas dans le texte — le modèle l'a cité "
            "de mémoire au lieu de le relire."
        )
    if occurrences > 1:
        return None, (
            f"Ce passage apparaît {occurrences} fois : impossible de savoir "
            "lequel corriger."
        )
    return texte.index(avant), None


async def discuter(
    llm: _Llm,
    texte: str,
    consigne: str,
    *,
    historique: Sequence[dict] = (),
    timeout: float = 180.0,
    num_ctx: int = 8192,
    num_predict: int = 2048,
) -> Discussion:
    """Un tour de discussion sur le texte extrait. N'applique rien."""

    messages = [{"role": "system", "content": _CONSIGNE_CHAT}]
    for tour in historique:
        role = "assistant" if tour.get("role") in ("ia", "assistant") else "user"
        messages.append({"role": role, "content": str(tour.get("content", ""))})
    messages.append(
        {
            "role": "user",
            "content": f"Texte extrait :\n---\n{texte}\n---\n\nDemande : {consigne}",
        }
    )

    brut = await llm.chat(
        messages,
        timeout=timeout,
        num_ctx=num_ctx,
        num_predict=num_predict,
        schema=_CORRECTIONS_SCHEMA,
    )
    try:
        charge = json.loads(brut)
    except (json.JSONDecodeError, TypeError):
        return Discussion(
            "Je n'ai pas su répondre à cette demande — reformulez-la.", [], []
        )

    retenues: List[Correction] = []
    refusees: List[dict] = []
    for brute in charge.get("corrections") or []:
        avant = str(brute.get("avant", ""))
        apres = str(brute.get("apres", ""))
        position, raison = _situer(texte, avant)
        if raison is not None:
            refusees.append({"avant": avant, "apres": apres, "raison": raison})
            continue
        if avant == apres:
            continue
        bouges = _difference(avant, apres)
        retenues.append(
            Correction(
                avant=avant,
                apres=apres,
                position=position,
                symboles_modifies=bouges,
                avertissement=(
                    "⚠️ Cette correction CHANGE des symboles porteurs de sens : "
                    + ", ".join(bouges)
                    + ". Vérifiez sur le document d'origine avant d'accepter."
                    if bouges
                    else None
                ),
            )
        )

    return Discussion(
        reponse=str(charge.get("reponse", "")).strip(),
        corrections=retenues,
        refusees=refusees,
    )
