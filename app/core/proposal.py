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

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import List, Optional, Protocol, Sequence


class _Llm(Protocol):
    async def complete(self, system: str, user: str) -> str: ...


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
