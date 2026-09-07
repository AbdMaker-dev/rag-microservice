"""On ne corrige un cahier que par PREUVE.

Ce que ces tests protègent tient en une phrase : une formule inventée mais
crédible est pire qu'une formule visiblement cassée. Awa repère la seconde,
jamais la première — et elle réviserait dessus.
"""

import inspect

from app.core import repair as module
from app.core.repair import Proof, looks_broken, proof_from_hits, repair_by_proof


def _proof(text: str, distance: float) -> Proof:
    return Proof(text=text, title="Cours validé", locator="p. 4", distance=distance)


def test_un_passage_prouve_est_remplace_par_le_texte_validé():
    """La correction est une CITATION : le texte substitué est littéralement
    celui du cours validé, jamais une reformulation du modèle."""

    result = repair_by_proof(
        "z = r(cos0 + isin0)",
        lambda _: _proof("z = r(cos θ + i sin θ)", 0.08),
    )
    segment = result.segments[0]
    assert segment.status == "corrige"
    assert segment.text == "z = r(cos θ + i sin θ)"
    assert segment.proof is not None and segment.proof.title == "Cours validé"
    assert "NOTEBOOK_SEGMENTS_CORRECTED" in result.warnings


def test_sans_preuve_un_passage_casse_est_SIGNALÉ_pas_devine():
    """Aucune couverture validée → on le dit. C'est la règle qui empêche
    l'invention plausible."""

    result = repair_by_proof("z = r(cos θ + i sin θ", lambda _: None)
    segment = result.segments[0]
    assert segment.status == "a-verifier"
    assert segment.text == "z = r(cos θ + i sin θ"  # inchangé
    assert segment.reason == "PARENTHESE_NON_FERMEE"
    assert "NOTEBOOK_SEGMENTS_TO_CHECK" in result.warnings


def test_une_preuve_trop_lointaine_ne_corrige_rien():
    """Un passage du même chapitre qui parle d'autre chose n'est pas une
    preuve. Le seuil vit dans le module, pas chez l'appelant."""

    result = repair_by_proof(
        "La rotation conserve les distances",
        lambda _: _proof("Le produit scalaire est bilinéaire", 0.62),
    )
    assert result.segments[0].status == "inchange"
    assert result.segments[0].text == "La rotation conserve les distances"


def test_un_passage_deja_conforme_n_est_pas_recopié():
    """Quand le cours validé dit déjà exactement ça, on ne touche à rien :
    une correction cosmétique ferait douter Awa de sa propre copie."""

    result = repair_by_proof(
        "Une similitude directe conserve les angles orientés.",
        lambda _: _proof("une similitude directe conserve les angles orientés", 0.05),
    )
    assert result.segments[0].status == "inchange"
    assert result.corrected == 0


def test_le_texte_reconstruit_garde_l_ordre_des_lignes():
    """Un cahier se lit dans l'ordre où il a été écrit."""

    def find(segment: str):
        return _proof("LIGNE CORRIGÉE", 0.05) if segment == "b" else None

    result = repair_by_proof("a\n\nb\nc", find)
    assert [s.original for s in result.segments] == ["a", "b", "c"]
    assert result.text() == "a\nLIGNE CORRIGÉE\nc"


def test_les_signes_de_mauvaise_lecture_sont_mecaniques():
    """On ne juge pas le SENS d'un passage — seulement si la lecture a
    déraillé. Un cahier peut contenir une phrase fausse, ce n'est pas notre
    affaire ; une parenthèse ouverte et jamais fermée, si."""

    assert looks_broken("f(x) = 2x + 1") == ""
    assert looks_broken("f(x = 2x + 1") == "PARENTHESE_NON_FERMEE"
    assert looks_broken("Le mot est illisible �") == "CARACTERE_ILLISIBLE"
    assert looks_broken("f(x) = 2x + 1 avec x réel") == ""
    assert looks_broken("¤¶§¤¶ ¤¶§¤ ¶§¤¶§ ¤¶§") == "CARACTERES_INATTENDUS"
    assert looks_broken("") == ""


def test_une_transcription_vide_le_dit():
    result = repair_by_proof("   \n\n  ", lambda _: None)
    assert result.segments == []
    assert "NOTEBOOK_EMPTY_TRANSCRIPTION" in result.warnings


def test_la_meilleure_preuve_est_la_plus_proche():
    hits = [
        {"content": "loin", "distance": 0.5, "title": "A", "locator": "p.1"},
        {"content": "proche", "distance": 0.05, "title": "B", "locator": "p.2"},
    ]
    proof = proof_from_hits(hits)
    assert proof is not None and proof.text == "proche" and proof.title == "B"


def test_aucune_preuve_disponible_rend_None():
    """Rendre None plutôt qu'une preuve faible : c'est ce qui fait remonter
    « signalé » au lieu de « corrigé »."""

    assert proof_from_hits([]) is None
    assert proof_from_hits([{"content": "  ", "distance": 0.01}]) is None


def test_aucun_modele_de_langue_n_intervient_dans_la_reparation():
    """Garde-fou : le jour où quelqu'un voudra « améliorer » la réparation
    avec le modèle, ce test tombera — et c'est exactement le but."""

    source = inspect.getsource(module)
    for interdit in ("chat(", "complete(", "ollama", "Llm"):
        assert interdit not in source
