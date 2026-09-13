"""Les garde-fous de la proposition — gardés par des tests, pas par des notes.

Chaque test rejoue une façon de se tromper qu'on a observée, ou qu'on
refuse d'observer une première fois sur un vrai cours.
"""

import pytest

from app.core.proposal import INCERTAIN, porteurs, proposer


class _Llm:
    """Un modèle qui répond ce qu'on lui dit de répondre."""

    def __init__(self, reponse: str) -> None:
        self.reponse = reponse
        self.vu = []

    async def complete(self, system: str, user: str) -> str:
        self.vu.append((system, user))
        return self.reponse


@pytest.mark.asyncio
async def test_repare_la_mise_en_forme_sans_avertissement():
    """Le cas sûr : des espaces remis, aucun symbole touché."""

    llm = _Llm("Son angle θ = arg(a).")
    r = await proposer(llm, "Son angle θ=arg(a) .")

    assert r.proposition == "Son angle θ = arg(a)."
    assert r.changee is True
    assert r.symboles_modifies == []
    assert r.avertissement is None


@pytest.mark.asyncio
async def test_un_symbole_ajoute_est_nomme():
    """L'échec du 29/08 : un « √ » apparu dans une inégalité.

    On ne cache pas la proposition — Alioune a choisi que le modèle
    propose — mais on NOMME ce qu'elle change, pour qu'un professeur
    pressé ne l'accepte pas sans voir.
    """

    llm = _Llm("U+V ≤ √(U²+V²)")
    r = await proposer(llm, "U+V ≤ U+V")

    assert r.proposition is not None
    assert "+√" in r.symboles_modifies
    assert r.avertissement is not None
    assert "CHANGE des symboles" in r.avertissement


@pytest.mark.asyncio
async def test_un_chiffre_change_est_nomme():
    """Une date, un pourcentage, une année : pas que les maths."""

    llm = _Llm("La bataille eut lieu en 1815.")
    r = await proposer(llm, "La bataille eut lieu en 1813.")

    assert "+5" in r.symboles_modifies
    assert "-3" in r.symboles_modifies
    assert r.avertissement is not None


@pytest.mark.asyncio
async def test_le_modele_a_le_droit_de_ne_pas_savoir():
    llm = _Llm(INCERTAIN)
    r = await proposer(llm, "f(x)=∫ ... dx")

    assert r.proposition is None
    assert r.incertaine is True
    assert r.changee is False
    assert "n'a pas su" in r.avertissement


@pytest.mark.asyncio
async def test_une_reecriture_est_ecartee_pas_montree():
    """Refuser, et ne même pas afficher : une reformulation crédible est
    précisément ce qu'on cherche à éviter."""

    llm = _Llm(
        "Une similitude directe est une transformation du plan qui conserve "
        "les angles orientés et multiplie les distances par un rapport "
        "strictement positif appelé rapport de la similitude."
    )
    r = await proposer(llm, "Son angle θ=arg(a) .")

    assert r.proposition is None
    assert r.incertaine is True
    assert "réécrit" in r.avertissement


@pytest.mark.asyncio
async def test_rien_a_corriger_ne_propose_rien():
    llm = _Llm("Son angle θ=arg(a).")
    r = await proposer(llm, "Son angle θ=arg(a).")

    assert r.proposition is None
    assert r.changee is False
    assert r.incertaine is False
    assert r.avertissement is None


@pytest.mark.asyncio
async def test_les_signalements_sont_transmis_comme_indice():
    llm = _Llm("Son angle θ = arg(a).")
    await proposer(llm, "Son angle θ=arg(a) .", signalements=["THIN", "FORMULA"])

    _, user = llm.vu[0]
    assert "THIN" in user and "FORMULA" in user


@pytest.mark.asyncio
async def test_la_consigne_interdit_de_completer():
    """La consigne est un garde-fou : si elle change, ce test le dit."""

    llm = _Llm(INCERTAIN)
    await proposer(llm, "quelque chose")

    system, _ = llm.vu[0]
    assert "JAMAIS" in system
    assert "changer un chiffre" in system
    assert INCERTAIN in system


def test_les_exposants_ne_comptent_pas_pour_des_symboles_differents():
    """NFKC d'abord : « U² » et « U2 » portent le même 2."""

    assert porteurs("U²")["2"] == 1
    assert porteurs("U2")["2"] == 1
