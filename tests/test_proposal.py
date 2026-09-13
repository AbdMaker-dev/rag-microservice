"""Les garde-fous de la proposition — gardés par des tests, pas par des notes.

Chaque test rejoue une façon de se tromper qu'on a observée, ou qu'on
refuse d'observer une première fois sur un vrai cours.
"""

import pytest

import json

from app.core.proposal import INCERTAIN, discuter, porteurs, proposer


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


# ─────────────────────── le chat de relecture ───────────────────────


class _Chat:
    """Un modèle qui rend le JSON qu'on lui dit de rendre."""

    def __init__(self, charge):
        self.charge = charge
        self.vu = []

    async def chat(self, messages, **kw):
        self.vu.append((messages, kw))
        return json.dumps(self.charge, ensure_ascii=False)

    async def complete(self, system, user):  # non utilisé ici
        raise AssertionError("le chat n'appelle pas complete")


TEXTE = (
    "## p. 1\n"
    "Une similitude directe a pour écriture z' = az + b.\n"
    "Son angle θ=arg(a) .\n"
    "Le rapport vaut |a|.\n"
)


@pytest.mark.asyncio
async def test_une_correction_situee_est_retenue():
    chat = _Chat({
        "reponse": "J'ai remis les espaces autour du signe égal.",
        "corrections": [{"avant": "Son angle θ=arg(a) .", "apres": "Son angle θ = arg(a)."}],
    })
    d = await discuter(chat, TEXTE, "corrige l'espacement de l'angle")

    assert len(d.corrections) == 1
    c = d.corrections[0]
    assert c.position == TEXTE.index("Son angle")
    assert c.symboles_modifies == []
    assert d.refusees == []


@pytest.mark.asyncio
async def test_un_passage_cite_de_memoire_est_refuse():
    """Le modèle désigne un endroit qui n'existe pas : on ne devine pas."""

    chat = _Chat({
        "reponse": "Corrigé.",
        "corrections": [{"avant": "Son angle theta = arg(a).", "apres": "autre chose"}],
    })
    d = await discuter(chat, TEXTE, "corrige l'angle")

    assert d.corrections == []
    assert len(d.refusees) == 1
    assert "ne se trouve pas" in d.refusees[0]["raison"]


@pytest.mark.asyncio
async def test_un_passage_ambigu_est_refuse():
    """Deux occurrences : appliquer au hasard serait pire que refuser."""

    texte = "le centre\nle centre\n"
    chat = _Chat({
        "reponse": "Corrigé.",
        "corrections": [{"avant": "le centre", "apres": "le point Ω"}],
    })
    d = await discuter(chat, texte, "renomme le centre")

    assert d.corrections == []
    assert "2 fois" in d.refusees[0]["raison"]


@pytest.mark.asyncio
async def test_un_symbole_change_dans_le_chat_est_signale():
    chat = _Chat({
        "reponse": "J'ai corrigé la formule.",
        "corrections": [{"avant": "z' = az + b", "apres": "z' = az - b"}],
    })
    d = await discuter(chat, TEXTE, "corrige la formule")

    c = d.corrections[0]
    assert "+-" in c.symboles_modifies or "-+" in c.symboles_modifies
    assert c.avertissement is not None


@pytest.mark.asyncio
async def test_le_texte_entier_est_donne_au_modele():
    """C'est la raison d'être du chat : comprendre « cette partie »."""

    chat = _Chat({"reponse": "ok", "corrections": []})
    await discuter(chat, TEXTE, "que dit le paragraphe sur le rapport ?")

    messages, _ = chat.vu[0]
    assert TEXTE in messages[-1]["content"]
    assert messages[0]["role"] == "system"
    # La consigne doit interdire la réécriture : c'est elle qui borne la
    # sortie, et elle est aussi fragile qu'un commentaire si rien ne la lit.
    assert "Tu ne le" in messages[0]["content"]
    assert "réécris jamais" in messages[0]["content"]
    assert "n'invente aucun contenu" in messages[0]["content"]


@pytest.mark.asyncio
async def test_l_historique_est_rejoue_dans_l_ordre():
    chat = _Chat({"reponse": "ok", "corrections": []})
    await discuter(
        chat, TEXTE, "et maintenant le rapport",
        historique=[
            {"role": "prof", "content": "corrige l'angle"},
            {"role": "ia", "content": "c'est fait"},
        ],
    )

    messages, _ = chat.vu[0]
    assert messages[1] == {"role": "user", "content": "corrige l'angle"}
    assert messages[2] == {"role": "assistant", "content": "c'est fait"}


@pytest.mark.asyncio
async def test_une_reponse_illisible_ne_casse_pas_l_ecran():
    class _Casse(_Chat):
        async def chat(self, messages, **kw):
            return "ceci n'est pas du JSON"

    d = await discuter(_Casse({}), TEXTE, "corrige")
    assert d.corrections == []
    assert "reformulez" in d.reponse
