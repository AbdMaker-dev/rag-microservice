"""La correction d'encodage doit réparer sans jamais inventer."""

from __future__ import annotations

from app.core.encoding import apply_plan, build_plan


def _mots(textes, police="ENNLHC+TimesNewRomanPSMT"):
    return [[{"text": t, "fontname": police} for t in textes]]


def test_detecte_le_decalage_mac_roman_et_retablit_les_accents():
    abimes = ["compŽtences", "dŽmontrer", "propriŽtŽs", "gŽomŽtrique", "Žtablir",
              "orthogonalitŽ", "parallŽlisme", "mŽtriques", "bilinŽaires",
              "DŽfinition", "Žlve", "rŽel", "numŽro", "annŽe", "donnŽe",
              "arrtŽ", "carrŽ", "degrŽ", "entiŽre", "unitŽ", "qualitŽ"]

    plan = build_plan(_mots(abimes))

    assert plan.words["compŽtences"] == "compétences"
    assert plan.words["propriŽtŽs"] == "propriétés"
    assert plan.unreadable_fonts == []


def test_laisse_intacte_une_police_deja_correcte():
    sains = ["compétences", "démontrer", "propriétés", "géométrique", "établir",
             "orthogonalité", "parallélisme", "métriques", "bilinéaires",
             "définition", "élève", "réel", "numéro", "année", "donnée",
             "arrêté", "carré", "degré", "entière", "unité", "qualité"]

    plan = build_plan(_mots(sains))

    assert plan.is_empty


def test_la_police_symbol_devient_des_mathematiques_pas_des_devises():
    """« £ » en police Symbol est un « ≤ », pas une livre sterling.

    C'est le cas qui justifie tout le module : traiter Symbol comme du texte
    transformerait une inégalité en montant d'argent.
    """

    mots = [[{"text": "£", "fontname": "Symbol"},
             {"text": "(cid:219)", "fontname": "Symbol"},
             {"text": "l", "fontname": "Symbol"}]]

    plan = build_plan(mots)

    assert plan.words["£"] == "≤"
    assert plan.words["(cid:219)"] == "⇔"
    assert plan.words["l"] == "λ"


def test_un_mot_ambigu_reste_abime_plutot_que_corrige_au_hasard():
    """Le même caractère, deux polices, deux corrections contradictoires.

    « ¥ » vaut « • » dans une police de texte décalée et « ∞ » en Symbol.
    Aucune ne peut être préférée sans deviner, donc on n'y touche pas : un mot
    visiblement abîmé se repère, un mot corrigé à tort ne se repère pas.
    """

    texte = [{"text": t, "fontname": "ENNLHC+TimesNewRomanPSMT"}
             for t in ["compŽtences", "dŽmontrer", "propriŽtŽs", "gŽomŽtrique",
                       "Žtablir", "orthogonalitŽ", "parallŽlisme", "mŽtriques",
                       "bilinŽaires", "DŽfinition", "rŽel", "numŽro", "annŽe",
                       "donnŽe", "carrŽ", "degrŽ", "unitŽ", "qualitŽ",
                       "entiŽre", "arrtŽ", "ŽlŽve", "¥"]]
    symbole = [{"text": "¥", "fontname": "Symbol"}]

    plan = build_plan([texte + symbole])

    assert plan.words["compŽtences"] == "compétences"
    assert "¥" not in plan.words


def test_une_correction_ne_doit_pas_fragmenter_un_mot():
    """« DÈfinition » ne doit pas devenir « D»finition ».

    Les deux fragments « D » et « finition » sont plausibles isolément, ce qui
    trompe une mesure naïve. Le mot doit rester d'un seul tenant.
    """

    plan = build_plan(_mots(["DÈfinition"] * 25, police="Times-Roman"))

    assert plan.words.get("DÈfinition", "DÈfinition") in {"DÈfinition", "Définition"}
    assert plan.words.get("DÈfinition") != "D»finition"


def test_signale_une_police_qu_aucune_table_ne_rend_lisible():
    illisibles = [""] * 30

    plan = build_plan(_mots(illisibles, police="ENNMNA+Wingdings"))

    assert "ENNMNA+Wingdings" in plan.unreadable_fonts


def test_apply_plan_remplace_les_mots_et_preserve_la_mise_en_page():
    plan = build_plan(_mots(["compŽtences"] * 25))
    texte = "| Contenus | compŽtences exigibles |\n\nAutre ligne"

    rendu = apply_plan(texte, plan)

    assert "compétences" in rendu
    assert rendu.count("\n") == texte.count("\n")
    assert rendu.startswith("| Contenus |")


def test_une_lettre_courante_n_est_pas_grecisee_par_une_occurrence_symbol():
    """Le « a » de « il a hérité » ne doit pas devenir « α ».

    Le caractère « a » vaut légitimement alpha en police Symbol. S'il suffisait
    d'une occurrence en Symbol pour décider, toutes les autres — françaises —
    seraient réécrites. Une occurrence inchangée doit donc peser autant qu'une
    occurrence corrigée.
    """

    pages = [[
        {"text": "a", "fontname": "Symbol"},
        {"text": "a", "fontname": "Times-Roman"},
        {"text": "hérité", "fontname": "Times-Roman"},
    ]]

    plan = build_plan(pages)

    assert "a" not in plan.words
    assert apply_plan("le Sénégal a hérité", plan) == "le Sénégal a hérité"
