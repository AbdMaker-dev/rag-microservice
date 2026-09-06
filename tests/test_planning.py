"""Le planning du ministère devient des chapitres ordonnés.

Les deux documents réels du Sénégal n'ont pas la même forme : le lycée
écrit « Thème/Chapitre » et « Objectifs spécifiques », le moyen ajoute une
colonne « Parties » et dit « Compétences exigibles ». Un planning qui se
lit par position de colonne casserait sur l'un des deux.
"""

from app.core.planning import parse_planning

TS2 = """
| ANNEE SCOLAIRE : 2025 – 2026 DISCIPLINE : MATHEMATIQUES NIVEAU : TS2 (5h/semaine) |
| Mois d’octobre |
| Périodes | Thème/Chapitre |  | Objectifs spécifiques | Acquis à évaluer |
| Semaine 2 Semaine 3 Semaine 4 | Chapitre 1 : Fonctions numériques (Limites, dérivées). |  | 1. Déterminer l'image d'un intervalle 2. Utiliser le théorème des valeurs intermédiaires | 1. Déterminer l'image d'un intervalle |
| Mois de novembre |
| Semaine 1 | Chapitre 1 (suite et fin) |
| Semaine 2 Semaine 3 | Chapitre 2 : Les nombres complexes |  | 1. Connaître les formes d'un complexe |  |
"""

MOYEN = """
| DISCIPLINE : Mathématiques |
| NIVEAU : Sixième (06 heures/semaine) |
| Périodes | Parties | Leçons / Contenus | Compétences exigibles |
| OCTOBRE |
| Semaine 1 |  |  |  |
| Semaine 2 | Activité Numérique | LES NOMBRES DECIMAUX 1) Entiers naturels 2) Ensemble IN | 1. Identifier l'ensemble des entiers naturels 2. Identifier chiffre, nombre |
"""


def test_le_planning_du_lycee_est_lu_avec_son_en_tete():
    """L'année, la matière, le niveau et l'horaire viennent du document —
    l'admin n'a rien à ressaisir quand ils y sont."""

    planning = parse_planning(TS2)
    assert planning.school_year == "2025-2026"
    assert planning.subject == "MATHEMATIQUES"
    assert planning.grade_label == "TS2"
    assert planning.weekly_hours == 5
    assert planning.warnings == []


def test_les_chapitres_sont_ordonnes_et_dates():
    """L'ordre est ce dont la plateforme a besoin ; le mois et la semaine
    sont le repère qu'on montre."""

    entries = parse_planning(TS2).entries
    assert [e.position for e in entries] == [1, 2]
    assert entries[0].chapter == "Fonctions numériques"
    assert entries[0].details == "Limites, dérivées"
    assert entries[0].periods[0].month == "octobre"
    assert entries[0].periods[0].weeks == [2, 3, 4]


def test_un_chapitre_qui_deborde_ne_cree_pas_un_doublon():
    """« Chapitre 1 (suite et fin) » en novembre est LE MÊME chapitre :
    deux entrées feraient croire à deux cours à créer."""

    entries = parse_planning(TS2).entries
    assert len(entries) == 2
    assert len(entries[0].periods) == 2
    assert entries[0].periods[1].month == "novembre"
    assert entries[0].periods[1].weeks == [1]


def test_les_objectifs_numerotes_deviennent_des_elements():
    entries = parse_planning(TS2).entries
    assert entries[0].objectives == [
        "Déterminer l'image d'un intervalle",
        "Utiliser le théorème des valeurs intermédiaires",
    ]


def test_le_planning_du_moyen_a_une_autre_forme_et_se_lit_aussi():
    """Colonne « Parties » en plus, « Compétences exigibles » au lieu
    d'« Objectifs » : les rôles se lisent dans l'en-tête, pas par position."""

    planning = parse_planning(MOYEN)
    assert planning.subject == "Mathématiques"
    assert planning.grade_label == "Sixième"
    assert planning.weekly_hours == 6
    entry = planning.entries[0]
    assert entry.part == "Activité Numérique"
    assert entry.chapter == "LES NOMBRES DECIMAUX"
    assert entry.details.startswith("1) Entiers naturels")
    assert entry.periods[0].month == "octobre"


def test_une_semaine_sans_lecon_ne_leve_aucune_alerte():
    """Le calendrier officiel contient des semaines vides — rentrée, semaine
    banalisée. Les signaler noierait les vraies anomalies."""

    planning = parse_planning(MOYEN)
    assert "PLANNING_ROW_WITHOUT_CHAPTER" not in planning.warnings
    assert len(planning.entries) == 1


def test_l_annee_manquante_est_dite_pas_devinee():
    """Un planning sans année reste exploitable : c'est l'admin qui
    complète. On ne lui invente pas une année scolaire."""

    planning = parse_planning(MOYEN)
    assert planning.school_year is None
    assert "PLANNING_YEAR_MISSING" in planning.warnings


def test_un_document_qui_n_est_pas_un_planning_le_dit():
    """Un cours déposé par erreur ne doit pas rendre une liste vide sans
    explication — l'admin doit comprendre que le fichier est le mauvais."""

    planning = parse_planning("Le produit scalaire est une forme bilinéaire.")
    assert planning.entries == []
    assert "PLANNING_NOT_RECOGNISED" in planning.warnings


def test_aucun_modele_de_langue_n_est_appele():
    """Garde-fou : un planning se recopie, il ne se paraphrase pas. Une
    semaine inventée décalerait une année scolaire en silence."""

    import inspect

    from app.core import planning as module

    source = inspect.getsource(module)
    for interdit in ("chat(", "complete(", "ollama", "Llm", "llm"):
        assert interdit not in source
