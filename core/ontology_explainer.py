"""
core/ontology_explainer.py — Fait « parler » l'ontologie OntoHESS.

Jusqu'ici l'ontologie servait uniquement à produire des variables booléennes
(compute_symbolic_states) consommées par les modèles. Ce module l'utilise comme
SOURCE D'EXPLICATION : il lit les règles SWRL réellement présentes dans
ontologies/OntoHESS2.owl, les évalue avec les grandeurs de l'instant analysé, et
produit une chaîne d'inférence lisible.

Honnêteté scientifique — deux limites assumées et affichées à l'utilisateur :

1. Le moteur d'exécution de l'application n'est PAS un raisonneur OWL/SWRL. Les
   règles sont lues depuis l'ontologie et évaluées ici numériquement ; le solveur
   de ems_core applique en parallèle une reproduction à seuils fixes de ces mêmes
   règles (voir l'avertissement de compute_symbolic_states).
2. Seules les règles dont toutes les variables sont connues à l'instant analysé
   peuvent être évaluées ; les autres sont signalées « indéterminable ».
"""

from functools import lru_cache
from pathlib import Path

import ems_core as core
from core.format import SEPARATEURS_PLOTLY, nombre


CHEMIN_OWL = core.ROOT_DIR / "ontologies" / "OntoHESS2.owl"

# Traduction des termes de l'ontologie en langage courant.
VOCABULAIRE = {
    "hasPower": "puissance demandée par la charge",
    "hasPowerBattery": "puissance de la batterie",
    "hasOutputPowerBattery": "puissance fournie par une batterie",
    "hasOutputPowerConverter": "puissance de sortie du convertisseur",
    "hasInputPowerConverter": "puissance d'entrée du convertisseur",
    "hasOperatingModeDriventrain": "mode de fonctionnement de la chaîne de traction",
    "hasVoltageBattery": "tension de la batterie",
    "hasCurrentBattery": "courant de la batterie",
    "hasSocBattery": "état de charge de la batterie",
    "hasMaxDischargeCurrentBattery": "courant maximal de décharge",
    "hasMaxChargeCurrentBattery": "courant maximal de recharge",
    "pEB_max_value": "puissance maximale de décharge de la batterie Énergie",
    "pEB_min_value": "puissance maximale de recharge de la batterie Énergie",
    "socEB_minThreshold": "seuil minimal de SOC de la batterie Énergie",
    "socEB_maxThreshold": "seuil maximal de SOC de la batterie Énergie",
    "socPB_minThreshold": "seuil minimal de SOC de la batterie Puissance",
    "socPB_maxThreshold": "seuil maximal de SOC de la batterie Puissance",
}

CLASSES_FR = {
    "BatteryEB": "Batterie Énergie",
    "BatteryPB": "Batterie Puissance",
    "Converter": "Convertisseur",
    "Load": "Charge (moteur)",
    "HESS": "Système hybride de stockage",
    "Overload": "Surcharge",
    "OverloadCondition": "Condition de surcharge",
    "NormalOperation": "Fonctionnement normal",
    "SOCState": "État de charge",
    "PowerState": "État de puissance",
}

COMPARATEURS = {
    "greaterThan": ">",
    "lessThan": "<",
    "greaterThanOrEqual": "≥",
    "lessThanOrEqual": "≤",
    "equal": "=",
    "notEqual": "≠",
}

CALCULS = {"multiply": "×", "subtract": "−", "add": "+", "divide": "÷"}

# Correspondance entre les variables SWRL et les grandeurs de la simulation.
# Établie à partir des règles réellement présentes dans OntoHESS2.owl.
LIAISON_VARIABLES = {
    "P": "p_dem",
    "soc": "soc_eb",
    "smin": "soc_eb_min",
    "pmax": "p_eb_max",
    "pmin": "p_eb_min",
    "veb": "v_eb",
    "vpb": "v_pb",
    "ieb": "i_eb",
}


def _fr(nom):
    return VOCABULAIRE.get(nom, CLASSES_FR.get(nom, nom))


@lru_cache(maxsize=1)
def charger_regles():
    """Lit les règles SWRL de l'ontologie. Retourne une liste de dictionnaires
    {id, classes, conditions, calculs, conclusions}. Liste vide si rdflib est
    absent ou le fichier introuvable (l'application reste fonctionnelle)."""
    try:
        from rdflib import Graph, Namespace, RDF
    except ImportError:
        return []

    if not CHEMIN_OWL.exists():
        return []

    graphe = Graph()
    try:
        graphe.parse(str(CHEMIN_OWL))
    except Exception:  # noqa: BLE001
        return []

    SWRL = Namespace("http://www.w3.org/2003/11/swrl#")

    def suite(noeud):
        elements = []
        while noeud and noeud != RDF.nil:
            item = graphe.value(noeud, RDF.first)
            if item is not None:
                elements.append(item)
            noeud = graphe.value(noeud, RDF.rest)
        return elements

    def lire_atome(atome):
        type_atome = str(graphe.value(atome, RDF.type)).split("#")[-1]
        if type_atome == "ClassAtom":
            return ("classe", str(graphe.value(atome, SWRL.classPredicate)).split("#")[-1])
        if type_atome in ("DatavaluedPropertyAtom", "IndividualPropertyAtom"):
            arguments = [
                str(graphe.value(atome, a)).split("#")[-1]
                for a in (SWRL.argument1, SWRL.argument2)
                if graphe.value(atome, a) is not None
            ]
            return ("propriete", str(graphe.value(atome, SWRL.propertyPredicate)).split("#")[-1], arguments)
        if type_atome == "BuiltinAtom":
            arguments = [str(a).split("#")[-1] for a in suite(graphe.value(atome, SWRL.arguments))]
            return ("builtin", str(graphe.value(atome, SWRL.builtin)).split("#")[-1], arguments)
        return ("autre", type_atome)

    brutes = []
    for implication in graphe.subjects(RDF.type, SWRL.Imp):
        corps = [lire_atome(a) for a in suite(graphe.value(implication, SWRL.body))]
        tete = [lire_atome(a) for a in suite(graphe.value(implication, SWRL.head))]
        brutes.append(
            {
                "classes": [a[1] for a in corps if a[0] == "classe"],
                "conditions": [a for a in corps if a[0] == "builtin" and a[1] in COMPARATEURS],
                "calculs": [a for a in corps if a[0] == "builtin" and a[1] in CALCULS],
                "conclusions": [a[1] for a in tete if a[0] == "propriete"],
                # Détail complet, pour parcourir les règles : grandeurs lues dans
                # le corps et affectations de la tête, avec leurs arguments.
                "lectures": [(a[1], a[2]) for a in corps if a[0] == "propriete"],
                "affectations": [(a[1], a[2]) for a in tete if a[0] == "propriete"],
            }
        )

    # Les identifiants doivent être STABLES d'une exécution à l'autre : rdflib
    # attribue des noms de nœuds anonymes variables, donc on numérote selon une
    # signature du contenu de la règle, et non selon l'ordre de lecture.
    def signature(regle):
        return (
            tuple(sorted(regle["conclusions"])),
            tuple(sorted(regle["classes"])),
            tuple(sorted((c[1], tuple(c[2])) for c in regle["conditions"])),
            tuple(sorted((c[1], tuple(c[2])) for c in regle["calculs"])),
        )

    brutes.sort(key=lambda r: repr(signature(r)))
    for indice, regle in enumerate(brutes, start=1):
        regle["id"] = f"R{indice}"
    return brutes


@lru_cache(maxsize=1)
def classes_ontologie():
    """Noms des classes OWL réellement déclarées dans l'ontologie."""
    try:
        from rdflib import Graph, RDF, OWL
    except ImportError:
        return frozenset()
    if not CHEMIN_OWL.exists():
        return frozenset()
    graphe = Graph()
    try:
        graphe.parse(str(CHEMIN_OWL))
    except Exception:  # noqa: BLE001
        return frozenset()
    return frozenset(str(c).split("#")[-1] for c in graphe.subjects(RDF.type, OWL.Class))


def diagnostic_configuration(soc_eb0, soc_pb0, nb_strategies, objectif):
    """Pré-diagnostic de la configuration AVANT simulation, à partir des classes
    réellement déclarées dans l'ontologie.

    L'ontologie n'intervient donc plus seulement pour expliquer une décision,
    mais aussi pour valider la cohérence de l'expérience à réaliser.
    """
    presentes = classes_ontologie()

    attendus = [
        ("HESS", "Architecture reconnue : système hybride de stockage"),
        ("BatteryEB", "Source d'énergie : batterie Énergie"),
        ("BatteryPB", "Source d'énergie : batterie Puissance"),
        ("Converter", "Organe de répartition : convertisseur"),
        ("ManagementStrategy", "Objet d'étude : stratégie de gestion d'énergie"),
    ]
    contexte = [
        {"concept": nom, "libelle": libelle, "reconnu": nom in presentes}
        for nom, libelle in attendus
    ]

    contraintes_owl = [
        ("SOCCondition", "Préservation des états de charge"),
        ("OverloadCondition", "Protection contre la surcharge"),
        ("PowerThreshold", "Respect des seuils de puissance"),
    ]
    contraintes = [
        {"concept": nom, "libelle": libelle, "reconnu": nom in presentes}
        for nom, libelle in contraintes_owl
    ]

    alertes = []
    if soc_eb0 < 0.35:
        alertes.append(
            f"L'ontologie associe un SOC initial de {nombre(soc_eb0 * 100, 0)} % à la classe "
            "`SOCCondition` : la batterie Énergie atteindra rapidement sa limite de "
            "fonctionnement et sera protégée par le filtre."
        )
    if soc_pb0 < 0.35:
        alertes.append(
            f"SOC initial de la batterie Puissance à {nombre(soc_pb0 * 100, 0)} % : sa capacité "
            "à absorber les pics de demande sera fortement réduite."
        )

    if objectif == "Comparer plusieurs stratégies" and nb_strategies < 3:
        conseils = [
            "Pour une comparaison pertinente, sélectionnez au moins une stratégie d'IA "
            "en plus des deux références."
        ]
    elif objectif == "Générer des résultats pour une publication":
        conseils = [
            "Pour des chiffres publiables, privilégiez la précision « Validation » et "
            "conservez l'ensemble des stratégies."
        ]
    elif objectif == "Étudier une stratégie en détail":
        conseils = [
            "Une seule stratégie d'IA suffit : les références restent incluses comme "
            "point de comparaison."
        ]
    else:
        conseils = [
            "Les deux stratégies de référence sont toujours simulées : elles servent "
            "de point de comparaison."
        ]

    coherent = all(c["reconnu"] for c in contexte)
    return {
        "contexte": contexte,
        "contraintes": contraintes,
        "objectif": objectif,
        "alertes": alertes,
        "conseils": conseils,
        "conclusion": (
            "Configuration cohérente avec une étude des stratégies de gestion d'énergie "
            "d'un système hybride de stockage."
            if coherent
            else "Certains concepts attendus sont absents de l'ontologie : le diagnostic "
            "est partiel."
        ),
    }


@lru_cache(maxsize=1)
def vocabulaire_ontologie():
    """Relations (ObjectProperty), attributs (DatatypeProperty) et individus
    réellement déclarés dans l'ontologie."""
    try:
        from rdflib import Graph, RDF, OWL
    except ImportError:
        return (), (), ()
    if not CHEMIN_OWL.exists():
        return (), (), ()
    graphe = Graph()
    try:
        graphe.parse(str(CHEMIN_OWL))
    except Exception:  # noqa: BLE001
        return (), (), ()

    def noms(type_owl):
        return tuple(sorted({str(s).split("#")[-1] for s in graphe.subjects(RDF.type, type_owl)}))

    return noms(OWL.ObjectProperty), noms(OWL.DatatypeProperty), noms(OWL.NamedIndividual)


# États de fonctionnement réellement déclarés comme individus dans l'ontologie.
# Les identifiants OWL gardent le mot « Overload », mais l'état signifie
# seulement que la demande dépasse la limite de la batterie Énergie seule
# (règles R9/R10/R12, seuils pEB_max_value / pEB_min_value) : le HESS, lui,
# n'est pas en surcharge. Les libellés affichés le disent tel quel.
ETATS_ONTOLOGIE = {
    "state_Normal": "Dans les limites de la batterie Énergie",
    "state_Overload_High": "Traction au-delà de la limite de la batterie Énergie",
    "state_Overload_Low": "Récupération au-delà de la limite de la batterie Énergie",
}

# Versions courtes, pour les nœuds de graphe.
ETATS_ONTOLOGIE_COURTS = {
    "state_Normal": "Dans les limites EB",
    "state_Overload_High": "Traction > limite EB",
    "state_Overload_Low": "Récup. > limite EB",
}

# Libellés des états symboliques de compute_symbolic_states (entrées des
# modèles neurosymboliques). Seuls libellés utilisés par les pages.
LIBELLES_SYMBOLIQUES = {
    "high_power_demand": "Forte demande de puissance",
    "regenerative_braking": "Freinage régénératif",
    "zero_power_demand": "Demande quasi nulle",
    "converter_risk": "Convertisseur proche de sa limite",
    "EB_available": "Batterie Énergie disponible",
    "PB_available": "Batterie Puissance disponible",
    "EB_low_SOC": "SOC batterie Énergie faible",
    "PB_low_SOC": "SOC batterie Puissance faible",
}


def etat_fonctionnement(p_dem):
    """Individu `state_*` de l'ontologie correspondant à la puissance demandée,
    avec les seuils que comparent les règles SWRL (pEB_max_value, pEB_min_value)."""
    if p_dem > core.P_EB_MAX_W:
        return "state_Overload_High"
    if p_dem < core.P_EB_MIN_W:
        return "state_Overload_Low"
    return "state_Normal"


def etat_instant(p_dem, soc_eb, soc_pb, p_eb=None):
    """Source UNIQUE de ce que les pages affichent comme « état » à un instant :
    l'état de fonctionnement inféré par l'ontologie, et les états symboliques
    fournis aux modèles neurosymboliques.

    p_eb : puissance réellement fournie par la batterie Énergie à cet instant
    (P_EB de la trajectoire). Toutes les pages passent cette même grandeur.
    """
    cle = etat_fonctionnement(p_dem)
    return {
        "fonctionnement": cle,
        "libelle": ETATS_ONTOLOGIE[cle],
        "symboliques": core.compute_symbolic_states(p_dem, soc_eb, soc_pb, p_eb=p_eb),
    }


def interpretation_ontologique(p_dem, soc_eb, soc_pb):
    """Chaîne « mesures → concepts de l'ontologie → état inféré ».

    Les mesures sont rattachées aux propriétés réellement déclarées dans
    OntoHESS2.owl, et l'état inféré est l'un des individus `state_*` de
    l'ontologie, déterminé par les seuils que les règles SWRL comparent.
    """
    relations, attributs, individus = vocabulaire_ontologie()

    observations = [
        {
            "mesure": f"SOC de la batterie Énergie = {nombre(soc_eb * 100, 0)} %",
            "propriete": "hasSocBattery",
            "individu": "batteryE1",
        },
        {
            "mesure": f"SOC de la batterie Puissance = {nombre(soc_pb * 100, 0)} %",
            "propriete": "hasSocBattery",
            "individu": "batteryP1",
        },
        {
            "mesure": f"Puissance demandée = {nombre(p_dem / 1000, 1)} kW",
            "propriete": "hasPower",
            "individu": "load1",
        },
    ]

    # L'état est déterminé par les mêmes seuils que ceux comparés dans les règles.
    etat = etat_fonctionnement(p_dem)
    if etat == "state_Overload_High":
        justification = (
            f"la puissance demandée ({nombre(p_dem / 1000, 1)} kW) dépasse `pEB_max_value` "
            f"({nombre(core.P_EB_MAX_W / 1000, 1)} kW)"
        )
    elif etat == "state_Overload_Low":
        justification = (
            f"la puissance récupérée ({nombre(p_dem / 1000, 1)} kW) dépasse la capacité de "
            f"recharge `pEB_min_value` ({nombre(core.P_EB_MIN_W / 1000, 1)} kW)"
        )
    else:
        justification = (
            f"la puissance demandée ({nombre(p_dem / 1000, 1)} kW) reste dans les limites "
            f"`pEB_min_value` … `pEB_max_value`"
        )

    eb_sous_seuil = soc_eb <= core.SOC_EB_MIN
    deductions = [
        {
            "concept": etat,
            "libelle": ETATS_ONTOLOGIE[etat],
            "justification": justification,
            "present": etat in individus,
        },
        {
            "concept": "SOCCondition",
            "libelle": (
                "SOC de la batterie Énergie sous son seuil"
                if eb_sous_seuil
                else "SOC de la batterie Énergie au-dessus de son seuil"
            ),
            "justification": (
                f"`hasSocBattery` = {nombre(soc_eb * 100, 0)} % comparé à `socEB_minThreshold` "
                f"= {nombre(core.SOC_EB_MIN * 100, 0)} %"
            ),
            "present": "SOCCondition" in classes_ontologie(),
        },
    ]

    # Relations effectivement mobilisées PAR CE raisonnement : elles dépendent de
    # l'état inféré (leadsToOverload n'a de sens qu'en situation de surcharge).
    candidates = ["hasThreshold", "hasSOCState", "dependsOnSOC"]
    if etat != "state_Normal":
        candidates += ["leadsToOverload", "triggersState"]
    mobilisees = [r for r in candidates if r in relations]

    return {
        "observations": observations,
        "deductions": deductions,
        "etat": etat,
        "relations": mobilisees,
        "attributs": [
            a for a in ("hasSocBattery", "hasPower", "socEB_minThreshold", "pEB_max_value", "pEB_min_value", "hasAlpha")
            if a in attributs
        ],
        "individus": [i for i in ("hess1", "batteryE1", "batteryP1", "converter1", "load1") if i in individus],
    }


HYPOTHESES = [
    "Tensions de pack considérées constantes (valeurs nominales).",
    "Convertisseur représenté par un modèle de puissance simplifié avec ses limites.",
    "Aucun modèle thermique ni de vieillissement des batteries.",
    "Cycle de conduite connu à l'avance (pas de prédiction en ligne).",
    "Conditions environnementales (météo, pente) non prises en compte.",
]


def contexte_numerique(p_dem, soc_eb, soc_pb):
    """Valeurs de l'instant analysé, sous les noms utilisés par les règles."""
    return {
        "p_dem": float(p_dem),
        "soc_eb": float(soc_eb),
        "soc_pb": float(soc_pb),
        "soc_eb_min": float(core.SOC_EB_MIN),
        "p_eb_max": float(core.P_EB_MAX_W),
        "p_eb_min": float(core.P_EB_MIN_W),
        "v_eb": float(core.V_EB_PACK_NOM),
        "v_pb": float(core.V_PB_PACK_NOM),
    }


def _resoudre(argument, contexte):
    """Résout un argument SWRL : littéral numérique ou variable connue."""
    try:
        return float(argument)
    except (TypeError, ValueError):
        cle = LIAISON_VARIABLES.get(argument)
        return contexte.get(cle) if cle else None


def _format_valeur(nom_variable, valeur):
    if valeur is None:
        return "?"
    if nom_variable in ("soc", "smin"):
        return f"{nombre(valeur * 100, 0)} %"
    if abs(valeur) >= 1000:
        return f"{nombre(valeur / 1000, 1)} kW"
    return f"{nombre(valeur, 2)}"


# Lecture en clair des règles SWRL de mode de fonctionnement et de répartition.
# Elles sont identifiées par leurs conditions, pas par leur numéro : le numéro
# dépend de l'ordre de lecture du fichier, les conditions non.
def _cle_conditions(conditions):
    return frozenset((op, tuple(args)) for _, op, args in conditions)


LECTURE_REGLES = {
    frozenset({("greaterThan", ("P", "0")), ("lessThanOrEqual", ("P", "pmax"))}):
        ("mode", "la traction reste dans les limites de la batterie Énergie"),
    frozenset({("greaterThan", ("P", "pmax"))}):
        ("mode", "la traction dépasse la limite de la batterie Énergie : la PB doit assister"),
    frozenset({("lessThanOrEqual", ("soc", "smin"))}):
        ("mode", "la batterie Énergie est à son SOC minimal : elle doit être protégée"),
    frozenset({("lessThan", ("P", "0"))}):
        ("mode", "le véhicule freine : de l'énergie est récupérée"),
    frozenset({("lessThanOrEqual", ("soc", "smin")), ("greaterThan", ("P", "0"))}):
        ("repartition", "la PB fournit toute la puissance, l'EB est protégée"),
    frozenset({("greaterThan", ("soc", "smin")), ("greaterThan", ("P", "pmax"))}):
        ("repartition", "l'EB fournit sa puissance maximale, la PB complète"),
    frozenset({("lessThan", ("P", "pmin"))}):
        ("repartition", "l'EB absorbe jusqu'à sa limite, la PB absorbe le surplus"),
    frozenset({("greaterThan", ("soc", "smin")), ("greaterThan", ("P", "0")), ("lessThanOrEqual", ("P", "pmax"))}):
        ("repartition", "l'EB fournit toute la puissance, la PB reste au repos"),
    frozenset({("lessThan", ("P", "0")), ("greaterThanOrEqual", ("P", "pmin"))}):
        ("repartition", "l'EB absorbe toute l'énergie récupérée"),
}

# Conditions de ces règles en français courant, seuils compris.
CONDITIONS_EN_FRANCAIS = {
    ("greaterThan", ("P", "0")): "le véhicule est en traction",
    ("lessThan", ("P", "0")): "le véhicule freine",
    ("greaterThan", ("P", "pmax")):
        f"la demande dépasse la limite de décharge de la batterie Énergie ({nombre(core.P_EB_MAX_W / 1000, 1)} kW)",
    ("lessThanOrEqual", ("P", "pmax")):
        f"la demande ne dépasse pas la limite de décharge de la batterie Énergie ({nombre(core.P_EB_MAX_W / 1000, 1)} kW)",
    ("lessThan", ("P", "pmin")):
        f"la puissance récupérée dépasse ce que la batterie Énergie peut absorber ({nombre(-core.P_EB_MIN_W / 1000, 1)} kW)",
    ("greaterThanOrEqual", ("P", "pmin")):
        f"la batterie Énergie peut absorber toute la puissance récupérée (jusqu'à {nombre(-core.P_EB_MIN_W / 1000, 1)} kW)",
    ("lessThanOrEqual", ("soc", "smin")):
        f"le SOC de la batterie Énergie a atteint son minimum ({nombre(core.SOC_EB_MIN * 100, 0)} %)",
    ("greaterThan", ("soc", "smin")):
        f"le SOC de la batterie Énergie est au-dessus de son minimum ({nombre(core.SOC_EB_MIN * 100, 0)} %)",
}


def regle_en_phrase(regle):
    """« Si …, alors … » en français courant pour une règle de mode ou de
    répartition ; conditions en notation brute pour les autres."""
    morceaux = [
        CONDITIONS_EN_FRANCAIS.get((op, tuple(args)), f"{args[0]} {COMPARATEURS[op]} {args[1]}")
        for _, op, args in regle["conditions"]
        if len(args) >= 2
    ]
    type_regle, lecture = lire_regle(regle)
    if not morceaux:
        return lecture[0].upper() + lecture[1:] + "."
    si = ", et ".join(morceaux) if len(morceaux) > 2 else " et que ".join(morceaux)
    mode = next((args[1] for p, args in regle.get("affectations", []) if p == "hasOperatingModeDriventrain"), None)
    if type_regle == "mode" and mode in SENS_DES_MODES:
        return f"Si {si}, alors le mode de fonctionnement est « {mode} » : {SENS_DES_MODES[mode]}."
    return f"Si {si}, alors {lecture}."


# Sens des modes de fonctionnement affectés par les règles R9 à R12.
SENS_DES_MODES = {
    "Normal": "l'EB suffit à fournir la demande",
    "Surcharge": "la PB doit assister l'EB (le HESS lui-même n'est pas en surcharge)",
    "ProtectionEB": "l'EB doit être protégée",
    "Recuperation": "de l'énergie est récupérée",
}


# Règles floues : libellé court, et concepts de l'ontologie qu'elles mobilisent
# (classes d'OntoHESS entre parenthèses).
REGLES_FLOUES = {
    "R1_PB_low_traction": ("R1 · PB basse en traction", [("SOC de la PB bas", "SOCState"), ("traction", "PowerState")]),
    "R2_EB_low_PB_available": ("R2 · EB basse, PB disponible", [("SOC de l'EB bas", "SOCState"), ("PB disponible", "SOCState")]),
    "R3_strong_traction": ("R3 · forte traction", [("forte demande de traction", "PowerState")]),
    "R4_zero_demand": ("R4 · demande nulle", [("demande quasi nulle", "PowerState")]),
    "R5_regenerative_braking": ("R5 · freinage", [("freinage régénératif", "PowerState")]),
    "R5b_PB_high_recharge": ("R5b · PB pleine en recharge", [("SOC de la PB élevé", "SOCState"), ("récupération", "PowerState")]),
    "R7_two_low_SOC": ("R7 · deux SOC bas", [("SOC des deux batteries bas", "SOCState")]),
}

def calculs_en_clair(regle):
    """Calculs d'une règle (builtins SWRL) : « z = x − y »."""
    sortie = []
    for _, op, args in regle["calculs"]:
        if len(args) >= 3:
            sortie.append(f"{args[0]} = {f' {CALCULS[op]} '.join(args[1:])}")
    return sortie


def regles_floues():
    """Les sept règles floues du moteur (ems_core), décrites à partir de ses
    constantes : conditions, conclusion (part de la PB), sens et concepts de
    l'ontologie mobilisés. Le moteur calcule alpha comme la moyenne des
    conclusions pondérée par l'activation de chaque règle."""
    conditions = {
        "R1_PB_low_traction": "SOC de la PB bas ET traction",
        "R2_EB_low_PB_available": "SOC de l'EB bas ET PB disponible (SOC moyen ou haut) ET traction",
        "R3_strong_traction": "forte traction ET PB disponible (SOC moyen ou haut)",
        "R4_zero_demand": "demande nulle ET accélération stable",
        "R5_regenerative_braking": "récupération ET PB rechargeable (SOC bas ou moyen)",
        "R5b_PB_high_recharge": "récupération ET SOC de la PB haut",
        "R7_two_low_SOC": "SOC de l'EB bas ET SOC de la PB bas",
    }
    return [
        {
            "cle": cle,
            "libelle": REGLES_FLOUES.get(cle, (cle, []))[0],
            "si": conditions.get(cle, "—"),
            "alpha": float(core.FUZZY_RULE_CONSEQUENTS[i]),
            "sens": core.RULE_LABELS_FR.get(cle, ""),
            "concepts": REGLES_FLOUES.get(cle, (cle, []))[1],
        }
        for i, cle in enumerate(core.FUZZY_RULE_NAMES)
    ]


def termes_flous():
    """Définition chiffrée des termes employés par les règles floues."""
    kw = lambda w: f"{nombre(w / 1000, 1)} kW"  # noqa: E731
    return [
        ("SOC de l'EB bas", f"plein en dessous de {nombre(core.SOC_LOW_FULL_EB * 100, 0)} %, nul au-dessus de {nombre(core.SOC_LOW_THRESHOLD * 100, 0)} %"),
        ("SOC de la PB bas", f"plein en dessous de {nombre(core.SOC_LOW_FULL_PB * 100, 0)} %, nul au-dessus de {nombre(core.SOC_LOW_THRESHOLD * 100, 0)} %"),
        ("SOC moyen", "trapèze 25 – 35 – 65 – 75 %"),
        ("SOC haut", "nul en dessous de 70 %, plein au-dessus de 80 %"),
        ("Traction", f"de {nombre(core.EPS_POWER_W, 0)} W jusqu'à la limite de l'EB ({kw(core.P_EB_MAX_W)}) et au-delà"),
        ("Forte traction", f"nulle en dessous de {kw(0.6 * core.P_EB_MAX_W)}, pleine au-delà de {kw(core.P_EB_MAX_W)}"),
        ("Demande nulle", f"|P| ≤ {nombre(core.EPS_POWER_W, 0)} W (s'annule à {nombre(2 * core.EPS_POWER_W, 0)} W)"),
        ("Récupération", f"puissance négative ; forte en dessous de {kw(core.P_EB_MIN_W)}"),
        ("Accélération stable", "|a| ≤ 0,1 m/s² (s'annule à 0,4 m/s²)"),
    ]


def contributions_floues(forces):
    """Contribution exacte de chaque règle floue à alpha : wᵢ·cᵢ / Σw.
    Leur somme vaut la sortie floue ; None si aucune règle n'est activée
    (le moteur applique alors sa répartition par défaut)."""
    import numpy as np

    forces = np.asarray(forces, dtype=float)
    somme = forces.sum()
    if somme <= 1e-9:
        return None
    return forces * core.FUZZY_RULE_CONSEQUENTS / somme


# Variables des règles SWRL, en clair.
GLOSSAIRE_VARIABLES = {
    "P": "puissance demandée",
    "pmax": "puissance maximale de décharge de l'EB",
    "pmin": "puissance maximale de recharge de l'EB",
    "soc": "SOC de l'EB",
    "smin": "SOC minimal de l'EB",
    "imax": "courant maximal de l'EB",
    "imin": "courant minimal de l'EB",
    "ibp_nom": "courant nominal de la PB",
    "ibe_nom2": "courant nominal de l'EB",
    "icharge": "courant de charge",
    "veb": "tension de l'EB",
    "vpb": "tension de la PB",
    "ieb": "courant de l'EB",
}


def lire_regle(regle):
    """(type, lecture) d'une règle : « mode », « repartition » ou « calcul »."""
    if not regle["conditions"]:
        return "calcul", "calcule une grandeur (" + ", ".join(dict.fromkeys(_fr(c) for c in regle["conclusions"])) + ")"
    cle = _cle_conditions(regle["conditions"])
    if cle in LECTURE_REGLES:
        return LECTURE_REGLES[cle]
    return "courant", "fixe le courant d'une batterie selon ses limites"


def conditions_en_clair(regle):
    """Conditions d'une règle, écrites avec les noms des variables."""
    return [
        f"{args[0]} {COMPARATEURS[op]} {args[1]}"
        for _, op, args in regle["conditions"]
        if len(args) >= 2
    ]


def alpha_ontologie_vect(p_dem, soc_eb):
    """Répartition (part de la PB) prescrite par les règles de répartition
    d'OntoHESS, pour des tableaux de puissance et de SOC. Mêmes conditions que
    les règles lues dans le fichier OWL (seuils pEB_max, pEB_min, socEB_min).
    NaN si la demande est quasi nulle : aucune règle de répartition ne s'applique."""
    import numpy as np

    p = np.asarray(p_dem, dtype=float)
    soc = np.asarray(soc_eb, dtype=float)
    pmax, pmin, smin = core.P_EB_MAX_W, core.P_EB_MIN_W, core.SOC_EB_MIN
    alpha = np.full(p.shape, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        traction = p > core.EPS_POWER_W
        alpha[traction & (soc <= smin)] = 1.0
        alpha[traction & (soc > smin) & (p <= pmax)] = 0.0
        haut = traction & (soc > smin) & (p > pmax)
        alpha[haut] = (p[haut] - pmax) / p[haut]
        recup = p < -core.EPS_POWER_W
        alpha[recup & (p >= pmin)] = 0.0
        fort = recup & (p < pmin)
        alpha[fort] = (p[fort] - pmin) / p[fort]
    return alpha


def repartition_ontologie(p_dem, soc_eb, soc_pb):
    """Règle de répartition d'OntoHESS activée à cet instant et la part de la
    PB qu'elle prescrit. None si aucune règle de répartition ne s'applique."""
    activees, _, _ = evaluer_regles(p_dem, soc_eb, soc_pb)
    regle = next((r for r in activees if r["type"] == "repartition"), None)
    if regle is None or abs(p_dem) <= core.EPS_POWER_W:
        return None
    return {
        "regle": regle,
        "alpha": float(alpha_ontologie_vect([p_dem], [soc_eb])[0]),
    }


def hierarchie_classes():
    """{classe racine: [sous-classes]} des classes nommées de l'ontologie."""
    graphe = _graphe()
    if graphe is None:
        return {}
    from rdflib import OWL, RDF, RDFS

    classes = {str(c).split("#")[-1] for c in graphe.subjects(RDF.type, OWL.Class)}
    parents = {}
    for c in graphe.subjects(RDF.type, OWL.Class):
        nom = str(c).split("#")[-1]
        parents[nom] = [
            str(p).split("#")[-1] for p in graphe.objects(c, RDFS.subClassOf)
            if str(p).split("#")[-1] in classes
        ]
    racines = sorted(c for c, ps in parents.items() if not ps)
    arbre = {r: [] for r in racines}
    for c, ps in sorted(parents.items()):
        for r in racines:
            if r in ps:
                arbre[r].append(c)
    return arbre


def relations_ontologie():
    """[(relation, domaines, portées)] avec les seules classes nommées."""
    graphe = _graphe()
    if graphe is None:
        return []
    from rdflib import OWL, RDF, RDFS

    classes = {str(c).split("#")[-1] for c in graphe.subjects(RDF.type, OWL.Class)}

    def nommees(noeuds):
        return sorted({str(n).split("#")[-1] for n in noeuds} & classes)

    return sorted(
        (
            str(p).split("#")[-1],
            nommees(graphe.objects(p, RDFS.domain)),
            nommees(graphe.objects(p, RDFS.range)),
        )
        for p in graphe.subjects(RDF.type, OWL.ObjectProperty)
    )


def individus_ontologie():
    """[(individu, classes nommées, [(propriété, valeur)])]."""
    graphe = _graphe()
    if graphe is None:
        return []
    from rdflib import OWL, RDF, Literal

    classes = {str(c).split("#")[-1] for c in graphe.subjects(RDF.type, OWL.Class)}
    sortie = []
    for i in graphe.subjects(RDF.type, OWL.NamedIndividual):
        types = sorted({str(t).split("#")[-1] for t in graphe.objects(i, RDF.type)} & classes)
        props = [
            (str(p).split("#")[-1], o.toPython() if isinstance(o, Literal) else str(o).split("#")[-1])
            for p, o in graphe.predicate_objects(i)
            if p != RDF.type
        ]
        sortie.append((str(i).split("#")[-1], types, sorted(props, key=lambda kv: kv[0])))
    return sorted(sortie)


@lru_cache(maxsize=1)
def _graphe():
    try:
        from rdflib import Graph
    except ImportError:
        return None
    if not CHEMIN_OWL.exists():
        return None
    graphe = Graph()
    try:
        graphe.parse(str(CHEMIN_OWL))
    except Exception:  # noqa: BLE001
        return None
    return graphe


def evaluer_regles(p_dem, soc_eb, soc_pb):
    """Évalue chaque règle de l'ontologie avec les valeurs de l'instant.

    Retourne (activees, non_activees, indeterminables). Chaque élément porte le
    détail des conditions, avec la valeur réelle confrontée au seuil — c'est ce
    qui permet d'expliquer aussi pourquoi une règle n'a PAS été activée.
    """
    contexte = contexte_numerique(p_dem, soc_eb, soc_pb)
    activees, non_activees, indeterminables = [], [], []

    for regle in charger_regles():
        if not regle["conditions"]:
            continue  # règle purement calculatoire (P = V × I, etc.)

        details, satisfaite, connue = [], True, True
        for _, operateur, arguments in regle["conditions"]:
            if len(arguments) < 2:
                continue
            gauche, droite = arguments[0], arguments[1]
            vg, vd = _resoudre(gauche, contexte), _resoudre(droite, contexte)
            if vg is None or vd is None:
                connue = False
                details.append({"texte": f"{gauche} {COMPARATEURS[operateur]} {droite}", "ok": None})
                continue
            ok = {
                "greaterThan": vg > vd,
                "lessThan": vg < vd,
                "greaterThanOrEqual": vg >= vd,
                "lessThanOrEqual": vg <= vd,
                "equal": abs(vg - vd) < 1e-9,
                "notEqual": abs(vg - vd) >= 1e-9,
            }[operateur]
            satisfaite = satisfaite and ok
            details.append(
                {
                    "texte": (
                        f"{_format_valeur(gauche, vg)} {COMPARATEURS[operateur]} "
                        f"{_format_valeur(droite, vd)}"
                    ),
                    "ok": ok,
                }
            )

        type_regle, lecture = lire_regle(regle)
        entree = {
            "id": regle["id"],
            "type": type_regle,
            "lecture": lecture,
            "classes": [CLASSES_FR.get(c, c) for c in regle["classes"]],
            # Une règle qui conclut sur les deux batteries répète la même propriété.
            "conclusions": list(dict.fromkeys(_fr(c) for c in regle["conclusions"])),
            "details": details,
        }
        if not connue:
            indeterminables.append(entree)
        elif satisfaite:
            activees.append(entree)
        else:
            non_activees.append(entree)

    return activees, non_activees, indeterminables


def concepts_actifs(p_dem, soc_eb, soc_pb, p_eb=None):
    """Concepts reconnus à cet instant, avec la mesure qui les justifie.

    Le premier est l'état de fonctionnement inféré par l'ontologie (le même que
    sur les autres pages) ; les suivants sont les états symboliques fournis aux
    modèles, rattachés aux classes présentes dans OntoHESS2.owl."""
    etat = etat_instant(p_dem, soc_eb, soc_pb, p_eb=p_eb)
    etats = etat["symboliques"]
    p_kw = float(p_dem) / 1000.0
    au_dela = etat["fonctionnement"] != "state_Normal"

    return [
        {
            "concept": etat["fonctionnement"],
            "libelle": etat["libelle"],
            "actif": au_dela,
            "mesure": (
                f"puissance demandée {nombre(p_kw, 1, signe=True)} kW, limites de la batterie Énergie "
                f"{nombre(core.P_EB_MIN_W / 1000, 1)} … {nombre(core.P_EB_MAX_W / 1000, 1)} kW"
            ),
            "consequence": (
                "la batterie Puissance doit compléter la batterie Énergie"
                if au_dela
                else "la batterie Énergie peut assurer seule la demande"
            ),
        },
        {
            "concept": "PowerState (forte demande)",
            "libelle": LIBELLES_SYMBOLIQUES["high_power_demand"],
            "actif": bool(etats["high_power_demand"]),
            "mesure": f"puissance demandée {nombre(abs(p_kw), 1)} kW, seuil {nombre(core.HIGH_POWER_THRESHOLD_W / 1000, 0)} kW",
            "consequence": "la batterie Puissance est davantage sollicitée",
        },
        {
            "concept": "SOCCondition",
            "libelle": LIBELLES_SYMBOLIQUES["EB_available"],
            "actif": bool(etats["EB_available"]),
            "mesure": f"SOC de l'EB {nombre(soc_eb * 100, 0)} %, seuil minimal {nombre(core.SOC_EB_MIN * 100, 0)} %",
            "consequence": "la batterie Énergie peut fournir de la puissance",
        },
        {
            "concept": "SOCState (bas)",
            "libelle": LIBELLES_SYMBOLIQUES["EB_low_SOC"],
            "actif": bool(etats["EB_low_SOC"]),
            "mesure": f"SOC de l'EB {nombre(soc_eb * 100, 0)} %",
            "consequence": "la batterie Énergie doit être protégée",
        },
        {
            "concept": "PowerState (récupération)",
            "libelle": LIBELLES_SYMBOLIQUES["regenerative_braking"],
            "actif": bool(etats["regenerative_braking"]),
            "mesure": f"puissance demandée {nombre(p_kw, 1, signe=True)} kW",
            "consequence": "l'énergie récupérée est dirigée vers les batteries",
        },
        {
            "concept": "ConverterPower (limite)",
            "libelle": LIBELLES_SYMBOLIQUES["converter_risk"],
            "actif": bool(etats["converter_risk"]),
            "mesure": f"seuil d'alerte {nombre(core.CONVERTER_RISK_THRESHOLD * 100, 0)} % de la capacité",
            "consequence": "la sollicitation du convertisseur doit être limitée",
        },
    ]


def indice_confiance(p_dem, soc_eb, soc_pb, correction, alpha_ecart=0.0):
    """Indice de confiance dans la décision, borné 0-100, avec ses raisons.

    Il ne s'agit PAS d'une probabilité produite par un modèle : c'est une mesure
    de marge par rapport aux situations limites (SOC proche du seuil, demande
    proche de la limite de l'EB, correction du filtre). Le libellé l'indique.
    """
    raisons_pour, raisons_contre = [], []
    score = 100.0

    marge_soc = (soc_eb - core.SOC_EB_MIN) / max(core.SOC_EB_MIN, 1e-6)
    if marge_soc < 0.15:
        score -= 25
        raisons_contre.append(f"SOC de l'EB proche du seuil minimal ({nombre(soc_eb * 100, 0)} %)")
    else:
        raisons_pour.append("états de charge éloignés des seuils critiques")

    ratio_p = abs(p_dem) / max(core.P_EB_MAX_W, 1e-6)
    if 0.85 <= ratio_p <= 1.15:
        score -= 20
        raisons_contre.append("puissance demandée très proche de la limite de la batterie Énergie")
    else:
        raisons_pour.append("puissance demandée éloignée des limites du système")

    if correction:
        score -= 30
        raisons_contre.append("le filtre physique a dû corriger la décision")
    else:
        raisons_pour.append("décision acceptée sans correction du filtre")

    if alpha_ecart > 0.15:
        score -= 15
        raisons_contre.append(f"écart notable avec la proposition initiale ({nombre(alpha_ecart, 2)})")

    return max(0.0, min(100.0, score)), raisons_pour, raisons_contre


def contrefactuels(p_dem, soc_eb, soc_pb):
    """Phrases « que se serait-il passé si… », calculées à partir des seuils
    réels du système et de l'écart avec la situation courante."""
    phrases = []
    p_kw = p_dem / 1000.0
    seuil_kw = core.P_EB_MAX_W / 1000.0

    if p_dem > core.P_EB_MAX_W:
        phrases.append(
            f"Si la demande avait été inférieure à {nombre(seuil_kw, 1)} kW (au lieu de "
            f"{nombre(p_kw, 1)} kW), la batterie Énergie aurait pu fournir seule la puissance."
        )
    elif p_dem > core.EPS_POWER_W:
        phrases.append(
            f"Si la demande avait dépassé {nombre(seuil_kw, 1)} kW (au lieu de {nombre(p_kw, 1)} kW), "
            "la batterie Puissance aurait dû compléter."
        )

    if soc_eb <= core.SOC_EB_MIN + 0.05:
        phrases.append(
            f"Si le SOC de la batterie Énergie avait dépassé "
            f"{nombre((core.SOC_EB_MIN + 0.05) * 100, 0)} %, elle n'aurait pas été protégée "
            "et aurait pris une part plus importante."
        )
    else:
        phrases.append(
            f"Si le SOC de la batterie Énergie était descendu sous "
            f"{nombre(core.SOC_EB_MIN * 100, 0)} %, elle aurait été protégée et la batterie "
            "Puissance aurait pris le relais."
        )

    if p_dem < -core.EPS_POWER_W:
        phrases.append(
            "Si le véhicule n'avait pas été en freinage, aucune récupération d'énergie "
            "n'aurait été déclenchée."
        )
    else:
        phrases.append(
            "Si le véhicule avait freiné, l'énergie récupérée aurait été dirigée vers "
            "les batteries au lieu d'être consommée."
        )

    return phrases


def expliquer_importances(noms_lisibles, importances):
    """Transforme des importances (gradients) en une phrase en langage naturel."""
    total = float(sum(abs(v) for v in importances))
    if total <= 0:
        return "Aucune variable ne se détache nettement à cet instant."

    parts = sorted(
        ((nom, 100.0 * abs(v) / total) for nom, v in zip(noms_lisibles, importances)),
        key=lambda kv: kv[1],
        reverse=True,
    )
    principale, pct1 = parts[0]
    if len(parts) > 1:
        seconde, pct2 = parts[1]
        return (
            f"Le modèle s'est principalement appuyé sur « {principale} » ({nombre(pct1, 0)} %) "
            f"et sur « {seconde} » ({nombre(pct2, 0)} %). Les autres variables ont eu une "
            "influence secondaire."
        )
    return f"Le modèle s'est appuyé presque exclusivement sur « {principale} » ({nombre(pct1, 0)} %)."


def chaine_inference(p_dem, soc_eb, soc_pb, part_eb, part_pb, correction, p_eb=None):
    """Construit la séquence Mesures → Concepts → Règles → Décision → Validation."""
    concepts = concepts_actifs(p_dem, soc_eb, soc_pb, p_eb=p_eb)
    activees, _, _ = evaluer_regles(p_dem, soc_eb, soc_pb)

    return {
        "mesures": [
            f"Puissance demandée : {nombre(p_dem / 1000, 1)} kW",
            f"SOC batterie Énergie : {nombre(soc_eb * 100, 0)} %",
            f"SOC batterie Puissance : {nombre(soc_pb * 100, 0)} %",
        ],
        "concepts": [c for c in concepts if c["actif"]],
        "concepts_absents": [c for c in concepts if not c["actif"]],
        "regles": activees,
        "decision": f"Énergie {nombre(part_eb, 0)} % · Puissance {nombre(part_pb, 0)} %",
        "validation": (
            "Décision corrigée par le filtre physique"
            if correction
            else "Décision acceptée sans correction"
        ),
    }
