"""
core/ontology_explainer.py — Fait « parler » l'ontologie OntoHESS.

Ce module utilise l'ontologie comme SOURCE D'EXPLICATION : il lit les règles
logiques (SWRL) réellement présentes dans ontologies/OntoHESS2.owl, les évalue
avec les grandeurs de l'instant analysé, et les écrit en langage courant, en
français ou en anglais (core/i18n.py).

Deux limites assumées et affichées à l'utilisateur :

1. L'application ne contient pas de moteur de raisonnement OWL/SWRL : les
   règles sont lues depuis l'ontologie et évaluées ici numériquement.
2. Seules les règles dont toutes les grandeurs sont connues à l'instant analysé
   peuvent être évaluées ; les autres sont signalées comme non évaluables.
"""

from functools import lru_cache

import ems_core as core
from core.format import nombre
from core.i18n import lib, tr


CHEMIN_OWL = core.ROOT_DIR / "ontologies" / "OntoHESS2.owl"

# Termes de l'ontologie en langage courant (français, anglais).
VOCABULAIRE = {
    "hasPower": ("puissance demandée par la charge", "power demanded by the load"),
    "hasPowerBattery": ("puissance de la batterie", "battery power"),
    "hasOutputPowerBattery": ("puissance fournie par une batterie", "power supplied by a battery"),
    "hasOutputPowerConverter": ("puissance de sortie du convertisseur", "converter output power"),
    "hasInputPowerConverter": ("puissance d'entrée du convertisseur", "converter input power"),
    "hasOperatingModeDriventrain": ("mode de fonctionnement de la chaîne de traction", "drivetrain operating mode"),
    "hasVoltageBattery": ("tension de la batterie", "battery voltage"),
    "hasCurrentBattery": ("courant de la batterie", "battery current"),
    "hasSocBattery": ("état de charge de la batterie", "battery state of charge"),
    "hasMaxDischargeCurrentBattery": ("courant maximal de décharge", "maximum discharge current"),
    "hasMaxChargeCurrentBattery": ("courant maximal de recharge", "maximum charge current"),
    "pEB_max_value": ("puissance maximale de décharge de la batterie Énergie", "maximum discharge power of the Energy battery"),
    "pEB_min_value": ("puissance maximale de recharge de la batterie Énergie", "maximum charge power of the Energy battery"),
    "socEB_minThreshold": ("seuil minimal de SOC de la batterie Énergie", "minimum SOC threshold of the Energy battery"),
    "socEB_maxThreshold": ("seuil maximal de SOC de la batterie Énergie", "maximum SOC threshold of the Energy battery"),
    "socPB_minThreshold": ("seuil minimal de SOC de la batterie Puissance", "minimum SOC threshold of the Power battery"),
    "socPB_maxThreshold": ("seuil maximal de SOC de la batterie Puissance", "maximum SOC threshold of the Power battery"),
}

# Noms des principales classes (concepts) de l'ontologie.
CLASSES = {
    "BatteryEB": ("Batterie Énergie", "Energy battery"),
    "BatteryPB": ("Batterie Puissance", "Power battery"),
    "Converter": ("Convertisseur", "Converter"),
    "Load": ("Charge (moteur)", "Load (motor)"),
    "HESS": ("Système hybride de stockage", "Hybrid energy storage system"),
    "Overload": ("Dépassement de limite", "Limit exceeded"),
    "OverloadCondition": ("Condition de dépassement", "Limit-exceeded condition"),
    "NormalOperation": ("Fonctionnement normal", "Normal operation"),
    "SOCState": ("État de charge", "State of charge"),
    "PowerState": ("État de puissance", "Power state"),
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

# Correspondance entre les variables des règles et les grandeurs de la simulation.
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


def nom_clair(nom: str) -> str:
    """Terme de l'ontologie en langage courant (le nom lui-même s'il est inconnu)."""
    if nom in VOCABULAIRE:
        return lib(VOCABULAIRE[nom])
    return nom_classe(nom)


def nom_classe(nom: str) -> str:
    return lib(CLASSES[nom]) if nom in CLASSES else nom


@lru_cache(maxsize=1)
def charger_regles():
    """Lit les règles SWRL de l'ontologie. Retourne une liste de dictionnaires
    {id, classes, conditions, calculs, conclusions, lectures, affectations}.
    Liste vide si rdflib est absent ou le fichier introuvable."""
    graphe = _graphe()
    if graphe is None:
        return []
    from rdflib import Namespace, RDF

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
    graphe = _graphe()
    if graphe is None:
        return frozenset()
    from rdflib import OWL, RDF

    return frozenset(str(c).split("#")[-1] for c in graphe.subjects(RDF.type, OWL.Class))


def diagnostic_configuration(soc_eb0, soc_pb0, nb_strategies):
    """Vérification de la configuration AVANT simulation, à partir des classes
    réellement déclarées dans l'ontologie : l'ontologie sert aussi à valider
    l'expérience, et pas seulement à expliquer les décisions."""
    presentes = classes_ontologie()

    attendus = [
        ("HESS", ("Architecture reconnue : système hybride de stockage", "Recognised architecture: hybrid storage system")),
        ("BatteryEB", ("Source d'énergie : batterie Énergie", "Energy source: Energy battery")),
        ("BatteryPB", ("Source d'énergie : batterie Puissance", "Energy source: Power battery")),
        ("Converter", ("Organe de répartition : convertisseur", "Power-sharing device: converter")),
        ("ManagementStrategy", ("Objet d'étude : stratégie de gestion d'énergie", "Subject of study: energy management strategy")),
    ]
    contexte = [{"concept": nom, "libelle": lib(libelle), "reconnu": nom in presentes} for nom, libelle in attendus]

    contraintes_owl = [
        ("SOCCondition", ("Préservation des états de charge", "Preservation of the states of charge")),
        ("OverloadCondition", ("Protection contre les dépassements de limite", "Protection against exceeded limits")),
        ("PowerThreshold", ("Respect des seuils de puissance", "Compliance with power thresholds")),
    ]
    contraintes = [{"concept": nom, "libelle": lib(libelle), "reconnu": nom in presentes} for nom, libelle in contraintes_owl]

    alertes = []
    if soc_eb0 < 0.35:
        alertes.append(tr(
            "SOC initial de la batterie Énergie à {s} % : elle atteindra vite son seuil minimal et "
            "sera alors protégée par le filtre de sécurité.",
            "Initial SOC of the Energy battery at {s} %: it will soon reach its minimum threshold "
            "and will then be protected by the safety filter.",
            s=nombre(soc_eb0 * 100, 0),
        ))
    if soc_pb0 < 0.35:
        alertes.append(tr(
            "SOC initial de la batterie Puissance à {s} % : elle pourra beaucoup moins absorber "
            "les pics de demande.",
            "Initial SOC of the Power battery at {s} %: it will be much less able to absorb "
            "demand peaks.",
            s=nombre(soc_pb0 * 100, 0),
        ))

    conseils = [tr(
        "Pour une comparaison pertinente, sélectionnez au moins une stratégie d'IA en plus des "
        "deux références.",
        "For a meaningful comparison, select at least one AI strategy in addition to the two "
        "reference strategies.",
    )] if nb_strategies < 3 else []

    coherent = all(c["reconnu"] for c in contexte)
    return {
        "contexte": contexte,
        "contraintes": contraintes,
        "alertes": alertes,
        "conseils": conseils,
        "conclusion": tr(
            "Configuration cohérente avec une étude des stratégies de gestion d'énergie d'un "
            "système hybride de stockage.",
            "Configuration consistent with a study of energy management strategies for a hybrid "
            "storage system.",
        ) if coherent else tr(
            "Certains concepts attendus sont absents de l'ontologie : la vérification est partielle.",
            "Some expected concepts are missing from the ontology: the check is partial.",
        ),
    }


@lru_cache(maxsize=1)
def vocabulaire_ontologie():
    """Relations (ObjectProperty), attributs (DatatypeProperty) et individus
    réellement déclarés dans l'ontologie."""
    graphe = _graphe()
    if graphe is None:
        return (), (), ()
    from rdflib import OWL, RDF

    def noms(type_owl):
        return tuple(sorted({str(s).split("#")[-1] for s in graphe.subjects(RDF.type, type_owl)}))

    return noms(OWL.ObjectProperty), noms(OWL.DatatypeProperty), noms(OWL.NamedIndividual)


# États de fonctionnement réellement déclarés comme individus dans l'ontologie.
# Les identifiants OWL gardent le mot « Overload », mais l'état signifie
# seulement que la demande dépasse la limite de la batterie Énergie seule
# (seuils pEB_max_value / pEB_min_value) : le HESS, lui, n'est pas en surcharge.
ETATS_ONTOLOGIE = {
    "state_Normal": ("Dans les limites de la batterie Énergie", "Within the Energy battery's limits"),
    "state_Overload_High": (
        "Traction au-delà de la limite de la batterie Énergie",
        "Traction beyond the Energy battery's limit",
    ),
    "state_Overload_Low": (
        "Récupération au-delà de la limite de la batterie Énergie",
        "Regeneration beyond the Energy battery's limit",
    ),
}

# Versions courtes, pour les nœuds de graphe.
ETATS_ONTOLOGIE_COURTS = {
    "state_Normal": ("Dans les limites EB", "Within EB limits"),
    "state_Overload_High": ("Traction > limite EB", "Traction > EB limit"),
    "state_Overload_Low": ("Récup. > limite EB", "Regen. > EB limit"),
}

# États qualitatifs déduits par l'ontologie et fournis aux modèles
# neuro-symboliques (compute_symbolic_states).
LIBELLES_SYMBOLIQUES = {
    "high_power_demand": ("Forte demande de puissance", "High power demand"),
    "regenerative_braking": ("Freinage avec récupération", "Regenerative braking"),
    "zero_power_demand": ("Demande quasi nulle", "Near-zero demand"),
    "converter_risk": ("Convertisseur proche de sa limite", "Converter close to its limit"),
    "EB_available": ("Batterie Énergie disponible", "Energy battery available"),
    "PB_available": ("Batterie Puissance disponible", "Power battery available"),
    "EB_low_SOC": ("SOC de la batterie Énergie faible", "Low Energy battery SOC"),
    "PB_low_SOC": ("SOC de la batterie Puissance faible", "Low Power battery SOC"),
}


def libelle_etat(cle: str, court: bool = False) -> str:
    return lib((ETATS_ONTOLOGIE_COURTS if court else ETATS_ONTOLOGIE)[cle])


def libelle_symbolique(cle: str) -> str:
    return lib(LIBELLES_SYMBOLIQUES[cle]) if cle in LIBELLES_SYMBOLIQUES else cle


def etat_fonctionnement(p_dem):
    """Individu `state_*` de l'ontologie correspondant à la puissance demandée,
    avec les seuils que comparent les règles (pEB_max_value, pEB_min_value)."""
    if p_dem > core.P_EB_MAX_W:
        return "state_Overload_High"
    if p_dem < core.P_EB_MIN_W:
        return "state_Overload_Low"
    return "state_Normal"


def etat_instant(p_dem, soc_eb, soc_pb, p_eb=None):
    """Source UNIQUE de ce que les pages affichent comme « état » à un instant :
    l'état de fonctionnement inféré par l'ontologie, et les états qualitatifs
    fournis aux modèles neuro-symboliques.

    p_eb : puissance réellement fournie par la batterie Énergie à cet instant
    (P_EB de la trajectoire). Toutes les pages passent cette même grandeur.
    """
    cle = etat_fonctionnement(p_dem)
    return {
        "fonctionnement": cle,
        "libelle": libelle_etat(cle),
        "symboliques": core.compute_symbolic_states(p_dem, soc_eb, soc_pb, p_eb=p_eb),
    }


HYPOTHESES = [
    ("Tensions des packs constantes (valeurs nominales), sans variation avec le SOC.",
     "Constant pack voltages (nominal values), with no variation with SOC."),
    ("Convertisseur représenté par sa puissance et ses limites, sans modèle électrique détaillé.",
     "Converter represented by its power and limits, without a detailed electrical model."),
    ("Pas de modèle thermique ni de vieillissement des batteries.",
     "No thermal or ageing model of the batteries."),
    ("Pertes non comptées dans le SOC par défaut, comme dans l'article de référence (option ci-dessous).",
     "Losses not counted in the SOC by default, as in the reference paper (option below)."),
    ("Cycle de conduite connu à l'avance ; météo et pente non prises en compte.",
     "Driving cycle known in advance; weather and road slope not taken into account."),
]


def hypotheses():
    return [lib(h) for h in HYPOTHESES]


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
    """Résout un argument de règle : nombre écrit en clair ou variable connue."""
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
    return nombre(valeur, 2)


# Lecture en clair des règles de mode de fonctionnement et de répartition.
# Elles sont identifiées par leurs conditions, pas par leur numéro : le numéro
# dépend de l'ordre de lecture du fichier, les conditions non.
def _cle_conditions(conditions):
    return frozenset((op, tuple(args)) for _, op, args in conditions)


LECTURE_REGLES = {
    frozenset({("greaterThan", ("P", "0")), ("lessThanOrEqual", ("P", "pmax"))}): ("mode", (
        "la traction reste dans les limites de la batterie Énergie",
        "traction stays within the Energy battery's limits",
    )),
    frozenset({("greaterThan", ("P", "pmax"))}): ("mode", (
        "la traction dépasse la limite de la batterie Énergie : la PB doit assister",
        "traction exceeds the Energy battery's limit: the PB must assist",
    )),
    frozenset({("lessThanOrEqual", ("soc", "smin"))}): ("mode", (
        "la batterie Énergie est à son SOC minimal : elle doit être protégée",
        "the Energy battery is at its minimum SOC: it must be protected",
    )),
    frozenset({("lessThan", ("P", "0"))}): ("mode", (
        "le véhicule freine : de l'énergie est récupérée",
        "the vehicle is braking: energy is recovered",
    )),
    frozenset({("lessThanOrEqual", ("soc", "smin")), ("greaterThan", ("P", "0"))}): ("repartition", (
        "la PB fournit toute la puissance, l'EB est protégée",
        "the PB supplies all the power, the EB is protected",
    )),
    frozenset({("greaterThan", ("soc", "smin")), ("greaterThan", ("P", "pmax"))}): ("repartition", (
        "l'EB fournit sa puissance maximale, la PB complète",
        "the EB supplies its maximum power, the PB supplies the rest",
    )),
    frozenset({("lessThan", ("P", "pmin"))}): ("repartition", (
        "l'EB absorbe jusqu'à sa limite, la PB absorbe le surplus",
        "the EB absorbs up to its limit, the PB absorbs the surplus",
    )),
    frozenset({("greaterThan", ("soc", "smin")), ("greaterThan", ("P", "0")), ("lessThanOrEqual", ("P", "pmax"))}): ("repartition", (
        "l'EB fournit toute la puissance, la PB reste au repos",
        "the EB supplies all the power, the PB stays idle",
    )),
    frozenset({("lessThan", ("P", "0")), ("greaterThanOrEqual", ("P", "pmin"))}): ("repartition", (
        "l'EB absorbe toute l'énergie récupérée",
        "the EB absorbs all the recovered energy",
    )),
}

# Conditions de ces règles en langage courant ; les seuils sont remplis à l'affichage.
CONDITIONS_EN_CLAIR = {
    ("greaterThan", ("P", "0")): ("le véhicule est en traction", "the vehicle is in traction"),
    ("lessThan", ("P", "0")): ("le véhicule freine", "the vehicle is braking"),
    ("greaterThan", ("P", "pmax")): (
        "la demande dépasse la limite de décharge de la batterie Énergie ({pmax} kW)",
        "demand exceeds the Energy battery's discharge limit ({pmax} kW)",
    ),
    ("lessThanOrEqual", ("P", "pmax")): (
        "la demande ne dépasse pas la limite de décharge de la batterie Énergie ({pmax} kW)",
        "demand does not exceed the Energy battery's discharge limit ({pmax} kW)",
    ),
    ("lessThan", ("P", "pmin")): (
        "la puissance récupérée dépasse ce que la batterie Énergie peut absorber ({pmin} kW)",
        "the recovered power exceeds what the Energy battery can absorb ({pmin} kW)",
    ),
    ("greaterThanOrEqual", ("P", "pmin")): (
        "la batterie Énergie peut absorber toute la puissance récupérée (jusqu'à {pmin} kW)",
        "the Energy battery can absorb all the recovered power (up to {pmin} kW)",
    ),
    ("lessThanOrEqual", ("soc", "smin")): (
        "le SOC de la batterie Énergie a atteint son minimum ({smin} %)",
        "the Energy battery's SOC has reached its minimum ({smin} %)",
    ),
    ("greaterThan", ("soc", "smin")): (
        "le SOC de la batterie Énergie est au-dessus de son minimum ({smin} %)",
        "the Energy battery's SOC is above its minimum ({smin} %)",
    ),
}

# Sens des modes de fonctionnement affectés par les règles R9 à R12.
SENS_DES_MODES = {
    "Normal": ("l'EB suffit à fournir la demande", "the EB can supply the demand on its own"),
    "Surcharge": (
        "la PB doit assister l'EB (le HESS lui-même n'est pas en surcharge)",
        "the PB must assist the EB (the HESS itself is not overloaded)",
    ),
    "ProtectionEB": ("l'EB doit être protégée", "the EB must be protected"),
    "Recuperation": ("de l'énergie est récupérée", "energy is being recovered"),
}


def _condition_en_clair(op, args):
    modele = CONDITIONS_EN_CLAIR.get((op, tuple(args)))
    if modele is None:
        return f"{args[0]} {COMPARATEURS[op]} {args[1]}"
    return lib(modele).format(
        pmax=nombre(core.P_EB_MAX_W / 1000, 1),
        pmin=nombre(-core.P_EB_MIN_W / 1000, 1),
        smin=nombre(core.SOC_EB_MIN * 100, 0),
    )


def regle_en_phrase(regle):
    """« Si …, alors … » en langage courant pour une règle de mode ou de
    répartition ; conditions en notation brute pour les autres."""
    morceaux = [_condition_en_clair(op, args) for _, op, args in regle["conditions"] if len(args) >= 2]
    type_regle, lecture = lire_regle(regle)
    if not morceaux:
        return lecture[0].upper() + lecture[1:] + "."
    if len(morceaux) > 2:
        si = tr(", et ", ", and ").join(morceaux)
    else:
        si = tr(" et que ", " and ").join(morceaux)
    mode = next((args[1] for p, args in regle.get("affectations", []) if p == "hasOperatingModeDriventrain"), None)
    if type_regle == "mode" and mode in SENS_DES_MODES:
        return tr(
            "Si {si}, alors le mode de fonctionnement est « {mode} » : {sens}.",
            "If {si}, then the operating mode is “{mode}”: {sens}.",
            si=si, mode=mode, sens=lib(SENS_DES_MODES[mode]),
        )
    return tr("Si {si}, alors {lecture}.", "If {si}, then {lecture}.", si=si, lecture=lecture)


# Règles floues : libellé court, et concepts de l'ontologie qu'elles mobilisent
# (classes d'OntoHESS entre parenthèses).
REGLES_FLOUES = {
    "R1_PB_low_traction": (("R1 · PB basse en traction", "R1 · Low PB in traction"),
                           [(("SOC de la PB bas", "low PB SOC"), "SOCState"), (("traction", "traction"), "PowerState")]),
    "R2_EB_low_PB_available": (("R2 · EB basse, PB disponible", "R2 · Low EB, PB available"),
                               [(("SOC de l'EB bas", "low EB SOC"), "SOCState"), (("PB disponible", "PB available"), "SOCState")]),
    "R3_strong_traction": (("R3 · forte traction", "R3 · Strong traction"),
                           [(("forte demande de traction", "high traction demand"), "PowerState")]),
    "R4_zero_demand": (("R4 · demande nulle", "R4 · Zero demand"),
                       [(("demande quasi nulle", "near-zero demand"), "PowerState")]),
    "R5_regenerative_braking": (("R5 · freinage", "R5 · Braking"),
                                [(("freinage avec récupération", "regenerative braking"), "PowerState")]),
    "R5b_PB_high_recharge": (("R5b · PB pleine en recharge", "R5b · Full PB while charging"),
                             [(("SOC de la PB élevé", "high PB SOC"), "SOCState"), (("récupération", "regeneration"), "PowerState")]),
    "R7_two_low_SOC": (("R7 · deux SOC bas", "R7 · Both SOCs low"),
                       [(("SOC des deux batteries bas", "both batteries' SOC low"), "SOCState")]),
}

# Ce que chaque règle floue cherche à faire (ems_core.RULE_LABELS_FR, et sa traduction).
SENS_REGLES_FLOUES = {
    "R1_PB_low_traction": (
        "limiter la PB, car elle est presque déchargée, même en phase de traction",
        "limit the PB, as it is almost empty, even during traction",
    ),
    "R2_EB_low_PB_available": (
        "faire porter l'effort sur la PB, car l'EB est faible tandis que la PB reste disponible",
        "put the load on the PB, as the EB is low while the PB is still available",
    ),
    "R3_strong_traction": (
        "solliciter davantage la PB en raison d'une forte demande de traction",
        "use the PB more because of a high traction demand",
    ),
    "R4_zero_demand": (
        "maintenir une répartition stable, car la demande est presque nulle",
        "keep a stable split, as the demand is almost zero",
    ),
    "R5_regenerative_braking": (
        "orienter la récupération d'énergie vers la PB",
        "send the recovered energy to the PB",
    ),
    "R5b_PB_high_recharge": (
        "limiter la recharge de la PB, car elle est déjà fortement chargée",
        "limit the PB charging, as it is already well charged",
    ),
    "R7_two_low_SOC": (
        "adopter une répartition prudente, car les deux batteries présentent un SOC faible",
        "adopt a cautious split, as both batteries have a low SOC",
    ),
    "DEFAULT": (
        "appliquer la répartition par défaut, car aucune règle ne domine clairement",
        "apply the default split, as no rule clearly dominates",
    ),
}


def libelle_regle_floue(cle: str) -> str:
    return lib(REGLES_FLOUES[cle][0]) if cle in REGLES_FLOUES else cle


def sens_regle_floue(cle: str) -> str:
    return lib(SENS_REGLES_FLOUES[cle]) if cle in SENS_REGLES_FLOUES else ""


def concepts_regle_floue(cle: str):
    """[(concept en clair, classe de l'ontologie)] d'une règle floue."""
    return [(lib(nom), classe) for nom, classe in REGLES_FLOUES.get(cle, (None, []))[1]]


def calculs_en_clair(regle):
    """Calculs d'une règle (opérations SWRL) : « z = x − y »."""
    sortie = []
    for _, op, args in regle["calculs"]:
        if len(args) >= 3:
            sortie.append(f"{args[0]} = {f' {CALCULS[op]} '.join(args[1:])}")
    return sortie


def regles_floues():
    """Les sept règles floues du moteur (ems_core), décrites à partir de ses
    constantes : conditions, conclusion (part de la PB), sens et concepts de
    l'ontologie mobilisés. Le moteur calcule alpha comme la moyenne des
    conclusions, pondérée par le degré d'activation de chaque règle."""
    conditions = {
        "R1_PB_low_traction": ("SOC de la PB bas ET traction", "low PB SOC AND traction"),
        "R2_EB_low_PB_available": (
            "SOC de l'EB bas ET PB disponible (SOC moyen ou haut) ET traction",
            "low EB SOC AND PB available (medium or high SOC) AND traction",
        ),
        "R3_strong_traction": (
            "forte traction ET PB disponible (SOC moyen ou haut)",
            "strong traction AND PB available (medium or high SOC)",
        ),
        "R4_zero_demand": ("demande nulle ET accélération stable", "zero demand AND steady acceleration"),
        "R5_regenerative_braking": (
            "récupération ET PB rechargeable (SOC bas ou moyen)",
            "regeneration AND PB can be charged (low or medium SOC)",
        ),
        "R5b_PB_high_recharge": ("récupération ET SOC de la PB haut", "regeneration AND high PB SOC"),
        "R7_two_low_SOC": ("SOC de l'EB bas ET SOC de la PB bas", "low EB SOC AND low PB SOC"),
    }
    return [
        {
            "cle": cle,
            "libelle": libelle_regle_floue(cle),
            "si": lib(conditions[cle]) if cle in conditions else "—",
            "alpha": float(core.FUZZY_RULE_CONSEQUENTS[i]),
            "sens": sens_regle_floue(cle),
            "concepts": concepts_regle_floue(cle),
        }
        for i, cle in enumerate(core.FUZZY_RULE_NAMES)
    ]


def termes_flous():
    """Définition chiffrée des termes employés par les règles floues."""
    kw = lambda w: f"{nombre(w / 1000, 1)} kW"  # noqa: E731
    pct = lambda s: f"{nombre(s * 100, 0)} %"  # noqa: E731
    w = lambda p: f"{nombre(p, 0)} W"  # noqa: E731
    return [
        (tr("SOC de l'EB bas", "Low EB SOC"), tr(
            "vrai en dessous de {a}, faux au-dessus de {b}", "true below {a}, false above {b}",
            a=pct(core.SOC_LOW_FULL_EB), b=pct(core.SOC_LOW_THRESHOLD))),
        (tr("SOC de la PB bas", "Low PB SOC"), tr(
            "vrai en dessous de {a}, faux au-dessus de {b}", "true below {a}, false above {b}",
            a=pct(core.SOC_LOW_FULL_PB), b=pct(core.SOC_LOW_THRESHOLD))),
        (tr("SOC moyen", "Medium SOC"), tr(
            "monte de 25 à 35 %, vrai de 35 à 65 %, redescend jusqu'à 75 %",
            "rises from 25 to 35 %, true from 35 to 65 %, falls back until 75 %")),
        (tr("SOC haut", "High SOC"), tr("faux en dessous de 70 %, vrai au-dessus de 80 %", "false below 70 %, true above 80 %")),
        (tr("Traction", "Traction"), tr(
            "de {a} jusqu'à la limite de l'EB ({b}) et au-delà", "from {a} up to the EB limit ({b}) and beyond",
            a=w(core.EPS_POWER_W), b=kw(core.P_EB_MAX_W))),
        (tr("Forte traction", "Strong traction"), tr(
            "faux en dessous de {a}, vrai au-delà de {b}", "false below {a}, true above {b}",
            a=kw(0.6 * core.P_EB_MAX_W), b=kw(core.P_EB_MAX_W))),
        (tr("Demande nulle", "Zero demand"), tr(
            "|P| ≤ {a} (faux à partir de {b})", "|P| ≤ {a} (false from {b})",
            a=w(core.EPS_POWER_W), b=w(2 * core.EPS_POWER_W))),
        (tr("Récupération", "Regeneration"), tr(
            "puissance négative ; forte en dessous de {a}", "negative power; strong below {a}", a=kw(core.P_EB_MIN_W))),
        (tr("Accélération stable", "Steady acceleration"), tr(
            "|a| ≤ 0,1 m/s² (faux à partir de 0,4 m/s²)", "|a| ≤ 0.1 m/s² (false from 0.4 m/s²)")),
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


# Variables des règles de l'ontologie, en clair.
GLOSSAIRE_VARIABLES = {
    "P": ("puissance demandée", "power demand"),
    "pmax": ("puissance maximale de décharge de l'EB", "EB maximum discharge power"),
    "pmin": ("puissance maximale de recharge de l'EB", "EB maximum charge power"),
    "soc": ("SOC de l'EB", "EB SOC"),
    "smin": ("SOC minimal de l'EB", "EB minimum SOC"),
    "imax": ("courant maximal de l'EB", "EB maximum current"),
    "imin": ("courant minimal de l'EB", "EB minimum current"),
    "ibp_nom": ("courant nominal de la PB", "PB nominal current"),
    "ibe_nom2": ("courant nominal de l'EB", "EB nominal current"),
    "icharge": ("courant de charge", "load current"),
    "veb": ("tension de l'EB", "EB voltage"),
    "vpb": ("tension de la PB", "PB voltage"),
    "ieb": ("courant de l'EB", "EB current"),
}


def glossaire():
    return [(v, lib(s)) for v, s in GLOSSAIRE_VARIABLES.items()]


def lire_regle(regle):
    """(type, lecture) d'une règle : « mode », « repartition », « courant » ou « calcul »."""
    if not regle["conditions"]:
        grandeurs = ", ".join(dict.fromkeys(nom_clair(c) for c in regle["conclusions"]))
        return "calcul", tr("calcule une grandeur ({g})", "computes a quantity ({g})", g=grandeurs)
    cle = _cle_conditions(regle["conditions"])
    if cle in LECTURE_REGLES:
        type_regle, lecture = LECTURE_REGLES[cle]
        return type_regle, lib(lecture)
    return "courant", tr("fixe le courant d'une batterie selon ses limites", "sets a battery current within its limits")


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
            continue  # règle de calcul pur (P = V × I, etc.)

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
            "classes": [nom_classe(c) for c in regle["classes"]],
            # Une règle qui conclut sur les deux batteries répète la même propriété.
            "conclusions": list(dict.fromkeys(nom_clair(c) for c in regle["conclusions"])),
            "details": details,
        }
        if not connue:
            indeterminables.append(entree)
        elif satisfaite:
            activees.append(entree)
        else:
            non_activees.append(entree)

    return activees, non_activees, indeterminables


def contrefactuels(p_dem, soc_eb, soc_pb):
    """Phrases « que se serait-il passé si… », calculées à partir des seuils
    réels du système et de l'écart avec la situation courante."""
    phrases = []
    p = nombre(p_dem / 1000.0, 1)
    seuil = nombre(core.P_EB_MAX_W / 1000.0, 1)

    if p_dem > core.P_EB_MAX_W:
        phrases.append(tr(
            "Si la demande avait été inférieure à {s} kW (au lieu de {p} kW), la batterie Énergie "
            "aurait pu fournir seule la puissance.",
            "Had the demand been below {s} kW (instead of {p} kW), the Energy battery could have "
            "supplied the power on its own.",
            s=seuil, p=p,
        ))
    elif p_dem > core.EPS_POWER_W:
        phrases.append(tr(
            "Si la demande avait dépassé {s} kW (au lieu de {p} kW), la batterie Puissance aurait "
            "dû compléter.",
            "Had the demand exceeded {s} kW (instead of {p} kW), the Power battery would have had "
            "to supply the rest.",
            s=seuil, p=p,
        ))

    if soc_eb <= core.SOC_EB_MIN + 0.05:
        phrases.append(tr(
            "Si le SOC de la batterie Énergie avait dépassé {s} %, elle n'aurait pas été protégée "
            "et aurait pris une part plus importante.",
            "Had the Energy battery's SOC been above {s} %, it would not have been protected and "
            "would have taken a larger share.",
            s=nombre((core.SOC_EB_MIN + 0.05) * 100, 0),
        ))
    else:
        phrases.append(tr(
            "Si le SOC de la batterie Énergie était descendu sous {s} %, elle aurait été protégée "
            "et la batterie Puissance aurait pris le relais.",
            "Had the Energy battery's SOC dropped below {s} %, it would have been protected and the "
            "Power battery would have taken over.",
            s=nombre(core.SOC_EB_MIN * 100, 0),
        ))

    if p_dem < -core.EPS_POWER_W:
        phrases.append(tr(
            "Si le véhicule n'avait pas été en freinage, aucune récupération d'énergie n'aurait "
            "été déclenchée.",
            "Had the vehicle not been braking, no energy recovery would have been triggered.",
        ))
    else:
        phrases.append(tr(
            "Si le véhicule avait freiné, l'énergie récupérée aurait été dirigée vers les batteries "
            "au lieu d'être consommée.",
            "Had the vehicle been braking, the recovered energy would have been sent to the "
            "batteries instead of being consumed.",
        ))

    return phrases
