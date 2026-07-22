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


CHEMIN_OWL = core.ROOT_DIR / "ontologies" / "OntoHESS2.owl"

# Traduction des termes de l'ontologie en langage courant.
VOCABULAIRE = {
    "hasPower": "puissance demandée par la charge",
    "hasPowerBattery": "puissance de la batterie",
    "hasOutputPowerBattery": "puissance fournie par une batterie",
    "hasOutputPowerConverter": "puissance de sortie du convertisseur",
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
            return ("propriete", str(graphe.value(atome, SWRL.propertyPredicate)).split("#")[-1])
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
            f"L'ontologie associe un SOC initial de {soc_eb0 * 100:.0f} % à la classe "
            "`SOCCondition` : la batterie Énergie atteindra rapidement sa limite de "
            "fonctionnement et sera protégée par le filtre."
        )
    if soc_pb0 < 0.35:
        alertes.append(
            f"SOC initial de la batterie Puissance à {soc_pb0 * 100:.0f} % : sa capacité "
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
ETATS_ONTOLOGIE = {
    "state_Normal": "Fonctionnement normal",
    "state_Overload_High": "Surcharge en traction",
    "state_Overload_Low": "Surcharge en récupération",
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
            "mesure": f"SOC de la batterie Énergie = {soc_eb * 100:.0f} %",
            "propriete": "hasSocBattery",
            "individu": "batteryE1",
        },
        {
            "mesure": f"SOC de la batterie Puissance = {soc_pb * 100:.0f} %",
            "propriete": "hasSocBattery",
            "individu": "batteryP1",
        },
        {
            "mesure": f"Puissance demandée = {p_dem / 1000:.1f} kW",
            "propriete": "hasPower",
            "individu": "load1",
        },
    ]

    # L'état est déterminé par les mêmes seuils que ceux comparés dans les règles.
    if p_dem > core.P_EB_MAX_W:
        etat = "state_Overload_High"
        justification = (
            f"la puissance demandée ({p_dem / 1000:.1f} kW) dépasse `pEB_max_value` "
            f"({core.P_EB_MAX_W / 1000:.1f} kW)"
        )
    elif p_dem < core.P_EB_MIN_W:
        etat = "state_Overload_Low"
        justification = (
            f"la puissance récupérée ({p_dem / 1000:.1f} kW) dépasse la capacité de "
            f"recharge `pEB_min_value` ({core.P_EB_MIN_W / 1000:.1f} kW)"
        )
    else:
        etat = "state_Normal"
        justification = (
            f"la puissance demandée ({p_dem / 1000:.1f} kW) reste dans les limites "
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
                f"`hasSocBattery` = {soc_eb * 100:.0f} % comparé à `socEB_minThreshold` "
                f"= {core.SOC_EB_MIN * 100:.0f} %"
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
        return f"{valeur * 100:.0f} %"
    if abs(valeur) >= 1000:
        return f"{valeur / 1000:.1f} kW"
    return f"{valeur:.2f}"


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

        entree = {
            "id": regle["id"],
            "classes": [CLASSES_FR.get(c, c) for c in regle["classes"]],
            "conclusions": [_fr(c) for c in regle["conclusions"]],
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
    """Concepts de l'ontologie reconnus à cet instant, avec la mesure qui les
    justifie. S'appuie sur compute_symbolic_states (reproduction à seuils fixes
    des règles) et nomme les classes réellement présentes dans OntoHESS2.owl."""
    etats = core.compute_symbolic_states(p_dem, soc_eb, soc_pb, p_eb=p_eb)
    p_kw = float(p_dem) / 1000.0

    return [
        {
            "concept": "Overload",
            "libelle": "Surcharge (forte demande de puissance)",
            "actif": bool(etats["high_power_demand"]),
            "mesure": f"puissance demandée {abs(p_kw):.1f} kW, seuil {core.HIGH_POWER_THRESHOLD_W / 1000:.0f} kW",
            "consequence": "la batterie Puissance est davantage sollicitée",
        },
        {
            "concept": "NormalOperation",
            "libelle": "Fonctionnement normal de la batterie Énergie",
            "actif": bool(etats["EB_available"]),
            "mesure": f"SOC de l'EB {soc_eb * 100:.0f} %, seuil minimal {core.SOC_EB_MIN * 100:.0f} %",
            "consequence": "la batterie Énergie peut fournir de la puissance",
        },
        {
            "concept": "SOCState (bas)",
            "libelle": "État de charge bas de la batterie Énergie",
            "actif": bool(etats["EB_low_SOC"]),
            "mesure": f"SOC de l'EB {soc_eb * 100:.0f} %",
            "consequence": "la batterie Énergie doit être protégée",
        },
        {
            "concept": "PowerState (récupération)",
            "libelle": "Freinage récupératif",
            "actif": bool(etats["regenerative_braking"]),
            "mesure": f"puissance demandée {p_kw:+.1f} kW",
            "consequence": "l'énergie récupérée est dirigée vers les batteries",
        },
        {
            "concept": "ConverterPower (limite)",
            "libelle": "Convertisseur proche de sa limite",
            "actif": bool(etats["converter_risk"]),
            "mesure": f"seuil d'alerte {core.CONVERTER_RISK_THRESHOLD * 100:.0f} % de la capacité",
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
        raisons_contre.append(f"SOC de l'EB proche du seuil minimal ({soc_eb * 100:.0f} %)")
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
        raisons_contre.append(f"écart notable avec la proposition initiale ({alpha_ecart:.2f})")

    return max(0.0, min(100.0, score)), raisons_pour, raisons_contre


def contrefactuels(p_dem, soc_eb, soc_pb):
    """Phrases « que se serait-il passé si… », calculées à partir des seuils
    réels du système et de l'écart avec la situation courante."""
    phrases = []
    p_kw = p_dem / 1000.0
    seuil_kw = core.P_EB_MAX_W / 1000.0

    if p_dem > core.P_EB_MAX_W:
        phrases.append(
            f"Si la demande avait été inférieure à {seuil_kw:.0f} kW (au lieu de "
            f"{p_kw:.1f} kW), la batterie Énergie aurait pu fournir seule la puissance."
        )
    elif p_dem > core.EPS_POWER_W:
        phrases.append(
            f"Si la demande avait dépassé {seuil_kw:.0f} kW (au lieu de {p_kw:.1f} kW), "
            "la batterie Puissance aurait dû compléter."
        )

    if soc_eb <= core.SOC_EB_MIN + 0.05:
        phrases.append(
            f"Si le SOC de la batterie Énergie avait dépassé "
            f"{(core.SOC_EB_MIN + 0.05) * 100:.0f} %, elle n'aurait pas été protégée "
            "et aurait pris une part plus importante."
        )
    else:
        phrases.append(
            f"Si le SOC de la batterie Énergie était descendu sous "
            f"{core.SOC_EB_MIN * 100:.0f} %, elle aurait été protégée et la batterie "
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
            f"Le modèle s'est principalement appuyé sur « {principale} » ({pct1:.0f} %) "
            f"et sur « {seconde} » ({pct2:.0f} %). Les autres variables ont eu une "
            "influence secondaire."
        )
    return f"Le modèle s'est appuyé presque exclusivement sur « {principale} » ({pct1:.0f} %)."


def chaine_inference(p_dem, soc_eb, soc_pb, part_eb, part_pb, correction, p_eb=None):
    """Construit la séquence Mesures → Concepts → Règles → Décision → Validation."""
    concepts = concepts_actifs(p_dem, soc_eb, soc_pb, p_eb=p_eb)
    activees, _, _ = evaluer_regles(p_dem, soc_eb, soc_pb)

    return {
        "mesures": [
            f"Puissance demandée : {p_dem / 1000:.1f} kW",
            f"SOC batterie Énergie : {soc_eb * 100:.0f} %",
            f"SOC batterie Puissance : {soc_pb * 100:.0f} %",
        ],
        "concepts": [c for c in concepts if c["actif"]],
        "concepts_absents": [c for c in concepts if not c["actif"]],
        "regles": activees,
        "decision": f"Énergie {part_eb:.0f} % · Puissance {part_pb:.0f} %",
        "validation": (
            "Décision corrigée par le filtre physique"
            if correction
            else "Décision acceptée sans correction"
        ),
    }
