"""
core/presentation.py — Textes de présentation des stratégies et de l'ontologie,
communs à l'application Streamlit et à l'application web.

Chaque fonction retourne ses textes dans la langue choisie (core/i18n.py). Les
étapes des chaînes de décision portent un rôle (« demande », « reference »,
« secondaire », « decision ») : chaque interface en déduit la couleur.
"""

import ems_core as core
from core import xai
from core.format import nombre
from core.i18n import lib, tr

DEMANDE, REFERENCE, SECONDAIRE, DECISION = "demande", "reference", "secondaire", "decision"


def principe_commun() -> list:
    """Chaîne commune à toutes les stratégies : [(texte, rôle)]."""
    return [
        ("P_dem + SOC_EB + SOC_PB", DEMANDE),
        (tr("Stratégie EMS", "EMS strategy"), DECISION),
        (tr("alpha : répartition", "alpha: power split"), DECISION),
        (tr("Filtre de sécurité", "Safety filter"), SECONDAIRE),
        (tr("Modèle physique du HESS", "Physical model of the HESS"), SECONDAIRE),
        (tr("SOC, puissances, courants", "SOC, powers, currents"), REFERENCE),
    ]


def architecture_electrique() -> str:
    part_conv = (core.V_EB_PACK_NOM - core.V_PB_PACK_NOM) / core.V_EB_PACK_NOM
    return tr(
        "Le HESS suit l'architecture en cascade à source de courant contrôlée de Fonseca de Freitas et al. [1] :\n"
        "- la batterie Puissance est branchée directement sur le bus continu ;\n"
        "- le convertisseur est placé **en série** entre les deux batteries : il règle le courant de "
        "la batterie Énergie et ne traite que la différence de tension entre elles, soit **{p} %** de "
        "la puissance de la batterie Énergie avec {veb} V et {vpb} V ;\n"
        "- le convertisseur n'étant pas réversible en tension, la batterie Énergie doit rester à une "
        "tension supérieure à celle de la batterie Puissance.",
        "The HESS follows the controlled current source cascade architecture of Fonseca de Freitas et al. [1]:\n"
        "- the Power battery is connected directly to the DC bus;\n"
        "- the converter is placed **in series** between the two batteries: it sets the Energy "
        "battery's current and only processes the voltage difference between them, i.e. **{p} %** of "
        "the Energy battery's power with {veb} V and {vpb} V;\n"
        "- as the converter is not voltage-reversible, the Energy battery must stay at a higher "
        "voltage than the Power battery.",
        p=nombre(part_conv * 100, 1), veb=nombre(core.V_EB_PACK_NOM, 0), vpb=nombre(core.V_PB_PACK_NOM, 1),
    )


REFERENCE_ARTICLE = (
    "[1] C. A. Fonseca de Freitas, P. Bartholomeus, X. Margueron, P. Le Moigne, « Partial Power "
    "Converter for Electric Vehicle Hybrid Energy Storage System Using a Controlled Current Source "
    "Cascade Architecture », IEEE Access, vol. 12, 2024 (fig. 8, eq. (9)–(16))."
)


def _grandeurs(colonnes):
    return ", ".join(xai.libelle_entree(c) for c in colonnes)


def fiches_strategies() -> dict:
    """Fiche de chaque stratégie : nature, rôle, fonctionnement, ce qu'elle a
    appris, grandeurs utilisées, et sa chaîne de décision (« flux »)."""
    fenetre = core.LSTM_WINDOW
    reserve = nombre(core.MLP_NS_RESERVE_PB_SOC * 100, 0)
    correction_max = nombre(core.MLP_NS_MAX_DELTA * 100, 0)
    demande_soc = tr("P_dem, SOC", "P_dem, SOC")
    return {
        "EMS_power_limitation": {
            "nature": tr("Règle physique fixe", "Fixed physical rule"),
            "role": tr(
                "Référence physique, toujours calculable et sans apprentissage : c'est la stratégie "
                "« power limitation » de l'article de référence [1] (fig. 10), à laquelle on compare les autres.",
                "Physical reference, always computable and without learning: it is the “power limitation” "
                "strategy of the reference paper [1] (fig. 10), against which the others are compared.",
            ),
            "fonctionnement": [
                tr("La batterie Énergie fournit la puissance en priorité, jusqu'à sa puissance maximale.",
                   "The Energy battery supplies the power first, up to its maximum power."),
                tr("Quand la demande dépasse cette limite, la batterie Puissance fournit le complément ; si "
                   "le SOC de la batterie Énergie atteint son minimum, la batterie Puissance fournit tout.",
                   "When the demand exceeds this limit, the Power battery supplies the rest; if the Energy "
                   "battery's SOC reaches its minimum, the Power battery supplies everything."),
                tr("Rien n'est appris : le comportement découle des seules limites physiques des batteries.",
                   "Nothing is learned: the behaviour follows only from the physical limits of the batteries."),
            ],
            "appris": tr("Aucun apprentissage : règle fixe.", "No learning: fixed rule."),
            "grandeurs": tr("Puissance demandée, SOC de la batterie Énergie", "Power demand, Energy battery SOC"),
            "flux": [
                (tr("P_dem, SOC EB", "P_dem, EB SOC"), DEMANDE),
                (tr("Limite de puissance de l'EB", "EB power limit"), REFERENCE),
                ("alpha", DECISION),
            ],
        },
        "EMS_fuzzy_logic": {
            "nature": tr("Règles expertes à degrés de vérité (logique floue)", "Expert rules with degrees of truth (fuzzy logic)"),
            "role": tr(
                "Traduire le savoir d'un expert en une décision continue, sans réseau de neurones. Sa "
                "décision sert aussi de point de départ à NS-MLP.",
                "Turn an expert's knowledge into a continuous decision, without a neural network. Its "
                "decision is also the starting point of NS-MLP.",
            ),
            "fonctionnement": [
                tr("Sept règles du type « si le SOC de la batterie Puissance est bas et que le véhicule est "
                   "en traction, alors… ». Une condition n'est pas vraie ou fausse : elle est vraie à un "
                   "certain degré, entre 0 et 1.",
                   "Seven rules such as “if the Power battery's SOC is low and the vehicle is in traction, "
                   "then…”. A condition is not true or false: it is true to a certain degree, between 0 and 1."),
                tr("Ces conditions reprennent les concepts de l'ontologie OntoHESS (états de charge, états de puissance).",
                   "These conditions use the concepts of the OntoHESS ontology (states of charge, power states)."),
                tr("Chaque règle propose une part pour la batterie Puissance ; la décision est la moyenne de "
                   "ces propositions, pondérée par le degré de vérité de chaque règle. On sait donc "
                   "exactement ce que chaque règle apporte à la décision.",
                   "Each rule proposes a share for the Power battery; the decision is the average of these "
                   "proposals, weighted by each rule's degree of truth. We therefore know exactly what each "
                   "rule contributes to the decision."),
            ],
            "appris": tr("Aucun apprentissage : règles fixées par l'expert.", "No learning: rules set by the expert."),
            "grandeurs": tr("SOC des deux batteries, puissance demandée, accélération", "SOC of both batteries, power demand, acceleration"),
            "flux": [
                (demande_soc, DEMANDE),
                (tr("Degré de vérité de chaque condition", "Degree of truth of each condition"), REFERENCE),
                (tr("Règles SI … ALORS", "IF … THEN rules"), REFERENCE),
                ("alpha", DECISION),
            ],
        },
        "EMS_MLP": {
            "nature": tr("Réseau de neurones sans mémoire", "Neural network without memory"),
            "role": tr(
                "Apprendre directement la répartition à partir de l'état du système à l'instant présent. "
                "C'est la référence « apprentissage seul », sans connaissances expertes.",
                "Learn the split directly from the state of the system at the present time. It is the "
                "“learning only” reference, without expert knowledge.",
            ),
            "fonctionnement": [
                tr("Une fonction ajustée sur des exemples (deux étages de {a} puis {b} neurones) : elle "
                   "reçoit l'état du système et rend alpha, entre 0 et 1.",
                   "A function fitted on examples (two stages of {a} then {b} neurons): it receives the "
                   "state of the system and returns alpha, between 0 and 1.",
                   a=core.MLP_HIDDEN_1, b=core.MLP_HIDDEN_2),
                tr("Elle ne voit qu'un instant à la fois : aucune mémoire du passé.",
                   "It only sees one time step at a time: no memory of the past."),
            ],
            "appris": tr(
                "La répartition alpha* qui, à chaque instant des données d'apprentissage, minimise un coût "
                "physique à cinq termes (sollicitation en puissance, énergie échangée, risque sur le SOC, "
                "sollicitation du convertisseur, continuité de la décision). Ce coût est minimisé instant "
                "par instant, sans anticiper la suite du cycle.",
                "The split alpha* which, at each time step of the training data, minimises a five-term "
                "physical cost (power stress, energy exchanged, SOC risk, converter stress, continuity of "
                "the decision). This cost is minimised step by step, without anticipating the rest of the cycle.",
            ),
            "grandeurs": _grandeurs(core.MLP_INPUT_COLS),
            "flux": [
                (tr("État à l'instant présent", "State at the present time"), DEMANDE),
                (tr("Réseau MLP", "MLP network"), SECONDAIRE),
                ("alpha", DECISION),
            ],
        },
        "EMS_MLP_neurosymbolic": {
            "nature": tr("Neuro-symbolique : règles expertes corrigées par un réseau", "Neuro-symbolic: expert rules corrected by a network"),
            "role": tr(
                "Corriger la logique floue sans la remplacer : garder une décision explicable tout en profitant de l'apprentissage.",
                "Correct the fuzzy logic without replacing it: keep an explainable decision while benefiting from learning.",
            ),
            "fonctionnement": [
                tr("Le réseau ne donne pas alpha : il donne une petite correction, ajoutée à la décision de la logique floue.",
                   "The network does not output alpha: it outputs a small correction, added to the fuzzy-logic decision."),
                tr("Cette correction est limitée à ±{c} points de répartition.",
                   "This correction is limited to ±{c} points of the split.", c=correction_max),
                tr("La décision se décompose donc exactement en « règles + correction » : c'est la seule "
                   "stratégie à apprentissage dont on lit directement la part due aux règles.",
                   "The decision therefore splits exactly into “rules + correction”: it is the only "
                   "learning-based strategy in which the share due to the rules can be read directly."),
                tr("Garde-fou de l'ontologie : quand le SOC de la batterie Puissance passe sous {r} %, les "
                   "règles R14 et R16 d'OntoHESS reprennent la main en traction (la batterie Énergie "
                   "d'abord, la batterie Puissance pour le seul surplus), afin de garder son énergie pour les pics.",
                   "Ontology safeguard: when the Power battery's SOC drops below {r} %, OntoHESS rules R14 "
                   "and R16 take over in traction (the Energy battery first, the Power battery only for the "
                   "surplus), so as to keep its energy for the peaks.", r=reserve),
            ],
            "appris": tr(
                "La même répartition alpha* que le MLP, apprise sous forme de correction de la logique floue.",
                "The same split alpha* as the MLP, learned as a correction of the fuzzy logic.",
            ),
            "grandeurs": tr(
                "État à l'instant présent, décision de la logique floue, états déduits par l'ontologie, et "
                "trois prévisions de NS-LSTM (demande à venir, variations de SOC des deux batteries) : 17 grandeurs.",
                "State at the present time, fuzzy-logic decision, states inferred by the ontology, and three "
                "NS-LSTM forecasts (upcoming demand, SOC variations of both batteries): 17 quantities.",
            ),
            "flux": [
                (demande_soc, DEMANDE),
                (tr("Règles floues", "Fuzzy rules"), REFERENCE),
                (tr("alpha des règles", "alpha from the rules"), DECISION),
                (tr("+ correction du réseau (±{c} pts)", "+ network correction (±{c} pts)", c=correction_max), SECONDAIRE),
                (tr("garde-fou R14/R16", "safeguard R14/R16"), REFERENCE),
                (tr("alpha final", "final alpha"), DECISION),
            ],
        },
        "EMS_LSTM": {
            "nature": tr("Réseau de neurones à mémoire", "Neural network with memory"),
            "role": tr(
                "Tenir compte du passé récent du cycle, et non du seul instant présent, pour anticiper l'évolution du système.",
                "Take the recent past of the cycle into account, not only the present time, to anticipate how the system evolves.",
            ),
            "fonctionnement": [
                tr("Il lit les {w} dernières secondes du cycle.", "It reads the last {w} seconds of the cycle.", w=fenetre),
                tr("Il ne donne pas alpha directement : il prévoit la demande à venir et la variation de SOC "
                   "de chaque batterie au pas suivant. La répartition est déduite de la variation de SOC "
                   "prévue pour la batterie Puissance.",
                   "It does not output alpha directly: it forecasts the upcoming demand and the SOC variation "
                   "of each battery at the next step. The split is deduced from the SOC variation forecast "
                   "for the Power battery."),
            ],
            "appris": tr(
                "La demande et les variations de SOC des données de référence, elles-mêmes produites par le modèle physique.",
                "The demand and SOC variations of the reference data, themselves produced by the physical model.",
            ),
            "grandeurs": _grandeurs(core.LSTM_FEATURE_COLS),
            "flux": [
                (tr("{w} dernières secondes", "Last {w} seconds", w=fenetre), DEMANDE),
                (tr("Réseau LSTM", "LSTM network"), SECONDAIRE),
                (tr("Variations de SOC prévues", "Forecast SOC variations"), SECONDAIRE),
                ("alpha", DECISION),
            ],
        },
        "EMS_LSTM_neurosymbolic": {
            "nature": tr("Neuro-symbolique : réseau à mémoire informé par l'ontologie", "Neuro-symbolic: network with memory informed by the ontology"),
            "role": tr(
                "Donner au réseau à mémoire, en plus des grandeurs physiques, des états déduits par "
                "l'ontologie : une décision informée à la fois par le passé récent et par les connaissances expertes.",
                "Give the network with memory, in addition to the physical quantities, states inferred by "
                "the ontology: a decision informed both by the recent past and by expert knowledge.",
            ),
            "fonctionnement": [
                tr("Même fonctionnement que le LSTM (lecture des {w} dernières secondes).",
                   "Same operation as the LSTM (reads the last {w} seconds).", w=fenetre),
                tr("Il reçoit en plus quatre états déduits par l'ontologie : forte demande, freinage avec "
                   "récupération, demande nulle, convertisseur proche de sa limite.",
                   "It additionally receives four states inferred by the ontology: high demand, regenerative "
                   "braking, zero demand, converter close to its limit."),
                tr("Les connaissances expertes n'interviennent qu'en entrée : il n'y a ni règles de départ ni correction limitée.",
                   "Expert knowledge only comes in as inputs: there are neither starting rules nor a limited correction."),
            ],
            "appris": tr("Les mêmes grandeurs que le LSTM.", "The same quantities as the LSTM."),
            "grandeurs": _grandeurs(core.LSTM_NS_FEATURE_COLS),
            "flux": [
                (tr("{w} dernières secondes + 4 états de l'ontologie", "Last {w} seconds + 4 ontology states", w=fenetre), DEMANDE),
                (tr("Réseau LSTM", "LSTM network"), SECONDAIRE),
                ("alpha", DECISION),
            ],
        },
        "EMS_GNN": {
            "nature": tr("Réseau de neurones sur le schéma du système", "Neural network on the system diagram"),
            "role": tr(
                "Représenter explicitement la structure du HESS : chaque composant échange de l'information "
                "avec ceux auxquels il est relié, avant la décision.",
                "Represent the structure of the HESS explicitly: each component exchanges information with "
                "those it is connected to, before the decision.",
            ),
            "fonctionnement": [
                tr("Le système est décrit par cinq composants reliés : batterie Énergie, batterie Puissance, convertisseur, moteur, véhicule.",
                   "The system is described by five connected components: Energy battery, Power battery, converter, motor, vehicle."),
                tr("Chaque composant porte ses grandeurs physiques (SOC, courants et puissances limites, capacité, demande, accélération).",
                   "Each component carries its physical quantities (SOC, current and power limits, capacity, demand, acceleration)."),
                tr("L'information circule {n} fois entre composants voisins ; une moyenne sur l'ensemble donne ensuite alpha.",
                   "Information passes {n} times between neighbouring components; an average over all of them then gives alpha.",
                   n=core.GNN_NUM_LAYERS),
            ],
            "appris": tr(
                "La répartition du modèle physique : le GNN apprend à l'imiter, d'où des résultats presque identiques aux siens.",
                "The split of the physical model: the GNN learns to imitate it, hence results almost identical to its own.",
            ),
            "grandeurs": tr("Grandeurs physiques de chacun des cinq composants", "Physical quantities of each of the five components"),
            "flux": [
                (tr("Schéma du HESS (5 composants)", "HESS diagram (5 components)"), DEMANDE),
                (tr("Réseau GNN", "GNN network"), SECONDAIRE),
                ("alpha", DECISION),
            ],
        },
    }


def synthese_strategies() -> dict:
    """{stratégie: {apprentissage, appris, memoire, ontologie, lecture}} pour le tableau de synthèse."""
    oui, non = tr("Oui", "Yes"), tr("Non", "No")
    memoire = tr("Oui : {w} dernières secondes", "Yes: last {w} seconds", w=core.LSTM_WINDOW)
    alpha_etoile = tr("Répartition de moindre coût à chaque instant (alpha*)", "Lowest-cost split at each time step (alpha*)")
    soc_reference = tr("Variations de SOC du modèle physique", "SOC variations of the physical model")
    lignes = {
        "EMS_power_limitation": (non, "—", non, tr("Mêmes décisions que les règles R13 à R17", "Same decisions as rules R13 to R17")),
        "EMS_fuzzy_logic": (non, "—", non, tr("Ses règles utilisent les concepts de l'ontologie", "Its rules use the ontology's concepts")),
        "EMS_MLP": (oui, alpha_etoile, non, non),
        "EMS_MLP_neurosymbolic": (oui, alpha_etoile, non,
                                  tr("Oui : règles floues, états déduits, garde-fou", "Yes: fuzzy rules, inferred states, safeguard")),
        "EMS_LSTM": (oui, soc_reference, memoire, non),
        "EMS_LSTM_neurosymbolic": (oui, soc_reference, memoire, tr("Oui : 4 états déduits en entrée", "Yes: 4 inferred states as inputs")),
        "EMS_GNN": (oui, tr("Décisions du modèle physique", "Decisions of the physical model"), non,
                    tr("En partie : schéma des composants du HESS", "Partly: diagram of the HESS components")),
    }
    return {
        cle: {"apprentissage": a, "appris": b, "memoire": c, "ontologie": d, "lecture": lib(xai.TRANSPARENCE[cle][1])}
        for cle, (a, b, c, d) in lignes.items()
    }


# Ontologie

def categories_ontologie() -> dict:
    """{catégorie de l'ontologie: (nom, ce qu'elle regroupe)}."""
    return {
        "Component": (
            tr("Composants", "Components"),
            tr("Les éléments physiques : batterie Énergie, batterie Puissance, convertisseur, charge "
               "(moteur) et l'architecture qui les relie.",
               "The physical elements: Energy battery, Power battery, converter, load (motor) and the "
               "architecture that connects them."),
        ),
        "ElectricalQuantity": (
            tr("Grandeurs électriques", "Electrical quantities"),
            tr("Les puissances, courants, tensions et résistances internes, et les seuils qui les bornent "
               "(puissance maximale de la batterie Énergie, courant maximal…).",
               "The powers, currents, voltages and internal resistances, and the thresholds that bound "
               "them (maximum power of the Energy battery, maximum current…)."),
        ),
        "ManagementStrategy": (
            tr("Stratégie de gestion", "Management strategy"),
            tr("Ce qu'examine une stratégie de gestion : conditions sur le SOC, sur la charge, sur le "
               "dépassement d'une limite, et le calcul de la puissance du convertisseur.",
               "What a management strategy examines: conditions on the SOC, on the load, on exceeding a "
               "limit, and the calculation of the converter power."),
        ),
        "SystemState": (
            tr("États du système", "System states"),
            tr("Les états du système : fonctionnement dans les limites de la batterie Énergie ou au-delà, "
               "états de charge et états de puissance.",
               "The system states: operation within the Energy battery's limits or beyond, states of "
               "charge and power states."),
        ),
    }


TYPES_REGLES = {
    "mode": ("Mode de fonctionnement", "Operating mode"),
    "repartition": ("Répartition de la puissance", "Power split"),
    "courant": ("Limitation de courant", "Current limitation"),
    "calcul": ("Calcul d'une grandeur", "Calculation of a quantity"),
}
SUJETS_REGLES = {
    "eb": ("batterie Énergie", "Energy battery"), "pb": ("batterie Puissance", "Power battery"),
    "l": ("charge", "load"), "c": ("convertisseur", "converter"),
}


def usages_ontologie() -> dict:
    """{stratégie: comment elle utilise l'ontologie}."""
    aucun = tr("Aucun usage.", "Not used.")
    return {
        "EMS_power_limitation": tr(
            "Ne lit pas l'ontologie, mais prend exactement les décisions de ses règles de répartition R13 à R17.",
            "Does not read the ontology, but makes exactly the decisions of its split rules R13 to R17.",
        ),
        "EMS_fuzzy_logic": tr(
            "Ses règles reposent sur les concepts « état de charge » et « état de puissance » de l'ontologie.",
            "Its rules rely on the ontology's “state of charge” and “power state” concepts.",
        ),
        "EMS_MLP": aucun,
        "EMS_LSTM": aucun,
        "EMS_GNN": tr(
            "Pas d'usage direct ; son schéma reprend les composants du HESS (batteries, convertisseur, moteur, véhicule).",
            "No direct use; its diagram uses the components of the HESS (batteries, converter, motor, vehicle).",
        ),
        "EMS_MLP_neurosymbolic": tr(
            "Part des règles floues, reçoit des états déduits par l'ontologie, et reste sous le contrôle de ses règles R14 et R16.",
            "Starts from the fuzzy rules, receives states inferred by the ontology, and stays under the control of its rules R14 and R16.",
        ),
        "EMS_LSTM_neurosymbolic": tr(
            "Reçoit quatre états déduits par l'ontologie : forte demande, freinage, demande nulle, convertisseur proche de sa limite.",
            "Receives four states inferred by the ontology: high demand, braking, zero demand, converter close to its limit.",
        ),
    }


def chaines_neuro_symboliques() -> dict:
    """Les deux façons d'associer l'ontologie à un réseau : {stratégie: (titre, chaîne, légende)}."""
    return {
        "EMS_MLP_neurosymbolic": (
            tr("**NS-MLP** · les règles décident, le réseau corrige", "**NS-MLP** · the rules decide, the network corrects"),
            [
                (tr("P_dem, SOC", "P_dem, SOC"), REFERENCE),
                (tr("Règles floues", "Fuzzy rules"), SECONDAIRE),
                (tr("Décision des règles", "Decision of the rules"), DECISION),
                (tr("Correction du réseau (±{c} pts)", "Network correction (±{c} pts)", c=nombre(core.MLP_NS_MAX_DELTA * 100, 0)), SECONDAIRE),
                (tr("Garde-fou R14/R16", "Safeguard R14/R16"), REFERENCE),
                (tr("Décision finale", "Final decision"), DECISION),
            ],
            tr(
                "La décision se décompose exactement : part des règles + correction du réseau. Sous {r} % "
                "de SOC de la batterie Puissance, les règles R14 et R16 reprennent la main en traction pour "
                "préserver sa réserve.",
                "The decision splits exactly: share from the rules + network correction. Below {r} % SOC of "
                "the Power battery, rules R14 and R16 take over in traction to preserve its reserve.",
                r=nombre(core.MLP_NS_RESERVE_PB_SOC * 100, 0),
            ),
        ),
        "EMS_LSTM_neurosymbolic": (
            tr("**NS-LSTM** · l'ontologie informe le réseau", "**NS-LSTM** · the ontology informs the network"),
            [
                (tr("{w} dernières secondes + 4 états déduits par l'ontologie", "Last {w} seconds + 4 states inferred by the ontology",
                    w=core.LSTM_WINDOW), REFERENCE),
                (tr("Réseau LSTM", "LSTM network"), SECONDAIRE),
                (tr("Décision finale", "Final decision"), DECISION),
            ],
            tr(
                "Les états déduits sont des entrées parmi d'autres : leur poids dans la décision se mesure après coup.",
                "The inferred states are inputs among others: their weight in the decision is measured afterwards.",
            ),
        ),
    }
