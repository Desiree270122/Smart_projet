"""
Fonctionnement des stratégies EMS : comment chacune transforme l'état du HESS
(demande, états de charge) en une répartition de la puissance, avec quelles
connaissances, et ce qu'elle a appris.
"""

import sys
from pathlib import Path

DOSSIER_PROJET = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DOSSIER_PROJET))

import pandas as pd
import streamlit as st

import ems_core as core
from core import xai
from core.format import nombre
from core.i18n import lib, tr
from core.navigation import pied_navigation
from core.resultats import famille, nom_affichage
from core.style import COULEUR_DECISION, COULEUR_DEMANDE, COULEUR_REFERENCE, COULEUR_SECONDAIRE, flux_html


st.title(tr("🧠 Fonctionnement des stratégies EMS", "🧠 How the EMS strategies work"))
st.caption(tr(
    "Comment chaque stratégie transforme l'état du HESS en une répartition de la puissance.",
    "How each strategy turns the state of the HESS into a power split.",
))


# Le principe commun à toutes les stratégies

st.subheader(tr("Le principe commun", "The common principle"))
st.markdown(
    flux_html([
        ("P_dem + SOC_EB + SOC_PB", COULEUR_DEMANDE),
        (tr("Stratégie EMS", "EMS strategy"), COULEUR_DECISION),
        (tr("alpha : répartition", "alpha: power split"), COULEUR_DECISION),
        (tr("Filtre de sécurité", "Safety filter"), COULEUR_SECONDAIRE),
        (tr("Modèle physique du HESS", "Physical model of the HESS"), COULEUR_SECONDAIRE),
        (tr("SOC, puissances, courants", "SOC, powers, currents"), COULEUR_REFERENCE),
    ]),
    unsafe_allow_html=True,
)
eq1, eq2 = st.columns(2)
eq1.latex(r"P_{PB} = \alpha \times P_{dem}")
eq2.latex(r"P_{EB} = (1 - \alpha) \times P_{dem}")
st.caption(tr(
    "Toutes les stratégies donnent la même grandeur, alpha : la part de la puissance confiée à la "
    "batterie Puissance. Un filtre de sécurité vérifie ensuite les limites de courant, de puissance "
    "et de SOC, et corrige alpha si nécessaire.",
    "All strategies output the same quantity, alpha: the share of power assigned to the Power "
    "battery. A safety filter then checks the current, power and SOC limits, and corrects alpha if "
    "necessary.",
))

with st.expander(tr("Architecture électrique du HESS (convertisseur à puissance partielle)",
                    "Electrical architecture of the HESS (partial-power converter)")):
    part_conv = (core.V_EB_PACK_NOM - core.V_PB_PACK_NOM) / core.V_EB_PACK_NOM
    st.markdown(tr(
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
    ))
    st.latex(
        r"P_{conv} = (V_{EB} - V_{PB})\,I_{EB} = P_{EB}\,\frac{V_{EB} - V_{PB}}{V_{EB}}"
        r"\qquad P_{dem} = P_{EB} + P_{PB}"
    )
    st.caption(
        "[1] C. A. Fonseca de Freitas, P. Bartholomeus, X. Margueron, P. Le Moigne, « Partial Power "
        "Converter for Electric Vehicle Hybrid Energy Storage System Using a Controlled Current Source "
        "Cascade Architecture », IEEE Access, vol. 12, 2024 (fig. 8, eq. (9)–(16))."
    )


def _grandeurs(colonnes):
    return ", ".join(xai.libelle_entree(c) for c in colonnes)


fenetre = core.LSTM_WINDOW
reserve = nombre(core.MLP_NS_RESERVE_PB_SOC * 100, 0)
correction_max = nombre(core.MLP_NS_MAX_DELTA * 100, 0)

# Fiche de chaque stratégie : nature, rôle, fonctionnement, ce qu'elle a appris, grandeurs utilisées.
FICHES = {
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
    },
}

demande_soc = tr("P_dem, SOC", "P_dem, SOC")
FLUX = {
    "EMS_power_limitation": [
        (tr("P_dem, SOC EB", "P_dem, EB SOC"), COULEUR_DEMANDE),
        (tr("Limite de puissance de l'EB", "EB power limit"), COULEUR_REFERENCE),
        ("alpha", COULEUR_DECISION),
    ],
    "EMS_fuzzy_logic": [
        (demande_soc, COULEUR_DEMANDE),
        (tr("Degré de vérité de chaque condition", "Degree of truth of each condition"), COULEUR_REFERENCE),
        (tr("Règles SI … ALORS", "IF … THEN rules"), COULEUR_REFERENCE),
        ("alpha", COULEUR_DECISION),
    ],
    "EMS_MLP": [
        (tr("État à l'instant présent", "State at the present time"), COULEUR_DEMANDE),
        (tr("Réseau MLP", "MLP network"), COULEUR_SECONDAIRE),
        ("alpha", COULEUR_DECISION),
    ],
    "EMS_LSTM": [
        (tr("{w} dernières secondes", "Last {w} seconds", w=fenetre), COULEUR_DEMANDE),
        (tr("Réseau LSTM", "LSTM network"), COULEUR_SECONDAIRE),
        (tr("Variations de SOC prévues", "Forecast SOC variations"), COULEUR_SECONDAIRE),
        ("alpha", COULEUR_DECISION),
    ],
    "EMS_GNN": [
        (tr("Schéma du HESS (5 composants)", "HESS diagram (5 components)"), COULEUR_DEMANDE),
        (tr("Réseau GNN", "GNN network"), COULEUR_SECONDAIRE),
        ("alpha", COULEUR_DECISION),
    ],
    "EMS_MLP_neurosymbolic": [
        (demande_soc, COULEUR_DEMANDE),
        (tr("Règles floues", "Fuzzy rules"), COULEUR_REFERENCE),
        (tr("alpha des règles", "alpha from the rules"), COULEUR_DECISION),
        (tr("+ correction du réseau (±{c} pts)", "+ network correction (±{c} pts)", c=correction_max), COULEUR_SECONDAIRE),
        (tr("garde-fou R14/R16", "safeguard R14/R16"), COULEUR_REFERENCE),
        (tr("alpha final", "final alpha"), COULEUR_DECISION),
    ],
    "EMS_LSTM_neurosymbolic": [
        (tr("{w} dernières secondes + 4 états de l'ontologie", "Last {w} seconds + 4 ontology states", w=fenetre), COULEUR_DEMANDE),
        (tr("Réseau LSTM", "LSTM network"), COULEUR_SECONDAIRE),
        ("alpha", COULEUR_DECISION),
    ],
}


def _carte(cle):
    fiche = FICHES[cle]
    with st.container(border=True):
        st.markdown(f"#### {nom_affichage(cle)}")
        st.caption(f"{fiche['nature']}  ·  {tr('famille', 'family')} « {famille(cle)} »")
        st.markdown(flux_html(FLUX[cle]), unsafe_allow_html=True)
        with st.expander(tr("Détails", "Details")):
            titres = (
                tr("À quoi elle sert", "What it is for"), tr("Comment elle fonctionne", "How it works"),
                tr("Ce qu'elle a appris à reproduire", "What it learned to reproduce"), tr("Grandeurs utilisées", "Quantities used"),
            )
            st.markdown(f"**{titres[0]}** — {fiche['role']}")
            st.markdown(f"**{titres[1]}**\n" + "\n".join(f"- {point}" for point in fiche["fonctionnement"]))
            st.markdown(f"**{titres[2]}** — {fiche['appris']}")
            st.markdown(f"**{titres[3]}** — {fiche['grandeurs']}")


st.subheader(tr("Les sept stratégies", "The seven strategies"))
classiques = [c for c in core.MODEL_ORDER if c in FICHES and "neurosymbolic" not in c]
for i in range(0, len(classiques), 2):
    colonnes = st.columns(2)
    for col, cle in zip(colonnes, classiques[i:i + 2]):
        with col:
            _carte(cle)


# Les deux stratégies neuro-symboliques, au cœur du projet

st.subheader(tr("Pourquoi deux stratégies neuro-symboliques ?", "Why two neuro-symbolic strategies?"))
st.markdown(tr(
    "Les deux combinent connaissances expertes et apprentissage, mais pas au même endroit de la "
    "chaîne de décision. Les comparer montre ce que change la place donnée aux connaissances expertes.",
    "Both combine expert knowledge and learning, but not at the same place in the decision chain. "
    "Comparing them shows what the place given to expert knowledge changes.",
))
ns1, ns2 = st.columns(2)
with ns1:
    _carte("EMS_MLP_neurosymbolic")
    st.info(tr(
        "**NS-MLP** — les connaissances expertes sont le **socle de la décision** : les règles floues "
        "proposent une répartition, le réseau n'y ajoute qu'une correction limitée, et un garde-fou de "
        "l'ontologie (règles R14/R16) protège la réserve de la batterie Puissance.",
        "**NS-MLP** — expert knowledge is the **basis of the decision**: the fuzzy rules propose a "
        "split, the network only adds a limited correction, and an ontology safeguard (rules R14/R16) "
        "protects the Power battery's reserve.",
    ))
with ns2:
    _carte("EMS_LSTM_neurosymbolic")
    st.info(tr(
        "**NS-LSTM** — le réseau exploite **le passé récent du cycle** et reçoit des états déduits par "
        "l'ontologie comme informations supplémentaires. Les connaissances expertes éclairent la "
        "décision sans la structurer.",
        "**NS-LSTM** — the network uses **the recent past of the cycle** and receives states inferred "
        "by the ontology as additional information. Expert knowledge informs the decision without "
        "structuring it.",
    ))

st.divider()


# Tableau de synthèse

st.subheader(tr("Les stratégies en un tableau", "The strategies in one table"))
oui, non = tr("Oui", "Yes"), tr("Non", "No")
memoire = tr("Oui : {w} dernières secondes", "Yes: last {w} seconds", w=fenetre)
alpha_etoile = tr("Répartition de moindre coût à chaque instant (alpha*)", "Lowest-cost split at each time step (alpha*)")
soc_reference = tr("Variations de SOC du modèle physique", "SOC variations of the physical model")
# (apprentissage, mémoire du passé, usage de l'ontologie, ce qu'elle a appris à reproduire)
SYNTHESE = {
    "EMS_power_limitation": (non, non, tr("Mêmes décisions que les règles R13 à R17", "Same decisions as rules R13 to R17"), "—"),
    "EMS_fuzzy_logic": (non, non, tr("Ses règles utilisent les concepts de l'ontologie", "Its rules use the ontology's concepts"), "—"),
    "EMS_MLP": (oui, non, non, alpha_etoile),
    "EMS_MLP_neurosymbolic": (oui, non, tr("Oui : règles floues, états déduits, garde-fou", "Yes: fuzzy rules, inferred states, safeguard"), alpha_etoile),
    "EMS_LSTM": (oui, memoire, non, soc_reference),
    "EMS_LSTM_neurosymbolic": (oui, memoire, tr("Oui : 4 états déduits en entrée", "Yes: 4 inferred states as inputs"), soc_reference),
    "EMS_GNN": (oui, non, tr("En partie : schéma des composants du HESS", "Partly: diagram of the HESS components"),
                tr("Décisions du modèle physique", "Decisions of the physical model")),
}
col_strategie = tr("Stratégie", "Strategy")
st.dataframe(
    pd.DataFrame([
        {
            col_strategie: nom_affichage(cle),
            tr("Famille", "Family"): famille(cle),
            tr("Apprentissage", "Learning"): SYNTHESE[cle][0],
            tr("Ce qu'elle a appris à reproduire", "What it learned to reproduce"): SYNTHESE[cle][3],
            tr("Mémoire du passé", "Memory of the past"): SYNTHESE[cle][1],
            tr("Usage de l'ontologie", "Use of the ontology"): SYNTHESE[cle][2],
            tr("Comment la décision se lit", "How the decision can be read"): lib(xai.TRANSPARENCE[cle][1]),
        }
        for cle in core.MODEL_ORDER if cle in SYNTHESE
    ]).set_index(col_strategie),
    width="stretch",
)
st.caption(tr(
    "Les stratégies à apprentissage ont été réglées à l'avance, sur la première moitié du cycle "
    "Artemis ; l'application rejoue leurs décisions. Le détail de l'ontologie et de ses règles est "
    "dans « Base de connaissances ».",
    "The learning-based strategies were tuned beforehand, on the first half of the Artemis cycle; "
    "the app replays their decisions. The details of the ontology and its rules are in “Knowledge base”.",
))


pied_navigation("vues/9_Architecture_des_modeles.py")
