import sys
from pathlib import Path

DOSSIER_PROJET = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DOSSIER_PROJET))

import pandas as pd
import streamlit as st

import ems_core as core
from core.resultats import nom_affichage, famille, EXPLICABILITE


# Configuration de page gérée par le routeur Accueil.py.

st.title("🧠 Les modèles d'IA")

st.markdown(
    "*Toutes les stratégies poursuivent le même objectif : déterminer le coefficient "
    "`alpha(t)` qui répartit la puissance demandée entre la batterie Énergie et la "
    "batterie Puissance. Elles se rangent en quatre familles, celles que compare le "
    "projet 2SMART : règles fixes, ontologie seule, apprentissage seul et approche "
    "hybride neurosymbolique. Seules certaines s'appuient sur l'ontologie OntoHESS.*"
)


# Architecture de décision : deux chemins, avec ou sans connaissances expertes

st.subheader("Architecture globale de décision")


def _flux(etapes):
    html = "<div style='display:flex;align-items:center;flex-wrap:wrap;gap:6px;margin:.3rem 0'>"
    for i, (etape, coul) in enumerate(etapes):
        html += (
            f"<span style='border:1px solid {coul};color:{coul};border-radius:9px;"
            f"padding:6px 11px;font-weight:600;font-size:.86rem'>{etape}</span>"
        )
        if i < len(etapes) - 1:
            html += "<span style='color:#94A3B8;font-weight:800'>&#8594;</span>"
    return html + "</div>"


_DEBUT = [("Cycle de conduite", "#6B7280"), ("Variables physiques", "#6B7280")]
_FIN = [("Modèle EMS", "#3B82F6"), ("Filtre physique", "#22C55E"), ("Répartition EB / PB", "#22C55E")]

st.markdown("**Avec connaissances expertes** — logique floue, MLP et LSTM neurosymboliques")
st.markdown(
    _flux(_DEBUT + [("Concepts OntoHESS", "#F59E0B"), ("Règles floues ou états symboliques", "#F59E0B")] + _FIN),
    unsafe_allow_html=True,
)
st.markdown("**Sans connaissances expertes** — modèle physique, MLP, LSTM, GNN")
st.markdown(_flux(_DEBUT + _FIN), unsafe_allow_html=True)

st.caption(
    "Le modèle physique n'exécute pas l'ontologie : ses branches de décision "
    "correspondent aux règles SWRL R9 à R17 d'OntoHESS, ce qui permet de le relire "
    "avec l'ontologie a posteriori."
)

eq1, eq2 = st.columns(2)
with eq1:
    st.latex(r"P_{PB} = \alpha \times P_{dem}")
with eq2:
    st.latex(r"P_{EB} = (1 - \alpha) \times P_{dem}")

st.info(
    "Quelle que soit la stratégie, la décision passe ensuite par un filtre "
    "physique de sécurité qui vérifie les limites de courant, de puissance et de "
    "SOC des deux batteries, et corrige alpha si nécessaire."
)


# Architecture électrique : cascade à source de courant contrôlée ([1], fig. 8)

st.subheader("Architecture électrique du HESS")

_part_conv = (core.V_EB_PACK_NOM - core.V_PB_PACK_NOM) / core.V_EB_PACK_NOM
st.markdown(
    "Le HESS suit l'architecture « CCS cascade » de Fonseca de Freitas et al. [1] :\n"
    "- la batterie Puissance est branchée directement sur le bus DC ;\n"
    "- le convertisseur est placé **en série** entre les deux batteries : il pilote le "
    "courant de la batterie Énergie et ne traite que la différence de tension entre "
    f"elles, soit **{_part_conv * 100:.1f} %** de la puissance de l'EB avec "
    f"{core.V_EB_PACK_NOM:.0f} V et {core.V_PB_PACK_NOM:.1f} V ;\n"
    "- le convertisseur n'étant pas réversible en tension, la batterie Énergie doit "
    "rester à une tension supérieure à celle de la batterie Puissance."
)
st.latex(
    r"P_{conv} = (V_{EB} - V_{PB})\,I_{EB} = P_{EB}\,\frac{V_{EB} - V_{PB}}{V_{EB}}"
    r"\qquad P_{load} = P_{EB} + P_{PB}"
)
st.caption(
    "[1] C. A. Fonseca de Freitas, P. Bartholomeus, X. Margueron, P. Le Moigne, « Partial "
    "Power Converter for Electric Vehicle Hybrid Energy Storage System Using a Controlled "
    "Current Source Cascade Architecture », IEEE Access, vol. 12, 2024 (fig. 8, éq. (9)–(16))."
)

st.divider()


# Fiche détaillée par modèle : à quoi ça sert, comment ça fonctionne, sur quoi
# il a été entraîné.

FICHES = {
    "EMS_power_limitation": {
        "famille": "Règle déterministe (modèle physique)",
        "role": (
            "Servir de référence physique et de solution de secours : elle est "
            "toujours calculable, sans apprentissage. C'est l'EMS « power limitation » "
            "de l'article de référence [1] (fig. 10), à laquelle on compare les autres."
        ),
        "fonctionnement": [
            "Règle si-alors : l'EB (batterie d'énergie) fournit la puissance en "
            "priorité, dans la limite de sa puissance maximale.",
            "Dès que la demande dépasse cette limite, la PB (batterie de "
            "puissance) fournit le complément ; si le SOC de l'EB atteint son "
            "minimum, la PB fournit tout.",
            "Aucun paramètre appris : le comportement découle uniquement des "
            "limites physiques des batteries.",
        ],
        "cible": "Aucune : règle fixe.",
        "entrees": "Puissance demandée, SOC de l'EB",
    },
    "EMS_fuzzy_logic": {
        "famille": "Logique floue (inférence Mamdani)",
        "role": (
            "Traduire des connaissances d'expert en une décision continue, sans "
            "réseau de neurones. Sa sortie sert aussi de point de départ au MLP "
            "neurosymbolique."
        ),
        "fonctionnement": [
            "Sept règles pondérées combinent des concepts flous : SOC faible, "
            "forte traction, freinage régénératif, etc.",
            "Ces concepts sont ceux de l'ontologie OntoHESS (seuils de SOC, "
            "conditions de puissance) ; les règles elles-mêmes sont écrites dans le "
            "code, pas lues dans le fichier OWL.",
            "Le moteur d'inférence agrège les règles activées puis défuzzifie "
            "pour produire un alpha continu.",
        ],
        "cible": "Aucune : règles expertes fixées à la main.",
        "entrees": "SOC_EB, SOC_PB, puissance demandée, accélération",
    },
    "EMS_MLP": {
        "famille": "Réseau de neurones dense (perceptron multicouche)",
        "role": (
            "Apprendre directement la décision alpha à partir de l'état "
            "instantané du système. C'est la référence purement neuronale, sans "
            "composante symbolique."
        ),
        "fonctionnement": [
            "Réseau tabulaire à deux couches cachées "
            f"({core.MLP_HIDDEN_1} puis {core.MLP_HIDDEN_2} neurones), activation "
            "ReLU et sortie Sigmoid pour garder alpha entre 0 et 1.",
            "Il regarde un seul instant à la fois : pas de mémoire temporelle.",
            "La décision dépend de l'état courant (SOC des deux batteries, "
            "puissance, vitesse, accélération).",
        ],
        "cible": (
            "alpha* : pour chaque instant du jeu d'entraînement, la répartition qui "
            "minimise un coût physique à cinq termes (stress de puissance, débit "
            "d'énergie, risque SOC, sollicitation du convertisseur, continuité). "
            "Elle est calculée instant par instant, sans anticiper la suite du cycle."
        ),
        "entrees": ", ".join(core.MLP_INPUT_COLS),
    },
    "EMS_MLP_neurosymbolic": {
        "famille": "Neurosymbolique (correction résiduelle de la logique floue)",
        "role": (
            "Corriger la logique floue sans la remplacer : garder une décision "
            "explicable tout en profitant de l'apprentissage."
        ),
        "fonctionnement": [
            "Le réseau ne prédit pas alpha directement, mais une petite "
            "correction delta ajoutée à la sortie de la logique floue.",
            f"La correction est bornée : delta = {core.MLP_NS_MAX_DELTA} × tanh(...), "
            "puis alpha final est ramené dans l'intervalle [0, 1].",
            "La décision se décompose donc exactement en base floue + correction : "
            "c'est la seule stratégie apprise dont on peut lire la part symbolique.",
        ],
        "cible": "La même cible alpha* que le MLP, apprise sous forme de correction de la logique floue.",
        "entrees": (
            "État instantané, sortie de la logique floue, états symboliques de "
            "l'ontologie, et les trois prédictions du LSTM (demande future, variations "
            "de SOC des deux batteries) : 17 entrées, ce qui le rend dépendant du LSTM."
        ),
    },
    "EMS_LSTM": {
        "famille": "Réseau récurrent (mémoire temporelle)",
        "role": (
            "Tenir compte de l'historique récent du cycle plutôt que du seul "
            "instant présent, pour anticiper l'évolution du système."
        ),
        "fonctionnement": [
            f"Le modèle lit une fenêtre glissante des {core.LSTM_WINDOW} derniers "
            "instants du cycle.",
            f"Couche LSTM ({core.LSTM_NUM_LAYERS} couches, {core.LSTM_HIDDEN_SIZE} "
            "unités cachées) suivie d'une tête dense.",
            "Il ne prédit pas alpha : il prédit la demande future et les variations "
            "de SOC des deux batteries. L'application en déduit la répartition alpha, "
            "une interprétation qui reste à confirmer.",
        ],
        "cible": (
            "La demande et les variations de SOC du jeu de données de référence, "
            "lui-même produit par la règle physique."
        ),
        "entrees": ", ".join(core.LSTM_FEATURE_COLS),
    },
    "EMS_LSTM_neurosymbolic": {
        "famille": "Neurosymbolique temporel (états symboliques en entrée)",
        "role": (
            "Donner au LSTM, en plus des grandeurs physiques, les états symboliques "
            "de l'ontologie, pour une décision informée par l'historique et par "
            "les concepts métier."
        ),
        "fonctionnement": [
            f"Même architecture récurrente que le LSTM (fenêtre de "
            f"{core.LSTM_WINDOW} instants).",
            "Les entrées incluent en plus quatre états symboliques déduits de "
            "l'ontologie : forte demande, freinage régénératif, demande nulle, "
            "risque convertisseur.",
            "Pas de base floue ni de correction bornée : le symbolique n'intervient "
            "qu'en entrée, et la répartition est déduite des variations de SOC "
            "prédites, comme pour le LSTM.",
        ],
        "cible": "La même cible que le LSTM.",
        "entrees": ", ".join(core.LSTM_NS_FEATURE_COLS),
    },
    "EMS_GNN": {
        "famille": "Réseau de neurones sur graphe",
        "role": (
            "Représenter explicitement la structure physique du HESS et faire "
            "circuler l'information entre ses composants avant de décider."
        ),
        "fonctionnement": [
            "Le système est un graphe à cinq nœuds : "
            + ", ".join(core.GNN_NODE_NAMES) + ".",
            "Chaque nœud porte des caractéristiques physiques (SOC, courants et "
            "puissances maximales, capacité, demande, accélération).",
            f"{core.GNN_NUM_LAYERS} couches de convolution de graphe (GCNConv) "
            "propagent l'information entre nœuds voisins.",
            "Une agrégation global_mean_pool résume le graphe, puis une tête "
            "dense produit alpha.",
        ],
        "cible": (
            "La répartition de la règle physique : le GNN apprend à l'imiter, ce qui "
            "explique des résultats presque identiques à ceux du modèle physique."
        ),
        "entrees": ", ".join(core.GNN_CONTINUOUS_FEATURE_NAMES),
    },
}


for cle in core.MODEL_ORDER:
    fiche = FICHES.get(cle)
    if fiche is None:
        continue

    with st.container(border=True):
        st.subheader(nom_affichage(cle))
        st.caption(f"{fiche['famille']}  ·  famille « {famille(cle)} »")

        st.markdown("**À quoi ça sert**")
        st.write(fiche["role"])

        st.markdown("**Comment ça fonctionne**")
        st.markdown("\n".join(f"- {point}" for point in fiche["fonctionnement"]))

        st.markdown("**Sur quoi il a été entraîné**")
        st.write(fiche["cible"])

        st.markdown("**Données d'entrée**")
        st.write(fiche["entrees"])


st.divider()


# Rôle de l'ontologie dans la chaîne de décision

st.subheader("Comment intervient l'ontologie ?")

st.dataframe(
    pd.DataFrame(
        [
            {
                "Étape": "Variables physiques",
                "Rôle d'OntoHESS": "Décrit les composants du HESS (batteries, convertisseur, charge) et leurs grandeurs.",
            },
            {
                "Étape": "Concepts métier",
                "Rôle d'OntoHESS": "Définit les seuils et les états de fonctionnement : state_Normal, state_Overload_High, state_Overload_Low.",
            },
            {
                "Étape": "Règles expertes",
                "Rôle d'OntoHESS": "Fournit les règles SWRL comparant puissance et SOC aux seuils déclarés.",
            },
            {
                "Étape": "Modèles neurosymboliques",
                "Rôle d'OntoHESS": "Fournit les états symboliques donnés en entrée aux deux réseaux (et, pour le MLP, la base floue qu'il corrige).",
            },
            {
                "Étape": "Modèle physique",
                "Rôle d'OntoHESS": "Aucun à l'exécution ; ses branches correspondent aux règles SWRL R9 à R17, ce qui permet de le relire a posteriori.",
            },
            {
                "Étape": "Explicabilité",
                "Rôle d'OntoHESS": "Justifie la décision avec des concepts métier compréhensibles et des règles traçables.",
            },
        ]
    ).set_index("Étape"),
    width="stretch",
)

st.caption(
    "Les noms de concepts cités sont ceux réellement déclarés dans "
    "ontologies/OntoHESS2.owl. Voir la page « Base de connaissances » pour le "
    "raisonnement pas à pas."
)


# Synthèse comparative des familles

st.subheader("Comparaison des stratégies")

_NIVEAU_TEXTE = {3: "Par construction", 2: "Partielle", 1: "Post-hoc"}

_COMPARAISON = {
    "EMS_power_limitation": ("Non", "Non", "Non exécutée — branches ≙ règles SWRL R9–R17", "—"),
    "EMS_fuzzy_logic": ("Non", "Non", "Concepts (règles écrites dans le code)", "—"),
    "EMS_MLP": ("Oui", "Non", "Non", "alpha* instantané"),
    "EMS_LSTM": ("Oui", f"Oui — fenêtre de {core.LSTM_WINDOW} instants", "Non", "SOC de la règle physique"),
    "EMS_MLP_neurosymbolic": ("Oui", "Non", "Oui — base floue + états symboliques", "alpha* instantané"),
    "EMS_LSTM_neurosymbolic": ("Oui", f"Oui — fenêtre de {core.LSTM_WINDOW} instants", "Oui — 4 états symboliques en entrée", "SOC de la règle physique"),
    "EMS_GNN": ("Oui", "Structure du graphe", "Partiel — topologie du HESS", "Règle physique (imitation)"),
}

_lignes_comp = []
for cle in core.MODEL_ORDER:
    if cle not in _COMPARAISON:
        continue
    apprentissage, memoire, ontologie, cible = _COMPARAISON[cle]
    niveau, _ = EXPLICABILITE.get(cle, (0, ""))
    _lignes_comp.append(
        {
            "Stratégie": nom_affichage(cle),
            "Famille": famille(cle),
            "Apprentissage": apprentissage,
            "Cible d'entraînement": cible,
            "Mémoire temporelle": memoire,
            "Ontologie": ontologie,
            "Explicabilité": _NIVEAU_TEXTE.get(niveau, "—"),
        }
    )

st.dataframe(pd.DataFrame(_lignes_comp).set_index("Stratégie"), width="stretch")

st.caption(
    "Les modèles neuronaux ont été entraînés hors ligne ; l'application charge "
    "leurs poids et rejoue leurs décisions. Seul le MLP neurosymbolique réutilise la "
    "logique floue comme socle ; le LSTM neurosymbolique ajoute des états "
    "symboliques à ses entrées."
)


from core.navigation import pied_navigation

pied_navigation("vues/9_Architecture_des_modeles.py")
