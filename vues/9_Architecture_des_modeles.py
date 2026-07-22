import sys
from pathlib import Path

DOSSIER_PROJET = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DOSSIER_PROJET))

import pandas as pd
import streamlit as st

import ems_core as core
from core.resultats import nom_affichage, EXPLICABILITE


# Configuration de page gérée par le routeur Accueil.py.

st.title("🧠 Les modèles d'IA")

st.markdown(
    "*Toutes les stratégies poursuivent le même objectif : déterminer le coefficient "
    "`alpha(t)` qui répartit la puissance demandée entre la batterie Énergie et la "
    "batterie Puissance. Elles emploient des mécanismes décisionnels différents — "
    "physiques, symboliques, neuronaux ou neuro-symboliques — tout en s'appuyant sur "
    "les mêmes connaissances métier issues de l'ontologie OntoHESS.*"
)


# Architecture globale : situer l'ontologie AVANT les modèles

st.subheader("Architecture globale de décision")

_ETAPES_ARCHI = [
    ("Cycle de conduite", "#6B7280"),
    ("Variables physiques", "#6B7280"),
    ("Ontologie OntoHESS", "#F59E0B"),
    ("Connaissances métier", "#F59E0B"),
    ("Règles expertes", "#F59E0B"),
    ("Modèle EMS", "#3B82F6"),
    ("Filtre physique", "#22C55E"),
    ("Répartition EB / PB", "#22C55E"),
]

_flux = "<div style='display:flex;align-items:center;flex-wrap:wrap;gap:6px;margin:.3rem 0'>"
for _i, (_etape, _coul) in enumerate(_ETAPES_ARCHI):
    _flux += (
        f"<span style='border:1px solid {_coul};color:{_coul};border-radius:9px;"
        f"padding:6px 11px;font-weight:600;font-size:.86rem'>{_etape}</span>"
    )
    if _i < len(_ETAPES_ARCHI) - 1:
        _flux += "<span style='color:#94A3B8;font-weight:800'>&#8594;</span>"
_flux += "</div>"
st.markdown(_flux, unsafe_allow_html=True)

st.caption(
    "L'ontologie intervient **avant** les modèles : elle fournit les concepts métier "
    "et les règles sur lesquels s'appuient la logique floue et les variantes "
    "neuro-symboliques."
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

st.divider()


# Fiche détaillée par modèle : à quoi ça sert, comment ça fonctionne.
# Le résumé « comment ça fonctionne » et le rôle proviennent d'ems_core
# (MODEL_CONSTRUCTION_SUMMARY) ; on ajoute ici les détails d'architecture.

FICHES = {
    "EMS_power_limitation": {
        "famille": "Règle déterministe (modèle physique)",
        "role": (
            "Servir de référence physique et de solution de secours : elle est "
            "toujours calculable, sans apprentissage. C'est la stratégie à "
            "laquelle on compare toutes les autres."
        ),
        "fonctionnement": [
            "Règle si-alors : l'EB (batterie d'énergie) fournit la puissance en "
            "priorité, dans la limite de sa puissance maximale.",
            "Dès que la demande dépasse cette limite, la PB (batterie de "
            "puissance) fournit le complément.",
            "Aucun paramètre appris : le comportement découle uniquement des "
            "limites physiques des batteries.",
        ],
        "entrees": "Puissance demandée, SOC de l'EB",
    },
    "EMS_fuzzy_logic": {
        "famille": "Logique floue (inférence Mamdani)",
        "role": (
            "Traduire des connaissances d'expert en une décision continue, sans "
            "réseau de neurones. Sa sortie sert aussi de point de départ aux deux "
            "variantes neurosymboliques."
        ),
        "fonctionnement": [
            "Sept règles pondérées combinent des concepts flous : SOC faible, "
            "forte traction, freinage régénératif, etc.",
            "L'ontologie OntoHESS formalise ces concepts métier (seuils de SOC, "
            "conditions de surcharge, états de fonctionnement) et les grandeurs "
            "qu'ils comparent ; ils alimentent directement les règles floues du "
            "moteur d'inférence. La logique floue ne sort donc pas de nulle part.",
            "Le moteur d'inférence agrège les règles activées puis défuzzifie "
            "pour produire un alpha continu.",
        ],
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
        "entrees": ", ".join(core.MLP_INPUT_COLS),
    },
    "EMS_MLP_neurosymbolic": {
        "famille": "Neurosymbolique (correction résiduelle)",
        "role": (
            "Corriger la logique floue sans la remplacer : garder une décision "
            "explicable tout en profitant de l'apprentissage."
        ),
        "fonctionnement": [
            "Le réseau ne prédit pas alpha directement, mais une petite "
            "correction delta ajoutée à la sortie de la logique floue.",
            f"La correction est bornée : delta = {core.MLP_NS_MAX_DELTA} × tanh(...), "
            "puis alpha final est ramené dans l'intervalle [0, 1].",
            "Les états symboliques reçus en entrée sont déduits des concepts définis "
            "dans l'ontologie OntoHESS. Le réseau apprend donc uniquement à ajuster "
            "une décision déjà cohérente avec les connaissances expertes.",
        ],
        "entrees": (
            "État instantané, sortie de la logique floue, états symboliques et "
            "sorties du LSTM (17 entrées au total)"
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
            "Il est entraîné à prédire des variations de SOC (plutôt que des "
            "valeurs absolues), ce qui limite le décalage entre entraînement et "
            "test, et propose aussi une valeur d'alpha.",
        ],
        "entrees": ", ".join(core.LSTM_FEATURE_COLS),
    },
    "EMS_LSTM_neurosymbolic": {
        "famille": "Neurosymbolique temporel",
        "role": (
            "Combiner l'anticipation temporelle du LSTM avec la connaissance "
            "symbolique, pour une décision à la fois informée par l'historique et "
            "encadrée par les règles."
        ),
        "fonctionnement": [
            f"Même architecture récurrente que le LSTM (fenêtre de "
            f"{core.LSTM_WINDOW} instants).",
            "Les entrées incluent en plus des états symboliques déduits de "
            "l'ontologie : forte demande, freinage régénératif, demande nulle, "
            "risque convertisseur.",
            "La prédiction temporelle est corrigée par la composante symbolique "
            "issue de la logique floue, elle-même fondée sur OntoHESS.",
        ],
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
        "entrees": ", ".join(core.GNN_CONTINUOUS_FEATURE_NAMES),
    },
}


for cle in core.MODEL_ORDER:
    fiche = FICHES.get(cle)
    if fiche is None:
        continue

    with st.container(border=True):
        st.subheader(nom_affichage(cle))
        st.caption(fiche["famille"])

        st.markdown("**À quoi ça sert**")
        st.write(fiche["role"])

        st.markdown("**Comment ça fonctionne**")
        st.markdown("\n".join(f"- {point}" for point in fiche["fonctionnement"]))

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
                "Étape": "Modèles neuro-symboliques",
                "Rôle d'OntoHESS": "Génère les états symboliques utilisés comme entrées supplémentaires du réseau.",
            },
            {
                "Étape": "Explicabilité",
                "Rôle d'OntoHESS": "Justifie la décision avec des concepts métier compréhensibles et des règles traçables.",
            },
        ]
    ).set_index("Étape"),
    use_container_width=True,
)

st.caption(
    "Les noms de concepts cités sont ceux réellement déclarés dans "
    "ontologies/OntoHESS2.owl. Voir la page « Base de connaissances » pour le "
    "raisonnement pas à pas."
)


# Synthèse comparative des familles

st.subheader("Comparaison des familles de stratégies")

_NIVEAU_TEXTE = {3: "Par construction", 2: "Partielle", 1: "Post-hoc"}

_COMPARAISON = {
    "EMS_power_limitation": ("Non", "Non", "Oui — règles SWRL de l'OWL"),
    "EMS_fuzzy_logic": ("Non", "Non", "Oui — concepts et seuils"),
    "EMS_MLP": ("Oui", "Non", "Non"),
    "EMS_LSTM": ("Oui", f"Oui — fenêtre de {core.LSTM_WINDOW} instants", "Non"),
    "EMS_MLP_neurosymbolic": ("Oui", "Non", "Oui — états symboliques"),
    "EMS_LSTM_neurosymbolic": ("Oui", f"Oui — fenêtre de {core.LSTM_WINDOW} instants", "Oui — états symboliques"),
    "EMS_GNN": ("Oui", "Structure du graphe", "Partiel — topologie du HESS"),
}

_lignes_comp = []
for cle in core.MODEL_ORDER:
    if cle not in _COMPARAISON:
        continue
    apprentissage, memoire, ontologie = _COMPARAISON[cle]
    niveau, _ = EXPLICABILITE.get(cle, (0, ""))
    _lignes_comp.append(
        {
            "Stratégie": nom_affichage(cle),
            "Apprentissage": apprentissage,
            "Mémoire temporelle": memoire,
            "Ontologie": ontologie,
            "Explicabilité": _NIVEAU_TEXTE.get(niveau, "—"),
        }
    )

st.dataframe(pd.DataFrame(_lignes_comp).set_index("Stratégie"), use_container_width=True)

st.info(
    "Le modèle physique n'est pas dépourvu d'ontologie : ses branches de décision "
    "(l'EB fournit seule, la PB complète, l'EB est protégée) correspondent aux règles "
    "SWRL déclarées dans OntoHESS2.owl. C'est vérifiable dans la page "
    "« Pourquoi cette décision ? »."
)

st.caption(
    "Les modèles neuronaux ont été entraînés hors ligne ; l'application charge "
    "leurs poids et rejoue leurs décisions. Les variantes neuro-symboliques "
    "réutilisent la logique floue comme socle interprétable."
)


from core.navigation import pied_navigation

pied_navigation("vues/9_Architecture_des_modeles.py")
