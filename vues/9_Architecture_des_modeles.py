import sys
from pathlib import Path

DOSSIER_PROJET = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DOSSIER_PROJET))

import streamlit as st

import ems_core as core
from core.resultats import nom_affichage


# Configuration de page gérée par le routeur Accueil.py.

st.title("Architecture des modèles")

st.markdown(
    "*Les sept stratégies décident toutes de la même variable : le coefficient "
    "`alpha(t)`, fraction de la puissance demandée confiée à la batterie de "
    "puissance. Elles diffèrent par la façon dont elles calculent cette valeur.*"
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
            "Les concepts ont été formalisés hors ligne à partir de l'ontologie "
            "OntoHESS.",
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
            "On garde ainsi la lisibilité de la règle floue, ajustée à la marge "
            "par le réseau.",
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
            "Les entrées incluent en plus des états symboliques : forte demande, "
            "freinage régénératif, demande nulle, risque convertisseur.",
            "La prédiction temporelle est corrigée par la composante symbolique "
            "issue de la logique floue.",
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

st.caption(
    "Les modèles neuronaux ont été entraînés hors ligne ; l'application charge "
    "leurs poids et rejoue leurs décisions. Les variantes neurosymboliques "
    "réutilisent la logique floue comme socle interprétable."
)


from core.navigation import pied_navigation

pied_navigation("vues/9_Architecture_des_modeles.py")
