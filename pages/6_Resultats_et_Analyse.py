
import sys
from pathlib import Path

DOSSIER_PROJET = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DOSSIER_PROJET))

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from ems_core import MODEL_DISPLAY_NAMES


# ============================================================
# Configuration de la page
# ============================================================

st.set_page_config(
    page_title="2SMART — Résultats & Analyse",
    layout="wide",
)

st.title("Résultats & Analyse")


# ============================================================
# Données : résultats de RÉFÉRENCE précalculés (chargés via le pont),
# ou résultats d'une simulation de cycle personnalisé si elle existe.
# Aucune simulation lourde n'est relancée ici.
# ============================================================

from core.resultats import assurer_donnees_session

try:
    _source = assurer_donnees_session(st)
except FileNotFoundError as exc:
    st.error(str(exc))
    st.info("Lance une fois le précalcul :  `python scripts/run_simulations.py`")
    st.stop()

st.caption(f"Source des données : {_source}")

if (
    "resultats_simulation" not in st.session_state
    or "cycle_pret" not in st.session_state
):
    st.warning("Aucune donnée disponible.")
    st.stop()


resultats = st.session_state["resultats_simulation"]
df = st.session_state["cycle_pret"]

if not resultats:
    st.warning(
        "Aucune stratégie n’a produit de résultat exploitable."
    )
    st.stop()


st.write(
    "Cette page compare l’évolution des états de charge, des puissances et "
    "des courants pour les stratégies EMS simulées. Les SOC peuvent être "
    "affichés pour toutes les stratégies, tandis que les puissances et les "
    "courants sont présentés sur une sélection réduite afin de garder les "
    "graphes lisibles."
)


# ============================================================
# Configuration visuelle des stratégies
# ============================================================

ORDRE_TRACE = [
    # Arrière-plan
    "EMS_GNN",
    "EMS_LSTM",
    "EMS_MLP",

    # Milieu
    "EMS_MLP_neurosymbolic",
    "EMS_LSTM_neurosymbolic",

    # Avant-plan
    "EMS_fuzzy_logic",
    "EMS_power_limitation",
]

COULEURS_STRATEGIES = {
    "EMS_power_limitation": "#1f77b4",
    "EMS_fuzzy_logic": "#ff7f0e",
    "EMS_MLP": "#2ca02c",
    "EMS_LSTM": "#d62728",
    "EMS_MLP_neurosymbolic": "#9467bd",
    "EMS_LSTM_neurosymbolic": "#e377c2",
    "EMS_GNN": "#8c564b",
}

STRATEGIES_ARRIERE_PLAN = {
    "EMS_GNN",
    "EMS_LSTM",
    "EMS_MLP",
}

STRATEGIES_AVANT_PLAN = {
    "EMS_power_limitation",
    "EMS_fuzzy_logic",
}


def nom_affiche(nom):
    """Retourne le nom lisible de la stratégie."""
    return MODEL_DISPLAY_NAMES.get(nom, nom)


def ordonner_strategies(strategies):
    """
    Ordonne les stratégies pour tracer d'abord les courbes les plus couvrantes.
    Les stratégies tracées en premier apparaissent en arrière-plan.
    """
    strategies_ordonnees = [
        nom for nom in ORDRE_TRACE
        if nom in strategies
    ]

    for nom in strategies:
        if nom not in strategies_ordonnees:
            strategies_ordonnees.append(nom)

    return strategies_ordonnees


def style_courbe(nom, rang):
    """
    Définit le style graphique selon la stratégie.
    Les stratégies les plus couvrantes sont plus transparentes et en arrière-plan.
    """
    if nom in STRATEGIES_ARRIERE_PLAN:
        return {
            "color": COULEURS_STRATEGIES.get(nom, None),
            "linewidth": 1.7,
            "alpha": 0.58,
            "zorder": 1 + rang,
        }

    if nom in STRATEGIES_AVANT_PLAN:
        return {
            "color": COULEURS_STRATEGIES.get(nom, None),
            "linewidth": 2.4,
            "alpha": 1.0,
            "zorder": 20 + rang,
        }

    return {
        "color": COULEURS_STRATEGIES.get(nom, None),
        "linewidth": 2.1,
        "alpha": 0.90,
        "zorder": 10 + rang,
    }


def obtenir_temps(n):
    """
    Retourne l’axe temporel.
    Si la colonne time existe, elle est utilisée. Sinon, on utilise l’indice.
    """
    if "time" in df.columns:
        temps = df["time"].to_numpy()
    else:
        temps = np.arange(len(df))

    return temps[:n]


def tracer_signal(ax, nom, x, y, rang):
    """
    Trace un signal en appliquant le style associé à la stratégie.
    """
    style = style_courbe(nom, rang)

    ax.plot(
        x,
        y,
        label=nom_affiche(nom),
        color=style["color"],
        linewidth=style["linewidth"],
        alpha=style["alpha"],
        zorder=style["zorder"],
    )


def strategies_defaut_detail(strategies_disponibles):
    """
    Définit les stratégies affichées par défaut pour les puissances et courants.
    On privilégie les références lisibles.
    """
    preferences = [
        "EMS_power_limitation",
        "EMS_fuzzy_logic",
        "EMS_MLP_neurosymbolic",
    ]

    selection = [
        nom for nom in preferences
        if nom in strategies_disponibles
    ]

    if selection:
        return selection

    return strategies_disponibles[: min(2, len(strategies_disponibles))]


# ============================================================
# Sélection des stratégies
# ============================================================

strategies_disponibles = ordonner_strategies(
    list(resultats.keys())
)

st.subheader("Sélection des stratégies")

strategies_soc = st.multiselect(
    "Stratégies à afficher pour les SOC",
    strategies_disponibles,
    default=strategies_disponibles,
    format_func=nom_affiche,
)

strategies_detaillees = st.multiselect(
    "Stratégies à afficher pour les puissances et les courants",
    strategies_disponibles,
    default=strategies_defaut_detail(strategies_disponibles),
    format_func=nom_affiche,
    help=(
        "Pour les puissances et les courants, il est conseillé de sélectionner "
        "1 à 3 stratégies maximum afin d’éviter des graphes trop chargés."
    ),
)


if not strategies_soc:
    st.info(
        "Sélectionne au moins une stratégie pour afficher l’évolution des SOC."
    )
    st.stop()


strategies_soc = ordonner_strategies(strategies_soc)
strategies_detaillees = ordonner_strategies(strategies_detaillees)


# ============================================================
# 1. Évolution des SOC
# ============================================================

st.subheader("1. Évolution des SOC de l’EB et de la PB")

fig_soc, axes_soc = plt.subplots(
    1,
    2,
    figsize=(13.5, 4.8),
    sharex=True,
)

for rang, nom in enumerate(strategies_soc):
    traj = resultats[nom]

    soc_eb = np.asarray(traj["SOC_EB"], dtype=float)[:-1]
    soc_pb = np.asarray(traj["SOC_PB"], dtype=float)[:-1]

    n = min(len(soc_eb), len(soc_pb), len(df))
    temps = obtenir_temps(n)

    tracer_signal(
        axes_soc[0],
        nom,
        temps,
        soc_eb[:n],
        rang,
    )

    tracer_signal(
        axes_soc[1],
        nom,
        temps,
        soc_pb[:n],
        rang,
    )


axes_soc[0].set_title("Évolution du SOC de l’EB")
axes_soc[0].set_xlabel("Temps (s)")
axes_soc[0].set_ylabel("SOC_EB")
axes_soc[0].grid(True, alpha=0.3)
axes_soc[0].set_ylim(0.15, 1.05)

axes_soc[1].set_title("Évolution du SOC de la PB")
axes_soc[1].set_xlabel("Temps (s)")
axes_soc[1].set_ylabel("SOC_PB")
axes_soc[1].grid(True, alpha=0.3)
axes_soc[1].set_ylim(0.15, 1.05)

axes_soc[1].legend(
    loc="upper right",
    fontsize=8,
    framealpha=0.85,
)

plt.tight_layout()
st.pyplot(fig_soc)
plt.close(fig_soc)


# ============================================================
# 2. Évolution des puissances
# ============================================================

st.subheader("2. Répartition des puissances")

if not strategies_detaillees:
    st.info(
        "Sélectionne au moins une stratégie pour afficher les puissances."
    )

else:
    fig_p, axes_p = plt.subplots(
        1,
        2,
        figsize=(13.5, 4.8),
        sharex=True,
    )

    for rang, nom in enumerate(strategies_detaillees):
        traj = resultats[nom]

        p_eb = np.asarray(traj["P_EB"], dtype=float)
        p_pb = np.asarray(traj["P_PB"], dtype=float)

        n = min(len(p_eb), len(p_pb), len(df))
        temps = obtenir_temps(n)

        tracer_signal(
            axes_p[0],
            nom,
            temps,
            p_eb[:n],
            rang,
        )

        tracer_signal(
            axes_p[1],
            nom,
            temps,
            p_pb[:n],
            rang,
        )

    axes_p[0].set_title("Puissance fournie par l’EB")
    axes_p[0].set_xlabel("Temps (s)")
    axes_p[0].set_ylabel("P_EB (W)")
    axes_p[0].grid(True, alpha=0.3)

    axes_p[1].set_title("Puissance fournie par la PB")
    axes_p[1].set_xlabel("Temps (s)")
    axes_p[1].set_ylabel("P_PB (W)")
    axes_p[1].grid(True, alpha=0.3)

    axes_p[1].legend(
        loc="upper right",
        fontsize=8,
        framealpha=0.85,
    )

    plt.tight_layout()
    st.pyplot(fig_p)
    plt.close(fig_p)


# ============================================================
# 3. Évolution des courants
# ============================================================

st.subheader("3. Évolution des courants")

if not strategies_detaillees:
    st.info(
        "Sélectionne au moins une stratégie pour afficher les courants."
    )

else:
    fig_i, axes_i = plt.subplots(
        1,
        2,
        figsize=(13.5, 4.8),
        sharex=True,
    )

    for rang, nom in enumerate(strategies_detaillees):
        traj = resultats[nom]

        i_eb = np.asarray(traj["I_EB"], dtype=float)
        i_pb = np.asarray(traj["I_PB"], dtype=float)

        n = min(len(i_eb), len(i_pb), len(df))
        temps = obtenir_temps(n)

        tracer_signal(
            axes_i[0],
            nom,
            temps,
            i_eb[:n],
            rang,
        )

        tracer_signal(
            axes_i[1],
            nom,
            temps,
            i_pb[:n],
            rang,
        )

    axes_i[0].set_title("Courant de l’EB")
    axes_i[0].set_xlabel("Temps (s)")
    axes_i[0].set_ylabel("I_EB (A)")
    axes_i[0].grid(True, alpha=0.3)

    axes_i[1].set_title("Courant de la PB")
    axes_i[1].set_xlabel("Temps (s)")
    axes_i[1].set_ylabel("I_PB (A)")
    axes_i[1].grid(True, alpha=0.3)

    axes_i[1].legend(
        loc="upper right",
        fontsize=8,
        framealpha=0.85,
    )

    plt.tight_layout()
    st.pyplot(fig_i)
    plt.close(fig_i)


# ============================================================
# 4. Lecture rapide
# ============================================================

st.subheader("4. Lecture rapide")

st.info(
    "Les SOC peuvent être affichés pour toutes les stratégies afin de comparer "
    "le comportement global du HESS. En revanche, les puissances et les courants "
    "sont volontairement affichés sur une sélection réduite, car ces signaux sont "
    "plus rapides, plus variables et deviennent difficiles à lire lorsque trop de "
    "modèles sont superposés."
)


# ============================================================
# Navigation vers l’analyse instantanée
# ============================================================

st.divider()

if st.button(
    "Consulter l’analyse instantanée",
    type="primary",
):
    st.switch_page(
        "pages/4_Moteur_Neurosymbolique.py"
    )