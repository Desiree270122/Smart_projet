
import sys
from pathlib import Path

DOSSIER_PROJET = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DOSSIER_PROJET))

import numpy as np
import plotly.graph_objects as go
import pandas as pd
import streamlit as st

from ems_core import MODEL_DISPLAY_NAMES


# Configuration de page gérée par le routeur Accueil.py.

st.title("Moteur Neuro-Symbolique — analyse à l'instant t")


# Données : résultats de RÉFÉRENCE précalculés (via le pont), ou
# simulation de cycle personnalisé si elle existe. Aucun calcul lourd ici.

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
        "Aucune stratégie n'a produit de résultat exploitable."
    )
    st.stop()


# Sélection de l'instant à analyser

nombre_points_max = min(
    len(traj["P_EB"])
    for traj in resultats.values()
)

instant_choisi = st.slider(
    "Instant du cycle à analyser",
    0,
    nombre_points_max - 1,
    nombre_points_max // 2,
)

strategie_graphe = st.selectbox(
    "Stratégie de référence pour le graphe (P_EB, P_PB, alpha affichés au survol)",
    list(resultats.keys()),
    format_func=lambda n: n,
)


ligne = df.iloc[instant_choisi]


st.write(
    f"**Temps : {float(ligne['time']):.0f} s** — "
    f"Puissance demandée : {float(ligne['hasPower']) / 1000:.2f} kW"
)


# Courbe de la puissance demandée sur tout le cycle

st.subheader("Puissance demandée sur le cycle")

st.caption(
    "Déplace le curseur ci-dessus pour changer l'instant (repère rouge). "
    "Survole la courbe pour lire, à n'importe quel point (y compris les pics) : "
    "P_dem, ainsi que P_EB, P_PB et alpha de la stratégie choisie. Zoom possible."
)

_t = df["time"].to_numpy()[:nombre_points_max]
_p = df["hasPower"].to_numpy()[:nombre_points_max] / 1000.0
_t_sel = float(ligne["time"])
_p_sel = float(ligne["hasPower"]) / 1000.0

# Données de la stratégie choisie, pour enrichir le survol (P_EB, P_PB, alpha).
_traj_g = resultats[strategie_graphe]
_custom = np.column_stack(
    [
        np.asarray(_traj_g["P_EB"], dtype=float)[:nombre_points_max] / 1000.0,
        np.asarray(_traj_g["P_PB"], dtype=float)[:nombre_points_max] / 1000.0,
        np.asarray(_traj_g["alpha_final"], dtype=float)[:nombre_points_max],
    ]
)

fig_puissance = go.Figure()
fig_puissance.add_trace(
    go.Scatter(
        x=_t,
        y=_p,
        mode="lines",
        line=dict(color="#5B8DEF", width=1),
        name="Puissance demandée",
        customdata=_custom,
        hovertemplate=(
            "Temps : %{x:.0f} s<br>P_dem : %{y:.2f} kW<br>"
            "P_EB : %{customdata[0]:.2f} kW<br>"
            "P_PB : %{customdata[1]:.2f} kW<br>"
            "alpha : %{customdata[2]:.2f}<extra></extra>"
        ),
    )
)
fig_puissance.add_trace(
    go.Scatter(
        x=[_t_sel],
        y=[_p_sel],
        mode="markers",
        marker=dict(color="#E5484D", size=11),
        name=f"Instant sélectionné ({_t_sel:.0f} s)",
        hovertemplate="Instant sélectionné<br>Temps : %{x:.0f} s<br>Puissance : %{y:.2f} kW<extra></extra>",
    )
)
fig_puissance.add_vline(x=_t_sel, line=dict(color="#E5484D", dash="dash", width=1.5))
fig_puissance.update_layout(
    xaxis_title="Temps (s)",
    yaxis_title="Puissance demandée (kW)",
    height=380,
    hovermode="closest",
    margin=dict(t=30, b=40),
    legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1.0),
)
st.plotly_chart(fig_puissance, use_container_width=True)


# Construction du tableau comparatif

st.subheader("Détail par stratégie à cet instant")

lignes_tableau = []

for nom, traj in resultats.items():
    lignes_tableau.append(
        {
            "Stratégie": nom,
            "Alpha appliqué": float(
                traj["alpha_final"][instant_choisi]
            ),
            "P_EB (kW)": float(
                traj["P_EB"][instant_choisi]
            ) / 1000,
            "P_PB (kW)": float(
                traj["P_PB"][instant_choisi]
            ) / 1000,
            "I_EB (A)": float(
                traj["I_EB"][instant_choisi]
            ),
            "I_PB (A)": float(
                traj["I_PB"][instant_choisi]
            ),
            "SOC_EB": float(
                traj["SOC_EB"][instant_choisi]
            ),
            "SOC_PB": float(
                traj["SOC_PB"][instant_choisi]
            ),
            "Correction appliquée": (
                "Oui"
                if bool(
                    traj["correction_applied"][instant_choisi]
                )
                else "Non"
            ),
        }
    )


st.dataframe(
    pd.DataFrame(lignes_tableau).round(4),
    use_container_width=True,
    hide_index=True,
)


# Navigation vers la comparaison des stratégies

from core.navigation import pied_navigation

pied_navigation("vues/4_Moteur_Neurosymbolique.py")