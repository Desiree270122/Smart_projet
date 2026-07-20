"""
Page « Résultats & Analyse » — orientée analyse scientifique.

Au lieu d'afficher 14 000 points bruts (illisibles), on résume chaque stratégie
par des indicateurs (KPI), on garde les courbes de SOC (parlantes) en version
interactive, et on remplace les séries de puissance/courant par des boxplots +
des analyses automatiques. Le lecteur n'a jamais à interpréter seul.
"""

import sys
from pathlib import Path

DOSSIER_PROJET = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DOSSIER_PROJET))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ems_core import MODEL_DISPLAY_NAMES
from core.resultats import (
    assurer_donnees_session,
    calculer_metriques,
    statistiques_detaillees,
    nom_affichage,
)
from core.navigation import pied_navigation


# Configuration de page gérée par le routeur Accueil.py.

st.title("Résultats et Analyse")


# Données (résultats précalculés via le pont)

try:
    _source = assurer_donnees_session(st)
except FileNotFoundError as exc:
    st.error(str(exc))
    st.info("Lance une fois le précalcul :  `python scripts/run_simulations.py`")
    st.stop()

st.caption(f"Source des données : {_source}")

if "resultats_simulation" not in st.session_state or "cycle_pret" not in st.session_state:
    st.warning("Aucune donnée disponible.")
    st.stop()

resultats = st.session_state["resultats_simulation"]
df = st.session_state["cycle_pret"]
donnees = {"resultats": resultats, "cycle_df": df}

if not resultats:
    st.warning("Aucune stratégie n'a produit de résultat exploitable.")
    st.stop()

stats = statistiques_detaillees(donnees)
metriques = calculer_metriques(donnees)
noms = list(resultats.keys())


def _meilleur(cle, sens="max"):
    """Stratégie optimisant une statistique (max = plus haut, min = plus bas)."""
    paires = [(n, stats[n][cle]) for n in noms]
    return (max if sens == "max" else min)(paires, key=lambda kv: kv[1])[0]


# 1. Tableau de bord (KPI)

st.header("1. Tableau de bord")

tableau = pd.DataFrame(
    {
        "Stratégie": [nom_affichage(n) for n in noms],
        "SOC_EB final (%)": [stats[n]["soc_eb_final"] * 100 for n in noms],
        "SOC_PB final (%)": [stats[n]["soc_pb_final"] * 100 for n in noms],
        "Énergie EB (Wh)": [stats[n]["energie_eb_wh"] for n in noms],
        "Énergie PB (Wh)": [stats[n]["energie_pb_wh"] for n in noms],
        "I_EB RMS (A)": [stats[n]["i_eb_rms"] for n in noms],
        "I_PB RMS (A)": [stats[n]["i_pb_rms"] for n in noms],
        "P_EB max (kW)": [stats[n]["p_eb_max"] / 1000 for n in noms],
        "P_PB max (kW)": [stats[n]["p_pb_max"] / 1000 for n in noms],
        "Violations SOC": [metriques[n]["nb_violations"] for n in noms],
    }
).set_index("Stratégie")

st.dataframe(tableau.style.format("{:.1f}"), use_container_width=True)

r1, r2, r3 = st.columns(3)
r1.metric("Préserve le mieux l'EB", nom_affichage(_meilleur("soc_eb_final", "max")))
r2.metric("Préserve le mieux la PB", nom_affichage(_meilleur("soc_pb_final", "max")))
r3.metric("Pics de courant PB les plus faibles", nom_affichage(_meilleur("i_pb_max", "min")))


# 2. Évolution des SOC (interactif)

st.header("2. Évolution des états de charge (SOC)")


def _courbe_soc(cle_soc, titre):
    fig = go.Figure()
    for n in noms:
        y = np.asarray(resultats[n][cle_soc], dtype=float) * 100.0
        x = np.arange(len(y))
        pas = max(1, len(y) // 2000)  # sous-échantillonnage pour rester fluide
        fig.add_trace(
            go.Scatter(x=x[::pas], y=y[::pas], mode="lines", name=nom_affichage(n))
        )
    fig.update_layout(
        title=titre,
        xaxis_title="Temps (s)",
        yaxis_title="SOC (%)",
        height=420,
        legend_title="Stratégie",
        margin=dict(t=50, b=40),
    )
    return fig


col_eb, col_pb = st.columns(2)
with col_eb:
    st.plotly_chart(_courbe_soc("SOC_EB", "SOC batterie Énergie"), use_container_width=True)
with col_pb:
    st.plotly_chart(_courbe_soc("SOC_PB", "SOC batterie Puissance"), use_container_width=True)

meilleur_eb = _meilleur("soc_eb_final", "max")
st.info(
    f"**Analyse automatique.** SOC_EB final le plus élevé : "
    f"**{nom_affichage(meilleur_eb)}** ({stats[meilleur_eb]['soc_eb_final'] * 100:.0f} %). "
    "Les stratégies neuro-symboliques tendent à mieux préserver la batterie "
    "d'énergie, tandis que les stratégies déterministes la sollicitent davantage "
    "en fin de cycle."
)


# 3. Répartition des puissances (boxplots)

st.header("3. Répartition des puissances")


def _boxplot(cle_p, titre):
    fig = go.Figure()
    for n in noms:
        y = np.asarray(resultats[n][cle_p], dtype=float) / 1000.0
        fig.add_trace(go.Box(y=y, name=nom_affichage(n), boxpoints=False))
    fig.update_layout(
        title=titre,
        yaxis_title="Puissance (kW)",
        height=420,
        showlegend=False,
        margin=dict(t=50, b=40),
    )
    return fig


col_peb, col_ppb = st.columns(2)
with col_peb:
    st.plotly_chart(_boxplot("P_EB", "Puissance batterie Énergie"), use_container_width=True)
with col_ppb:
    st.plotly_chart(_boxplot("P_PB", "Puissance batterie Puissance"), use_container_width=True)

pb_max_strat = _meilleur("p_pb_max", "max")
st.info(
    "**Analyse automatique.** La médiane et la dispersion montrent quelle "
    "stratégie sollicite le plus chaque batterie. Pic de puissance PB le plus "
    f"élevé : **{nom_affichage(pb_max_strat)}** ({stats[pb_max_strat]['p_pb_max'] / 1000:.1f} kW)."
)


# 4. Sollicitation électrique (courants)

st.header("4. Sollicitation électrique (courants)")


def _boxplot_courant(cle_i, titre):
    fig = go.Figure()
    for n in noms:
        y = np.asarray(resultats[n][cle_i], dtype=float)
        fig.add_trace(go.Box(y=y, name=nom_affichage(n), boxpoints=False))
    fig.update_layout(
        title=titre,
        yaxis_title="Courant (A)",
        height=420,
        showlegend=False,
        margin=dict(t=50, b=40),
    )
    return fig


col_ieb, col_ipb = st.columns(2)
with col_ieb:
    st.plotly_chart(_boxplot_courant("I_EB", "Courant batterie Énergie"), use_container_width=True)
with col_ipb:
    st.plotly_chart(_boxplot_courant("I_PB", "Courant batterie Puissance"), use_container_width=True)

ipb_rms_strat = _meilleur("i_pb_rms", "min")
ipb_max_strat = _meilleur("i_pb_max", "max")
st.info(
    "**Analyse automatique.** Courant RMS PB le plus faible (le plus régulier) : "
    f"**{nom_affichage(ipb_rms_strat)}**. Pic de courant PB le plus élevé : "
    f"**{nom_affichage(ipb_max_strat)}** ({stats[ipb_max_strat]['i_pb_max']:.0f} A)."
)


# 5. Comparaison des stratégies (scores normalisés)

st.header("5. Comparaison des stratégies")

# axes : (libellé, clé, sens) — sens "max" = plus haut = mieux, "min" = plus bas = mieux
AXES = [
    ("SOC EB préservé", "soc_eb_final", "max"),
    ("SOC PB préservé", "soc_pb_final", "max"),
    ("Pics courant PB faibles", "i_pb_max", "min"),
    ("Stabilité courant PB", "i_pb_rms", "min"),
]


def _score(valeurs, v, sens):
    lo, hi = min(valeurs), max(valeurs)
    if hi - lo < 1e-12:
        return 5.0
    x = (v - lo) / (hi - lo)
    return round(5.0 * (x if sens == "max" else (1.0 - x)), 1)


scores = {}
for libelle, cle, sens in AXES:
    vals = [stats[n][cle] for n in noms]
    scores[libelle] = [_score(vals, stats[n][cle], sens) for n in noms]

tableau_scores = pd.DataFrame(scores, index=[nom_affichage(n) for n in noms])
tableau_scores["Score global"] = tableau_scores.mean(axis=1).round(1)
st.dataframe(tableau_scores.style.format("{:.1f}"), use_container_width=True)

fig_radar = go.Figure()
for i, n in enumerate(noms):
    fig_radar.add_trace(
        go.Scatterpolar(
            r=[scores[lib][i] for lib, _, _ in AXES],
            theta=[lib for lib, _, _ in AXES],
            fill="toself",
            name=nom_affichage(n),
        )
    )
fig_radar.update_layout(
    polar=dict(radialaxis=dict(range=[0, 5])),
    height=500,
    title="Profil comparatif (0 = faible, 5 = excellent)",
)
st.plotly_chart(fig_radar, use_container_width=True)


# 6. Conclusion automatique

st.header("6. Conclusion")

meilleur_global = tableau_scores["Score global"].idxmax()
st.success(
    f"**{meilleur_global}** obtient le meilleur score global sur ce cycle. "
    f"Le modèle **{nom_affichage(_meilleur('soc_eb_final', 'max'))}** préserve le mieux "
    f"la batterie d'énergie, tandis que **{nom_affichage(_meilleur('p_pb_max', 'max'))}** "
    "présente les plus fortes sollicitations en puissance sur la batterie de puissance. "
    "La logique floue constitue généralement un compromis entre stabilité et réactivité ; "
    "la stratégie déterministe (EB-priority) sert de référence simple mais sollicite "
    "davantage la batterie d'énergie."
)


pied_navigation("vues/6_Resultats_et_Analyse.py")
