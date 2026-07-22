"""
Page « Explorer les résultats » — vue narrative sur l'ensemble du cycle.

Plutôt que 14 000 points bruts, on raconte le cycle : points forts, tendances
et profil de chaque stratégie, avec une analyse automatique à chaque étape et
une synthèse finale. Les tableaux détaillés restent accessibles, mais repliés.
"""

import sys
from pathlib import Path

DOSSIER_PROJET = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DOSSIER_PROJET))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.resultats import (
    assurer_donnees_session,
    calculer_metriques,
    statistiques_detaillees,
    nom_affichage,
)
from core.navigation import pied_navigation


# Configuration de page gérée par le routeur Accueil.py.

C_EB = "#3B82F6"
C_PB = "#22C55E"
C_GRIS = "#94A3B8"


st.title("📈 Explorer les résultats")
st.caption(
    "Que s'est-il passé sur l'ensemble du cycle ? Points forts, tendances et "
    "profil de chaque stratégie, avec une analyse automatique à chaque étape."
)


try:
    _source = assurer_donnees_session(st)
except FileNotFoundError as exc:
    st.error(str(exc))
    st.info("Lance une fois le précalcul :  `python scripts/run_simulations.py`")
    st.stop()

if "resultats_simulation" not in st.session_state or "cycle_pret" not in st.session_state:
    st.warning("Aucune donnée disponible.")
    st.stop()

resultats = st.session_state["resultats_simulation"]
df = st.session_state["cycle_pret"]
donnees = {"resultats": resultats, "cycle_df": df}

if not resultats:
    st.warning("Aucune stratégie n'a produit de résultat exploitable.")
    st.stop()

st.caption(f"Source des données : {_source}")

stats = statistiques_detaillees(donnees)
metriques = calculer_metriques(donnees)
noms = list(resultats.keys())


def _meilleur(cle, sens="max"):
    """Stratégie optimisant une statistique (max = plus haut, min = plus bas)."""
    paires = [(n, stats[n][cle]) for n in noms]
    return (max if sens == "max" else min)(paires, key=lambda kv: kv[1])[0]


def _min_violations():
    return min(noms, key=lambda n: metriques[n]["nb_violations"])


# Points forts du cycle

st.subheader("🌟 Points forts du cycle")

k1, k2, k3, k4 = st.columns(4)
best_eb = _meilleur("soc_eb_final", "max")
best_pb = _meilleur("soc_pb_final", "max")
best_i = _meilleur("i_pb_rms", "min")
best_v = _min_violations()
k1.metric("Préserve le mieux l'EB", nom_affichage(best_eb), f"{stats[best_eb]['soc_eb_final'] * 100:.0f} %")
k2.metric("Préserve le mieux la PB", nom_affichage(best_pb), f"{stats[best_pb]['soc_pb_final'] * 100:.0f} %")
k3.metric("Courant PB le plus régulier", nom_affichage(best_i), f"{stats[best_i]['i_pb_rms']:.0f} A RMS")
k4.metric("Le moins de violations SOC", nom_affichage(best_v), f"{metriques[best_v]['nb_violations']:.0f}")

with st.expander("Voir le tableau détaillé (toutes les stratégies)"):
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


# Évolution des états de charge

st.subheader("📉 Comment évoluent les batteries ?")


def _courbe_soc(cle_soc, titre, couleur_titre):
    fig = go.Figure()
    for n in noms:
        y = np.asarray(resultats[n][cle_soc], dtype=float) * 100.0
        x = np.arange(len(y))
        pas = max(1, len(y) // 2000)
        fig.add_trace(go.Scatter(x=x[::pas], y=y[::pas], mode="lines", name=nom_affichage(n)))
    fig.update_layout(
        title=titre, xaxis_title="Temps (s)", yaxis_title="SOC (%)", height=400,
        legend_title="Stratégie", margin=dict(t=50, b=40),
    )
    return fig


col_eb, col_pb = st.columns(2)
with col_eb:
    st.plotly_chart(_courbe_soc("SOC_EB", "SOC batterie Énergie", C_EB), use_container_width=True)
with col_pb:
    st.plotly_chart(_courbe_soc("SOC_PB", "SOC batterie Puissance", C_PB), use_container_width=True)

st.info(
    f"Analyse automatique — SOC_EB final le plus élevé : **{nom_affichage(best_eb)}** "
    f"({stats[best_eb]['soc_eb_final'] * 100:.0f} %). Plus une courbe descend, plus la "
    "batterie a été sollicitée sur le cycle."
)


# Répartition des puissances

st.subheader("⚡ Comment se répartit la puissance ?")


def _boxplot(cle_p, titre):
    fig = go.Figure()
    for n in noms:
        y = np.asarray(resultats[n][cle_p], dtype=float) / 1000.0
        fig.add_trace(go.Box(y=y, name=nom_affichage(n), boxpoints=False))
    fig.update_layout(title=titre, yaxis_title="Puissance (kW)", height=400, showlegend=False, margin=dict(t=50, b=40))
    return fig


col_peb, col_ppb = st.columns(2)
with col_peb:
    st.plotly_chart(_boxplot("P_EB", "Puissance batterie Énergie"), use_container_width=True)
with col_ppb:
    st.plotly_chart(_boxplot("P_PB", "Puissance batterie Puissance"), use_container_width=True)

pb_max_strat = _meilleur("p_pb_max", "max")
st.info(
    "Analyse automatique — la boîte montre la dispersion : plus elle est haute, plus "
    f"la batterie encaisse de pics. Pic de puissance PB le plus élevé : "
    f"**{nom_affichage(pb_max_strat)}** ({stats[pb_max_strat]['p_pb_max'] / 1000:.1f} kW)."
)


# Sollicitation électrique (courants)

st.subheader("🔌 Quelle sollicitation électrique ?")


def _boxplot_courant(cle_i, titre):
    fig = go.Figure()
    for n in noms:
        y = np.asarray(resultats[n][cle_i], dtype=float)
        fig.add_trace(go.Box(y=y, name=nom_affichage(n), boxpoints=False))
    fig.update_layout(title=titre, yaxis_title="Courant (A)", height=400, showlegend=False, margin=dict(t=50, b=40))
    return fig


col_ieb, col_ipb = st.columns(2)
with col_ieb:
    st.plotly_chart(_boxplot_courant("I_EB", "Courant batterie Énergie"), use_container_width=True)
with col_ipb:
    st.plotly_chart(_boxplot_courant("I_PB", "Courant batterie Puissance"), use_container_width=True)

ipb_max_strat = _meilleur("i_pb_max", "max")
st.info(
    f"Analyse automatique — courant PB le plus régulier : **{nom_affichage(best_i)}**. "
    f"Pic de courant PB le plus élevé : **{nom_affichage(ipb_max_strat)}** "
    f"({stats[ipb_max_strat]['i_pb_max']:.0f} A). Un courant plus régulier ménage la batterie."
)


# Profil comparatif (radar)

st.subheader("🧭 Profil comparatif des stratégies")

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

with st.expander("Voir les scores par axe"):
    st.dataframe(tableau_scores.style.format("{:.1f}"), use_container_width=True)


# Verdict + à retenir

st.subheader("📋 Verdict")

meilleur_global = tableau_scores["Score global"].idxmax()
score_global = tableau_scores["Score global"].max()
with st.container(border=True):
    st.markdown(
        f"Sur l'ensemble du cycle, **{meilleur_global}** obtient le meilleur profil global "
        f"({score_global:.1f}/5)."
    )
    st.markdown("Points marquants :")
    st.markdown(
        f"- Préserve le mieux l'EB : **{nom_affichage(best_eb)}**\n"
        f"- Préserve le mieux la PB : **{nom_affichage(best_pb)}**\n"
        f"- Courant PB le plus régulier : **{nom_affichage(best_i)}**\n"
        f"- Le moins de violations SOC : **{nom_affichage(best_v)}**"
    )

st.subheader("🎓 Ce qu'il faut retenir")

ns_noms = [n for n in noms if "neurosymbolic" in n]
soc_ns = np.mean([stats[n]["soc_eb_final"] for n in ns_noms]) if ns_noms else None
soc_det = stats["EMS_power_limitation"]["soc_eb_final"] if "EMS_power_limitation" in stats else None

phrase = f"Sur ce cycle, aucune stratégie ne domine partout : le meilleur profil global est {meilleur_global}."
if soc_ns is not None and soc_det is not None:
    if soc_ns > soc_det:
        phrase += (
            " Les variantes neuro-symboliques préservent en moyenne mieux la batterie "
            "d'énergie que la stratégie déterministe de référence."
        )
    else:
        phrase += (
            " Ici, la stratégie déterministe de référence préserve autant ou mieux la "
            "batterie d'énergie que les variantes neuro-symboliques."
        )
st.info(phrase)


pied_navigation("vues/6_Resultats_et_Analyse.py")
