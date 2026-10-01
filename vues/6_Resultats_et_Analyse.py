"""
Résultats de simulation : évolution du HESS au cours du cycle et performances
de la stratégie sélectionnée. Pas de classement ici : la comparaison entre
stratégies est dans « Comparaison des stratégies EMS ».
"""

import sys
from pathlib import Path

DOSSIER_PROJET = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DOSSIER_PROJET))

__________________________________________________import numpy as np
import plotly.graph_objects as go
import streamlit as st

import ems_core as core
from core.pertes import RENDEMENT_CONVERTISSEUR, pertes_par_pas
from core.resultats import assurer_donnees_session, calculer_metriques, nom_affichage
from core.style import (
    COULEUR_CONVERTISSEUR,
    COULEUR_DEMANDE,
    COULEUR_EB,
    COULEUR_PB,
    COULEUR_REFERENCE,
    COULEUR_SECONDAIRE,
    COULEUR_VIOLATION,
)
from core.navigation import pied_navigation


st.title("📈 Résultats de simulation")
st.caption(
    "Évolution du HESS au cours du cycle et analyse des performances de la stratégie "
    "sélectionnée."
)

try:
    source = assurer_donnees_session(st)
except FileNotFoundError as exc:
    st.error(str(exc))
    st.info("Lancez une fois le précalcul :  `python scripts/run_simulations.py`")
    st.stop()

resultats = st.session_state.get("resultats_simulation")
df = st.session_state.get("cycle_pret")
if not resultats or df is None:
    st.warning("Aucune donnée disponible.")
    st.stop()

noms = list(resultats.keys())
c_s, c_src = st.columns([2, 3])
strategie = c_s.selectbox(
    "Stratégie", noms,
    index=noms.index("EMS_power_limitation") if "EMS_power_limitation" in noms else 0,
    format_func=nom_affichage,
)
c_src.caption(f"Source des données : {source}")

traj = resultats[strategie]
n = min(len(df), len(traj["P_EB"]))


@st.cache_data(show_spinner="Calcul des indicateurs…")
def _indicateurs(strategie, signature):
    return calculer_metriques({"resultats": {strategie: resultats[strategie]}, "cycle_df": df})[strategie]


m = _indicateurs(strategie, float(np.nansum(traj["alpha_final"])))
if m["energie_non_servie_wh"] >= 1.0:
    st.warning(
        f"{nom_affichage(strategie)} ne fournit pas toute la puissance demandée "
        f"({m['energie_non_servie_wh']:.0f} Wh manquants, jusqu'à {m['ecart_puissance_max_kw']:.1f} kW "
        "à un instant) : son énergie consommée et son rendement en sont flattés."
    )


# Les six indicateurs du protocole

l1 = st.columns(3)
l1[0].metric("M1 · Énergie consommée", f"{m['energie_km_wh']:.1f} Wh/km", border=True,
             help=f"{m['energie_consommee_wh'] / 1000:.2f} kWh sur le cycle, pertes estimées comprises")
l1[1].metric("M2 · Rendement du HESS", f"{m['rendement_hess'] * 100:.2f} %", border=True,
             help="Estimé : pertes calculées après coup sur une simulation sans pertes")
l1[2].metric("M3 · Écart moyen des SOC", f"{m['rmse_delta_soc'] * 100:.1f} pts", border=True,
             help=f"Écart quadratique moyen ; écart maximal {m['delta_soc_max'] * 100:.1f} points")
l2 = st.columns(3)
l2[0].metric("M4 · Pertes du convertisseur", f"{m['pertes_convertisseur_wh']:.0f} Wh", border=True)
l2[1].metric("M5 · Violations", f"{m['nb_violations'] + m['nb_violations_courant']:.0f}", border=True,
             help=f"{m['nb_violations']:.0f} de SOC, {m['nb_violations_courant']:.0f} de courant")
l2[2].metric("M6 · Erreur de suivi de puissance", f"{m['rmse_puissance_kw']:.2f} kW", border=True,
             help=f"Écart quadratique moyen en traction ; écart maximal {m['ecart_puissance_max_kw']:.1f} kW")

temps_min = (df["time"].to_numpy(dtype=float)[:n] if "time" in df.columns else np.arange(n)) / 60.0
p_dem = df["hasPower"].to_numpy(dtype=float)[:n]
p_eb = np.asarray(traj["P_EB"], dtype=float)[:n]
p_pb = np.asarray(traj["P_PB"], dtype=float)[:n]


# 1 — Répartition de la puissance

st.subheader("Comment la demande de puissance est-elle répartie entre les deux batteries ?")
st.caption("Faites glisser la réglette sous le graphique pour zoomer sur une portion du cycle.")
fig_p = go.Figure(
    [
        go.Scattergl(x=temps_min, y=p_dem / 1000, name="Demande", line=dict(color=COULEUR_DEMANDE, width=1.2)),
        go.Scattergl(x=temps_min, y=p_eb / 1000, name="Batterie Énergie", line=dict(color=COULEUR_EB, width=1.2)),
        go.Scattergl(x=temps_min, y=p_pb / 1000, name="Batterie Puissance", line=dict(color=COULEUR_PB, width=1.2)),
    ]
)
fig_p.add_hline(y=core.P_EB_MAX_W / 1000, line=dict(color=COULEUR_SECONDAIRE, dash="dot", width=1),
                annotation_text="limite de la batterie Énergie", annotation_position="top left")
fig_p.update_layout(
    height=380, margin=dict(t=10, b=10, l=10, r=10), hovermode="x unified", yaxis_title="Puissance (kW)",
    xaxis=dict(title="Temps (min)", rangeslider=dict(visible=True, thickness=0.08), range=[60, 90]),
    legend=dict(orientation="h", y=1.12, x=0),
)
st.plotly_chart(fig_p, width="stretch")


# 2 — États de charge

st.subheader("Évolution des états de charge")
x_soc = np.arange(len(traj["SOC_EB"])) * core.DT_SECONDS / 60.0
fig_soc = go.Figure(
    [
        go.Scatter(x=x_soc, y=np.asarray(traj["SOC_EB"], float) * 100, name="Batterie Énergie",
                   line=dict(color=COULEUR_EB, width=2)),
        go.Scatter(x=x_soc, y=np.asarray(traj["SOC_PB"], float) * 100, name="Batterie Puissance",
                   line=dict(color=COULEUR_PB, width=2)),
    ]
)
fig_soc.add_hline(y=core.SOC_EB_MIN * 100, line=dict(color=COULEUR_VIOLATION, dash="dot", width=1),
                  annotation_text="SOC minimal", annotation_position="bottom right")
fig_soc.add_hline(y=core.SOC_EB_MAX * 100, line=dict(color=COULEUR_REFERENCE, dash="dot", width=1),
                  annotation_text="SOC maximal", annotation_position="top right")
fig_soc.update_layout(
    height=320, margin=dict(t=10, b=40, l=10, r=10), hovermode="x unified",
    xaxis_title="Temps (min)", yaxis=dict(title="SOC (%)", range=[0, 105]),
    legend=dict(orientation="h", y=1.12, x=0),
)
st.plotly_chart(fig_soc, width="stretch")


# 3 — Pertes

st.subheader("Pertes estimées")
st.caption(
    "Pertes cumulées au fil du cycle : effet Joule dans chaque batterie (R·I²) et pertes du "
    "convertisseur, qui ne traite que la différence de tension entre les batteries. "
    "Estimation faite après coup : la simulation elle-même est sans pertes."
)
pas = pertes_par_pas(traj)
h = core.DT_SECONDS / 3600.0
cumul = {k: np.cumsum(v[:n]) * h for k, v in pas.items()}
total = cumul["eb"] + cumul["pb"] + cumul["convertisseur"]
pas_graphe = max(1, n // 2000)
fig_l = go.Figure(
    [
        go.Scatter(x=temps_min[::pas_graphe], y=cumul["eb"][::pas_graphe], name="Batterie Énergie",
                   line=dict(color=COULEUR_EB, width=1.8)),
        go.Scatter(x=temps_min[::pas_graphe], y=cumul["pb"][::pas_graphe], name="Batterie Puissance",
                   line=dict(color=COULEUR_PB, width=1.8)),
        go.Scatter(x=temps_min[::pas_graphe], y=cumul["convertisseur"][::pas_graphe], name="Convertisseur",
                   line=dict(color=COULEUR_CONVERTISSEUR, width=1.8)),
        go.Scatter(x=temps_min[::pas_graphe], y=total[::pas_graphe], name="Pertes totales",
                   line=dict(color=COULEUR_REFERENCE, width=2.2, dash="dash")),
    ]
)
fig_l.update_layout(
    height=320, margin=dict(t=10, b=40, l=10, r=10), hovermode="x unified",
    xaxis_title="Temps (min)", yaxis_title="Pertes cumulées (Wh)", legend=dict(orientation="h", y=1.12, x=0),
)
st.plotly_chart(fig_l, width="stretch")
p1, p2, p3, p4 = st.columns(4)
p1.metric("Batterie Énergie", f"{cumul['eb'][-1]:.0f} Wh")
p2.metric("Batterie Puissance", f"{cumul['pb'][-1]:.0f} Wh")
p3.metric("Convertisseur", f"{cumul['convertisseur'][-1]:.0f} Wh")
p4.metric("Pertes totales", f"{total[-1]:.0f} Wh")
with st.expander("Hypothèses du calcul des pertes"):
    st.markdown(
        f"- Résistances internes calculées à partir des cellules : {core.CELL_EB_RINT_OHM * 1000:.0f} mΩ × "
        f"{core.CELL_EB_N_SERIE}/{core.CELL_EB_N_PARALLELE} pour l'EB, {core.CELL_PB_RINT_OHM * 1000:.1f} mΩ × "
        f"{core.CELL_PB_N_SERIE}/{core.CELL_PB_N_PARALLELE} pour la PB.\n"
        f"- Convertisseur : rendement de {RENDEMENT_CONVERTISSEUR * 100:.1f} % (mesuré à puissance nominale, "
        "plus faible à charge partielle), appliqué à la seule puissance qu'il traite.\n"
        "- Tensions constantes, sans variation avec le SOC."
    )


pied_navigation("vues/6_Resultats_et_Analyse.py")
