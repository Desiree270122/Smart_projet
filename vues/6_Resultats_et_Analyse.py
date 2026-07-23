
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
from core.style import couleur, COULEUR_NEUTRE
from core.navigation import pied_navigation
 
 
st.title("Explorer les résultats")
st.caption(
    "Comment une stratégie se comporte-t-elle sur le cycle, et en quoi diffère-t-elle "
    "d'une référence ? Choisissez la stratégie à examiner."
)
 
try:
    source = assurer_donnees_session(st)
except FileNotFoundError as exc:
    st.error(str(exc))
    st.info("Lancez une fois le précalcul :  `python scripts/run_simulations.py`")
    st.stop()
 
if "resultats_simulation" not in st.session_state or "cycle_pret" not in st.session_state:
    st.warning("Aucune donnée disponible.")
    st.stop()
 
resultats = st.session_state["resultats_simulation"]
df = st.session_state["cycle_pret"]
 
if not resultats:
    st.warning("Aucune stratégie n'a produit de résultat exploitable.")
    st.stop()
 
 
@st.cache_data(show_spinner="Calcul des indicateurs…")
def _indicateurs(cle_source):
    donnees = {"resultats": resultats, "cycle_df": df}
    return statistiques_detaillees(donnees), calculer_metriques(donnees)
 
 
stats, metriques = _indicateurs(source)
noms = list(resultats.keys())
 
 
def _valeur(n, cle):
    if cle == "nb_violations":
        return float(metriques[n]["nb_violations"])
    return float(stats[n][cle])
 
 
# (libellé, clé, sens favorable, format, unité affichée)
INDICATEURS = [
    ("Préservation batterie Énergie", "soc_eb_final", "max", "{:.1f}", "% de SOC final"),
    ("Préservation batterie Puissance", "soc_pb_final", "max", "{:.1f}", "% de SOC final"),
    ("Stabilité du courant PB", "i_pb_rms", "min", "{:.0f}", "A RMS"),
    ("Respect des contraintes SOC", "nb_violations", "min", "{:.0f}", "violations"),
]
 
ECHELLE = {"soc_eb_final": 100.0, "soc_pb_final": 100.0, "i_pb_rms": 1.0, "nb_violations": 1.0}
 
 
# 1 — Choix de la stratégie et de la référence
 
col_s, col_r = st.columns(2)
 
cible = col_s.selectbox(
    "Stratégie à explorer",
    noms,
    format_func=nom_affichage,
)
 
defaut_ref = next((n for n in noms if "power_limitation" in n), noms[0])
autres = [n for n in noms if n != cible]
ref = col_r.selectbox(
    "Comparer à",
    autres,
    index=autres.index(defaut_ref) if defaut_ref in autres else 0,
    format_func=nom_affichage,
)
 
st.caption(f"Source des données : {source}")
st.divider()
 
 
# 2 — Écarts chiffrés, pas d'étoiles
 
st.subheader("Écarts à la référence")
 
cols = st.columns(len(INDICATEURS))
for col, (lib, cle, sens, f, unite) in zip(cols, INDICATEURS):
    v_c = _valeur(cible, cle) * ECHELLE[cle]
    v_r = _valeur(ref, cle) * ECHELLE[cle]
    ecart = v_c - v_r
    col.metric(
        lib,
        f.format(v_c) + f" {unite}",
        delta=("—" if abs(ecart) < 1e-9 else f.format(ecart)),
        delta_color=("normal" if sens == "max" else "inverse"),
    )
st.caption(f"Écart calculé face à {nom_affichage(ref)}. Vert = avantage pour la stratégie explorée.")
 
 
# 3 — Position relative : l'écart, pas le rang
 
st.subheader("Position parmi les sept stratégies")
st.caption(
    "Chaque ligne place les sept stratégies entre la pire (0) et la meilleure (1) valeur "
    "observée sur ce cycle. Des points serrés signifient que le critère ne départage pas."
)
 
fig_pos = go.Figure()
for lib, cle, sens, f, unite in INDICATEURS:
    vals = {n: _valeur(n, cle) for n in noms}
    lo, hi = min(vals.values()), max(vals.values())
    for n in noms:
        x = 0.5 if hi - lo < 1e-12 else (vals[n] - lo) / (hi - lo)
        if sens == "min":
            x = 1.0 - x
        vedette = n in (cible, ref)
        fig_pos.add_trace(
            go.Scatter(
                x=[x],
                y=[lib],
                mode="markers",
                marker=dict(
                    size=16 if vedette else 9,
                    color=couleur(n) if vedette else COULEUR_NEUTRE,
                    opacity=1.0 if vedette else 0.45,
                    line=dict(width=1.5 if vedette else 0, color="#FFFFFF"),
                ),
                hovertemplate=f"{nom_affichage(n)}<br>{f.format(vals[n] * ECHELLE[cle])} {unite}<extra></extra>",
                showlegend=False,
            )
        )
 
fig_pos.update_layout(
    height=90 * len(INDICATEURS) + 60,
    margin=dict(t=20, b=40, l=10, r=20),
    xaxis=dict(title="0 = pire des sept   ·   1 = meilleure des sept", range=[-0.06, 1.06]),
    yaxis=dict(title=None, autorange="reversed"),
)
st.plotly_chart(fig_pos, use_container_width=True)
 
 
# 4 — Trajectoires : deux courbes, pas sept
 
st.subheader("Utilisation des batteries")
st.caption(
    "Une pente plus faible signifie que la batterie a été moins sollicitée sur le cycle."
)
 
 
def _trajectoire(cle_soc, titre):
    fig = go.Figure()
    for n in (ref, cible):
        y = np.asarray(resultats[n][cle_soc], dtype=float) * 100.0
        pas = max(1, len(y) // 2000)
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(y))[::pas],
                y=y[::pas],
                mode="lines",
                name=nom_affichage(n),
                line=dict(
                    color=couleur(n),
                    width=1.4 if n == ref else 2.2,
                    dash="dot" if n == ref else "solid",
                ),
            )
        )
    fig.update_layout(
        title=titre, xaxis_title="Temps (s)", yaxis_title="SOC (%)",
        height=340, margin=dict(t=45, b=40, l=50, r=15),
        legend=dict(orientation="h", y=-0.22, x=0),
    )
    return fig
 
 
g1, g2 = st.columns(2)
g1.plotly_chart(_trajectoire("SOC_EB", "Batterie Énergie"), use_container_width=True)
g2.plotly_chart(_trajectoire("SOC_PB", "Batterie Puissance"), use_container_width=True)
 
chute_c = (float(resultats[cible]["SOC_EB"][0]) - float(resultats[cible]["SOC_EB"][-1])) * 100
chute_r = (float(resultats[ref]["SOC_EB"][0]) - float(resultats[ref]["SOC_EB"][-1])) * 100
diff = chute_c - chute_r
if abs(diff) < 0.5:
    st.markdown(
        f"{nom_affichage(cible)} décharge la batterie Énergie de {chute_c:.1f} points sur le "
        f"cycle, soit un écart négligeable avec {nom_affichage(ref)} ({chute_r:.1f} points)."
    )
else:
    sens_txt = "moins" if diff < 0 else "plus"
    st.markdown(
        f"{nom_affichage(cible)} décharge la batterie Énergie de {chute_c:.1f} points, "
        f"soit {abs(diff):.1f} points de {sens_txt} que {nom_affichage(ref)} "
        f"({chute_r:.1f} points)."
    )
 
 
# 5 — Courant PB
 
st.subheader("Sollicitation de la batterie Puissance")
st.caption(
    "Le courant efficace est l'indicateur lié au vieillissement électrochimique : "
    "une distribution resserrée traduit un fonctionnement plus doux."
)
 
fig_i = go.Figure()
for n in noms:
    vedette = n in (cible, ref)
    fig_i.add_trace(
        go.Box(
            y=np.asarray(resultats[n]["I_PB"], dtype=float),
            name=nom_affichage(n),
            boxpoints=False,
            marker_color=couleur(n) if vedette else COULEUR_NEUTRE,
            opacity=1.0 if vedette else 0.35,
        )
    )
fig_i.update_layout(
    yaxis_title="Courant (A)", height=400, showlegend=False,
    margin=dict(t=20, b=90, l=50, r=15),
)
st.plotly_chart(fig_i, use_container_width=True)
 
rms_c, rms_r = _valeur(cible, "i_pb_rms"), _valeur(ref, "i_pb_rms")
pic_c = float(stats[cible]["i_pb_max"])
ecart_rel = (rms_c - rms_r) / rms_r * 100 if rms_r else 0.0
st.markdown(
    f"{nom_affichage(cible)} : {rms_c:.0f} A RMS, pic à {pic_c:.0f} A. "
    f"Soit {abs(ecart_rel):.0f} % {'de moins' if ecart_rel < 0 else 'de plus'} "
    f"que {nom_affichage(ref)} ({rms_r:.0f} A RMS)."
)
 
 
# 6 — Table de référence, repliée
 
with st.expander("Indicateurs bruts, toutes stratégies"):
    tableau = pd.DataFrame(
        {
            "Stratégie": [nom_affichage(n) for n in noms],
            "SOC_EB final (%)": [stats[n]["soc_eb_final"] * 100 for n in noms],
            "SOC_PB final (%)": [stats[n]["soc_pb_final"] * 100 for n in noms],
            "Énergie EB (Wh)": [stats[n]["energie_eb_wh"] for n in noms],
            "Énergie PB (Wh)": [stats[n]["energie_pb_wh"] for n in noms],
            "I_EB RMS (A)": [stats[n]["i_eb_rms"] for n in noms],
            "I_PB RMS (A)": [stats[n]["i_pb_rms"] for n in noms],
            "I_PB max (A)": [stats[n]["i_pb_max"] for n in noms],
            "P_PB max (kW)": [stats[n]["p_pb_max"] / 1000 for n in noms],
            "Violations SOC": [metriques[n]["nb_violations"] for n in noms],
        }
    ).set_index("Stratégie")
    st.dataframe(
        tableau.style.format(
            {
                "SOC_EB final (%)": "{:.1f}", "SOC_PB final (%)": "{:.1f}",
                "Énergie EB (Wh)": "{:.1f}", "Énergie PB (Wh)": "{:.1f}",
                "I_EB RMS (A)": "{:.0f}", "I_PB RMS (A)": "{:.0f}",
                "I_PB max (A)": "{:.0f}", "P_PB max (kW)": "{:.1f}",
                "Violations SOC": "{:.0f}",
            }
        ),
        use_container_width=True,
    )
    st.caption(
        "Pour classer les stratégies entre elles, voir la page « Comparer les méthodes »."
    )
 
 
pied_navigation("vues/6_Resultats_et_Analyse.py")
 
