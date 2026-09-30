import sys
from pathlib import Path

DOSSIER_PROJET = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DOSSIER_PROJET))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ems_core import DT_SECONDS
from core.resultats import (
    assurer_donnees_session,
    calculer_metriques,
    statistiques_detaillees,
    nom_affichage,
)
from core.style import couleur
from core.navigation import pied_navigation


st.title("📈 Explorer les résultats")
st.caption(
    "Comment une stratégie se comporte-t-elle sur le cycle, comparée à une référence ? "
    "Pour classer toutes les stratégies entre elles, voir « Comparer les méthodes »."
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


@st.cache_data(show_spinner="Calcul des indicateurs…")
def _indicateurs(cle_source):
    donnees = {"resultats": resultats, "cycle_df": df}
    return statistiques_detaillees(donnees), calculer_metriques(donnees)


stats, metriques = _indicateurs(source)
noms = list(resultats.keys())


def _valeur(n, cle):
    return float(metriques[n][cle] if cle in metriques[n] else stats[n][cle])


# (libellé, clé, sens favorable, format, unité, échelle)
INDICATEURS = [
    ("SOC final de l'EB", "soc_eb_final", "max", "{:.1f}", "%", 100.0),
    ("SOC final de la PB", "soc_pb_final", "max", "{:.1f}", "%", 100.0),
    ("Courant efficace de la PB", "i_pb_rms", "min", "{:.1f}", "A", 1.0),
    ("Pertes estimées", "pertes_totales_wh", "min", "{:.0f}", "Wh", 1.0),
    ("Demande non fournie", "energie_non_servie_wh", "min", "{:.0f}", "Wh", 1.0),
    ("Violations de SOC", "nb_violations", "min", "{:.0f}", "", 1.0),
]


# 1 — Choix de la stratégie et de la référence

col_s, col_r = st.columns(2)
defaut_ref = next((n for n in noms if "power_limitation" in n), noms[0])
defaut_cible = next((n for n in noms if n != defaut_ref), noms[0])
cible = col_s.selectbox("Stratégie à explorer", noms, index=noms.index(defaut_cible), format_func=nom_affichage)
autres = [n for n in noms if n != cible]
ref = col_r.selectbox(
    "Comparer à", autres,
    index=autres.index(defaut_ref) if defaut_ref in autres else 0, format_func=nom_affichage,
)
st.caption(f"Source des données : {source}")


def _libelle(n):
    return f"{nom_affichage(n)} ({'explorée' if n == cible else 'référence'})"


def _trait(n):
    if n == cible:
        return dict(color=couleur(n), width=2.4)
    return dict(color=couleur(n), width=1.8, dash="dash")


for n in (cible, ref):
    if _valeur(n, "energie_non_servie_wh") >= 1.0:
        st.warning(
            f"{nom_affichage(n)} ne fournit pas toute la puissance demandée "
            f"({_valeur(n, 'energie_non_servie_wh'):.0f} Wh manquants) : le modèle étant sans "
            "pertes, cette énergie reste dans les batteries et fait paraître son SOC final meilleur."
        )


# 2 — Écarts à la référence

st.subheader("Écarts à la référence")
st.caption(
    f"Valeur de {nom_affichage(cible)}, écart avec {nom_affichage(ref)} (vert : avantage "
    f"pour {nom_affichage(cible)}), et rang parmi les {len(noms)} stratégies."
)


def _rang(cle, sens, f, echelle, n):
    """Rang (ex æquo à la précision affichée) de la stratégie n."""
    valeurs = sorted({f.format(_valeur(m, cle) * echelle) for m in noms}, key=float, reverse=(sens == "max"))
    return valeurs.index(f.format(_valeur(n, cle) * echelle)) + 1


for ligne in (INDICATEURS[:3], INDICATEURS[3:]):
    colonnes = st.columns(3)
    for col, (lib, cle, sens, f, unite, echelle) in zip(colonnes, ligne):
        v_c, v_r = _valeur(cible, cle) * echelle, _valeur(ref, cle) * echelle
        ecart = v_c - v_r
        unite_ecart = " pts" if unite == "%" else (f" {unite}" if unite else "")
        with col:
            st.metric(
                lib,
                f"{f.format(v_c)} {unite}".strip(),
                delta=None if f.format(abs(ecart)) == f.format(0.0) else f.format(ecart) + unite_ecart,
                delta_color="normal" if sens == "max" else "inverse",
                border=True,
            )
            rang = _rang(cle, sens, f, echelle, cible)
            st.caption(f"{rang}{'er' if rang == 1 else 'e'} sur {len(noms)}  ·  {'plus haut' if sens == 'max' else 'plus bas'} = mieux")


# 3 — Utilisation des batteries

st.subheader("Utilisation des batteries")
st.caption("Une pente plus faible signifie que la batterie a été moins sollicitée.")


def _trajectoire(cle_soc, titre):
    fig = go.Figure()
    for n in (ref, cible):
        y = np.asarray(resultats[n][cle_soc], dtype=float) * 100.0
        pas = max(1, len(y) // 2000)
        fig.add_trace(
            go.Scatter(
                x=(np.arange(len(y)) * DT_SECONDS / 60.0)[::pas], y=y[::pas], mode="lines",
                name=_libelle(n), line=_trait(n),
                hovertemplate="%{x:.0f} min : %{y:.1f} %<extra></extra>",
            )
        )
    fig.update_layout(
        title=titre, xaxis_title="Temps écoulé (min)", yaxis=dict(title="SOC (%)", range=[0, 102]),
        height=340, margin=dict(t=45, b=40, l=50, r=15), hovermode="x unified",
        legend=dict(orientation="h", y=-0.25, x=0),
    )
    return fig


g1, g2 = st.columns(2)
g1.plotly_chart(_trajectoire("SOC_EB", "Batterie Énergie"), width="stretch")
g2.plotly_chart(_trajectoire("SOC_PB", "Batterie Puissance"), width="stretch")


# 4 — Sollicitation de la batterie Puissance

st.subheader("Sollicitation de la batterie Puissance")
st.caption(
    "Monotone du courant : les valeurs du courant PB (charge et décharge confondues), "
    "triées de la plus forte à la plus faible. Plus la courbe est basse, plus la batterie "
    "est ménagée ; le courant efficace, lié au vieillissement, se joue surtout à gauche."
)

# Une boîte à moustaches ne convient pas : le courant PB est quasi nul la moitié
# du temps, les boîtes s'écrasent sur 0 et seules les moustaches restent visibles.
fig_i = go.Figure()
part_active = {}
for n in (ref, cible):
    i_abs = np.abs(np.asarray(resultats[n]["I_PB"], dtype=float))
    part_active[n] = float(np.mean(i_abs > 1.0) * 100)
    tri = np.sort(i_abs)[::-1]
    part = np.arange(len(tri)) / len(tri) * 100.0
    pas = max(1, len(tri) // 2000)
    fig_i.add_trace(
        go.Scatter(
            x=part[::pas], y=tri[::pas], mode="lines", name=_libelle(n), line=_trait(n),
            hovertemplate="≥ %{y:.0f} A pendant %{x:.1f} % du cycle<extra></extra>",
        )
    )
fig_i.update_layout(
    xaxis=dict(title="Part du cycle (%)", range=[0, min(100.0, max(part_active.values()) * 1.15 + 1)]),
    yaxis_title="Courant de la PB (A, valeur absolue)",
    height=360, margin=dict(t=20, b=40, l=50, r=15),
    legend=dict(orientation="h", y=-0.22, x=0),
)
st.plotly_chart(fig_i, width="stretch")
st.caption("L'axe horizontal s'arrête là où le courant devient nul pour les deux stratégies.")

rms_c, rms_r = _valeur(cible, "i_pb_rms"), _valeur(ref, "i_pb_rms")
ecart_rel = (rms_c - rms_r) / rms_r * 100 if rms_r else 0.0
st.markdown(
    f"**{nom_affichage(cible)}** sollicite la batterie Puissance pendant "
    f"**{part_active[cible]:.0f} %** du cycle (contre {part_active[ref]:.0f} % pour "
    f"{nom_affichage(ref)}), avec un courant efficace de **{rms_c:.1f} A** "
    f"({ecart_rel:+.0f} %) et un pic à {stats[cible]['i_pb_max']:.0f} A."
)


# 5 — Indicateurs bruts, repliés

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
            "Pertes estimées (Wh)": [metriques[n]["pertes_totales_wh"] for n in noms],
            "Demande non fournie (Wh)": [metriques[n]["energie_non_servie_wh"] for n in noms],
            "Violations SOC": [metriques[n]["nb_violations"] for n in noms],
        }
    ).set_index("Stratégie")
    st.dataframe(
        tableau.style.format("{:.1f}").format(
            {"I_PB max (A)": "{:.0f}", "Pertes estimées (Wh)": "{:.0f}",
             "Demande non fournie (Wh)": "{:.0f}", "Violations SOC": "{:.0f}"}
        ),
        width="stretch",
    )


pied_navigation("vues/6_Resultats_et_Analyse.py")
