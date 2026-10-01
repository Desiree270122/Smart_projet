"""
Comparaison des stratégies EMS : les sept stratégies, sur un même cycle et
selon un protocole commun (M1 à M6). La page décrit les résultats ; elle ne
désigne pas de gagnant — la conclusion appartient à l'analyse.
"""

import sys
from pathlib import Path

DOSSIER_PROJET = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DOSSIER_PROJET))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from core.resultats import (
    assurer_donnees_session,
    calculer_metriques,
    famille,
    nom_affichage,
    PAIRES_SYMBOLIQUE,
)
from core.style import couleur
from core import xai
from core.navigation import pied_navigation


# Les six métriques du protocole : (clé, libellé court, unité, échelle, format)
METRIQUES = [
    ("energie_km_wh", "M1 · Énergie consommée", "Wh/km", 1.0, "{:.1f}"),
    ("rendement_hess", "M2 · Rendement du HESS", "%", 100.0, "{:.2f}"),
    ("rmse_delta_soc", "M3 · Écart entre les SOC", "pts", 100.0, "{:.1f}"),
    ("pertes_convertisseur_wh", "M4 · Pertes du convertisseur", "Wh", 1.0, "{:.0f}"),
    ("violations_totales", "M5 · Violations de contraintes", "", 1.0, "{:.0f}"),
    ("rmse_puissance_kw", "M6 · Erreur de suivi de puissance", "kW", 1.0, "{:.2f}"),
]


def _titre(lib, unite):
    return f"{lib} ({unite})" if unite else lib


st.title("📊 Comparaison des stratégies EMS")
st.caption(
    "Comparaison des performances des sept stratégies sur un même cycle et selon un "
    "protocole commun : même profil de conduite, mêmes batteries, même convertisseur, "
    "même filtre de sécurité."
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


@st.cache_data(show_spinner="Calcul des métriques…")
def _metriques(signature):
    m = calculer_metriques({"resultats": resultats, "cycle_df": df})
    for v in m.values():
        v["violations_totales"] = v["nb_violations"] + v["nb_violations_courant"]
    return m


metriques = _metriques(tuple(sorted((n, float(np.nansum(t["alpha_final"]))) for n, t in resultats.items())))
st.caption(f"Source des données : {source}  ·  {len(noms)} stratégies")

with st.expander("Définition des métriques"):
    st.markdown(
        "- **M1 · Énergie consommée** : énergie tirée des batteries sur le cycle, pertes estimées "
        "comprises, rapportée à la distance parcourue.\n"
        "- **M2 · Rendement du HESS** : énergie de traction fournie, rapportée à cette énergie "
        "augmentée des pertes (batteries et convertisseur).\n"
        "- **M3 · Écart entre les SOC** : écart quadratique moyen entre les SOC des deux batteries.\n"
        "- **M4 · Pertes du convertisseur** : pertes du convertisseur à puissance partielle, qui ne "
        "traite que (V_EB − V_PB)·I_EB.\n"
        "- **M5 · Violations de contraintes** : nombre de pas où une limite de SOC ou de courant est dépassée.\n"
        "- **M6 · Erreur de suivi de puissance** : écart quadratique moyen entre puissance fournie "
        "et demandée en traction."
    )
    st.latex(
        r"M_1 = \frac{1}{D}\Big(\int_0^T (P_{EB} + P_{PB})\,dt + E_{pertes}\Big)"
        r"\qquad M_2 = \frac{E_{traction}}{E_{traction} + E_{pertes}}"
        r"\qquad M_3 = \sqrt{\tfrac{1}{N}\textstyle\sum_k (SOC_{EB} - SOC_{PB})^2}"
    )
    st.latex(
        r"M_4 = \int_0^T (1-\eta)\,\lvert (V_{EB}-V_{PB})\,I_{EB}\rvert\,dt"
        r"\qquad M_5 = \textstyle\sum_k \mathbf{1}(\text{limite dépassée})"
        r"\qquad M_6 = \sqrt{\tfrac{1}{N}\textstyle\sum_k (P_{HESS} - P_{dem})^2}"
    )
    st.caption(
        "Le modèle du HESS étant simulé sans pertes, M1, M2 et M4 utilisent des pertes estimées "
        "après coup (résistances internes des cellules, rendement du convertisseur de 95,5 %) : "
        "ils comparent les stratégies, sans prétendre à la valeur absolue."
    )


# 1 — Les six métriques

st.subheader("Les six métriques du protocole")
st.dataframe(
    pd.DataFrame(
        [
            {
                "Stratégie": nom_affichage(n),
                "Famille": famille(n),
                **{
                    _titre(lib, u): f.format(metriques[n][cle] * ech)
                    for cle, lib, u, ech, f in METRIQUES
                },
            }
            for n in noms
        ]
    ).set_index("Stratégie"),
    width="stretch",
)

fig_m = make_subplots(
    rows=2, cols=3, shared_yaxes=True, horizontal_spacing=0.04, vertical_spacing=0.16,
    subplot_titles=[_titre(lib, u) for _, lib, u, _, _ in METRIQUES],
)
for k, (cle, lib, u, ech, f) in enumerate(METRIQUES):
    valeurs = [metriques[n][cle] * ech for n in noms]
    fig_m.add_trace(
        go.Bar(
            y=[nom_affichage(n) for n in noms], x=valeurs, orientation="h",
            marker_color=[couleur(n) for n in noms],
            text=[f.format(v) for v in valeurs], textposition="outside", cliponaxis=False,
            hovertemplate="%{y} : %{x}<extra></extra>", showlegend=False,
        ),
        row=k // 3 + 1, col=k % 3 + 1,
    )
fig_m.update_yaxes(autorange="reversed")
fig_m.update_layout(height=540, margin=dict(t=40, b=20, l=10, r=40), bargap=0.25)
fig_m.update_annotations(font_size=12)
st.plotly_chart(fig_m, width="stretch")


# 2 — État de faisabilité

st.subheader("État de faisabilité")
st.caption(
    "Respect des contraintes (M5) et suivi de la puissance demandée (M6), en détail. Une "
    "stratégie qui ne fournit pas toute la demande consomme mécaniquement moins d'énergie : "
    "c'est à garder en tête en lisant M1 et M2."
)
st.dataframe(
    pd.DataFrame(
        [
            {
                "Stratégie": nom_affichage(n),
                "Violations de SOC": f"{metriques[n]['nb_violations']:.0f}",
                "Violations de courant": f"{metriques[n]['nb_violations_courant']:.0f}",
                "Erreur de suivi, RMSE (kW)": f"{metriques[n]['rmse_puissance_kw']:.2f}",
                "Écart maximal (kW)": f"{metriques[n]['ecart_puissance_max_kw']:.1f}",
                "Demande non fournie (Wh)": f"{metriques[n]['energie_non_servie_wh']:.0f}",
            }
            for n in noms
        ]
    ).set_index("Stratégie"),
    width="stretch",
)


# 3 — Contribution de l'approche neuro-symbolique

st.subheader("Contribution de l'approche neuro-symbolique")
st.caption(
    "Chaque modèle neuro-symbolique comparé au même réseau sans composante symbolique. "
    "NS-MLP part des règles floues et n'apprend qu'une correction bornée ; NS-LSTM reçoit "
    "des états symboliques en entrée supplémentaire."
)


def _variation(cle, a, b):
    """Écart de b par rapport à a : en points pour un pourcentage, en % sinon."""
    if cle in ("rendement_hess", "rmse_delta_soc"):
        return f"{(b - a) * 100:+.2f} pts"
    if abs(a) < 1e-12:
        return "—" if abs(b) < 1e-12 else f"{b:+.2f}"
    return f"{(b - a) / abs(a) * 100:+.1f} %"


colonnes_ns = st.columns(len(PAIRES_SYMBOLIQUE))
for col, (seul, ns) in zip(colonnes_ns, PAIRES_SYMBOLIQUE):
    if seul not in metriques or ns not in metriques:
        continue
    with col:
        st.markdown(f"**{nom_affichage(ns)} et {nom_affichage(seul)}**")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Indicateur": _titre(lib, u),
                        nom_affichage(seul): f.format(metriques[seul][cle] * ech),
                        nom_affichage(ns): f.format(metriques[ns][cle] * ech),
                        "Variation": _variation(cle, metriques[seul][cle], metriques[ns][cle]),
                    }
                    for cle, lib, u, ech, f in METRIQUES
                ]
            ).set_index("Indicateur"),
            width="stretch",
        )


# 4 — Explicabilité, analysée à part

st.subheader("Explicabilité — analyse séparée")
st.caption(
    "L'explicabilité ne se mesure pas comme une performance : elle est analysée à part, "
    "selon trois sous-critères. E1 et E2 décrivent la forme d'explication que permet "
    "l'architecture ; E3 est mesurée sur les décisions."
)
mesurer = st.toggle("Mesurer E3 · cohérence physique (≈ 30 s au premier calcul)", value=False)


@st.cache_data(show_spinner="Mesure de la cohérence physique des décisions (E3)…")
def _coherences(signature):
    return {n: xai.coherence_physique(n, df, resultats[n]) for n in noms}


coherences = _coherences(tuple(sorted((n, float(np.nansum(t["alpha_final"]))) for n, t in resultats.items()))) if mesurer else {}
st.dataframe(
    pd.DataFrame(
        [
            {
                "Stratégie": nom_affichage(n),
                "E1 · Transparence": xai.TRANSPARENCE.get(n, (0, "—", "—"))[1],
                "E2 · Traçabilité": xai.TRANSPARENCE.get(n, (0, "—", "—"))[2],
                "E3 · Cohérence physique": f"{coherences[n][0] * 100:.0f} %" if n in coherences else "—",
            }
            for n in noms
        ]
    ).set_index("Stratégie"),
    width="stretch",
)
if coherences:
    with st.expander("Détail de E3, contrainte par contrainte"):
        st.dataframe(
            pd.DataFrame(
                {nom_affichage(n): {k: f"{v * 100:.0f} %" for k, v in coherences[n][1].items()} for n in noms}
            ).T,
            width="stretch",
        )
        st.caption(
            "Sur 120 instants de traction, chaque contrainte augmente une grandeur (SOC de "
            "+5 points, demande de +1 kW) et vérifie que la puissance confiée à la PB évolue "
            "dans le sens attendu, à 1 % de la demande près."
        )


pied_navigation("vues/5_Comparaison_des_strategies.py")
