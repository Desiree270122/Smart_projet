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
from core.format import SEPARATEURS_PLOTLY, nombre

from core.resultats import (
    assurer_donnees_session,
    calculer_metriques,
    cycle_artemis_affiche,
    debut_partie_test,
    famille,
    nom_affichage,
    restreindre,
    PAIRES_SYMBOLIQUE,
)
from core.style import couleur
from core import xai
from core.navigation import pied_navigation


# Les six métriques du protocole : (clé, libellé court, unité, échelle, décimales)
METRIQUES = [
    ("energie_km_wh", "M1 · Énergie consommée", "Wh/km", 1.0, 1),
    ("rendement_hess", "M2 · Rendement du HESS", "%", 100.0, 2),
    ("rmse_delta_soc", "M3 · Écart entre les SOC", "pts", 100.0, 1),
    ("pertes_convertisseur_wh", "M4 · Pertes du convertisseur", "Wh", 1.0, 0),
    ("violations_totales", "M5 · Violations de contraintes", "", 1.0, 0),
    ("rmse_puissance_kw", "M6 · Erreur de suivi de puissance", "kW", 1.0, 2),
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
n_cycle = min([len(df)] + [len(t["P_EB"]) for t in resultats.values()])

# Les réseaux ont été entraînés sur les premiers 75 % du cycle Artemis : la partie
# test est la seule évaluation sur des situations qu'ils n'ont jamais vues.
debut_eval = 0
if cycle_artemis_affiche(st):
    partie = st.radio(
        "Instants évalués",
        ["Cycle complet", "Partie test seulement (25 % jamais vus à l'entraînement)"],
        horizontal=True,
        help="Les réseaux ont appris sur les premiers 50 % du cycle Artemis et ont été validés sur les 25 % suivants.",
    )
    if partie.startswith("Partie test"):
        debut_eval = debut_partie_test(n_cycle)
        st.caption(
            f"Évaluation sur les instants {debut_eval} à {n_cycle} ({nombre((n_cycle - debut_eval) / 60, 0)} min) : "
            "chaque stratégie y arrive avec ses propres états de charge, hérités du début du cycle."
        )
    else:
        st.caption(
            "Sur le cycle complet, 75 % des instants ont servi à entraîner ou valider les réseaux : "
            "leurs résultats y sont plus favorables que sur des situations nouvelles."
        )
else:
    st.caption("Cycle jamais vu par les réseaux pendant leur entraînement : toute la comparaison est une évaluation sur données nouvelles.")


@st.cache_data(show_spinner="Calcul des métriques…")
def _metriques(signature, debut):
    donnees = {"resultats": resultats, "cycle_df": df}
    if debut:
        donnees = restreindre(donnees, debut, n_cycle)
    m = calculer_metriques(donnees)
    for v in m.values():
        v["violations_totales"] = v["nb_violations"] + v["nb_violations_courant"]
    return m


signature = tuple(sorted((n, float(np.nansum(t["alpha_final"]))) for n, t in resultats.items()))
metriques = _metriques(signature, debut_eval)
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
        "Sauf si les pertes ont été incluses à la simulation (page « Lancer une simulation »), "
        "M1, M2 et M4 utilisent des pertes estimées après coup : résistances internes des "
        "cellules et rendement mesuré du convertisseur (courbe de l'article, fig. 33). Ils "
        "comparent les stratégies, sans prétendre à la valeur absolue."
    )


# 1 — Les six métriques

st.subheader("Les six métriques du protocole")
tableau = pd.DataFrame(
    [
        {
            "Stratégie": nom_affichage(n),
            "Famille": famille(n),
            **{
                _titre(lib, u): nombre(metriques[n][cle] * ech, f)
                for cle, lib, u, ech, f in METRIQUES
            },
        }
        for n in noms
    ]
).set_index("Stratégie")
st.dataframe(tableau, width="stretch")
st.download_button(
    "Télécharger le tableau (CSV)",
    pd.DataFrame(
        [
            {"Stratégie": nom_affichage(n), "Famille": famille(n),
             **{_titre(lib, u): round(metriques[n][cle] * ech, 4) for cle, lib, u, ech, _ in METRIQUES}}
            for n in noms
        ]
    ).to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig"),
    file_name="metriques_M1_M6.csv",
    mime="text/csv",
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
            text=[nombre(v, f) for v in valeurs], textposition="outside", cliponaxis=False,
            hovertemplate="%{y} : %{x}<extra></extra>", showlegend=False,
        ),
        row=k // 3 + 1, col=k % 3 + 1,
    )
fig_m.update_yaxes(autorange="reversed")
fig_m.update_layout(separators=SEPARATEURS_PLOTLY, height=540, margin=dict(t=40, b=20, l=10, r=40), bargap=0.25)
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
                "Violations de SOC": f"{nombre(metriques[n]['nb_violations'], 0)}",
                "Violations de courant": f"{nombre(metriques[n]['nb_violations_courant'], 0)}",
                "Erreur de suivi, RMSE (kW)": f"{nombre(metriques[n]['rmse_puissance_kw'], 2)}",
                "Écart maximal (kW)": f"{nombre(metriques[n]['ecart_puissance_max_kw'], 1)}",
                "Demande non fournie (Wh)": f"{nombre(metriques[n]['energie_non_servie_wh'], 0)}",
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
    "NS-MLP part des règles floues, n'apprend qu'une correction bornée et reste sous le contrôle "
    "d'un garde-fou symbolique ; NS-LSTM reçoit "
    "des états symboliques en entrée supplémentaire."
)


def _variation(cle, a, b):
    """Écart de b par rapport à a : en points pour un pourcentage, en % sinon."""
    if cle in ("rendement_hess", "rmse_delta_soc"):
        return f"{nombre((b - a) * 100, 2, signe=True)} pts"
    if abs(a) < 1e-12:
        return "—" if abs(b) < 1e-12 else f"{nombre(b, 2, signe=True)}"
    return f"{nombre((b - a) / abs(a) * 100, 1, signe=True)} %"


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
                        nom_affichage(seul): nombre(metriques[seul][cle] * ech, f),
                        nom_affichage(ns): nombre(metriques[ns][cle] * ech, f),
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

# E3 est précalculé avec les résultats de référence ; sinon, il se mesure à la demande.
coherences = dict(st.session_state.get("coherence_simulation") or {})
if not all(n in coherences for n in noms):
    if st.toggle("Mesurer E3 · cohérence physique (≈ 30 s au premier calcul)", value=False):

        @st.cache_data(show_spinner="Mesure de la cohérence physique des décisions (E3)…")
        def _coherences(signature):
            return {n: xai.coherence_physique(n, df, resultats[n]) for n in noms}

        coherences = _coherences(signature)
st.dataframe(
    pd.DataFrame(
        [
            {
                "Stratégie": nom_affichage(n),
                "E1 · Transparence": xai.TRANSPARENCE.get(n, (0, "—", "—"))[1],
                "E2 · Traçabilité": xai.TRANSPARENCE.get(n, (0, "—", "—"))[2],
                "E3 · Cohérence physique": f"{nombre(coherences[n][0] * 100, 0)} %" if n in coherences else "—",
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
                {nom_affichage(n): {k: f"{nombre(v * 100, 0)} %" for k, v in coherences[n][1].items()} for n in noms}
            ).T,
            width="stretch",
        )
        st.caption(
            "Sur 120 instants de traction, chaque contrainte augmente une grandeur (SOC de "
            "+5 points, demande de +1 kW) et vérifie que la puissance confiée à la PB évolue "
            "dans le sens attendu, à 1 % de la demande près."
        )


# 5 — Limites et perspectives

st.subheader("Limites et perspectives")
st.markdown(
    "- **Données d'entraînement.** Les réseaux ont appris sur les premiers 50 % du cycle Artemis "
    "et ont été validés sur les 25 % suivants : seule la partie test (sélecteur en haut de page) "
    "et le cycle WLTC mesurent leur comportement sur des situations nouvelles.\n"
    "- **Cibles d'entraînement différentes.** Le GNN imite le modèle physique, le MLP et NS-MLP "
    "apprennent la répartition optimale instant par instant (α*), les LSTM prédisent l'évolution "
    "des SOC. Une partie des écarts vient donc de ce que chaque réseau a appris, pas seulement de "
    "son architecture.\n"
    "- **Décision instantanée.** α* optimise chaque instant sans voir la suite du cycle : une "
    "stratégie qui l'imite peut vider la batterie Puissance trop tôt et manquer ensuite de "
    "puissance pour les pics.\n"
    "- **Pertes.** La référence est simulée sans pertes, comme dans l'article ; les pertes sont "
    "estimées après coup, ou incluses à la demande (page « Lancer une simulation »). Sous 1,2 kW, "
    "le rendement du convertisseur n'est pas mesuré : la valeur à 1,2 kW est conservée.\n"
    "- **Modèle de batterie.** Tensions constantes, sans variation avec le SOC ni la température, "
    "et sans vieillissement.\n"
    "- **Paramètres des packs.** Énergies calculées à partir des cellules (12,6 kWh et 3,6 kWh), "
    "différentes de celles de l'article (13,7 et 3,0 kWh) ; d'où un cycle Artemis de 6 "
    "répétitions au lieu de 7.\n"
    "- **Perspectives.** Cible d'entraînement commune à tous les réseaux, horizon de décision "
    "plus long (commande prédictive, apprentissage par renforcement), modèle de batterie à "
    "tension variable, validation sur d'autres cycles et sur banc."
)


pied_navigation("vues/5_Comparaison_des_strategies.py")
