"""
Tableau de bord : comprendre en quelques secondes ce qui se passe dans le HESS
pour un modèle de gestion d'énergie choisi — état du système, répartition de la
puissance, respect des contraintes — avec une synthèse rédigée.
"""

import sys
from pathlib import Path

DOSSIER_PROJET = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DOSSIER_PROJET))

import numpy as np
import plotly.graph_objects as go
import streamlit as st

import ems_core as core
from core.resultats import assurer_donnees_session, calculer_metriques, nom_affichage
from core.style import (
    COULEUR_CONVERTISSEUR,
    COULEUR_DECISION,
    COULEUR_DEMANDE,
    COULEUR_EB,
    COULEUR_PB,
    COULEUR_SECONDAIRE,
    COULEUR_VIOLATION,
)


st.markdown(
    """
    <style>
    .s2-titre{font-size:2.1rem;font-weight:800;letter-spacing:-.5px;margin:0}
    .s2-accroche{font-size:1.05rem;color:#AEB6C4;margin:.2rem 0 .8rem}
    .s2-flux{display:flex;align-items:center;flex-wrap:wrap;gap:6px;margin:.2rem 0 1rem}
    .s2-etape{border:1px solid;border-radius:9px;padding:6px 11px;font-weight:600;font-size:.88rem}
    .s2-fleche{color:#8B93A7;font-weight:800}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='s2-titre'>2SMART · Gestion d'énergie d'un HESS</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='s2-accroche'>Deux batteries complémentaires, une décision à chaque seconde : "
    "quelle part de la puissance confier à chacune ? L'application simule, compare et "
    "explique cette décision pour sept stratégies.</div>",
    unsafe_allow_html=True,
)


def _etape(texte, couleur):
    return f"<span class='s2-etape' style='border-color:{couleur};color:{couleur}'>{texte}</span>"


_fleche = "<span class='s2-fleche'>&#8594;</span>"
st.markdown(
    "<div class='s2-flux'>"
    + _fleche.join(
        [
            _etape("Demande du véhicule", COULEUR_DEMANDE),
            _etape("EMS : décision alpha", COULEUR_DECISION),
            _etape("Batterie Énergie (1 − alpha)", COULEUR_EB) + " " + _etape("Batterie Puissance (alpha)", COULEUR_PB),
            _etape("Convertisseur en série (EB)", COULEUR_CONVERTISSEUR),
            _etape("Bus DC et moteur", COULEUR_SECONDAIRE),
        ]
    )
    + "</div>",
    unsafe_allow_html=True,
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


# Choix du modèle et du cycle

noms = list(resultats.keys())
c_mod, c_cyc, c_btn = st.columns([2, 2, 1])
strategie = c_mod.selectbox(
    "Modèle EMS", noms,
    index=noms.index("EMS_power_limitation") if "EMS_power_limitation" in noms else 0,
    format_func=nom_affichage,
)
cycle = (
    "Artemis urbain + routier, répété 7 fois (référence)"
    if source == "référence précalculée" else "Cycle personnalisé (simulation de la session)"
)
c_cyc.selectbox("Cycle de conduite", [cycle], help="Pour un autre cycle, lancez une simulation.")
with c_btn:
    st.write("")
    if st.button("▶ Autre cycle", width="stretch", help="Préparer et lancer une simulation sur un autre cycle"):
        st.switch_page("vues/2_Preparation_donnees.py")

traj = resultats[strategie]
n = min(len(df), len(traj["P_EB"]))


@st.cache_data(show_spinner="Calcul des indicateurs…")
def _indicateurs(strategie, signature):
    return calculer_metriques({"resultats": {strategie: resultats[strategie]}, "cycle_df": df})[strategie]


m = _indicateurs(strategie, float(np.nansum(traj["alpha_final"])))
p_dem = df["hasPower"].to_numpy(dtype=float)[:n]
p_eb = np.asarray(traj["P_EB"], dtype=float)[:n]
p_pb = np.asarray(traj["P_PB"], dtype=float)[:n]
soc_eb = np.asarray(traj["SOC_EB"], dtype=float)
soc_pb = np.asarray(traj["SOC_PB"], dtype=float)
temps_min = (df["time"].to_numpy(dtype=float)[:n] if "time" in df.columns else np.arange(n)) / 60.0
traction = p_dem > core.EPS_POWER_W
violations = int(m["nb_violations"] + m["nb_violations_courant"])
non_fourni = m["energie_non_servie_wh"]


# Indicateurs de l'état du système

st.subheader(f"État du système avec {nom_affichage(strategie)}")
l1 = st.columns(3)
l1[0].metric("Puissance demandée (max)", f"{p_dem.max() / 1000:.1f} kW",
             help=f"Moyenne en traction : {p_dem[traction].mean() / 1000:.1f} kW", border=True)
l1[1].metric("SOC final · batterie Énergie", f"{soc_eb[-1] * 100:.1f} %", border=True)
l1[2].metric("SOC final · batterie Puissance", f"{soc_pb[-1] * 100:.1f} %", border=True)
l2 = st.columns(3)
l2[0].metric("Rendement du HESS (estimé)", f"{m['rendement_hess'] * 100:.2f} %", border=True)
l2[1].metric("Écart entre les SOC (RMSE)", f"{m['rmse_delta_soc'] * 100:.1f} pts",
             help=f"Écart maximal : {m['delta_soc_max'] * 100:.1f} points", border=True)
l2[2].metric(
    "Contraintes et demande",
    "✓ respectées" if violations == 0 and non_fourni < 1 else "✗ non respectées",
    help=f"{violations} violation(s) de SOC ou de courant ; {non_fourni:.0f} Wh de demande non fournie",
    border=True,
)


# Les deux graphiques clés

st.subheader("Évolution des états de charge")
fig_soc = go.Figure(
    [
        go.Scatter(x=np.arange(len(soc_eb)) * core.DT_SECONDS / 60.0, y=soc_eb * 100, name="Batterie Énergie",
                   line=dict(color=COULEUR_EB, width=2)),
        go.Scatter(x=np.arange(len(soc_pb)) * core.DT_SECONDS / 60.0, y=soc_pb * 100, name="Batterie Puissance",
                   line=dict(color=COULEUR_PB, width=2)),
    ]
)
fig_soc.add_hline(y=core.SOC_EB_MIN * 100, line=dict(color=COULEUR_VIOLATION, dash="dot", width=1),
                  annotation_text="SOC minimal", annotation_position="bottom right")
fig_soc.update_layout(
    height=300, margin=dict(t=10, b=40, l=10, r=10), hovermode="x unified",
    xaxis_title="Temps (min)", yaxis=dict(title="SOC (%)", range=[0, 102]),
    legend=dict(orientation="h", y=1.12, x=0),
)
st.plotly_chart(fig_soc, width="stretch")

st.subheader("Répartition de la puissance")
st.caption(
    "Le véhicule demande une puissance ; l'EMS la répartit entre les deux batteries. "
    "Faites glisser la réglette sous le graphique pour zoomer sur une portion du cycle."
)
fig_p = go.Figure(
    [
        go.Scattergl(x=temps_min, y=p_dem / 1000, name="Demande", line=dict(color=COULEUR_DEMANDE, width=1.2)),
        go.Scattergl(x=temps_min, y=p_eb / 1000, name="Batterie Énergie", line=dict(color=COULEUR_EB, width=1.2)),
        go.Scattergl(x=temps_min, y=p_pb / 1000, name="Batterie Puissance", line=dict(color=COULEUR_PB, width=1.2)),
    ]
)
fig_p.update_layout(
    height=380, margin=dict(t=10, b=10, l=10, r=10), hovermode="x unified",
    yaxis_title="Puissance (kW)",
    xaxis=dict(title="Temps (min)", rangeslider=dict(visible=True, thickness=0.08), range=[60, 90]),
    legend=dict(orientation="h", y=1.12, x=0),
)
st.plotly_chart(fig_p, width="stretch")


# Synthèse rédigée

st.subheader("Analyse de la simulation")
part_pb = float(np.sum(p_pb[traction])) / float(np.sum(p_dem[traction])) * 100 if traction.any() else 0.0
temps_pb = float(np.mean(p_pb[traction] > 100.0)) * 100 if traction.any() else 0.0
distance = float(np.sum(df["speed"].to_numpy(dtype=float)[:n])) * core.DT_SECONDS / 1000 if "speed" in df.columns else float("nan")
phrases = [
    f"Sur ce cycle ({n * core.DT_SECONDS / 60:.0f} min, {distance:.0f} km), la demande atteint "
    f"{p_dem.max() / 1000:.1f} kW en traction et {p_dem.min() / 1000:.1f} kW au freinage.",
    f"{nom_affichage(strategie)} confie **{part_pb:.0f} %** de l'énergie de traction à la batterie "
    f"Puissance, qui intervient pendant {temps_pb:.0f} % du temps de traction.",
    f"L'écart entre les deux SOC atteint au plus **{m['delta_soc_max'] * 100:.1f} points** (écart "
    f"quadratique moyen {m['rmse_delta_soc'] * 100:.1f} points) ; les SOC finaux sont de "
    f"{soc_eb[-1] * 100:.1f} % (Énergie) et {soc_pb[-1] * 100:.1f} % (Puissance).",
    (
        "Aucune violation de SOC ni de courant, et toute la puissance demandée a été fournie."
        if violations == 0 and non_fourni < 1
        else f"**Attention** : {violations} violation(s) de contrainte et {non_fourni:.0f} Wh de "
        "demande non fournie ; les autres indicateurs de ce modèle sont donc flattés."
    ),
    f"Rendement estimé du HESS : **{m['rendement_hess'] * 100:.2f} %** ({m['pertes_totales_wh']:.0f} Wh "
    f"de pertes estimées, dont {m['pertes_convertisseur_wh']:.0f} Wh dans le convertisseur).",
]
with st.container(border=True):
    st.markdown(" ".join(phrases))

b1, b2, b3 = st.columns(3)
if b1.button("🔍 Pourquoi ces décisions ?", width="stretch", type="primary"):
    st.switch_page("vues/7_Explicabilite.py")
if b2.button("⚖️ Comparer les modèles", width="stretch"):
    st.switch_page("vues/5_Comparaison_des_strategies.py")
if b3.button("📈 Simuler un autre cycle", width="stretch"):
    st.switch_page("vues/2_Preparation_donnees.py")


with st.expander("Le projet en bref"):
    st.markdown(
        "- **Le problème** : un véhicule électrique équipé de deux batteries complémentaires "
        "doit décider, à chaque instant, laquelle fournit la puissance demandée.\n"
        "- **Pourquoi c'est difficile** : les objectifs se contredisent (autonomie, durée de "
        "vie, rendement), les contraintes physiques sont strictes et les cycles de conduite "
        "très variables.\n"
        "- **La réponse de 2SMART** : comparer quatre familles d'approches (règles fixes, "
        "ontologie seule, apprentissage seul, hybride neuro-symbolique) et expliquer chaque "
        "décision, sous le contrôle d'un filtre physique de sécurité.\n"
        "- **L'architecture** : cascade à source de courant contrôlée (Fonseca de Freitas et "
        "al., IEEE Access 2024) ; le convertisseur, en série, ne traite qu'environ 10 % de la "
        "puissance de la batterie Énergie."
    )
    e1, e2 = st.columns(2)
    e1.latex(r"P_{PB} = \alpha \times P_{dem}")
    e2.latex(r"P_{EB} = (1 - \alpha) \times P_{dem}")
    st.caption(
        "alpha = 0 : toute la puissance vient de la batterie Énergie ; alpha = 1 : toute la "
        "puissance vient de la batterie Puissance. La décision passe ensuite par le filtre "
        "physique de sécurité."
    )
