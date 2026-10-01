"""
Tableau de bord : comprendre en quelques secondes ce qui se passe dans le HESS
pour une stratégie de gestion d'énergie choisie — état du système, répartition
de la puissance, respect des contraintes — avec une synthèse rédigée.
"""

import sys
from pathlib import Path

DOSSIER_PROJET = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DOSSIER_PROJET))

import numpy as np
import plotly.graph_objects as go
import streamlit as st

import ems_core as core
from core.format import nombre, separateurs_plotly
from core.lecture import aide_survol, bulle, courbe_explication, lire_repartition, lire_soc
from core.i18n import tr
from core.resultats import assurer_donnees_session, calculer_metriques, choisir_cycle, nom_affichage
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

st.markdown(
    "<div class='s2-titre'>" + tr("2SMART · Gestion d'énergie d'un HESS", "2SMART · Energy management of a HESS") + "</div>",
    unsafe_allow_html=True,
)
st.markdown(
    "<div class='s2-accroche'>"
    + tr(
        "Deux batteries complémentaires, une décision à chaque seconde : quelle part de la "
        "puissance confier à chacune ? L'application simule, compare et explique cette décision "
        "pour sept stratégies de gestion d'énergie (EMS).",
        "Two complementary batteries, one decision every second: what share of the power should "
        "each one supply? The application simulates, compares and explains this decision for "
        "seven energy management strategies (EMS).",
    )
    + "</div>",
    unsafe_allow_html=True,
)


def _etape(texte, couleur):
    return f"<span class='s2-etape' style='border-color:{couleur};color:{couleur}'>{texte}</span>"


_fleche = "<span class='s2-fleche'>&#8594;</span>"
st.markdown(
    "<div class='s2-flux'>"
    + _fleche.join(
        [
            _etape(tr("Demande du véhicule", "Vehicle demand"), COULEUR_DEMANDE),
            _etape(tr("EMS : décision alpha", "EMS: alpha decision"), COULEUR_DECISION),
            _etape(tr("Batterie Énergie (1 − alpha)", "Energy battery (1 − alpha)"), COULEUR_EB)
            + " " + _etape(tr("Batterie Puissance (alpha)", "Power battery (alpha)"), COULEUR_PB),
            _etape(tr("Convertisseur en série (EB)", "Series converter (EB)"), COULEUR_CONVERTISSEUR),
            _etape(tr("Bus DC et moteur", "DC bus and motor"), COULEUR_SECONDAIRE),
        ]
    )
    + "</div>",
    unsafe_allow_html=True,
)

try:
    source = assurer_donnees_session(st)
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

resultats = st.session_state.get("resultats_simulation")
df = st.session_state.get("cycle_pret")
if not resultats or df is None:
    st.warning(tr("Aucune donnée disponible.", "No data available."))
    st.stop()


# Choix de la stratégie et du cycle

noms = list(resultats.keys())
c_mod, c_cyc, c_btn = st.columns([2, 2, 1])
strategie = c_mod.selectbox(
    tr("Stratégie EMS", "EMS strategy"), noms,
    index=noms.index("EMS_power_limitation") if "EMS_power_limitation" in noms else 0,
    format_func=nom_affichage, key="strategie_accueil",
)
choisir_cycle(st, c_cyc)
with c_btn:
    st.write("")
    if st.button(tr("▶ Autre cycle", "▶ Other cycle"), width="stretch",
                 help=tr("Préparer et lancer une simulation sur un autre cycle", "Prepare and run a simulation on another cycle")):
        st.switch_page("vues/2_Preparation_donnees.py")

traj = resultats[strategie]
n = min(len(df), len(traj["P_EB"]))


@st.cache_data(show_spinner=False)
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
nom = nom_affichage(strategie)
eb, pb = tr("Batterie Énergie", "Energy battery"), tr("Batterie Puissance", "Power battery")


# Indicateurs de l'état du système

st.subheader(tr("État du système avec {s}", "System state with {s}", s=nom))
l1 = st.columns(3)
l1[0].metric(
    tr("Puissance demandée (max)", "Power demand (max)"), f"{nombre(p_dem.max() / 1000, 1)} kW",
    help=tr("Moyenne en traction : {p} kW", "Mean in traction: {p} kW", p=nombre(p_dem[traction].mean() / 1000, 1)),
    border=True,
)
l1[1].metric(tr("SOC final · batterie Énergie", "Final SOC · Energy battery"), f"{nombre(soc_eb[-1] * 100, 1)} %", border=True)
l1[2].metric(tr("SOC final · batterie Puissance", "Final SOC · Power battery"), f"{nombre(soc_pb[-1] * 100, 1)} %", border=True)
l2 = st.columns(3)
l2[0].metric(tr("Rendement du HESS (estimé)", "HESS efficiency (estimated)"), f"{nombre(m['rendement_hess'] * 100, 2)} %", border=True)
l2[1].metric(
    tr("Écart moyen entre les SOC", "Mean gap between the SOCs"), tr("{v} pts", "{v} pts", v=nombre(m["rmse_delta_soc"] * 100, 1)),
    help=tr("Écart quadratique moyen ; écart maximal : {v} points", "Root-mean-square gap; maximum gap: {v} points",
            v=nombre(m["delta_soc_max"] * 100, 1)),
    border=True,
)
respectees = violations == 0 and non_fourni < 1
l2[2].metric(
    tr("Contraintes et demande", "Constraints and demand"),
    tr("✓ respectées", "✓ met") if respectees else tr("✗ non respectées", "✗ not met"),
    help=tr(
        "{v} dépassement(s) de SOC ou de courant ; {e} Wh de demande non fournie",
        "{v} SOC or current limit violation(s); {e} Wh of demand not supplied",
        v=violations, e=nombre(non_fourni, 0),
    ),
    border=True,
)


# Les deux graphiques clés

st.subheader(tr("Évolution des états de charge", "State of charge over time"))
fig_soc = go.Figure(
    [
        go.Scatter(x=np.arange(len(soc_eb)) * core.DT_SECONDS / 60.0, y=soc_eb * 100, name=eb,
                   line=dict(color=COULEUR_EB, width=2), hovertemplate=bulle(eb, "%{y:.1f} %")),
        go.Scatter(x=np.arange(len(soc_pb)) * core.DT_SECONDS / 60.0, y=soc_pb * 100, name=pb,
                   line=dict(color=COULEUR_PB, width=2), hovertemplate=bulle(pb, "%{y:.1f} %")),
        courbe_explication(np.arange(len(soc_eb)) * core.DT_SECONDS / 60.0, soc_eb * 100, lire_soc(traj), acceleree=False),
    ]
)
fig_soc.add_hline(y=core.SOC_EB_MIN * 100, line=dict(color=COULEUR_VIOLATION, dash="dot", width=1),
                  annotation_text=tr("SOC minimal", "Minimum SOC"), annotation_position="bottom right")
fig_soc.update_layout(
    separators=separateurs_plotly(), height=300, margin=dict(t=10, b=40, l=10, r=10), hovermode="x unified",
    xaxis=dict(title=tr("Temps (min)", "Time (min)"), hoverformat=".1f"), yaxis=dict(title="SOC (%)", range=[0, 102]),
    legend=dict(orientation="h", y=1.12, x=0),
)
st.plotly_chart(fig_soc, width="stretch")

st.subheader(tr("Répartition de la puissance", "Power split"))
st.caption(tr(
    "Le véhicule demande une puissance ; l'EMS la répartit entre les deux batteries. Faites "
    "glisser la réglette sous le graphique pour zoomer sur une portion du cycle.",
    "The vehicle demands a power; the EMS splits it between the two batteries. Drag the slider "
    "below the chart to zoom in on part of the cycle.",
) + " " + aide_survol())
# Portion du cycle affichée à l'ouverture (15 minutes), ajustable avec la réglette.
debut_vue = float(min(60.0, max(0.0, temps_min[-1] - 15.0)))
vue = [debut_vue, float(min(debut_vue + 15.0, temps_min[-1]))]
fig_p = go.Figure(
    [
        # Courbe non accélérée : c'est elle que la réglette reproduit en miniature.
        go.Scatter(x=temps_min, y=p_dem / 1000, name=tr("Demande", "Demand"), line=dict(color=COULEUR_DEMANDE, width=1.2),
                   hovertemplate=bulle(tr("Demande", "Demand"), "%{y:.1f} kW")),
        go.Scattergl(x=temps_min, y=p_eb / 1000, name=eb, line=dict(color=COULEUR_EB, width=1.2),
                     hovertemplate=bulle(eb, "%{y:.1f} kW")),
        go.Scattergl(x=temps_min, y=p_pb / 1000, name=pb, line=dict(color=COULEUR_PB, width=1.2),
                     hovertemplate=bulle(pb, "%{y:.1f} kW")),
        courbe_explication(temps_min, p_dem / 1000, lire_repartition(p_dem, traj, n)),
    ]
)
fig_p.update_layout(
    separators=separateurs_plotly(), height=380, margin=dict(t=10, b=10, l=10, r=10), hovermode="x unified",
    yaxis_title=tr("Puissance (kW)", "Power (kW)"),
    xaxis=dict(title=tr("Temps (min)", "Time (min)"), rangeslider=dict(visible=True, thickness=0.08), range=vue, hoverformat=".1f"),
    legend=dict(orientation="h", y=1.12, x=0),
)
st.plotly_chart(fig_p, width="stretch")


# Synthèse rédigée

st.subheader(tr("Analyse de la simulation", "Simulation summary"))
part_pb = float(np.sum(p_pb[traction])) / float(np.sum(p_dem[traction])) * 100 if traction.any() else 0.0
temps_pb = float(np.mean(p_pb[traction] > 100.0)) * 100 if traction.any() else 0.0
distance = float(np.sum(df["speed"].to_numpy(dtype=float)[:n])) * core.DT_SECONDS / 1000 if "speed" in df.columns else float("nan")
phrases = [
    tr(
        "Sur ce cycle ({d} min, {km} km), la demande atteint {pmax} kW en traction et {pmin} kW au freinage.",
        "Over this cycle ({d} min, {km} km), demand reaches {pmax} kW in traction and {pmin} kW when braking.",
        d=nombre(n * core.DT_SECONDS / 60, 0), km=nombre(distance, 0),
        pmax=nombre(p_dem.max() / 1000, 1), pmin=nombre(p_dem.min() / 1000, 1),
    ),
    tr(
        "{s} confie **{part} %** de l'énergie de traction à la batterie Puissance, qui intervient "
        "pendant {t} % du temps de traction.",
        "{s} gives **{part} %** of the traction energy to the Power battery, which is used during "
        "{t} % of the traction time.",
        s=nom, part=nombre(part_pb, 0), t=nombre(temps_pb, 0),
    ),
    tr(
        "L'écart entre les deux SOC atteint au plus **{max} points** (écart quadratique moyen "
        "{rms} points) ; les SOC finaux sont de {eb} % (Énergie) et {pb} % (Puissance).",
        "The gap between the two SOCs reaches at most **{max} points** (root-mean-square gap "
        "{rms} points); the final SOCs are {eb} % (Energy) and {pb} % (Power).",
        max=nombre(m["delta_soc_max"] * 100, 1), rms=nombre(m["rmse_delta_soc"] * 100, 1),
        eb=nombre(soc_eb[-1] * 100, 1), pb=nombre(soc_pb[-1] * 100, 1),
    ),
    tr(
        "Aucun dépassement de SOC ni de courant, et toute la puissance demandée a été fournie.",
        "No SOC or current limit was exceeded, and all the demanded power was supplied.",
    ) if respectees else tr(
        "**Attention** : {v} dépassement(s) de limite et {e} Wh de demande non fournie ; les autres "
        "indicateurs de cette stratégie sont donc flattés.",
        "**Warning**: {v} limit violation(s) and {e} Wh of demand not supplied; the other "
        "indicators of this strategy are therefore flattering.",
        v=violations, e=nombre(non_fourni, 0),
    ),
    tr(
        "Rendement estimé du HESS : **{r} %** ({p} Wh de pertes estimées, dont {c} Wh dans le convertisseur).",
        "Estimated HESS efficiency: **{r} %** ({p} Wh of estimated losses, including {c} Wh in the converter).",
        r=nombre(m["rendement_hess"] * 100, 2), p=nombre(m["pertes_totales_wh"], 0), c=nombre(m["pertes_convertisseur_wh"], 0),
    ),
]
with st.container(border=True):
    st.markdown(" ".join(phrases))

b1, b2, b3 = st.columns(3)
if b1.button(tr("🔍 Pourquoi ces décisions ?", "🔍 Why these decisions?"), width="stretch", type="primary"):
    st.switch_page("vues/7_Explicabilite.py")
if b2.button(tr("⚖️ Comparer les stratégies", "⚖️ Compare the strategies"), width="stretch"):
    st.switch_page("vues/5_Comparaison_des_strategies.py")
if b3.button(tr("📈 Simuler un autre cycle", "📈 Simulate another cycle"), width="stretch"):
    st.switch_page("vues/2_Preparation_donnees.py")


with st.expander(tr("Le projet en bref", "The project in brief")):
    st.markdown(tr(
        "- **Le problème** : un véhicule électrique équipé de deux batteries complémentaires doit "
        "décider, à chaque instant, laquelle fournit la puissance demandée.\n"
        "- **Pourquoi c'est difficile** : les objectifs se contredisent (autonomie, durée de vie, "
        "rendement), les contraintes physiques sont strictes et les cycles de conduite très variables.\n"
        "- **La réponse de 2SMART** : comparer quatre familles d'approches (règles fixes, ontologie "
        "seule, apprentissage seul, hybride neuro-symbolique) et expliquer chaque décision, sous le "
        "contrôle d'un filtre physique de sécurité.\n"
        "- **L'architecture** : cascade à source de courant contrôlée (Fonseca de Freitas et al., "
        "IEEE Access 2024) ; le convertisseur, en série, ne traite qu'environ 10 % de la puissance "
        "de la batterie Énergie.",
        "- **The problem**: an electric vehicle with two complementary batteries must decide, at "
        "every instant, which one supplies the demanded power.\n"
        "- **Why it is hard**: the objectives conflict (range, lifetime, efficiency), the physical "
        "constraints are strict and driving cycles vary widely.\n"
        "- **The 2SMART approach**: compare four families of approaches (fixed rules, ontology "
        "only, learning only, neuro-symbolic hybrid) and explain every decision, under the control "
        "of a physical safety filter.\n"
        "- **The architecture**: controlled current source cascade (Fonseca de Freitas et al., "
        "IEEE Access 2024); the converter, in series, processes only about 10 % of the Energy "
        "battery's power.",
    ))
    e1, e2 = st.columns(2)
    e1.latex(r"P_{PB} = \alpha \times P_{dem}")
    e2.latex(r"P_{EB} = (1 - \alpha) \times P_{dem}")
    st.caption(tr(
        "alpha = 0 : toute la puissance vient de la batterie Énergie ; alpha = 1 : toute la "
        "puissance vient de la batterie Puissance. La décision passe ensuite par le filtre "
        "physique de sécurité.",
        "alpha = 0: all the power comes from the Energy battery; alpha = 1: all the power comes "
        "from the Power battery. The decision then goes through the physical safety filter.",
    ))
