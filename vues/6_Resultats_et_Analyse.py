"""
Résultats de simulation : évolution du HESS au cours du cycle pour la stratégie
choisie — répartition de la puissance, états de charge, charge et décharge de
chaque composant, pertes. La comparaison entre stratégies est dans
« Comparaison des stratégies EMS ».
"""

import sys
from pathlib import Path

DOSSIER_PROJET = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DOSSIER_PROJET))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import ems_core as core
from core.format import nombre, separateurs_plotly
from core.i18n import langue, tr
from core.navigation import pied_navigation
from core.pertes import pertes_par_pas
from core.resultats import (
    assurer_donnees_session, calculer_metriques, choisir_cycle, cycle_artemis_affiche, debut_partie_test,
    nom_affichage,
)
from core.style import (
    COULEUR_CONVERTISSEUR,
    COULEUR_DEMANDE,
    COULEUR_EB,
    COULEUR_PB,
    COULEUR_REFERENCE,
    COULEUR_SECONDAIRE,
    COULEUR_VIOLATION,
)


st.title(tr("📈 Résultats de simulation", "📈 Simulation results"))
st.caption(tr(
    "Évolution du HESS au cours du cycle pour la stratégie choisie.",
    "How the HESS evolves over the cycle for the selected strategy.",
))

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

noms = list(resultats.keys())
c_s, c_cycle = st.columns(2)
strategie = c_s.selectbox(
    tr("Stratégie", "Strategy"), noms,
    index=noms.index("EMS_power_limitation") if "EMS_power_limitation" in noms else 0,
    format_func=nom_affichage, key="strategie_resultats",
)
choisir_cycle(st, c_cycle)

traj = resultats[strategie]
n = min(len(df), len(traj["P_EB"]))


@st.cache_data(show_spinner=False)
def _indicateurs(strategie, signature):
    return calculer_metriques({"resultats": {strategie: resultats[strategie]}, "cycle_df": df})[strategie]


m = _indicateurs(strategie, float(np.nansum(traj["alpha_final"])))
if m["energie_non_servie_wh"] >= 1.0:
    st.warning(tr(
        "{s} ne fournit pas toute la puissance demandée ({e} Wh manquants, jusqu'à {p} kW à un "
        "instant) : son énergie consommée et son rendement en sont flattés.",
        "{s} does not deliver all the demanded power ({e} Wh missing, up to {p} kW at one time "
        "step): its energy consumption and efficiency are flattered as a result.",
        s=nom_affichage(strategie), e=nombre(m["energie_non_servie_wh"], 0), p=nombre(m["ecart_puissance_max_kw"], 1),
    ))


# Les six métriques du protocole

l1 = st.columns(3)
l1[0].metric(tr("M1 · Énergie consommée", "M1 · Energy consumed"), f"{nombre(m['energie_km_wh'], 1)} Wh/km", border=True,
             help=tr("{e} kWh sur le cycle, pertes comprises", "{e} kWh over the cycle, losses included",
                     e=nombre(m["energie_consommee_wh"] / 1000, 2)))
l1[1].metric(tr("M2 · Rendement du HESS", "M2 · HESS efficiency"), f"{nombre(m['rendement_hess'] * 100, 2)} %", border=True,
             help=tr("Énergie de traction fournie, divisée par cette énergie augmentée des pertes",
                     "Traction energy delivered, divided by that energy plus the losses"))
l1[2].metric(tr("M3 · Écart entre les SOC", "M3 · Gap between SOCs"), f"{nombre(m['rmse_delta_soc'] * 100, 1)} pts", border=True,
             help=tr("Écart quadratique moyen ; écart maximal {x} points", "Root-mean-square gap; maximum gap {x} points",
                     x=nombre(m["delta_soc_max"] * 100, 1)))
l2 = st.columns(3)
l2[0].metric(tr("M4 · Pertes du convertisseur", "M4 · Converter losses"), f"{nombre(m['pertes_convertisseur_wh'], 0)} Wh", border=True)
l2[1].metric(tr("M5 · Dépassements de limite", "M5 · Limit violations"),
             nombre(m["nb_violations"] + m["nb_violations_courant"], 0), border=True,
             help=tr("{a} de SOC, {b} de courant", "{a} of SOC, {b} of current",
                     a=nombre(m["nb_violations"], 0), b=nombre(m["nb_violations_courant"], 0)))
l2[2].metric(tr("M6 · Écart de puissance", "M6 · Power shortfall"), f"{nombre(m['rmse_puissance_kw'], 2)} kW", border=True,
             help=tr("Écart quadratique moyen entre puissance fournie et demandée en traction ; écart maximal {x} kW",
                     "Root-mean-square gap between delivered and demanded traction power; maximum gap {x} kW",
                     x=nombre(m["ecart_puissance_max_kw"], 1)))

temps_min = (df["time"].to_numpy(dtype=float)[:n] if "time" in df.columns else np.arange(n)) / 60.0
p_dem = df["hasPower"].to_numpy(dtype=float)[:n]
p_eb = np.asarray(traj["P_EB"], dtype=float)[:n]
p_pb = np.asarray(traj["P_PB"], dtype=float)[:n]
p_conv = (core.V_EB_PACK_NOM - core.V_PB_PACK_NOM) * np.asarray(traj["I_EB"], dtype=float)[:n]

# Portion du cycle affichée à l'ouverture (15 minutes), ajustable avec la réglette.
debut_vue = float(min(60.0, max(0.0, temps_min[-1] - 15.0)))
vue = [debut_vue, float(min(debut_vue + 15.0, temps_min[-1]))]
nom_eb, nom_pb = tr("Batterie Énergie", "Energy battery"), tr("Batterie Puissance", "Power battery")
nom_conv, axe_temps = tr("Convertisseur", "Converter"), tr("Temps (min)", "Time (min)")
aide_reglette = tr(
    "Faites glisser la réglette sous le graphique pour parcourir le cycle.",
    "Drag the slider under the chart to move through the cycle.",
)


# 1 — Répartition de la puissance

st.subheader(tr(
    "Comment la demande de puissance est-elle répartie entre les deux batteries ?",
    "How is the power demand shared between the two batteries?",
))
st.caption(aide_reglette)
fig_p = go.Figure([
    # Courbe non accélérée : c'est elle que la réglette reproduit en miniature.
    go.Scatter(x=temps_min, y=p_dem / 1000, name=tr("Demande", "Demand"), line=dict(color=COULEUR_DEMANDE, width=1.2)),
    go.Scattergl(x=temps_min, y=p_eb / 1000, name=nom_eb, line=dict(color=COULEUR_EB, width=1.2)),
    go.Scattergl(x=temps_min, y=p_pb / 1000, name=nom_pb, line=dict(color=COULEUR_PB, width=1.2)),
])
fig_p.add_hline(y=core.P_EB_MAX_W / 1000, line=dict(color=COULEUR_SECONDAIRE, dash="dot", width=1),
                annotation_text=tr("limite de la batterie Énergie", "Energy battery limit"), annotation_position="top left")
if cycle_artemis_affiche(st):
    fig_p.add_vrect(
        x0=temps_min[debut_partie_test(n)], x1=temps_min[-1], fillcolor=COULEUR_SECONDAIRE, opacity=0.12, line_width=0,
        annotation_text=tr("dernier quart : jamais vu à l'apprentissage", "last quarter: never seen in training"),
        annotation_position="top right",
    )
fig_p.update_layout(
    separators=separateurs_plotly(), height=380, margin=dict(t=10, b=10, l=10, r=10), hovermode="x unified",
    yaxis_title=tr("Puissance (kW)", "Power (kW)"),
    xaxis=dict(title=axe_temps, rangeslider=dict(visible=True, thickness=0.08), range=vue),
    legend=dict(orientation="h", y=1.12, x=0),
)
st.plotly_chart(fig_p, width="stretch")


# 2 — États de charge

st.subheader(tr("Évolution des états de charge", "State of charge over time"))
x_soc = np.arange(len(traj["SOC_EB"])) * core.DT_SECONDS / 60.0
fig_soc = go.Figure([
    go.Scatter(x=x_soc, y=np.asarray(traj["SOC_EB"], float) * 100, name=nom_eb, line=dict(color=COULEUR_EB, width=2)),
    go.Scatter(x=x_soc, y=np.asarray(traj["SOC_PB"], float) * 100, name=nom_pb, line=dict(color=COULEUR_PB, width=2)),
])
fig_soc.add_hline(y=core.SOC_EB_MIN * 100, line=dict(color=COULEUR_VIOLATION, dash="dot", width=1),
                  annotation_text=tr("SOC minimal", "Minimum SOC"), annotation_position="bottom right")
fig_soc.add_hline(y=core.SOC_EB_MAX * 100, line=dict(color=COULEUR_REFERENCE, dash="dot", width=1),
                  annotation_text=tr("SOC maximal", "Maximum SOC"), annotation_position="top right")
fig_soc.update_layout(
    separators=separateurs_plotly(), height=320, margin=dict(t=10, b=40, l=10, r=10), hovermode="x unified",
    xaxis_title=axe_temps, yaxis=dict(title="SOC (%)", range=[0, 105]),
    legend=dict(orientation="h", y=1.12, x=0),
)
st.plotly_chart(fig_soc, width="stretch")


# 3 — Charge et décharge de chaque composant

st.subheader(tr(
    "Charge et décharge des batteries et du convertisseur",
    "Charge and discharge of the batteries and the converter",
))
st.caption(tr(
    "Au-dessus de zéro, le composant se décharge : il fournit de la puissance. En dessous, il se "
    "recharge : il reçoit l'énergie récupérée au freinage. Les pointillés marquent ses limites. ",
    "Above zero the component discharges: it supplies power. Below zero it charges: it receives "
    "the energy recovered while braking. The dotted lines mark its limits. ",
) + aide_reglette)

# (nom, puissance, couleur, limite de décharge, limite de recharge, seuil d'activité en W)
composants = [
    (nom_eb, p_eb, COULEUR_EB, core.P_EB_MAX_W, core.P_EB_MIN_W, core.EPS_POWER_W),
    (nom_pb, p_pb, COULEUR_PB, core.P_PB_MAX_W, core.P_PB_MIN_W, core.EPS_POWER_W),
    (nom_conv, p_conv, COULEUR_CONVERTISSEUR, core.P_CONV_MAX_W, core.P_CONV_MIN_W, core.EPS_POWER_W * 0.1),
]


def _transparent(couleur_hex, opacite):
    r, g, b = (int(couleur_hex[i:i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{opacite})"


fig_cd = make_subplots(
    rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.07,
    subplot_titles=[f"{nom} (kW)" for nom, *_ in composants],
)
for ligne, (nom, puissance, coul, lim_dech, lim_rech, _) in enumerate(composants, start=1):
    # Dernière ligne en courbes non accélérées : la réglette les reproduit en miniature.
    Courbe = go.Scatter if ligne == len(composants) else go.Scattergl
    fig_cd.add_trace(
        Courbe(
            x=temps_min, y=np.clip(puissance, 0, None) / 1000, name=tr("Décharge", "Discharge"), legendgroup="d",
            showlegend=False, mode="lines", line=dict(color=coul, width=1), fill="tozeroy", fillcolor=_transparent(coul, 0.55),
            hovertemplate=tr("décharge", "discharge") + " %{y:.2f} kW<extra></extra>",
        ),
        row=ligne, col=1,
    )
    fig_cd.add_trace(
        Courbe(
            x=temps_min, y=np.clip(puissance, None, 0) / 1000, name=tr("Recharge", "Charge"), legendgroup="c",
            showlegend=False, mode="lines", line=dict(color=coul, width=1, dash="dot"), fill="tozeroy",
            fillcolor=_transparent(coul, 0.22),
            hovertemplate=tr("recharge", "charge") + " %{y:.2f} kW<extra></extra>",
        ),
        row=ligne, col=1,
    )
    # Les limites ne sont tracées que si elles sont à l'échelle des puissances atteintes.
    amplitude = max(float(np.max(np.abs(puissance))), 1.0)
    for limite, texte, position in (
        (lim_dech, tr("limite de décharge", "discharge limit"), "top left"),
        (lim_rech, tr("limite de recharge", "charge limit"), "bottom left"),
    ):
        if abs(limite) <= 1.5 * amplitude:
            fig_cd.add_hline(
                y=limite / 1000, line=dict(color=COULEUR_SECONDAIRE, dash="dot", width=1), row=ligne, col=1,
                annotation_text=f"{texte} {nombre(limite / 1000, 2 if abs(limite) < 5000 else 1)} kW",
                annotation_position=position, annotation_font_size=10,
            )
fig_cd.update_xaxes(range=vue)
fig_cd.update_xaxes(title_text=axe_temps, rangeslider=dict(visible=True, thickness=0.06), row=3, col=1)
fig_cd.update_layout(separators=separateurs_plotly(), height=720, margin=dict(t=30, b=10, l=10, r=10),
                     hovermode="x unified", showlegend=False)
fig_cd.update_annotations(selector=dict(yanchor="bottom", xanchor="center"), font_size=13)
st.plotly_chart(fig_cd, width="stretch")

heures = core.DT_SECONDS / 3600.0
col_composant = tr("Composant", "Component")
st.dataframe(
    pd.DataFrame([
        {
            col_composant: nom,
            tr("Énergie fournie en décharge (Wh)", "Energy supplied while discharging (Wh)"):
                nombre(np.clip(puissance, 0, None).sum() * heures, 0),
            tr("Énergie reçue en recharge (Wh)", "Energy received while charging (Wh)"):
                nombre(-np.clip(puissance, None, 0).sum() * heures, 0),
            tr("Temps en décharge", "Time discharging"): f"{nombre(np.mean(puissance > seuil) * 100, 0)} %",
            tr("Temps en recharge", "Time charging"): f"{nombre(np.mean(puissance < -seuil) * 100, 0)} %",
            tr("Puissance maximale atteinte (kW)", "Peak power reached (kW)"):
                f"{nombre(puissance.max() / 1000, 2)} / {nombre(puissance.min() / 1000, 2)}",
            tr("Limites décharge / recharge (kW)", "Discharge / charge limits (kW)"):
                f"{nombre(lim_dech / 1000, 2)} / {nombre(lim_rech / 1000, 2)}",
        }
        for nom, puissance, _, lim_dech, lim_rech, seuil in composants
    ]).set_index(col_composant),
    width="stretch",
)
st.caption(tr(
    "Le convertisseur, placé en série avec la batterie Énergie, ne traite que la différence de "
    "tension entre les deux batteries : sa puissance vaut (V_EB − V_PB)·I_EB, environ {p} % de celle "
    "de la batterie Énergie, dont elle suit le sens.",
    "The converter, in series with the Energy battery, only processes the voltage difference "
    "between the two batteries: its power is (V_EB − V_PB)·I_EB, about {p} % of the Energy "
    "battery's power, and has the same sign.",
    p=nombre((core.V_EB_PACK_NOM - core.V_PB_PACK_NOM) / core.V_EB_PACK_NOM * 100, 1),
))


# 4 — Pertes

st.subheader(tr("Pertes", "Losses"))
st.caption(tr(
    "Pertes cumulées au fil du cycle : échauffement de chaque batterie (R·I²) et pertes du convertisseur.",
    "Cumulative losses over the cycle: heating of each battery (R·I²) and converter losses.",
))
pas = pertes_par_pas(traj)
cumul = {k: np.cumsum(v[:n]) * heures for k, v in pas.items()}
total = cumul["eb"] + cumul["pb"] + cumul["convertisseur"]
pas_graphe = max(1, n // 2000)
fig_l = go.Figure([
    go.Scatter(x=temps_min[::pas_graphe], y=cumul["eb"][::pas_graphe], name=nom_eb, line=dict(color=COULEUR_EB, width=1.8)),
    go.Scatter(x=temps_min[::pas_graphe], y=cumul["pb"][::pas_graphe], name=nom_pb, line=dict(color=COULEUR_PB, width=1.8)),
    go.Scatter(x=temps_min[::pas_graphe], y=cumul["convertisseur"][::pas_graphe], name=nom_conv,
               line=dict(color=COULEUR_CONVERTISSEUR, width=1.8)),
    go.Scatter(x=temps_min[::pas_graphe], y=total[::pas_graphe], name=tr("Pertes totales", "Total losses"),
               line=dict(color=COULEUR_REFERENCE, width=2.2, dash="dash")),
])
fig_l.update_layout(
    separators=separateurs_plotly(), height=320, margin=dict(t=10, b=40, l=10, r=10), hovermode="x unified",
    xaxis_title=axe_temps, yaxis_title=tr("Pertes cumulées (Wh)", "Cumulative losses (Wh)"),
    legend=dict(orientation="h", y=1.12, x=0),
)
st.plotly_chart(fig_l, width="stretch")
p1, p2, p3, p4 = st.columns(4)
p1.metric(nom_eb, f"{nombre(cumul['eb'][-1], 0)} Wh")
p2.metric(nom_pb, f"{nombre(cumul['pb'][-1], 0)} Wh")
p3.metric(nom_conv, f"{nombre(cumul['convertisseur'][-1], 0)} Wh")
p4.metric(tr("Pertes totales", "Total losses"), f"{nombre(total[-1], 0)} Wh")

pertes_dans_soc = bool((st.session_state.get("meta_simulation") or {}).get("pertes_dans_soc"))
reserve = (
    (float(traj["SOC_EB"][-1]) - core.SOC_EB_MIN) * core.ENERGY_EB_WH
    + (float(traj["SOC_PB"][-1]) - core.SOC_PB_MIN) * core.ENERGY_PB_WH
)
if pertes_dans_soc:
    st.info(tr(
        "Les pertes sont comptées dans cette simulation : il reste {r} Wh utilisables en fin de cycle.",
        "Losses are counted in this simulation: {r} Wh remain usable at the end of the cycle.",
        r=nombre(reserve, 0),
    ))
elif reserve - total[-1] >= 0:
    st.success(tr(
        "Cette simulation ne compte pas les pertes dans les états de charge. Il reste {r} Wh "
        "utilisables en fin de cycle, plus que les {t} Wh de pertes estimées : en les comptant, la "
        "stratégie finirait le cycle ({m} Wh de marge).",
        "This simulation does not count the losses in the states of charge. {r} Wh remain usable at "
        "the end of the cycle, more than the {t} Wh of estimated losses: counting them, the strategy "
        "would still complete the cycle ({m} Wh margin).",
        r=nombre(reserve, 0), t=nombre(total[-1], 0), m=nombre(reserve - total[-1], 0),
    ))
else:
    st.warning(tr(
        "Cette simulation ne compte pas les pertes dans les états de charge. Il reste {r} Wh "
        "utilisables en fin de cycle, moins que les {t} Wh de pertes estimées : en les comptant, il "
        "manquerait environ {m} Wh. Pour le vérifier, relancez la simulation en incluant les pertes "
        "(page « Lancer une simulation »).",
        "This simulation does not count the losses in the states of charge. {r} Wh remain usable at "
        "the end of the cycle, less than the {t} Wh of estimated losses: counting them, about {m} Wh "
        "would be missing. To check, run the simulation again with losses included (“Run a "
        "simulation” page).",
        r=nombre(reserve, 0), t=nombre(total[-1], 0), m=nombre(total[-1] - reserve, 0),
    ))
with st.expander(tr("Hypothèses du calcul des pertes", "Assumptions of the loss calculation")):
    st.markdown(tr(
        "- Résistance interne de chaque pack, calculée à partir des cellules : {reb} mΩ × {seb}/{peb} "
        "pour la batterie Énergie, {rpb} mΩ × {spb}/{ppb} pour la batterie Puissance.\n"
        "- Convertisseur : rendement mesuré dans l'article de référence (fig. 33), de 91,5 % à 1,2 kW "
        "à 95,5 % vers 2,5 kW, appliqué à la seule puissance qu'il traite ; en dessous de 1,2 kW, la "
        "valeur à 1,2 kW est conservée.\n"
        "- Tensions constantes, sans variation avec le SOC.",
        "- Internal resistance of each pack, computed from the cells: {reb} mΩ × {seb}/{peb} for the "
        "Energy battery, {rpb} mΩ × {spb}/{ppb} for the Power battery.\n"
        "- Converter: efficiency measured in the reference paper (fig. 33), from 91.5 % at 1.2 kW to "
        "95.5 % around 2.5 kW, applied only to the power it processes; below 1.2 kW, the value at "
        "1.2 kW is kept.\n"
        "- Constant voltages, with no variation with SOC.",
        reb=nombre(core.CELL_EB_RINT_OHM * 1000, 0), seb=core.CELL_EB_N_SERIE, peb=core.CELL_EB_N_PARALLELE,
        rpb=nombre(core.CELL_PB_RINT_OHM * 1000, 1), spb=core.CELL_PB_N_SERIE, ppb=core.CELL_PB_N_PARALLELE,
    ))


# Export

export = pd.DataFrame({
    "temps_s": temps_min * 60.0,
    "P_demande_W": p_dem,
    "P_EB_W": p_eb,
    "P_PB_W": p_pb,
    "P_convertisseur_W": p_conv,
    "alpha": np.asarray(traj["alpha_final"], dtype=float)[:n],
    "SOC_EB": np.asarray(traj["SOC_EB"], dtype=float)[:n],
    "SOC_PB": np.asarray(traj["SOC_PB"], dtype=float)[:n],
    "pertes_EB_W": pas["eb"][:n],
    "pertes_PB_W": pas["pb"][:n],
    "pertes_convertisseur_W": pas["convertisseur"][:n],
    "P_non_fournie_W": np.asarray(traj["P_unserved"], dtype=float)[:n],
})
st.download_button(
    tr("Télécharger les courbes de {s} (CSV)", "Download the curves of {s} (CSV)", s=nom_affichage(strategie)),
    export.to_csv(index=False, sep=";", decimal="." if langue() == "en" else ",").encode("utf-8-sig"),
    file_name=f"courbes_{strategie}.csv",
    mime="text/csv",
)
st.caption(tr("Données : {s}", "Data: {s}", s=source))


pied_navigation("vues/6_Resultats_et_Analyse.py")
