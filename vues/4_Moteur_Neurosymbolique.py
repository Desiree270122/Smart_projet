

import sys
from pathlib import Path

DOSSIER_PROJET = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DOSSIER_PROJET))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from ems_core import (
    alpha_fuzzy_calc,
    compute_symbolic_states,
    V_EB_PACK_NOM,
)
from core.resultats import assurer_donnees_session, nom_affichage
from core import ontology_explainer as ox
from core.navigation import pied_navigation



C_EB = "#3B82F6"
C_PB = "#22C55E"
C_GRIS = "#94A3B8"



LIB_ETATS = {
    "high_power_demand": "Forte demande de puissance",
    "regenerative_braking": "Freinage régénératif",
    "zero_power_demand": "Demande quasi nulle",
    "converter_risk": "Convertisseur proche de sa limite",
    "EB_available": "Batterie Énergie disponible",
    "PB_available": "Batterie Puissance disponible",
    "EB_low_SOC": "SOC batterie Énergie faible",
    "PB_low_SOC": "SOC batterie Puissance faible",
}


def _alpha_fuzzy(soc_eb, soc_pb, p_dem, accel):
    return float(
        alpha_fuzzy_calc(np.array([soc_eb]), np.array([soc_pb]), np.array([p_dem]), np.array([accel]))["alpha"][0]
    )


def _fleche():
    st.markdown(f"<div style='color:{C_GRIS};font-size:1.1em'>&#8595;</div>", unsafe_allow_html=True)


st.title("📊 Analyse instantanée")
st.caption(
    "Que décide chaque stratégie à un instant précis, et comment cette décision "
    "se construit-elle ? Pour comparer les stratégies sur l'ensemble du cycle, "
    "voir « Comparer les méthodes »."
)

try:
    _source = assurer_donnees_session(st)
except FileNotFoundError as exc:
    st.error(str(exc))
    st.info("Lance une fois le précalcul :  `python scripts/run_simulations.py`")
    st.stop()

resultats = st.session_state.get("resultats_simulation")
df = st.session_state.get("cycle_pret")
if not resultats or df is None:
    st.warning("Aucune donnée disponible.")
    st.stop()

st.caption(f"Source des données : {_source}")

noms = list(resultats.keys())
n = min([len(df)] + [len(tr["P_EB"]) for tr in resultats.values()])


# Un seul sélecteur de stratégie + le curseur d'instant, en tête : ils pilotent
# tout le reste de la page.

col_i, col_s = st.columns([2, 1])
with col_i:
    if "time" in df.columns and n > 1:
        _t = df["time"].to_numpy()[:n]
        t_min, t_max = int(_t[0]), int(_t[-1])
        t_choisi = st.slider("Instant du cycle (s)", t_min, t_max, int((t_min + t_max) // 2))
        instant = int(np.abs(_t - t_choisi).argmin())
    else:
        instant = st.slider("Échantillon", 0, n - 1, n // 2)
with col_s:
    strategie = st.selectbox("Stratégie analysée", noms, format_func=nom_affichage)

ligne = df.iloc[instant]
t_sel = float(ligne["time"]) if "time" in df.columns else float(instant)
speed_kmh = (float(ligne["speed"]) * 3.6) if "speed" in df.columns else 0.0
accel = float(ligne["hasAcceleration"]) if "hasAcceleration" in df.columns else 0.0
p_dem = float(ligne["hasPower"])

tr_sel = resultats[strategie]
soc_eb_sel = float(tr_sel["SOC_EB"][instant])
soc_pb_sel = float(tr_sel["SOC_PB"][instant])
i_eb_sel = float(tr_sel["I_EB"][instant])
alpha_final = float(tr_sel["alpha_final"][instant])
alpha_req = float(tr_sel["alpha_requested"][instant]) if "alpha_requested" in tr_sel else alpha_final
corr_sel = bool(tr_sel["correction_applied"][instant]) if "correction_applied" in tr_sel else False

# Source UNIQUE des états symboliques : celle que les modèles consomment.
etats = compute_symbolic_states(p_dem, soc_eb_sel, soc_pb_sel, p_eb=i_eb_sel * V_EB_PACK_NOM)
etats_actifs = [LIB_ETATS[k] for k in LIB_ETATS if etats.get(k)]

# État de fonctionnement inféré par l'ontologie (remplace l'ancien _scenario).
interp = ox.interpretation_ontologique(p_dem, soc_eb_sel, soc_pb_sel)
etat_op = ox.ETATS_ONTOLOGIE[interp["etat"]]


# Situation du véhicule (texte en caption, pas en st.metric qui tronque)

st.subheader("Situation du véhicule")
c1, c2, c3 = st.columns(3)
c1.metric("Vitesse", f"{speed_kmh:.0f} km/h")
c2.metric("Puissance demandée", f"{p_dem / 1000:.1f} kW")
c3.metric("Accélération", f"{accel:+.1f} m/s²")
st.caption(
    f"Instant : t = {t_sel:.0f} s.  ·  État de fonctionnement inféré par l'ontologie : "
    f"**{etat_op}** (`{interp['etat']}`), sur les états de charge de {nom_affichage(strategie)}."
)


# Décision de chaque stratégie : table compacte (dense et rigoureuse) + barre alpha

st.subheader("Décision de chaque stratégie")

lignes = []
for nom in noms:
    tr = resultats[nom]
    a = float(tr["alpha_final"][instant])
    a_req = float(tr["alpha_requested"][instant]) if "alpha_requested" in tr else a
    se = float(tr["SOC_EB"][instant])
    sp = float(tr["SOC_PB"][instant])
    a_fz = _alpha_fuzzy(se, sp, p_dem, accel)
    corr = bool(tr["correction_applied"][instant]) if "correction_applied" in tr else False
    lignes.append(
        {
            "Stratégie": nom_affichage(nom),
            "alpha (part PB)": round(a, 2),
            "P_EB (kW)": round(float(tr["P_EB"][instant]) / 1000, 1),
            "P_PB (kW)": round(float(tr["P_PB"][instant]) / 1000, 1),
            "Écart / base floue": round(a - a_fz, 2),
            "Écart / demandé": round(a - a_req, 2),
            "Filtre": "corrigée" if corr else "—",
        }
    )
st.dataframe(pd.DataFrame(lignes).set_index("Stratégie"), use_container_width=True)
st.caption(
    "alpha est la fraction de puissance confiée à la PB. « Écart / base floue » = "
    "distance à ce que proposerait la logique floue ; « Écart / demandé » = correction "
    "apportée par le filtre de sécurité."
)

ordre = sorted(noms, key=lambda x: float(resultats[x]["alpha_final"][instant]), reverse=True)
fig_alpha = go.Figure(
    go.Bar(
        x=[float(resultats[x]["alpha_final"][instant]) for x in ordre][::-1],
        y=[nom_affichage(x) for x in ordre][::-1],
        orientation="h",
        marker_color=[(C_PB if x == strategie else C_GRIS) for x in ordre][::-1],
        text=[f"{float(resultats[x]['alpha_final'][instant]):.2f}" for x in ordre][::-1],
        textposition="outside",
        hoverinfo="skip",
    )
)
fig_alpha.update_layout(
    height=40 * len(noms) + 70,
    margin=dict(t=10, b=30, l=10, r=50),
    xaxis=dict(title="alpha — part confiée à la PB", range=[0, 1]),
    showlegend=False,
)
st.plotly_chart(fig_alpha, use_container_width=True)
st.caption(f"En vert : la stratégie sélectionnée ({nom_affichage(strategie)}).")


# Évolution locale, pour la stratégie sélectionnée uniquement

st.subheader("📈 Évolution autour de cet instant")

demi = 150
i0, i1 = max(0, instant - demi), min(n, instant + demi + 1)
xx = df["time"].to_numpy()[i0:i1] if "time" in df.columns else np.arange(i0, i1)

fig = make_subplots(
    rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
    subplot_titles=("Puissances (kW)", "États de charge (%)"),
)
fig.add_trace(go.Scatter(x=xx, y=df["hasPower"].to_numpy()[i0:i1] / 1000.0, name="P_dem", line=dict(color=C_GRIS)), row=1, col=1)
fig.add_trace(go.Scatter(x=xx, y=np.asarray(tr_sel["P_EB"], float)[i0:i1] / 1000.0, name="P_EB", line=dict(color=C_EB)), row=1, col=1)
fig.add_trace(go.Scatter(x=xx, y=np.asarray(tr_sel["P_PB"], float)[i0:i1] / 1000.0, name="P_PB", line=dict(color=C_PB)), row=1, col=1)
fig.add_trace(go.Scatter(x=xx, y=np.asarray(tr_sel["SOC_EB"], float)[i0:i1] * 100.0, name="SOC_EB", line=dict(color=C_EB, dash="dot")), row=2, col=1)
fig.add_trace(go.Scatter(x=xx, y=np.asarray(tr_sel["SOC_PB"], float)[i0:i1] * 100.0, name="SOC_PB", line=dict(color=C_PB, dash="dot")), row=2, col=1)
fig.add_vline(x=t_sel, line=dict(color="#EF4444", dash="dash"))
fig.update_layout(height=460, margin=dict(t=50, b=30), hovermode="x unified", legend=dict(orientation="h", y=1.12))
st.plotly_chart(fig, use_container_width=True)


# Pipeline de décision — adapté à la stratégie sélectionnée

st.subheader(f"🧩 Comment {nom_affichage(strategie)} construit sa décision")

is_ns = "neurosymbolic" in strategie
is_fuzzy = strategie == "EMS_fuzzy_logic"
is_phys = strategie == "EMS_power_limitation"

alpha_fuzzy_sel = _alpha_fuzzy(soc_eb_sel, soc_pb_sel, p_dem, accel)
etats_txt = ", ".join(etats_actifs) if etats_actifs else "aucun état particulier"
filtre_txt = "correction appliquée" if corr_sel else "aucune correction"
decision_txt = f"{alpha_final * 100:.0f} % pour la PB"

if is_ns:
    etapes = [
        ("Ontologie OntoHESS", "concepts et seuils décrivant le HESS"),
        ("États symboliques détectés", etats_txt),
        ("Règles expertes floues", f"{alpha_fuzzy_sel * 100:.0f} % pour la PB"),
        ("Correction du réseau neuronal", f"{(alpha_req - alpha_fuzzy_sel) * 100:+.0f} %"),
        ("Filtre physique de sécurité", filtre_txt),
        ("Décision finale", decision_txt),
    ]
    note = (
        "La décision part des connaissances de l'ontologie, puis des règles floues ; "
        "le réseau n'apporte qu'une correction bornée."
    )
elif is_fuzzy:
    etapes = [
        ("Ontologie OntoHESS", "concepts et seuils décrivant le HESS"),
        ("États symboliques détectés", etats_txt),
        ("Règles expertes floues", f"{alpha_fuzzy_sel * 100:.0f} % pour la PB"),
        ("Filtre physique de sécurité", filtre_txt),
        ("Décision finale", decision_txt),
    ]
    note = "Décision entièrement issue des règles expertes formalisées par l'ontologie."
elif is_phys:
    etapes = [
        ("Ontologie OntoHESS", "règles SWRL de priorité à la batterie Énergie"),
        ("États symboliques détectés", etats_txt),
        ("Règle physique déterministe", "l'EB fournit en priorité, dans ses limites"),
        ("Filtre physique de sécurité", filtre_txt),
        ("Décision finale", decision_txt),
    ]
    note = "Décision déterministe dont les branches correspondent aux règles SWRL de l'ontologie."
else:
    etapes = [
        ("Réseau neuronal", "prédit directement la répartition"),
        ("Filtre physique de sécurité", filtre_txt),
        ("Décision finale", decision_txt),
    ]
    note = (
        f"À titre de repère, la logique floue proposerait {alpha_fuzzy_sel * 100:.0f} % "
        "pour la PB — mais cette stratégie ne s'appuie pas dessus."
    )

with st.container(border=True):
    for i, (titre, detail) in enumerate(etapes):
        st.markdown(f"**{titre}**" + (f" — {detail}" if detail else ""))
        if i < len(etapes) - 1:
            _fleche()

st.caption(note)


# Évaluer sur l'ensemble du cycle (à la place du classement instantané)

st.divider()
st.subheader("Et sur l'ensemble du cycle ?")
st.markdown(
    "Un EMS ne se juge pas sur un instant isolé : envoyer de la puissance vers la PB "
    "maintenant ne se justifie qu'au regard de ce qu'il en reste plus tard dans le "
    "cycle. Le classement des stratégies se fait donc sur le cycle complet."
)
if st.button("Aller à « Comparer les méthodes »", type="primary"):
    st.switch_page("vues/5_Comparaison_des_strategies.py")


pied_navigation("vues/4_Moteur_Neurosymbolique.py")
