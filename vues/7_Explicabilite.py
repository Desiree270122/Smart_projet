import sys
from pathlib import Path

DOSSIER_PROJET = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DOSSIER_PROJET))

import numpy as np
import torch
import plotly.graph_objects as go
import streamlit as st

from ems_core import (
    alpha_fuzzy_calc,
    compute_symbolic_states,
    load_mlp_simple,
    charger_scaler,
    appliquer_scaler,
    FUZZY_RULE_NAMES,
    RULE_LABELS_FR,
    MLP_INPUT_COLS,
    MLP_SCALER_FILE,
    DEVICE,
    EPS_POWER_W,
    V_EB_PACK_NOM,
    V_PB_PACK_NOM,
    P_EB_MAX_W,
    SOC_EB_MIN,
)
from core.resultats import assurer_donnees_session, nom_affichage
from core.navigation import pied_navigation


# Étiquettes lisibles des 5 entrées du MLP.
LABELS_FEATURES = {
    "SOC_EB": "SOC EB",
    "SOC_PB": "SOC PB",
    "hasPower": "P_dem",
    "speed": "Vitesse",
    "hasAcceleration": "Accélération",
}

# Les 4 états symboliques réellement utilisés par les modèles neuro-symboliques.
ETATS_NS = {
    "high_power_demand": "Forte demande de puissance",
    "regenerative_braking": "Freinage / récupération",
    "zero_power_demand": "Demande quasi nulle",
    "converter_risk": "Convertisseur proche de sa limite",
}


@st.cache_resource(show_spinner=False)
def _charger_mlp():
    modele = load_mlp_simple()
    modele.eval()
    return modele, charger_scaler(MLP_SCALER_FILE)


st.title("Pourquoi cette décision ?")
st.caption(
    "Justification de la répartition de puissance à un instant donné. Le contenu "
    "s'adapte au modèle : on ne montre que ce que la stratégie utilise réellement."
)

try:
    assurer_donnees_session(st)
except FileNotFoundError as exc:
    st.error(str(exc))
    st.info("Lance une fois le précalcul :  `python scripts/run_simulations.py`")
    st.stop()

resultats = st.session_state.get("resultats_simulation")
df = st.session_state.get("cycle_pret")
if not resultats or df is None:
    st.warning("Aucune donnée disponible.")
    st.stop()

n = min(len(traj["P_EB"]) for traj in resultats.values())

col_t, col_s = st.columns([2, 1])
with col_t:
    instant = st.slider("Instant analysé", 0, n - 1, n // 2)
with col_s:
    strategie = st.selectbox(
        "Stratégie",
        list(resultats.keys()),
        format_func=nom_affichage,
    )

traj = resultats[strategie]

t_sel = float(df["time"].iloc[instant]) if "time" in df.columns else float(instant)
speed = float(df["speed"].iloc[instant]) if "speed" in df.columns else 0.0
accel = float(df["hasAcceleration"].iloc[instant]) if "hasAcceleration" in df.columns else 0.0
p_dem = float(df["hasPower"].iloc[instant])
soc_eb = float(traj["SOC_EB"][instant])
soc_pb = float(traj["SOC_PB"][instant])
p_eb = float(traj["P_EB"][instant])
p_pb = float(traj["P_PB"][instant])
alpha_final = float(traj["alpha_final"][instant]) if "alpha_final" in traj else 0.0
alpha_requested = float(traj["alpha_requested"][instant]) if "alpha_requested" in traj else alpha_final
correction = bool(traj["correction_applied"][instant]) if "correction_applied" in traj else False

if p_dem > EPS_POWER_W:
    mode = "Traction"
elif p_dem < -EPS_POWER_W:
    mode = "Freinage / récupération"
else:
    mode = "Arrêt / roue libre"


def kw(x):
    return f"{x / 1000.0:.1f} kW"


# 1. Contexte

st.header("1. Contexte")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Temps", f"{t_sel:.0f} s")
c2.metric("Mode", mode)
c3.metric("Puissance demandée", kw(p_dem))
c4.metric("Vitesse", f"{speed * 3.6:.0f} km/h")

demi = 150
i0, i1 = max(0, instant - demi), min(n, instant + demi + 1)
xx = df["time"].to_numpy()[i0:i1]
fig_ctx = go.Figure()
fig_ctx.add_trace(go.Scatter(x=xx, y=df["hasPower"].to_numpy()[i0:i1] / 1000.0, name="P_dem", line=dict(color="#8B93A7")))
fig_ctx.add_trace(go.Scatter(x=xx, y=np.asarray(traj["P_EB"], float)[i0:i1] / 1000.0, name="P_EB", line=dict(color="#5B8DEF")))
fig_ctx.add_trace(go.Scatter(x=xx, y=np.asarray(traj["P_PB"], float)[i0:i1] / 1000.0, name="P_PB", line=dict(color="#30A46C")))
fig_ctx.add_vline(x=t_sel, line=dict(color="#E5484D", dash="dash"))
fig_ctx.update_layout(
    title="Puissances autour de l'instant analysé",
    xaxis_title="Temps (s)",
    yaxis_title="Puissance (kW)",
    height=340,
    margin=dict(t=40, b=40),
    hovermode="x unified",
)
st.plotly_chart(fig_ctx, use_container_width=True)


# 2. Vérification des conditions — dépend du modèle (fidélité)

st.header("2. Ce que la stratégie utilise réellement")

if strategie == "EMS_power_limitation":
    st.markdown("**Stratégie physique déterministe** — priorité à la batterie Énergie, dans ses limites.")
    cond1 = soc_eb > SOC_EB_MIN
    cond2 = 0 < p_dem <= P_EB_MAX_W
    st.markdown(
        f"- SOC EB ({soc_eb * 100:.0f} %) au-dessus du minimum ({SOC_EB_MIN * 100:.0f} %) : "
        f"**{'oui' if cond1 else 'non'}**\n"
        f"- Demande ({kw(p_dem)}) dans la limite de l'EB ({kw(P_EB_MAX_W)}) : "
        f"**{'oui' if cond2 else 'non'}**"
    )
    st.caption(
        "Règle : l'EB fournit la puissance qu'elle peut (jusqu'à sa limite), la PB complète le reste."
    )

elif strategie == "EMS_fuzzy_logic":
    st.markdown("**Logique floue** — les règles expertes et leur force (poids des règles).")
    res = alpha_fuzzy_calc(np.array([soc_eb]), np.array([soc_pb]), np.array([p_dem]), np.array([accel]))
    forces = np.asarray(res["strengths"][0], dtype=float)
    fig_r = go.Figure(go.Bar(x=list(FUZZY_RULE_NAMES), y=forces, marker_color="#5B8DEF"))
    fig_r.update_layout(title="Force de chaque règle (0 à 1)", yaxis_title="Force", height=320, margin=dict(t=40, b=80))
    st.plotly_chart(fig_r, use_container_width=True)
    actives = [(FUZZY_RULE_NAMES[i], forces[i]) for i in range(len(FUZZY_RULE_NAMES)) if forces[i] > 0.05]
    for nom_r, f in sorted(actives, key=lambda x: x[1], reverse=True):
        st.markdown(f"- **{RULE_LABELS_FR.get(nom_r, nom_r)}** — force {f * 100:.0f} %")

elif strategie == "EMS_MLP":
    st.markdown(
        "**Réseau de neurones (MLP)** — pas de règles internes lisibles. La seule "
        "explication fidèle est l'**influence de chaque entrée** sur la décision "
        "(gradient × entrée à cet instant)."
    )
    modele, scaler = _charger_mlp()
    vals = {"SOC_EB": soc_eb, "SOC_PB": soc_pb, "hasPower": p_dem, "speed": speed, "hasAcceleration": accel}
    brut = np.array([vals[col] for col in MLP_INPUT_COLS], dtype=np.float64)
    brut_s = appliquer_scaler(brut, scaler) if scaler is not None else brut
    x = torch.tensor([np.asarray(brut_s).tolist()], dtype=torch.float32, device=DEVICE, requires_grad=True)
    a = modele(x)
    a.sum().backward()
    contrib = (x.grad[0].cpu().numpy() * np.asarray(brut_s, dtype=float))
    couleurs = ["#E5484D" if v < 0 else "#5B8DEF" for v in contrib]
    fig_f = go.Figure(go.Bar(x=[LABELS_FEATURES[c] for c in MLP_INPUT_COLS], y=contrib, marker_color=couleurs))
    fig_f.update_layout(title="Influence de chaque entrée sur alpha", yaxis_title="Contribution", height=320, margin=dict(t=40, b=40))
    st.plotly_chart(fig_f, use_container_width=True)
    st.caption("Bleu = pousse vers plus de PB ; rouge = pousse vers plus d'EB.")

elif strategie == "EMS_MLP_neurosymbolic":
    st.markdown(
        "**MLP neuro-symbolique** — la décision se décompose : base floue + correction "
        "neuronale bornée + filtre physique."
    )
    alpha_fuzzy = float(alpha_fuzzy_calc(np.array([soc_eb]), np.array([soc_pb]), np.array([p_dem]), np.array([accel]))["alpha"][0])
    delta = alpha_requested - alpha_fuzzy
    d1, d2, d3 = st.columns(3)
    d1.metric("Base floue", f"{alpha_fuzzy * 100:.0f} %")
    d2.metric("Correction réseau", f"{delta * 100:+.0f} %")
    d3.metric("Après filtre", f"{alpha_final * 100:.0f} %")
    st.caption(
        f"La logique floue proposait {alpha_fuzzy * 100:.0f} % pour la PB ; le réseau a "
        f"corrigé de {delta * 100:+.0f} % ; le filtre de sécurité a abouti à "
        f"{alpha_final * 100:.0f} %."
    )
    etats = compute_symbolic_states(p_dem, soc_eb, soc_pb)
    st.markdown("**États symboliques vérifiés :**")
    for cle, lib in ETATS_NS.items():
        st.markdown(f"- {'[oui]' if etats[cle] else '[non]'} {lib}")

else:
    st.markdown(
        f"**{nom_affichage(strategie)}** — modèle appris (temporel ou relationnel). "
        "Son raisonnement interne n'est pas directement lisible sur cette page ; on "
        "affiche donc la décision et sa vérification physique."
    )
    if "neurosymbolic" in strategie:
        etats = compute_symbolic_states(p_dem, soc_eb, soc_pb)
        st.markdown("**États symboliques encadrant la décision :**")
        for cle, lib in ETATS_NS.items():
            st.markdown(f"- {'[oui]' if etats[cle] else '[non]'} {lib}")


# 3. Décision prise (avec calcul des courants)

st.header("3. Décision prise")

total_mag = abs(p_eb) + abs(p_pb)
part_eb = 100.0 * abs(p_eb) / total_mag if total_mag > 1.0 else 0.0
part_pb = 100.0 * abs(p_pb) / total_mag if total_mag > 1.0 else 0.0

g1, g2, g3 = st.columns(3)
g1.metric("alpha appliqué", f"{alpha_final:.2f}")
g2.metric("Batterie Énergie", kw(p_eb), f"{part_eb:.0f} %")
g3.metric("Batterie Puissance", kw(p_pb), f"{part_pb:.0f} %")

i_eb = p_eb / V_EB_PACK_NOM
i_pb = p_pb / V_PB_PACK_NOM
st.markdown("**Calcul des courants** (I = P / V) :")
st.markdown(
    f"- I_EB = {p_eb:.0f} W / {V_EB_PACK_NOM:.0f} V = **{i_eb:.1f} A**\n"
    f"- I_PB = {p_pb:.0f} W / {V_PB_PACK_NOM:.1f} V = **{i_pb:.1f} A**"
)


# 4. Pourquoi (et pourquoi pas autre chose)

st.header("4. Pourquoi cette répartition ?")

raisons = []
if p_dem > EPS_POWER_W and part_eb >= part_pb:
    raisons.append("la demande reste dans ce que l'EB peut fournir, on la privilégie")
if p_dem > EPS_POWER_W and part_pb > part_eb:
    raisons.append("la demande dépasse le confort de l'EB, la PB prend le complément")
if soc_eb <= SOC_EB_MIN + 1e-3:
    raisons.append("l'EB est à son minimum : elle est protégée")
if p_dem < -EPS_POWER_W:
    raisons.append("phase de freinage : l'énergie est dirigée vers les batteries")

if raisons:
    for r in raisons:
        st.markdown(f"- {r}")

if strategie in ("EMS_MLP_neurosymbolic", "EMS_LSTM_neurosymbolic"):
    st.markdown(
        f"Répartition **proposée par le modèle** : {alpha_requested * 100:.0f} % pour la PB ; "
        f"**après filtre de sécurité** : {alpha_final * 100:.0f} %."
    )

if correction:
    st.warning(
        "Le filtre de sécurité a **corrigé** la répartition proposée pour respecter les "
        "limites physiques des batteries et du convertisseur."
    )


pied_navigation("vues/7_Explicabilite.py")
