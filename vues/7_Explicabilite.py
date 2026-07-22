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
    construire_graphe_instant,
    load_mlp_simple,
    load_lstm_seul,
    load_lstm_neurosymbolic,
    load_gnn_simple,
    charger_scaler,
    appliquer_scaler,
    FUZZY_RULE_NAMES,
    RULE_LABELS_FR,
    MLP_INPUT_COLS,
    MLP_SCALER_FILE,
    LSTM_FEATURE_COLS,
    LSTM_NS_FEATURE_COLS,
    LSTM_WINDOW,
    LSTM_SCALER_FILE,
    LSTM_NS_SCALER_FILE,
    GNN_SCALER_FILE,
    GNN_NODE_NAMES,
    DEVICE,
    EPS_POWER_W,
    V_EB_PACK_NOM,
    V_PB_PACK_NOM,
    P_EB_MAX_W,
    P_EB_MIN_W,
    SOC_EB_MIN,
)
from core.resultats import assurer_donnees_session, nom_affichage
from core.navigation import pied_navigation


# Étiquettes lisibles des entrées / composants.
LABELS_FEATURES = {
    "SOC_EB": "SOC EB",
    "SOC_PB": "SOC PB",
    "hasPower": "P_dem",
    "speed": "Vitesse",
    "hasAcceleration": "Accélération",
    "hasTotalForce": "Force totale",
    "I_EB": "I_EB",
    "high_power_demand": "Forte demande",
    "regenerative_braking": "Freinage",
    "zero_power_demand": "Demande nulle",
    "converter_risk": "Risque convert.",
}

LABELS_NOEUDS = {
    "energy_battery": "Batterie Énergie",
    "power_battery": "Batterie Puissance",
    "converter": "Convertisseur",
    "motor": "Moteur",
    "vehicle": "Véhicule",
}

# Les 4 états symboliques réellement utilisés par les modèles neuro-symboliques.
ETATS_NS = {
    "high_power_demand": "Forte demande de puissance",
    "regenerative_braking": "Freinage / récupération",
    "zero_power_demand": "Demande quasi nulle",
    "converter_risk": "Convertisseur proche de sa limite",
}
ETATS_NS_KEYS = set(ETATS_NS)


@st.cache_resource(show_spinner=False)
def _charger_mlp():
    modele = load_mlp_simple()
    modele.eval()
    return modele, charger_scaler(MLP_SCALER_FILE)


@st.cache_resource(show_spinner=False)
def _charger_lstm(ns):
    modele = load_lstm_neurosymbolic() if ns else load_lstm_seul()
    modele.eval()
    scaler = charger_scaler(LSTM_NS_SCALER_FILE if ns else LSTM_SCALER_FILE)
    cols = LSTM_NS_FEATURE_COLS if ns else LSTM_FEATURE_COLS
    return modele, scaler, list(cols)


@st.cache_resource(show_spinner=False)
def _charger_gnn_xai():
    res = load_gnn_simple()
    modele = res[0] if isinstance(res, tuple) else res
    modele.eval()
    return modele, charger_scaler(GNN_SCALER_FILE)


def _fenetre_lstm(cols, instant, df, traj):
    """Reconstruit la fenêtre temporelle (LSTM_WINDOW pas) vue par le LSTM à
    l'instant t : features physiques depuis le cycle et la trajectoire, états
    symboliques recalculés par pas pour les variantes neuro-symboliques."""
    idx = [max(0, instant - LSTM_WINDOW + 1 + k) for k in range(LSTM_WINDOW)]

    def valeurs(col):
        if col in ("SOC_EB", "SOC_PB", "I_EB"):
            return np.asarray(traj[col], dtype=float)[idx]
        if col in ETATS_NS_KEYS:
            vals = []
            for j in idx:
                pj = float(df["hasPower"].iloc[j])
                sej = float(traj["SOC_EB"][j])
                spj = float(traj["SOC_PB"][j])
                iej = float(traj["I_EB"][j])
                etats_j = compute_symbolic_states(pj, sej, spj, p_eb=iej * V_EB_PACK_NOM)
                vals.append(float(etats_j[col]))
            return np.asarray(vals, dtype=float)
        return df[col].to_numpy(dtype=float)[idx]

    return np.stack([valeurs(c) for c in cols], axis=1).astype(np.float32)


def _attribution_lstm(strategie, instant, df, traj):
    """Importance de chaque entrée (|gradient × entrée| sommé sur la fenêtre)."""
    ns = "neurosymbolic" in strategie
    modele, scaler, cols = _charger_lstm(ns)
    fen = _fenetre_lstm(cols, instant, df, traj)
    if scaler is not None:
        for k in range(fen.shape[0]):
            try:
                fen[k, :] = appliquer_scaler(fen[k, :], scaler)
            except Exception:  # noqa: BLE001
                pass
    x = torch.tensor(fen[None, ...], dtype=torch.float32, device=DEVICE, requires_grad=True)
    with torch.backends.cudnn.flags(enabled=False):
        sortie = modele(x)
        sortie.sum().backward()
    imp = np.abs(x.grad[0].cpu().numpy() * fen).sum(axis=0)
    return cols, imp


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
    st.markdown(
        f"- SOC EB : **{soc_eb * 100:.0f} %** (minimum {SOC_EB_MIN * 100:.0f} %)\n"
        f"- Demande : **{kw(p_dem)}** (limite EB en décharge {kw(P_EB_MAX_W)}, "
        f"en recharge {kw(P_EB_MIN_W)})"
    )
    # Règle réellement appliquée à cet instant (branche active de la logique EB-priority).
    if abs(p_dem) <= EPS_POWER_W:
        regle = "demande quasi nulle : aucune batterie n'est sollicitée."
    elif p_dem < 0:
        if p_dem < P_EB_MIN_W:
            regle = (
                f"freinage fort : l'EB absorbe jusqu'à sa limite ({kw(P_EB_MIN_W)}), "
                f"la PB absorbe le surplus ({kw(p_dem - P_EB_MIN_W)})."
            )
        else:
            regle = "freinage modéré : l'EB absorbe toute l'énergie récupérée."
    elif soc_eb <= SOC_EB_MIN:
        regle = (
            f"l'EB est à son SOC minimum ({soc_eb * 100:.0f} %) : elle est protégée, "
            "la PB fournit toute la demande."
        )
    elif p_dem <= P_EB_MAX_W:
        regle = "la demande tient dans la limite de l'EB : l'EB fournit seule, la PB reste au repos."
    else:
        regle = (
            f"la demande dépasse la limite de l'EB : l'EB donne son maximum ({kw(P_EB_MAX_W)}), "
            f"la PB complète ({kw(p_dem - P_EB_MAX_W)})."
        )
    st.success(f"**Règle appliquée ici** : {regle}")

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

elif strategie in ("EMS_LSTM", "EMS_LSTM_neurosymbolic"):
    ns = "neurosymbolic" in strategie
    st.markdown(
        f"**Modèle temporel (LSTM{'-NS' if ns else ''})** — il tient compte des "
        f"{LSTM_WINDOW} dernières secondes. Influence réelle de chaque entrée sur sa "
        "prédiction (gradient × entrée sur la fenêtre) :"
    )
    cols, imp = _attribution_lstm(strategie, instant, df, traj)
    total = imp.sum()
    pct = imp / total * 100.0 if total > 0 else imp
    labels = [LABELS_FEATURES.get(c, c) for c in cols]
    ordre = list(np.argsort(pct)[::-1])
    fig_l = go.Figure(
        go.Bar(x=[labels[i] for i in ordre], y=[pct[i] for i in ordre], marker_color="#5B8DEF")
    )
    fig_l.update_layout(title="Importance des entrées (%)", yaxis_title="%", height=340, margin=dict(t=40, b=110))
    st.plotly_chart(fig_l, use_container_width=True)
    if ns:
        etats = compute_symbolic_states(p_dem, soc_eb, soc_pb)
        st.markdown("**États symboliques (entrées supplémentaires du modèle NS) :**")
        for cle, lib in ETATS_NS.items():
            st.markdown(f"- {'[oui]' if etats[cle] else '[non]'} {lib}")

elif strategie == "EMS_GNN":
    st.markdown(
        "**Réseau de graphes (GNN)** — il raisonne sur la structure du HESS. "
        "Influence réelle de chaque composant (nœud du graphe) sur la décision "
        "(gradient × entrée) :"
    )
    modele_g, scaler_g = _charger_gnn_xai()
    x_g, edge = construire_graphe_instant(p_dem, soc_eb, soc_pb, accel, scaler_g)
    x_g = x_g.to(DEVICE).clone().requires_grad_(True)
    edge = edge.to(DEVICE)
    batch = torch.zeros(x_g.shape[0], dtype=torch.long, device=DEVICE)
    sortie_g = modele_g(x_g, edge, batch)
    sortie_g.sum().backward()
    imp_g = np.abs((x_g.grad * x_g).detach().cpu().numpy()).sum(axis=1)
    total_g = imp_g.sum()
    pct_g = imp_g / total_g * 100.0 if total_g > 0 else imp_g
    noms = [LABELS_NOEUDS.get(n, n) for n in GNN_NODE_NAMES]
    fig_g = go.Figure(go.Bar(x=noms, y=pct_g, marker_color="#30A46C"))
    fig_g.update_layout(title="Importance de chaque composant (%)", yaxis_title="%", height=340, margin=dict(t=40, b=60))
    st.plotly_chart(fig_g, use_container_width=True)

else:
    st.info("Stratégie non reconnue pour l'explication détaillée.")


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
