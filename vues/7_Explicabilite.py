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
    HIGH_POWER_THRESHOLD_W,
    CONVERTER_RISK_THRESHOLD,
    P_CONV_MAX_W,
    P_CONV_MIN_W,
    estimate_p_conv,
)
from core.resultats import assurer_donnees_session, nom_affichage
from core.navigation import pied_navigation
from core import ontology_explainer as ox


# Couleurs de la charte (batterie Énergie = bleu, Puissance = vert).
C_EB = "#5B8DEF"
C_PB = "#30A46C"
C_OK = "#30A46C"
C_NON = "#E5484D"
C_GRIS = "#8B93A7"


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

ETATS_NS = {
    "high_power_demand": "Forte demande de puissance",
    "regenerative_braking": "Freinage / récupération",
    "zero_power_demand": "Demande quasi nulle",
    "converter_risk": "Convertisseur proche de sa limite",
}
ETATS_NS_KEYS = set(ETATS_NS)


def _etats_ns_detail(p_dem, soc_eb, soc_pb, p_eb=None):
    """Détail dynamique des 4 états symboliques : pour chacun, l'état (oui/non),
    la grandeur mesurée et le seuil. Reproduit exactement les conditions de
    compute_symbolic_states pour que l'état affiché corresponde à la valeur."""
    etats = compute_symbolic_states(p_dem, soc_eb, soc_pb, p_eb=p_eb)

    p_kw = p_dem / 1000.0
    seuil_kw = HIGH_POWER_THRESHOLD_W / 1000.0
    eps_kw = EPS_POWER_W / 1000.0

    p_conv = float(estimate_p_conv(np.array([p_eb if p_eb is not None else p_dem]))[0])
    util = p_conv / P_CONV_MAX_W if p_conv >= 0 else abs(p_conv) / abs(P_CONV_MIN_W)

    details = {
        "high_power_demand": (
            f"|P_dem| = {abs(p_kw):.1f} kW "
            f"{'≥' if etats['high_power_demand'] else '<'} seuil {seuil_kw:.0f} kW"
        ),
        "regenerative_braking": (
            f"P_dem = {p_kw:+.1f} kW — "
            f"{'négative, énergie récupérée' if etats['regenerative_braking'] else 'non négative'}"
        ),
        "zero_power_demand": (
            f"|P_dem| = {abs(p_kw):.2f} kW "
            f"{'≤' if etats['zero_power_demand'] else '>'} seuil {eps_kw:.1f} kW"
        ),
        "converter_risk": (
            f"charge convertisseur = {util * 100:.0f} % "
            f"{'≥' if etats['converter_risk'] else '<'} seuil {CONVERTER_RISK_THRESHOLD * 100:.0f} %"
        ),
    }
    return [
        {"libelle": lib, "actif": bool(etats[cle]), "detail": details[cle]}
        for cle, lib in ETATS_NS.items()
    ]


def _pastille(actif):
    couleur = C_OK if actif else C_NON
    txt = "oui" if actif else "non"
    return f"<span style='color:{couleur};font-weight:600'>&#9679; {txt}</span>"


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
    """Reconstruit la fenêtre temporelle (LSTM_WINDOW pas) vue par le LSTM."""
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


def _donut_repartition(part_eb, part_pb):
    fig = go.Figure(
        go.Pie(
            labels=["Batterie Énergie", "Batterie Puissance"],
            values=[part_eb, part_pb],
            hole=0.58,
            marker_colors=[C_EB, C_PB],
            textinfo="label+percent",
            sort=False,
            direction="clockwise",
        )
    )
    fig.update_layout(height=300, margin=dict(t=10, b=10, l=10, r=10), showlegend=False)
    return fig


def _jauge_alpha(alpha):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=alpha * 100.0,
            number={"suffix": " %"},
            title={"text": "alpha — part confiée à la PB"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": C_PB},
                "steps": [
                    {"range": [0, 50], "color": "#EAF1FD"},
                    {"range": [50, 100], "color": "#E7F6EE"},
                ],
            },
        )
    )
    fig.update_layout(height=260, margin=dict(t=50, b=10))
    return fig


def _timeline(etapes):
    with st.container(border=True):
        for i, e in enumerate(etapes):
            st.markdown(f"**{i + 1}.** {e}")
            if i < len(etapes) - 1:
                st.markdown(
                    f"<div style='color:{C_GRIS};margin:-8px 0 -8px 8px;font-size:1.1em'>&#8595;</div>",
                    unsafe_allow_html=True,
                )


def _graphe_gnn(pct_g, edge):
    """Schéma du HESS : cinq nœuds colorés selon leur importance réelle dans la
    décision du GNN. La topologie est schématique ; les couleurs (importances)
    sont calculées à cet instant."""
    positions = [(0.0, 1.0), (0.0, -1.0), (1.2, 0.0), (2.4, 0.0), (3.6, 0.0)]

    ei = edge.detach().cpu().numpy()
    edge_x, edge_y = [], []
    for k in range(ei.shape[1]):
        a, b = int(ei[0, k]), int(ei[1, k])
        if a < len(positions) and b < len(positions):
            edge_x += [positions[a][0], positions[b][0], None]
            edge_y += [positions[a][1], positions[b][1], None]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(color="#C7CCD6", width=2), hoverinfo="skip")
    )
    labels = [LABELS_NOEUDS.get(n, n) for n in GNN_NODE_NAMES]
    fig.add_trace(
        go.Scatter(
            x=[p[0] for p in positions],
            y=[p[1] for p in positions],
            mode="markers+text",
            marker=dict(
                size=[34 + p * 0.9 for p in pct_g],
                color=list(pct_g),
                colorscale="YlGnBu",
                showscale=True,
                colorbar=dict(title="%"),
                line=dict(color="white", width=2),
            ),
            text=[f"{l}<br>{p:.0f} %" for l, p in zip(labels, pct_g)],
            textposition="bottom center",
            hoverinfo="text",
        )
    )
    fig.update_layout(
        height=380,
        showlegend=False,
        margin=dict(t=20, b=20, l=10, r=10),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, range=[-1.8, 1.6]),
    )
    return fig


def _etapes_raisonnement(mode, p_dem, soc_eb, soc_pb, part_eb, part_pb, correction):
    etapes = [f"Situation : {mode.lower()}, puissance demandée {abs(p_dem) / 1000.0:.1f} kW"]
    if abs(p_dem) <= EPS_POWER_W:
        etapes.append("Demande quasi nulle : les batteries sont peu sollicitées")
    elif p_dem < 0:
        etapes.append("Phase de freinage : de l'énergie est disponible à la récupération")
        etapes.append(f"Orientation de la récupération : PB {part_pb:.0f} %, EB {part_eb:.0f} %")
    else:
        etapes.append(f"État des batteries : SOC EB {soc_eb * 100:.0f} %, SOC PB {soc_pb * 100:.0f} %")
        if part_eb >= part_pb:
            etapes.append("La batterie Énergie peut porter l'essentiel ; la Puissance couvre les pics")
        else:
            etapes.append("La demande dépasse le confort de l'Énergie : la Puissance prend le relais")
    etapes.append(f"Décision : Énergie {part_eb:.0f} %, Puissance {part_pb:.0f} %")
    if correction:
        etapes.append("Ajustement final par le filtre de sécurité physique")
    return etapes


def _resume_texte(strategie, p_dem, part_eb, part_pb, correction):
    nom = nom_affichage(strategie)
    if abs(p_dem) <= EPS_POWER_W:
        coeur = "aucune batterie n'est réellement sollicitée."
    elif p_dem < 0:
        coeur = (
            f"l'énergie de freinage est récupérée (Puissance {part_pb:.0f} %, "
            f"Énergie {part_eb:.0f} %)."
        )
    elif part_eb >= part_pb:
        coeur = (
            f"la batterie Énergie fournit l'essentiel ({part_eb:.0f} %) et la "
            f"batterie Puissance complète ({part_pb:.0f} %)."
        )
    else:
        coeur = (
            f"la batterie Puissance prend le relais ({part_pb:.0f} %) pour "
            f"soulager la batterie Énergie ({part_eb:.0f} %)."
        )
    fin = (
        " La répartition a été validée sans correction."
        if not correction
        else " La répartition a été ajustée par le filtre de sécurité."
    )
    return f"Avec la stratégie {nom}, {coeur}{fin}"


# Interface

st.title("Centre d'explicabilité")
st.caption(
    "Comprendre en quelques secondes pourquoi l'algorithme a réparti la puissance "
    "de cette façon. Le contenu s'adapte au modèle : on ne montre que ce qu'il "
    "utilise réellement."
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
    strategie = st.selectbox("Stratégie", list(resultats.keys()), format_func=nom_affichage)

traj = resultats[strategie]

t_sel = float(df["time"].iloc[instant]) if "time" in df.columns else float(instant)
speed = float(df["speed"].iloc[instant]) if "speed" in df.columns else 0.0
accel = float(df["hasAcceleration"].iloc[instant]) if "hasAcceleration" in df.columns else 0.0
p_dem = float(df["hasPower"].iloc[instant])
soc_eb = float(traj["SOC_EB"][instant])
soc_pb = float(traj["SOC_PB"][instant])
p_eb = float(traj["P_EB"][instant])
p_pb = float(traj["P_PB"][instant])
p_eb_instant = float(traj["I_EB"][instant]) * V_EB_PACK_NOM
alpha_final = float(traj["alpha_final"][instant]) if "alpha_final" in traj else 0.0
alpha_requested = float(traj["alpha_requested"][instant]) if "alpha_requested" in traj else alpha_final
correction = bool(traj["correction_applied"][instant]) if "correction_applied" in traj else False

if p_dem > EPS_POWER_W:
    mode = "Traction"
elif p_dem < -EPS_POWER_W:
    mode = "Freinage / récupération"
else:
    mode = "Arrêt / roue libre"

total_mag = abs(p_eb) + abs(p_pb)
part_eb = 100.0 * abs(p_eb) / total_mag if total_mag > 1.0 else 0.0
part_pb = 100.0 * abs(p_pb) / total_mag if total_mag > 1.0 else 0.0


def kw(x):
    return f"{x / 1000.0:.1f} kW"


tab_decision, tab_pourquoi, tab_raison, tab_physique, tab_science = st.tabs(
    ["1. Décision", "2. Pourquoi ?", "3. Raisonnement", "4. Vérification", "5. Détails scientifiques"]
)
tab_apercu = tab_decision


# Vue d'ensemble : situation, décision visuelle, résumé en langage naturel

with tab_apercu:
    st.subheader("Situation actuelle")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Mode", mode)
    c2.metric("Puissance demandée", kw(p_dem))
    c3.metric("Vitesse", f"{speed * 3.6:.0f} km/h")
    c4.metric("SOC Énergie", f"{soc_eb * 100:.0f} %")
    c5.metric("SOC Puissance", f"{soc_pb * 100:.0f} %")

    st.subheader("Décision prise")
    g1, g2 = st.columns(2)
    with g1:
        if total_mag > 1.0:
            st.plotly_chart(_donut_repartition(part_eb, part_pb), use_container_width=True)
        else:
            st.info("Demande quasi nulle : les batteries sont au repos.")
    with g2:
        st.plotly_chart(_jauge_alpha(alpha_final), use_container_width=True)

    st.subheader("Résumé")
    st.success(_resume_texte(strategie, p_dem, part_eb, part_pb, correction))
    if correction:
        st.warning(
            "Le filtre de sécurité a corrigé la répartition proposée pour respecter "
            "les limites physiques des batteries et du convertisseur."
        )


# Pourquoi ? — causes, confiance, contrefactuels (source : ontologie OntoHESS)

with tab_pourquoi:
    _concepts = ox.concepts_actifs(p_dem, soc_eb, soc_pb, p_eb=p_eb_instant)
    _actifs = [c for c in _concepts if c["actif"]]

    st.subheader("Les causes principales")
    if _actifs:
        for c in _actifs:
            with st.container(border=True):
                st.markdown(f"**{c['libelle']}**  ·  concept `{c['concept']}`")
                st.caption(f"Constat : {c['mesure']} — conséquence : {c['consequence']}.")
    else:
        st.info("Aucun concept particulier n'est reconnu : la situation est nominale.")

    st.subheader("Niveau de confiance")
    _conf, _pour, _contre = ox.indice_confiance(
        p_dem, soc_eb, soc_pb, correction, abs(alpha_final - alpha_requested)
    )
    st.progress(_conf / 100.0, text=f"Confiance dans la décision : {_conf:.0f} %")
    cc1, cc2 = st.columns(2)
    with cc1:
        for r in _pour:
            st.markdown(f"- {r}")
    with cc2:
        for r in _contre:
            st.markdown(f"- {r}")
    st.caption(
        "Cet indice n'est pas une probabilité produite par un modèle : il mesure la "
        "marge par rapport aux situations limites (seuils de SOC, limites de puissance, "
        "correction du filtre)."
    )

    st.subheader("Que se serait-il passé si…")
    for phrase in ox.contrefactuels(p_dem, soc_eb, soc_pb):
        st.markdown(f"- {phrase}")


# Raisonnement du modèle : chaîne d'inférence + ce que le modèle utilise réellement

with tab_raison:
    st.subheader("Chaîne d'inférence")
    st.caption(
        "Des mesures jusqu'à la validation physique, en passant par les concepts et "
        "les règles de l'ontologie OntoHESS."
    )
    _chaine = ox.chaine_inference(
        p_dem, soc_eb, soc_pb, part_eb, part_pb, correction, p_eb=p_eb_instant
    )
    _etapes_chaine = [
        "**Mesures** — " + " · ".join(_chaine["mesures"]),
        "**Concepts reconnus** — "
        + (", ".join(c["libelle"] for c in _chaine["concepts"]) or "situation nominale"),
        "**Règles activées** — "
        + (
            ", ".join(f"{r['id']}" for r in _chaine["regles"])
            or "aucune règle numériquement évaluable"
        ),
        "**Décision** — " + _chaine["decision"],
        "**Validation** — " + _chaine["validation"],
    ]
    _timeline(_etapes_chaine)

    if _chaine["regles"]:
        st.markdown("**Justification logique des règles activées**")
        for r in _chaine["regles"]:
            with st.container(border=True):
                st.markdown(f"**{r['id']}** — conclut sur : {', '.join(r['conclusions'])}")
                for d in r["details"]:
                    st.markdown(f"- prémisse vérifiée : `{d['texte']}`")
    st.caption(
        "Règles lues directement dans ontologies/OntoHESS2.owl et évaluées avec les "
        "valeurs de cet instant. Le solveur applique en parallèle une reproduction à "
        "seuils fixes de ces mêmes règles."
    )

    st.subheader("Fil du raisonnement")
    _timeline(_etapes_raisonnement(mode, p_dem, soc_eb, soc_pb, part_eb, part_pb, correction))

    st.subheader("Puissances autour de l'instant")
    demi = 150
    i0, i1 = max(0, instant - demi), min(n, instant + demi + 1)
    xx = df["time"].to_numpy()[i0:i1]
    fig_ctx = go.Figure()
    fig_ctx.add_trace(go.Scatter(x=xx, y=df["hasPower"].to_numpy()[i0:i1] / 1000.0, name="P_dem", line=dict(color=C_GRIS)))
    fig_ctx.add_trace(go.Scatter(x=xx, y=np.asarray(traj["P_EB"], float)[i0:i1] / 1000.0, name="P_EB", line=dict(color=C_EB)))
    fig_ctx.add_trace(go.Scatter(x=xx, y=np.asarray(traj["P_PB"], float)[i0:i1] / 1000.0, name="P_PB", line=dict(color=C_PB)))
    fig_ctx.add_vline(x=t_sel, line=dict(color=C_NON, dash="dash"))
    fig_ctx.update_layout(
        xaxis_title="Temps (s)", yaxis_title="Puissance (kW)", height=320,
        margin=dict(t=20, b=40), hovermode="x unified",
    )
    st.plotly_chart(fig_ctx, use_container_width=True)

    st.subheader("Ce que la stratégie utilise réellement")

    if strategie == "EMS_power_limitation":
        st.markdown("**Stratégie physique déterministe** — priorité à la batterie Énergie, dans ses limites.")
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
        st.markdown("**Logique floue** — règles expertes et leur force à cet instant.")
        res = alpha_fuzzy_calc(np.array([soc_eb]), np.array([soc_pb]), np.array([p_dem]), np.array([accel]))
        forces = np.asarray(res["strengths"][0], dtype=float)
        fig_r = go.Figure(go.Bar(x=list(FUZZY_RULE_NAMES), y=forces, marker_color=C_EB))
        fig_r.update_layout(title="Force de chaque règle (0 à 1)", yaxis_title="Force", height=320, margin=dict(t=40, b=80))
        st.plotly_chart(fig_r, use_container_width=True)
        actives = [(FUZZY_RULE_NAMES[i], forces[i]) for i in range(len(FUZZY_RULE_NAMES)) if forces[i] > 0.05]
        for nom_r, f in sorted(actives, key=lambda x: x[1], reverse=True):
            st.markdown(f"- **{RULE_LABELS_FR.get(nom_r, nom_r)}** — force {f * 100:.0f} %")

    elif strategie == "EMS_MLP":
        st.markdown(
            "**Réseau de neurones (MLP)** — pas de règles internes lisibles. "
            "L'explication fidèle est l'influence de chaque entrée sur la décision "
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
        couleurs = [C_NON if v < 0 else C_EB for v in contrib]
        fig_f = go.Figure(go.Bar(x=[LABELS_FEATURES[c] for c in MLP_INPUT_COLS], y=contrib, marker_color=couleurs))
        fig_f.update_layout(title="Influence de chaque entrée sur alpha", yaxis_title="Contribution", height=320, margin=dict(t=40, b=40))
        st.plotly_chart(fig_f, use_container_width=True)
        st.caption("Bleu = pousse vers plus de PB ; rouge = pousse vers plus d'EB.")

    elif strategie == "EMS_MLP_neurosymbolic":
        st.markdown(
            "**MLP neuro-symbolique** — la décision se décompose : base floue + "
            "correction neuronale bornée + filtre physique."
        )
        alpha_fuzzy = float(alpha_fuzzy_calc(np.array([soc_eb]), np.array([soc_pb]), np.array([p_dem]), np.array([accel]))["alpha"][0])
        delta = alpha_requested - alpha_fuzzy
        d1, d2, d3 = st.columns(3)
        d1.metric("Base floue", f"{alpha_fuzzy * 100:.0f} %")
        d2.metric("Correction réseau", f"{delta * 100:+.0f} %")
        d3.metric("Après filtre", f"{alpha_final * 100:.0f} %")
        st.caption(
            f"La logique floue proposait {alpha_fuzzy * 100:.0f} % pour la PB ; le réseau a "
            f"corrigé de {delta * 100:+.0f} % ; le filtre a abouti à {alpha_final * 100:.0f} %."
        )

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
        fig_l = go.Figure(go.Bar(x=[labels[i] for i in ordre], y=[pct[i] for i in ordre], marker_color=C_EB))
        fig_l.update_layout(title="Importance des entrées (%)", yaxis_title="%", height=340, margin=dict(t=40, b=110))
        st.plotly_chart(fig_l, use_container_width=True)

    elif strategie == "EMS_GNN":
        st.markdown(
            "**Réseau de graphes (GNN)** — il raisonne sur la structure du HESS. "
            "Chaque nœud est coloré selon son influence réelle sur la décision "
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
        st.plotly_chart(_graphe_gnn(pct_g, edge), use_container_width=True)
        st.caption("Topologie schématique ; les couleurs représentent l'importance calculée à cet instant.")

    else:
        st.info("Stratégie non reconnue pour l'explication détaillée.")


# Validation physique : contraintes, courants, correction

with tab_physique:
    st.subheader("Répartition et courants")
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

    st.subheader("Contraintes vérifiées")
    soc_ok = soc_eb > SOC_EB_MIN
    dem_ok = p_dem <= P_EB_MAX_W if p_dem > 0 else True
    st.markdown(
        f"- {_pastille(soc_ok)} SOC Énergie au-dessus du minimum "
        f"({soc_eb * 100:.0f} % vs {SOC_EB_MIN * 100:.0f} %)",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"- {_pastille(dem_ok)} Demande dans la limite de l'EB "
        f"({kw(p_dem)} vs {kw(P_EB_MAX_W)})",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"- {_pastille(not correction)} Décision "
        f"{'acceptée sans correction' if not correction else 'corrigée par le filtre de sécurité'}",
        unsafe_allow_html=True,
    )

    st.subheader("Décision, exprimée comme une règle")
    p_dem_kw = p_dem / 1000.0
    seuil_eb_kw = P_EB_MAX_W / 1000.0
    if abs(p_dem) <= EPS_POWER_W:
        st.markdown(
            f"- **Si** la demande est quasi nulle (P_dem = {p_dem_kw:.2f} kW), "
            "**alors** aucune batterie n'est réellement sollicitée."
        )
    elif p_dem < -EPS_POWER_W:
        st.markdown(
            f"- **Si** le véhicule freine (P_dem = {p_dem_kw:+.1f} kW), **alors** "
            f"l'énergie récupérée recharge les batteries : {part_pb:.0f} % vers la PB, "
            f"{part_eb:.0f} % vers l'EB."
        )
    elif part_pb <= 5.0:
        st.markdown(
            f"- **Si** la demande ({p_dem_kw:.1f} kW) reste dans ce que l'EB peut fournir "
            f"seule (≤ {seuil_eb_kw:.0f} kW), l'EB **fournit toute** la puissance "
            "**parce que** c'est la batterie d'énergie et que la PB est préservée pour les pics."
        )
    elif part_eb >= part_pb:
        st.markdown(
            f"- **Si** la demande ({p_dem_kw:.1f} kW) approche la limite de l'EB, **alors** "
            f"l'EB fournit l'essentiel ({part_eb:.0f} %) et la PB complète ({part_pb:.0f} %)."
        )
    else:
        st.markdown(
            f"- **Sinon** (demande de {p_dem_kw:.1f} kW au-delà du confort de l'EB), la "
            f"**PB prend le relais** : {part_pb:.0f} % contre {part_eb:.0f} % pour l'EB."
        )
    if soc_eb <= SOC_EB_MIN + 1e-3:
        st.markdown(
            "- **Si** l'EB atteint son SOC minimal, **alors** elle est protégée et la PB assure la demande."
        )


# Justification scientifique : états symboliques détaillés, décomposition NS

with tab_science:
    est_ns = strategie in ("EMS_MLP_neurosymbolic", "EMS_LSTM_neurosymbolic")
    if est_ns:
        st.subheader("États symboliques (entrées supplémentaires du modèle NS)")
        st.caption("Vert = état actif, rouge = inactif. Chaque état est recalculé à cet instant.")
        lignes = _etats_ns_detail(p_dem, soc_eb, soc_pb, p_eb=p_eb_instant)
        for rangee in (lignes[:2], lignes[2:]):
            cols = st.columns(2)
            for col, item in zip(cols, rangee):
                with col:
                    with st.container(border=True):
                        st.markdown(
                            f"{_pastille(item['actif'])} **{item['libelle']}**",
                            unsafe_allow_html=True,
                        )
                        st.caption(item["detail"])
        st.markdown(
            f"Répartition **proposée par le modèle** : {alpha_requested * 100:.0f} % pour la PB ; "
            f"**après filtre de sécurité** : {alpha_final * 100:.0f} %."
        )
    else:
        st.info(
            "Cette stratégie n'utilise pas d'états symboliques. Voir l'onglet "
            "« Raisonnement du modèle » pour son explication fidèle (règles, gradients "
            "ou importance des composants)."
        )

    st.subheader("Règles de l'ontologie non activées")
    st.caption(
        "Expliquer pourquoi une règle ne s'applique pas est aussi informatif que "
        "justifier celles qui s'appliquent."
    )
    _, _non_act, _indet = ox.evaluer_regles(p_dem, soc_eb, soc_pb)
    for r in _non_act[:6]:
        echecs = [d["texte"] for d in r["details"] if d["ok"] is False]
        if echecs:
            st.markdown(f"- **{r['id']}** ignorée : condition non remplie — `{' ; '.join(echecs)}`")
    if _indet:
        st.caption(
            f"{len(_indet)} règle(s) ne sont pas évaluables ici : elles portent sur des "
            "grandeurs non disponibles à cet instant (tensions et courants internes du "
            "convertisseur)."
        )

    st.subheader("Grandeurs brutes à cet instant")
    st.markdown(
        f"- Temps : **{t_sel:.0f} s**\n"
        f"- P_dem : **{kw(p_dem)}** — P_EB : **{kw(p_eb)}** — P_PB : **{kw(p_pb)}**\n"
        f"- SOC EB : **{soc_eb * 100:.1f} %** — SOC PB : **{soc_pb * 100:.1f} %**\n"
        f"- alpha demandé : **{alpha_requested:.3f}** — alpha appliqué : **{alpha_final:.3f}**"
    )


pied_navigation("vues/7_Explicabilite.py")
