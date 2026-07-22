import sys
from pathlib import Path

DOSSIER_PROJET = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DOSSIER_PROJET))

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

import ems_core as core
from ems_core import alpha_fuzzy_calc, EPS_POWER_W
from core.resultats import assurer_donnees_session, nom_affichage


# Configuration de page gérée par le routeur Accueil.py.

C_EB = "#3B82F6"
C_PB = "#22C55E"
C_OK = "#22C55E"
C_ATTENTION = "#F59E0B"
C_GRIS = "#94A3B8"
MEDAILLES = ["🥇", "🥈", "🥉", "4e", "5e", "6e", "7e"]


def _scenario(p_dem, speed_kmh, accel):
    if abs(p_dem) <= EPS_POWER_W:
        return "Arrêt / roue libre"
    if p_dem < 0:
        return "Freinage / récupération"
    if accel > 0.8:
        return "Forte accélération"
    if speed_kmh > 90:
        return "Vitesse élevée"
    return "Traction régulière"


def _pensee(part_eb, part_pb, correction, p_dem):
    if abs(p_dem) <= EPS_POWER_W:
        texte = "Demande quasi nulle : je sollicite peu les batteries."
    elif p_dem < 0:
        cible = "Puissance" if part_pb >= part_eb else "Énergie"
        texte = f"Freinage : je dirige la récupération surtout vers la batterie {cible}."
    elif part_eb >= part_pb:
        texte = f"Je privilégie la batterie Énergie ({part_eb:.0f} %) car la demande reste gérable."
    else:
        texte = f"Je sollicite davantage la batterie Puissance ({part_pb:.0f} %) pour absorber le pic."
    if correction:
        texte += " Ma proposition a été ajustée par le filtre de sécurité."
    return texte


def _tag_decision(part_eb, part_pb, correction):
    if correction:
        return "décision corrigée par le filtre"
    if abs(part_eb - part_pb) <= 10:
        return "bon équilibre EB/PB"
    if part_eb > part_pb:
        return "préserve la batterie Puissance"
    return "soulage la batterie Énergie"


def _classement(resultats, instant, p_dem):
    """Classe les stratégies à cet instant par coût physique multi-objectif
    (candidate_metrics) : plus bas = mieux. Score 0-100 normalisé entre elles."""
    couts = {}
    for nom, tr in resultats.items():
        a = float(tr["alpha_final"][instant])
        se = float(tr["SOC_EB"][instant])
        sp = float(tr["SOC_PB"][instant])
        try:
            m = core.candidate_metrics(np.array([a]), p_dem, se, sp, None)
            couts[nom] = float(m["total_cost"][0])
        except Exception:  # noqa: BLE001
            couts[nom] = float("nan")

    finis = [c for c in couts.values() if c == c]
    lo, hi = (min(finis), max(finis)) if finis else (0.0, 1.0)

    lignes = []
    for nom, c in couts.items():
        if c != c:
            score = 0.0
        elif hi - lo < 1e-12:
            score = 100.0
        else:
            score = 100.0 * (hi - c) / (hi - lo)
        lignes.append({"nom": nom, "cout": c, "score": score})
    lignes.sort(key=lambda x: (-x["score"], x["nom"]))
    return lignes


st.title("📊 Analyse instantanée")
st.caption(
    "Comprenez comment chaque stratégie réagit à un instant précis du cycle de "
    "conduite. Ici on compare les approches ; pour explorer en profondeur une "
    "seule stratégie, voir « Pourquoi cette décision ? »."
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

n = min(len(tr["P_EB"]) for tr in resultats.values())
instant = st.slider("Instant du cycle à analyser", 0, n - 1, n // 2)

ligne = df.iloc[instant]
t_sel = float(ligne["time"]) if "time" in df.columns else float(instant)
speed = float(ligne["speed"]) if "speed" in df.columns else 0.0
speed_kmh = speed * 3.6
accel = float(ligne["hasAcceleration"]) if "hasAcceleration" in df.columns else 0.0
p_dem = float(ligne["hasPower"])
scenario = _scenario(p_dem, speed_kmh, accel)


# Situation actuelle

st.subheader("Situation actuelle")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Vitesse", f"{speed_kmh:.0f} km/h")
c2.metric("Puissance demandée", f"{p_dem / 1000:.1f} kW")
c3.metric("Accélération", f"{accel:+.1f} m/s²")
c4.metric("Situation", scenario)
st.caption(f"Instant analysé : t = {t_sel:.0f} s.")


# Décisions des modèles (cartes, plus de dataframe brut)

st.subheader("📊 Décisions des modèles")

noms = list(resultats.keys())
par_ligne = 4
for i0 in range(0, len(noms), par_ligne):
    bloc = noms[i0 : i0 + par_ligne]
    cols = st.columns(par_ligne)
    for col, nom in zip(cols, bloc):
        tr = resultats[nom]
        p_eb = float(tr["P_EB"][instant])
        p_pb = float(tr["P_PB"][instant])
        alpha = float(tr["alpha_final"][instant])
        corr = bool(tr["correction_applied"][instant]) if "correction_applied" in tr else False
        tot = abs(p_eb) + abs(p_pb)
        part_eb = 100.0 * abs(p_eb) / tot if tot > 1.0 else 0.0
        part_pb = 100.0 * abs(p_pb) / tot if tot > 1.0 else 0.0
        with col:
            with st.container(border=True):
                st.markdown(f"**{nom_affichage(nom)}**")
                st.markdown(
                    f"<span style='color:{C_EB};font-weight:700'>EB {part_eb:.0f} %</span> · "
                    f"<span style='color:{C_PB};font-weight:700'>PB {part_pb:.0f} %</span>",
                    unsafe_allow_html=True,
                )
                st.caption(f"alpha = {alpha:.2f}")
                coul = C_ATTENTION if corr else C_OK
                mot = "corrigée" if corr else "acceptée"
                st.markdown(
                    f"<span style='color:{coul}'>&#9679; {mot}</span>",
                    unsafe_allow_html=True,
                )


# Évolution autour de cet instant (une seule figure, stratégie au choix)

st.subheader("📈 Évolution autour de cet instant")

strategie_focus = st.selectbox(
    "Stratégie détaillée sur les courbes",
    noms,
    format_func=nom_affichage,
)
tr_f = resultats[strategie_focus]

demi = 150
i0, i1 = max(0, instant - demi), min(n, instant + demi + 1)
xx = df["time"].to_numpy()[i0:i1]

fig = make_subplots(
    rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
    subplot_titles=("Puissances (kW)", "États de charge (%)"),
)
fig.add_trace(go.Scatter(x=xx, y=df["hasPower"].to_numpy()[i0:i1] / 1000.0, name="P_dem", line=dict(color=C_GRIS)), row=1, col=1)
fig.add_trace(go.Scatter(x=xx, y=np.asarray(tr_f["P_EB"], float)[i0:i1] / 1000.0, name="P_EB", line=dict(color=C_EB)), row=1, col=1)
fig.add_trace(go.Scatter(x=xx, y=np.asarray(tr_f["P_PB"], float)[i0:i1] / 1000.0, name="P_PB", line=dict(color=C_PB)), row=1, col=1)
fig.add_trace(go.Scatter(x=xx, y=np.asarray(tr_f["SOC_EB"], float)[i0:i1] * 100.0, name="SOC_EB", line=dict(color=C_EB, dash="dot")), row=2, col=1)
fig.add_trace(go.Scatter(x=xx, y=np.asarray(tr_f["SOC_PB"], float)[i0:i1] * 100.0, name="SOC_PB", line=dict(color=C_PB, dash="dot")), row=2, col=1)
fig.add_vline(x=t_sel, line=dict(color="#EF4444", dash="dash"))
fig.update_layout(height=460, margin=dict(t=50, b=30), hovermode="x unified", legend=dict(orientation="h", y=1.12))
st.plotly_chart(fig, use_container_width=True)


# Construction de la décision (variante neuro-symbolique)

if "EMS_MLP_neurosymbolic" in resultats:
    st.subheader("🧠 Construction de la décision (MLP neuro-symbolique)")
    tr_ns = resultats["EMS_MLP_neurosymbolic"]
    soc_eb_ns = float(tr_ns["SOC_EB"][instant])
    soc_pb_ns = float(tr_ns["SOC_PB"][instant])
    alpha_final_ns = float(tr_ns["alpha_final"][instant])
    alpha_req_ns = float(tr_ns["alpha_requested"][instant]) if "alpha_requested" in tr_ns else alpha_final_ns
    alpha_fuzzy = float(
        alpha_fuzzy_calc(np.array([soc_eb_ns]), np.array([soc_pb_ns]), np.array([p_dem]), np.array([accel]))["alpha"][0]
    )
    delta = alpha_req_ns - alpha_fuzzy
    with st.container(border=True):
        st.markdown(f"**Règles floues** — {alpha_fuzzy * 100:.0f} % pour la PB")
        st.markdown(f"<div style='color:{C_GRIS}'>&#8595;</div>", unsafe_allow_html=True)
        st.markdown(f"**Correction IA** — {delta * 100:+.0f} %")
        st.markdown(f"<div style='color:{C_GRIS}'>&#8595;</div>", unsafe_allow_html=True)
        st.markdown("**Filtre physique de sécurité**")
        st.markdown(f"<div style='color:{C_GRIS}'>&#8595;</div>", unsafe_allow_html=True)
        st.markdown(f"**Décision finale** — {alpha_final_ns * 100:.0f} % pour la PB")


# Raisonnement de chaque modèle (langage naturel)

st.subheader("💬 Raisonnement de chaque modèle")
for nom in noms:
    tr = resultats[nom]
    p_eb = float(tr["P_EB"][instant])
    p_pb = float(tr["P_PB"][instant])
    corr = bool(tr["correction_applied"][instant]) if "correction_applied" in tr else False
    tot = abs(p_eb) + abs(p_pb)
    part_eb = 100.0 * abs(p_eb) / tot if tot > 1.0 else 0.0
    part_pb = 100.0 * abs(p_pb) / tot if tot > 1.0 else 0.0
    st.markdown(f"- **{nom_affichage(nom)}** : « {_pensee(part_eb, part_pb, corr, p_dem)} »")


# Classement à cet instant

st.subheader("🏆 Quelle stratégie est la plus performante ?")

if abs(p_dem) <= EPS_POWER_W:
    st.info("Demande quasi nulle à cet instant : le classement n'est pas significatif.")
    classement = []
else:
    classement = _classement(resultats, instant, p_dem)
    cols = st.columns(len(classement))
    for col, item, medaille in zip(cols, classement, MEDAILLES):
        tr = resultats[item["nom"]]
        p_eb = float(tr["P_EB"][instant])
        p_pb = float(tr["P_PB"][instant])
        corr = bool(tr["correction_applied"][instant]) if "correction_applied" in tr else False
        tot = abs(p_eb) + abs(p_pb)
        part_eb = 100.0 * abs(p_eb) / tot if tot > 1.0 else 0.0
        part_pb = 100.0 * abs(p_pb) / tot if tot > 1.0 else 0.0
        with col:
            with st.container(border=True):
                st.markdown(
                    f"<div style='text-align:center;font-size:22px'>{medaille}</div>"
                    f"<div style='text-align:center;font-weight:700'>{nom_affichage(item['nom'])}</div>"
                    f"<div style='text-align:center;color:{C_GRIS};font-size:.85em'>score {item['score']:.0f}/100</div>"
                    f"<div style='text-align:center;color:{C_GRIS};font-size:.8em'>{_tag_decision(part_eb, part_pb, corr)}</div>",
                    unsafe_allow_html=True,
                )


# Verdict + à retenir

st.subheader("📋 Verdict")
if classement:
    meilleur = classement[0]
    tr_b = resultats[meilleur["nom"]]
    corr_b = bool(tr_b["correction_applied"][instant]) if "correction_applied" in tr_b else False
    raisons = ["coût physique le plus faible à cet instant"]
    if not corr_b:
        raisons.append("décision acceptée sans correction du filtre")
    with st.container(border=True):
        st.markdown(
            f"À cet instant, **{len(classement)} stratégies** ont été comparées. "
            f"La stratégie **{nom_affichage(meilleur['nom'])}** obtient le meilleur score "
            f"({meilleur['score']:.0f}/100)."
        )
        st.markdown("Pourquoi :")
        for r in raisons:
            st.markdown(f"- {r}")

    st.subheader("🎓 Ce qu'il faut retenir")
    alphas = np.array([float(resultats[nm]["alpha_final"][instant]) for nm in noms])
    ecart = float(alphas.max() - alphas.min())
    ns_noms = [nm for nm in noms if "neurosymbolic" in nm]
    corr_ns = sum(
        1 for nm in ns_noms if "correction_applied" in resultats[nm] and bool(resultats[nm]["correction_applied"][instant])
    )
    corr_autres = sum(
        1
        for nm in noms
        if nm not in ns_noms and "correction_applied" in resultats[nm] and bool(resultats[nm]["correction_applied"][instant])
    )
    if ecart < 0.10:
        phrase = "À cet instant, les modèles prennent des décisions très proches."
    else:
        phrase = f"À cet instant, les modèles divergent nettement (écart d'alpha de {ecart:.2f})."
    if ns_noms and corr_ns < corr_autres:
        phrase += " Les variantes neuro-symboliques respectent mieux les contraintes physiques (moins de corrections)."
    st.info(phrase)


from core.navigation import pied_navigation

pied_navigation("vues/4_Moteur_Neurosymbolique.py")
