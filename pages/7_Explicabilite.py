"""
Page « Pourquoi cette décision ? » — analyse d'une décision EMS.

Ce n'est PAS un tableau de variables internes : c'est une justification, en
langage métier, de la répartition de puissance choisie à un instant donné.
Chaque stratégie raconte : quelle était la situation, quelles contraintes ont
été détectées, quel raisonnement a été suivi, et pourquoi la puissance a été
répartie ainsi entre l'EB et la PB.
"""

import sys
from pathlib import Path

DOSSIER_PROJET = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DOSSIER_PROJET))

import numpy as np
import streamlit as st

from ems_core import (
    analyser_capacites_hess,
    compute_symbolic_states,
    alpha_fuzzy_calc,
    FUZZY_RULE_NAMES,
    RULE_LABELS_FR,
    MODEL_DISPLAY_NAMES,
    EPS_POWER_W,
)
from core.resultats import assurer_donnees_session
from core.navigation import pied_navigation


# Configuration de page gérée par le routeur Accueil.py.

st.title("Pourquoi cette décision ?")
st.caption("Analyse et justification d'une décision EMS, à un instant donné, en langage physique.")


# ------------------------------------------------------------
# Données (résultats précalculés via le pont)
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# Choix de l'instant et de la stratégie
# ------------------------------------------------------------

n_points = min(len(traj["P_EB"]) for traj in resultats.values())

col_t, col_s = st.columns([2, 1])
with col_t:
    instant = st.slider("Instant analysé", 0, n_points - 1, n_points // 2)
with col_s:
    strategie = st.selectbox(
        "Stratégie",
        list(resultats.keys()),
        format_func=lambda n: MODEL_DISPLAY_NAMES.get(n, n),
    )

traj = resultats[strategie]

t_sel = float(df["time"].iloc[instant]) if "time" in df.columns else float(instant)
vitesse_ms = float(df["speed"].iloc[instant]) if "speed" in df.columns else None
accel = float(df["hasAcceleration"].iloc[instant]) if "hasAcceleration" in df.columns else 0.0
p_dem = float(df["hasPower"].iloc[instant])
soc_eb = float(traj["SOC_EB"][instant])
soc_pb = float(traj["SOC_PB"][instant])
p_eb = float(traj["P_EB"][instant])
p_pb = float(traj["P_PB"][instant])
alpha_final = float(traj["alpha_final"][instant])
alpha_requested = float(traj["alpha_requested"][instant]) if "alpha_requested" in traj else alpha_final
correction = bool(traj["correction_applied"][instant]) if "correction_applied" in traj else False

cap = analyser_capacites_hess(p_dem, soc_eb, soc_pb)
etats = compute_symbolic_states(p_dem, soc_eb, soc_pb)


def kw(x):
    return f"{x / 1000.0:.1f} kW"


regime = "Traction" if p_dem > EPS_POWER_W else ("Freinage / récupération" if p_dem < -EPS_POWER_W else "Arrêt")

total_mag = abs(p_eb) + abs(p_pb)
part_eb = 100.0 * abs(p_eb) / total_mag if total_mag > 1.0 else 0.0
part_pb = 100.0 * abs(p_pb) / total_mag if total_mag > 1.0 else 0.0


# ============================================================
# 1. Situation du véhicule
# ============================================================

st.header("1. Situation du véhicule")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Temps", f"{t_sel:.0f} s")
c2.metric("Mode", regime)
c3.metric("Puissance demandée", kw(p_dem))
c4.metric("Accélération", f"{accel:.2f} m/s²")


# ============================================================
# 2. Analyse des capacités
# ============================================================

st.header("2. Que peut fournir chaque batterie ?")

a1, a2, a3 = st.columns(3)
a1.metric("Batterie Énergie disponible", kw(cap["eb_dispo_max_W"]))
a2.metric("Batterie Puissance disponible", kw(cap["pb_dispo_max_W"]))
a3.metric("Demande", kw(p_dem))

if p_dem > EPS_POWER_W:
    if p_dem <= cap["eb_dispo_max_W"] + EPS_POWER_W:
        st.success(
            "Conclusion : la batterie Énergie peut couvrir presque toute la demande. "
            "La batterie Puissance n'intervient qu'en complément."
        )
    elif cap["faisable"]:
        st.success(
            "Conclusion : la demande dépasse ce que l'EB peut fournir seule. "
            "La batterie Puissance doit participer."
        )
    else:
        st.error(
            f"Conclusion : la demande n'est pas entièrement réalisable "
            f"(manque de {kw(cap['P_non_servie_W'])})."
        )
elif p_dem < -EPS_POWER_W:
    st.info("Conclusion : phase de freinage — l'énergie est récupérée dans les batteries.")
else:
    st.info("Conclusion : demande quasi nulle, le véhicule est à l'arrêt ou en roue libre.")


# ============================================================
# 3. Raisonnement du modèle
# ============================================================

st.header("3. Raisonnement suivi")

st.markdown("Le système observe la situation suivante :")

observations = []
if etats["high_power_demand"]:
    observations.append("la demande de puissance est forte")
elif etats["zero_power_demand"]:
    observations.append("la demande est quasi nulle")
else:
    observations.append("la demande est modérée")
observations.append(
    "le SOC de la batterie Énergie est satisfaisant"
    if not etats["EB_low_SOC"]
    else "le SOC de la batterie Énergie est bas"
)
observations.append(
    "le SOC de la batterie Puissance est satisfaisant"
    if not etats["PB_low_SOC"]
    else "le SOC de la batterie Puissance est bas"
)
observations.append(
    "le convertisseur est proche de sa limite"
    if etats["converter_risk"]
    else "aucune contrainte de convertisseur n'est détectée"
)

for obs in observations:
    st.markdown(f"- {obs}")

if part_eb >= part_pb:
    st.markdown(
        "Le système **privilégie donc la batterie Énergie**, et ne sollicite la "
        "batterie Puissance qu'en complément."
    )
else:
    st.markdown(
        "Le système **sollicite davantage la batterie Puissance**, car l'énergie "
        "seule ne suffit pas ou doit être préservée."
    )


# ============================================================
# 4. Décision
# ============================================================

st.header("4. Décision prise")

dec1, dec2 = st.columns(2)
with dec1:
    st.markdown(f"**Batterie Énergie : {kw(p_eb)}** ({part_eb:.0f} %)")
    st.progress(min(1.0, part_eb / 100.0))
with dec2:
    st.markdown(f"**Batterie Puissance : {kw(p_pb)}** ({part_pb:.0f} %)")
    st.progress(min(1.0, part_pb / 100.0))


# ============================================================
# 5. Pourquoi pas une autre décision ?
# ============================================================

st.header("5. Pourquoi pas une autre répartition ?")

raisons = []
if p_dem > EPS_POWER_W and p_dem <= cap["eb_dispo_max_W"] + EPS_POWER_W:
    raisons.append(
        "la demande reste inférieure à ce que la batterie Énergie peut fournir seule"
    )
if not etats["EB_low_SOC"]:
    raisons.append("le SOC de la batterie Énergie est suffisant")
if not etats["high_power_demand"]:
    raisons.append("aucune forte demande n'impose de solliciter davantage la batterie Puissance")
if etats["PB_low_SOC"]:
    raisons.append("la batterie Puissance est presque vide et doit être préservée")
if etats["converter_risk"]:
    raisons.append("le convertisseur est proche de sa limite, ce qui limite la répartition")

if raisons:
    st.markdown("La batterie Puissance ne fournit pas davantage parce que :")
    for r in raisons:
        st.markdown(f"- {r}")
else:
    st.markdown(
        "La répartition découle directement des capacités disponibles et de la "
        "demande à cet instant."
    )


# ============================================================
# 6. Raisonnement propre au modèle
# ============================================================

st.header("6. Comment cette stratégie raisonne-t-elle ?")


def _explication_modele():
    if strategie == "EMS_power_limitation":
        return (
            "**Stratégie déterministe (priorité Énergie).** La batterie Énergie "
            "fournit en priorité la puissance qu'elle peut, jusqu'à sa limite ; la "
            "batterie Puissance ne fournit que le complément nécessaire."
        )

    if strategie == "EMS_fuzzy_logic":
        res = alpha_fuzzy_calc(
            np.array([soc_eb]), np.array([soc_pb]), np.array([p_dem]), np.array([accel])
        )
        forces = np.asarray(res["strengths"][0], dtype=float)
        actives = [
            RULE_LABELS_FR.get(FUZZY_RULE_NAMES[i], FUZZY_RULE_NAMES[i])
            for i in range(len(FUZZY_RULE_NAMES))
            if forces[i] > 0.05
        ]
        texte = "**Logique floue.** Les règles expertes activées sont :\n"
        texte += "\n".join(f"- {a}" for a in actives) if actives else "- aucune règle dominante"
        texte += f"\n\nLeur combinaison conduit à confier **{part_pb:.0f} %** de la demande à la PB."
        return texte

    if strategie == "EMS_MLP":
        return (
            f"**Réseau de neurones (MLP).** Le modèle estime directement la fraction "
            f"**{alpha_final * 100:.0f} %** à confier à la batterie Puissance, à partir "
            "de la situation instantanée (puissance demandée, SOC des deux batteries). "
            "Les entrées les plus influentes sont typiquement P_dem, SOC_EB et SOC_PB."
        )

    if strategie == "EMS_MLP_neurosymbolic":
        return (
            f"**MLP neuro-symbolique.** Le réseau propose une répartition intégrant "
            f"déjà des indicateurs symboliques ; la valeur proposée était "
            f"**{alpha_requested * 100:.0f} %** pour la PB, puis le filtre physique de "
            f"sécurité l'a ajustée à **{alpha_final * 100:.0f} %** pour respecter les "
            "contraintes réelles des batteries et du convertisseur."
        )

    if strategie == "EMS_LSTM":
        return (
            "**Modèle temporel (LSTM).** Le modèle tient compte des dernières secondes "
            "du cycle pour anticiper l'évolution de la demande, et sollicite la batterie "
            "Puissance en conséquence."
        )

    if strategie == "EMS_LSTM_neurosymbolic":
        return (
            f"**LSTM neuro-symbolique.** Le modèle temporel anticipe l'évolution de la "
            f"demande ; les indicateurs symboliques encadrent ensuite la répartition. "
            f"Valeur proposée : **{alpha_requested * 100:.0f} %**, valeur finale après "
            f"filtre de sécurité : **{alpha_final * 100:.0f} %**."
        )

    if strategie == "EMS_GNN":
        return (
            "**Réseau de graphes (GNN).** Le modèle considère la structure du HESS "
            "(batterie Énergie, convertisseur, batterie Puissance et charge, reliés entre "
            "eux) et calcule la répartition en tenant compte des interactions entre composants."
        )

    return "Stratégie EMS."


st.markdown(_explication_modele())

if correction:
    st.warning(
        "À cet instant, le **filtre de sécurité** a corrigé la répartition proposée "
        "pour rester dans les limites physiques des batteries et du convertisseur."
    )


pied_navigation("pages/7_Explicabilite.py")
