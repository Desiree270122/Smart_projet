"""
Page « Lancer une simulation » — assistant scientifique.

L'interface ne se contente pas d'afficher des options : elle aide à construire
une configuration cohérente et à en comprendre les conséquences avant
l'exécution (analyse de configuration, contrôle de cohérence, validation).
La logique de simulation elle-même est inchangée.
"""

import os
import re
import sys
import time
from pathlib import Path

DOSSIER_PROJET = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DOSSIER_PROJET))

import pandas as pd
import streamlit as st
import torch

from ems_core import (
    simuler_toutes_strategies,
    set_alpha_grid_step,
    load_mlp_simple,
    load_mlp_neurosymbolic,
    load_lstm_seul,
    load_lstm_neurosymbolic,
    load_gnn_simple,
)
from core.resultats import EXPLICABILITE, NIVEAUX_EXPLICABILITE
from core.navigation import pied_navigation

# Configuration de page gérée par le routeur Accueil.py.

try:
    torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
except Exception:
    pass


MODELES_NEURONAUX = [
    "EMS_MLP",
    "EMS_MLP_neurosymbolic",
    "EMS_LSTM",
    "EMS_LSTM_neurosymbolic",
    "EMS_GNN",
]

# Famille et coût relatif en temps de calcul (poids servant à l'estimation).
FAMILLES = {
    "EMS_MLP": ("Réseau dense (MLP)", "Rapide", 1.0),
    "EMS_MLP_neurosymbolic": ("Neuro-symbolique", "Modéré", 1.5),
    "EMS_LSTM": ("Réseau récurrent (LSTM)", "Modéré", 1.5),
    "EMS_LSTM_neurosymbolic": ("Neuro-symbolique temporel", "Modéré", 1.5),
    "EMS_GNN": ("Réseau de graphes (GNN)", "Lent", 3.0),
}

CLES_RESULTATS_A_SUPPRIMER = [
    "resultats_simulation",
    "avertissements_simulation",
    "signature_simulation",
    "modele_actif",
    "alpha_star_reference",
    "erreurs_chargement",
    "_sim_custom_faite",
]


def supprimer_anciens_resultats():
    for cle in CLES_RESULTATS_A_SUPPRIMER:
        st.session_state.pop(cle, None)


@st.cache_resource(show_spinner=False)
def _charger_modeles_deterministes():
    """Charge les 4 modèles neuronaux « standard » (MLP, MLP-NS, LSTM, LSTM-NS)."""
    modeles, erreurs = {}, {}
    chargeurs = {
        "EMS_MLP": load_mlp_simple,
        "EMS_MLP_neurosymbolic": load_mlp_neurosymbolic,
        "EMS_LSTM": load_lstm_seul,
        "EMS_LSTM_neurosymbolic": load_lstm_neurosymbolic,
    }
    for nom, chargeur in chargeurs.items():
        try:
            m = chargeur()
            if hasattr(m, "eval"):
                m.eval()
            modeles[nom] = m
        except Exception as exc:  # noqa: BLE001
            erreurs[nom] = str(exc)
    return modeles, erreurs


@st.cache_resource(show_spinner=False)
def _charger_gnn():
    """Charge le GNN à part (import torch_geometric coûteux, à la demande)."""
    gnn_model, _gnn_scaler = load_gnn_simple()
    if hasattr(gnn_model, "eval"):
        gnn_model.eval()
    return gnn_model


def _charger_modeles(avec_gnn):
    modeles_det, erreurs_det = _charger_modeles_deterministes()
    modeles = dict(modeles_det)
    erreurs = dict(erreurs_det)
    if avec_gnn:
        try:
            modeles["EMS_GNN"] = _charger_gnn()
        except Exception as exc:  # noqa: BLE001
            erreurs["EMS_GNN"] = str(exc)
    return modeles, erreurs


@st.cache_data(show_spinner=False)
def _simuler_en_cache(df, soc_eb0, soc_pb0, signature_modeles, pas_alpha, _modeles_charges):
    set_alpha_grid_step(pas_alpha)
    with torch.inference_mode():
        return simuler_toutes_strategies(df, soc_eb0, soc_pb0, _modeles_charges)


def _niveau_explicabilite(code):
    niveau, _ = EXPLICABILITE.get(code, (0, ""))
    return {3: "Par construction", 2: "Partielle", 1: "Post-hoc"}.get(niveau, "—")


st.title("▶️ Lancer une simulation")
st.caption(
    "Construisez une configuration cohérente, vérifiez ses conséquences, puis "
    "exécutez la simulation sur votre propre cycle de conduite."
)

st.info(
    "Cette page relance un calcul complet — comptez plusieurs minutes. Pour une "
    "démonstration immédiate, les pages Comparer, Explorer et « Pourquoi cette "
    "décision ? » utilisent déjà des résultats précalculés."
)

if "cycle_pret" not in st.session_state:
    st.warning("Aucun cycle préparé. Commencez par la page « Préparer une simulation ».")
    if st.button("Aller à la préparation"):
        st.switch_page("vues/2_Preparation_donnees.py")
    st.stop()

df = st.session_state["cycle_pret"].copy()
nb_points = len(df)
duree_cycle_s = float(df["time"].iloc[-1]) if "time" in df.columns and nb_points else 0.0


# ① Cycle de conduite

st.subheader("① Cycle de conduite")

col_cyc, col_mod = st.columns([3, 1])
with col_cyc:
    with st.container(border=True):
        st.markdown("**Cycle actuellement chargé**")
        i1, i2 = st.columns(2)
        i1.metric("Instants", f"{nb_points:,}".replace(",", " "))
        i2.metric("Durée du cycle", f"{duree_cycle_s / 60:.1f} min" if duree_cycle_s else "—")
with col_mod:
    st.write("")
    if st.button("Modifier le cycle", use_container_width=True):
        st.switch_page("vues/2_Preparation_donnees.py")


# ② Stratégies à simuler

st.subheader("② Stratégies à simuler")
st.caption(
    "Les deux stratégies de référence (modèle physique et logique floue) sont "
    "toujours incluses. Cochez les modèles d'IA à ajouter."
)

base_table = pd.DataFrame(
    [
        {
            "Simuler": True,
            "Stratégie": code,
            "Famille": FAMILLES[code][0],
            "Explicabilité": _niveau_explicabilite(code),
            "Coût en temps": FAMILLES[code][1],
        }
        for code in MODELES_NEURONAUX
    ]
)

table_editee = st.data_editor(
    base_table,
    key="table_strategies",
    hide_index=True,
    use_container_width=True,
    disabled=["Stratégie", "Famille", "Explicabilité", "Coût en temps"],
    column_config={
        "Simuler": st.column_config.CheckboxColumn("Simuler", help="Inclure cette stratégie"),
    },
)

selected = set(table_editee.loc[table_editee["Simuler"], "Stratégie"])
# Dépendance : le MLP neuro-symbolique réutilise en interne le LSTM.
dependance_ajoutee = "EMS_MLP_neurosymbolic" in selected and "EMS_LSTM" not in selected
if "EMS_MLP_neurosymbolic" in selected:
    selected.add("EMS_LSTM")


# ③ Conditions initiales

st.subheader("③ Conditions initiales")
c1, c2 = st.columns(2)
soc_eb0 = c1.slider("SOC initial batterie Énergie (%)", 20, 100, 100) / 100.0
soc_pb0 = c2.slider("SOC initial batterie Puissance (%)", 20, 100, 100) / 100.0

if "SOC_EB" in df.columns and nb_points > 0:
    df.loc[df.index[0], "SOC_EB"] = soc_eb0
if "SOC_PB" in df.columns and nb_points > 0:
    df.loc[df.index[0], "SOC_PB"] = soc_pb0


# ④ Résolution

st.subheader("④ Résolution")
resolution = st.radio(
    "Précision du calcul (plus précis = plus lent)",
    ["Rapide", "Standard", "Précise"],
    index=1,
    horizontal=True,
    help="« Rapide » suffit pour explorer ; « Précise » pour des chiffres finaux.",
)
pas_alpha = {"Rapide": 0.005, "Standard": 0.002, "Précise": 0.001}[resolution]
facteur_resolution = {"Rapide": 1.0, "Standard": 1.5, "Précise": 2.5}[resolution]


# ⑤ Analyse de configuration (réagit aux choix)

st.subheader("⑤ Analyse de configuration")

nb_total = len(selected) + 2  # + modèle physique et logique floue
ns_choisis = [m for m in selected if "neurosymbolic" in m]
poids = sum(FAMILLES[m][2] for m in selected if m in FAMILLES) * facteur_resolution
est_lo, est_hi = int(round(poids * 1.5)), int(round(poids * 4.0))

if poids <= 3:
    complexite = "Faible"
elif poids <= 8:
    complexite = "Modérée"
else:
    complexite = "Élevée"

observations = [f"{nb_total} stratégies seront comparées (dont les deux références)."]
if ns_choisis:
    observations.append(
        f"{len(ns_choisis)} approche(s) neuro-symbolique(s) sélectionnée(s) : "
        "décision explicable par construction."
    )
if dependance_ajoutee:
    observations.append(
        "Le LSTM a été ajouté automatiquement : le MLP neuro-symbolique s'appuie sur ses sorties."
    )
if "EMS_GNN" in selected:
    observations.append("Le modèle GNN augmentera nettement le temps de calcul.")
if len(selected) >= 4:
    observations.append("Configuration riche : la comparaison sera pertinente mais plus longue.")
if not selected:
    observations.append("Aucun modèle d'IA sélectionné : seules les deux références seront simulées.")

for obs in observations:
    st.markdown(f"- {obs}")

e1, e2 = st.columns(2)
e1.metric("Complexité du calcul", complexite)
e2.metric("Temps estimé", f"≈ {est_lo} à {est_hi} min" if poids else "—")
st.caption(
    "Estimation indicative : le temps réel dépend de la machine, de la longueur du "
    "cycle et de la résolution choisie."
)


# Contrôle de cohérence des conditions initiales

alertes = []
if soc_eb0 < 0.30 or soc_pb0 < 0.30:
    alertes.append(
        f"États de charge initiaux faibles (Énergie {soc_eb0 * 100:.0f} %, "
        f"Puissance {soc_pb0 * 100:.0f} %) : les performances risquent d'être dégradées "
        "et les violations plus nombreuses."
    )
if abs(soc_eb0 - soc_pb0) >= 0.50:
    alertes.append(
        "Les deux batteries démarrent très déséquilibrées : le comportement initial "
        "sera atypique."
    )

for a in alertes:
    st.warning(a)


# ⑥ Configuration et validation

st.subheader("⑥ Configuration de la simulation")

col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
with col_cfg1:
    with st.container(border=True):
        st.markdown("**Cycle**")
        st.caption(f"{nb_points:,} instants".replace(",", " "))
        st.caption(f"{duree_cycle_s / 60:.1f} min" if duree_cycle_s else "durée inconnue")
with col_cfg2:
    with st.container(border=True):
        st.markdown(f"**Stratégies — {nb_total}**")
        st.caption("Modèle physique, logique floue")
        st.caption(", ".join(sorted(selected)) if selected else "aucun modèle d'IA")
with col_cfg3:
    with st.container(border=True):
        st.markdown("**Conditions et résolution**")
        st.caption(f"SOC Énergie {soc_eb0 * 100:.0f} % · SOC Puissance {soc_pb0 * 100:.0f} %")
        st.caption(f"Résolution {resolution} (pas alpha {pas_alpha})")

st.markdown("**Validation**")
if alertes:
    st.markdown("- ⚠️ Conditions initiales inhabituelles (voir avertissements ci-dessus)")
else:
    st.markdown("- ✔️ Conditions initiales dans une plage usuelle")
st.markdown(f"- ✔️ Cycle chargé ({nb_points:,} instants)".replace(",", " "))
if selected:
    st.markdown(f"- ✔️ {len(selected)} modèle(s) d'IA sélectionné(s)")
else:
    st.markdown("- ⚠️ Aucun modèle d'IA sélectionné : seules les références seront simulées")

st.success("Configuration prête." if not alertes else "Configuration exécutable, avec réserves.")

if st.button("Lancer la simulation", type="primary"):
    supprimer_anciens_resultats()
    modeles_tous, erreurs = _charger_modeles(avec_gnn=("EMS_GNN" in selected))
    modeles_charges = {k: v for k, v in modeles_tous.items() if k in selected}
    erreurs_pertinentes = {k: v for k, v in erreurs.items() if k in selected}

    if "EMS_GNN" in selected and "EMS_GNN" not in modeles_charges:
        st.warning("Le modèle GNN n'a pas pu être chargé : il sera ignoré.")

    debut = time.time()
    with st.spinner("Simulation en cours… (cela peut prendre plusieurs minutes)"):
        resultats, avertissements = _simuler_en_cache(
            df,
            soc_eb0,
            soc_pb0,
            tuple(sorted(modeles_charges.keys())),
            pas_alpha,
            modeles_charges,
        )

    st.session_state["resultats_simulation"] = resultats
    st.session_state["avertissements_simulation"] = avertissements
    st.session_state["erreurs_chargement"] = erreurs_pertinentes
    st.session_state["duree_simulation"] = time.time() - debut
    st.session_state["nb_points_sim"] = nb_points
    st.session_state["_sim_custom_faite"] = True


# Résumé d'exécution

if st.session_state.get("_sim_custom_faite"):
    resultats = st.session_state["resultats_simulation"]
    avertissements = st.session_state.get("avertissements_simulation", [])
    erreurs_ch = st.session_state.get("erreurs_chargement", {})
    duree = st.session_state.get("duree_simulation", 0.0)
    nb = st.session_state.get("nb_points_sim", nb_points)

    st.divider()
    st.subheader("Résumé d'exécution")

    timings, autres = [], []
    for msg in avertissements:
        m = re.match(r"\[timing\]\s*(\S+)\s*:\s*([\d.]+)", msg)
        if m:
            timings.append((m.group(1), float(m.group(2))))
        else:
            autres.append(msg)

    minutes, secondes = int(duree // 60), int(duree % 60)

    st.success("Simulation terminée.")
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Instants simulés", f"{nb:,}".replace(",", " "))
    r2.metric("Stratégies", len(resultats))
    r3.metric("Erreurs", len(erreurs_ch))
    r4.metric("Temps total", f"{minutes} min {secondes} s")

    if erreurs_ch:
        for code, msg in erreurs_ch.items():
            st.error(f"{code} n'a pas pu être chargé : {msg}")

    if timings:
        with st.expander("Durée par stratégie"):
            for code, s in sorted(timings, key=lambda kv: -kv[1]):
                st.markdown(f"- {code} : {s:.0f} s")

    st.markdown("**Résultats disponibles**")
    b1, b2, b3 = st.columns(3)
    if b1.button("Comparer les méthodes", type="primary"):
        st.switch_page("vues/5_Comparaison_des_strategies.py")
    if b2.button("Explorer les résultats"):
        st.switch_page("vues/6_Resultats_et_Analyse.py")
    if b3.button("Comprendre une décision"):
        st.switch_page("vues/7_Explicabilite.py")

    if autres:
        with st.expander("Détails techniques"):
            for msg in autres:
                st.caption(msg)


pied_navigation("vues/8_Simulation_cycle_personnalise.py")
