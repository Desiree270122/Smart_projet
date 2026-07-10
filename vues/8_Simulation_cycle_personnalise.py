"""
Page « Nouvelle simulation » — assistant simple pour lancer une simulation sur
un cycle personnalisé.

Orientée utilisateur (physicien) : pas de logs internes, pas de noms de
notebooks. Un déroulé en étapes (cycle, stratégies, conditions, résolution,
lancement), puis un résumé clair avec des boutons vers l'analyse. Les détails
techniques éventuels sont relégués dans un volet repliable.
"""

import os
import re
import sys
import time
from pathlib import Path

DOSSIER_PROJET = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DOSSIER_PROJET))

import streamlit as st
import torch

from ems_core import (
    simuler_toutes_strategies,
    set_alpha_grid_step,
    MODEL_DISPLAY_NAMES,
    load_mlp_simple,
    load_mlp_neurosymbolic,
    load_lstm_seul,
    load_lstm_neurosymbolic,
    load_gnn_simple,
)
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

CLES_RESULTATS_A_SUPPRIMER = [
    "resultats_simulation",
    "avertissements_simulation",
    "signature_simulation",
    "modele_actif",
    "alpha_star_reference",
    "_sim_custom_faite",
]


def nom_affiche(code):
    return code


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


# ============================================================
# Page
# ============================================================

st.title("Nouvelle simulation")

st.info(
    "Cette page relance une simulation complète sur le cycle préparé — c'est "
    "**long** (plusieurs minutes). Pour la démonstration, les pages Comparaison, "
    "Analyse détaillée et « Pourquoi cette décision ? » affichent déjà des "
    "résultats précalculés, instantanés. N'utilise cette page que pour tester "
    "**ton propre cycle**."
)

if "cycle_pret" not in st.session_state:
    st.warning("Aucun cycle préparé. Commence par la page Préparation.")
    if st.button("Aller à la préparation"):
        st.switch_page("vues/2_Preparation_donnees.py")
    st.stop()

df = st.session_state["cycle_pret"].copy()
nb_points = len(df)


# ------------------------------------------------------------
# Étape 1 — Cycle
# ------------------------------------------------------------

st.header("Étape 1 — Cycle de conduite")
st.write(
    f"Cycle préparé : **{nb_points:,} instants**. "
    "Pour changer de cycle, retourne à la page Préparation.".replace(",", " ")
)


# ------------------------------------------------------------
# Étape 2 — Stratégies
# ------------------------------------------------------------

st.header("Étape 2 — Stratégies à simuler")
st.caption(
    "Les stratégies de référence (priorité Énergie et logique floue) sont "
    "toujours incluses. Choisis les modèles d'IA à ajouter."
)

cols = st.columns(3)
selection = {}
for i, m in enumerate(MODELES_NEURONAUX):
    with cols[i % 3]:
        selection[m] = st.checkbox(nom_affiche(m), value=True)

selected = {m for m, v in selection.items() if v}
# Dépendance : le MLP neuro-symbolique réutilise en interne le LSTM.
if "EMS_MLP_neurosymbolic" in selected:
    selected.add("EMS_LSTM")


# ------------------------------------------------------------
# Étape 3 — Conditions initiales
# ------------------------------------------------------------

st.header("Étape 3 — Conditions initiales")
c1, c2 = st.columns(2)
soc_eb0 = c1.slider("SOC initial batterie Énergie (%)", 20, 100, 100) / 100.0
soc_pb0 = c2.slider("SOC initial batterie Puissance (%)", 20, 100, 100) / 100.0

if "SOC_EB" in df.columns and nb_points > 0:
    df.loc[df.index[0], "SOC_EB"] = soc_eb0
if "SOC_PB" in df.columns and nb_points > 0:
    df.loc[df.index[0], "SOC_PB"] = soc_pb0


# ------------------------------------------------------------
# Étape 4 — Résolution
# ------------------------------------------------------------

st.header("Étape 4 — Résolution")
resolution = st.radio(
    "Précision du calcul (plus précis = plus lent)",
    ["Rapide", "Standard", "Précise"],
    horizontal=True,
    help="« Rapide » suffit pour explorer ; « Précise » pour des chiffres finaux.",
)
pas_alpha = {"Rapide": 0.005, "Standard": 0.002, "Précise": 0.001}[resolution]


# ------------------------------------------------------------
# Étape 5 — Lancer
# ------------------------------------------------------------

st.header("Étape 5 — Lancer")

st.markdown("**Exécution de la simulation**")
st.caption(
    "La simulation est exécutée stratégie par stratégie au sein d'un calcul "
    "global. L'application affiche un indicateur d'exécution pendant le "
    "traitement, puis présente les temps de calcul détaillés une fois la "
    "simulation terminée."
)

if st.button("Lancer la simulation", type="primary"):
    supprimer_anciens_resultats()
    modeles_tous, _erreurs = _charger_modeles(avec_gnn=("EMS_GNN" in selected))
    modeles_charges = {k: v for k, v in modeles_tous.items() if k in selected}

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
    st.session_state["duree_simulation"] = time.time() - debut
    st.session_state["nb_points_sim"] = nb_points
    st.session_state["_sim_custom_faite"] = True


# ------------------------------------------------------------
# Résumé final (uniquement si une simulation a été lancée ici)
# ------------------------------------------------------------

if st.session_state.get("_sim_custom_faite"):
    resultats = st.session_state["resultats_simulation"]
    avertissements = st.session_state.get("avertissements_simulation", [])
    duree = st.session_state.get("duree_simulation", 0.0)
    nb = st.session_state.get("nb_points_sim", nb_points)

    st.divider()
    st.header("Simulation terminée")

    timings, autres = [], []
    for msg in avertissements:
        m = re.match(r"\[timing\]\s*(\S+)\s*:\s*([\d.]+)", msg)
        if m:
            timings.append((m.group(1), float(m.group(2))))
        else:
            autres.append(msg)

    minutes, secondes = int(duree // 60), int(duree % 60)
    st.success(
        f"{len(resultats)} stratégies simulées sur {nb:,} instants "
        f"en {minutes} min {secondes} s.".replace(",", " ")
    )

    if timings:
        st.markdown("**Durée par stratégie**")
        for code, s in timings:
            st.markdown(f"- {nom_affiche(code)} : {s:.0f} s")

    st.markdown("**Tous les résultats sont disponibles :**")
    b1, b2, b3 = st.columns(3)
    if b1.button("Comparer les stratégies", type="primary"):
        st.switch_page("vues/5_Comparaison_des_strategies.py")
    if b2.button("Voir l'analyse détaillée"):
        st.switch_page("vues/6_Resultats_et_Analyse.py")
    if b3.button("Comprendre une décision"):
        st.switch_page("vues/7_Explicabilite.py")

    if autres:
        with st.expander("Détails techniques (développeur)"):
            for msg in autres:
                st.caption(msg)


pied_navigation("vues/8_Simulation_cycle_personnalise.py")
