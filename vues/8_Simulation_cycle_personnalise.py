"""
Page « Lancer une simulation » : que va-t-on simuler ?

Trois étapes : le cycle, les stratégies, le lancement. Toutes les stratégies
sélectionnées sont simulées sur le même cycle avec le même modèle physique du
HESS ; l'analyse des résultats se fait ensuite dans « Résultats de simulation ».
La logique de simulation elle-même est inchangée.
"""

import os
import re
import sys
import time
from pathlib import Path

DOSSIER_PROJET = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DOSSIER_PROJET))

import numpy as np
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
    DT_SECONDS,
)
from core.resultats import FAMILLES, nom_affichage
from core.navigation import pied_navigation
from core.style import flux_html, COULEUR_DECISION, COULEUR_EB, COULEUR_REFERENCE, COULEUR_SECONDAIRE
from core import ontology_explainer as ox

# Configuration de page gérée par le routeur Accueil.py.

try:
    torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
except Exception:
    pass


# Les deux références (modèle physique, logique floue) sont toujours simulées.
REFERENCES = ["EMS_power_limitation", "EMS_fuzzy_logic"]

# Coût relatif en temps de calcul des modèles d'IA (poids de l'estimation).
COUT_CALCUL = {
    "EMS_MLP": ("rapide", 1.0),
    "EMS_MLP_neurosymbolic": ("modéré", 1.5),
    "EMS_LSTM": ("modéré", 1.5),
    "EMS_LSTM_neurosymbolic": ("modéré", 1.5),
    "EMS_GNN": ("lent", 3.0),
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


def kw(x):
    return f"{x / 1000.0:.1f} kW"


st.title("▶️ Lancer une simulation")
st.caption(
    "Choisir le cycle et les stratégies à simuler, puis exécuter la même simulation "
    "physique pour toutes les stratégies sélectionnées."
)

if "cycle_pret" not in st.session_state:
    st.warning("Aucun cycle préparé. Commencez par la page « Préparer une simulation ».")
    if st.button("Aller à la préparation"):
        st.switch_page("vues/2_Preparation_donnees.py")
    st.stop()

df = st.session_state["cycle_pret"].copy()
nb_points = len(df)
duree_cycle_s = float(df["time"].iloc[-1]) if "time" in df.columns and nb_points else nb_points * DT_SECONDS
p_dem = df["hasPower"].to_numpy(dtype=float) if "hasPower" in df.columns else np.zeros(nb_points)


# Étape 1 — le cycle

st.subheader("1 · Le cycle")
with st.container(border=True):
    origine = "préparé par vous" if "prep_meta" in st.session_state else "cycle de référence"
    c = st.columns(4)
    c[0].metric("Cycle utilisé", origine)
    c[1].metric("Durée", f"{duree_cycle_s / 60:.0f} min", help=f"{nb_points:,} instants".replace(",", " "))
    c[2].metric("Demande maximale", kw(p_dem.max(initial=0.0)))
    c[3].metric("Freinage maximal", kw(-p_dem.min(initial=0.0)))
    st.caption(
        f"Énergie de traction demandée : {np.clip(p_dem, 0, None).sum() * DT_SECONDS / 3.6e6:.2f} kWh ; "
        f"énergie récupérable au freinage : {-np.clip(p_dem, None, 0).sum() * DT_SECONDS / 3.6e6:.2f} kWh."
    )
    if st.button("Modifier le cycle"):
        st.switch_page("vues/2_Preparation_donnees.py")

    with st.expander("Conditions initiales et précision du calcul"):
        def _soc_initial(cle):
            return int(np.clip(round(float(st.session_state.get(cle, 1.0)) * 100), 20, 100))

        s1, s2 = st.columns(2)
        soc_eb0 = s1.slider("SOC initial de la batterie Énergie (%)", 20, 100, _soc_initial("soc_eb0")) / 100.0
        soc_pb0 = s2.slider("SOC initial de la batterie Puissance (%)", 20, 100, _soc_initial("soc_pb0")) / 100.0
        resolution = st.radio(
            "Finesse de la recherche de alpha par le filtre de sécurité (plus fin = plus lent)",
            ["Exploration", "Analyse", "Validation"],
            index=1,
            horizontal=True,
            help="« Exploration » pour dégrossir ; « Validation » pour des chiffres publiables.",
        )
        st.markdown("**Hypothèses de simulation**")
        for hypothese in ox.HYPOTHESES:
            st.markdown(f"- {hypothese}")

pas_alpha = {"Exploration": 0.005, "Analyse": 0.002, "Validation": 0.001}[resolution]
facteur_resolution = {"Exploration": 1.0, "Analyse": 1.5, "Validation": 2.5}[resolution]

if "SOC_EB" in df.columns and nb_points > 0:
    df.loc[df.index[0], "SOC_EB"] = soc_eb0
if "SOC_PB" in df.columns and nb_points > 0:
    df.loc[df.index[0], "SOC_PB"] = soc_pb0


# Étape 2 — les stratégies

st.subheader("2 · Les stratégies")
st.caption(
    "Le modèle physique et la logique floue sont toujours simulés : ils servent de "
    "référence aux autres pages. Cochez les modèles d'IA à ajouter."
)

selected = set()
colonnes = st.columns(len(FAMILLES))
for col, (nom_famille, membres) in zip(colonnes, FAMILLES.items()):
    with col:
        with st.container(border=True):
            st.markdown(f"**{nom_famille}**")
            for cle in membres:
                if cle in REFERENCES:
                    st.checkbox(nom_affichage(cle), value=True, disabled=True, key=f"sim_{cle}",
                                help="Toujours simulée : stratégie de référence")
                elif st.checkbox(nom_affichage(cle), value=True, key=f"sim_{cle}",
                                 help=f"Temps de calcul {COUT_CALCUL[cle][0]}"):
                    selected.add(cle)

# Dépendance : NS-MLP reçoit en entrée les prédictions du LSTM.
if "EMS_MLP_neurosymbolic" in selected and "EMS_LSTM" not in selected:
    selected.add("EMS_LSTM")
    st.caption("Le LSTM sera aussi simulé : NS-MLP utilise ses prédictions comme entrées.")


# Étape 3 — lancer

nb_total = len(selected) + len(REFERENCES)
poids = sum(COUT_CALCUL[m][1] for m in selected) * facteur_resolution
est_lo, est_hi = int(round(poids * 1.5)), int(round(poids * 4.0))

st.subheader("3 · Lancer")
with st.container(border=True):
    st.markdown("**Toutes les stratégies sont comparées dans les mêmes conditions**")
    st.markdown(
        flux_html(
            [
                ("Même profil de conduite", COULEUR_REFERENCE),
                (f"{nb_total} EMS", COULEUR_DECISION),
                ("Même modèle physique du HESS", COULEUR_SECONDAIRE),
                ("SOC · puissance · courant · pertes", COULEUR_EB),
            ]
        ),
        unsafe_allow_html=True,
    )
    st.caption(
        "Chaque EMS propose sa répartition alpha à chaque instant ; le même filtre de sécurité "
        "et le même modèle de batteries calculent ensuite l'évolution du système. Seule la "
        "décision change d'une stratégie à l'autre."
    )

    diagnostic = ox.diagnostic_configuration(soc_eb0, soc_pb0, nb_total, "Comparer plusieurs stratégies")
    for alerte in diagnostic["alertes"]:
        st.warning(alerte)
    with st.expander("Vérifications de l'ontologie avant lancement"):
        d1, d2 = st.columns(2)
        with d1:
            st.markdown("**Contexte identifié**")
            for element in diagnostic["contexte"]:
                st.markdown(f"- {'✔️' if element['reconnu'] else '—'} {element['libelle']}  ·  `{element['concept']}`")
        with d2:
            st.markdown("**Contraintes principales**")
            for element in diagnostic["contraintes"]:
                st.markdown(f"- {'✔️' if element['reconnu'] else '—'} {element['libelle']}  ·  `{element['concept']}`")
        for conseil in diagnostic["conseils"]:
            st.info(conseil)
        st.success(diagnostic["conclusion"])
        st.caption(
            "Concepts vérifiés par lecture directe des classes déclarées dans ontologies/OntoHESS2.owl."
        )

    st.caption(
        f"{nb_total} stratégies · précision « {resolution} » · temps estimé : "
        + (f"{est_lo} à {est_hi} min (selon la machine et la longueur du cycle)." if poids else "quelques secondes.")
    )
    lancer = st.button("🚀 Lancer la simulation", type="primary")

if lancer:
    supprimer_anciens_resultats()
    modeles_tous, erreurs = _charger_modeles(avec_gnn=("EMS_GNN" in selected))
    modeles_charges = {k: v for k, v in modeles_tous.items() if k in selected}
    erreurs_pertinentes = {k: v for k, v in erreurs.items() if k in selected}

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
    st.session_state["pas_alpha"] = pas_alpha
    st.session_state["soc_eb0"] = soc_eb0
    st.session_state["soc_pb0"] = soc_pb0
    st.session_state["avertissements_simulation"] = avertissements
    st.session_state["erreurs_chargement"] = erreurs_pertinentes
    st.session_state["duree_simulation"] = time.time() - debut
    st.session_state["_source_donnees"] = "simulation lancée sur cette page"
    st.session_state["_sim_custom_faite"] = True


# Après le calcul : un simple compte rendu, l'analyse est dans « Résultats de simulation »

if st.session_state.get("_sim_custom_faite"):
    resultats = st.session_state["resultats_simulation"]
    erreurs_ch = st.session_state.get("erreurs_chargement", {})
    duree = st.session_state.get("duree_simulation", 0.0)

    with st.container(border=True):
        st.markdown("### ✓ Simulation terminée")
        st.markdown(
            f"**{len(resultats)} stratégies simulées** · durée : {int(duree // 60)} min {int(duree % 60)} s"
        )
        for code, msg in erreurs_ch.items():
            st.error(f"{nom_affichage(code)} n'a pas pu être chargé et n'a pas été simulé : {msg}")
        if st.button("Voir les résultats →", type="primary"):
            st.switch_page("vues/6_Resultats_et_Analyse.py")

    timings, autres = [], []
    for msg in st.session_state.get("avertissements_simulation", []):
        m = re.match(r"\[timing\]\s*(\S+)\s*:\s*([\d.]+)", msg)
        if m:
            timings.append((m.group(1), float(m.group(2))))
        else:
            autres.append(msg)
    if timings or autres:
        with st.expander("Détails techniques de l'exécution"):
            for code, s in sorted(timings, key=lambda kv: -kv[1]):
                st.markdown(f"- {nom_affichage(code)} : {s:.0f} s")
            for msg in autres:
                st.caption(msg)


pied_navigation("vues/8_Simulation_cycle_personnalise.py")
