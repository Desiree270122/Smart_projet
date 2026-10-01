"""
Page « Lancer une simulation » : que va-t-on simuler ?

Trois étapes : le cycle, les stratégies, le lancement. Toutes les stratégies
choisies sont simulées sur le même cycle avec le même modèle physique du HESS ;
l'analyse des résultats se fait ensuite dans « Résultats de simulation ».
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

import ems_core as core
from ems_core import (
    DT_SECONDS,
    load_gnn_simple,
    load_lstm_neurosymbolic,
    load_lstm_seul,
    load_mlp_neurosymbolic,
    load_mlp_simple,
    set_alpha_grid_step,
    set_pertes,
    simuler_toutes_strategies,
)
from core import ontology_explainer as ox
from core.format import nombre
from core.i18n import lib, tr
from core.navigation import pied_navigation
from core.resultats import (
    CLE_PERSONNALISE, CYCLES_REFERENCE, FAMILLES, afficher_simulation_personnalisee,
    charger_reference, libelle_cycle, nom_affichage,
)
from core.style import COULEUR_DECISION, COULEUR_EB, COULEUR_REFERENCE, COULEUR_SECONDAIRE, flux_html

try:
    torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
except Exception:  # noqa: BLE001
    pass


# Les deux références (modèle physique, logique floue) sont toujours simulées.
REFERENCES = ["EMS_power_limitation", "EMS_fuzzy_logic"]

# Temps de calcul relatif des stratégies à apprentissage : (libellé, poids de l'estimation).
COUT_CALCUL = {
    "EMS_MLP": (("rapide", "fast"), 1.0),
    "EMS_MLP_neurosymbolic": (("moyen", "medium"), 1.5),
    "EMS_LSTM": (("moyen", "medium"), 1.5),
    "EMS_LSTM_neurosymbolic": (("moyen", "medium"), 1.5),
    "EMS_GNN": (("long", "slow"), 3.0),
}

# Précision du calcul : pas de recherche de la répartition par le filtre de sécurité, et facteur de durée.
PRECISIONS = {
    "rapide": (("Rapide", "Fast"), 0.005, 1.0),
    "normale": (("Normale", "Standard"), 0.002, 1.5),
    "fine": (("Fine", "Fine"), 0.001, 2.5),
}


@st.cache_resource(show_spinner=False)
def _charger_modeles_deterministes():
    """Charge les quatre réseaux MLP, NS-MLP, LSTM et NS-LSTM."""
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
    """Charge le GNN à part (plus long à charger, seulement à la demande)."""
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
def _simuler_en_cache(df, soc_eb0, soc_pb0, signature_modeles, pas_alpha, pertes, _modeles_charges):
    # Réglages globaux du moteur : posés pour ce calcul, puis remis par défaut,
    # pour ne pas déteindre sur les autres sessions du même serveur.
    set_alpha_grid_step(pas_alpha)
    set_pertes(*pertes)
    try:
        with torch.inference_mode():
            return simuler_toutes_strategies(df, soc_eb0, soc_pb0, _modeles_charges)
    finally:
        set_pertes(False)
        set_alpha_grid_step(core.ALPHA_GRID_STEP_DEFAUT)


@st.cache_data(show_spinner=False)
def _cycle_reference(cle):
    return charger_reference(CYCLES_REFERENCE[cle][1])["cycle_df"]


def kw(x):
    return f"{nombre(x / 1000.0, 1)} kW"


st.title(tr("▶️ Lancer une simulation", "▶️ Run a simulation"))
st.caption(tr(
    "Choisir le cycle et les stratégies, puis lancer la même simulation physique pour toutes les "
    "stratégies choisies.",
    "Choose the cycle and the strategies, then run the same physical simulation for all the "
    "selected strategies.",
))


# Étape 1 — le cycle

cles_cycles = (["prepare"] if "cycle_prepare" in st.session_state else []) + [
    c for c, (_, chemin) in CYCLES_REFERENCE.items() if Path(chemin).exists()
]
if not cles_cycles:
    st.warning(tr(
        "Aucun cycle disponible. Commencez par la page « Préparer une simulation ».",
        "No cycle available. Start with the “Prepare a simulation” page.",
    ))
    if st.button(tr("Aller à la préparation", "Go to preparation")):
        st.switch_page("vues/2_Preparation_donnees.py")
    st.stop()

st.subheader(tr("1 · Le cycle", "1 · The cycle"))
with st.container(border=True):
    c_choix, c_btn = st.columns([3, 1])
    cle_cycle = c_choix.selectbox(tr("Cycle à simuler", "Cycle to simulate"), cles_cycles, format_func=libelle_cycle)
    with c_btn:
        st.write("")
        if st.button(tr("Préparer un autre cycle", "Prepare another cycle"), width="stretch"):
            st.switch_page("vues/2_Preparation_donnees.py")

    df = (st.session_state["cycle_prepare"] if cle_cycle == "prepare" else _cycle_reference(cle_cycle)).copy()
    nb_points = len(df)
    duree_cycle_s = float(df["time"].iloc[-1]) if "time" in df.columns and nb_points else nb_points * DT_SECONDS
    p_dem = df["hasPower"].to_numpy(dtype=float) if "hasPower" in df.columns else np.zeros(nb_points)

    c = st.columns(4)
    c[0].metric(tr("Durée", "Duration"), f"{nombre(duree_cycle_s / 60, 0)} min")
    c[1].metric(tr("Distance", "Distance"), f"{nombre(df['speed'].sum() * DT_SECONDS / 1000, 0)} km" if "speed" in df.columns else "—")
    c[2].metric(tr("Demande maximale", "Peak demand"), kw(p_dem.max(initial=0.0)))
    c[3].metric(tr("Freinage maximal", "Peak braking"), kw(-p_dem.min(initial=0.0)))
    energie_nette = p_dem.sum() * DT_SECONDS / 3.6e6
    energie_utile = ((1 - core.SOC_EB_MIN) * core.ENERGY_EB_WH + (1 - core.SOC_PB_MIN) * core.ENERGY_PB_WH) / 1000
    st.caption(tr(
        "Énergie nette demandée : {a} kWh, pour {b} kWh utilisables dans le HESS entre 100 % et le SOC minimal.",
        "Net energy demanded: {a} kWh, for {b} kWh usable in the HESS between 100 % and the minimum SOC.",
        a=nombre(energie_nette, 2), b=nombre(energie_utile, 2),
    ))
    if energie_nette > energie_utile:
        st.warning(tr(
            "Ce cycle demande plus d'énergie que le HESS n'en contient : toutes les stratégies "
            "manqueront d'énergie avant la fin. Réduisez le nombre de répétitions.",
            "This cycle demands more energy than the HESS contains: every strategy will run out of "
            "energy before the end. Reduce the number of repetitions.",
        ))

    with st.expander(tr("États de charge initiaux et précision du calcul", "Initial states of charge and calculation precision")):
        def _soc_initial(cle):
            return int(np.clip(round(float(st.session_state.get(cle, 1.0)) * 100), 20, 100))

        s1, s2 = st.columns(2)
        suffixe = "_prepare" if cle_cycle == "prepare" else "_aucun"
        soc_eb0 = s1.slider(tr("SOC initial de la batterie Énergie (%)", "Initial SOC of the Energy battery (%)"),
                            20, 100, _soc_initial("soc_eb0" + suffixe)) / 100.0
        soc_pb0 = s2.slider(tr("SOC initial de la batterie Puissance (%)", "Initial SOC of the Power battery (%)"),
                            20, 100, _soc_initial("soc_pb0" + suffixe)) / 100.0
        precision = st.radio(
            tr("Précision du calcul (plus fin = plus long)", "Calculation precision (finer = slower)"),
            list(PRECISIONS), index=1, horizontal=True, format_func=lambda p: lib(PRECISIONS[p][0]),
            help=tr(
                "Pas avec lequel le filtre de sécurité cherche la répartition la plus proche de celle "
                "demandée : 0,5 %, 0,2 % ou 0,1 % de la puissance.",
                "Step with which the safety filter searches for the split closest to the requested "
                "one: 0.5 %, 0.2 % or 0.1 % of the power.",
            ),
        )
        st.markdown(tr("**Hypothèses de la simulation**", "**Simulation assumptions**"))
        st.markdown("\n".join(f"- {h}" for h in ox.hypotheses()))

    with st.expander(tr("Pertes : non comptées par défaut, comme dans l'article de référence",
                        "Losses: not counted by default, as in the reference paper")):
        avec_pertes = st.checkbox(
            tr("Compter les pertes dans le calcul des états de charge", "Count the losses when computing the states of charge"),
            help=tr(
                "Chaque batterie fournit alors aussi ses pertes par effet Joule, et la batterie Énergie celles du convertisseur.",
                "Each battery then also supplies its Joule losses, and the Energy battery those of the converter.",
            ),
        )
        p1, p2, p3 = st.columns(3)
        aide_r = tr("{r} mΩ par cellule × {s} en série / {p} en parallèle", "{r} mΩ per cell × {s} in series / {p} in parallel",
                    r="{r}", s="{s}", p="{p}")
        r_eb = p1.number_input(
            tr("Résistance interne de la batterie Énergie (mΩ)", "Internal resistance of the Energy battery (mΩ)"),
            1.0, 5000.0, round(core.R_EB_PACK_OHM * 1000, 1), 10.0, disabled=not avec_pertes,
            help=aide_r.format(r=nombre(core.CELL_EB_RINT_OHM * 1000, 0), s=core.CELL_EB_N_SERIE, p=core.CELL_EB_N_PARALLELE),
        )
        r_pb = p2.number_input(
            tr("Résistance interne de la batterie Puissance (mΩ)", "Internal resistance of the Power battery (mΩ)"),
            1.0, 5000.0, round(core.R_PB_PACK_OHM * 1000, 1), 10.0, disabled=not avec_pertes,
            help=aide_r.format(r=nombre(core.CELL_PB_RINT_OHM * 1000, 1), s=core.CELL_PB_N_SERIE, p=core.CELL_PB_N_PARALLELE),
        )
        mode_rendement = p3.radio(
            tr("Rendement du convertisseur", "Converter efficiency"), ["courbe", "constant"],
            format_func={"courbe": tr("Courbe mesurée (article, fig. 33)", "Measured curve (paper, fig. 33)"),
                         "constant": tr("Valeur constante", "Constant value")}.get,
            disabled=not avec_pertes,
        )
        rendement_constant = None
        if mode_rendement == "constant":
            rendement_constant = st.slider(tr("Rendement constant (%)", "Constant efficiency (%)"), 80.0, 100.0, 95.5, 0.1,
                                           disabled=not avec_pertes) / 100.0
        st.caption(tr(
            "Courbe mesurée : 91,5 % à 1,2 kW, 95,5 % vers 2,5 kW. Ici le convertisseur ne traite "
            "qu'environ 10 % de la puissance de la batterie Énergie, souvent moins de 1,2 kW : la "
            "valeur à 1,2 kW est alors conservée, ce qui sous-estime légèrement ses pertes.",
            "Measured curve: 91.5 % at 1.2 kW, 95.5 % around 2.5 kW. Here the converter only "
            "processes about 10 % of the Energy battery's power, often less than 1.2 kW: the value at "
            "1.2 kW is then kept, which slightly underestimates its losses.",
        ))
pertes = (avec_pertes, rendement_constant, r_eb / 1000.0, r_pb / 1000.0)
_, pas_alpha, facteur_duree = PRECISIONS[precision]

if "SOC_EB" in df.columns and nb_points > 0:
    df.loc[df.index[0], "SOC_EB"] = soc_eb0
if "SOC_PB" in df.columns and nb_points > 0:
    df.loc[df.index[0], "SOC_PB"] = soc_pb0


# Étape 2 — les stratégies

st.subheader(tr("2 · Les stratégies", "2 · The strategies"))
st.caption(tr(
    "Le modèle physique et la logique floue sont toujours simulés : ils servent de référence. "
    "Cochez les stratégies à apprentissage à ajouter.",
    "The physical model and fuzzy logic are always simulated: they serve as references. Tick the "
    "learning-based strategies to add.",
))

selected = set()
colonnes = st.columns(len(FAMILLES))
for col, (nom_famille, membres) in zip(colonnes, FAMILLES.items()):
    with col:
        with st.container(border=True):
            st.markdown(f"**{lib(nom_famille)}**")
            for cle in membres:
                if cle in REFERENCES:
                    st.checkbox(nom_affichage(cle), value=True, disabled=True, key=f"sim_{cle}",
                                help=tr("Toujours simulée : stratégie de référence", "Always simulated: reference strategy"))
                elif st.checkbox(nom_affichage(cle), value=True, key=f"sim_{cle}",
                                 help=tr("Temps de calcul : {t}", "Computing time: {t}", t=lib(COUT_CALCUL[cle][0]))):
                    selected.add(cle)

# NS-MLP reçoit en entrée les prévisions de NS-LSTM.
if "EMS_MLP_neurosymbolic" in selected and "EMS_LSTM_neurosymbolic" not in selected:
    selected.add("EMS_LSTM_neurosymbolic")
    st.caption(tr(
        "NS-LSTM sera aussi simulé : NS-MLP utilise ses prévisions.",
        "NS-LSTM will also be simulated: NS-MLP uses its forecasts.",
    ))


# Étape 3 — lancer

nb_total = len(selected) + len(REFERENCES)
poids = sum(COUT_CALCUL[m][1] for m in selected) * facteur_duree
est_lo, est_hi = int(round(poids * 1.5)), int(round(poids * 4.0))

st.subheader(tr("3 · Lancer", "3 · Run"))
with st.container(border=True):
    st.markdown(tr(
        "**Toutes les stratégies sont comparées dans les mêmes conditions**",
        "**All strategies are compared under the same conditions**",
    ))
    st.markdown(
        flux_html([
            (tr("Même profil de conduite", "Same driving profile"), COULEUR_REFERENCE),
            (tr("{n} stratégies EMS", "{n} EMS strategies", n=nb_total), COULEUR_DECISION),
            (tr("Même modèle physique du HESS", "Same physical model of the HESS"), COULEUR_SECONDAIRE),
            (tr("SOC · puissances · courants · pertes", "SOC · powers · currents · losses"), COULEUR_EB),
        ]),
        unsafe_allow_html=True,
    )
    st.caption(tr(
        "À chaque instant, chaque stratégie propose sa répartition de la puissance ; le même filtre de "
        "sécurité et le même modèle de batteries calculent ensuite l'évolution du système. Seule la "
        "décision change d'une stratégie à l'autre.",
        "At each time step, each strategy proposes its power split; the same safety filter and the "
        "same battery model then compute how the system evolves. Only the decision changes from one "
        "strategy to another.",
    ))

    diagnostic = ox.diagnostic_configuration(soc_eb0, soc_pb0, nb_total)
    for alerte in diagnostic["alertes"]:
        st.warning(alerte)
    with st.expander(tr("Vérifications avant lancement (base de connaissances OntoHESS)",
                        "Checks before running (OntoHESS knowledge base)")):
        d1, d2 = st.columns(2)
        with d1:
            st.markdown(tr("**Système reconnu**", "**System recognised**"))
            st.markdown("\n".join(f"- {'✔️' if e['reconnu'] else '—'} {e['libelle']}" for e in diagnostic["contexte"]))
        with d2:
            st.markdown(tr("**Contraintes prises en compte**", "**Constraints taken into account**"))
            st.markdown("\n".join(f"- {'✔️' if e['reconnu'] else '—'} {e['libelle']}" for e in diagnostic["contraintes"]))
        for conseil in diagnostic["conseils"]:
            st.info(conseil)
        st.success(diagnostic["conclusion"])

    st.caption(
        tr("{n} stratégies · précision « {p} » · durée estimée : ", "{n} strategies · “{p}” precision · estimated time: ",
           n=nb_total, p=lib(PRECISIONS[precision][0]).lower())
        + (tr("{a} à {b} min, selon l'ordinateur et la longueur du cycle.", "{a} to {b} min, depending on the computer and the cycle length.",
              a=est_lo, b=est_hi) if poids else tr("quelques secondes.", "a few seconds."))
    )
    lancer = st.button(tr("🚀 Lancer la simulation", "🚀 Run the simulation"), type="primary")

if lancer:
    modeles_tous, erreurs = _charger_modeles(avec_gnn=("EMS_GNN" in selected))
    modeles_charges = {k: v for k, v in modeles_tous.items() if k in selected}
    erreurs_pertinentes = {k: v for k, v in erreurs.items() if k in selected}

    debut = time.time()
    with st.spinner(tr("Simulation en cours… (cela peut prendre plusieurs minutes)", "Simulation running… (this may take several minutes)")):
        resultats, avertissements = _simuler_en_cache(
            df, soc_eb0, soc_pb0, tuple(sorted(modeles_charges.keys())), pas_alpha, pertes, modeles_charges,
        )

    afficher_simulation_personnalisee(
        st,
        {
            "resultats": resultats,
            "cycle_df": df,
            "avertissements": avertissements,
            "meta": {
                "cycle_cle": cle_cycle,
                "soc_eb0": soc_eb0,
                "soc_pb0": soc_pb0,
                "pas_alpha": pas_alpha,
                "pertes_dans_soc": avec_pertes,
            },
        },
    )
    st.session_state["erreurs_chargement"] = erreurs_pertinentes
    st.session_state["duree_simulation"] = time.time() - debut
    st.session_state["_sim_custom_faite"] = True
    st.rerun()


# Après le calcul : un simple compte rendu ; l'analyse est dans « Résultats de simulation »

if st.session_state.get("_sim_custom_faite") and CLE_PERSONNALISE in st.session_state:
    derniere = st.session_state[CLE_PERSONNALISE]
    resultats = derniere["resultats"]
    erreurs_ch = st.session_state.get("erreurs_chargement", {})
    duree = st.session_state.get("duree_simulation", 0.0)
    meta = derniere["meta"]

    with st.container(border=True):
        st.markdown(tr("### ✓ Simulation terminée", "### ✓ Simulation complete"))
        st.markdown(tr(
            "**{n} stratégies simulées** sur « {c} » ({p}) · durée : {m} min {s} s",
            "**{n} strategies simulated** on “{c}” ({p}) · time: {m} min {s} s",
            n=len(resultats), c=libelle_cycle(meta.get("cycle_cle", "")),
            p=tr("pertes comptées", "losses counted") if meta.get("pertes_dans_soc") else tr("sans pertes", "without losses"),
            m=int(duree // 60), s=int(duree % 60),
        ))
        st.caption(tr(
            "Les pages d'analyse affichent maintenant cette simulation (« Ma dernière simulation » dans leur choix de cycle).",
            "The analysis pages now show this simulation (“My last simulation” in their cycle selector).",
        ))
        for code in erreurs_ch:
            st.error(tr(
                "{s} n'a pas pu être chargée et n'a pas été simulée.", "{s} could not be loaded and was not simulated.",
                s=nom_affichage(code),
            ))
        if st.button(tr("Voir les résultats →", "See the results →"), type="primary"):
            st.switch_page("vues/6_Resultats_et_Analyse.py")

    durees = []
    for msg in derniere.get("avertissements", []):
        m = re.match(r"\[timing\]\s*(\S+)\s*:\s*([\d.]+)", msg)
        if m:
            durees.append((m.group(1), float(m.group(2))))
    if durees:
        with st.expander(tr("Durée de calcul par stratégie", "Computing time per strategy")):
            st.markdown("\n".join(f"- {nom_affichage(code)} : {nombre(s, 0)} s" for code, s in sorted(durees, key=lambda kv: -kv[1])))


pied_navigation("vues/8_Simulation_cycle_personnalise.py")
