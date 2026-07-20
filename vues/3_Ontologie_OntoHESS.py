"""
Page « Raisonnement intelligent du système HESS ».

Présentation orientée MÉTIER (physiciens, électrotechniciens, spécialistes HESS),
pas informatique. La page répond à une seule question : « Pourquoi l'EMS a-t-il
réparti la puissance ainsi entre les deux batteries ? »

Le formalisme (ontologie OWL, règles) est réutilisé tel quel dans le code mais
présenté en langage physique. Les détails techniques (classes, triplets RDF…)
sont relégués dans un volet repliable pour les experts.
"""

import sys
from pathlib import Path

DOSSIER_PROJET = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DOSSIER_PROJET))

import numpy as np
import streamlit as st
import torch

from ems_core import (
    analyser_capacites_hess,
    compute_symbolic_states,
    alpha_fuzzy_calc,
    resoudre_decision_physique,
    appliquer_scaler,
    charger_scaler,
    load_mlp_simple,
    MLP_INPUT_COLS,
    MLP_SCALER_FILE,
    DEVICE,
    FUZZY_RULE_NAMES,
    RULE_LABELS_FR,
    EPS_POWER_W,
)
from core.resultats import assurer_donnees_session, nom_affichage
from core.navigation import pied_navigation


@st.cache_resource(show_spinner=False)
def _charger_mlp_pour_whatif():
    """Charge le MLP (et son scaler) pour recalculer sa vraie décision en mode
    What-if. Mis en cache : chargé une seule fois."""
    modele = load_mlp_simple()
    modele.eval()
    return modele, charger_scaler(MLP_SCALER_FILE)


# Configuration de page gérée par le routeur Accueil.py.

st.title("Raisonnement intelligent du système HESS")
st.caption(
    "Comprendre la décision énergétique.  ·  "
    "Technologie utilisée : ontologie OWL OntoHESS + règles expertes."
)


# Données : résultats de référence précalculés (via le pont)

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


# Choix de l'instant et de la stratégie de référence

n_points = min(len(traj["P_EB"]) for traj in resultats.values())

col_t, col_s = st.columns([2, 1])

with col_t:
    instant = st.slider("Instant analysé", 0, n_points - 1, n_points // 2)

with col_s:
    strategie = st.selectbox(
        "Stratégie de référence",
        list(resultats.keys()),
        format_func=nom_affichage,
    )

traj = resultats[strategie]

t_sel = float(df["time"].iloc[instant]) if "time" in df.columns else float(instant)
vitesse_ms = float(df["speed"].iloc[instant]) if "speed" in df.columns else None
accel = float(df["hasAcceleration"].iloc[instant]) if "hasAcceleration" in df.columns else 0.0
soc_eb = float(traj["SOC_EB"][instant])
soc_pb = float(traj["SOC_PB"][instant])
p_dem_cycle = float(df["hasPower"].iloc[instant])

# Mode « What-if » : on peut remplacer manuellement la puissance demandée, tout
# en gardant l'état réel de l'instant (SOC, vitesse, accélération...).
utiliser_cycle = st.checkbox("Utiliser la puissance du cycle", value=True)
if utiliser_cycle:
    p_dem = p_dem_cycle
else:
    p_dem = (
        st.number_input(
            "Puissance demandée (kW)",
            value=round(p_dem_cycle / 1000.0, 2),
            step=1.0,
            format="%.2f",
        )
        * 1000.0
    )
mode_whatif = not utiliser_cycle

cap = analyser_capacites_hess(p_dem, soc_eb, soc_pb)
etats = compute_symbolic_states(p_dem, soc_eb, soc_pb)

if utiliser_cycle:
    # Décision réellement prise par la stratégie choisie (précalculée).
    p_eb = float(traj["P_EB"][instant])
    p_pb = float(traj["P_PB"][instant])
    correction = (
        bool(traj["correction_applied"][instant])
        if "correction_applied" in traj
        else False
    )
else:
    # What-if : la répartition précalculée ne correspond plus à cette puissance,
    # on recalcule la décision réellement cohérente avec la puissance modifiée.
    if strategie == "EMS_MLP":
        # Vraie décision du modèle MLP, recalculée pour la puissance modifiée.
        modele_mlp, scaler_mlp = _charger_mlp_pour_whatif()
        vals_mlp = {
            "SOC_EB": soc_eb,
            "SOC_PB": soc_pb,
            "hasPower": p_dem,
            "speed": float(df["speed"].iloc[instant]) if "speed" in df.columns else 0.0,
            "hasAcceleration": accel,
        }
        brut_mlp = np.array([vals_mlp[col] for col in MLP_INPUT_COLS], dtype=np.float64)
        if scaler_mlp is not None:
            brut_mlp = appliquer_scaler(brut_mlp, scaler_mlp)
        x_mlp = torch.tensor([brut_mlp.tolist()], dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            alpha_req = float(modele_mlp(x_mlp).item())
        source_decision_wi = "modèle MLP"
    else:
        # Autres stratégies : proposition des règles floues (interprétable).
        alpha_req = float(
            alpha_fuzzy_calc(
                np.array([soc_eb]),
                np.array([soc_pb]),
                np.array([p_dem]),
                np.array([accel]),
            )["alpha"][0]
        )
        source_decision_wi = "règles floues"

    decision_wi = resoudre_decision_physique(alpha_req, p_dem, soc_eb, soc_pb)
    p_eb = float(decision_wi["P_EB_final"])
    p_pb = float(decision_wi["P_PB_final"])
    correction = bool(decision_wi["correction_applied"])


def kw(x):
    return f"{x / 1000.0:.1f} kW"


if p_dem > EPS_POWER_W:
    regime = "Traction"
elif p_dem < -EPS_POWER_W:
    regime = "Freinage / récupération"
else:
    regime = "Arrêt / roue libre"


# 1. Situation actuelle

st.header("1. Situation actuelle")

if mode_whatif:
    st.warning(
        f"**Mode d'analyse : What-if** — puissance modifiée manuellement "
        f"(**{kw(p_dem)}** au lieu de {kw(p_dem_cycle)} du cycle). Les autres "
        "variables (SOC, vitesse, accélération) restent celles de cet instant ; "
        f"la décision est recalculée ({source_decision_wi} + filtre physique de sécurité)."
    )
else:
    st.caption("Mode d'analyse : Référence — puissance issue du cycle.")

s1, s2, s3, s4 = st.columns(4)
s1.metric("Temps", f"{t_sel:.0f} s")
s2.metric("Vitesse", f"{vitesse_ms * 3.6:.0f} km/h" if vitesse_ms is not None else "n/a")
s3.metric(
    "Puissance demandée",
    kw(p_dem),
    "modifiée" if mode_whatif else None,
)
s4.metric("Mode de fonctionnement", regime)


# 2. État du HESS

st.header("2. État des composants du HESS")

etat_col1, etat_col2, etat_col3 = st.columns(3)

with etat_col1:
    with st.container(border=True):
        st.markdown("**Batterie Énergie**")
        st.markdown(f"SOC : **{soc_eb * 100:.0f} %**")
        st.markdown(f"Puissance disponible : **{kw(cap['eb_dispo_max_W'])}**")
        st.markdown(f"Puissance fournie : **{kw(p_eb)}**")
        st.markdown(f"Puissance restante : **{kw(cap['eb_dispo_max_W'] - max(0.0, p_eb))}**")
        etat = "Disponible" if cap["eb_dispo_max_W"] > 1.0 else "Indisponible"
        if etats["EB_low_SOC"]:
            etat += " (SOC au minimum)" if cap["eb_dispo_max_W"] <= 1.0 else " (presque vide)"
        st.markdown(f"État : {etat}")

with etat_col2:
    with st.container(border=True):
        st.markdown("**Batterie Puissance**")
        st.markdown(f"SOC : **{soc_pb * 100:.0f} %**")
        st.markdown(f"Puissance disponible : **{kw(cap['pb_dispo_max_W'])}**")
        st.markdown(f"Puissance fournie : **{kw(p_pb)}**")
        st.markdown(f"Puissance restante : **{kw(cap['pb_dispo_max_W'] - max(0.0, p_pb))}**")
        etat = "Disponible" if cap["pb_dispo_max_W"] > 1.0 else "Indisponible"
        if etats["PB_low_SOC"]:
            etat += " (SOC au minimum)" if cap["pb_dispo_max_W"] <= 1.0 else " (presque vide)"
        st.markdown(f"État : {etat}")

with etat_col3:
    with st.container(border=True):
        st.markdown("**Convertisseur**")
        conv = "Proche de sa limite" if etats["converter_risk"] else "Fonctionnement normal"
        st.markdown(f"État : {conv}")
        st.markdown("**Filtre de sécurité**")
        st.markdown("Correction appliquée" if correction else "Aucune correction")


# 3. Analyse des capacités

st.header("3. Que peut réellement fournir chaque batterie ?")

cap1, cap2, cap3 = st.columns(3)
cap1.metric("Batterie Énergie — max", kw(cap["eb_dispo_max_W"]))
cap2.metric("Batterie Puissance — max", kw(cap["pb_dispo_max_W"]))
cap3.metric("HESS — total disponible", kw(cap["hess_dispo_max_W"]))

if p_dem > EPS_POWER_W:
    eb_seule = p_dem <= cap["eb_dispo_max_W"] + EPS_POWER_W
    pb_seule = p_dem <= cap["pb_dispo_max_W"] + EPS_POWER_W

    if eb_seule:
        conclusion = "La batterie Énergie peut satisfaire **seule** la demande."
    elif pb_seule and not eb_seule:
        conclusion = (
            "La batterie Énergie ne peut pas satisfaire seule la demande : "
            "la batterie Puissance doit participer."
        )
    elif cap["faisable"]:
        conclusion = (
            "Aucune batterie ne suffit seule : les **deux ensemble** couvrent la "
            f"demande ({kw(cap['hess_dispo_max_W'])} disponibles pour {kw(p_dem)})."
        )
    else:
        conclusion = (
            f"**Demande non réalisable** : puissance disponible insuffisante "
            f"(manque de {kw(cap['P_non_servie_W'])})."
        )

    if cap["faisable"]:
        st.success(f"Le système conclut : {conclusion}")
    else:
        st.error(f"Le système conclut : {conclusion}")
        st.info(
            "**Réaction du système** : aucune répartition ne permet de couvrir la "
            "demande en respectant les limites physiques (SOC minimal des batteries, "
            "convertisseur). Les batteries fournissent le maximum autorisé et "
            f"**{kw(cap['P_non_servie_W'])} restent non servis**. Le filtre de sécurité "
            "protège les batteries contre la décharge profonde, au détriment de la "
            "performance : en pratique, la propulsion du véhicule serait limitée à cet "
            "instant (le conducteur n'obtient pas toute la puissance demandée)."
        )
elif p_dem < -EPS_POWER_W:
    st.info(
        "Phase de freinage : l'énergie récupérée est absorbée par les batteries "
        f"(non récupérée : {kw(cap['P_regen_rejetee_W'])})."
    )
else:
    st.info("Demande quasi nulle : le véhicule est à l'arrêt ou en roue libre.")


# 4. Raisonnement intelligent

st.header("4. Raisonnement du système")

st.subheader("Ce que le système détecte")

phrases_etats = []
if etats["EB_available"]:
    phrases_etats.append("La batterie Énergie est disponible.")
else:
    phrases_etats.append("La batterie Énergie n'est plus disponible (SOC au minimum).")
if etats["PB_available"]:
    phrases_etats.append("La batterie Puissance est disponible.")
else:
    phrases_etats.append("La batterie Puissance n'est plus disponible (SOC au minimum).")
if etats["EB_low_SOC"]:
    phrases_etats.append("La batterie Énergie est presque vide.")
if etats["PB_low_SOC"]:
    phrases_etats.append("La batterie Puissance est presque vide.")
if etats["high_power_demand"]:
    phrases_etats.append("Forte demande de puissance.")
if etats["regenerative_braking"]:
    phrases_etats.append("Le véhicule est en freinage régénératif.")
if etats["zero_power_demand"]:
    phrases_etats.append("La demande de puissance est quasi nulle.")
if etats["converter_risk"]:
    phrases_etats.append("Le convertisseur est proche de sa limite.")
else:
    phrases_etats.append("Aucune limitation du convertisseur.")

for phrase in phrases_etats:
    st.markdown(f"- {phrase}")

st.subheader("Règle experte appliquée")

res_fuzzy = alpha_fuzzy_calc(
    np.array([soc_eb]),
    np.array([soc_pb]),
    np.array([p_dem]),
    np.array([accel]),
)
forces = np.asarray(res_fuzzy["strengths"][0], dtype=float)
regle_dominante = str(res_fuzzy["dominant_rule"][0])

regles_actives = sorted(
    [
        (FUZZY_RULE_NAMES[i], forces[i])
        for i in range(len(FUZZY_RULE_NAMES))
        if forces[i] > 0.05
    ],
    key=lambda x: x[1],
    reverse=True,
)

if not regles_actives:
    st.info(
        "Aucune règle experte ne domine clairement : le système applique une "
        "répartition prudente par défaut."
    )
else:
    for nom_regle, force in regles_actives:
        libelle = RULE_LABELS_FR.get(nom_regle, nom_regle)
        principale = nom_regle == regle_dominante
        with st.container(border=True):
            st.markdown(
                ("**Règle principale**" if principale else "**Règle secondaire**")
                + f" — {libelle}."
            )
            st.progress(min(1.0, float(force)), text=f"Intensité : {force * 100:.0f} %")


# 5. Décision finale

st.header("5. Décision finale")

if mode_whatif:
    st.caption(
        f"Mode What-if : répartition **recalculée** pour la puissance modifiée, "
        f"à partir de la décision du **{source_decision_wi}**, puis passée au filtre "
        "physique de sécurité."
    )

total_mag = abs(p_eb) + abs(p_pb)
part_eb = 100.0 * abs(p_eb) / total_mag if total_mag > 1.0 else 0.0
part_pb = 100.0 * abs(p_pb) / total_mag if total_mag > 1.0 else 0.0

d1, d2 = st.columns(2)
with d1:
    st.metric("Batterie Énergie fournit", kw(p_eb), f"{part_eb:.0f} % de la répartition")
with d2:
    st.metric("Batterie Puissance fournit", kw(p_pb), f"{part_pb:.0f} % de la répartition")

st.markdown(
    f"La batterie Énergie fournit **{kw(p_eb)}** ({part_eb:.0f} %) et la batterie "
    f"Puissance fournit **{kw(p_pb)}** ({part_pb:.0f} %). "
    + (
        "Le filtre de sécurité a **corrigé** la répartition proposée pour respecter "
        "les limites physiques des batteries et du convertisseur."
        if correction
        else "La décision respectait déjà les contraintes physiques : aucune correction "
        "n'a été nécessaire."
    )
)


# 6. Détails techniques (optionnels) — pour experts

st.divider()

with st.expander("Détails techniques : ontologie OWL OntoHESS (pour experts)"):
    chemins = [
        DOSSIER_PROJET / "ontologies" / "OntoHESS2.owl",
        DOSSIER_PROJET / "ontology" / "OntoHESS2.owl",
        DOSSIER_PROJET / "ontologie" / "OntoHESS2.owl",
        DOSSIER_PROJET / "OntoHESS2.owl",
    ]
    chemin_owl = next((c for c in chemins if c.exists()), None)

    if chemin_owl is None:
        st.info("Fichier OWL introuvable — la logique de décision reste opérationnelle.")
    else:
        try:
            from rdflib import Graph, RDF, OWL

            graphe = Graph()
            graphe.parse(str(chemin_owl))
            t1, t2, t3 = st.columns(3)
            t1.metric("Triplets RDF", len(graphe))
            t2.metric("Classes OWL", len(set(graphe.subjects(RDF.type, OWL.Class))))
            t3.metric("Individus", len(set(graphe.subjects(RDF.type, OWL.NamedIndividual))))
            st.caption(f"Ontologie chargée depuis : {chemin_owl.name}")
        except Exception as exc:  # noqa: BLE001
            st.info(f"Ontologie non analysée : {exc}")

with st.expander(f"Détails techniques : les {len(FUZZY_RULE_NAMES)} règles expertes"):
    for nom_regle in FUZZY_RULE_NAMES:
        st.markdown(f"- **{nom_regle}** : {RULE_LABELS_FR.get(nom_regle, '')}")


pied_navigation("vues/3_Ontologie_OntoHESS.py")
