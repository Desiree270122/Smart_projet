

import pandas as pd
import streamlit as st

from core.resultats import (
    charger_reference,
    calculer_metriques,
    meilleure_strategie,
    nom_affichage,
    CRITERES,
)


# Configuration de page gérée par le routeur Accueil.py.


@st.cache_data(show_spinner="Chargement des résultats précalculés…")
def _charger():
    donnees = charger_reference()
    metriques = calculer_metriques(donnees)
    return donnees["meta"], metriques


st.title("Comparaison des stratégies EMS")

try:
    meta, metriques = _charger()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.info(
        "Cette page lit un résultat calculé hors-ligne. Lance une fois :\n\n"
        "`python scripts/run_simulations.py`"
    )
    st.stop()


st.caption(
    f"Cycle **{meta['cycle_nom']}** — {meta['nb_points']:,} échantillons — "
    f"grille alpha {meta['pas_alpha']} — précalculé en {meta['duree_simulation_s']:.0f} s"
    .replace(",", " ")
)


# ------------------------------------------------------------
# Colonnes affichées + sens de tri
# ------------------------------------------------------------

COLONNES = {
    "cout_physique_moyen": ("Coût physique moyen", "{:.4f}"),
    "nb_violations": ("Violations SOC", "{:.0f}"),
    "nb_corrections": ("Corrections", "{:.0f}"),
    "taux_faisabilite": ("Faisabilité %", "{:.1%}"),
    "soc_eb_final": ("SOC_EB final", "{:.3f}"),
    "soc_pb_final": ("SOC_PB final", "{:.3f}"),
    "desequilibre_soc_moyen": ("Déséquilibre SOC moyen", "{:.4f}"),
    "energie_non_servie_wh": ("Énergie non servie (Wh)", "{:.1f}"),
    "regen_rejetee_wh": ("Régén. rejetée (Wh)", "{:.1f}"),
}


critere = st.selectbox(
    "Critère pour désigner la meilleure stratégie",
    list(CRITERES.keys()),
    index=list(CRITERES.keys()).index("Sécurité physique"),
)

metrique_critere, sens = CRITERES[critere]
meilleure_cle, meilleure_val = meilleure_strategie(metriques, critere)


# ------------------------------------------------------------
# Verdict
# ------------------------------------------------------------

st.subheader("Verdict")

if meilleure_cle is None:
    st.warning("Métrique indisponible pour ce critère.")
else:
    st.success(
        f"Selon le critère **« {critere} »**, la meilleure stratégie est "
        f"**{nom_affichage(meilleure_cle)}** "
        f"({COLONNES[metrique_critere][0]} = {COLONNES[metrique_critere][1].format(meilleure_val)})."
    )


# ------------------------------------------------------------
# Tableau de classement (trié selon le critère)
# ------------------------------------------------------------

ordre = sorted(
    metriques.items(),
    key=lambda kv: kv[1].get(metrique_critere, float("inf")),
    reverse=(sens == "max"),
)

lignes = []
for nom, m in ordre:
    ligne = {"Stratégie": nom_affichage(nom)}
    for cle, (libelle, fmt) in COLONNES.items():
        valeur = m.get(cle, float("nan"))
        ligne[libelle] = fmt.format(valeur) if valeur == valeur else "—"  # NaN -> —
    lignes.append(ligne)

tableau = pd.DataFrame(lignes).set_index("Stratégie")
st.dataframe(tableau, use_container_width=True)


# ------------------------------------------------------------
# Note d'honnêteté scientifique (coût vs conformité)
# ------------------------------------------------------------

from core.navigation import pied_navigation

pied_navigation("pages/5_Comparaison_des_strategies.py")
