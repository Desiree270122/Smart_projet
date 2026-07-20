

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


# Classement général (moyenne des scores normalisés sur tous les critères)

def _scores_globaux(metriques):
    """Score 0-1 par stratégie : moyenne des scores normalisés sur les critères
    (1 = meilleur). Permet un classement synthétique."""
    scores = {n: [] for n in metriques}
    for _crit, (metrique, sens_c) in CRITERES.items():
        vals = {n: m.get(metrique, float("nan")) for n, m in metriques.items()}
        finis = [v for v in vals.values() if v == v]
        if not finis:
            continue
        lo, hi = min(finis), max(finis)
        for n, v in vals.items():
            if v != v:
                continue
            x = 0.5 if hi - lo < 1e-12 else (v - lo) / (hi - lo)
            scores[n].append(x if sens_c == "max" else 1.0 - x)
    return {n: (sum(s) / len(s) if s else 0.0) for n, s in scores.items()}


classement = sorted(_scores_globaux(metriques).items(), key=lambda kv: kv[1], reverse=True)

st.subheader("Classement général")

col_cl, col_hi = st.columns(2)
with col_cl:
    for rang, (nom_s, sc) in enumerate(classement, start=1):
        st.markdown(f"{rang}. **{nom_affichage(nom_s)}**  ({sc * 100:.0f} %)")
with col_hi:
    st.markdown(
        f"- Meilleur coût : **{nom_affichage(meilleure_strategie(metriques, 'Coût énergétique')[0])}**\n"
        f"- Meilleure sécurité : **{nom_affichage(meilleure_strategie(metriques, 'Sécurité physique')[0])}**\n"
        f"- Meilleur équilibre SOC : **{nom_affichage(meilleure_strategie(metriques, 'Équilibre EB/PB')[0])}**\n"
        f"- Meilleure explicabilité : **{nom_affichage(meilleure_strategie(metriques, 'Explicabilité')[0])}**"
    )

st.divider()


# Colonnes affichées + sens de tri

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


# Verdict

st.subheader("Verdict")

if meilleure_cle is None:
    st.warning("Métrique indisponible pour ce critère.")
else:
    st.success(
        f"Selon le critère **« {critere} »**, la meilleure stratégie est "
        f"**{nom_affichage(meilleure_cle)}** "
        f"({COLONNES[metrique_critere][0]} = {COLONNES[metrique_critere][1].format(meilleure_val)})."
    )


# Meilleure stratégie pour CHAQUE critère (vue d'ensemble)

st.markdown("**Meilleure stratégie par critère**")

POURQUOI = {
    "Sécurité physique": "le moins de violations SOC",
    "Coût énergétique": "coût physique minimal",
    "Préservation EB": "SOC final EB le plus élevé",
    "Préservation PB": "SOC final PB le plus élevé",
    "Équilibre EB/PB": "déséquilibre SOC le plus faible",
    "Performance globale": "coût physique minimal",
    "Explicabilité": "le moins de corrections du filtre",
}

lignes_critere = []
for crit in CRITERES:
    cle_c, val_c = meilleure_strategie(metriques, crit)
    metrique_c, _sens_c = CRITERES[crit]
    fmt_c = COLONNES.get(metrique_c, (crit, "{:.4f}"))[1]
    lignes_critere.append(
        {
            "Critère": crit,
            "Meilleure stratégie": nom_affichage(cle_c) if cle_c else "—",
            "Valeur": fmt_c.format(val_c) if cle_c and val_c == val_c else "—",
            "Pourquoi": POURQUOI.get(crit, ""),
        }
    )

st.dataframe(
    pd.DataFrame(lignes_critere).set_index("Critère"),
    use_container_width=True,
)


# Tableau de classement (trié selon le critère)

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


from core.navigation import pied_navigation

pied_navigation("vues/5_Comparaison_des_strategies.py")
