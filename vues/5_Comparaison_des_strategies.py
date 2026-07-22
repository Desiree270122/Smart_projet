

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.resultats import (
    charger_reference,
    calculer_metriques,
    meilleure_strategie,
    nom_affichage,
    CRITERES,
    EXPLICABILITE,
    NIVEAUX_EXPLICABILITE,
)


# Configuration de page gérée par le routeur Accueil.py.


@st.cache_data(show_spinner="Chargement des résultats précalculés…")
def _charger():
    donnees = charger_reference()
    metriques = calculer_metriques(donnees)
    return donnees["meta"], metriques, donnees["resultats"]


st.title("Comparaison des stratégies EMS")

try:
    meta, metriques, resultats = _charger()
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
        f"- Meilleur alignement au filtre : "
        f"**{nom_affichage(meilleure_strategie(metriques, 'Alignement au filtre physique')[0])}**"
    )

st.divider()


# Évolution des SOC de toutes les stratégies (comparaison visuelle)

st.subheader("Évolution des états de charge (comparaison)")

_noms = list(resultats.keys())


def _courbe_soc(cle_soc, titre):
    fig = go.Figure()
    for nom_s in _noms:
        y = np.asarray(resultats[nom_s][cle_soc], dtype=float) * 100.0
        x = np.arange(len(y))
        pas = max(1, len(y) // 2000)
        fig.add_trace(go.Scatter(x=x[::pas], y=y[::pas], mode="lines", name=nom_affichage(nom_s)))
    fig.update_layout(
        title=titre,
        xaxis_title="Temps (s)",
        yaxis_title="SOC (%)",
        height=380,
        legend_title="Stratégie",
        margin=dict(t=50, b=40),
    )
    return fig


col_soc_eb, col_soc_pb = st.columns(2)
with col_soc_eb:
    st.plotly_chart(_courbe_soc("SOC_EB", "SOC batterie Énergie"), use_container_width=True)
with col_soc_pb:
    st.plotly_chart(_courbe_soc("SOC_PB", "SOC batterie Puissance"), use_container_width=True)

st.caption(
    f"Les {len(_noms)} stratégies (logique floue et GNN inclus) sont comparées sur le même cycle. "
    "Clique dans la légende pour isoler une stratégie."
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


# Toutes les stratégies, critère par critère (et non le seul gagnant)

st.subheader("Toutes les stratégies, critère par critère")

lignes_matrice = []
for crit, (metrique_c, sens_c) in CRITERES.items():
    fmt_c = COLONNES.get(metrique_c, (crit, "{:.4f}"))[1]
    vals = {n: metriques[n].get(metrique_c, float("nan")) for n in metriques}
    finis = {n: v for n, v in vals.items() if v == v}
    best = (min if sens_c == "min" else max)(finis, key=lambda n: finis[n]) if finis else None

    ligne = {"Critère": crit}
    for n in metriques:
        v = vals[n]
        texte = fmt_c.format(v) if v == v else "—"
        if n == best:
            texte += " ★"
        ligne[nom_affichage(n)] = texte
    lignes_matrice.append(ligne)

st.dataframe(
    pd.DataFrame(lignes_matrice).set_index("Critère"),
    use_container_width=True,
)
st.caption(
    f"Les {len(metriques)} stratégies sont affichées pour chaque critère. "
    "★ = meilleure valeur du critère. Tous ces critères sont **mesurés** sur le cycle simulé."
)


# Explicabilité : propriété structurelle, déclarée et justifiée (jamais mesurée)

st.subheader("Explicabilité des modèles")

st.caption(
    "L'explicabilité n'est pas une grandeur mesurable sur une trajectoire : c'est une "
    "propriété de la structure du modèle. Elle est donc déclarée et justifiée ci-dessous, "
    "et n'entre pas dans le classement chiffré."
)

lignes_x = []
for n in metriques:
    niveau, justification = EXPLICABILITE.get(n, (0, "—"))
    lignes_x.append(
        {
            "Stratégie": nom_affichage(n),
            "Niveau": NIVEAUX_EXPLICABILITE.get(niveau, "—"),
            "Pourquoi": justification,
        }
    )

st.dataframe(
    pd.DataFrame(lignes_x).set_index("Stratégie"),
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
