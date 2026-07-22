"""
Page « Comparer les méthodes » — parcours narratif.

Le cœur scientifique (chargement, métriques, scores, critères) est inchangé et
reste la source de vérité : cette page n'en modifie que la présentation, en
guidant l'utilisateur de la vue d'ensemble jusqu'aux tableaux détaillés.
"""

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

C_BLEU = "#3B82F6"
C_VERT = "#22C55E"
C_ORANGE = "#F59E0B"
C_GRIS = "#94A3B8"
MEDAILLES = ["🥇", "🥈", "🥉"]


@st.cache_data(show_spinner="Chargement des résultats précalculés…")
def _charger():
    donnees = charger_reference()
    metriques = calculer_metriques(donnees)
    return donnees["meta"], metriques, donnees["resultats"]


st.title("⚖️ Comparer les méthodes")
st.caption(
    "Quelle stratégie de gestion d'énergie choisir, et pourquoi ? Cette page "
    "compare les sept stratégies sur l'ensemble du cycle, du résumé général "
    "jusqu'aux métriques détaillées."
)

try:
    meta, metriques, resultats = _charger()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.info(
        "Cette page lit un résultat calculé hors-ligne. Lance une fois :\n\n"
        "`python scripts/run_simulations.py`"
    )
    st.stop()


# Bloc 1 — Vue d'ensemble

st.subheader("📊 Vue d'ensemble")

v1, v2, v3, v4, v5 = st.columns(5)
v1.metric("Cycle", str(meta["cycle_nom"]))
v2.metric("Points simulés", f"{meta['nb_points']:,}".replace(",", " "))
v3.metric("Précalcul", f"{meta['duree_simulation_s']:.0f} s")
v4.metric("Stratégies", len(metriques))
v5.metric("Critères mesurés", len(CRITERES))
st.caption(f"Grille de recherche d'alpha : pas de {meta['pas_alpha']}.")


# Cœur scientifique inchangé : scores normalisés moyennés sur tous les critères.

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


scores_globaux = _scores_globaux(metriques)
classement = sorted(scores_globaux.items(), key=lambda kv: kv[1], reverse=True)


# Colonnes affichées + sens de tri (libellés et formats des métriques)

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


# Bloc 2 — Classement général

st.subheader("🏆 Classement général")
st.caption(
    "Les stratégies sont comparées sur l'ensemble des critères mesurés, chacun "
    "normalisé entre elles, afin d'obtenir une vue globale de leurs performances."
)

podium = classement[:3]
cols_podium = st.columns(len(podium))
for col, (nom_s, sc), medaille in zip(cols_podium, podium, MEDAILLES):
    with col:
        with st.container(border=True):
            st.markdown(
                f"<div style='text-align:center;font-size:30px'>{medaille}</div>"
                f"<div style='text-align:center;font-weight:700;font-size:1.05rem'>{nom_affichage(nom_s)}</div>"
                f"<div style='text-align:center;color:{C_BLEU};font-weight:800;font-size:1.4rem'>{sc * 100:.0f} %</div>",
                unsafe_allow_html=True,
            )

with st.expander("Voir le classement complet"):
    for rang, (nom_s, sc) in enumerate(classement, start=1):
        st.markdown(f"{rang}. **{nom_affichage(nom_s)}** — {sc * 100:.0f} %")


# Bloc 3 — Les points forts

st.subheader("⭐ Les points forts")
st.caption("Qui excelle sur quoi ? Chaque carte désigne la stratégie qui domine un critère.")

POINTS_FORTS = [
    ("Coût énergétique", "Coût le plus faible", C_BLEU),
    ("Sécurité physique", "Le moins de violations", C_VERT),
    ("Équilibre EB/PB", "Meilleur équilibre SOC", C_ORANGE),
    ("Alignement au filtre physique", "Le moins de corrections", C_GRIS),
]

cols_forts = st.columns(4)
for col, (crit, libelle, couleur) in zip(cols_forts, POINTS_FORTS):
    cle_f, val_f = meilleure_strategie(metriques, crit)
    metrique_f, _ = CRITERES[crit]
    fmt_f = COLONNES.get(metrique_f, (crit, "{:.4f}"))[1]
    with col:
        with st.container(border=True):
            st.markdown(f"<div style='color:{couleur};font-weight:700'>{libelle}</div>", unsafe_allow_html=True)
            st.markdown(f"**{nom_affichage(cle_f) if cle_f else '—'}**")
            st.caption(fmt_f.format(val_f) if cle_f and val_f == val_f else "—")


# Bloc 4 — Comparaison des comportements

st.subheader("📈 Évolution des états de charge")
st.caption(
    "Ces courbes montrent comment chaque stratégie sollicite la batterie Énergie "
    "et la batterie Puissance au fil du cycle. Plus une courbe descend, plus la "
    "batterie a été utilisée."
)

_noms = list(resultats.keys())


def _courbe_soc(cle_soc, titre):
    fig = go.Figure()
    for nom_s in _noms:
        y = np.asarray(resultats[nom_s][cle_soc], dtype=float) * 100.0
        x = np.arange(len(y))
        pas = max(1, len(y) // 2000)
        fig.add_trace(go.Scatter(x=x[::pas], y=y[::pas], mode="lines", name=nom_affichage(nom_s)))
    fig.update_layout(
        title=titre, xaxis_title="Temps (s)", yaxis_title="SOC (%)", height=380,
        legend_title="Stratégie", margin=dict(t=50, b=40),
    )
    return fig


col_soc_eb, col_soc_pb = st.columns(2)
with col_soc_eb:
    st.plotly_chart(_courbe_soc("SOC_EB", "SOC batterie Énergie"), use_container_width=True)
with col_soc_pb:
    st.plotly_chart(_courbe_soc("SOC_PB", "SOC batterie Puissance"), use_container_width=True)

_best_eq = meilleure_strategie(metriques, "Équilibre EB/PB")[0]
_best_eb = meilleure_strategie(metriques, "Préservation EB")[0]
st.info(
    f"Lecture automatique — **{nom_affichage(_best_eq)}** conserve le meilleur équilibre "
    f"entre les deux batteries sur le cycle, et **{nom_affichage(_best_eb)}** termine avec "
    "la batterie Énergie la mieux préservée."
)


# Bloc 5 — Analyse selon un critère + recommandation

st.subheader("🔍 Analyse selon un critère")
st.caption("Choisissez le critère que vous souhaitez privilégier : la recommandation s'adapte.")

critere = st.selectbox(
    "Critère à privilégier",
    list(CRITERES.keys()),
    index=list(CRITERES.keys()).index("Sécurité physique"),
)

metrique_critere, sens = CRITERES[critere]
meilleure_cle, meilleure_val = meilleure_strategie(metriques, critere)

st.markdown("#### 💡 Recommandation")

if meilleure_cle is None:
    st.warning("Métrique indisponible pour ce critère.")
else:
    raisons = [
        f"{COLONNES[metrique_critere][0]} = "
        f"{COLONNES[metrique_critere][1].format(meilleure_val)} (le meilleur des sept)"
    ]
    if metriques[meilleure_cle].get("nb_violations", 1) == 0:
        raisons.append("aucune violation SOC sur l'ensemble du cycle")
    autres_titres = [
        c for c in CRITERES if c != critere and meilleure_strategie(metriques, c)[0] == meilleure_cle
    ]
    if autres_titres:
        raisons.append("également en tête sur : " + ", ".join(autres_titres))

    with st.container(border=True):
        st.markdown(
            f"Si vous privilégiez **« {critere} »**, la stratégie recommandée est "
            f"**{nom_affichage(meilleure_cle)}**."
        )
        st.markdown("Pourquoi :")
        for r in raisons:
            st.markdown(f"- {r}")


# Bloc 6 — Synthèse automatique

st.subheader("🧠 Synthèse")

_valeurs_scores = list(scores_globaux.values())
_ecart = (max(_valeurs_scores) - min(_valeurs_scores)) if _valeurs_scores else 0.0
_ns = [n for n in metriques if "neurosymbolic" in n]
_non_ns = [n for n in metriques if n not in _ns]
_viol_ns = float(np.mean([metriques[n]["nb_violations"] for n in _ns])) if _ns else None
_viol_autres = float(np.mean([metriques[n]["nb_violations"] for n in _non_ns])) if _non_ns else None
_best_secu = meilleure_strategie(metriques, "Sécurité physique")[0]

_phrases = []
if _ecart < 0.15:
    _phrases.append(
        f"Sur le cycle {meta['cycle_nom']}, les sept stratégies présentent des "
        "performances globales assez proches."
    )
else:
    _phrases.append(
        f"Sur le cycle {meta['cycle_nom']}, les stratégies se différencient nettement "
        f"(écart de {_ecart * 100:.0f} points entre la première et la dernière)."
    )
if _viol_ns is not None and _viol_autres is not None:
    if _viol_ns < _viol_autres:
        _phrases.append(
            f"Les variantes neuro-symboliques réduisent les violations SOC "
            f"({_viol_ns:.0f} en moyenne contre {_viol_autres:.0f} pour les autres)."
        )
    elif _viol_ns > _viol_autres:
        _phrases.append(
            f"Ici, les variantes neuro-symboliques ne réduisent pas les violations SOC "
            f"({_viol_ns:.0f} en moyenne contre {_viol_autres:.0f} pour les autres)."
        )
_phrases.append(
    f"La meilleure sécurité physique revient à **{nom_affichage(_best_secu)}**. "
    "Le choix final dépend donc de l'objectif recherché."
)
st.info(" ".join(_phrases))


# Bloc 7 — Détails (repliés : secondaires pour la plupart des lecteurs)

st.subheader("📋 Détails")

with st.expander("Toutes les stratégies, critère par critère"):
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

    st.dataframe(pd.DataFrame(lignes_matrice).set_index("Critère"), use_container_width=True)
    st.caption("★ = meilleure valeur du critère. Tous ces critères sont mesurés sur le cycle simulé.")

with st.expander("Tableau complet des métriques (trié selon le critère choisi)"):
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
    st.dataframe(pd.DataFrame(lignes).set_index("Stratégie"), use_container_width=True)

with st.expander("Explicabilité des modèles (propriété structurelle)"):
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
    st.dataframe(pd.DataFrame(lignes_x).set_index("Stratégie"), use_container_width=True)


from core.navigation import pied_navigation

pied_navigation("vues/5_Comparaison_des_strategies.py")
