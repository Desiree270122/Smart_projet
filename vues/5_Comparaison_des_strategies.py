

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


COULEURS = {
    "EMS_power_limitation": "#9AA0A6",
    "EMS_fuzzy_logic": "#C9A227",
    "EMS_MLP": "#6FB1E8",
    "EMS_LSTM": "#1F6FB2",
    "EMS_GNN": "#8E6FD0",
    "EMS_MLP_neurosymbolic": "#E8734A",
    "EMS_LSTM_neurosymbolic": "#2E9E6B",
}
COULEUR_DEFAUT = "#9AA0A6"


def couleur(nom):
    return COULEURS.get(nom, COULEURS.get(nom_affichage(nom), COULEUR_DEFAUT))


COLONNES = {
    "cout_physique_moyen": ("Coût physique moyen", "{:.4f}"),
    "nb_violations": ("Violations SOC", "{:.0f}"),
    "nb_corrections": ("Corrections", "{:.0f}"),
    "taux_faisabilite": ("Faisabilité", "{:.1%}"),
    "soc_eb_final": ("SOC_EB final", "{:.3f}"),
    "soc_pb_final": ("SOC_PB final", "{:.3f}"),
    "desequilibre_soc_moyen": ("Déséquilibre SOC moyen", "{:.4f}"),
    "energie_non_servie_wh": ("Énergie non servie (Wh)", "{:.1f}"),
    "regen_rejetee_wh": ("Régén. rejetée (Wh)", "{:.1f}"),
}


def _criteres_uniques():
    """Deux entrees de CRITERES pointent vers la meme metrique : on ne garde
    que la premiere, sinon le score composite la pondere deux fois."""
    vus, sortie = set(), {}
    for nom_c, (met, sens_c) in CRITERES.items():
        if met in vus:
            continue
        vus.add(met)
        sortie[nom_c] = (met, sens_c)
    return sortie


CRIT = _criteres_uniques()


def fmt(met, val):
    if val != val:
        return "—"
    return COLONNES.get(met, ("", "{:.4f}"))[1].format(val)


@st.cache_data(show_spinner="Chargement des résultats précalculés…")
def _charger():
    donnees = charger_reference()
    return donnees["meta"], calculer_metriques(donnees), donnees["resultats"]


st.title("Comparer les méthodes")
st.caption(
    "Quelle stratégie de gestion d'énergie choisir, et pourquoi ? Choisissez "
    "d'abord le critère qui compte pour vous : tout le reste de la page s'y adapte."
)

try:
    meta, metriques, resultats = _charger()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.info("Cette page lit un résultat calculé hors-ligne. Lancez une fois :\n\n"
            "`python scripts/run_simulations.py`")
    st.stop()

noms = list(metriques.keys())

st.caption(
    f"{meta['cycle_nom']}  ·  {meta['nb_points']:,}".replace(",", " ")
    + f" points  ·  {len(noms)} stratégies  ·  {len(CRIT)} critères"
    + f"  ·  précalcul {meta['duree_simulation_s']:.0f} s"
    + f"  ·  pas d'alpha {meta['pas_alpha']}"
)

st.divider()


# 1 — Le critère pilote la page

critere = st.selectbox(
    "Critère à privilégier",
    list(CRIT.keys()),
    index=list(CRIT.keys()).index("Sécurité physique") if "Sécurité physique" in CRIT else 0,
)
met_c, sens = CRIT[critere]
best_cle, best_val = meilleure_strategie(metriques, critere)


# 2 — La réponse

if best_cle is None:
    st.warning("Métrique indisponible pour ce critère.")
else:
    raisons = [f"{COLONNES[met_c][0]} = {fmt(met_c, best_val)}, la meilleure des sept"]
    if met_c != "nb_violations" and metriques[best_cle].get("nb_violations", 1) == 0:
        raisons.append("aucune violation SOC sur l'ensemble du cycle")
    autres = [c for c in CRIT if c != critere and meilleure_strategie(metriques, c)[0] == best_cle]
    if autres:
        raisons.append("également en tête sur : " + ", ".join(autres))

    with st.container(border=True):
        st.markdown(f"### {nom_affichage(best_cle)}")
        st.caption(f"Stratégie recommandée si vous privilégiez « {critere} »")
        for r in raisons:
            st.markdown(f"- {r}")


# 3 — Classement sur ce critère

st.subheader("Classement sur ce critère")

vals = {n: metriques[n].get(met_c, float("nan")) for n in noms}
finis = {n: v for n, v in vals.items() if v == v}
ordre = sorted(finis, key=lambda n: finis[n], reverse=(sens == "max"))

fig_rang = go.Figure(
    go.Bar(
        x=[finis[n] for n in ordre][::-1],
        y=[nom_affichage(n) for n in ordre][::-1],
        orientation="h",
        marker_color=[couleur(n) for n in ordre][::-1],
        text=[fmt(met_c, finis[n]) for n in ordre][::-1],
        textposition="outside",
        hoverinfo="skip",
    )
)
fig_rang.update_layout(
    height=40 * len(ordre) + 80,
    margin=dict(t=10, b=30, l=10, r=60),
    xaxis_title=COLONNES[met_c][0] + (" (plus haut = mieux)" if sens == "max" else " (plus bas = mieux)"),
    yaxis_title=None,
    showlegend=False,
)
st.plotly_chart(fig_rang, use_container_width=True)


# 4 — Tableau unique, stratégies en lignes

st.subheader("Toutes les stratégies, critère par critère")
st.caption("★ = meilleure valeur. La flèche indique le sens favorable.")

meilleurs = {}
for nom_c, (met, sens_c) in CRIT.items():
    v = {n: metriques[n].get(met, float("nan")) for n in noms}
    f = {n: x for n, x in v.items() if x == x}
    meilleurs[nom_c] = (min if sens_c == "min" else max)(f, key=lambda n: f[n]) if f else None

lignes = []
for n in ordre + [x for x in noms if x not in ordre]:
    ligne = {"Stratégie": nom_affichage(n)}
    for nom_c, (met, sens_c) in CRIT.items():
        entete = f"{nom_c} {'↑' if sens_c == 'max' else '↓'}"
        texte = fmt(met, metriques[n].get(met, float("nan")))
        ligne[entete] = texte + (" ★" if meilleurs[nom_c] == n else "")
    lignes.append(ligne)

st.dataframe(pd.DataFrame(lignes).set_index("Stratégie"), use_container_width=True)


# 5 — Courbes SOC, légende unique

st.subheader("Évolution des états de charge")
st.caption(
    "Plus une courbe descend, plus la batterie a été sollicitée. "
    "Les deux graphes partagent la même légende."
)

chips = "".join(
    f"<span style='display:inline-flex;align-items:center;margin:0 14px 6px 0;font-size:0.85rem'>"
    f"<span style='width:14px;height:3px;background:{couleur(n)};margin-right:6px'></span>"
    f"{nom_affichage(n)}</span>"
    for n in noms
)
st.markdown(f"<div style='margin-bottom:8px'>{chips}</div>", unsafe_allow_html=True)


def courbe(cle_soc, titre):
    fig = go.Figure()
    for n in noms:
        y = np.asarray(resultats[n][cle_soc], dtype=float) * 100.0
        pas = max(1, len(y) // 2000)
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(y))[::pas],
                y=y[::pas],
                mode="lines",
                name=nom_affichage(n),
                line=dict(color=couleur(n), width=1.6),
            )
        )
    fig.update_layout(
        title=titre, xaxis_title="Temps (s)", yaxis_title="SOC (%)",
        height=360, showlegend=False, margin=dict(t=45, b=40, l=50, r=15),
    )
    return fig


g1, g2 = st.columns(2)
g1.plotly_chart(courbe("SOC_EB", "Batterie Énergie"), use_container_width=True)
g2.plotly_chart(courbe("SOC_PB", "Batterie Puissance"), use_container_width=True)


# 6 — Détails

st.subheader("Détails")

with st.expander("Score composite toutes stratégies confondues"):
    def _scores(metriques):
        s = {n: [] for n in metriques}
        for _c, (met, sens_c) in CRIT.items():
            v = {n: m.get(met, float("nan")) for n, m in metriques.items()}
            f = [x for x in v.values() if x == x]
            if not f:
                continue
            lo, hi = min(f), max(f)
            for n, x in v.items():
                if x != x:
                    continue
                r = 0.5 if hi - lo < 1e-12 else (x - lo) / (hi - lo)
                s[n].append(r if sens_c == "max" else 1.0 - r)
        return {n: (sum(v) / len(v) if v else 0.0) for n, v in s.items()}

    sc = _scores(metriques)
    clst = sorted(sc.items(), key=lambda kv: kv[1], reverse=True)
    ecart = (max(sc.values()) - min(sc.values())) * 100 if sc else 0.0

    st.caption(
        "Moyenne des scores normalisés sur tous les critères, chacun ramené à "
        "l'intervalle observé entre les sept stratégies. Ce score pondère tous "
        "les critères également, ce qui n'est pas un choix neutre : il sert de "
        "repère, pas de verdict."
    )
    for rang, (n, v) in enumerate(clst, start=1):
        st.markdown(f"{rang}. {nom_affichage(n)} — {v * 100:.0f} %")
    if ecart < 15:
        st.caption(
            f"Écart de {ecart:.0f} points entre la première et la dernière : "
            "les premières positions ne sont pas significativement départagées."
        )

with st.expander("Tableau complet des métriques brutes"):
    ordre_m = sorted(
        metriques.items(),
        key=lambda kv: kv[1].get(met_c, float("inf")),
        reverse=(sens == "max"),
    )
    brut = []
    for n, m in ordre_m:
        ligne = {"Stratégie": nom_affichage(n)}
        for cle, (lib, f) in COLONNES.items():
            v = m.get(cle, float("nan"))
            ligne[lib] = f.format(v) if v == v else "—"
        brut.append(ligne)
    st.dataframe(pd.DataFrame(brut).set_index("Stratégie"), use_container_width=True)

with st.expander("Explicabilité des modèles"):
    st.caption(
        "L'explicabilité n'est pas mesurable sur une trajectoire : c'est une "
        "propriété de la structure du modèle. Elle est déclarée ci-dessous et "
        "n'entre pas dans le classement chiffré."
    )
    for n in noms:
        niveau, justif = EXPLICABILITE.get(n, (0, "—"))
        st.markdown(
            f"**{nom_affichage(n)}** — {NIVEAUX_EXPLICABILITE.get(niveau, '—')}  \n{justif}"

        )


from core.navigation import pied_navigation

pied_navigation("vues/5_Comparaison_des_strategies.py")