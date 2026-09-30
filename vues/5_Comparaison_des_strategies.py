

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.resultats import (
    charger_reference,
    calculer_metriques,
    nom_affichage,
    CRITERES,
)
from core.style import couleur


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


def _rangs_competition(finis, sens, affiche):
    """Rang 1..N avec ex æquo (style sportif : 1, 1, puis 3). Deux valeurs
    identiques une fois affichées (affiche : valeur -> texte) sont ex æquo : le
    classement ne départage pas ce que le tableau montre égal.
    Retourne (ordre trié, {stratégie: rang})."""
    ordre = sorted(finis, key=lambda n: finis[n], reverse=(sens == "max"))
    rangs, rang, precedent = {}, 0, None
    for i, n in enumerate(ordre):
        v = affiche(finis[n])
        if v != precedent:
            rang = i + 1
        rangs[n] = rang
        precedent = v
    return ordre, rangs


def _rang_txt(rang):
    return f"{rang}{'er' if rang == 1 else 'e'}"


@st.cache_data(show_spinner="Chargement des résultats précalculés…")
def _charger():
    donnees = charger_reference()
    return donnees["meta"], calculer_metriques(donnees), donnees["resultats"]


st.title("⚖️ Comparer les méthodes")
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

# Une stratégie qui n'a pas fourni toute la puissance demandée est sortie du
# classement : le modèle du HESS étant sans pertes, l'énergie non fournie reste
# dans les batteries et fait paraître son SOC final meilleur à tort.
SEUIL_NON_SERVI_WH = 1.0
non_servie = {n: metriques[n].get("energie_non_servie_wh", 0.0) for n in noms}
classees = [n for n in noms if non_servie[n] < SEUIL_NON_SERVI_WH] or list(noms)
hors_classement = [n for n in noms if n not in classees]


def _wh(x):
    return f"{x:,.0f} Wh".replace(",", " ")

st.caption(
    f"{meta['cycle_nom']}  ·  {meta['nb_points']:,}".replace(",", " ")
    + f" points  ·  {len(noms)} stratégies  ·  {len(CRIT)} critères"
    + f"  ·  précalcul {meta['duree_simulation_s']:.0f} s"
    + f"  ·  pas d'alpha {meta['pas_alpha']}"
)

st.divider()


# 1 — Le critère pilote la page

# Par défaut, un critère qui départage nettement les stratégies classées :
# « Sécurité physique » les met toutes à égalité (0 violation) et « Coût
# physique » ne sépare le GNN du modèle physique que de 0,2 %.
CRITERE_DEFAUT = "Alignement au filtre physique"
critere = st.selectbox(
    "Critère à privilégier",
    list(CRIT.keys()),
    index=list(CRIT.keys()).index(CRITERE_DEFAUT) if CRITERE_DEFAUT in CRIT else 0,
)
met_c, sens = CRIT[critere]


def _classement(critere_c):
    """(ordre, rangs) des stratégies classées sur un critère, ex æquo compris."""
    met, sens_c = CRIT[critere_c]
    v = {n: metriques[n].get(met, float("nan")) for n in classees}
    v = {n: x for n, x in v.items() if x == x}
    return _rangs_competition(v, sens_c, lambda x: fmt(met, x))


def _premiers(critere_c):
    """Stratégies classées premières sur un critère, ex æquo compris."""
    return {n for n, k in _classement(critere_c)[1].items() if k == 1}


vals = {n: metriques[n].get(met_c, float("nan")) for n in noms}
finis = {n: v for n, v in vals.items() if v == v}
ordre, rangs = _classement(critere)
gagnants = [n for n in ordre if rangs.get(n) == 1]  # ex æquo au rang 1
# Hors classement : affichées après les autres, dans le même sens de tri.
ordre_hors = sorted((n for n in hors_classement if n in finis), key=lambda n: finis[n], reverse=(sens == "max"))


# 2 — La réponse

if not gagnants:
    st.warning("Métrique indisponible pour ce critère.")
else:
    ref = gagnants[0]
    raisons = [
        f"{COLONNES[met_c][0]} = {fmt(met_c, finis[ref])} "
        + (
            f"(meilleur résultat parmi les {len(classees)} stratégies qui fournissent toute la demande)"
            if hors_classement
            else f"(meilleur résultat parmi les {len(noms)} stratégies évaluées)"
        )
    ]
    if met_c != "nb_violations" and all(metriques[g].get("nb_violations", 1) == 0 for g in gagnants):
        raisons.append("aucune violation des contraintes de SOC sur l'ensemble du cycle")
    # Un autre critère n'est cité que si CHAQUE stratégie en tête y est aussi première.
    autres = [c for c in CRIT if c != critere and set(gagnants) <= _premiers(c)]
    if autres:
        raisons.append(
            ("également classée première pour : " if len(gagnants) == 1
             else "également classées premières ensemble pour : ")
            + ", ".join(autres)
        )

    _noms_g = [f"**{nom_affichage(g)}**" for g in gagnants]
    _liste_g = " et ".join([", ".join(_noms_g[:-1]), _noms_g[-1]]) if len(_noms_g) > 1 else _noms_g[0]

    with st.container(border=True):
        if len(gagnants) == 1:
            st.markdown(f"### 🏆 {nom_affichage(ref)}")
            st.caption(f"Stratégie recommandée lorsque le critère « {critere} » est prioritaire.")
        else:
            st.markdown("### 🏆 Première place (égalité)")
            st.markdown(
                f"Les stratégies {_liste_g} obtiennent un **résultat identique** et occupent "
                f"la première place lorsque le critère « {critere} » est prioritaire."
            )
        st.markdown("**Résultats clés**")
        for r in raisons:
            st.markdown(f"- {r}")

if hors_classement:
    st.warning(
        "**Hors classement** : "
        + ", ".join(f"{nom_affichage(n)} ({_wh(non_servie[n])} non fournis)" for n in hors_classement)
        + ". Ces stratégies n'ont pas fourni toute la puissance demandée sur le cycle. "
        "Le modèle du HESS étant sans pertes, l'énergie non fournie reste dans les "
        "batteries : leur SOC final paraît meilleur, mais ce n'est pas un gain."
    )


# 3 — Classement sur ce critère

st.subheader("🏅 Classement sur ce critère")
st.caption(
    "Les stratégies au résultat identique (à la précision affichée) partagent le même rang."
    + (" En grisé, en bas : les stratégies hors classement." if ordre_hors else "")
)

barres = ordre + ordre_hors
fig_rang = go.Figure(
    go.Bar(
        x=[finis[n] for n in barres][::-1],
        y=[
            (f"{_rang_txt(rangs[n])}  " if n in rangs else "hors cl.  ") + nom_affichage(n)
            for n in barres
        ][::-1],
        orientation="h",
        marker=dict(
            color=[couleur(n) for n in barres][::-1],
            opacity=[1.0 if n in rangs else 0.35 for n in barres][::-1],
        ),
        text=[
            fmt(met_c, finis[n]) + ("" if n in rangs else f"  ·  {_wh(non_servie[n])} non fournis")
            for n in barres
        ][::-1],
        textposition="outside",
        hoverinfo="skip",
    )
)
fig_rang.update_layout(
    height=40 * len(barres) + 80,
    margin=dict(t=10, b=30, l=10, r=60),
    xaxis_title=COLONNES[met_c][0] + (" (plus haut = mieux)" if sens == "max" else " (plus bas = mieux)"),
    yaxis_title=None,
    showlegend=False,
)
st.plotly_chart(fig_rang, width="stretch")


# 4 — Tableau unique, stratégies en lignes

st.subheader("📊 Toutes les stratégies, critère par critère")
st.caption(
    "★ = meilleure valeur parmi les stratégies classées (plusieurs ★ si ex æquo). "
    "La flèche indique le sens favorable."
)

meilleurs = {nom_c: _premiers(nom_c) for nom_c in CRIT}

lignes = []
for n in ordre + [x for x in noms if x not in ordre]:
    ligne = {"Stratégie": nom_affichage(n) + ("" if n in classees else " (hors classement)")}
    for nom_c, (met, sens_c) in CRIT.items():
        entete = f"{nom_c} {'↑' if sens_c == 'max' else '↓'}"
        texte = fmt(met, metriques[n].get(met, float("nan")))
        ligne[entete] = texte + (" ★" if n in meilleurs[nom_c] else "")
    ligne["Demande non fournie (Wh)"] = f"{non_servie[n]:.0f}"
    lignes.append(ligne)

st.dataframe(pd.DataFrame(lignes).set_index("Stratégie"), width="stretch")


# 5 — Courbes SOC, légende unique

st.subheader("📉 Évolution des états de charge")
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
g1.plotly_chart(courbe("SOC_EB", "Batterie Énergie"), width="stretch")
g2.plotly_chart(courbe("SOC_PB", "Batterie Puissance"), width="stretch")


# 6 — Détails

st.subheader("📋 Détails")

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

    sc = _scores({n: metriques[n] for n in classees})
    _, rangs_sc = _rangs_competition(sc, "max", lambda v: f"{v * 100:.0f}")
    clst = sorted(sc.items(), key=lambda kv: kv[1], reverse=True)
    ecart = (max(sc.values()) - min(sc.values())) * 100 if sc else 0.0

    st.caption(
        "Moyenne des scores normalisés sur tous les critères, chacun ramené à "
        f"l'intervalle observé entre les {len(classees)} stratégies classées. Ce score "
        "pondère tous les critères également, ce qui n'est pas un choix neutre : il "
        "sert de repère, pas de verdict."
    )
    for n, v in clst:
        st.markdown(f"{_rang_txt(rangs_sc[n])}. {nom_affichage(n)} — {v * 100:.0f} %")
    for n in hors_classement:
        st.markdown(f"— {nom_affichage(n)} — hors classement ({_wh(non_servie[n])} non fournis)")
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
    st.dataframe(pd.DataFrame(brut).set_index("Stratégie"), width="stretch")

from core.navigation import pied_navigation

pied_navigation("vues/5_Comparaison_des_strategies.py")