

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ems_core as ec
from core.pertes import RENDEMENT_CONVERTISSEUR, bilan_pertes, resistances_packs
from core.resultats import (
    charger_reference,
    calculer_metriques,
    famille,
    nom_affichage,
    CRITERES,
    FAMILLES,
    PAIRES_SYMBOLIQUE,
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
    "pertes_totales_wh": ("Pertes estimées (Wh)", "{:.0f}"),
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


# Les quatre familles de l'offre de stage

st.subheader("Les quatre familles de stratégies")
st.caption(
    "Le projet 2SMART compare quatre approches : règles fixes, ontologie seule, "
    "apprentissage seul et approche hybride (ici neurosymbolique). Pour chacune, sa "
    "meilleure stratégie sur le critère choisi."
)
lignes_f = []
for fam, membres in FAMILLES.items():
    presents = [n for n in membres if n in finis]
    if not presents:
        continue
    classes_f = [n for n in presents if n in rangs]
    meilleure = min(classes_f, key=lambda n: rangs[n]) if classes_f else None
    lignes_f.append(
        {
            "Famille": fam,
            "Stratégies": ", ".join(nom_affichage(n) for n in presents),
            "Meilleure": nom_affichage(meilleure) if meilleure else "aucune (demande non fournie)",
            COLONNES[met_c][0]: fmt(met_c, finis[meilleure]) if meilleure else "—",
            "Rang": _rang_txt(rangs[meilleure]) if meilleure else "hors classement",
        }
    )
st.dataframe(pd.DataFrame(lignes_f).set_index("Famille"), width="stretch")


# 3 — Classement sur ce critère

st.subheader("Classement sur ce critère")
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

st.subheader("Toutes les stratégies, critère par critère")
st.caption(
    "★ = meilleure valeur parmi les stratégies classées (plusieurs ★ si ex æquo). "
    "La flèche indique le sens favorable."
)

meilleurs = {nom_c: _premiers(nom_c) for nom_c in CRIT}

lignes = []
for n in ordre + [x for x in noms if x not in ordre]:
    ligne = {
        "Stratégie": nom_affichage(n) + ("" if n in classees else " (hors classement)"),
        "Famille": famille(n),
    }
    for nom_c, (met, sens_c) in CRIT.items():
        entete = f"{nom_c} {'↑' if sens_c == 'max' else '↓'}"
        texte = fmt(met, metriques[n].get(met, float("nan")))
        ligne[entete] = texte + (" ★" if n in meilleurs[nom_c] else "")
    ligne["Demande non fournie (Wh)"] = f"{non_servie[n]:.0f}"
    lignes.append(ligne)

st.dataframe(pd.DataFrame(lignes).set_index("Stratégie"), width="stretch")


# Apport du symbolique : même réseau, sans puis avec composante symbolique

st.subheader("Apport du symbolique")
st.caption(
    "Même réseau, sans puis avec composante symbolique : c'est la comparaison qui isole "
    "ce qu'apporte le symbolique. Les deux variantes ne l'intègrent pas de la même façon : "
    "le MLP neurosymbolique part de la base floue et n'apprend qu'une correction bornée ; "
    "le LSTM neurosymbolique reçoit quatre états symboliques en entrée supplémentaire."
)


def _ecart(met, a, b):
    """Écart de b par rapport à a : en points pour un SOC, en % sinon."""
    if met.startswith("soc_"):
        return f"{(b - a) * 100:+.2f} pts"
    return f"{(b - a) / abs(a) * 100:+.0f} %" if a else "—"


CRIT_ABLATION = list(CRIT.items()) + [("Demande non fournie", ("energie_non_servie_wh", "min"))]
for seul, ns in PAIRES_SYMBOLIQUE:
    if seul not in metriques or ns not in metriques:
        continue
    lignes_a = []
    for nom_c, (met, sens_c) in CRIT_ABLATION:
        a, b = metriques[seul].get(met, float("nan")), metriques[ns].get(met, float("nan"))
        if a != a or b != b:
            continue
        if fmt(met, a) == fmt(met, b):
            verdict = "égal"
        else:
            verdict = "mieux" if ((b < a) if sens_c == "min" else (b > a)) else "moins bien"
        lignes_a.append(
            {
                "Critère": nom_c,
                nom_affichage(seul): fmt(met, a),
                nom_affichage(ns): fmt(met, b),
                "Écart": _ecart(met, a, b),
                "Avec le symbolique": verdict,
            }
        )
    st.markdown(f"**{nom_affichage(seul)} → {nom_affichage(ns)}**")
    st.dataframe(pd.DataFrame(lignes_a).set_index("Critère"), width="stretch")
    if seul in hors_classement or ns in hors_classement:
        st.caption(
            "Au moins une des deux stratégies ne fournit pas toute la demande : à "
            "interpréter avec prudence (voir l'encadré « Hors classement »)."
        )


# Bilan des pertes (objectif du projet : rendement global et pertes du convertisseur)

st.subheader("Bilan des pertes")
st.caption(
    "Pertes par effet Joule dans chaque batterie (R·I²) et pertes du convertisseur, qui "
    "ne traite que la différence de tension entre les deux batteries (architecture en "
    "cascade). Estimation faite après coup : les trajectoires ont été simulées sans pertes."
)

bilans = {n: bilan_pertes(resultats[n]) for n in noms}
ordre_p = sorted(noms, key=lambda n: bilans[n]["total_wh"])
etiquettes_p = [nom_affichage(n) + ("" if n in classees else " (hors cl.)") for n in ordre_p][::-1]
COMPOSANTES = (
    ("eb_wh", "Batterie Énergie (R·I²)", "#5B8DEF"),
    ("pb_wh", "Batterie Puissance (R·I²)", "#30A46C"),
    ("convertisseur_wh", "Convertisseur", "#E0A030"),
)
fig_p = go.Figure()
for cle, lib, coul in COMPOSANTES:
    fig_p.add_trace(
        go.Bar(
            y=etiquettes_p,
            x=[bilans[n][cle] for n in ordre_p][::-1],
            name=lib,
            orientation="h",
            marker_color=coul,
            hovertemplate=f"{lib} : %{{x:.0f}} Wh<extra></extra>",
        )
    )
fig_p.add_trace(
    go.Scatter(
        y=etiquettes_p,
        x=[bilans[n]["total_wh"] for n in ordre_p][::-1],
        mode="text",
        text=[f"  {bilans[n]['total_wh']:.0f} Wh ({bilans[n]['part_traction'] * 100:.1f} %)" for n in ordre_p][::-1],
        textposition="middle right",
        showlegend=False,
        hoverinfo="skip",
    )
)
fig_p.update_layout(
    barmode="stack",
    height=40 * len(noms) + 110,
    margin=dict(t=10, b=40, l=10, r=130),
    xaxis_title="Pertes sur le cycle (Wh) — entre parenthèses : part de l'énergie de traction",
    legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0),
)
st.plotly_chart(fig_p, width="stretch")

_ref_p = "EMS_power_limitation"
_meilleure_p = min(classees, key=lambda n: bilans[n]["total_wh"])
if _ref_p in bilans and _meilleure_p != _ref_p:
    _gain = (bilans[_meilleure_p]["total_wh"] / bilans[_ref_p]["total_wh"] - 1) * 100
    st.markdown(
        f"Parmi les stratégies qui fournissent toute la demande, **{nom_affichage(_meilleure_p)}** "
        f"a le moins de pertes ({bilans[_meilleure_p]['total_wh']:.0f} Wh), soit {_gain:+.0f} % "
        f"par rapport au {nom_affichage(_ref_p).lower()} ({bilans[_ref_p]['total_wh']:.0f} Wh)."
    )

r_eb, r_pb = resistances_packs()
with st.expander("Hypothèses et équations du bilan"):
    st.markdown(
        f"- Batterie Énergie : {ec.CELL_EB_RINT_OHM * 1000:.0f} mΩ par cellule, "
        f"{ec.CELL_EB_N_SERIE} en série × {ec.CELL_EB_N_PARALLELE} en parallèle, soit "
        f"**{r_eb * 1000:.0f} mΩ** pour le pack.\n"
        f"- Batterie Puissance : {ec.CELL_PB_RINT_OHM * 1000:.1f} mΩ par cellule, "
        f"{ec.CELL_PB_N_SERIE} en série × {ec.CELL_PB_N_PARALLELE} en parallèle, soit "
        f"**{r_pb * 1000:.0f} mΩ** pour le pack.\n"
        f"- Convertisseur : rendement de {RENDEMENT_CONVERTISSEUR * 100:.1f} %, mesuré à puissance "
        "nominale ([1], fig. 33) ; il est plus faible à charge partielle.\n"
        "- Tensions constantes (valeurs nominales), sans variation avec le SOC.\n"
        "- Trajectoires simulées sans pertes : ce bilan compare les stratégies, mais ne "
        "dit pas si le pack terminerait le cycle une fois les pertes prises en compte."
    )
    st.latex(
        r"P_{conv} = (V_{EB} - V_{PB})\,I_{EB} \qquad "
        r"P_{pertes} = R_{EB}\,I_{EB}^2 + R_{PB}\,I_{PB}^2 + (1 - \eta)\,\lvert P_{conv} \rvert"
    )
    st.caption(
        "[1] C. A. Fonseca de Freitas et al., « Partial Power Converter for Electric "
        "Vehicle Hybrid Energy Storage System Using a Controlled Current Source Cascade "
        "Architecture », IEEE Access, vol. 12, 2024 (fig. 8, éq. (9)–(16))."
    )


st.caption(
    "Pour suivre l'évolution des états de charge d'une stratégie face à une référence, "
    "voir « Explorer les résultats »."
)


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