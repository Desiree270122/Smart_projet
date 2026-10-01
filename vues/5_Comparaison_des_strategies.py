"""
Comparaison des stratégies EMS : les sept stratégies, sur un même cycle et
selon un protocole commun (M1 à M6), l'explicabilité (E1 à E3), puis une
conclusion calculée sur tous les critères : quelle stratégie retenir ?
"""

import sys
from pathlib import Path

DOSSIER_PROJET = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DOSSIER_PROJET))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from core import verdict, xai
from core.format import nombre, separateurs_plotly
from core.i18n import langue, lib, tr
from core.navigation import pied_navigation
from core.resultats import (
    CYCLES_REFERENCE,
    PAIRES_SYMBOLIQUE,
    assurer_donnees_session,
    calculer_metriques,
    charger_reference,
    choisir_cycle,
    famille,
    libelle_cycle,
    nom_affichage,
)
from core.style import couleur


# Les six métriques du protocole : (clé, libellé, unité, échelle, décimales)
METRIQUES = [
    ("energie_km_wh", ("M1 · Énergie consommée", "M1 · Energy consumed"), "Wh/km", 1.0, 1),
    ("rendement_hess", ("M2 · Rendement du HESS", "M2 · HESS efficiency"), "%", 100.0, 2),
    ("rmse_delta_soc", ("M3 · Écart entre les SOC", "M3 · Gap between SOCs"), "pts", 100.0, 1),
    ("pertes_convertisseur_wh", ("M4 · Pertes du convertisseur", "M4 · Converter losses"), "Wh", 1.0, 0),
    ("violations_totales", ("M5 · Dépassements de limite", "M5 · Limit violations"), "", 1.0, 0),
    ("rmse_puissance_kw", ("M6 · Écart de puissance", "M6 · Power shortfall"), "kW", 1.0, 2),
]


def _titre(libelle, unite):
    return f"{lib(libelle)} ({unite})" if unite else lib(libelle)


st.title(tr("📊 Comparaison des stratégies EMS", "📊 EMS strategy comparison"))
st.caption(tr(
    "Les sept stratégies dans les mêmes conditions : même profil de conduite, mêmes batteries, "
    "même convertisseur, même filtre de sécurité. En bas de page : quelle stratégie retenir.",
    "The seven strategies under the same conditions: same driving profile, same batteries, same "
    "converter, same safety filter. At the bottom of the page: which strategy to choose.",
))

try:
    source = assurer_donnees_session(st)
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

resultats = st.session_state.get("resultats_simulation")
df = st.session_state.get("cycle_pret")
if not resultats or df is None:
    st.warning(tr("Aucune donnée disponible.", "No data available."))
    st.stop()

noms = list(resultats.keys())
signature = tuple(sorted((n, float(np.nansum(t["alpha_final"]))) for n, t in resultats.items()))

choisir_cycle(st, st.columns([2, 3])[0])


@st.cache_data(show_spinner=False)
def _metriques(signature):
    m = calculer_metriques({"resultats": resultats, "cycle_df": df})
    for v in m.values():
        v["violations_totales"] = v["nb_violations"] + v["nb_violations_courant"]
    return m


metriques = _metriques(signature)
st.caption(tr("Données : {s} · {n} stratégies", "Data: {s} · {n} strategies", s=source, n=len(noms)))

with st.expander(tr("Définition des métriques", "Definition of the metrics")):
    st.markdown(tr(
        "- **M1 · Énergie consommée** : énergie tirée des batteries sur le cycle, pertes comprises, "
        "rapportée à la distance parcourue.\n"
        "- **M2 · Rendement du HESS** : énergie de traction fournie, divisée par cette énergie "
        "augmentée des pertes (batteries et convertisseur).\n"
        "- **M3 · Écart entre les SOC** : écart quadratique moyen entre les états de charge des deux batteries.\n"
        "- **M4 · Pertes du convertisseur** : il ne traite que la puissance (V_EB − V_PB)·I_EB.\n"
        "- **M5 · Dépassements de limite** : nombre d'instants où une limite de SOC ou de courant est dépassée.\n"
        "- **M6 · Écart de puissance** : écart quadratique moyen entre la puissance fournie et la "
        "puissance demandée en traction ; il est nul quand toute la demande est fournie.",
        "- **M1 · Energy consumed**: energy drawn from the batteries over the cycle, losses included, "
        "per distance travelled.\n"
        "- **M2 · HESS efficiency**: traction energy delivered, divided by that energy plus the "
        "losses (batteries and converter).\n"
        "- **M3 · Gap between SOCs**: root-mean-square gap between the two batteries' states of charge.\n"
        "- **M4 · Converter losses**: it only processes the power (V_EB − V_PB)·I_EB.\n"
        "- **M5 · Limit violations**: number of time steps where an SOC or current limit is exceeded.\n"
        "- **M6 · Power shortfall**: root-mean-square gap between delivered and demanded traction "
        "power; it is zero when the whole demand is met.",
    ))
    st.latex(
        r"M_1 = \frac{1}{D}\Big(\int_0^T (P_{EB} + P_{PB})\,dt + E_{pertes}\Big)"
        r"\qquad M_2 = \frac{E_{traction}}{E_{traction} + E_{pertes}}"
        r"\qquad M_3 = \sqrt{\tfrac{1}{N}\textstyle\sum_k (SOC_{EB} - SOC_{PB})^2}"
    )
    st.latex(
        r"M_4 = \int_0^T (1-\eta)\,\lvert (V_{EB}-V_{PB})\,I_{EB}\rvert\,dt"
        r"\qquad M_6 = \sqrt{\tfrac{1}{N}\textstyle\sum_k (P_{HESS} - P_{dem})^2}"
    )
    st.caption(tr(
        "Sauf si les pertes ont été incluses à la simulation (page « Lancer une simulation »), M1, M2 "
        "et M4 utilisent des pertes estimées après coup : résistances internes des cellules et "
        "rendement mesuré du convertisseur. Ils servent à comparer les stratégies entre elles.",
        "Unless losses were included in the simulation (“Run a simulation” page), M1, M2 and M4 use "
        "losses estimated afterwards: internal resistances of the cells and measured converter "
        "efficiency. They are meant for comparing the strategies with one another.",
    ))


# 1 — Les six métriques

st.subheader(tr("Les six métriques du protocole", "The six metrics of the protocol"))
col_strategie, col_famille = tr("Stratégie", "Strategy"), tr("Famille", "Family")
st.dataframe(
    pd.DataFrame([
        {
            col_strategie: nom_affichage(n),
            col_famille: famille(n),
            **{_titre(libelle, u): nombre(metriques[n][cle] * ech, dec) for cle, libelle, u, ech, dec in METRIQUES},
        }
        for n in noms
    ]).set_index(col_strategie),
    width="stretch",
)
st.download_button(
    tr("Télécharger le tableau (CSV)", "Download the table (CSV)"),
    pd.DataFrame([
        {col_strategie: nom_affichage(n), col_famille: famille(n),
         **{_titre(libelle, u): round(metriques[n][cle] * ech, 4) for cle, libelle, u, ech, _ in METRIQUES}}
        for n in noms
    ]).to_csv(index=False, sep=";", decimal="." if langue() == "en" else ",").encode("utf-8-sig"),
    file_name="metriques_M1_M6.csv",
    mime="text/csv",
)

fig_m = make_subplots(
    rows=2, cols=3, shared_yaxes=True, horizontal_spacing=0.04, vertical_spacing=0.16,
    subplot_titles=[_titre(libelle, u) for _, libelle, u, _, _ in METRIQUES],
)
for k, (cle, libelle, u, ech, dec) in enumerate(METRIQUES):
    valeurs = [metriques[n][cle] * ech for n in noms]
    fig_m.add_trace(
        go.Bar(
            y=[nom_affichage(n) for n in noms], x=valeurs, orientation="h",
            marker_color=[couleur(n) for n in noms],
            text=[nombre(v, dec) for v in valeurs], textposition="outside", cliponaxis=False,
            hovertemplate="%{y} : %{text}<extra></extra>", showlegend=False,
        ),
        row=k // 3 + 1, col=k % 3 + 1,
    )
fig_m.update_yaxes(autorange="reversed")
fig_m.update_layout(separators=separateurs_plotly(), height=540, margin=dict(t=40, b=20, l=10, r=40), bargap=0.25)
fig_m.update_annotations(font_size=12)
st.plotly_chart(fig_m, width="stretch")


# 2 — Demande fournie et limites respectées

st.subheader(tr("La demande est-elle fournie, les limites respectées ?", "Is the demand met, are the limits respected?"))
st.caption(tr(
    "Détail de M5 et M6. Une stratégie qui ne fournit pas toute la puissance demandée consomme "
    "mécaniquement moins d'énergie : ses valeurs de M1, M2 et M4 sont alors flattées.",
    "Detail of M5 and M6. A strategy that does not deliver all the demanded power mechanically "
    "consumes less energy: its M1, M2 and M4 values are then flattered.",
))
st.dataframe(
    pd.DataFrame([
        {
            col_strategie: nom_affichage(n),
            tr("Dépassements de SOC", "SOC violations"): nombre(metriques[n]["nb_violations"], 0),
            tr("Dépassements de courant", "Current violations"): nombre(metriques[n]["nb_violations_courant"], 0),
            tr("Écart de puissance moyen (kW)", "Mean power shortfall (kW)"): nombre(metriques[n]["rmse_puissance_kw"], 2),
            tr("Écart de puissance maximal (kW)", "Maximum power shortfall (kW)"): nombre(metriques[n]["ecart_puissance_max_kw"], 1),
            tr("Énergie non fournie (Wh)", "Energy not delivered (Wh)"): nombre(metriques[n]["energie_non_servie_wh"], 0),
        }
        for n in noms
    ]).set_index(col_strategie),
    width="stretch",
)


# 3 — Apport de l'approche neuro-symbolique

st.subheader(tr("Apport de l'approche neuro-symbolique", "Contribution of the neuro-symbolic approach"))
st.caption(tr(
    "Chaque stratégie neuro-symbolique comparée au même réseau de neurones sans connaissances "
    "expertes. NS-MLP part des règles expertes, n'apprend qu'une correction limitée et reste sous "
    "le contrôle d'un garde-fou de l'ontologie ; NS-LSTM reçoit en plus des états déduits par l'ontologie.",
    "Each neuro-symbolic strategy compared with the same neural network without expert knowledge. "
    "NS-MLP starts from the expert rules, only learns a limited correction and stays under the "
    "control of an ontology safeguard; NS-LSTM additionally receives states inferred by the ontology.",
))


def _variation(cle, a, b):
    """Écart de b par rapport à a : en points pour un pourcentage, en % sinon."""
    if cle in ("rendement_hess", "rmse_delta_soc"):
        return f"{nombre((b - a) * 100, 2, signe=True)} pts"
    if abs(a) < 1e-12:
        return "—" if abs(b) < 1e-12 else nombre(b, 2, signe=True)
    return f"{nombre((b - a) / abs(a) * 100, 1, signe=True)} %"


col_metrique = tr("Métrique", "Metric")
colonnes_ns = st.columns(len(PAIRES_SYMBOLIQUE))
for col, (seul, ns) in zip(colonnes_ns, PAIRES_SYMBOLIQUE):
    if seul not in metriques or ns not in metriques:
        continue
    with col:
        st.markdown(tr("**{a} et {b}**", "**{a} and {b}**", a=nom_affichage(ns), b=nom_affichage(seul)))
        st.dataframe(
            pd.DataFrame([
                {
                    col_metrique: _titre(libelle, u),
                    nom_affichage(seul): nombre(metriques[seul][cle] * ech, dec),
                    nom_affichage(ns): nombre(metriques[ns][cle] * ech, dec),
                    tr("Variation", "Change"): _variation(cle, metriques[seul][cle], metriques[ns][cle]),
                }
                for cle, libelle, u, ech, dec in METRIQUES
            ]).set_index(col_metrique),
            width="stretch",
        )


# 4 — Explicabilité

st.subheader(tr("Explicabilité", "Explainability"))
st.caption(tr(
    "Peut-on comprendre une décision ? E1 et E2 décrivent ce que permet la construction de la "
    "stratégie ; E3 est mesurée : quand on augmente un SOC ou la demande, la décision évolue-t-elle "
    "dans le sens attendu par la physique ?",
    "Can a decision be understood? E1 and E2 describe what the design of the strategy allows; E3 is "
    "measured: when an SOC or the demand is increased, does the decision move in the direction "
    "expected from physics?",
))

# E3 est calculée à l'avance avec les résultats de référence ; sinon, à la demande.
coherences = dict(st.session_state.get("coherence_simulation") or {})
if not all(n in coherences for n in noms):
    if st.toggle(tr("Mesurer E3 · cohérence physique (environ 30 s)", "Measure E3 · physical consistency (about 30 s)"), value=False):

        @st.cache_data(show_spinner=False)
        def _coherences(signature):
            return {n: xai.coherence_physique(n, df, resultats[n]) for n in noms}

        with st.spinner(tr("Mesure de la cohérence physique…", "Measuring physical consistency…")):
            coherences = _coherences(signature)

st.dataframe(
    pd.DataFrame([
        {
            col_strategie: nom_affichage(n),
            tr("E1 · Comment la décision se lit", "E1 · How the decision can be read"): lib(xai.TRANSPARENCE[n][1]) if n in xai.TRANSPARENCE else "—",
            tr("E2 · Comment elle s'explique", "E2 · How it is explained"): lib(xai.TRANSPARENCE[n][2]) if n in xai.TRANSPARENCE else "—",
            tr("E3 · Cohérence physique", "E3 · Physical consistency"): f"{nombre(coherences[n][0] * 100, 0)} %" if n in coherences else "—",
        }
        for n in noms
    ]).set_index(col_strategie),
    width="stretch",
)
if coherences:
    with st.expander(tr("Détail de E3, test par test", "Detail of E3, test by test")):
        st.dataframe(
            pd.DataFrame({
                nom_affichage(n): {xai.libelle_contrainte(k): f"{nombre(v * 100, 0)} %" for k, v in coherences[n][1].items()}
                for n in noms if n in coherences
            }).T,
            width="stretch",
        )
        st.caption(tr(
            "Sur 120 instants de traction, chaque test augmente une grandeur (SOC de +5 points, "
            "demande de +1 kW) et vérifie que la puissance confiée à la batterie Puissance évolue "
            "dans le sens indiqué, à 1 % de la demande près.",
            "Over 120 traction time steps, each test increases one quantity (SOC by +5 points, demand "
            "by +1 kW) and checks that the power assigned to the Power battery moves in the stated "
            "direction, to within 1 % of the demand.",
        ))


# 5 — Conclusion : quelle stratégie retenir, tous critères confondus ?

st.subheader(tr("Quelle stratégie retenir ?", "Which strategy should be chosen?"))
st.caption(tr(
    "Conclusion calculée sur tous les critères, en trois étapes : (1) une stratégie doit fournir "
    "toute la demande et respecter les limites, sinon elle est écartée ; (2) les autres sont "
    "comparées deux à deux sur chaque critère ; (3) on vérifie que le résultat ne dépend pas de "
    "l'importance donnée à chaque critère.",
    "Conclusion computed over all criteria, in three steps: (1) a strategy must deliver the whole "
    "demand and respect the limits, otherwise it is set aside; (2) the others are compared pairwise "
    "on each criterion; (3) we check that the result does not depend on the importance given to "
    "each criterion.",
))
avec_explicabilite = st.checkbox(
    tr("Tenir compte de l'explicabilité (E1 à E3) en plus des performances (M1 à M4)",
       "Take explainability (E1 to E3) into account in addition to performance (M1 to M4)"),
    value=True,
)
liste_criteres = verdict.criteres(avec_explicabilite)


def _raison(non_fourni, depassements):
    morceaux = []
    if non_fourni >= 1.0:
        morceaux.append(tr("{e} Wh non fournis", "{e} Wh not delivered", e=nombre(non_fourni, 0)))
    if depassements:
        morceaux.append(tr("{d} dépassement(s) de limite", "{d} limit violation(s)", d=depassements))
    return ", ".join(morceaux) or tr("demande non suivie", "demand not followed")


def _liste(noms_strategies):
    libelles = [nom_affichage(n) for n in noms_strategies]
    if len(libelles) <= 1:
        return "".join(libelles)
    return ", ".join(libelles[:-1]) + tr(" et ", " and ") + libelles[-1]


def _score(valeur):
    return nombre(valeur, 2, signe=True)


def _forts(metriques_eval, strategie, strategies):
    forts = verdict.points_forts(metriques_eval, strategie, strategies, liste_criteres)
    return ", ".join(verdict.libelle(c).split(" · ")[0] for c in forts) or "—"


# Sur le cycle affiché

sur_ce_cycle = verdict.evaluer(verdict.completer(metriques, coherences), liste_criteres)
st.markdown(tr("**Sur ce cycle**", "**On this cycle**"))
if avec_explicabilite and not all(n in coherences for n in noms):
    st.caption(tr(
        "E3 n'est pas mesurée pour cette simulation (bouton ci-dessus) : ce critère ne départage pas les stratégies ici.",
        "E3 has not been measured for this simulation (toggle above): this criterion does not separate the strategies here.",
    ))
if sur_ce_cycle["ecartees"]:
    st.markdown(
        tr("Étape 1 — écartées : ", "Step 1 — set aside: ")
        + tr(" ; ", "; ").join(f"**{nom_affichage(n)}** ({_raison(*r)})" for n, r in sur_ce_cycle["ecartees"].items())
        + "."
    )
else:
    st.markdown(tr(
        "Étape 1 — toutes les stratégies fournissent la demande et respectent les limites.",
        "Step 1 — all strategies deliver the demand and respect the limits.",
    ))

classement = sorted(sur_ce_cycle["retenues"], key=lambda n: -sur_ce_cycle["score"][n])
if classement:
    g_col, t_col = st.columns([2, 3])
    with g_col:
        fig_s = go.Figure(
            go.Bar(
                y=[nom_affichage(n) for n in classement], x=[sur_ce_cycle["score"][n] for n in classement],
                orientation="h", marker_color=[couleur(n) for n in classement],
                text=[_score(sur_ce_cycle["score"][n]) for n in classement], textposition="outside", cliponaxis=False,
                hovertemplate="%{y} : %{text}<extra></extra>",
            )
        )
        fig_s.add_vline(x=0, line=dict(color="#8B93A7", width=1))
        fig_s.update_layout(
            separators=separateurs_plotly(), height=60 + 42 * len(classement), margin=dict(t=10, b=40, l=10, r=40),
            xaxis=dict(title=tr("Score (comparaisons gagnées − perdues)", "Score (comparisons won − lost)"), range=[-1.15, 1.15]),
            yaxis=dict(autorange="reversed"), showlegend=False,
        )
        st.plotly_chart(fig_s, width="stretch")
    with t_col:
        st.dataframe(
            pd.DataFrame([
                {
                    col_strategie: nom_affichage(n),
                    tr("Score", "Score"): _score(sur_ce_cycle["score"][n]),
                    tr("En tête dans … des pondérations", "First in … of the weightings"): f"{nombre(sur_ce_cycle['en_tete'][n] * 100, 0)} %",
                    tr("Critères où elle n'est battue par aucune autre", "Criteria where no other beats it"): _forts(metriques, n, classement),
                }
                for n in classement
            ]).set_index(col_strategie),
            width="stretch",
        )
    st.caption(tr(
        "Étapes 2 et 3 — le score va de −1 (battue partout) à +1 (meilleure partout). La colonne "
        "suivante chiffre la solidité du résultat : part des 5 000 jeux de poids tirés au hasard où "
        "la stratégie arrive première.",
        "Steps 2 and 3 — the score ranges from −1 (beaten everywhere) to +1 (best everywhere). The "
        "next column measures how solid the result is: share of the 5,000 randomly drawn sets of "
        "weights in which the strategy comes first.",
    ))


# Sur les deux cycles de référence : la conclusion générale

@st.cache_data(show_spinner=False)
def _metriques_reference(cle, modification):
    donnees = charger_reference(CYCLES_REFERENCE[cle][1])
    return verdict.completer(
        calculer_metriques({"resultats": donnees["resultats"], "cycle_df": donnees["cycle_df"]}),
        donnees.get("coherence"),
    )


cycles_presents = [c for c, (_, chemin) in CYCLES_REFERENCE.items() if Path(chemin).exists()]
if len(cycles_presents) >= 2:
    evaluations = {c: _metriques_reference(c, Path(CYCLES_REFERENCE[c][1]).stat().st_mtime) for c in cycles_presents}
    bilan = verdict.conclure(evaluations, liste_criteres)
    communes = sorted(bilan["communes"], key=lambda n: -bilan["score"][n])
    toutes = list(next(iter(evaluations.values())))

    st.markdown(tr("**Sur les deux cycles de référence**", "**On the two reference cycles**"))
    lignes = []
    for n in communes + [x for x in toutes if x not in communes]:
        ligne = {col_strategie: nom_affichage(n)}
        for c in cycles_presents:
            ecart = verdict.ecartees(evaluations[c]).get(n)
            if ecart:
                ligne[libelle_cycle(c)] = tr("écartée ({r})", "set aside ({r})", r=_raison(*ecart))
            elif n in communes:
                ligne[libelle_cycle(c)] = _score(bilan["par_evaluation"][c]["score"][n])
            else:
                ligne[libelle_cycle(c)] = tr("demande fournie", "demand met")
        ligne[tr("Score moyen", "Mean score")] = _score(bilan["score"][n]) if n in communes else "—"
        ligne[tr("En tête dans … des pondérations", "First in … of the weightings")] = (
            f"{nombre(bilan['en_tete'][n] * 100, 0)} %" if n in communes else "—"
        )
        lignes.append(ligne)
    st.dataframe(pd.DataFrame(lignes).set_index(col_strategie), width="stretch")
    st.caption(tr(
        "Seules les stratégies qui fournissent toute la demande sur les deux cycles sont comparées "
        "entre elles ; leurs scores sont recalculés entre elles seules. Le cycle WLTC n'a jamais "
        "servi à régler les stratégies à apprentissage.",
        "Only the strategies that deliver the whole demand on both cycles are compared with one "
        "another; their scores are recomputed among themselves only. The WLTC cycle was never used "
        "to tune the learning-based strategies.",
    ))

    if communes:
        en_tete = verdict.meilleures(bilan["score"])
        premiere = en_tete[0]
        part = bilan["en_tete"][premiere]
        phrases = []
        if len(en_tete) == 1:
            phrases.append(tr(
                "**{s} est la stratégie à retenir** sur l'ensemble des critères : parmi les {k} "
                "stratégies qui fournissent toute la demande sur les deux cycles ({l}), elle obtient "
                "le meilleur score moyen ({sc}) et arrive en tête dans {p} % des pondérations.",
                "**{s} is the strategy to choose** over all criteria: among the {k} strategies that "
                "deliver the whole demand on both cycles ({l}), it has the best mean score ({sc}) and "
                "comes first in {p} % of the weightings.",
                s=nom_affichage(premiere), k=len(communes), l=_liste(communes),
                sc=_score(bilan["score"][premiere]), p=nombre(part * 100, 0),
            ))
        else:
            phrases.append(tr(
                "**{l} arrivent à égalité** sur l'ensemble des critères (scores moyens {sc}), parmi "
                "les {k} stratégies qui fournissent toute la demande sur les deux cycles.",
                "**{l} are tied** over all criteria (mean scores {sc}), among the {k} strategies that "
                "deliver the whole demand on both cycles.",
                l=_liste(en_tete), sc=" / ".join(_score(bilan["score"][n]) for n in en_tete), k=len(communes),
            ))
        if len(communes) > 1 and len(en_tete) == 1:
            seconde = communes[1]
            phrases.append(tr(
                "{s} suit (score {sc}, en tête dans {p} % des pondérations).",
                "{s} comes next (score {sc}, first in {p} % of the weightings).",
                s=nom_affichage(seconde), sc=_score(bilan["score"][seconde]), p=nombre(bilan["en_tete"][seconde] * 100, 0),
            ))
            phrases.append(
                tr("Ce résultat est solide : il tient quelle que soit l'importance donnée à chaque critère, ou presque.",
                   "This result is solid: it holds whatever importance is given to each criterion, or nearly so.")
                if part >= 0.70 else
                tr("Ce résultat dépend de l'importance donnée à chaque critère : l'écart avec la suivante est faible.",
                   "This result depends on the importance given to each criterion: the gap with the next one is small.")
            )
        if avec_explicabilite:
            perf = verdict.conclure(evaluations, verdict.criteres(False))
            tete_perf = verdict.meilleures(perf["score"])
            phrases.append(tr(
                "Sur les seules performances (M1 à M4), {l} est en tête dans {p} % des pondérations.",
                "On performance alone (M1 to M4), {l} comes first in {p} % of the weightings.",
                l=_liste(tete_perf), p=nombre(sum(perf["en_tete"][n] for n in tete_perf) * 100, 0),
            ))
        hors = [n for n in toutes if n not in communes]
        if len(hors) > 1:
            phrases.append(tr(
                "{l} ne sont pas retenues : elles ne fournissent pas toute la demande sur au moins un des deux cycles.",
                "{l} are not retained: they do not deliver the whole demand on at least one of the two cycles.",
                l=_liste(hors),
            ))
        elif hors:
            phrases.append(tr(
                "{l} n'est pas retenue : elle ne fournit pas toute la demande sur au moins un des deux cycles.",
                "{l} is not retained: it does not deliver the whole demand on at least one of the two cycles.",
                l=_liste(hors),
            ))
        st.success(" ".join(phrases))
    else:
        st.warning(tr(
            "Aucune stratégie ne fournit toute la demande sur les deux cycles : pas de conclusion générale.",
            "No strategy delivers the whole demand on both cycles: no overall conclusion.",
        ))

with st.expander(tr("Méthode et seuils", "Method and thresholds")):
    st.markdown(tr(
        "- **Étape 1.** Sont écartées les stratégies qui laissent de la puissance non fournie (M6) ou "
        "dépassent une limite de SOC ou de courant (M5).\n"
        "- **Étape 2.** Sur chaque critère, une stratégie gagne une comparaison quand elle fait mieux "
        "qu'une autre d'un écart supérieur au seuil ci-dessous ; en dessous, les deux sont jugées "
        "équivalentes. Score = moyenne, sur les critères, de (comparaisons gagnées − perdues) / "
        "nombre d'adversaires. Tous les critères ont le même poids.\n"
        "- **Étape 3.** 5 000 jeux de poids sont tirés au hasard ; on compte la part où chaque "
        "stratégie est première.\n"
        "- M1, M2 et M4 mesurent tous trois des pertes : les performances énergétiques pèsent donc "
        "trois critères, l'équilibre des batteries un seul.",
        "- **Step 1.** Strategies that leave power undelivered (M6) or exceed an SOC or current "
        "limit (M5) are set aside.\n"
        "- **Step 2.** On each criterion, a strategy wins a comparison when it does better than "
        "another by more than the threshold below; under it, the two are judged equivalent. Score = "
        "mean, over the criteria, of (comparisons won − lost) / number of opponents. All criteria "
        "have the same weight.\n"
        "- **Step 3.** 5,000 sets of weights are drawn at random; we count the share in which each "
        "strategy comes first.\n"
        "- M1, M2 and M4 all measure losses: energy performance therefore counts for three criteria, "
        "battery balance for only one.",
    ))
    col_critere = tr("Critère", "Criterion")
    st.dataframe(
        pd.DataFrame([
            {
                col_critere: verdict.libelle(c),
                tr("Meilleur quand il est", "Better when"): tr("plus bas", "lower") if c["sens"] == "min" else tr("plus haut", "higher"),
                tr("Seuil d'indifférence", "Indifference threshold"): lib(c["seuil_txt"]),
            }
            for c in liste_criteres
        ]).set_index(col_critere),
        width="stretch",
    )
    st.caption(tr(
        "Méthodes de référence : comparaison par paires à seuils (PROMETHEE II) et analyse de "
        "robustesse par tirage des poids (SMAA).",
        "Reference methods: pairwise comparison with thresholds (PROMETHEE II) and robustness "
        "analysis by weight sampling (SMAA).",
    ))


pied_navigation("vues/5_Comparaison_des_strategies.py")
