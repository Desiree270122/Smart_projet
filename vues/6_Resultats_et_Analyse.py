"""
Page « Explorer les résultats » — orientée explication, pas visualisation.

La page ne cherche pas à montrer tout ce qui est calculable, mais à répondre à
une question : pourquoi une stratégie offre-t-elle un meilleur compromis que
les autres ? Le classement repose sur une somme de rangs (aucune note
arbitraire, aucune normalisation inventée), ce qui le rend explicable.
"""

import sys
from pathlib import Path

DOSSIER_PROJET = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DOSSIER_PROJET))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.resultats import (
    assurer_donnees_session,
    calculer_metriques,
    statistiques_detaillees,
    nom_affichage,
)
from core.navigation import pied_navigation


# Configuration de page gérée par le routeur Accueil.py.

C_EB = "#3B82F6"
C_PB = "#22C55E"
VERT_CLAIR = "background-color: rgba(34,197,94,.25)"


st.title("📈 Explorer les résultats")
st.caption(
    "Pourquoi une stratégie offre-t-elle un meilleur compromis que les autres ? "
    "Cette page explique les arbitrages réalisés par chacune sur l'ensemble du cycle."
)


try:
    _source = assurer_donnees_session(st)
except FileNotFoundError as exc:
    st.error(str(exc))
    st.info("Lance une fois le précalcul :  `python scripts/run_simulations.py`")
    st.stop()

if "resultats_simulation" not in st.session_state or "cycle_pret" not in st.session_state:
    st.warning("Aucune donnée disponible.")
    st.stop()

resultats = st.session_state["resultats_simulation"]
df = st.session_state["cycle_pret"]
donnees = {"resultats": resultats, "cycle_df": df}

if not resultats:
    st.warning("Aucune stratégie n'a produit de résultat exploitable.")
    st.stop()

st.caption(f"Source des données : {_source}")

stats = statistiques_detaillees(donnees)
metriques = calculer_metriques(donnees)
noms = list(resultats.keys())


# Critères de comparaison : (libellé, valeur par stratégie, sens)
CRITERES_CYCLE = [
    ("Préservation batterie Énergie", lambda n: stats[n]["soc_eb_final"], "max"),
    ("Préservation batterie Puissance", lambda n: stats[n]["soc_pb_final"], "max"),
    ("Stabilité du courant PB", lambda n: stats[n]["i_pb_rms"], "min"),
    ("Respect des contraintes SOC", lambda n: metriques[n]["nb_violations"], "min"),
]


def _rangs(valeur, sens):
    """Rang 1..N sur un critère (1 = meilleur). Aucune note inventée."""
    ordonnes = sorted(noms, key=valeur, reverse=(sens == "max"))
    return {n: i + 1 for i, n in enumerate(ordonnes)}


rangs_par_critere = {lib: _rangs(val, sens) for lib, val, sens in CRITERES_CYCLE}
somme_rangs = {n: sum(rangs_par_critere[lib][n] for lib, _, _ in CRITERES_CYCLE) for n in noms}
classement = sorted(noms, key=lambda n: (somme_rangs[n], n))
rang_final = {n: i + 1 for i, n in enumerate(classement)}


def _etoiles(rang):
    """Étoiles dérivées du rang sur le critère (pas d'un score arbitraire)."""
    return "★" * max(1, 6 - rang) + "☆" * (5 - max(1, 6 - rang))


def _meilleur(libelle):
    return min(noms, key=lambda n: rangs_par_critere[libelle][n])


# 1. Résumé du cycle

st.subheader("🌟 Résumé du cycle")

best_eb = _meilleur("Préservation batterie Énergie")
best_pb = _meilleur("Préservation batterie Puissance")
best_i = _meilleur("Stabilité du courant PB")
best_v = _meilleur("Respect des contraintes SOC")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Préserve le mieux l'EB", nom_affichage(best_eb), f"{stats[best_eb]['soc_eb_final'] * 100:.0f} %")
k2.metric("Préserve le mieux la PB", nom_affichage(best_pb), f"{stats[best_pb]['soc_pb_final'] * 100:.0f} %")
k3.metric("Courant PB le plus stable", nom_affichage(best_i), f"{stats[best_i]['i_pb_rms']:.0f} A RMS")
k4.metric("Le moins de violations", nom_affichage(best_v), f"{metriques[best_v]['nb_violations']:.0f}")

_gagnants = {best_eb, best_pb, best_i, best_v}
if len(_gagnants) > 1:
    st.info(
        f"**Aucune stratégie ne domine tous les critères** : {len(_gagnants)} stratégies "
        "différentes arrivent en tête selon le critère considéré. Cette page explique "
        "les compromis réalisés par chacune."
    )
else:
    st.info(
        f"**{nom_affichage(best_eb)}** arrive en tête sur les quatre critères, ce qui est "
        "inhabituel : la suite détaille ses arbitrages."
    )

with st.expander("Voir le tableau détaillé de tous les indicateurs"):
    tableau = pd.DataFrame(
        {
            "Stratégie": [nom_affichage(n) for n in noms],
            "SOC_EB final (%)": [stats[n]["soc_eb_final"] * 100 for n in noms],
            "SOC_PB final (%)": [stats[n]["soc_pb_final"] * 100 for n in noms],
            "Énergie EB (Wh)": [stats[n]["energie_eb_wh"] for n in noms],
            "Énergie PB (Wh)": [stats[n]["energie_pb_wh"] for n in noms],
            "I_EB RMS (A)": [stats[n]["i_eb_rms"] for n in noms],
            "I_PB RMS (A)": [stats[n]["i_pb_rms"] for n in noms],
            "I_PB max (A)": [stats[n]["i_pb_max"] for n in noms],
            "P_PB max (kW)": [stats[n]["p_pb_max"] / 1000 for n in noms],
            "Violations SOC": [metriques[n]["nb_violations"] for n in noms],
        }
    ).set_index("Stratégie")
    st.dataframe(tableau.style.format("{:.1f}"), use_container_width=True)


# 2. Comment les batteries ont été utilisées ?

st.subheader("📉 Comment les batteries ont été utilisées ?")


def _courbe_soc(cle_soc, titre):
    fig = go.Figure()
    for n in noms:
        y = np.asarray(resultats[n][cle_soc], dtype=float) * 100.0
        x = np.arange(len(y))
        pas = max(1, len(y) // 2000)
        fig.add_trace(go.Scatter(x=x[::pas], y=y[::pas], mode="lines", name=nom_affichage(n)))
    fig.update_layout(
        title=titre, xaxis_title="Temps (s)", yaxis_title="SOC (%)", height=400,
        legend_title="Stratégie", margin=dict(t=50, b=40),
    )
    return fig


col_eb, col_pb = st.columns(2)
with col_eb:
    st.plotly_chart(_courbe_soc("SOC_EB", "SOC batterie Énergie"), use_container_width=True)
with col_pb:
    st.plotly_chart(_courbe_soc("SOC_PB", "SOC batterie Puissance"), use_container_width=True)

# Vitesse de décharge de l'EB : écart entre SOC initial et SOC final, en points.
_chute_eb = {
    n: (float(resultats[n]["SOC_EB"][0]) - float(resultats[n]["SOC_EB"][-1])) * 100.0 for n in noms
}
_lent = min(noms, key=lambda n: _chute_eb[n])
_rapide = max(noms, key=lambda n: _chute_eb[n])

st.markdown("**Analyse automatique**")
st.markdown(
    f"Les courbes montrent que **{nom_affichage(_lent)}** décharge le plus lentement la "
    f"batterie d'énergie ({_chute_eb[_lent]:.1f} points de SOC sur le cycle). À l'inverse, "
    f"**{nom_affichage(_rapide)}** la sollicite le plus fortement "
    f"({_chute_eb[_rapide]:.1f} points). Une pente plus faible traduit une meilleure "
    "préservation de la batterie d'énergie."
)


# 3. Comment la batterie Puissance est sollicitée ?

st.subheader("🔌 Comment la batterie Puissance est sollicitée ?")
st.caption(
    "C'est la batterie que l'on cherche à protéger : le courant efficace est "
    "directement lié au vieillissement électrochimique."
)

fig_ipb = go.Figure()
for n in noms:
    fig_ipb.add_trace(
        go.Box(y=np.asarray(resultats[n]["I_PB"], dtype=float), name=nom_affichage(n), boxpoints=False)
    )
fig_ipb.update_layout(
    title="Courant batterie Puissance", yaxis_title="Courant (A)", height=420,
    showlegend=False, margin=dict(t=50, b=40),
)
st.plotly_chart(fig_ipb, use_container_width=True)

_ns = [n for n in noms if "neurosymbolic" in n]
_non_ns = [n for n in noms if n not in _ns]
st.markdown("**Analyse automatique**")
if _ns and _non_ns:
    _rms_ns = float(np.mean([stats[n]["i_pb_rms"] for n in _ns]))
    _rms_autres = float(np.mean([stats[n]["i_pb_rms"] for n in _non_ns]))
    _pic_ns = float(np.max([stats[n]["i_pb_max"] for n in _ns]))
    _comparatif = "inférieur" if _rms_ns < _rms_autres else "supérieur"
    st.markdown(
        f"Les stratégies neuro-symboliques présentent un courant efficace moyen de "
        f"**{_rms_ns:.0f} A**, {_comparatif} à celui des autres stratégies "
        f"(**{_rms_autres:.0f} A**), avec un pic maximal de {_pic_ns:.0f} A. "
        f"Le courant le plus régulier revient à **{nom_affichage(best_i)}** "
        f"({stats[best_i]['i_pb_rms']:.0f} A RMS), ce qui limite le stress électrochimique."
    )


# 4. Comparaison des stratégies (somme de rangs, sans note arbitraire)

st.subheader("📊 Comparaison des stratégies")
st.caption(
    "Chaque stratégie est classée sur les quatre critères (rang 1 = meilleure), puis les "
    "rangs sont additionnés. Aucun critère n'est privilégié et aucune note n'est inventée : "
    "la somme de rangs la plus faible désigne le meilleur compromis."
)

comparatif = pd.DataFrame(
    {
        "Stratégie": [nom_affichage(n) for n in classement],
        "SOC EB final (%)": [stats[n]["soc_eb_final"] * 100 for n in classement],
        "SOC PB final (%)": [stats[n]["soc_pb_final"] * 100 for n in classement],
        "I_PB RMS (A)": [stats[n]["i_pb_rms"] for n in classement],
        "Violations SOC": [metriques[n]["nb_violations"] for n in classement],
        "Somme des rangs": [somme_rangs[n] for n in classement],
        "Rang": [rang_final[n] for n in classement],
    }
).set_index("Stratégie")


def _surligner(col):
    if col.name in ("SOC EB final (%)", "SOC PB final (%)"):
        cible = col.max()
    elif col.name in ("I_PB RMS (A)", "Violations SOC", "Somme des rangs", "Rang"):
        cible = col.min()
    else:
        return ["" for _ in col]
    return [VERT_CLAIR if v == cible else "" for v in col]


st.dataframe(
    comparatif.style.apply(_surligner, axis=0).format(
        {
            "SOC EB final (%)": "{:.1f}",
            "SOC PB final (%)": "{:.1f}",
            "I_PB RMS (A)": "{:.0f}",
            "Violations SOC": "{:.0f}",
            "Somme des rangs": "{:.0f}",
            "Rang": "{:.0f}",
        }
    ),
    use_container_width=True,
)


# 5. Verdict expliqué (fusionne l'ancien verdict et « ce qu'il faut retenir »)

gagnant = classement[0]

st.subheader(f"🧠 Pourquoi {nom_affichage(gagnant)} offre le meilleur compromis ?")

for libelle, valeur, sens in CRITERES_CYCLE:
    r = rangs_par_critere[libelle][gagnant]
    with st.container(border=True):
        c_gauche, c_droite = st.columns([1, 3])
        c_gauche.markdown(f"**{_etoiles(r)}**")
        v = valeur(gagnant)
        if "Préservation" in libelle:
            texte_val = f"SOC final de {v * 100:.1f} %"
        elif "courant" in libelle:
            texte_val = f"{v:.0f} A RMS"
        else:
            texte_val = f"{v:.0f} violation(s)"
        c_droite.markdown(f"**{libelle}** — rang {r} sur {len(noms)} · {texte_val}")

st.markdown("**Pourquoi les autres ne l'emportent pas**")
for n in classement[1:4]:
    forces = [lib for lib, _, _ in CRITERES_CYCLE if rangs_par_critere[lib][n] == 1]
    faiblesses = sorted(
        [(rangs_par_critere[lib][n], lib) for lib, _, _ in CRITERES_CYCLE], reverse=True
    )[0]
    txt_force = ", ".join(forces) if forces else "aucun critère en tête"
    st.markdown(
        f"- **{nom_affichage(n)}** → en tête sur : {txt_force} ; mais rang "
        f"{faiblesses[0]} sur « {faiblesses[1]} »."
    )

with st.container(border=True):
    st.markdown("**🧠 Explication de la décision**")
    st.markdown(
        f"**{nom_affichage(gagnant)}** n'a pas été retenu parce qu'il possède la meilleure "
        "valeur sur un unique indicateur, mais parce qu'il obtient la somme de rangs la plus "
        f"faible ({somme_rangs[gagnant]}) sur les quatre critères réunis :"
    )
    st.markdown(
        "\n".join(
            f"- {lib} : rang {rangs_par_critere[lib][gagnant]}" for lib, _, _ in CRITERES_CYCLE
        )
    )
    st.markdown(
        "La décision résulte donc d'un **compromis entre plusieurs objectifs**, et non de "
        "l'optimisation d'un seul critère. Une autre stratégie peut rester préférable si "
        "un objectif précis prime — le tableau ci-dessus permet de le vérifier critère par critère."
    )


pied_navigation("vues/6_Resultats_et_Analyse.py")
