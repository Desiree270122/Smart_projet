

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
from core import verdict as vd
import numpy as np


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
    "Quelle stratégie de gestion d'énergie choisir, et pourquoi ? Le verdict tient compte "
    "de tous les critères à la fois ; l'onglet « Critère par critère » détaille chacun."
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


tab_verdict, tab_critere = st.tabs(["🏆 Verdict tous critères", "Critère par critère"])


with tab_verdict:
    st.markdown(
        "Toutes les stratégies sont simulées dans les mêmes conditions : même cycle, mêmes "
        "batteries, même convertisseur, même filtre de sécurité. Pour affirmer qu'un modèle "
        "est meilleur, il doit l'être **sur l'ensemble des métriques**, pas seulement sur "
        "celle qu'on préfère."
    )

    # Matrice des métriques du protocole
    st.markdown("#### Les métriques, stratégie par stratégie")
    _matrice = vd.CRITERES_PRINCIPAUX + vd.ELIMINATOIRES

    def _titre(c):
        return c["libelle"] + (" (" + c["unite"] + ")" if c["unite"] else "")

    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Stratégie": nom_affichage(n),
                    **{
                        _titre(c): c["fmt"].format(vd.valeur(metriques, n, c["cle"]) * c["echelle"])
                        for c in _matrice
                    },
                }
                for n in noms
            ]
        ).set_index("Stratégie"),
        width="stretch",
    )
    with st.expander("Définition des métriques"):
        st.markdown(
            "- **M1 · Énergie consommée** : énergie tirée des batteries sur le cycle, pertes "
            "estimées comprises, rapportée à la distance parcourue.\n"
            "- **M2 · Rendement du HESS** : énergie de traction fournie, rapportée à cette "
            "énergie augmentée des pertes (batteries et convertisseur).\n"
            "- **M3 · Équilibrage des SOC** : écart quadratique moyen entre les SOC des deux batteries.\n"
            "- **M4 · Pertes du convertisseur** : pertes du convertisseur à puissance partielle, "
            "qui ne traite que (V_EB − V_PB)·I_EB.\n"
            "- **M5 · Respect des contraintes** : nombre de pas où une limite de SOC ou de courant est dépassée.\n"
            "- **M6 · Suivi de puissance** : écart quadratique moyen entre puissance fournie et "
            "demandée en traction.\n"
            "- **M7 · Sollicitation des batteries** : courant efficace de chaque batterie, lié à "
            "son vieillissement (objectif « prolonger la durée de vie » du projet)."
        )
        st.latex(
            r"M_1 = \frac{1}{D}\int_0^T \big(P_{EB} + P_{PB}\big)\,dt + \frac{E_{pertes}}{D}"
            r"\qquad M_2 = \frac{E_{traction}}{E_{traction} + E_{pertes}}"
        )
        st.latex(
            r"M_3 = \sqrt{\tfrac{1}{N}\textstyle\sum_k \big(SOC_{EB}(k) - SOC_{PB}(k)\big)^2}"
            r"\qquad M_6 = \sqrt{\tfrac{1}{N}\textstyle\sum_k \big(P_{HESS}(k) - P_{dem}(k)\big)^2}"
            r"\qquad M_7 = \sqrt{\tfrac{1}{N}\textstyle\sum_k I(k)^2}"
        )
        st.caption(
            "Le modèle du HESS étant simulé sans pertes, M1, M2 et M4 utilisent les pertes "
            "estimées après coup (bilan de l'onglet « Critère par critère ») : ils comparent "
            "les stratégies, sans prétendre à la valeur absolue."
        )

    st.markdown("#### Critères retenus pour le verdict")
    t1, t2 = st.columns(2)
    avec_compl = t1.toggle(
        "Ajouter les critères complémentaires (corrections du filtre, coût physique, SOC finaux)", value=False,
    )
    inclure_xai = t2.toggle(
        "Ajouter l'explicabilité (niveau déclaré par modèle, non mesuré)", value=False,
    )
    criteres_v = (
        vd.CRITERES_PRINCIPAUX
        + (vd.CRITERES_COMPLEMENTAIRES if avec_compl else [])
        + ([vd.CRITERE_EXPLICABILITE] if inclure_xai else [])
    )

    # Étape 1 — critères éliminatoires
    retenues, exclues = vd.eliminer(metriques)
    st.markdown("#### 1. Critères éliminatoires (M5, M6)")
    st.caption(
        "Une stratégie doit respecter les contraintes (M5) et fournir la puissance demandée "
        "(M6). Sinon, elle paraît plus sobre (M1) et plus efficace (M2) qu'elle ne l'est : "
        "l'énergie qu'elle n'a pas fournie reste dans les batteries."
    )
    for n, raison in exclues.items():
        st.markdown(f"- **{nom_affichage(n)}** est éliminée : {raison}.")
    st.markdown("Restent en lice : " + ", ".join(f"**{nom_affichage(n)}**" for n in retenues) + ".")

    if len(retenues) < 2:
        st.info("Moins de deux stratégies restent en lice : aucune comparaison n'est possible.")
    else:
        # Étape 2 — dominance
        st.markdown("#### 2. Dominance")
        st.caption(
            "Une stratégie en domine une autre si elle est au moins aussi bonne sur tous les "
            "critères et meilleure sur au moins un : elle l'emporte quelle que soit "
            "l'importance donnée à chaque critère."
        )
        dom = vd.dominances(metriques, retenues, criteres_v)
        for a, b in dom:
            st.markdown(f"- **{nom_affichage(a)}** domine **{nom_affichage(b)}**.")
        non_dominees = [n for n in retenues if n not in {b for _, b in dom}]
        st.markdown(
            ("Aucune stratégie n'en domine une autre : chacune a au moins un point fort. " if not dom else "")
            + "Stratégies non dominées : " + ", ".join(f"**{nom_affichage(n)}**" for n in non_dominees) + "."
        )

        # Étape 3 — comparaison deux à deux
        st.markdown("#### 3. Comparaison deux à deux")
        st.caption(
            "Chaque case indique sur combien de critères la stratégie de la ligne est meilleure "
            "que celle de la colonne, puis moins bonne. Un écart plus petit que le seuil "
            "d'indifférence du critère compte comme une égalité."
        )
        paires = vd.bilan_paires(metriques, retenues, criteres_v)
        st.dataframe(
            pd.DataFrame(
                {
                    nom_affichage(b): [
                        "—" if a == b else f"{len(paires[(a, b)][0])} – {len(paires[(a, b)][1])}"
                        for a in retenues
                    ]
                    for b in retenues
                },
                index=[nom_affichage(a) for a in retenues],
            ),
            width="stretch",
        )
        c_a, c_b = st.columns(2)
        pa = c_a.selectbox("Comparer", retenues, format_func=nom_affichage, key="paire_a")
        pb = c_b.selectbox("à", [n for n in retenues if n != pa], format_func=nom_affichage, key="paire_b")
        mieux, moins = paires[(pa, pb)]
        egaux = [c["libelle"] for c in criteres_v if c["libelle"] not in mieux + moins]
        st.markdown(
            f"Face à {nom_affichage(pb)}, **{nom_affichage(pa)}** fait mieux sur : "
            f"{', '.join(mieux) or 'aucun critère'} ; moins bien sur : "
            f"{', '.join(moins) or 'aucun critère'} ; aussi bien sur : "
            f"{', '.join(egaux) or 'aucun critère'}."
        )
        with st.expander("Seuils d'indifférence"):
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Critère": c["libelle"],
                            "Objectif du projet": c["objectif"],
                            "Sens": "plus haut = mieux" if c["sens"] == "max" else "plus bas = mieux",
                            "Écart jugé significatif au-delà de": c["justification"],
                        }
                        for c in criteres_v
                    ]
                ).set_index("Critère"),
                width="stretch",
            )

        # Étape 4 — pondération et robustesse
        st.markdown("#### 4. Pondération et robustesse")
        st.caption(
            "Le score d'une stratégie est la moyenne pondérée de ses victoires moins ses "
            "défaites, critère par critère (méthode PROMETHEE II). Comme toute pondération est "
            "discutable, on en tire 5 000 au hasard et on compte la part où chaque stratégie "
            "est première."
        )
        with st.expander("Ajuster l'importance des critères (par défaut : toutes égales)"):
            colonnes_p = st.columns(3)
            poids = np.array(
                [
                    colonnes_p[i % 3].slider(c["libelle"], 0, 5, 1, key=f"poids_{c['cle']}")
                    for i, c in enumerate(criteres_v)
                ],
                dtype=float,
            )
        flux = vd.flux_par_critere(metriques, retenues, criteres_v)
        scores = flux @ poids / poids.sum() if poids.sum() > 0 else flux.mean(axis=1)
        premiers, rang_moyen = vd.robustesse(flux)

        ordre_v = list(np.argsort(-premiers))
        fig_v = go.Figure(
            go.Bar(
                y=[nom_affichage(retenues[i]) for i in ordre_v][::-1],
                x=[premiers[i] * 100 for i in ordre_v][::-1],
                orientation="h",
                marker_color=[couleur(retenues[i]) for i in ordre_v][::-1],
                text=[f"{premiers[i] * 100:.0f} %" for i in ordre_v][::-1],
                textposition="outside",
                hoverinfo="skip",
            )
        )
        fig_v.update_layout(
            height=45 * len(retenues) + 80, margin=dict(t=10, b=40, l=10, r=50),
            xaxis=dict(title="Part des pondérations où la stratégie est première (%)", range=[0, 110]),
            showlegend=False,
        )
        st.plotly_chart(fig_v, width="stretch")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Stratégie": nom_affichage(retenues[i]),
                        "Score (votre pondération)": f"{scores[i]:+.2f}",
                        "Première dans": f"{premiers[i] * 100:.0f} % des pondérations",
                        "Rang moyen": f"{rang_moyen[i]:.1f}",
                    }
                    for i in sorted(range(len(retenues)), key=lambda i: -scores[i])
                ]
            ).set_index("Stratégie"),
            width="stretch",
        )

        # Conclusion
        st.markdown("#### Conclusion")
        i_rob = int(np.argmax(premiers))
        i_poids = int(np.argmax(scores))
        meilleur = retenues[i_rob]

        def _points_forts(i):
            forts = [criteres_v[k]["libelle"] for k in np.argsort(-flux[i]) if flux[i, k] > 0]
            return ", ".join(forts[:3]) or "aucun critère en particulier"

        if premiers[i_rob] >= vd.SEUIL_ROBUSTE and meilleur in non_dominees:
            st.success(
                f"**{nom_affichage(meilleur)} est le meilleur modèle** sur ce cycle, tous critères "
                f"confondus : premier dans {premiers[i_rob] * 100:.0f} % des pondérations, et "
                f"aucune stratégie ne le domine. Points forts : {_points_forts(i_rob)}."
            )
        else:
            candidats = [i for i in ordre_v if premiers[i] >= 0.10]
            st.info(
                "**Aucun modèle n'est meilleur sur tous les plans.** "
                f"{nom_affichage(meilleur)} est le plus souvent premier ({premiers[i_rob] * 100:.0f} % "
                "des pondérations), ce qui ne suffit pas à conclure : le classement dépend de "
                f"l'importance donnée aux critères. Avec votre pondération, c'est "
                f"**{nom_affichage(retenues[i_poids])}** qui l'emporte."
            )
            st.markdown(
                "Le choix dépend donc des priorités :\n"
                + "\n".join(
                    f"- **{nom_affichage(retenues[i])}** l'emporte quand on privilégie : {_points_forts(i)} "
                    f"(première dans {premiers[i] * 100:.0f} % des pondérations)"
                    for i in candidats
                )
            )
        st.caption(
            "Verdict valable pour ce cycle et ces modèles entraînés. Pour le généraliser, il "
            "faudrait le répéter sur plusieurs cycles de conduite et plusieurs entraînements "
            "par réseau, avec les pertes intégrées à la simulation."
        )


with tab_critere:
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