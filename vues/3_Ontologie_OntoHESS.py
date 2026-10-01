"""
Page « Base de connaissances » : présente l'ontologie OntoHESS dans un ordre
pédagogique — à quoi elle sert, ce qu'elle décrit, ses règles en langage
courant, son rôle dans chaque stratégie. Les détails destinés aux spécialistes
de l'ontologie sont repliés. On peut aussi tester ses règles sur une situation.

L'explication d'une décision du cycle, instant par instant, est dans la page
« Pourquoi cette décision ? ».
"""

import sys
from pathlib import Path

DOSSIER_PROJET = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DOSSIER_PROJET))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ems_core as core
from core import ontology_explainer as ox
from core.format import nombre, separateurs_plotly
from core.i18n import lib, tr
from core.navigation import pied_navigation
from core.resultats import assurer_donnees_session, nom_affichage
from core.style import COULEUR_DECISION, COULEUR_REFERENCE, COULEUR_SECONDAIRE, flux_html


# L'état déduit est une décision (violet), les composants restent neutres.
C_ETAT = COULEUR_DECISION
C_NOEUD = COULEUR_REFERENCE

# Les quatre grandes catégories de concepts : (nom, ce qu'elle regroupe).
CATEGORIES = {
    "Component": (
        ("Composants", "Components"),
        ("Les éléments physiques : batterie Énergie, batterie Puissance, convertisseur, charge "
         "(moteur) et l'architecture qui les relie.",
         "The physical elements: Energy battery, Power battery, converter, load (motor) and the "
         "architecture that connects them."),
    ),
    "ElectricalQuantity": (
        ("Grandeurs électriques", "Electrical quantities"),
        ("Les puissances, courants, tensions et résistances internes, et les seuils qui les bornent "
         "(puissance maximale de la batterie Énergie, courant maximal…).",
         "The powers, currents, voltages and internal resistances, and the thresholds that bound "
         "them (maximum power of the Energy battery, maximum current…)."),
    ),
    "ManagementStrategy": (
        ("Stratégie de gestion", "Management strategy"),
        ("Ce qu'examine une stratégie de gestion : conditions sur le SOC, sur la charge, sur le "
         "dépassement d'une limite, et le calcul de la puissance du convertisseur.",
         "What a management strategy examines: conditions on the SOC, on the load, on exceeding a "
         "limit, and the calculation of the converter power."),
    ),
    "SystemState": (
        ("États du système", "System states"),
        ("Les états du système : fonctionnement dans les limites de la batterie Énergie ou au-delà, "
         "états de charge et états de puissance.",
         "The system states: operation within the Energy battery's limits or beyond, states of "
         "charge and power states."),
    ),
}

TYPES = {
    "mode": ("Mode de fonctionnement", "Operating mode"),
    "repartition": ("Répartition de la puissance", "Power split"),
    "courant": ("Limitation de courant", "Current limitation"),
    "calcul": ("Calcul d'une grandeur", "Calculation of a quantity"),
}
SUJETS = {
    "eb": ("batterie Énergie", "Energy battery"), "pb": ("batterie Puissance", "Power battery"),
    "l": ("charge", "load"), "c": ("convertisseur", "converter"),
}


def _sujet(s):
    return lib(SUJETS[s]) if s in SUJETS else s


def _graphe_connaissances(etat_actif):
    """Objets décrits par l'ontologie et leurs relations ; l'état déduit pour la
    situation testée est mis en avant."""
    noeuds = {
        "hess1": (0.0, 2.0, tr("Système HESS", "HESS system")),
        "managementStrategy1": (-2.2, 2.0, tr("Stratégie de gestion", "Management strategy")),
        "batteryE1": (-1.6, 1.0, tr("Batterie Énergie", "Energy battery")),
        "batteryP1": (0.0, 1.0, tr("Batterie Puissance", "Power battery")),
        "converter1": (1.6, 1.0, tr("Convertisseur", "Converter")),
        "load1": (1.6, 0.0, tr("Charge (moteur)", "Load (motor)")),
        **{
            cle: (x, -1.0, ox.libelle_etat(cle, court=True))
            for cle, x in (("state_Normal", -1.6), ("state_Overload_High", 0.0), ("state_Overload_Low", 1.6))
        },
    }
    aretes = [
        ("hess1", "managementStrategy1"), ("hess1", "batteryE1"), ("hess1", "batteryP1"),
        ("hess1", "converter1"), ("converter1", "batteryE1"), ("converter1", "load1"),
        ("hess1", "state_Normal"), ("hess1", "state_Overload_High"), ("hess1", "state_Overload_Low"),
    ]
    fig = go.Figure()
    for a, b in aretes:
        actif = b == etat_actif
        fig.add_trace(
            go.Scatter(
                x=[noeuds[a][0], noeuds[b][0]], y=[noeuds[a][1], noeuds[b][1]], mode="lines",
                line=dict(color=C_ETAT if actif else "#9AA0AA", width=3 if actif else 1.2),
                hoverinfo="skip", showlegend=False,
            )
        )
    etats = [c.startswith("state_") for c in noeuds]
    fig.add_trace(
        go.Scatter(
            x=[v[0] for v in noeuds.values()], y=[v[1] for v in noeuds.values()],
            mode="markers+text",
            marker=dict(
                size=[38 if c == etat_actif else (24 if e else 30) for c, e in zip(noeuds, etats)],
                color=[C_ETAT if c == etat_actif else (COULEUR_SECONDAIRE if e else C_NOEUD) for c, e in zip(noeuds, etats)],
                line=dict(color="white", width=2),
            ),
            text=[v[2] for v in noeuds.values()], textposition="bottom center", textfont=dict(size=11),
            hovertext=[tr("{l} (nom dans l'ontologie : {c})", "{l} (name in the ontology: {c})", l=v[2], c=c) for c, v in noeuds.items()],
            hoverinfo="text", showlegend=False,
        )
    )
    fig.update_layout(
        separators=separateurs_plotly(), height=380, margin=dict(t=10, b=10, l=10, r=10),
        xaxis=dict(visible=False, range=[-3.0, 2.8]), yaxis=dict(visible=False, range=[-1.9, 2.5]),
    )
    return fig


st.title(tr("📚 Base de connaissances", "📚 Knowledge base"))
st.markdown(tr("#### OntoHESS : les connaissances sur le HESS, écrites noir sur blanc",
               "#### OntoHESS: knowledge about the HESS, written down explicitly"))
st.caption(tr(
    "Une ontologie est une description structurée d'un domaine : ici, les composants du HESS, leurs "
    "grandeurs, leurs états et les règles d'un expert pour interpréter la situation du système. La "
    "page « Pourquoi cette décision ? » l'applique à chaque instant du cycle.",
    "An ontology is a structured description of a domain: here, the components of the HESS, their "
    "quantities, their states and an expert's rules for interpreting the situation of the system. "
    "The “Why this decision?” page applies it at each time step of the cycle.",
))

regles = ox.charger_regles()
relations, attributs, individus = ox.vocabulaire_ontologie()
hierarchie = ox.hierarchie_classes()

if not regles or not individus:
    st.warning(tr("L'ontologie n'a pas pu être lue.", "The ontology could not be read."))
    st.stop()

st.markdown(tr("**Contenu de l'ontologie**", "**Contents of the ontology**"))
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric(tr("Concepts", "Concepts"), len(ox.classes_ontologie()),
          help=tr("Les notions décrites : batterie, convertisseur, seuil, état…", "The notions described: battery, converter, threshold, state…"))
m2.metric(tr("Relations", "Relations"), len(relations),
          help=tr("Les liens entre concepts : « est composé de », « alimente »…", "The links between concepts: “is composed of”, “powers”…"))
m3.metric(tr("Propriétés", "Properties"), len(attributs),
          help=tr("Les grandeurs attachées aux concepts : tension, SOC, puissance maximale…", "The quantities attached to concepts: voltage, SOC, maximum power…"))
m4.metric(tr("Objets décrits", "Objects described"), len(individus),
          help=tr("Les éléments concrets du HESS étudié, avec leurs valeurs", "The actual elements of the HESS studied, with their values"))
m5.metric(tr("Règles", "Rules"), len(regles), help=tr("Règles logiques du type « si … alors … »", "Logical rules of the form “if … then …”"))

st.markdown(
    flux_html([
        (tr("Situation du HESS", "Situation of the HESS"), COULEUR_REFERENCE),
        (tr("Grandeurs électriques + SOC", "Electrical quantities + SOC"), COULEUR_REFERENCE),
        (tr("État du système", "System state"), COULEUR_REFERENCE),
        (tr("Règles expertes", "Expert rules"), COULEUR_SECONDAIRE),
        (tr("État déduit", "Inferred state"), COULEUR_SECONDAIRE),
        (tr("Décision de gestion", "Management decision"), COULEUR_DECISION),
    ]),
    unsafe_allow_html=True,
)
st.caption(tr(
    "Le chemin que décrit l'ontologie : de la situation mesurée à la décision de gestion.",
    "The path described by the ontology: from the measured situation to the management decision.",
))


# A — À quoi sert OntoHESS ?

st.subheader(tr("A. À quoi sert OntoHESS ?", "A. What is OntoHESS for?"))
st.markdown(tr(
    "- **Décrire** le système avec un vocabulaire commun : batteries, convertisseur, grandeurs, seuils.\n"
    "- **Raisonner** : à partir de la puissance demandée et des SOC, ses règles en déduisent l'état "
    "de fonctionnement et une répartition de référence de la puissance.\n"
    "- **Expliquer** : chaque décision d'une stratégie peut être relue avec ces règles, et les "
    "stratégies neuro-symboliques s'appuient directement sur ses concepts.",
    "- **Describe** the system with a shared vocabulary: batteries, converter, quantities, thresholds.\n"
    "- **Reason**: from the power demand and the SOCs, its rules infer the operating state and a "
    "reference power split.\n"
    "- **Explain**: each decision of a strategy can be re-read with these rules, and the "
    "neuro-symbolic strategies rely directly on its concepts.",
))


# B — Que décrit OntoHESS ?

st.subheader(tr("B. Que décrit OntoHESS ?", "B. What does OntoHESS describe?"))
colonnes = st.columns(len(hierarchie))
for col, (racine, enfants) in zip(colonnes, hierarchie.items()):
    nom_cat, description = CATEGORIES.get(racine, ((racine, racine), ("", "")))
    with col:
        with st.container(border=True):
            st.markdown(f"**{lib(nom_cat)}**")
            st.caption(lib(description))
            st.markdown(tr("{n} concepts", "{n} concepts", n=len(enfants)))

with st.expander(tr("Détails pour les spécialistes de l'ontologie", "Details for ontology specialists")):
    st.caption(tr(
        "Les noms ci-dessous sont ceux du fichier de l'ontologie (en anglais).",
        "The names below are those of the ontology file.",
    ))
    st.markdown(tr("**Concepts**, rangés sous leurs quatre grandes catégories", "**Concepts**, grouped under their four main categories"))
    colonnes = st.columns(len(hierarchie))
    for col, (racine, enfants) in zip(colonnes, hierarchie.items()):
        with col:
            st.markdown(f"`{racine}`")
            st.markdown("\n".join(
                f"- `{c}`" + (f" — {ox.nom_classe(c)}" if c in ox.CLASSES else "") for c in enfants
            ))

    st.markdown(tr(
        "**Objets décrits** : les éléments concrets du HESS étudié, avec les valeurs que l'ontologie leur attribue",
        "**Objects described**: the actual elements of the HESS studied, with the values the ontology gives them",
    ))
    col_objet = tr("Objet", "Object")
    st.dataframe(
        pd.DataFrame([
            {
                col_objet: nom,
                tr("Concepts", "Concepts"): ", ".join(types),
                tr("Propriétés", "Properties"): " ; ".join(f"{p} = {v}" for p, v in props),
            }
            for nom, types, props in ox.individus_ontologie()
        ]).set_index(col_objet),
        width="stretch",
    )

    # Cohérence entre l'ontologie et les paramètres de la simulation
    props_eb = dict(next((p for nom, _, p in ox.individus_ontologie() if nom == "batteryE1"), []))
    i_max_onto = props_eb.get("iEB_max_value")
    i_max_sim = core.P_EB_MAX_W / core.V_EB_PACK_NOM
    if i_max_onto is not None and abs(float(i_max_onto) - i_max_sim) > 0.01 * i_max_sim:
        st.warning(tr(
            "Écart entre l'ontologie et la simulation : le courant maximal de la batterie Énergie vaut "
            "{a} A dans l'ontologie, alors que la simulation la limite à {b} A ({p} kW).",
            "Mismatch between the ontology and the simulation: the Energy battery's maximum current is "
            "{a} A in the ontology, whereas the simulation limits it to {b} A ({p} kW).",
            a=nombre(float(i_max_onto), 2), b=nombre(i_max_sim, 1), p=nombre(core.P_EB_MAX_W / 1000, 1),
        ))

    st.markdown(tr("**Relations** entre concepts ({n})", "**Relations** between concepts ({n})", n=len(ox.relations_ontologie())))
    col_relation = tr("Relation", "Relation")
    st.dataframe(
        pd.DataFrame([
            {col_relation: r, tr("De", "From"): ", ".join(d) or "—", tr("Vers", "To"): ", ".join(p) or "—"}
            for r, d, p in ox.relations_ontologie()
        ]).set_index(col_relation),
        width="stretch",
    )

    st.markdown(tr("**Propriétés** ({n})", "**Properties** ({n})", n=len(attributs)))
    st.caption(" · ".join(f"`{a}`" for a in attributs))


# C — Quelles règles sont utilisées ?

st.subheader(tr("C. Quelles règles sont utilisées ?", "C. Which rules are used?"))
st.caption(tr(
    "Les règles lues à chaque instant du cycle, en langage courant. Elles reconnaissent d'abord le "
    "mode de fonctionnement, puis prescrivent une répartition de la puissance.",
    "The rules read at each time step of the cycle, in plain language. They first recognise the "
    "operating mode, then prescribe a power split.",
))
lues = {"mode": [], "repartition": []}
for r in regles:
    type_r = ox.lire_regle(r)[0]
    if type_r in lues:
        lues[type_r].append(r)

c_mode, c_rep = st.columns(2)
for col, type_r, titre_bloc in (
    (c_mode, "mode", tr("Reconnaître le mode de fonctionnement", "Recognise the operating mode")),
    (c_rep, "repartition", tr("Répartir la puissance entre les batteries", "Split the power between the batteries")),
):
    with col:
        with st.container(border=True):
            st.markdown(f"**{titre_bloc}**")
            st.markdown("\n".join(f"- **{r['id']}** — {ox.regle_en_phrase(r)}" for r in lues[type_r]))

autres = [r["id"] for r in regles if ox.lire_regle(r)[0] not in lues]
st.caption(tr(
    "Les {n} autres règles ({l}) calculent des grandeurs ou limitent des courants internes que la "
    "simulation ne calcule pas : elles se consultent ci-dessous.",
    "The {n} other rules ({l}) compute quantities or limit internal currents that the simulation "
    "does not compute: they can be browsed below.",
    n=len(autres), l=", ".join(autres),
))

# Vérification : les règles de répartition décrivent-elles le modèle physique ?
try:
    assurer_donnees_session(st)
    _res = st.session_state.get("resultats_simulation") or {}
    _df = st.session_state.get("cycle_pret")
except FileNotFoundError:
    _res, _df = {}, None
if "EMS_power_limitation" in _res and _df is not None:
    _t = _res["EMS_power_limitation"]
    _n = min(len(_df), len(_t["alpha_requested"]))
    _onto = ox.alpha_ontologie_vect(_df["hasPower"].to_numpy(dtype=float)[:_n], np.asarray(_t["SOC_EB"], float)[:_n])
    _m = ~np.isnan(_onto)
    _ecart = np.abs(np.asarray(_t["alpha_requested"], float)[:_n][_m] - _onto[_m])
    st.info(tr(
        "Les règles de répartition R13 à R17 décrivent exactement le modèle physique : sur les {n} "
        "instants du cycle où elles s'appliquent, la répartition qu'elles prescrivent et celle du "
        "modèle physique diffèrent au plus de {e} point.",
        "The split rules R13 to R17 describe the physical model exactly: over the {n} time steps of "
        "the cycle where they apply, the split they prescribe and that of the physical model differ by "
        "at most {e} point.",
        n=nombre(_m.sum(), 0), e=nombre(_ecart.max() * 100, 2),
    ))

st.markdown(tr("**Parcourir toutes les règles**", "**Browse all the rules**"))
onglet_onto, onglet_flou = st.tabs([
    tr("Règles de l'ontologie ({n})", "Ontology rules ({n})", n=len(regles)),
    tr("Règles de la logique floue ({n})", "Fuzzy-logic rules ({n})", n=len(ox.regles_floues())),
])

with onglet_onto:
    roles = ["toutes"] + list(TYPES)
    role = st.radio(
        tr("Rôle de la règle", "Role of the rule"), roles, horizontal=True, key="role_regle",
        format_func=lambda r: tr("Toutes", "All") if r == "toutes" else lib(TYPES[r]),
    )
    choix = [r for r in regles if role == "toutes" or ox.lire_regle(r)[0] == role]
    regle = st.selectbox(
        tr("Règle", "Rule"), choix, format_func=lambda r: f"{r['id']} — {ox.lire_regle(r)[1]}", key="regle_onto",
    )
    type_r, lecture = ox.lire_regle(regle)
    with st.container(border=True):
        st.markdown(f"#### {regle['id']} · {lib(TYPES[type_r])}")
        st.markdown(tr("**En clair** : {p}", "**In plain words**: {p}", p=ox.regle_en_phrase(regle)))
        st.caption(tr("Écriture de la règle dans l'ontologie :", "How the rule is written in the ontology:"))
        f1, f2 = st.columns(2)
        with f1:
            st.markdown(tr("**S'applique à**", "**Applies to**"))
            st.markdown(", ".join(ox.nom_classe(c) for c in regle["classes"]) or "—")
            st.markdown(tr("**Grandeurs lues**", "**Quantities read**"))
            st.markdown(
                "\n".join(f"- `{args[-1]}` : {ox.nom_clair(p)} ({_sujet(args[0])})" for p, args in regle["lectures"] if len(args) == 2)
                or "—"
            )
        with f2:
            st.markdown(tr("**Si**", "**If**"))
            st.markdown("\n".join(f"- `{c}`" for c in ox.conditions_en_clair(regle)) or tr("- toujours (pas de condition)", "- always (no condition)"))
            if regle["calculs"]:
                st.markdown(tr("**Calcule**", "**Computes**"))
                st.markdown("\n".join(f"- `{c}`" for c in ox.calculs_en_clair(regle)))
            st.markdown(tr("**Alors**", "**Then**"))
            st.markdown(
                "\n".join(f"- {ox.nom_clair(p)} ({_sujet(args[0])}) = `{args[1]}`" for p, args in regle["affectations"] if len(args) == 2)
                or "—"
            )
        st.caption(
            tr("Règle appliquée à chaque instant du cycle.", "Rule applied at each time step of the cycle.")
            if type_r in ("mode", "repartition") else
            tr("Règle non appliquée pendant la simulation : elle porte sur des courants et tensions "
               "internes que la simulation ne calcule pas, ou n'a pas de condition.",
               "Rule not applied during the simulation: it concerns internal currents and voltages that "
               "the simulation does not compute, or has no condition.")
        )

    with st.expander(tr("Vue d'ensemble des {n} règles", "Overview of the {n} rules", n=len(regles))):
        col_regle = tr("Règle", "Rule")
        st.dataframe(
            pd.DataFrame([
                {
                    col_regle: r["id"],
                    tr("Rôle", "Role"): lib(TYPES[ox.lire_regle(r)[0]]),
                    tr("Si", "If"): tr(" et ", " and ").join(ox.conditions_en_clair(r)) or tr("toujours", "always"),
                    tr("Alors", "Then"): ox.lire_regle(r)[1],
                }
                for r in regles
            ]).set_index(col_regle),
            width="stretch", height=420,
        )
    with st.expander(tr("Signification des symboles", "Meaning of the symbols")):
        st.markdown("\n".join(f"- `{v}` : {s}" for v, s in ox.glossaire()))

with onglet_flou:
    st.caption(tr(
        "Ces règles sont propres à la stratégie « logique floue » : elles ne figurent pas dans le "
        "fichier de l'ontologie, mais emploient ses concepts. Une condition y est vraie à un certain "
        "degré, entre 0 et 1 ; la décision est la moyenne des conclusions des règles, pondérée par ces "
        "degrés. La contribution de chaque règle à la décision est donc connue exactement.",
        "These rules belong to the “fuzzy logic” strategy: they are not in the ontology file, but use "
        "its concepts. A condition is true to a certain degree, between 0 and 1; the decision is the "
        "average of the rules' conclusions, weighted by these degrees. The contribution of each rule "
        "to the decision is therefore known exactly.",
    ))
    floues = ox.regles_floues()
    rf = st.selectbox(tr("Règle floue", "Fuzzy rule"), floues, format_func=lambda r: r["libelle"], key="regle_floue")
    with st.container(border=True):
        st.markdown(f"#### {rf['libelle']}")
        st.markdown(tr("**Si** {c}", "**If** {c}", c=rf["si"]))
        st.markdown(tr(
            "**Alors** confier **{a} %** de la puissance à la batterie Puissance, c'est-à-dire {s}.",
            "**Then** assign **{a} %** of the power to the Power battery, i.e. {s}.",
            a=nombre(rf["alpha"] * 100, 0), s=rf["sens"],
        ))
        if rf["concepts"]:
            st.markdown(
                tr("**Concepts de l'ontologie** : ", "**Ontology concepts**: ")
                + ", ".join(f"{nom} (`{classe}`)" for nom, classe in rf["concepts"])
            )
    col_regle_floue = tr("Règle", "Rule")
    st.dataframe(
        pd.DataFrame([
            {col_regle_floue: r["libelle"], tr("Si", "If"): r["si"],
             tr("Part de la batterie Puissance", "Power battery share"): f"{nombre(r['alpha'] * 100, 0)} %"}
            for r in floues
        ]).set_index(col_regle_floue),
        width="stretch",
    )
    with st.expander(tr("Définition chiffrée des conditions", "Numerical definition of the conditions")):
        st.markdown("\n".join(f"- **{terme}** : {definition}" for terme, definition in ox.termes_flous()))
        st.caption(tr(
            "Quand aucune règle n'est vraie, la répartition par défaut est de {a} % pour la batterie Puissance.",
            "When no rule is true, the default split is {a} % for the Power battery.",
            a=nombre(core.FUZZY_DEFAULT_ALPHA * 100, 0),
        ))


# D — Comment l'ontologie intervient dans les stratégies ?

st.subheader(tr("D. Comment l'ontologie intervient dans les stratégies ?", "D. How does the ontology come into the strategies?"))
aucun = tr("Aucun usage.", "Not used.")
USAGES = {
    "EMS_power_limitation": tr(
        "Ne lit pas l'ontologie, mais prend exactement les décisions de ses règles de répartition R13 à R17.",
        "Does not read the ontology, but makes exactly the decisions of its split rules R13 to R17.",
    ),
    "EMS_fuzzy_logic": tr(
        "Ses règles reposent sur les concepts « état de charge » et « état de puissance » de l'ontologie.",
        "Its rules rely on the ontology's “state of charge” and “power state” concepts.",
    ),
    "EMS_MLP": aucun,
    "EMS_LSTM": aucun,
    "EMS_GNN": tr(
        "Pas d'usage direct ; son schéma reprend les composants du HESS (batteries, convertisseur, moteur, véhicule).",
        "No direct use; its diagram uses the components of the HESS (batteries, converter, motor, vehicle).",
    ),
    "EMS_MLP_neurosymbolic": tr(
        "Part des règles floues, reçoit des états déduits par l'ontologie, et reste sous le contrôle de ses règles R14 et R16.",
        "Starts from the fuzzy rules, receives states inferred by the ontology, and stays under the control of its rules R14 and R16.",
    ),
    "EMS_LSTM_neurosymbolic": tr(
        "Reçoit quatre états déduits par l'ontologie : forte demande, freinage, demande nulle, convertisseur proche de sa limite.",
        "Receives four states inferred by the ontology: high demand, braking, zero demand, converter close to its limit.",
    ),
}
col_strategie = tr("Stratégie", "Strategy")
st.dataframe(
    pd.DataFrame([{col_strategie: nom_affichage(c), tr("Usage de l'ontologie", "Use of the ontology"): u} for c, u in USAGES.items()])
    .set_index(col_strategie),
    width="stretch",
)

st.markdown(tr(
    "**Les deux façons d'associer l'ontologie à un réseau de neurones**",
    "**The two ways of combining the ontology with a neural network**",
))
ns1, ns2 = st.columns(2)
with ns1:
    with st.container(border=True):
        st.markdown(tr("**NS-MLP** · les règles décident, le réseau corrige", "**NS-MLP** · the rules decide, the network corrects"))
        st.markdown(
            flux_html([
                (tr("P_dem, SOC", "P_dem, SOC"), COULEUR_REFERENCE),
                (tr("Règles floues", "Fuzzy rules"), COULEUR_SECONDAIRE),
                (tr("Décision des règles", "Decision of the rules"), COULEUR_DECISION),
                (tr("Correction du réseau (±{c} pts)", "Network correction (±{c} pts)", c=nombre(core.MLP_NS_MAX_DELTA * 100, 0)), COULEUR_SECONDAIRE),
                (tr("Garde-fou R14/R16", "Safeguard R14/R16"), COULEUR_REFERENCE),
                (tr("Décision finale", "Final decision"), COULEUR_DECISION),
            ]),
            unsafe_allow_html=True,
        )
        st.caption(tr(
            "La décision se décompose exactement : part des règles + correction du réseau. Sous {r} % "
            "de SOC de la batterie Puissance, les règles R14 et R16 reprennent la main en traction pour "
            "préserver sa réserve.",
            "The decision splits exactly: share from the rules + network correction. Below {r} % SOC of "
            "the Power battery, rules R14 and R16 take over in traction to preserve its reserve.",
            r=nombre(core.MLP_NS_RESERVE_PB_SOC * 100, 0),
        ))
with ns2:
    with st.container(border=True):
        st.markdown(tr("**NS-LSTM** · l'ontologie informe le réseau", "**NS-LSTM** · the ontology informs the network"))
        st.markdown(
            flux_html([
                (tr("{w} dernières secondes + 4 états déduits par l'ontologie", "Last {w} seconds + 4 states inferred by the ontology",
                    w=core.LSTM_WINDOW), COULEUR_REFERENCE),
                (tr("Réseau LSTM", "LSTM network"), COULEUR_SECONDAIRE),
                (tr("Décision finale", "Final decision"), COULEUR_DECISION),
            ]),
            unsafe_allow_html=True,
        )
        st.caption(tr(
            "Les états déduits sont des entrées parmi d'autres : leur poids dans la décision se mesure après coup.",
            "The inferred states are inputs among others: their weight in the decision is measured afterwards.",
        ))
st.caption(tr(
    "Pour toutes les stratégies, l'ontologie sert aussi à expliquer : chaque décision est relue avec "
    "ses règles dans la page « Pourquoi cette décision ? ».",
    "For all strategies, the ontology is also used to explain: each decision is re-read with its "
    "rules on the “Why this decision?” page.",
))


# Tester les règles sur une situation choisie

st.subheader(tr("Tester les règles sur une situation", "Test the rules on a situation"))
st.caption(tr(
    "Choisissez une situation : l'ontologie en déduit l'état de fonctionnement et la répartition de "
    "référence, règle par règle.",
    "Choose a situation: the ontology infers the operating state and the reference split, rule by rule.",
))
s1, s2, s3 = st.columns(3)
p_test = s1.slider(tr("Puissance demandée (kW)", "Power demand (kW)"), -50.0, 50.0, 15.0, 0.5) * 1000.0
soc_eb_test = s2.slider(tr("SOC de la batterie Énergie (%)", "Energy battery SOC (%)"), 0, 100, 60) / 100.0
soc_pb_test = s3.slider(tr("SOC de la batterie Puissance (%)", "Power battery SOC (%)"), 0, 100, 80) / 100.0

etat_test = ox.etat_fonctionnement(p_test)
activees, non_activees, _ = ox.evaluer_regles(p_test, soc_eb_test, soc_pb_test)
repart = ox.repartition_ontologie(p_test, soc_eb_test, soc_pb_test)

g_col, t_col = st.columns([1, 1])
with g_col:
    st.plotly_chart(_graphe_connaissances(etat_test), width="stretch")
with t_col:
    st.markdown(tr("État déduit : **{e}**", "Inferred state: **{e}**", e=ox.libelle_etat(etat_test)))
    for r in sorted((r for r in activees if r["type"] in ("mode", "repartition")), key=lambda r: r["type"]):
        premisses = tr(" et ", " and ").join(f"`{d['texte']}`" for d in r["details"])
        st.markdown(tr("- **{i}** — si {p}, alors {l}.", "- **{i}** — if {p}, then {l}.", i=r["id"], p=premisses, l=r["lecture"]))
    if repart is not None:
        a = repart["alpha"]
        st.success(tr(
            "Répartition de référence (règle {i}) : batterie Énergie {eb} kW, batterie Puissance {pb} kW (alpha = {a} %).",
            "Reference split (rule {i}): Energy battery {eb} kW, Power battery {pb} kW (alpha = {a} %).",
            i=repart["regle"]["id"], eb=nombre(p_test * (1 - a) / 1000, 1), pb=nombre(p_test * a / 1000, 1), a=nombre(a * 100, 0),
        ))
    else:
        st.info(tr("Demande quasi nulle : aucune règle de répartition ne s'applique.",
                   "Near-zero demand: no split rule applies."))

with st.expander(tr("Règles qui ne s'appliquent pas, et pourquoi", "Rules that do not apply, and why")):
    for r in non_activees:
        if r["type"] not in ("mode", "repartition"):
            continue
        echecs = " ; ".join(d["texte"] for d in r["details"] if d["ok"] is False)
        st.markdown(tr("- **{i}** ({l}) : `{e}` n'est pas vérifié", "- **{i}** ({l}): `{e}` does not hold",
                       i=r["id"], l=r["lecture"], e=echecs))


pied_navigation("vues/3_Ontologie_OntoHESS.py")
