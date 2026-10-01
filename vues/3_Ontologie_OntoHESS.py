"""
Page « Base de connaissances » : présente l'ontologie OntoHESS elle-même
dans un ordre pédagogique : à quoi elle sert, ce qu'elle décrit, ses règles
en langage courant, son rôle dans chaque stratégie ; les détails techniques
sont repliés. On peut aussi tester ses règles sur une situation choisie.

L'explication d'une décision du cycle, instant par instant, est dans la page
« Pourquoi cette décision ? » : elle n'est pas répétée ici.
"""

import sys
from pathlib import Path

DOSSIER_PROJET = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DOSSIER_PROJET))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from core.format import SEPARATEURS_PLOTLY, nombre

import ems_core as core
from core import ontology_explainer as ox
from core.navigation import pied_navigation
from core.resultats import assurer_donnees_session, nom_affichage
from core.style import flux_html, COULEUR_DECISION, COULEUR_REFERENCE, COULEUR_SECONDAIRE


# Palette commune : l'état inféré est une décision (violet), les composants restent neutres.
C_ETAT = COULEUR_DECISION
C_NOEUD = COULEUR_REFERENCE

RACINES_FR = {
    "Component": "Composants",
    "ElectricalQuantity": "Grandeurs électriques",
    "ManagementStrategy": "Stratégie de gestion",
    "SystemState": "États du système",
}


def _graphe_connaissances(etat_actif):
    """Individus déclarés dans OntoHESS2.owl et leurs relations ; l'état
    inféré pour la situation testée est mis en avant."""
    noeuds = {
        "hess1": (0.0, 2.0, "Système HESS"),
        "managementStrategy1": (-2.2, 2.0, "Stratégie de gestion"),
        "batteryE1": (-1.6, 1.0, "Batterie Énergie"),
        "batteryP1": (0.0, 1.0, "Batterie Puissance"),
        "converter1": (1.6, 1.0, "Convertisseur"),
        "load1": (1.6, 0.0, "Charge (moteur)"),
        **{
            cle: (x, -1.0, ox.ETATS_ONTOLOGIE_COURTS[cle])
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
            text=[f"{v[2]}<br><span style='font-size:9px'>{c}</span>" for c, v in noeuds.items()],
            textposition="bottom center", textfont=dict(size=10),
            hoverinfo="text", showlegend=False,
        )
    )
    fig.update_layout(
        separators=SEPARATEURS_PLOTLY, height=380, margin=dict(t=10, b=10, l=10, r=10),
        xaxis=dict(visible=False, range=[-3.0, 2.8]), yaxis=dict(visible=False, range=[-1.9, 2.5]),
    )
    return fig


# Configuration de page gérée par le routeur Accueil.py.

st.title("📚 Base de connaissances")
st.markdown("#### OntoHESS : formalisation des connaissances du HESS")
st.caption(
    "Cette ontologie décrit les composants du HESS, leurs grandeurs, leurs états et les "
    "règles utilisées pour interpréter la situation du système. La page « Pourquoi cette "
    "décision ? » l'applique à chaque instant du cycle."
)

regles = ox.charger_regles()
relations, attributs, individus = ox.vocabulaire_ontologie()
hierarchie = ox.hierarchie_classes()

if not regles or not individus:
    st.warning("Ontologie non chargée (fichier ontologies/OntoHESS2.owl ou rdflib absent).")
    st.stop()

st.markdown("**Contenu de l'ontologie**")
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Classes", len(ox.classes_ontologie()))
m2.metric("Relations", len(relations))
m3.metric("Attributs", len(attributs))
m4.metric("Individus", len(individus))
m5.metric("Règles SWRL", len(regles))

st.markdown(
    flux_html(
        [
            ("Situation du HESS", COULEUR_REFERENCE),
            ("Grandeurs électriques + SOC", COULEUR_REFERENCE),
            ("État du système", COULEUR_REFERENCE),
            ("Règles expertes", COULEUR_SECONDAIRE),
            ("État symbolique", COULEUR_SECONDAIRE),
            ("Décision de gestion", COULEUR_DECISION),
        ]
    ),
    unsafe_allow_html=True,
)
st.caption("Le chemin que formalise l'ontologie : de la situation mesurée à la décision de gestion.")


# A — À quoi sert OntoHESS ?

st.subheader("A. À quoi sert OntoHESS ?")
st.markdown(
    "- **Décrire** le système avec un vocabulaire commun : batteries, convertisseur, "
    "grandeurs, seuils.\n"
    "- **Raisonner** : à partir de la puissance demandée et des SOC, ses règles en "
    "déduisent l'état de fonctionnement et une répartition de référence de la puissance.\n"
    "- **Expliquer** : chaque décision d'une stratégie peut être relue avec ces règles, "
    "et les modèles neuro-symboliques s'appuient directement sur ses concepts."
)


# B — Que décrit OntoHESS ?

st.subheader("B. Que décrit OntoHESS ?")
DESCRIPTIONS = {
    "Component": "Les éléments physiques : batterie Énergie, batterie Puissance, convertisseur "
    "DC-DC, charge (moteur) et l'architecture qui les relie.",
    "ElectricalQuantity": "Les puissances, courants, tensions et résistances internes, et les "
    "seuils qui les bornent (puissance maximale de l'EB, courant maximal…).",
    "ManagementStrategy": "Ce qu'examine une stratégie de gestion : conditions sur le SOC, sur la "
    "charge, sur le dépassement de limite, et le calcul de la puissance du convertisseur.",
    "SystemState": "Les états du système : fonctionnement dans les limites ou au-delà de la limite "
    "de l'EB, états de charge et états de puissance.",
}
colonnes = st.columns(len(hierarchie))
for col, (racine, enfants) in zip(colonnes, hierarchie.items()):
    with col:
        with st.container(border=True):
            st.markdown(f"**{RACINES_FR.get(racine, racine)}**")
            st.caption(DESCRIPTIONS.get(racine, ""))
            st.markdown(f"{len(enfants)} classes")

with st.expander("Afficher les détails techniques de l'ontologie"):
    st.markdown("**Classes**, rangées sous leurs quatre grandes catégories")
    colonnes = st.columns(len(hierarchie))
    for col, (racine, enfants) in zip(colonnes, hierarchie.items()):
        with col:
            st.markdown(f"`{racine}`")
            st.markdown(
                "\n".join(
                    f"- `{c}`" + (f" — {ox.CLASSES_FR[c]}" if c in ox.CLASSES_FR else "")
                    for c in enfants
                )
            )

    st.markdown("**Individus** : les objets concrets du HESS étudié, avec les valeurs que l'ontologie leur attribue")
    lignes_ind = [
        {
            "Individu": nom,
            "Classes": ", ".join(types),
            "Propriétés": " ; ".join(f"{p} = {v}" for p, v in props),
        }
        for nom, types, props in ox.individus_ontologie()
    ]
    st.dataframe(pd.DataFrame(lignes_ind).set_index("Individu"), width="stretch")

    # Cohérence entre l'ontologie et les paramètres de la simulation
    _props_eb = dict(next((p for nom, _, p in ox.individus_ontologie() if nom == "batteryE1"), []))
    _i_max_onto = _props_eb.get("iEB_max_value")
    _i_max_sim = core.P_EB_MAX_W / core.V_EB_PACK_NOM
    if _i_max_onto is not None and abs(float(_i_max_onto) - _i_max_sim) > 0.01 * _i_max_sim:
        st.warning(
            f"Écart entre l'ontologie et la simulation : `iEB_max_value` vaut {nombre(float(_i_max_onto), 2)} A "
            f"dans l'ontologie, alors que la simulation limite l'EB à {nombre(_i_max_sim, 1)} A "
            f"({nombre(core.P_EB_MAX_W / 1000, 1)} kW). Les deux sources doivent être alignées."
        )

    st.markdown(f"**Relations** entre classes ({len(ox.relations_ontologie())})")
    st.dataframe(
        pd.DataFrame(
            [
                {"Relation": r, "De": ", ".join(d) or "—", "Vers": ", ".join(p) or "—"}
                for r, d, p in ox.relations_ontologie()
            ]
        ).set_index("Relation"),
        width="stretch",
    )

    st.markdown(f"**Attributs** ({len(attributs)})")
    st.caption(" · ".join(f"`{a}`" for a in attributs))


# C — Quelles règles sont utilisées ?

st.subheader("C. Quelles règles sont utilisées ?")
st.caption(
    "Les règles SWRL lues à chaque instant du cycle, en langage courant. Elles reconnaissent "
    "d'abord le mode de fonctionnement, puis prescrivent une répartition de la puissance."
)
TYPES_FR = {
    "mode": "Mode de fonctionnement",
    "repartition": "Répartition de la puissance",
    "courant": "Limitation de courant",
    "calcul": "Calcul d'une grandeur",
}
lues = {"mode": [], "repartition": []}
for r in regles:
    type_r = ox.lire_regle(r)[0]
    if type_r in lues:
        lues[type_r].append(r)

c_mode, c_rep = st.columns(2)
for col, type_r, titre in (
    (c_mode, "mode", "Reconnaître le mode de fonctionnement"),
    (c_rep, "repartition", "Répartir la puissance entre les batteries"),
):
    with col:
        with st.container(border=True):
            st.markdown(f"**{titre}**")
            st.markdown("\n".join(f"- **{r['id']}** — {ox.regle_en_phrase(r)}" for r in lues[type_r]))

autres = [r["id"] for r in regles if ox.lire_regle(r)[0] not in lues]
st.caption(
    f"Les {len(autres)} autres règles ({', '.join(autres)}) calculent des grandeurs ou limitent "
    "des courants internes que la simulation ne calcule pas : elles sont consultables ci-dessous."
)

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
    _nb = f"{_m.sum():,}".replace(",", " ")
    st.info(
        f"Les règles de répartition R13 à R17 décrivent exactement le {nom_affichage('EMS_power_limitation').lower()} : "
        f"sur les {_nb} instants du cycle où elles s'appliquent, la répartition qu'elles "
        f"prescrivent et celle demandée par cette stratégie diffèrent au plus de "
        f"{nombre(_ecart.max() * 100, 2)} point."
    )

st.markdown("**Parcourir toutes les règles**")
SUJETS = {"eb": "batterie Énergie", "pb": "batterie Puissance", "l": "charge", "c": "convertisseur"}


def _sujet(s):
    return SUJETS.get(s, s)


onglet_swrl, onglet_flou = st.tabs(
    [f"Règles SWRL de l'ontologie ({len(regles)})", f"Règles floues du moteur ({len(ox.regles_floues())})"]
)

with onglet_swrl:
    role = st.radio("Rôle", ["Toutes"] + list(TYPES_FR.values()), horizontal=True, key="role_regle")
    choix = [r for r in regles if role == "Toutes" or TYPES_FR[ox.lire_regle(r)[0]] == role]
    regle = st.selectbox(
        "Règle", choix, format_func=lambda r: f"{r['id']} — {ox.lire_regle(r)[1]}", key="regle_swrl",
    )
    type_r, lecture = ox.lire_regle(regle)
    evaluee = type_r in ("mode", "repartition")
    with st.container(border=True):
        st.markdown(f"#### {regle['id']} · {TYPES_FR[type_r]}")
        st.markdown(f"**En clair** : {ox.regle_en_phrase(regle)}")
        f1, f2 = st.columns(2)
        with f1:
            st.markdown("**S'applique à**")
            st.markdown(", ".join(ox.CLASSES_FR.get(c, c) for c in regle["classes"]) or "—")
            st.markdown("**Grandeurs lues**")
            st.markdown(
                "\n".join(
                    f"- `{args[-1]}` : {ox._fr(p)} ({_sujet(args[0])})"
                    for p, args in regle["lectures"] if len(args) == 2
                ) or "—"
            )
        with f2:
            st.markdown("**Si**")
            st.markdown("\n".join(f"- `{c}`" for c in ox.conditions_en_clair(regle)) or "- toujours (pas de condition)")
            if regle["calculs"]:
                st.markdown("**Calcule**")
                st.markdown("\n".join(f"- `{c}`" for c in ox.calculs_en_clair(regle)))
            st.markdown("**Alors**")
            st.markdown(
                "\n".join(
                    f"- {ox._fr(p)} ({_sujet(args[0])}) = `{args[1]}`"
                    for p, args in regle["affectations"] if len(args) == 2
                ) or "—"
            )
        st.caption(
            "Évaluée par l'application à chaque instant du cycle."
            if evaluee
            else "Non évaluée pendant la simulation : elle porte sur des grandeurs internes "
            "(courants, tensions) que la simulation ne calcule pas, ou n'a pas de condition."
        )

    with st.expander(f"Vue d'ensemble des {len(regles)} règles"):
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Règle": r["id"],
                        "Rôle": TYPES_FR[ox.lire_regle(r)[0]],
                        "Si": " et ".join(ox.conditions_en_clair(r)) or "toujours",
                        "Alors": ox.lire_regle(r)[1],
                    }
                    for r in regles
                ]
            ).set_index("Règle"),
            width="stretch",
            height=420,
        )
    with st.expander("Signification des variables"):
        st.markdown("\n".join(f"- `{v}` : {s}" for v, s in ox.GLOSSAIRE_VARIABLES.items()))

with onglet_flou:
    st.caption(
        "Ces règles sont définies dans le code du moteur flou, pas dans le fichier OWL ; "
        "elles emploient les concepts de l'ontologie. Le moteur calcule alpha comme la "
        "moyenne de leurs conclusions, pondérée par l'activation de chaque règle : la "
        "contribution de chaque règle à la décision est donc exacte."
    )
    floues = ox.regles_floues()
    rf = st.selectbox("Règle floue", floues, format_func=lambda r: r["libelle"], key="regle_floue")
    with st.container(border=True):
        st.markdown(f"#### {rf['libelle']}")
        st.markdown(f"**Si** {rf['si']}")
        st.markdown(f"**Alors** confier **{nombre(rf['alpha'] * 100, 0)} %** de la puissance à la PB, c'est-à-dire {rf['sens']}.")
        if rf["concepts"]:
            st.markdown(
                "**Concepts de l'ontologie** : "
                + ", ".join(f"{lib} (`{cl}`)" for lib, cl in rf["concepts"])
            )
    st.dataframe(
        pd.DataFrame(
            [
                {"Règle": r["libelle"], "Si": r["si"], "Part de la PB": f"{nombre(r['alpha'] * 100, 0)} %"}
                for r in floues
            ]
        ).set_index("Règle"),
        width="stretch",
    )
    with st.expander("Définition des termes flous"):
        st.markdown("\n".join(f"- **{t}** : {d}" for t, d in ox.termes_flous()))
        st.caption(
            f"Sans aucune règle activée, le moteur applique une répartition par défaut de "
            f"{nombre(core.FUZZY_DEFAULT_ALPHA * 100, 0)} % pour la PB."
        )


# D — Comment l'ontologie intervient dans les modèles ?

st.subheader("D. Comment l'ontologie intervient dans les modèles ?")
USAGES = {
    "EMS_power_limitation": "N'exécute pas l'ontologie, mais prend exactement la décision des règles de répartition R13 à R17.",
    "EMS_fuzzy_logic": "Ses règles floues reposent sur les concepts `SOCState` et `PowerState` ; elles sont écrites dans le code, pas lues dans le fichier OWL.",
    "EMS_MLP": "Aucun usage.",
    "EMS_LSTM": "Aucun usage.",
    "EMS_GNN": "Aucun usage direct ; son graphe reprend les composants du HESS (batteries, convertisseur, moteur, véhicule).",
    "EMS_MLP_neurosymbolic": "Part de la base floue (concepts de l'ontologie) et reçoit des états symboliques (SOC, puissance, convertisseur).",
    "EMS_LSTM_neurosymbolic": "Reçoit quatre états symboliques (forte demande, freinage, demande nulle, convertisseur chargé) en entrée.",
}
st.dataframe(
    pd.DataFrame(
        [{"Stratégie": nom_affichage(c), "Usage de l'ontologie": u} for c, u in USAGES.items()]
    ).set_index("Stratégie"),
    width="stretch",
)

st.markdown("**Les deux façons d'associer l'ontologie à un réseau de neurones**")
ns1, ns2 = st.columns(2)
with ns1:
    with st.container(border=True):
        st.markdown(f"**{nom_affichage('EMS_MLP_neurosymbolic')}** · les règles décident, le réseau corrige")
        st.markdown(
            flux_html(
                [
                    ("Entrées", COULEUR_REFERENCE),
                    ("Base floue", COULEUR_SECONDAIRE),
                    ("Décision symbolique", COULEUR_DECISION),
                    (f"Correction MLP bornée (±{nombre(core.MLP_NS_MAX_DELTA * 100, 0)} pts)", COULEUR_SECONDAIRE),
                    ("Garde-fou R14/R16", COULEUR_REFERENCE),
                    ("Décision finale", COULEUR_DECISION),
                ]
            ),
            unsafe_allow_html=True,
        )
        st.caption(
            "La décision se décompose exactement : part des règles + correction du réseau. "
            f"Sous {nombre(core.MLP_NS_RESERVE_PB_SOC * 100, 0)} % de SOC de la PB, les règles R14 et R16 "
            "reprennent la main en traction pour préserver sa réserve."
        )
with ns2:
    with st.container(border=True):
        st.markdown(f"**{nom_affichage('EMS_LSTM_neurosymbolic')}** · le symbolique informe le réseau")
        st.markdown(
            flux_html(
                [
                    (f"Historique temporel ({core.LSTM_WINDOW} s) + 4 états symboliques", COULEUR_REFERENCE),
                    ("LSTM", COULEUR_SECONDAIRE),
                    ("Décision finale", COULEUR_DECISION),
                ]
            ),
            unsafe_allow_html=True,
        )
        st.caption("Les états symboliques sont des entrées parmi d'autres : leur poids se mesure après coup.")
st.caption(
    "Pour toutes les stratégies, l'ontologie sert aussi à expliquer : chaque décision est "
    "relue avec ses règles dans la page « Pourquoi cette décision ? »."
)


# Tester les règles sur une situation choisie

st.subheader("Tester les règles sur une situation")
st.caption(
    "Choisissez une situation : l'ontologie en déduit l'état de fonctionnement et la "
    "répartition de référence, règle par règle."
)
s1, s2, s3 = st.columns(3)
p_test = s1.slider("Puissance demandée (kW)", -50.0, 50.0, 15.0, 0.5) * 1000.0
soc_eb_test = s2.slider("SOC de la batterie Énergie (%)", 0, 100, 60) / 100.0
soc_pb_test = s3.slider("SOC de la batterie Puissance (%)", 0, 100, 80) / 100.0

etat_test = ox.etat_fonctionnement(p_test)
activees, non_activees, _ = ox.evaluer_regles(p_test, soc_eb_test, soc_pb_test)
repart = ox.repartition_ontologie(p_test, soc_eb_test, soc_pb_test)

g_col, t_col = st.columns([1, 1])
with g_col:
    st.plotly_chart(_graphe_connaissances(etat_test), width="stretch")
with t_col:
    st.markdown(f"État inféré : **{ox.ETATS_ONTOLOGIE[etat_test]}** (`{etat_test}`)")
    for r in sorted((r for r in activees if r["type"] in ("mode", "repartition")), key=lambda r: r["type"]):
        premisses = " et ".join(f"`{d['texte']}`" for d in r["details"])
        st.markdown(f"- **{r['id']}** — si {premisses}, alors {r['lecture']}.")
    if repart is not None:
        a = repart["alpha"]
        st.success(
            f"Répartition de référence (règle {repart['regle']['id']}) : batterie Énergie "
            f"{nombre(p_test * (1 - a) / 1000, 1)} kW, batterie Puissance {nombre(p_test * a / 1000, 1)} kW "
            f"(alpha = {nombre(a * 100, 0)} %)."
        )
    else:
        st.info("Demande quasi nulle : aucune règle de répartition ne s'applique.")

with st.expander("Règles non activées, et pourquoi"):
    for r in non_activees:
        if r["type"] not in ("mode", "repartition"):
            continue
        echecs = [d["texte"] for d in r["details"] if d["ok"] is False]
        st.markdown(f"- **{r['id']}** ({r['lecture']}) : `{' ; '.join(echecs)}` non vérifié")


pied_navigation("vues/3_Ontologie_OntoHESS.py")
