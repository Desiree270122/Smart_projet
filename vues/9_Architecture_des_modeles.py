"""
Fonctionnement des stratégies EMS : comment chacune transforme l'état du HESS
(demande, états de charge) en une répartition de la puissance, avec quelles
connaissances, et ce qu'elle a appris. Les textes viennent de core/presentation.py,
partagé avec l'application web.
"""

import sys
from pathlib import Path

DOSSIER_PROJET = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DOSSIER_PROJET))

import pandas as pd
import streamlit as st

import ems_core as core
from core import presentation
from core.i18n import tr
from core.navigation import pied_navigation
from core.resultats import famille, nom_affichage
from core.style import COULEUR_DECISION, COULEUR_DEMANDE, COULEUR_REFERENCE, COULEUR_SECONDAIRE, flux_html

COULEURS_ROLES = {
    presentation.DEMANDE: COULEUR_DEMANDE, presentation.REFERENCE: COULEUR_REFERENCE,
    presentation.SECONDAIRE: COULEUR_SECONDAIRE, presentation.DECISION: COULEUR_DECISION,
}


def _chaine(etapes):
    return flux_html([(texte, COULEURS_ROLES[role]) for texte, role in etapes])


st.title(tr("🧠 Fonctionnement des stratégies EMS", "🧠 How the EMS strategies work"))
st.caption(tr(
    "Comment chaque stratégie transforme l'état du HESS en une répartition de la puissance.",
    "How each strategy turns the state of the HESS into a power split.",
))


# Le principe commun à toutes les stratégies

st.subheader(tr("Le principe commun", "The common principle"))
st.markdown(_chaine(presentation.principe_commun()), unsafe_allow_html=True)
eq1, eq2 = st.columns(2)
eq1.latex(r"P_{PB} = \alpha \times P_{dem}")
eq2.latex(r"P_{EB} = (1 - \alpha) \times P_{dem}")
st.caption(tr(
    "Toutes les stratégies donnent la même grandeur, alpha : la part de la puissance confiée à la "
    "batterie Puissance. Un filtre de sécurité vérifie ensuite les limites de courant, de puissance "
    "et de SOC, et corrige alpha si nécessaire.",
    "All strategies output the same quantity, alpha: the share of power assigned to the Power "
    "battery. A safety filter then checks the current, power and SOC limits, and corrects alpha if "
    "necessary.",
))

with st.expander(tr("Architecture électrique du HESS (convertisseur à puissance partielle)",
                    "Electrical architecture of the HESS (partial-power converter)")):
    st.markdown(presentation.architecture_electrique())
    st.latex(
        r"P_{conv} = (V_{EB} - V_{PB})\,I_{EB} = P_{EB}\,\frac{V_{EB} - V_{PB}}{V_{EB}}"
        r"\qquad P_{dem} = P_{EB} + P_{PB}"
    )
    st.caption(presentation.REFERENCE_ARTICLE)


FICHES = presentation.fiches_strategies()


def _carte(cle):
    fiche = FICHES[cle]
    with st.container(border=True):
        st.markdown(f"#### {nom_affichage(cle)}")
        st.caption(f"{fiche['nature']}  ·  {tr('famille', 'family')} « {famille(cle)} »")
        st.markdown(_chaine(fiche["flux"]), unsafe_allow_html=True)
        with st.expander(tr("Détails", "Details")):
            titres = (
                tr("À quoi elle sert", "What it is for"), tr("Comment elle fonctionne", "How it works"),
                tr("Ce qu'elle a appris à reproduire", "What it learned to reproduce"), tr("Grandeurs utilisées", "Quantities used"),
            )
            st.markdown(f"**{titres[0]}** — {fiche['role']}")
            st.markdown(f"**{titres[1]}**\n" + "\n".join(f"- {point}" for point in fiche["fonctionnement"]))
            st.markdown(f"**{titres[2]}** — {fiche['appris']}")
            st.markdown(f"**{titres[3]}** — {fiche['grandeurs']}")


st.subheader(tr("Les sept stratégies", "The seven strategies"))
classiques = [c for c in core.MODEL_ORDER if c in FICHES and "neurosymbolic" not in c]
for i in range(0, len(classiques), 2):
    colonnes = st.columns(2)
    for col, cle in zip(colonnes, classiques[i:i + 2]):
        with col:
            _carte(cle)


# Les deux stratégies neuro-symboliques, au cœur du projet

st.subheader(tr("Pourquoi deux stratégies neuro-symboliques ?", "Why two neuro-symbolic strategies?"))
st.markdown(tr(
    "Les deux combinent connaissances expertes et apprentissage, mais pas au même endroit de la "
    "chaîne de décision. Les comparer montre ce que change la place donnée aux connaissances expertes.",
    "Both combine expert knowledge and learning, but not at the same place in the decision chain. "
    "Comparing them shows what the place given to expert knowledge changes.",
))
ns1, ns2 = st.columns(2)
with ns1:
    _carte("EMS_MLP_neurosymbolic")
    st.info(tr(
        "**NS-MLP** — les connaissances expertes sont le **socle de la décision** : les règles floues "
        "proposent une répartition, le réseau n'y ajoute qu'une correction limitée, et un garde-fou de "
        "l'ontologie (règles R14/R16) protège la réserve de la batterie Puissance.",
        "**NS-MLP** — expert knowledge is the **basis of the decision**: the fuzzy rules propose a "
        "split, the network only adds a limited correction, and an ontology safeguard (rules R14/R16) "
        "protects the Power battery's reserve.",
    ))
with ns2:
    _carte("EMS_LSTM_neurosymbolic")
    st.info(tr(
        "**NS-LSTM** — le réseau exploite **le passé récent du cycle** et reçoit des états déduits par "
        "l'ontologie comme informations supplémentaires. Les connaissances expertes éclairent la "
        "décision sans la structurer.",
        "**NS-LSTM** — the network uses **the recent past of the cycle** and receives states inferred "
        "by the ontology as additional information. Expert knowledge informs the decision without "
        "structuring it.",
    ))

st.divider()


# Tableau de synthèse

st.subheader(tr("Les stratégies en un tableau", "The strategies in one table"))
SYNTHESE = presentation.synthese_strategies()
col_strategie = tr("Stratégie", "Strategy")
st.dataframe(
    pd.DataFrame([
        {
            col_strategie: nom_affichage(cle),
            tr("Famille", "Family"): famille(cle),
            tr("Apprentissage", "Learning"): SYNTHESE[cle]["apprentissage"],
            tr("Ce qu'elle a appris à reproduire", "What it learned to reproduce"): SYNTHESE[cle]["appris"],
            tr("Mémoire du passé", "Memory of the past"): SYNTHESE[cle]["memoire"],
            tr("Usage de l'ontologie", "Use of the ontology"): SYNTHESE[cle]["ontologie"],
            tr("Comment la décision se lit", "How the decision can be read"): SYNTHESE[cle]["lecture"],
        }
        for cle in core.MODEL_ORDER if cle in SYNTHESE
    ]).set_index(col_strategie),
    width="stretch",
)
st.caption(tr(
    "Les stratégies à apprentissage ont été réglées à l'avance, sur la première moitié du cycle "
    "Artemis ; l'application rejoue leurs décisions. Le détail de l'ontologie et de ses règles est "
    "dans « Base de connaissances ».",
    "The learning-based strategies were tuned beforehand, on the first half of the Artemis cycle; "
    "the app replays their decisions. The details of the ontology and its rules are in “Knowledge base”.",
))


pied_navigation("vues/9_Architecture_des_modeles.py")
