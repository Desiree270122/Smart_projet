"""
Point d'entrée de l'application 2SMART.

Ce fichier est un ROUTEUR : il définit le menu de navigation, organisé selon le
parcours naturel d'un physicien (préparer, analyser, comprendre), avec des
sections. Le contenu de chaque page est dans pages/.
"""

import streamlit as st


st.set_page_config(
    page_title="2SMART — HESS",
    layout="wide",
)


# Menu organisé par sections. Les clés du dictionnaire deviennent des
# séparateurs de section dans la barre latérale ; la section vide ("") place
# l'accueil tout en haut, sans titre de section.
menu = {
    "": [
        st.Page("pages/1_Accueil.py", title="Accueil", default=True),
    ],
    "Simulation": [
        st.Page("pages/2_Preparation_donnees.py", title="Préparation"),
        st.Page("pages/8_Simulation_cycle_personnalise.py", title="Nouvelle simulation"),
    ],
    "Analyse": [
        st.Page("pages/5_Comparaison_des_strategies.py", title="Comparaison des stratégies"),
        st.Page("pages/6_Resultats_et_Analyse.py", title="Analyse détaillée"),
        st.Page("pages/7_Explicabilite.py", title="Pourquoi cette décision ?"),
    ],
    "Intelligence": [
        st.Page("pages/3_Ontologie_OntoHESS.py", title="Base de connaissances HESS"),
        st.Page("pages/4_Moteur_Neurosymbolique.py", title="Raisonnement neuro-symbolique"),
    ],
}

st.navigation(menu).run()
