"""
Point d'entrée de l'application 2SMART.

Ce fichier est un ROUTEUR : il définit le menu de navigation, organisé selon le
parcours naturel d'un physicien (préparer, analyser, comprendre), avec des
sections. Le contenu de chaque page est dans vues/.
"""

import streamlit as st


st.set_page_config(
    page_title="2SMART — HESS",
    layout="wide",
)


# En-tête de la barre latérale (identité de l'application).
with st.sidebar:
    st.markdown(
        "<div style='font-size:1.4rem;font-weight:800;letter-spacing:-.5px;"
        "background:linear-gradient(90deg,#3B82F6,#22C55E);-webkit-background-clip:text;"
        "-webkit-text-fill-color:transparent;color:#3B82F6'>2SMART</div>"
        "<div style='color:#94A3B8;font-size:.8rem;margin-bottom:.4rem'>"
        "Plateforme HESS</div>",
        unsafe_allow_html=True,
    )


# Menu organisé par sections. Les clés du dictionnaire deviennent des
# séparateurs de section dans la barre latérale ; la section vide ("") place
# l'accueil tout en haut, sans titre de section.
menu = {
    "": [
        st.Page("vues/1_Accueil.py", title="Accueil", default=True),
    ],
    "Simulation": [
        st.Page("vues/2_Preparation_donnees.py", title="Préparation"),
        st.Page("vues/8_Simulation_cycle_personnalise.py", title="Nouvelle simulation"),
    ],
    "Analyse": [
        st.Page("vues/5_Comparaison_des_strategies.py", title="Comparaison des stratégies"),
        st.Page("vues/6_Resultats_et_Analyse.py", title="Analyse détaillée"),
        st.Page("vues/7_Explicabilite.py", title="Pourquoi cette décision ?"),
    ],
    "Intelligence": [
        st.Page("vues/9_Architecture_des_modeles.py", title="Architecture des modèles"),
        st.Page("vues/3_Ontologie_OntoHESS.py", title="Base de connaissances HESS"),
        st.Page("vues/4_Moteur_Neurosymbolique.py", title="Raisonnement neuro-symbolique"),
    ],
}

st.navigation(menu).run()
