"""
2SMART — Plateforme d'analyse, de simulation et d'explicabilité pour les
systèmes hybrides de stockage d'énergie (HESS).

Point d'entrée de l'application. La navigation suit le parcours de
l'utilisateur, et non la technologie sous-jacente :

    1. Préparer une simulation
    2. Exécuter une stratégie de gestion d'énergie
    3. Analyser les résultats
    4. Comprendre les décisions de l'intelligence artificielle

Chaque page est implémentée dans le dossier `vues/`.
"""

import streamlit as st


st.set_page_config(
    page_title="2SMART — HESS",
    layout="wide",
)


# En-tête de la barre latérale (identité de l'application).
with st.sidebar:
    st.markdown(
        "<div style='font-size:1.5rem;font-weight:800;letter-spacing:-.5px;"
        "background:linear-gradient(90deg,#3B82F6,#22C55E);-webkit-background-clip:text;"
        "-webkit-text-fill-color:transparent;color:#3B82F6'>2SMART</div>"
        "<div style='color:#94A3B8;font-size:.82rem'>Gestion intelligente de l'énergie</div>"
        "<div style='color:#94A3B8;font-size:.72rem;margin-bottom:.4rem'>Version 2.0</div>",
        unsafe_allow_html=True,
    )


# Menu organisé par sections, libellé par ce que l'utilisateur veut faire.
# Les clés deviennent des séparateurs de section dans la barre latérale ;
# la section vide ("") place l'accueil tout en haut, sans titre de section.
menu = {
    "": [
        st.Page("vues/1_Accueil.py", title="🏠 Accueil", default=True),
    ],
    "🚗 Simulation": [
        st.Page("vues/2_Preparation_donnees.py", title="📂 Préparer une simulation"),
        st.Page("vues/8_Simulation_cycle_personnalise.py", title="▶️ Lancer une simulation"),
    ],
    "📊 Résultats": [
        st.Page("vues/5_Comparaison_des_strategies.py", title="⚖️ Comparer les méthodes"),
        st.Page("vues/6_Resultats_et_Analyse.py", title="📈 Explorer les résultats"),
        st.Page("vues/7_Explicabilite.py", title="💡 Pourquoi cette décision ?"),
    ],
    "🧠 Intelligence artificielle": [
        st.Page("vues/9_Architecture_des_modeles.py", title="🧠 Les modèles d'IA"),
        st.Page("vues/3_Ontologie_OntoHESS.py", title="📚 Base de connaissances"),
        st.Page("vues/4_Moteur_Neurosymbolique.py", title="🤖 IA + règles expertes"),
    ],
}

st.navigation(menu).run()
