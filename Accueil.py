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

from core.resultats import CLE_CYCLE, CLE_CYCLE_DEMANDE, cycles_disponibles


st.set_page_config(
    page_title="2SMART — HESS",
    layout="wide",
)


# Cycle dont toutes les pages affichent les résultats. Une page peut demander
# un autre cycle (après une simulation) : la demande est appliquée ici, avant
# la création du widget, comme l'exige Streamlit.
with st.sidebar:
    if CLE_CYCLE_DEMANDE in st.session_state:
        st.session_state[CLE_CYCLE] = st.session_state.pop(CLE_CYCLE_DEMANDE)
    _cycles = cycles_disponibles(st)
    if _cycles:
        if st.session_state.get(CLE_CYCLE) not in _cycles:
            st.session_state[CLE_CYCLE] = next(iter(_cycles))
        st.selectbox(
            "Cycle étudié", list(_cycles), format_func=_cycles.get, key=CLE_CYCLE,
            help="Toutes les pages d'analyse affichent les résultats de ce cycle.",
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
        st.Page("vues/1_Accueil.py", title="🏠 Tableau de bord", default=True),
    ],
    "📈 Simulation": [
        st.Page("vues/2_Preparation_donnees.py", title="📂 Préparer une simulation"),
        st.Page("vues/8_Simulation_cycle_personnalise.py", title="▶️ Lancer une simulation"),
        st.Page("vues/6_Resultats_et_Analyse.py", title="📈 Résultats de simulation"),
    ],
    "⚖️ Comparaison": [
        st.Page("vues/5_Comparaison_des_strategies.py", title="📊 Comparaison des stratégies EMS"),
    ],
    "🔍 Explication": [
        st.Page("vues/7_Explicabilite.py", title="💡 Pourquoi cette décision ?"),
        st.Page("vues/9_Architecture_des_modeles.py", title="🧠 Architecture des stratégies EMS"),
        st.Page("vues/3_Ontologie_OntoHESS.py", title="📚 Base de connaissances"),
    ],
}

st.navigation(menu).run()
