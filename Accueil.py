"""
2SMART — Plateforme d'analyse, de simulation et d'explicabilité pour les
systèmes hybrides de stockage d'énergie (HESS).

Point d'entrée de l'application. La navigation suit le parcours de
l'utilisateur, et non la technologie sous-jacente :

    1. Préparer une simulation
    2. Exécuter une stratégie de gestion d'énergie
    3. Analyser les résultats
    4. Comprendre les décisions de l'intelligence artificielle

Chaque page est implémentée dans le dossier `vues/`. L'interface est en
français ou en anglais (core/i18n.py).
"""

import streamlit as st

from core.i18n import CLE_LANGUE, LANGUES, langue_du_navigateur, lib, tr
from core.navigation import SECTIONS, titre


st.set_page_config(
    page_title="2SMART — HESS",
    layout="wide",
)


with st.sidebar:
    # Langue de l'interface au premier affichage : celle demandée dans le lien
    # (…/?lang=en, pratique pour partager l'application), sinon celle du navigateur.
    if CLE_LANGUE not in st.session_state:
        demandee = str(st.query_params.get("lang", "")).lower()
        st.session_state[CLE_LANGUE] = demandee if demandee in LANGUES else langue_du_navigateur()
    st.radio(
        "🌐 Langue / Language", list(LANGUES), format_func=LANGUES.get, key=CLE_LANGUE, horizontal=True,
    )

    # Identité de l'application.
    st.markdown(
        "<div style='font-size:1.5rem;font-weight:800;letter-spacing:-.5px;"
        "background:linear-gradient(90deg,#3B82F6,#22C55E);-webkit-background-clip:text;"
        "-webkit-text-fill-color:transparent;color:#3B82F6'>2SMART</div>"
        "<div style='color:#94A3B8;font-size:.82rem'>"
        + tr("Gestion intelligente de l'énergie", "Smart energy management")
        + "</div><div style='color:#94A3B8;font-size:.72rem;margin-bottom:.4rem'>Version 2.0</div>",
        unsafe_allow_html=True,
    )


# Menu organisé par sections, libellé par ce que l'utilisateur veut faire.
# La section vide place le tableau de bord tout en haut, sans titre de section.
menu = {
    lib(section): [
        st.Page(page, title=titre(page), default=(page == "vues/1_Accueil.py")) for page in pages
    ]
    for section, pages in SECTIONS
}

st.navigation(menu).run()
