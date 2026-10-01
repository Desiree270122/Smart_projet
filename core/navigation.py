"""
core/navigation.py — Titres des pages et pied de navigation commun.

Le menu (Accueil.py) et le pied « page précédente / page suivante » de chaque
page utilisent les mêmes titres, dans la langue choisie.
"""

import streamlit as st

from core.i18n import lib, tr


# Titre de chaque page (français, anglais), dans l'ordre du parcours.
TITRES = {
    "vues/1_Accueil.py": ("🏠 Tableau de bord", "🏠 Dashboard"),
    "vues/2_Preparation_donnees.py": ("📂 Préparer une simulation", "📂 Prepare a simulation"),
    "vues/8_Simulation_cycle_personnalise.py": ("▶️ Lancer une simulation", "▶️ Run a simulation"),
    "vues/6_Resultats_et_Analyse.py": ("📈 Résultats de simulation", "📈 Simulation results"),
    "vues/5_Comparaison_des_strategies.py": ("📊 Comparaison des stratégies EMS", "📊 EMS strategy comparison"),
    "vues/7_Explicabilite.py": ("💡 Pourquoi cette décision ?", "💡 Why this decision?"),
    "vues/9_Architecture_des_modeles.py": ("🧠 Fonctionnement des stratégies EMS", "🧠 How the EMS strategies work"),
    "vues/3_Ontologie_OntoHESS.py": ("📚 Base de connaissances", "📚 Knowledge base"),
}

# Sections du menu : (titre, pages).
SECTIONS = [
    (("", ""), ["vues/1_Accueil.py"]),
    (("📈 Simulation", "📈 Simulation"), [
        "vues/2_Preparation_donnees.py", "vues/8_Simulation_cycle_personnalise.py", "vues/6_Resultats_et_Analyse.py",
    ]),
    (("⚖️ Comparaison", "⚖️ Comparison"), ["vues/5_Comparaison_des_strategies.py"]),
    (("🔍 Explication", "🔍 Explanation"), [
        "vues/7_Explicabilite.py", "vues/9_Architecture_des_modeles.py", "vues/3_Ontologie_OntoHESS.py",
    ]),
]

ORDRE_PAGES = list(TITRES)


def titre(page: str) -> str:
    return lib(TITRES[page])


def pied_navigation(cible_courante: str):
    """Pied de navigation : page précédente à gauche, page suivante à droite.

    cible_courante : chemin de la page appelante, par exemple
    "vues/5_Comparaison_des_strategies.py".
    """
    try:
        i = ORDRE_PAGES.index(cible_courante)
    except ValueError:
        return

    st.divider()
    col_prec, col_suiv = st.columns(2)

    if i > 0:
        cible_prec = ORDRE_PAGES[i - 1]
        with col_prec:
            if st.button(tr("Précédent : {t}", "Previous: {t}", t=titre(cible_prec)), width="stretch", key="nav_precedent"):
                st.switch_page(cible_prec)

    if i < len(ORDRE_PAGES) - 1:
        cible_suiv = ORDRE_PAGES[i + 1]
        with col_suiv:
            if st.button(
                tr("Suivant : {t}", "Next: {t}", t=titre(cible_suiv)), width="stretch", type="primary", key="nav_suivant",
            ):
                st.switch_page(cible_suiv)
