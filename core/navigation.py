"""
core/navigation.py — Pied de page de navigation cohérent pour toutes les pages.

Au lieu de boutons ad hoc mal alignés en bas de chaque page, on affiche un
pied uniforme : « page précédente » à gauche, « page suivante » à droite,
selon l'ordre logique de l'application.
"""

import streamlit as st


# Ordre de navigation de l'application. Accueil.py est l'entrée (hors dossier
# vues/) mais fait partie du parcours, donc inclus ici.
ORDRE_PAGES = [
    ("🏠 Accueil", "vues/1_Accueil.py"),
    ("📂 Préparer une simulation", "vues/2_Preparation_donnees.py"),
    ("▶️ Lancer une simulation", "vues/8_Simulation_cycle_personnalise.py"),
    ("⚖️ Comparer les méthodes", "vues/5_Comparaison_des_strategies.py"),
    ("📊 Analyse instantanée", "vues/4_Moteur_Neurosymbolique.py"),
    ("📈 Explorer les résultats", "vues/6_Resultats_et_Analyse.py"),
    ("💡 Pourquoi cette décision ?", "vues/7_Explicabilite.py"),
    ("🧠 Les modèles d'IA", "vues/9_Architecture_des_modeles.py"),
    ("📚 Base de connaissances", "vues/3_Ontologie_OntoHESS.py"),
]


def pied_navigation(cible_courante: str):
    """Affiche un pied de navigation cohérent (précédent / suivant).

    cible_courante : la cible `switch_page` de la page appelante,
    par exemple "vues/5_Comparaison_des_strategies.py".
    """
    cibles = [c for _, c in ORDRE_PAGES]
    try:
        i = cibles.index(cible_courante)
    except ValueError:
        return

    st.divider()
    col_prec, col_suiv = st.columns(2)

    if i > 0:
        libelle_prec, cible_prec = ORDRE_PAGES[i - 1]
        with col_prec:
            if st.button(
                f"Précédent : {libelle_prec}",
                use_container_width=True,
                key="nav_precedent",
            ):
                st.switch_page(cible_prec)

    if i < len(ORDRE_PAGES) - 1:
        libelle_suiv, cible_suiv = ORDRE_PAGES[i + 1]
        with col_suiv:
            if st.button(
                f"Suivant : {libelle_suiv}",
                use_container_width=True,
                type="primary",
                key="nav_suivant",
            ):
                st.switch_page(cible_suiv)
