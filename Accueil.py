"""
Accueil.py

Page d’accueil de l’application 2SMART.
Cette page présente l’objectif de l’application, le système HESS
et la navigation vers les principales fonctionnalités.
"""

import streamlit as st


# ============================================================
# Configuration de la page
# ============================================================

st.set_page_config(
    page_title="2SMART — Accueil",
    layout="wide",
)


# ============================================================
# Titre principal
# ============================================================

st.title("2SMART — Semi-Supervised MAnagement of dual battery electric vehicles")

st.write(
    "Ce projet vise à développer un système de gestion d’énergie pour un "
    "véhicule électrique équipé d’un système hybride de stockage "
    "d’énergie HESS. Cette application permet de superviser, simuler et "
    "comparer plusieurs stratégies EMS. Elle intègre les modèles entraînés, "
    "les règles expertes et les contraintes physiques afin d’analyser la "
    "répartition de puissance entre la batterie d’énergie et la batterie "
    "de puissance."
)


# ============================================================
# Objectifs de l'application
# ============================================================

st.subheader("Objectifs de l’application")

st.write(
    "L’application a été conçue pour répondre aux objectifs du projet 2SMART "
    "en fournissant un environnement de simulation, de supervision et de "
    "validation des stratégies de gestion d’énergie appliquées à un système "
    "hybride de stockage d’énergie HESS."
)

obj1, obj2, obj3 = st.columns(3)

with obj1:
    with st.container(border=True):
        st.markdown("**1. Intégration de l’ontologie**")
        st.write(
            "Exploiter les règles expertes issues de l’ontologie afin de "
            "structurer les connaissances sur les batteries, le convertisseur, "
            "la demande de puissance et les conditions de fonctionnement."
        )

with obj2:
    with st.container(border=True):
        st.markdown("**2. Environnement de simulation**")
        st.write(
            "Importer un cycle de conduite, préparer les données et simuler "
            "le comportement du système HESS à partir des variables disponibles, "
            "notamment la puissance demandée, les SOC et les puissances EB/PB."
        )

with obj3:
    with st.container(border=True):
        st.markdown("**3. Intégration des modèles entraînés**")
        st.write(
            "Charger et tester les stratégies EMS développées, notamment la "
            "logique floue, les modèles classiques et les variantes "
            "neurosymboliques."
        )


obj4, obj5 = st.columns(2)

with obj4:
    with st.container(border=True):
        st.markdown("**4. Couplage règles expertes — IA**")
        st.write(
            "Analyser l’apport des règles physiques et symboliques dans les "
            "décisions proposées par les modèles afin d’obtenir une stratégie "
            "plus sûre, interprétable et cohérente avec les contraintes du système."
        )

with obj5:
    with st.container(border=True):
        st.markdown("**5. Validation et comparaison**")
        st.write(
            "Comparer les différentes stratégies de gestion d’énergie à partir "
            "d’indicateurs communs : répartition de puissance, évolution des SOC, "
            "respect des contraintes physiques et comportement global du HESS."
        )
# ============================================================
# Variable de décision
# ============================================================

st.subheader("Variable de décision")

st.write(
    "La variable de décision principale est le coefficient `alpha(t)`. "
    "Il représente la fraction de la puissance demandée attribuée à la "
    "batterie de puissance. À chaque instant, une stratégie EMS propose "
    "une valeur de `alpha` afin de répartir la demande de puissance entre "
    "les deux batteries."
)

eq1, eq2 = st.columns(2)

with eq1:
    st.latex(r"P_{PB} = \alpha \times P_{dem}")

with eq2:
    st.latex(r"P_{EB} = (1 - \alpha) \times P_{dem}")


# ============================================================
# Modules disponibles
# ============================================================

st.subheader("Modules disponibles")

m1, m2, m3 = st.columns(3)

with m1:
    with st.container(border=True):
        st.markdown("**Préparation des données**")
        st.write(
            "Importer, vérifier et préparer le cycle de conduite utilisé "
            "pour les simulations."
        )

    with st.container(border=True):
        st.markdown("**Simulation globale**")
        st.write(
            "Appliquer une stratégie EMS sur l’ensemble du cycle et calculer "
            "`alpha`, `P_EB` et `P_PB`."
        )

with m2:
    with st.container(border=True):
        st.markdown("**Évolution du SOC**")
        st.write(
            "Analyser l’évolution de `SOC_EB` et `SOC_PB`, puis détecter les "
            "éventuelles violations des limites physiques."
        )

    with st.container(border=True):
        st.markdown("**Analyse instantanée**")
        st.write(
            "Étudier la décision prise par le modèle à un instant précis "
            "du cycle de conduite."
        )

with m3:
    with st.container(border=True):
        st.markdown("**Comparaison et optimisation**")
        st.write(
            "Comparer les performances des différentes stratégies EMS "
            "à partir d’indicateurs communs."
        )

    with st.container(border=True):
        st.markdown("**Explicabilité, ontologie et export**")
        st.write(
            "Expliquer les décisions des modèles, présenter les règles expertes "
            "et exporter les résultats pour une analyse externe."
        )




# ============================================================
# Démarrage
# ============================================================

st.subheader("Démarrage")

if "resultats_simulation" not in st.session_state:
    st.info(
        "Aucune simulation disponible pour le moment. "
        "Veuillez commencer par la préparation des données afin de charger "
        "et valider un cycle de conduite."
    )

    st.page_link(
        "pages/2_Preparation_donnees.py",
        label="Aller à la préparation des données",
    )

else:
    st.success(
        "Une simulation est déjà disponible. Vous pouvez consulter les résultats "
        "dans les pages Simulation globale, Évolution du SOC, Analyse instantanée "
        "ou Comparaison et optimisation."
    )