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
        st.markdown("**2 · Données & Prétraitement**")
        st.write(
            "Importer un cycle de conduite, calculer forces, puissance et "
            "courants, et préparer les données pour les modèles."
        )

    with st.container(border=True):
        st.markdown("**3 · Ontologie OntoHESS**")
        st.write(
            "Classes, propriétés, règles et connaissances expertes "
            "structurant le fonctionnement du HESS."
        )

with m2:
    with st.container(border=True):
        st.markdown("**4 · Moteur Neuro-Symbolique**")
        st.write(
            "À un instant t : puissance demandée, batterie(s) active(s), "
            "décision du modèle et explication."
        )

    with st.container(border=True):
        st.markdown("**5 · Comparaison des stratégies**")
        st.write(
            "Classement des stratégies selon plusieurs critères et "
            "désignation de la meilleure — résultats précalculés."
        )

with m3:
    with st.container(border=True):
        st.markdown("**6 · Résultats & Analyse**")
        st.write(
            "Évolution des SOC et des puissances, alpha dans le temps, "
            "violations et corrections."
        )

    with st.container(border=True):
        st.markdown("**7 · Explicabilité**")
        st.write(
            "Logique floue, états symboliques, filtre physique, alpha "
            "proposé vs appliqué : pourquoi le modèle décide ainsi."
        )




# ============================================================
# Démarrage
# ============================================================

st.subheader("Mode de fonctionnement")

mode1, mode2 = st.columns(2)

with mode1:
    with st.container(border=True):
        st.markdown("**Mode démonstration (par défaut)**")
        st.write(
            "Les pages Comparaison, Résultats et Explicabilité affichent des "
            "résultats **précalculés** : chargement **instantané**, idéal pour "
            "la soutenance."
        )

with mode2:
    with st.container(border=True):
        st.markdown("**Mode cycle personnalisé**")
        st.write(
            "La page *Simulation — cycle personnalisé* relance une simulation "
            "complète sur ton propre cycle : **plus long** (plusieurs minutes)."
        )


st.subheader("Synthèse")

try:
    from core.resultats import (
        charger_reference,
        calculer_metriques,
        meilleure_strategie,
        nom_affichage,
    )

    _donnees = charger_reference()
    _metriques = calculer_metriques(_donnees)
    _meta = _donnees["meta"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Meilleure — coût", nom_affichage(meilleure_strategie(_metriques, "Coût énergétique")[0]))
    c2.metric("Meilleure — sécurité", nom_affichage(meilleure_strategie(_metriques, "Sécurité physique")[0]))
    c3.metric("Meilleure — équilibre SOC", nom_affichage(meilleure_strategie(_metriques, "Équilibre EB/PB")[0]))

    st.caption(
        f"Cycle de référence : {_meta['nb_points']:,} échantillons — "
        f"{len(_metriques)} stratégies comparées.".replace(",", " ")
    )
    st.page_link(
        "pages/5_Comparaison_des_strategies.py",
        label="→ Voir la comparaison complète",
    )

except FileNotFoundError:
    st.info(
        "Résultats de référence non encore générés. Lance une fois, hors-ligne : "
        "`python scripts/run_simulations.py` — ou prépare un cycle et utilise la "
        "simulation personnalisée."
    )
    st.page_link(
        "pages/2_Preparation_donnees.py",
        label="Aller à la préparation des données",
    )