import streamlit as st


# ============================================================
# Titre + sous-titre
# ============================================================

st.title("2SMART — Semi-Supervised MAnagement of dual battery electric vehicles")

st.markdown(
    "*Plateforme d'analyse, de simulation et d'explicabilité des stratégies de "
    "gestion d'énergie d'un système hybride de stockage (HESS) pour véhicule "
    "électrique.*"
)


# ============================================================
# Chaîne de traitement (pipeline)
# ============================================================

st.subheader("Chaîne de traitement")

etapes = [
    "Cycle de conduite",
    "Prétraitement",
    "EMS (IA + règles)",
    "Filtre physique",
    "Résultats",
    "Explicabilité",
]
cols_pipe = st.columns(len(etapes))
for col, etape in zip(cols_pipe, etapes):
    with col:
        with st.container(border=True):
            st.markdown(f"**{etape}**")


# ============================================================
# Présentation
# ============================================================

st.subheader("Le projet")

st.write(
    "2SMART développe un système de gestion d'énergie pour un véhicule électrique "
    "équipé d'un système hybride de stockage (HESS), combinant une batterie "
    "d'énergie (EB) et une batterie de puissance (PB) via un convertisseur. "
    "L'application permet de simuler, comparer et expliquer plusieurs stratégies "
    "de répartition de la puissance entre ces deux batteries."
)


# ============================================================
# Objectifs
# ============================================================

st.subheader("Objectifs")

obj1, obj2, obj3 = st.columns(3)

with obj1:
    with st.container(border=True):
        st.markdown("**Connaissances expertes**")
        st.write(
            "Structurer les connaissances sur les batteries, le convertisseur et "
            "les conditions de fonctionnement (ontologie et règles)."
        )

with obj2:
    with st.container(border=True):
        st.markdown("**Environnement de simulation**")
        st.write(
            "Préparer un cycle de conduite et simuler le comportement du HESS "
            "(puissance demandée, SOC, puissances EB/PB)."
        )

with obj3:
    with st.container(border=True):
        st.markdown("**Stratégies EMS**")
        st.write(
            "Sept stratégies : règles, logique floue, modèles neuronaux et "
            "variantes neuro-symboliques."
        )

obj4, obj5 = st.columns(2)

with obj4:
    with st.container(border=True):
        st.markdown("**Couplage règles expertes — IA**")
        st.write(
            "Combiner apprentissage et connaissances physiques pour une décision "
            "plus sûre, interprétable et conforme aux contraintes du système."
        )

with obj5:
    with st.container(border=True):
        st.markdown("**Validation et comparaison**")
        st.write(
            "Comparer les stratégies sur des indicateurs communs : répartition de "
            "puissance, SOC, respect des contraintes physiques."
        )


# ============================================================
# Variable de décision alpha
# ============================================================

st.subheader("Variable de décision")

st.write(
    "La variable de décision est le coefficient `alpha(t)` : la fraction de la "
    "puissance demandée attribuée à la batterie de puissance. À chaque instant, "
    "une stratégie EMS propose une valeur de `alpha`."
)

eq1, eq2 = st.columns(2)
with eq1:
    st.latex(r"P_{PB} = \alpha \times P_{dem}")
with eq2:
    st.latex(r"P_{EB} = (1 - \alpha) \times P_{dem}")


# ============================================================
# Architecture de l'application
# ============================================================

st.subheader("Architecture de l'application")

m1, m2, m3 = st.columns(3)

with m1:
    with st.container(border=True):
        st.markdown("**Données et Prétraitement**")
        st.write("Importer un cycle, calculer forces, puissance et courants.")
    with st.container(border=True):
        st.markdown("**Base de connaissances HESS**")
        st.write("Règles et connaissances expertes structurant le fonctionnement du HESS.")

with m2:
    with st.container(border=True):
        st.markdown("**Raisonnement neuro-symbolique**")
        st.write("Décision à un instant t : batterie active, choix du modèle, explication.")
    with st.container(border=True):
        st.markdown("**Comparaison des stratégies**")
        st.write("Classement des stratégies selon plusieurs critères.")

with m3:
    with st.container(border=True):
        st.markdown("**Analyse détaillée**")
        st.write("Évolution des SOC, puissances, courants et indicateurs.")
    with st.container(border=True):
        st.markdown("**Pourquoi cette décision ?**")
        st.write("Justification de chaque décision EMS en langage physique.")


# ============================================================
# Deux modes d'utilisation
# ============================================================

st.subheader("Deux modes d'utilisation")

mode1, mode2 = st.columns(2)

with mode1:
    with st.container(border=True):
        st.markdown("**Mode démonstration**")
        st.write(
            "- Résultats précalculés\n"
            "- Chargement instantané\n"
            "- Comparaison des stratégies\n"
            "- Analyse et explicabilité\n"
            "- Idéal pour une soutenance"
        )

with mode2:
    with st.container(border=True):
        st.markdown("**Mode simulation personnalisée**")
        st.write(
            "- Import d'un nouveau cycle\n"
            "- Simulation complète\n"
            "- Temps de calcul plus long\n"
            "- Analyse d'un cycle inédit"
        )


# ============================================================
# Conclusion + accès
# ============================================================

st.info(
    "L'application permet de comparer sept stratégies EMS, d'analyser leurs "
    "performances, d'expliquer chaque décision et de vérifier leur compatibilité "
    "avec les contraintes physiques du système HESS."
)

acces1, acces2 = st.columns(2)
with acces1:
    if st.button("Explorer les résultats", type="primary", use_container_width=True):
        st.switch_page("vues/5_Comparaison_des_strategies.py")
with acces2:
    if st.button("Préparation des données", use_container_width=True):
        st.switch_page("vues/2_Preparation_donnees.py")
