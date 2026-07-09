
import sys
from pathlib import Path

DOSSIER_PROJET = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DOSSIER_PROJET))

import numpy as np
import pandas as pd
import streamlit as st

try:
    from rdflib import Graph, RDF, RDFS, OWL, URIRef, Literal
    RDFLIB_DISPONIBLE = True
except Exception:
    RDFLIB_DISPONIBLE = False
    Graph = None
    RDF = None
    RDFS = None
    OWL = None
    URIRef = None
    Literal = None

from ems_core import (
    compute_symbolic_states,
    RULE_LABELS_FR,
    MODEL_CONSTRUCTION_DETAILED,
    MODEL_DISPLAY_NAMES,
    MODEL_ORDER,
    alpha_fuzzy_calc,
)


# ============================================================
# Configuration Streamlit
# ============================================================

st.set_page_config(
    page_title="2SMART — Ontologie et règles",
    layout="wide",
)

st.title("Ontologie et règles")

st.write(
    "Cette page présente la couche symbolique du projet 2SMART. Elle montre "
    "comment les grandeurs physiques du HESS sont formalisées dans l’ontologie, "
    "traduites en états symboliques, puis reliées aux règles floues et aux "
    "stratégies EMS."
)

st.info(
    "Convention utilisée : `alpha_PB = P_PB / P_dem`. "
    "`alpha_PB = 0` signifie que la PB ne participe pas et que l’EB couvre "
    "toute la demande. `alpha_PB = 1` signifie que la PB couvre toute la demande."
)

with st.expander("Précision méthodologique"):
    st.write(
        "L’ontologie OWL est utilisée comme support de formalisation des "
        "connaissances physiques : concepts, relations, constantes, états "
        "symboliques et règles. La simulation Streamlit n’interroge pas un "
        "moteur d’inférence OWL/SWRL à chaque pas de temps ; elle utilise les "
        "fonctions Python et les règles floues dérivées de cette formalisation."
    )


# ============================================================
# Fonctions de recherche et de lecture OWL
# ============================================================

def trouver_fichier_ontologie():
    """
    Recherche le fichier OWL de l’ontologie HESS.

    Chemin principal attendu :
    Projet_Artemis2/ontologies/OntoHESS2.owl
    """

    chemins_possibles = [
        DOSSIER_PROJET / "ontologies" / "OntoHESS2.owl",
        DOSSIER_PROJET / "ontology" / "OntoHESS2.owl",
        DOSSIER_PROJET / "ontologie" / "OntoHESS2.owl",
        DOSSIER_PROJET / "OntoHESS2.owl",
    ]

    for chemin in chemins_possibles:
        if chemin.exists():
            return chemin

    return None


def nom_local(objet):
    """
    Transforme une URI RDF en nom court lisible.
    """
    texte = str(objet)

    if "#" in texte:
        return texte.split("#")[-1]

    if "/" in texte:
        return texte.rstrip("/").split("/")[-1]

    return texte


def label_ou_nom(graphe, entite):
    """
    Retourne le label RDFS d’une entité si disponible, sinon son nom local.
    """
    for label in graphe.objects(entite, RDFS.label):
        return str(label)

    return nom_local(entite)


def commentaire(graphe, entite):
    """
    Retourne le commentaire RDFS d’une entité si disponible.
    """
    for comment in graphe.objects(entite, RDFS.comment):
        return str(comment)

    return ""


def noms_uris_visibles(elements):
    """
    Nettoie les domaines, ranges et relations OWL.

    Les identifiants du type N935b5d63... sont des nœuds anonymes RDF
    générés automatiquement pour représenter des restrictions OWL.
    Ils ne sont pas utiles pour la lecture de la page, donc on les masque.
    """
    noms = []

    for element in elements:
        if isinstance(element, URIRef):
            nom = nom_local(element)

            if nom and not nom.startswith("N"):
                noms.append(nom)

    return sorted(set(noms))


def joindre_noms_uris(elements):
    """
    Joint les noms lisibles des URI RDF.
    """
    noms = noms_uris_visibles(elements)

    if noms:
        return ", ".join(noms)

    return "—"


def charger_graphe_owl(chemin_owl):
    """
    Charge le fichier OWL avec rdflib.
    """
    if not RDFLIB_DISPONIBLE:
        return None

    graphe = Graph()
    graphe.parse(str(chemin_owl))
    return graphe


def extraire_ontologie(graphe):
    """
    Extrait l’IRI principale de l’ontologie.
    """
    lignes = []

    for sujet in graphe.subjects(RDF.type, OWL.Ontology):
        lignes.append(
            {
                "Élément": "Ontology",
                "IRI / Nom": str(sujet),
                "Label": label_ou_nom(graphe, sujet),
                "Commentaire": commentaire(graphe, sujet),
            }
        )

    return pd.DataFrame(lignes)


def extraire_classes(graphe):
    """
    Extrait les classes OWL/RDFS.
    """
    classes = set(graphe.subjects(RDF.type, OWL.Class))
    classes.update(set(graphe.subjects(RDF.type, RDFS.Class)))

    lignes = []

    for classe in sorted(classes, key=lambda x: nom_local(x)):
        if isinstance(classe, URIRef):
            lignes.append(
                {
                    "Classe": nom_local(classe),
                    "Label": label_ou_nom(graphe, classe),
                    "Commentaire": commentaire(graphe, classe),
                    "IRI": str(classe),
                }
            )

    return pd.DataFrame(lignes)


def extraire_proprietes_objet(graphe):
    """
    Extrait les propriétés objet de l’ontologie.

    Les domaines/ranges anonymes liés aux restrictions OWL sont masqués pour
    éviter l’affichage d’identifiants techniques du type N935b5d63...
    """
    props = set(graphe.subjects(RDF.type, OWL.ObjectProperty))

    lignes = []

    for prop in sorted(props, key=lambda x: nom_local(x)):
        domaines = joindre_noms_uris(graphe.objects(prop, RDFS.domain))
        ranges = joindre_noms_uris(graphe.objects(prop, RDFS.range))

        lignes.append(
            {
                "Relation objet": nom_local(prop),
                "Domaine lisible": domaines,
                "Range lisible": ranges,
                "Commentaire": commentaire(graphe, prop),
            }
        )

    return pd.DataFrame(lignes)


def extraire_proprietes_donnees(graphe):
    """
    Extrait les propriétés de données de l’ontologie.

    Les domaines anonymes liés aux restrictions OWL sont masqués pour garder
    uniquement les classes lisibles.
    """
    props = set(graphe.subjects(RDF.type, OWL.DatatypeProperty))

    lignes = []

    for prop in sorted(props, key=lambda x: nom_local(x)):
        domaines = joindre_noms_uris(graphe.objects(prop, RDFS.domain))
        ranges = joindre_noms_uris(graphe.objects(prop, RDFS.range))

        lignes.append(
            {
                "Propriété de données": nom_local(prop),
                "Domaine lisible": domaines,
                "Type / Range lisible": ranges,
                "Commentaire": commentaire(graphe, prop),
            }
        )

    return pd.DataFrame(lignes)


def extraire_individus(graphe):
    """
    Extrait les individus de l’ontologie.

    On retire les classes, propriétés et éléments techniques OWL/RDF.
    """
    types_exclus = {
        OWL.Class,
        RDFS.Class,
        OWL.ObjectProperty,
        OWL.DatatypeProperty,
        OWL.AnnotationProperty,
        OWL.Ontology,
        OWL.Restriction,
    }

    lignes = []

    for sujet, _, type_obj in graphe.triples((None, RDF.type, None)):
        if type_obj in types_exclus:
            continue

        if not isinstance(sujet, URIRef):
            continue

        if str(type_obj).startswith(str(OWL)) and nom_local(type_obj) in {
            "NamedIndividual",
        }:
            continue

        lignes.append(
            {
                "Individu": nom_local(sujet),
                "Type": nom_local(type_obj),
                "Label": label_ou_nom(graphe, sujet),
                "Commentaire": commentaire(graphe, sujet),
            }
        )

    df_individus = pd.DataFrame(lignes)

    if df_individus.empty:
        return df_individus

    return (
        df_individus
        .drop_duplicates()
        .sort_values(["Type", "Individu"])
        .reset_index(drop=True)
    )


def extraire_constantes_litterales(graphe):
    """
    Extrait les valeurs littérales associées aux individus :
    constantes physiques, capacités, limites, tensions, etc.

    Les sujets anonymes RDF sont ignorés pour éviter les lignes techniques.
    """
    lignes = []

    proprietes_a_ignorer = {
        str(RDF.type),
        str(RDFS.label),
        str(RDFS.comment),
    }

    for sujet, predicat, objet in graphe:
        if str(predicat) in proprietes_a_ignorer:
            continue

        if not isinstance(sujet, URIRef):
            continue

        if isinstance(objet, Literal):
            lignes.append(
                {
                    "Individu / Sujet": nom_local(sujet),
                    "Propriété": nom_local(predicat),
                    "Valeur": str(objet),
                }
            )

    df_constantes = pd.DataFrame(lignes)

    if df_constantes.empty:
        return df_constantes

    return (
        df_constantes
        .drop_duplicates()
        .sort_values(["Individu / Sujet", "Propriété"])
        .reset_index(drop=True)
    )


def extraire_relations_individus(graphe):
    """
    Extrait les relations lisibles entre individus ou concepts.

    Les nœuds anonymes RDF du type N935b5d63... sont ignorés, car ils
    correspondent à des restrictions OWL internes et non à des concepts
    directement lisibles pour l’utilisateur.
    """
    lignes = []

    relations_techniques = {
        RDF.type,
        RDFS.label,
        RDFS.comment,
        RDFS.domain,
        RDFS.range,
    }

    for sujet, predicat, objet in graphe:
        if predicat in relations_techniques:
            continue

        if isinstance(objet, Literal):
            continue

        if not isinstance(sujet, URIRef):
            continue

        if not isinstance(objet, URIRef):
            continue

        sujet_nom = nom_local(sujet)
        objet_nom = nom_local(objet)
        relation_nom = nom_local(predicat)

        if sujet_nom.startswith("N") or objet_nom.startswith("N"):
            continue

        lignes.append(
            {
                "Sujet": sujet_nom,
                "Relation": relation_nom,
                "Objet": objet_nom,
            }
        )

    df_relations = pd.DataFrame(lignes)

    if df_relations.empty:
        return df_relations

    return (
        df_relations
        .drop_duplicates()
        .sort_values(["Sujet", "Relation", "Objet"])
        .reset_index(drop=True)
    )


def extraire_regles_swrl(graphe):
    """
    Extrait les règles SWRL si elles sont présentes.

    rdflib ne reconstruit pas toujours facilement les règles SWRL sous forme
    humaine. Cette fonction liste au minimum les individus de type swrl:Imp.
    """
    lignes = []

    swrl_imp = URIRef("http://www.w3.org/2003/11/swrl#Imp")
    swrl_body = URIRef("http://www.w3.org/2003/11/swrl#body")
    swrl_head = URIRef("http://www.w3.org/2003/11/swrl#head")

    for regle in graphe.subjects(RDF.type, swrl_imp):
        lignes.append(
            {
                "Règle SWRL": nom_local(regle),
                "Body": nom_local(next(graphe.objects(regle, swrl_body), "")),
                "Head": nom_local(next(graphe.objects(regle, swrl_head), "")),
                "Commentaire": commentaire(graphe, regle),
            }
        )

    return pd.DataFrame(lignes)


def afficher_dataframe_ou_message(df_data, message):
    """
    Affiche un dataframe ou un message si le tableau est vide.
    """
    if df_data is None or df_data.empty:
        st.info(message)
    else:
        st.dataframe(
            df_data,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# 1. Chaîne de connaissance et de décision
# ============================================================

st.subheader("1. Chaîne de connaissance et de décision")

st.write(
    "La couche symbolique permet de relier les mesures physiques aux états "
    "qualitatifs du système, puis aux règles et à la décision finale."
)

st.graphviz_chart(
    """
    digraph G {
        rankdir=LR;
        node [shape=box, style="rounded,filled", color="#4c566a", fillcolor="#1f2937", fontcolor="white"];

        A [label="Grandeurs physiques\\nP_dem, vitesse, accélération,\\nSOC_EB, SOC_PB"];
        B [label="Ontologie OWL\\nclasses, propriétés, individus,\\nconstantes physiques"];
        C [label="États symboliques\\nEB_available, PB_low_SOC,\\nhigh_power_demand, ..."];
        D [label="Règles floues / expertes\\nR1 ... R7"];
        E [label="Alpha_PB proposé\\nalpha_PB = P_PB / P_dem"];
        F [label="Filtre de sécurité\\nrespect des contraintes physiques"];
        G [label="Décision finale\\nP_EB et P_PB appliqués"];

        A -> B -> C -> D -> E -> F -> G;
    }
    """
)


# ============================================================
# 2. Chargement et contenu réel de l’ontologie OWL
# ============================================================

st.subheader("2. Contenu réel de l’ontologie OWL")

chemin_owl = trouver_fichier_ontologie()

if chemin_owl is None:
    st.warning(
        "Aucun fichier OWL n’a été trouvé. Place le fichier dans : "
        "`ontologies/OntoHESS2.owl`."
    )

    with st.expander("Chemins recherchés"):
        chemins_affiches = [
            DOSSIER_PROJET / "ontologies" / "OntoHESS2.owl",
            DOSSIER_PROJET / "ontology" / "OntoHESS2.owl",
            DOSSIER_PROJET / "ontologie" / "OntoHESS2.owl",
            DOSSIER_PROJET / "OntoHESS2.owl",
            Path(r"C:\Users\Admin\Desktop\Projet_Artemis2\ontologies\OntoHESS2.owl"),
        ]

        for chemin in chemins_affiches:
            st.code(str(chemin))

    graphe_owl = None

elif not RDFLIB_DISPONIBLE:
    st.error(
        "Le paquet `rdflib` n’est pas installé. Ajoute `rdflib` dans "
        "`requirements.txt`, puis relance l’application."
    )
    st.code("pip install rdflib")
    graphe_owl = None

else:
    try:
        graphe_owl = charger_graphe_owl(chemin_owl)

        st.success(f"Ontologie chargée : {chemin_owl}")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric("Triplets RDF", len(graphe_owl))

        with c2:
            st.metric("Classes", len(extraire_classes(graphe_owl)))

        with c3:
            st.metric("Individus", len(extraire_individus(graphe_owl)))

        tab_info, tab_classes, tab_props, tab_individus, tab_constantes, tab_relations, tab_swrl = st.tabs(
            [
                "Informations",
                "Classes",
                "Propriétés",
                "Individus",
                "Constantes",
                "Relations",
                "Règles SWRL",
            ]
        )

        with tab_info:
            afficher_dataframe_ou_message(
                extraire_ontologie(graphe_owl),
                "Aucune déclaration OWL.Ontology n’a été trouvée.",
            )

        with tab_classes:
            afficher_dataframe_ou_message(
                extraire_classes(graphe_owl),
                "Aucune classe OWL/RDFS n’a été trouvée.",
            )

        with tab_props:
            st.caption(
                "Les identifiants techniques générés par OWL/RDF pour les restrictions "
                "anonymes sont masqués afin de ne garder que les classes lisibles."
            )

            st.markdown("**Relations objet**")
            afficher_dataframe_ou_message(
                extraire_proprietes_objet(graphe_owl),
                "Aucune propriété objet n’a été trouvée.",
            )

            st.markdown("**Propriétés de données**")
            afficher_dataframe_ou_message(
                extraire_proprietes_donnees(graphe_owl),
                "Aucune propriété de données n’a été trouvée.",
            )

        with tab_individus:
            afficher_dataframe_ou_message(
                extraire_individus(graphe_owl),
                "Aucun individu OWL n’a été trouvé.",
            )

        with tab_constantes:
            afficher_dataframe_ou_message(
                extraire_constantes_litterales(graphe_owl),
                "Aucune constante littérale n’a été trouvée.",
            )

        with tab_relations:
            afficher_dataframe_ou_message(
                extraire_relations_individus(graphe_owl),
                "Aucune relation non littérale n’a été trouvée.",
            )

        with tab_swrl:
            afficher_dataframe_ou_message(
                extraire_regles_swrl(graphe_owl),
                "Aucune règle SWRL de type swrl:Imp n’a été trouvée.",
            )

    except Exception as exc:
        st.error(
            "Le fichier OWL a été trouvé mais n’a pas pu être chargé."
        )
        st.exception(exc)
        graphe_owl = None


# ============================================================
# 3. Concepts physiques du système
# ============================================================

st.subheader("3. Concepts physiques du système HESS")

st.write(
    "Le système hybride de stockage d’énergie repose sur deux batteries "
    "complémentaires, un convertisseur et un filtre de sécurité."
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    with st.container(border=True):
        st.markdown("### Batterie d’énergie (EB)")
        st.write(
            "Source énergétique principale. Elle fournit la majeure partie "
            "de l’énergie lorsque ses contraintes de puissance et de SOC le permettent."
        )

with col2:
    with st.container(border=True):
        st.markdown("### Batterie de puissance (PB)")
        st.write(
            "Source de puissance instantanée. Elle intervient lors des pics de "
            "demande, des fortes accélérations ou lorsque l’EB atteint une limite."
        )

with col3:
    with st.container(border=True):
        st.markdown("### Convertisseur")
        st.write(
            "Composant de liaison entre EB, PB et la charge. Il impose ses propres "
            "limites physiques de fonctionnement."
        )

with col4:
    with st.container(border=True):
        st.markdown("### Filtre de sécurité")
        st.write(
            "Vérifie et corrige la décision EMS afin que les puissances appliquées "
            "restent compatibles avec les contraintes physiques."
        )


# ============================================================
# 4. États symboliques dynamiques
# ============================================================

st.subheader("4. États symboliques à l’instant sélectionné")

st.write(
    "Les états symboliques traduisent les valeurs numériques en descriptions "
    "qualitatives : disponibilité des batteries, SOC faible, forte demande de "
    "puissance, freinage régénératif ou risque convertisseur."
)

donnees_simulation_disponibles = (
    "resultats_simulation" in st.session_state
    and "cycle_pret" in st.session_state
    and bool(st.session_state["resultats_simulation"])
)

if donnees_simulation_disponibles:
    resultats = st.session_state["resultats_simulation"]
    df = st.session_state["cycle_pret"]

    nombre_points_max = min(
        len(traj["P_EB"])
        for traj in resultats.values()
    )

    instant_choisi = st.slider(
        "Instant à analyser",
        0,
        nombre_points_max - 1,
        nombre_points_max // 2,
    )

    if "time" in df.columns:
        t_sel = float(df["time"].iloc[instant_choisi])
    else:
        t_sel = float(instant_choisi)

    p_dem = float(df.iloc[instant_choisi]["hasPower"])

    nom_strategie = st.selectbox(
        "Stratégie utilisée comme référence pour les SOC",
        list(resultats.keys()),
        format_func=lambda n: MODEL_DISPLAY_NAMES.get(n, n),
    )

    traj = resultats[nom_strategie]

    soc_eb_instant = float(traj["SOC_EB"][instant_choisi])
    soc_pb_instant = float(traj["SOC_PB"][instant_choisi])

    etats = compute_symbolic_states(
        p_dem,
        soc_eb_instant,
        soc_pb_instant,
    )

    etats_actifs = [
        nom
        for nom, valeur in etats.items()
        if valeur
    ]

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("Temps", f"{t_sel:.0f} s")

    with c2:
        st.metric("P_dem", f"{p_dem / 1000.0:.2f} kW")

    with c3:
        st.metric("SOC_EB", f"{soc_eb_instant:.3f}")

    with c4:
        st.metric("SOC_PB", f"{soc_pb_instant:.3f}")

    if etats_actifs:
        st.success(
            "États symboliques actifs : "
            + ", ".join(f"**{nom}**" for nom in etats_actifs)
        )
    else:
        st.info(
            "Aucun état symbolique particulier n’est actif à cet instant."
        )

    tableau_etats = pd.DataFrame(
        [
            {
                "État symbolique": nom,
                "Actif": "Oui" if valeur else "Non",
                "Lecture": (
                    "État actuellement vérifié"
                    if valeur
                    else "État non vérifié à cet instant"
                ),
            }
            for nom, valeur in etats.items()
        ]
    )

    st.dataframe(
        tableau_etats,
        use_container_width=True,
        hide_index=True,
    )

else:
    st.info(
        "Lance une simulation globale pour explorer les états symboliques "
        "à un instant précis."
    )


# ============================================================
# 5. Activation des règles floues
# ============================================================

st.subheader("5. Activation des règles floues")

st.write(
    "Cette section montre quelles règles floues sont activées à l’instant "
    "sélectionné. Cela permet de relier directement les états physiques et "
    "symboliques à la décision alpha_PB."
)

if donnees_simulation_disponibles:
    acceleration = (
        float(df.iloc[instant_choisi]["hasAcceleration"])
        if "hasAcceleration" in df.columns
        else 0.0
    )

    try:
        fuzzy_out = alpha_fuzzy_calc(
            np.array([soc_eb_instant]),
            np.array([soc_pb_instant]),
            np.array([p_dem]),
            np.array([acceleration]),
        )

        strengths = fuzzy_out["strengths"][0]
        dominant_rule = str(fuzzy_out["dominant_rule"][0])
        alpha_fuzzy = float(fuzzy_out["alpha"][0]) if "alpha" in fuzzy_out else np.nan

        noms_regles = [
            "R1_PB_low_traction",
            "R2_EB_low_PB_available",
            "R3_strong_traction",
            "R4_zero_demand",
            "R5_regenerative_braking",
            "R5b_PB_high_recharge",
            "R7_two_low_SOC",
        ]

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric("Règle dominante", dominant_rule)

        with c2:
            st.metric(
                "Degré max",
                f"{float(np.max(strengths)):.3f}",
            )

        with c3:
            st.metric(
                "Alpha_PB flou",
                f"{alpha_fuzzy:.3f}" if np.isfinite(alpha_fuzzy) else "n/a",
            )

        tableau_activation = pd.DataFrame(
            [
                {
                    "Règle": nom_regle,
                    "Degré d’activation": float(strength),
                    "Objectif": RULE_LABELS_FR.get(nom_regle, ""),
                }
                for nom_regle, strength in zip(noms_regles, strengths)
            ]
        ).sort_values(
            by="Degré d’activation",
            ascending=False,
        )

        st.dataframe(
            tableau_activation,
            use_container_width=True,
            hide_index=True,
        )

    except Exception as exc:
        st.error(
            "Impossible de calculer l’activation des règles floues pour cet instant."
        )
        st.exception(exc)

else:
    st.info(
        "La simulation est nécessaire pour afficher l’activation dynamique "
        "des règles floues."
    )


# ============================================================
# 6. Base complète des règles floues
# ============================================================

st.subheader("6. Base complète des règles floues")

st.write(
    "La base de règles floues formalise les situations importantes du HESS : "
    "SOC faible, forte traction, récupération d’énergie, demande quasi nulle "
    "ou protection d’une batterie."
)

lignes_regles = []

for nom_regle, description in RULE_LABELS_FR.items():
    if nom_regle == "DEFAULT":
        continue

    lignes_regles.append(
        {
            "Règle": nom_regle,
            "Objectif": description,
        }
    )

st.dataframe(
    pd.DataFrame(lignes_regles),
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# 7. Rôle des stratégies EMS
# ============================================================

st.subheader("7. Rôle des stratégies EMS dans l’architecture neurosymbolique")

st.write(
    "Ce tableau présente les sept stratégies EMS et leur lien avec la couche "
    "symbolique du projet."
)

tableau_strategies = pd.DataFrame(
    MODEL_CONSTRUCTION_DETAILED
).rename(
    columns={
        "modele": "Modèle",
        "type": "Type",
        "donnees_entree": "Données d’entrée",
        "role": "Rôle",
    }
)

if not tableau_strategies.empty:
    if "Modèle" in tableau_strategies.columns:
        tableau_strategies["Nom affiché"] = tableau_strategies["Modèle"].map(
            lambda x: MODEL_DISPLAY_NAMES.get(x, x)
        )

    liens_symboliques = {
        "EMS_power_limitation": "Règle déterministe de secours, priorité EB.",
        "EMS_fuzzy_logic": "Utilise directement les états et règles floues.",
        "EMS_MLP": "Modèle appris tabulaire, comparé à la logique symbolique.",
        "EMS_MLP_neurosymbolic": "Combine apprentissage MLP et variables symboliques.",
        "EMS_LSTM": "Modèle temporel, comparé aux règles sur une fenêtre.",
        "EMS_LSTM_neurosymbolic": "Mémoire temporelle enrichie par états symboliques.",
        "EMS_GNN": "Représentation graphe des composants HESS.",
    }

    if "Modèle" in tableau_strategies.columns:
        tableau_strategies["Lien symbolique"] = tableau_strategies["Modèle"].map(
            lambda x: liens_symboliques.get(x, "")
        )

    colonnes_prioritaires = [
        col
        for col in [
            "Nom affiché",
            "Type",
            "Données d’entrée",
            "Lien symbolique",
            "Rôle",
        ]
        if col in tableau_strategies.columns
    ]

    st.dataframe(
        tableau_strategies[colonnes_prioritaires],
        use_container_width=True,
        hide_index=True,
    )

else:
    st.info("Le tableau de construction des stratégies n’est pas disponible.")


# ============================================================
# 8. Traçabilité de la décision
# ============================================================

st.subheader("8. Traçabilité de la décision")

st.write(
    "La décision finale peut être retracée comme une succession d’étapes "
    "physiques, symboliques et décisionnelles."
)

with st.container(border=True):
    st.markdown(
        """
        **Étape 1 — Mesures physiques**  
        Le système observe `P_dem`, la vitesse, l’accélération, `SOC_EB` et `SOC_PB`.

        **Étape 2 — Ontologie OWL**  
        Les concepts du HESS, les composants, les propriétés et certaines constantes
        physiques sont formalisés dans l’ontologie.

        **Étape 3 — États symboliques**  
        Les valeurs numériques sont traduites en états qualitatifs comme
        `EB_available`, `PB_low_SOC`, `high_power_demand` ou `regenerative_braking`.

        **Étape 4 — Règles floues / expertes**  
        Les états activent des règles qui proposent une orientation de répartition.

        **Étape 5 — Proposition d’alpha_PB**  
        La stratégie EMS produit une valeur `alpha_PB`, représentant la part de
        puissance confiée à la PB.

        **Étape 6 — Filtre de sécurité physique**  
        La proposition est corrigée si elle viole une contrainte de SOC, de puissance,
        de courant ou de convertisseur.

        **Étape 7 — Décision finale**  
        La décision appliquée détermine `P_PB = alpha_PB * P_dem`, puis
        `P_EB = P_dem - P_PB`.
        """
    )


# ============================================================
# Navigation
# ============================================================

st.divider()

if st.button(
    "Consulter les résultats et l’analyse",
    type="primary",
):
    st.switch_page("pages/6_Resultats_et_Analyse.py")
