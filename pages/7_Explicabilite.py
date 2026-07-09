
import sys
from pathlib import Path

DOSSIER_PROJET = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DOSSIER_PROJET))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from ems_core import (
    EPS_POWER_W,
    P_EB_MAX_W,
    SOC_EB_MIN,
    alpha_fuzzy_calc,
    RULE_LABELS_FR,
    MODEL_ORDER,
    MODEL_DISPLAY_NAMES,
    MODEL_CONSTRUCTION_SUMMARY,
    load_gnn_test_graphs,
    GNN_NODE_NAMES,
)

import ems_core as _core


# ============================================================
# Configuration de la page
# ============================================================

st.set_page_config(
    page_title="2SMART — Explicabilité",
    layout="wide",
)

st.title("Explicabilité des modèles EMS")

st.write(
    "Cette page explique la décision prise par chaque stratégie EMS à un instant "
    "donné du cycle. Elle relie la décision du modèle aux grandeurs physiques : "
    "puissance demandée, répartition EB/PB, courants, SOC, état des batteries "
    "et intervention éventuelle du filtre de sécurité."
)

st.info(
    "Convention alpha utilisée dans le projet : `alpha = P_PB / P_dem`. "
    "`alpha = 0` signifie que la PB ne participe pas et que l’EB fournit toute "
    "la demande. `alpha = 1` signifie que la PB fournit toute la demande. "
    "Une stratégie à priorité EB se traduit donc généralement par un alpha faible."
)

st.caption(
    "Convention de signe : `P > 0` signifie que la batterie fournit de la puissance "
    "au véhicule, donc elle est en décharge. `P < 0` signifie que la batterie absorbe "
    "de la puissance, donc elle est en recharge. `P ≈ 0` signifie que la batterie est "
    "au repos."
)


# ============================================================
# Vérification des résultats disponibles
# ============================================================

from core.resultats import assurer_donnees_session

try:
    _source = assurer_donnees_session(st)
except FileNotFoundError as exc:
    st.error(str(exc))
    st.info("Lance une fois le précalcul :  `python scripts/run_simulations.py`")
    st.stop()

st.caption(f"Source des données : {_source}")

if (
    "resultats_simulation" not in st.session_state
    or "cycle_pret" not in st.session_state
):
    st.warning("Aucune donnée disponible.")
    st.stop()


resultats = st.session_state["resultats_simulation"]
df = st.session_state["cycle_pret"]

if not resultats:
    st.warning("Aucune stratégie n’a produit de résultat exploitable.")
    st.stop()


# ============================================================
# Fonctions utilitaires générales
# ============================================================

def nom_affiche(nom):
    """Retourne le nom lisible d’une stratégie EMS."""
    return MODEL_DISPLAY_NAMES.get(nom, nom)


def get_array(traj, key, n=None, default=0.0, dtype=float):
    """
    Récupère une variable de trajectoire sous forme de tableau numpy.

    Si la variable n’existe pas :
    - retourne un tableau vide si n n’est pas donné ;
    - retourne un tableau rempli avec default si n est donné.
    """
    if key in traj:
        return np.asarray(traj[key], dtype=dtype)

    if n is None:
        return np.asarray([], dtype=dtype)

    return np.full(n, default, dtype=dtype)


def valeur_ou_na(x, fmt="{:.3f}"):
    """Formate une valeur numérique ou retourne n/a."""
    try:
        if np.isfinite(x):
            return fmt.format(float(x))
    except Exception:
        pass

    return "n/a"


def obtenir_temps(n):
    """Retourne l’axe temporel."""
    if "time" in df.columns:
        return df["time"].to_numpy()[:n]

    return np.arange(n)


def phase_cycle(p_dem, seuil=50.0):
    """Détermine la phase physique du cycle."""
    if p_dem > seuil:
        return "Traction"

    if p_dem < -seuil:
        return "Freinage / récupération"

    return "Quasi-neutre"


def etat_batterie(p, seuil=50.0):
    """
    Détermine l’état instantané d’une batterie selon la puissance.

    Convention :
    - P > seuil  : décharge ;
    - P < -seuil : recharge ;
    - sinon      : repos.
    """
    if p > seuil:
        return "Décharge"

    if p < -seuil:
        return "Recharge"

    return "Repos"


def interpretation_alpha(alpha):
    """
    Interprète alpha selon la convention du projet :
    alpha = P_PB / P_dem.

    Donc :
    - alpha = 0 : PB ne participe pas, EB fournit toute la demande ;
    - alpha = 1 : PB fournit toute la demande.
    """
    if not np.isfinite(alpha):
        return "Alpha_PB non disponible."

    if abs(alpha) < 1e-6:
        return (
            "Alpha_PB est égal à 0 : la PB ne participe pas ; "
            "l’EB couvre donc toute la demande."
        )

    if alpha < 0.05:
        return (
            "Alpha_PB est très faible : la PB fournit une très faible part "
            "de la demande ; l’EB couvre donc presque toute la demande."
        )

    if alpha < 0.30:
        return (
            "Alpha_PB est faible : la PB fournit une faible part de la demande, "
            "tandis que l’EB reste majoritaire."
        )

    if alpha < 0.70:
        return (
            "Alpha_PB est intermédiaire : la demande est partagée entre EB et PB."
        )

    if alpha < 1.0 - 1e-6:
        return (
            "Alpha_PB est élevé : la PB fournit une part importante de la demande."
        )

    return (
        "Alpha_PB est égal à 1 : la PB couvre toute la demande, "
        "et l’EB ne fournit pas de puissance à cet instant."
    )


def extraire_valeurs_instantanees(traj, pas):
    """
    Extrait toutes les grandeurs utiles pour une stratégie à un instant donné.
    """
    p_eb = get_array(traj, "P_EB")
    n = len(p_eb)

    p_pb = get_array(traj, "P_PB", n=n)
    i_eb = get_array(traj, "I_EB", n=n)
    i_pb = get_array(traj, "I_PB", n=n)

    alpha_requested = get_array(
        traj,
        "alpha_requested",
        n=n,
        default=np.nan,
    )

    alpha_final = get_array(
        traj,
        "alpha_final",
        n=n,
        default=np.nan,
    )

    correction = get_array(
        traj,
        "correction_applied",
        n=n,
        default=False,
        dtype=bool,
    )

    soc_eb = get_array(traj, "SOC_EB")
    soc_pb = get_array(traj, "SOC_PB")

    temps = obtenir_temps(n)

    if "hasPower" in df.columns:
        p_dem = df["hasPower"].to_numpy(dtype=float)[:n]
    else:
        p_dem = p_eb + p_pb

    pas = max(0, min(int(pas), n - 1))

    valeurs = {
        "n": n,
        "temps": temps,
        "pas": pas,
        "t_sel": float(temps[pas]),
        "p_dem": p_dem,
        "p_eb": p_eb,
        "p_pb": p_pb,
        "i_eb": i_eb,
        "i_pb": i_pb,
        "soc_eb": soc_eb,
        "soc_pb": soc_pb,
        "alpha_requested": alpha_requested,
        "alpha_final": alpha_final,
        "correction": correction,
        "p_dem_sel": float(p_dem[pas]),
        "p_eb_sel": float(p_eb[pas]),
        "p_pb_sel": float(p_pb[pas]),
        "i_eb_sel": float(i_eb[pas]),
        "i_pb_sel": float(i_pb[pas]),
        "soc_eb_sel": float(soc_eb[pas]) if pas < len(soc_eb) else np.nan,
        "soc_pb_sel": float(soc_pb[pas]) if pas < len(soc_pb) else np.nan,
        "alpha_requested_sel": (
            float(alpha_requested[pas])
            if pas < len(alpha_requested)
            else np.nan
        ),
        "alpha_final_sel": (
            float(alpha_final[pas])
            if pas < len(alpha_final)
            else np.nan
        ),
        "correction_sel": bool(correction[pas]) if pas < len(correction) else False,
    }

    valeurs["phase"] = phase_cycle(valeurs["p_dem_sel"])
    valeurs["etat_eb"] = etat_batterie(valeurs["p_eb_sel"])
    valeurs["etat_pb"] = etat_batterie(valeurs["p_pb_sel"])

    if np.isfinite(valeurs["alpha_final_sel"]):
        valeurs["part_pb"] = valeurs["alpha_final_sel"]
        valeurs["part_eb"] = 1.0 - valeurs["alpha_final_sel"]
    else:
        valeurs["part_pb"] = np.nan
        valeurs["part_eb"] = np.nan

    return valeurs


# ============================================================
# Explications textuelles
# ============================================================

def expliquer_eb_priority(p_dem, soc_eb):
    """
    Explication de la stratégie déterministe à priorité EB.
    """
    if abs(p_dem) <= EPS_POWER_W:
        return (
            "La demande de puissance est quasi nulle. La stratégie ne sollicite "
            "pas significativement les batteries."
        )

    if p_dem < 0:
        return (
            "Le véhicule est en phase de récupération. La stratégie oriente "
            "prioritairement l’énergie récupérable vers l’EB, dans la limite "
            "des contraintes physiques."
        )

    if soc_eb <= SOC_EB_MIN:
        return (
            "Le SOC de l’EB est au seuil minimal. La stratégie protège l’EB et "
            "la PB doit prendre le relais lorsque cela est physiquement possible."
        )

    if p_dem <= P_EB_MAX_W:
        return (
            "La demande reste dans la puissance maximale admissible de l’EB. "
            "La stratégie donne donc la priorité à l’EB ; comme alpha représente "
            "la part de la PB, l’alpha attendu reste faible."
        )

    return (
        "La demande dépasse la puissance maximale admissible de l’EB. "
        "L’EB fournit sa limite et la PB fournit le complément."
    )


def generer_explication_physique(nom, valeurs):
    """
    Génère une explication automatique de la décision appliquée.
    """
    p_dem = valeurs["p_dem_sel"]
    p_eb = valeurs["p_eb_sel"]
    p_pb = valeurs["p_pb_sel"]
    alpha_req = valeurs["alpha_requested_sel"]
    alpha_final = valeurs["alpha_final_sel"]
    correction = valeurs["correction_sel"]

    texte = (
        f"À **t = {valeurs['t_sel']:.0f} s**, la demande vaut "
        f"**{p_dem / 1000.0:.2f} kW**. "
        f"La stratégie **{nom_affiche(nom)}** applique "
        f"`alpha_PB = {valeur_ou_na(alpha_final)}`. "
    )

    if np.isfinite(alpha_req):
        texte += (
            f"Avant filtrage, la valeur proposée était "
            f"`alpha_PB_requested = {alpha_req:.3f}`. "
        )

    texte += interpretation_alpha(alpha_final) + " "

    texte += (
        f"L’EB fournit **{p_eb / 1000.0:.2f} kW** et se trouve en "
        f"**{valeurs['etat_eb'].lower()}**. "
        f"La PB fournit **{p_pb / 1000.0:.2f} kW** et se trouve en "
        f"**{valeurs['etat_pb'].lower()}**. "
    )

    if correction:
        texte += (
            "Le filtre de sécurité a modifié la décision initiale, ce qui indique "
            "que la proposition du modèle devait être adaptée aux contraintes "
            "physiques du HESS."
        )
    else:
        texte += (
            "Aucune correction du filtre de sécurité n’a été nécessaire : la décision "
            "était compatible avec les contraintes physiques à cet instant."
        )

    return texte


# ============================================================
# Affichages Streamlit
# ============================================================

def afficher_resume_global(pas):
    """
    Affiche le contexte physique global de l’instant sélectionné.
    """
    ligne = df.iloc[pas]

    if "time" in df.columns:
        t_sel = float(df["time"].to_numpy()[pas])
    else:
        t_sel = float(pas)

    p_dem = float(ligne["hasPower"]) if "hasPower" in df.columns else np.nan

    acceleration = (
        float(ligne["hasAcceleration"])
        if "hasAcceleration" in df.columns
        else np.nan
    )

    phase = phase_cycle(p_dem) if np.isfinite(p_dem) else "n/a"

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("Temps", f"{t_sel:.0f} s")

    with c2:
        st.metric(
            "Puissance demandée",
            f"{p_dem / 1000.0:.2f} kW" if np.isfinite(p_dem) else "n/a",
        )

    with c3:
        st.metric(
            "Accélération",
            f"{acceleration:.3f}" if np.isfinite(acceleration) else "n/a",
        )

    with c4:
        st.metric("Phase", phase)


def afficher_metriques_decision(valeurs):
    """
    Affiche les métriques principales de la décision appliquée.
    """
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("Alpha_PB appliqué", valeur_ou_na(valeurs["alpha_final_sel"]))

    with c2:
        st.metric("Part EB", valeur_ou_na(valeurs["part_eb"]))

    with c3:
        st.metric("Part PB", valeur_ou_na(valeurs["part_pb"]))

    with c4:
        st.metric("Filtre", "Corrigé" if valeurs["correction_sel"] else "Non corrigé")

    c5, c6, c7, c8 = st.columns(4)

    with c5:
        st.metric("P_EB", f"{valeurs['p_eb_sel'] / 1000.0:.2f} kW")

    with c6:
        st.metric("P_PB", f"{valeurs['p_pb_sel'] / 1000.0:.2f} kW")

    with c7:
        st.metric("État EB", valeurs["etat_eb"])

    with c8:
        st.metric("État PB", valeurs["etat_pb"])

    c9, c10, c11, c12 = st.columns(4)

    with c9:
        st.metric("SOC_EB", valeur_ou_na(valeurs["soc_eb_sel"]))

    with c10:
        st.metric("SOC_PB", valeur_ou_na(valeurs["soc_pb_sel"]))

    with c11:
        st.metric("I_EB", f"{valeurs['i_eb_sel']:.2f} A")

    with c12:
        st.metric("I_PB", f"{valeurs['i_pb_sel']:.2f} A")


def afficher_tableau_decision(valeurs):
    """
    Affiche les grandeurs instantanées dans un tableau.
    """
    lignes = [
        {
            "Grandeur": "Puissance demandée P_dem",
            "Valeur": f"{valeurs['p_dem_sel'] / 1000.0:.3f} kW",
            "Lecture physique": valeurs["phase"],
        },
        {
            "Grandeur": "Alpha_PB demandé",
            "Valeur": valeur_ou_na(valeurs["alpha_requested_sel"]),
            "Lecture physique": "Sortie proposée avant filtre",
        },
        {
            "Grandeur": "Alpha_PB appliqué",
            "Valeur": valeur_ou_na(valeurs["alpha_final_sel"]),
            "Lecture physique": interpretation_alpha(valeurs["alpha_final_sel"]),
        },
        {
            "Grandeur": "Part EB",
            "Valeur": valeur_ou_na(valeurs["part_eb"]),
            "Lecture physique": "Part de la demande couverte par l’EB",
        },
        {
            "Grandeur": "Part PB",
            "Valeur": valeur_ou_na(valeurs["part_pb"]),
            "Lecture physique": "Part de la demande couverte par la PB",
        },
        {
            "Grandeur": "P_EB",
            "Valeur": f"{valeurs['p_eb_sel'] / 1000.0:.3f} kW",
            "Lecture physique": valeurs["etat_eb"],
        },
        {
            "Grandeur": "P_PB",
            "Valeur": f"{valeurs['p_pb_sel'] / 1000.0:.3f} kW",
            "Lecture physique": valeurs["etat_pb"],
        },
        {
            "Grandeur": "I_EB",
            "Valeur": f"{valeurs['i_eb_sel']:.3f} A",
            "Lecture physique": "Courant EB",
        },
        {
            "Grandeur": "I_PB",
            "Valeur": f"{valeurs['i_pb_sel']:.3f} A",
            "Lecture physique": "Courant PB",
        },
        {
            "Grandeur": "SOC_EB",
            "Valeur": valeur_ou_na(valeurs["soc_eb_sel"]),
            "Lecture physique": "État de charge EB",
        },
        {
            "Grandeur": "SOC_PB",
            "Valeur": valeur_ou_na(valeurs["soc_pb_sel"]),
            "Lecture physique": "État de charge PB",
        },
        {
            "Grandeur": "Correction du filtre",
            "Valeur": "Oui" if valeurs["correction_sel"] else "Non",
            "Lecture physique": (
                "Décision corrigée"
                if valeurs["correction_sel"]
                else "Décision conservée"
            ),
        },
    ]

    st.dataframe(
        pd.DataFrame(lignes),
        use_container_width=True,
        hide_index=True,
    )


def afficher_graphiques_locaux(nom, valeurs, fenetre=80):
    """
    Affiche les graphes locaux autour de l’instant sélectionné :
    - puissances ;
    - courants ;
    - alpha ;
    - SOC.
    """
    pas = valeurs["pas"]
    n = valeurs["n"]
    i0 = max(0, pas - fenetre)
    i1 = min(n, pas + fenetre + 1)

    temps = valeurs["temps"]
    t_sel = valeurs["t_sel"]

    tab_p, tab_i, tab_alpha, tab_soc = st.tabs(
        [
            "Puissances",
            "Courants",
            "Alpha",
            "SOC",
        ]
    )

    with tab_p:
        fig_p, ax_p = plt.subplots(figsize=(11, 4.5))

        ax_p.plot(
            temps[i0:i1],
            valeurs["p_dem"][i0:i1] / 1000.0,
            label="P_dem",
            linewidth=2.0,
            linestyle="--",
        )

        ax_p.plot(
            temps[i0:i1],
            valeurs["p_eb"][i0:i1] / 1000.0,
            label="P_EB",
            linewidth=2.0,
        )

        ax_p.plot(
            temps[i0:i1],
            valeurs["p_pb"][i0:i1] / 1000.0,
            label="P_PB",
            linewidth=2.0,
        )

        ax_p.axvline(
            t_sel,
            linestyle=":",
            linewidth=2.0,
            label="Instant sélectionné",
        )

        ax_p.axhline(
            0.0,
            linewidth=1.0,
            alpha=0.5,
        )

        ax_p.set_title(
            f"Puissances autour de l’instant sélectionné — {nom_affiche(nom)}"
        )
        ax_p.set_xlabel("Temps (s)")
        ax_p.set_ylabel("Puissance (kW)")
        ax_p.grid(True, alpha=0.3)
        ax_p.legend(fontsize=8)

        st.pyplot(fig_p)
        plt.close(fig_p)

        st.caption(
            "Lecture : P_dem est la demande véhicule. P_EB et P_PB indiquent "
            "comment cette demande est répartie entre la batterie d’énergie et "
            "la batterie de puissance."
        )

    with tab_i:
        fig_i, ax_i = plt.subplots(figsize=(11, 4.5))

        ax_i.plot(
            temps[i0:i1],
            valeurs["i_eb"][i0:i1],
            label="I_EB",
            linewidth=2.0,
        )

        ax_i.plot(
            temps[i0:i1],
            valeurs["i_pb"][i0:i1],
            label="I_PB",
            linewidth=2.0,
        )

        ax_i.axvline(
            t_sel,
            linestyle=":",
            linewidth=2.0,
            label="Instant sélectionné",
        )

        ax_i.axhline(
            0.0,
            linewidth=1.0,
            alpha=0.5,
        )

        ax_i.set_title(
            f"Courants autour de l’instant sélectionné — {nom_affiche(nom)}"
        )
        ax_i.set_xlabel("Temps (s)")
        ax_i.set_ylabel("Courant (A)")
        ax_i.grid(True, alpha=0.3)
        ax_i.legend(fontsize=8)

        st.pyplot(fig_i)
        plt.close(fig_i)

        st.caption(
            "Lecture : les courants permettent d’identifier les sollicitations "
            "instantanées imposées aux deux batteries."
        )

    with tab_alpha:
        fig_a, ax_a = plt.subplots(figsize=(11, 4.5))

        alpha_requested = valeurs["alpha_requested"][i0:i1]
        alpha_final = valeurs["alpha_final"][i0:i1]

        if np.isfinite(alpha_requested).any():
            ax_a.plot(
                temps[i0:i1],
                alpha_requested,
                label="Alpha_PB demandé",
                linewidth=2.0,
                linestyle="--",
            )

        if np.isfinite(alpha_final).any():
            ax_a.plot(
                temps[i0:i1],
                alpha_final,
                label="Alpha_PB appliqué",
                linewidth=2.0,
            )

        ax_a.axvline(
            t_sel,
            linestyle=":",
            linewidth=2.0,
            label="Instant sélectionné",
        )

        ax_a.axhline(
            0.0,
            linewidth=1.0,
            alpha=0.5,
        )

        ax_a.axhline(
            1.0,
            linewidth=1.0,
            alpha=0.5,
        )

        ax_a.set_title(
            f"Alpha_PB demandé et appliqué — {nom_affiche(nom)}"
        )
        ax_a.set_xlabel("Temps (s)")
        ax_a.set_ylabel("Alpha_PB")
        ax_a.set_ylim(-0.05, 1.05)
        ax_a.grid(True, alpha=0.3)
        ax_a.legend(fontsize=8)

        st.pyplot(fig_a)
        plt.close(fig_a)

        st.caption(
            "Lecture : Alpha_PB représente la part de la demande confiée à la PB. "
            "Si alpha demandé et alpha appliqué diffèrent, le filtre de sécurité "
            "a modifié la décision initiale."
        )

    with tab_soc:
        fig_s, ax_s = plt.subplots(figsize=(11, 4.5))

        soc_eb = valeurs["soc_eb"]
        soc_pb = valeurs["soc_pb"]

        n_soc = min(len(soc_eb), len(soc_pb), len(temps))
        j0 = max(0, pas - fenetre)
        j1 = min(n_soc, pas + fenetre + 1)

        ax_s.plot(
            temps[j0:j1],
            soc_eb[j0:j1],
            label="SOC_EB",
            linewidth=2.0,
        )

        ax_s.plot(
            temps[j0:j1],
            soc_pb[j0:j1],
            label="SOC_PB",
            linewidth=2.0,
        )

        ax_s.axvline(
            t_sel,
            linestyle=":",
            linewidth=2.0,
            label="Instant sélectionné",
        )

        ax_s.axhline(
            SOC_EB_MIN,
            linestyle="--",
            linewidth=1.0,
            alpha=0.6,
            label="Seuil SOC minimal",
        )

        ax_s.set_title(
            f"SOC autour de l’instant sélectionné — {nom_affiche(nom)}"
        )
        ax_s.set_xlabel("Temps (s)")
        ax_s.set_ylabel("SOC")
        ax_s.set_ylim(0.15, 1.05)
        ax_s.grid(True, alpha=0.3)
        ax_s.legend(fontsize=8)

        st.pyplot(fig_s)
        plt.close(fig_s)

        st.caption(
            "Lecture : le SOC permet de comprendre pourquoi une batterie peut être "
            "sollicitée, protégée ou rechargée."
        )


def afficher_bloc_explicabilite_physique(nom, traj, pas):
    """
    Affiche la partie commune d’explicabilité physique pour toutes les stratégies.
    """
    valeurs = extraire_valeurs_instantanees(traj, pas)

    st.markdown("### Décision appliquée")

    afficher_metriques_decision(valeurs)

    with st.container(border=True):
        st.write(
            generer_explication_physique(
                nom,
                valeurs,
            )
        )

    st.markdown("### Tableau des grandeurs instantanées")
    afficher_tableau_decision(valeurs)

    st.markdown("### Visualisation graphique locale")
    afficher_graphiques_locaux(nom, valeurs)


def afficher_gnnexplainer():
    """
    Affiche l’analyse GNNExplainer optionnelle.
    """
    with st.expander("Analyse GNNExplainer sur les graphes de test"):
        if not _core._import_torch_geometric():
            st.error(
                "Le paquet torch_geometric n’est pas installé. "
                "GNNExplainer n’est donc pas disponible."
            )
            return

        try:
            graph_data = load_gnn_test_graphs()
            graphes_test = graph_data["test"]

            indice_graphe = st.slider(
                "Graphe de test",
                0,
                len(graphes_test) - 1,
                0,
                key="gnn_test_idx",
            )

            st.caption(
                f"{len(graphes_test)} graphes de test sont disponibles. "
                "Cette analyse porte sur les graphes de test construits pendant "
                "l’entraînement du GNN."
            )

            if "EMS_GNN" not in st.session_state.get(
                "modeles_charges_cache",
                {},
            ):
                st.info(
                    "Le modèle EMS_GNN chargé n’est pas disponible dans la session "
                    "actuelle. Relance le chargement des modèles depuis la page "
                    "« Simulation globale » avant d’exécuter GNNExplainer."
                )
                return

            from torch_geometric.explain import Explainer, GNNExplainer

            modele_gnn = st.session_state[
                "modeles_charges_cache"
            ]["EMS_GNN"]

            explainer = Explainer(
                model=modele_gnn,
                algorithm=GNNExplainer(
                    epochs=100
                ),
                explanation_type="model",
                node_mask_type="attributes",
                edge_mask_type="object",
                model_config=dict(
                    mode="regression",
                    task_level="graph",
                    return_type="raw",
                ),
            )

            graphe = graphes_test[indice_graphe]

            explication = explainer(
                graphe.x,
                graphe.edge_index,
                batch=None,
            )

            importance_noeuds = (
                explication.node_mask
                .sum(dim=-1)
                .detach()
                .cpu()
                .numpy()
            )

            tableau_importance = pd.DataFrame(
                {
                    "Nœud": GNN_NODE_NAMES[:len(importance_noeuds)],
                    "Importance": importance_noeuds,
                }
            ).sort_values(
                "Importance",
                ascending=False,
            )

            st.dataframe(
                tableau_importance,
                use_container_width=True,
                hide_index=True,
            )

            fig_gnn, ax_gnn = plt.subplots(figsize=(7, 3))

            ax_gnn.bar(
                tableau_importance["Nœud"],
                tableau_importance["Importance"],
            )

            ax_gnn.set_ylabel("Importance du masque de nœud")
            ax_gnn.set_title("Importance des nœuds selon GNNExplainer")

            plt.xticks(rotation=20)
            plt.tight_layout()

            st.pyplot(fig_gnn)
            plt.close(fig_gnn)

        except FileNotFoundError as exc:
            st.error(str(exc))


# ============================================================
# Sélection de l’instant à expliquer
# ============================================================

nombre_points_max = min(
    len(traj["P_EB"])
    for traj in resultats.values()
)

st.subheader("Instant à expliquer")

instant_choisi = st.slider(
    "Pas de simulation",
    0,
    nombre_points_max - 1,
    nombre_points_max // 2,
)

if "time" in df.columns:
    temps_choisi = float(df["time"].to_numpy()[instant_choisi])
else:
    temps_choisi = float(instant_choisi)

st.caption(
    f"Pas sélectionné : **{instant_choisi}** — "
    f"temps correspondant : **{temps_choisi:.0f} s**."
)

afficher_resume_global(instant_choisi)


# ============================================================
# Onglets par stratégie
# ============================================================

codes_onglets = [
    nom
    for nom in MODEL_ORDER
    if nom in resultats
]

for nom in resultats:
    if nom not in codes_onglets:
        codes_onglets.append(nom)

noms_onglets = [
    nom_affiche(nom)
    for nom in codes_onglets
]

onglets = st.tabs(noms_onglets)


for nom, onglet in zip(codes_onglets, onglets):
    with onglet:
        traj = resultats[nom]

        st.markdown("### Principe de la stratégie")

        st.caption(
            MODEL_CONSTRUCTION_SUMMARY.get(
                nom,
                "Résumé de construction non disponible pour cette stratégie.",
            )
        )

        valeurs = extraire_valeurs_instantanees(
            traj,
            instant_choisi,
        )

        st.markdown("### Explication XAI de la décision")

        if nom == "EMS_power_limitation":
            st.write(
                expliquer_eb_priority(
                    valeurs["p_dem_sel"],
                    valeurs["soc_eb_sel"],
                )
            )

        elif nom == "EMS_fuzzy_logic":
            acceleration = (
                float(df.iloc[instant_choisi]["hasAcceleration"])
                if "hasAcceleration" in df.columns
                else 0.0
            )

            resultat_fuzzy = alpha_fuzzy_calc(
                np.array([valeurs["soc_eb_sel"]]),
                np.array([valeurs["soc_pb_sel"]]),
                np.array([valeurs["p_dem_sel"]]),
                np.array([acceleration]),
            )

            strengths = resultat_fuzzy["strengths"][0]

            regle_dominante = str(
                resultat_fuzzy["dominant_rule"][0]
            )

            poids_dominant = float(
                strengths.max()
            )

            st.write(
                f"À cet instant, la règle floue la plus active est "
                f"**{regle_dominante}**, avec un degré d’activation de "
                f"**{poids_dominant:.2f}**. Cette règle cherche à "
                f"{RULE_LABELS_FR.get(regle_dominante, 'appliquer une répartition adaptée à la situation')}."
            )

            noms_regles = [
                "R1_PB_low_traction",
                "R2_EB_low_PB_available",
                "R3_strong_traction",
                "R4_zero_demand",
                "R5_regenerative_braking",
                "R5b_PB_high_recharge",
                "R7_two_low_SOC",
            ]

            autres_regles = sorted(
                zip(
                    noms_regles,
                    strengths,
                ),
                key=lambda x: x[1],
                reverse=True,
            )

            regles_secondaires = [
                (nom_regle, poids)
                for nom_regle, poids in autres_regles[1:3]
                if poids > 0.05
            ]

            if regles_secondaires:
                phrase_secondaire = ", ".join(
                    f"{nom_regle} ({poids:.2f})"
                    for nom_regle, poids in regles_secondaires
                )

                st.caption(
                    "Règles secondaires partiellement actives : "
                    f"{phrase_secondaire}."
                )

        elif nom in (
            "EMS_MLP",
            "EMS_MLP_neurosymbolic",
        ):
            st.write(
                "Pour les modèles MLP, l’explication XAI repose sur l’influence "
                "des variables d’entrée sur la valeur `alpha_PB`. L’analyse globale "
                "peut être réalisée avec SHAP. Ici, l’application explique la "
                "décision appliquée en la reliant aux grandeurs physiques mesurées "
                "à l’instant sélectionné."
            )

        elif nom in (
            "EMS_LSTM",
            "EMS_LSTM_neurosymbolic",
        ):
            st.write(
                "Pour les modèles LSTM, l’explication doit tenir compte de la "
                "fenêtre temporelle utilisée par le modèle. Une analyse de type "
                "Integrated Gradients peut identifier les instants passés et les "
                "variables qui influencent la décision. Ici, l’application montre "
                "la décision appliquée et son interprétation physique."
            )

        elif nom == "EMS_GNN":
            st.write(
                "Pour le GNN, l’explication peut s’appuyer sur l’importance des "
                "nœuds et des arêtes du graphe HESS. L’application présente d’abord "
                "la décision physique appliquée à l’instant choisi. Une analyse "
                "GNNExplainer peut ensuite être lancée sur les graphes de test."
            )

        else:
            st.write(
                "Cette stratégie est expliquée à partir de ses grandeurs physiques "
                "instantanées."
            )

        afficher_bloc_explicabilite_physique(
            nom,
            traj,
            instant_choisi,
        )

        if nom == "EMS_GNN":
            afficher_gnnexplainer()


# ============================================================
# Navigation vers la page de l’ontologie
# ============================================================

st.divider()

if st.button(
    "Continuer vers l’ontologie et les règles expertes",
    type="primary",
):
    st.switch_page(
        "pages/3_Ontologie_OntoHESS.py"
    )
