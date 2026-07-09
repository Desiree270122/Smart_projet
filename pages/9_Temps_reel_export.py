

import io
import sys
import time
from pathlib import Path

DOSSIER_PROJET = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DOSSIER_PROJET))

import numpy as np
import pandas as pd
import streamlit as st

from ems_core import MODEL_DISPLAY_NAMES


st.set_page_config(
    page_title="Temps réel et exportation",
    layout="wide",
)

st.title("Temps réel et exportation")


# ============================================================
# Vérification de la présence des résultats de simulation
# ============================================================

if (
    "resultats_simulation" not in st.session_state
    or "cycle_pret" not in st.session_state
):
    st.warning(
        "Une simulation doit être réalisée avant de consulter cette page."
    )

    if st.button("Ouvrir la page de simulation globale"):
        st.switch_page("pages/3_Simulation_globale.py")

    st.stop()


resultats = st.session_state["resultats_simulation"]
df = st.session_state["cycle_pret"]


if not resultats:
    st.warning(
        "Aucune stratégie n'a produit de résultat exploitable."
    )
    st.stop()


# ============================================================
# 1. Relecture de la simulation en pseudo-temps réel
# ============================================================

st.subheader("1. Mode pseudo-temps réel")

st.write(
    "Cette section rejoue la simulation déjà calculée, instant après instant, "
    "comme si les données arrivaient en direct. Il ne s'agit pas d'une nouvelle "
    "inférence réalisée en temps réel, mais d'une relecture progressive des "
    "résultats existants."
)


nom_strategie_replay = st.selectbox(
    "Stratégie à rejouer",
    list(resultats.keys()),
    format_func=lambda n: MODEL_DISPLAY_NAMES.get(n, n),
)


traj_replay = resultats[nom_strategie_replay]

nombre_points = len(
    traj_replay["P_EB"]
)


if "replay_position" not in st.session_state:
    st.session_state["replay_position"] = 0


# ============================================================
# Commandes de lecture
# ============================================================

col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)


if col_btn1.button("Revenir au début"):
    st.session_state["replay_position"] = 0


if col_btn2.button("Pas précédent"):
    st.session_state["replay_position"] = max(
        0,
        st.session_state["replay_position"] - 1,
    )


if col_btn3.button("Pas suivant"):
    st.session_state["replay_position"] = min(
        nombre_points - 1,
        st.session_state["replay_position"] + 1,
    )


lecture_auto = col_btn4.checkbox(
    "Lecture automatique"
)


# ============================================================
# Paramètres de lecture
# ============================================================

col_vit1, col_vit2 = st.columns(2)


with col_vit1:
    vitesse_lecture = st.slider(
        "Vitesse de lecture, en secondes entre deux pas",
        0.1,
        2.0,
        0.8,
        0.1,
        key="vitesse_lecture_slider",
        help=(
            "Plus la valeur sélectionnée est élevée, "
            "plus la lecture est lente."
        ),
    )


with col_vit2:
    pas_par_lecture = st.number_input(
        "Nombre de pas à lire par lancement",
        min_value=5,
        max_value=500,
        value=20,
        step=5,
        key="pas_par_lecture_input",
    )


position = st.slider(
    "Position dans le cycle",
    0,
    nombre_points - 1,
    st.session_state["replay_position"],
    key="replay_slider",
)


st.session_state["replay_position"] = position


# Les emplacements des métriques sont créés une seule fois, en dehors de toute
# boucle. La reconstruction répétée de st.columns() ou st.metric() pendant une
# lecture automatique peut provoquer l'erreur du navigateur :
# « NotFoundError: Failed to execute 'removeChild' on 'Node' ».
#
# Seul le contenu de chaque emplacement est donc actualisé à chaque pas.
# La structure de l'interface reste inchangée.

zone_temps = st.empty()

col_m1, col_m2, col_m3, col_m4 = st.columns(4)

ph_power = col_m1.empty()
ph_alpha = col_m2.empty()
ph_soc_eb = col_m3.empty()
ph_soc_pb = col_m4.empty()

col_m5, col_m6 = st.columns(2)

ph_p_eb = col_m5.empty()
ph_p_pb = col_m6.empty()

zone_caption = st.empty()
zone_alerte = st.empty()


def afficher_etat_instant(pos):
    """
    Affiche l'état du système à un instant précis de la simulation.
    """

    ligne = df.iloc[pos]

    alpha_val = float(
        traj_replay["alpha_final"][pos]
    )

    p_eb = float(
        traj_replay["P_EB"][pos]
    )

    p_pb = float(
        traj_replay["P_PB"][pos]
    )

    soc_eb = float(
        traj_replay["SOC_EB"][pos]
    )

    soc_pb = float(
        traj_replay["SOC_PB"][pos]
    )

    correction = bool(
        traj_replay["correction_applied"][pos]
    )

    alerte = None

    if np.isnan(alpha_val):
        alerte = (
            "Une valeur invalide a été détectée. "
            "Il est recommandé de basculer vers la stratégie "
            "EMS_power_limitation."
        )

    elif abs(p_eb) > 1e6 or abs(p_pb) > 1e6:
        alerte = (
            "Une valeur de puissance anormalement élevée a été détectée."
        )

    zone_temps.write(
        f"**Temps : {float(ligne['time']):.0f} s** "
        f"(pas {pos} sur {nombre_points - 1})"
    )

    ph_power.metric(
        "Puissance demandée",
        f"{float(ligne['hasPower']) / 1000:.2f} kW",
    )

    ph_alpha.metric(
        "Alpha appliqué",
        f"{alpha_val:.3f}",
    )

    ph_soc_eb.metric(
        "SOC de l'EB",
        f"{soc_eb:.3f}",
    )

    ph_soc_pb.metric(
        "SOC de la PB",
        f"{soc_pb:.3f}",
    )

    ph_p_eb.metric(
        "Puissance de l'EB",
        f"{p_eb / 1000:.2f} kW",
    )

    ph_p_pb.metric(
        "Puissance de la PB",
        f"{p_pb / 1000:.2f} kW",
    )

    zone_caption.caption(
        "Le filtre de sécurité a corrigé la décision proposée à cet instant."
        if correction
        else ""
    )

    if alerte:
        zone_alerte.error(
            alerte
        )

    else:
        zone_alerte.empty()


afficher_etat_instant(
    position
)


# ============================================================
# Lecture automatique
# ============================================================

if lecture_auto:
    fin = min(
        nombre_points - 1,
        position + int(pas_par_lecture),
    )

    for pos in range(
        position,
        fin + 1,
    ):
        st.session_state["replay_position"] = pos

        afficher_etat_instant(
            pos
        )

        time.sleep(
            float(vitesse_lecture)
        )

    st.session_state["replay_position"] = fin

    st.rerun()


# ============================================================
# 2. Présentation de la gestion des erreurs
# ============================================================

st.subheader("2. Gestion des erreurs simulées")

st.write(
    "En conditions réelles, plusieurs incidents peuvent se produire : "
    "une perte de connexion avec le simulateur, une sortie invalide d'un modèle "
    "ou encore un dépassement du délai de calcul. Dans ces situations, la "
    "stratégie de secours recommandée consiste à basculer immédiatement vers "
    "EMS_power_limitation, qui reste toujours calculable et respecte les "
    "contraintes physiques du système."
)


# ============================================================
# 3. Exportation des résultats
# ============================================================

st.subheader("3. Exportation des résultats")

st.write(
    "Choisis une stratégie précise (par exemple EMS_fuzzy_logic) ou l'ensemble "
    "des stratégies, ainsi que le format d'exportation souhaité."
)


def construire_dataframe_export(nom, traj):
    """Construit le DataFrame exportable pour une seule stratégie."""
    n = len(traj["P_EB"])
    return pd.DataFrame({
        "strategie": [nom] * n,
        "time": df["time"].to_numpy()[:n],
        "hasPower": df["hasPower"].to_numpy()[:n],
        "alpha_requested": traj["alpha_requested"],
        "alpha_final": traj["alpha_final"],
        "P_EB": traj["P_EB"],
        "P_PB": traj["P_PB"],
        "I_EB": traj["I_EB"],
        "I_PB": traj["I_PB"],
        "SOC_EB": traj["SOC_EB"][:n],
        "SOC_PB": traj["SOC_PB"][:n],
        "cost": traj["cost"],
        "correction_applied": traj["correction_applied"],
        "P_unserved": traj["P_unserved"],
        "P_regen_curtailed": traj["P_regen_curtailed"],
    })


col_choix1, col_choix2 = st.columns(2)

with col_choix1:
    options_strategie_export = ["Toutes les stratégies (fichier combiné)"] + list(resultats.keys())
    strategie_export = st.selectbox(
        "Stratégie à exporter",
        options_strategie_export,
        format_func=lambda n: n if n.startswith("Toutes") else MODEL_DISPLAY_NAMES.get(n, n),
        key="strategie_export_select",
    )

with col_choix2:
    format_export_resultats = st.radio(
        "Format du fichier",
        ["CSV (.csv)", "Excel (.xlsx)"],
        horizontal=True,
        key="format_export_resultats",
    )

if strategie_export.startswith("Toutes"):
    df_export = pd.concat(
        [construire_dataframe_export(nom, traj) for nom, traj in resultats.items()],
        ignore_index=True,
    )
    nom_fichier_base = "resultats_toutes_strategies"
else:
    df_export = construire_dataframe_export(strategie_export, resultats[strategie_export])
    nom_fichier_base = f"resultats_{strategie_export}"

if format_export_resultats.startswith("CSV"):
    csv_bytes = df_export.to_csv(index=False, sep=";").encode("utf-8-sig")
    st.download_button(
        f"Télécharger « {nom_fichier_base}.csv »",
        data=csv_bytes,
        file_name=f"{nom_fichier_base}.csv",
        mime="text/csv",
        use_container_width=True,
        key="download_export_csv",
    )
else:
    try:
        buffer_excel = io.BytesIO()
        with pd.ExcelWriter(buffer_excel, engine="xlsxwriter") as writer:
            df_export.to_excel(writer, index=False, sheet_name="Resultats")
        st.download_button(
            f"Télécharger « {nom_fichier_base}.xlsx »",
            data=buffer_excel.getvalue(),
            file_name=f"{nom_fichier_base}.xlsx",
            mime=(
                "application/vnd.openxmlformats-officedocument."
                "spreadsheetml.sheet"
            ),
            use_container_width=True,
            key="download_export_xlsx",
        )
    except ImportError:
        st.error(
            "L'exportation au format Excel n'est pas disponible. "
            "Installe le module `xlsxwriter` (`pip install xlsxwriter`), "
            "ou choisis le format CSV."
        )

st.caption(
    "La génération automatique du rapport PDF et l'exportation groupée "
    "des figures seront ajoutées une fois la mise en forme du rapport finalisée."
)