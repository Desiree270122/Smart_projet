import sys
import io
from pathlib import Path

DOSSIER_PROJET = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DOSSIER_PROJET))

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import ems_core as core
from ems_core import (
    simuler_strategie_deterministe,
    eb_priority_alpha_single,
)


st.set_page_config(
    page_title="Préparation des données",
    layout="wide",
)

st.title("Préparation des données")

st.write(
    "Importe un fichier de cycle de conduite au format CSV, TXT, TSV ou Excel, "
    "avec ou sans ligne d'en-tête. Les colonnes de temps, de vitesse, de puissance "
    "et d'accélération sont détectées automatiquement à partir de leur nom, en "
    "français ou en anglais. Tu peux ensuite modifier les choix proposés."
)

# ============================================================
# 1. Importation du fichier
# ============================================================

st.subheader("1. Importation du fichier")

sans_entete = st.checkbox(
    "Ce fichier ne contient qu'une seule colonne : le profil de vitesse brut, "
    "sans ligne d'en-tête",
    key="sans_entete_checkbox",
)

fichier = st.file_uploader(
    "Fichier du cycle de conduite",
    type=["csv", "txt", "tsv", "xlsx", "xls"],
    key="fichier_cycle_uploader",
)

if fichier is None:
    st.info("En attente d'un fichier.")
    st.stop()

est_excel = fichier.name.lower().endswith((".xlsx", ".xls"))

# Le sélecteur de feuille, lorsqu'il y en a plusieurs, est créé ici, à la racine
# du script. Il reste ainsi au même emplacement à chaque nouvelle exécution et
# n'est jamais placé dans une fonction ou dans un bloc conditionnel imbriqué dans
# un try/except. Cette organisation évite les comportements instables de l'interface.
feuilles = None

if est_excel:
    try:
        fichier.seek(0)
        feuilles = pd.ExcelFile(fichier).sheet_names
    except Exception as exc:
        st.error(f"Impossible de lire ce fichier Excel : {exc}")
        st.stop()

feuille_choisie = feuilles[0] if feuilles else None

if feuilles and len(feuilles) > 1:
    feuille_choisie = st.selectbox(
        "Feuille à utiliser",
        feuilles,
        key="feuille_select",
    )


def _charger_fichier(
    f,
    sans_entete: bool,
    feuille,
) -> pd.DataFrame:
    """
    Charge un fichier CSV, TXT, TSV ou Excel dans un DataFrame,
    sans créer de widget.
    """
    entete = None if sans_entete else 0
    f.seek(0)

    if feuille is not None:
        return pd.read_excel(
            f,
            sheet_name=feuille,
            header=entete,
        )

    return pd.read_csv(
        f,
        sep=None,
        engine="python",
        header=entete,
    )


try:
    df_brut = _charger_fichier(
        fichier,
        sans_entete,
        feuille_choisie,
    )

except Exception as exc:
    st.error(
        f"Impossible de lire ce fichier : {exc}\n\n"
        "Pour un fichier Excel, vérifie que le paquet `openpyxl` est installé."
    )
    st.stop()

if sans_entete:
    df_brut.columns = [
        f"col_{i}"
        for i in range(df_brut.shape[1])
    ]

    if df_brut.shape[1] == 1:
        df_brut.columns = ["speed"]

else:
    df_brut.columns = [
        str(c).strip()
        for c in df_brut.columns
    ]

    if all(str(c).isdigit() for c in df_brut.columns):
        st.warning(
            "Les colonnes de ce fichier n'ont pas de nom : l'en-tête est absent "
            "ou n'a pas été détecté. Elles sont donc numérotées 0, 1, 2, etc. "
            "S'il s'agit d'un profil de vitesse brut à une seule colonne, coche "
            "la case ci-dessus."
        )

st.success(
    f"Fichier chargé : {df_brut.shape[0]} lignes "
    f"et {df_brut.shape[1]} colonnes."
)

with st.expander("Aperçu des données brutes"):
    st.dataframe(
        df_brut.head(20),
        use_container_width=True,
    )

# ============================================================
# 2. Sélection des colonnes, unités, répétitions
#    et paramètres du véhicule
# ============================================================

st.caption(
    "Tous les réglages ci-dessous sont regroupés dans un formulaire. "
    "La page ne sera actualisée qu'après un clic sur « Préparer les données ». "
    "Cela évite les instabilités d'affichage lorsque plusieurs champs apparaissent "
    "ou disparaissent en même temps, notamment dans la section "
    "« Paramètres du véhicule »."
)

colonnes = list(df_brut.columns)

guess_time = core.guess_column(
    colonnes,
    "time",
)

guess_speed = core.guess_column(
    colonnes,
    "speed",
)

guess_power = core.guess_column(
    colonnes,
    "power",
)

guess_accel = core.guess_column(
    colonnes,
    "acceleration",
)

with st.form(key="prep_form"):
    st.subheader("2. Sélection des colonnes")

    c1, c2 = st.columns(2)

    with c1:
        options_vitesse = ["(aucune)"] + colonnes

        index_vitesse = (
            options_vitesse.index(guess_speed)
            if guess_speed in colonnes
            else 0
        )

        speed_choice = st.selectbox(
            "Colonne de vitesse",
            options_vitesse,
            index=index_vitesse,
            key="speed_col_select",
        )

        speed_col = (
            None
            if speed_choice == "(aucune)"
            else speed_choice
        )

    with c2:
        options_puissance = [
            "(aucune — calculer à partir de la dynamique du véhicule)"
        ] + colonnes

        index_puissance = (
            options_puissance.index(guess_power)
            if guess_power in colonnes
            else 0
        )

        power_choice = st.selectbox(
            "Colonne de puissance demandée",
            options_puissance,
            index=index_puissance,
            key="power_col_select",
        )

        power_col = (
            None
            if power_choice.startswith("(aucune")
            else power_choice
        )

    c3, c4 = st.columns(2)

    with c3:
        options_accel = [
            "(aucune — considérée comme nulle)"
        ] + colonnes

        index_accel = (
            options_accel.index(guess_accel)
            if guess_accel in colonnes
            else 0
        )

        accel_choice = st.selectbox(
            "Colonne d'accélération (facultative)",
            options_accel,
            index=index_accel,
            key="accel_col_select",
        )

        accel_col = (
            None
            if accel_choice.startswith("(aucune")
            else accel_choice
        )

    with c4:
        pas_de_temps = df_brut.shape[1] == 1

        sans_colonne_temps = st.checkbox(
            "Aucune colonne de temps "
            "(échantillonnage à fréquence constante)",
            value=pas_de_temps,
            key="sans_temps_checkbox",
        )

    tc1, tc2 = st.columns(2)

    with tc1:
        frequence_hz = st.number_input(
            "Fréquence d'échantillonnage (Hz), utilisée en l'absence "
            "de colonne de temps",
            min_value=0.01,
            max_value=1000.0,
            value=1.0,
            step=0.1,
            help="1 Hz = un point par seconde (cas typique WLTC/Artemis).",
            key="freq_input",
        )

    with tc2:
        options_temps = colonnes

        index_temps = (
            options_temps.index(guess_time)
            if guess_time in colonnes
            else 0
        )

        time_col_choice = st.selectbox(
            "Colonne de temps, utilisée si le fichier en contient une",
            options_temps,
            index=index_temps,
            key="time_col_select",
        )

    time_col = (
        None
        if sans_colonne_temps
        else time_col_choice
    )

    st.subheader("3. Unité de la vitesse")

    if speed_col is not None:
        try:
            valeurs_vitesse = pd.to_numeric(
                df_brut[speed_col],
                errors="coerce",
            ).to_numpy(dtype=float)

            if pd.isna(valeurs_vitesse).all():
                raise ValueError(
                    "aucune valeur numérique dans la colonne sélectionnée"
                )

            vitesse_valide = True

        except Exception as exc:
            st.error(
                f"La colonne de vitesse « {speed_col} » "
                f"n'est pas numérique ({exc})."
            )

            valeurs_vitesse = None
            vitesse_valide = False

        if vitesse_valide:
            unite_detectee = core.detect_speed_unit(
                valeurs_vitesse
            )

            st.caption(
                f"Unité détectée automatiquement : **{unite_detectee}** "
                f"(vitesse maximale ≈ "
                f"{pd.Series(valeurs_vitesse).abs().max():.1f} "
                f"dans l'unité d'origine)."
            )

        else:
            unite_detectee = "km/h"

        unite_vitesse = st.radio(
            "Unité de la colonne de vitesse",
            ["km/h", "m/s"],
            index=["km/h", "m/s"].index(unite_detectee),
            horizontal=True,
            key="unite_radio",
        )

    else:
        vitesse_valide = False
        unite_vitesse = "km/h"

        st.caption(
            "Aucune colonne de vitesse sélectionnée : "
            "cette section est ignorée."
        )

    st.subheader("4. Répétition du cycle")

    if vitesse_valide and speed_col is not None:
        vitesse_ms_apercu = core.convert_speed_to_ms(
            valeurs_vitesse,
            unite_vitesse,
        )

        repetition_auto = core.detect_repetition(
            vitesse_ms_apercu
        )

        st.caption(
            "Détection automatique à titre indicatif : "
            f"le motif semble être répété {repetition_auto} fois."
        )

    else:
        repetition_auto = 1

    nombre_repetitions = st.number_input(
        "Nombre de répétitions du cycle à appliquer",
        min_value=1,
        max_value=50,
        value=int(repetition_auto),
        step=1,
        help=(
            "Indique la valeur exacte connue pour ton fichier plutôt que "
            "de te fier uniquement à la détection automatique."
        ),
        key="repetitions_input",
    )

    st.subheader(
        "5. Paramètres du véhicule "
        "(utilisés uniquement si aucune colonne de puissance n'est sélectionnée)"
    )

    st.caption(
        "La puissance est calculée à partir des forces aérodynamiques, "
        "de roulement, de gravité et d'accélération. Les valeurs par défaut "
        "ont été validées lors du cadrage du projet et peuvent être modifiées "
        "ci-dessous."
    )

    vc1, vc2, vc3 = st.columns(3)

    with vc1:
        masse = st.number_input(
            "Masse (kg)",
            min_value=1.0,
            value=core.VEHICLE_MASS_KG,
            step=10.0,
            key="masse_input",
        )

        cx = st.number_input(
            "Coefficient de traînée Cx",
            min_value=0.0,
            value=core.DRAG_COEFFICIENT_CX,
            step=0.01,
            key="cx_input",
        )

    with vc2:
        surface_frontale = st.number_input(
            "Surface frontale S (m²)",
            min_value=0.0,
            value=core.FRONTAL_AREA_M2,
            step=0.05,
            key="surface_input",
        )

        c0 = st.number_input(
            "Coefficient de roulement C0",
            min_value=0.0,
            value=core.ROLLING_C0,
            step=0.001,
            format="%.4f",
            key="c0_input",
        )

    with vc3:
        c1 = st.number_input(
            "Coefficient de roulement C1",
            min_value=0.0,
            value=core.ROLLING_C1,
            step=1e-7,
            format="%.2e",
            key="c1_input",
        )

        pente_deg = st.number_input(
            "Pente de la route (°)",
            value=0.0,
            step=0.5,
            key="pente_input",
        )

    rho = st.number_input(
        "Densité de l'air (kg/m³)",
        min_value=0.0,
        value=core.AIR_DENSITY_KG_M3,
        step=0.005,
        format="%.3f",
        key="rho_input",
    )

    valide = st.form_submit_button(
        "Préparer les données",
        type="primary",
    )

if not valide and "cycle_pret" not in st.session_state:
    st.stop()

if valide:
    if speed_col is None and power_col is None:
        st.error(
            "Sélectionne au moins une colonne de vitesse "
            "ou une colonne de puissance."
        )
        st.stop()

    if speed_col is not None and not vitesse_valide:
        st.error(
            "Corrige la colonne de vitesse avant de préparer les données."
        )
        st.stop()

    with st.spinner("Préparation en cours..."):
        # ----------------------------------------------------
        # Temps et vitesse du cycle de base, avant répétition
        # ----------------------------------------------------

        if time_col is not None:
            temps_brut = pd.to_numeric(
                df_brut[time_col],
                errors="coerce",
            ).to_numpy(dtype=float)

            pas_de_temps_s = (
                float(np.median(np.diff(temps_brut)))
                if len(temps_brut) > 1
                else 1.0
            )

            if pas_de_temps_s <= 0 or np.isnan(pas_de_temps_s):
                pas_de_temps_s = 1.0

        else:
            pas_de_temps_s = 1.0 / frequence_hz

        n_base = len(df_brut)

        temps_base = (
            np.arange(n_base)
            * pas_de_temps_s
        )

        df_base = pd.DataFrame()

        if speed_col is not None:
            df_base["speed"] = core.convert_speed_to_ms(
                pd.to_numeric(
                    df_brut[speed_col],
                    errors="coerce",
                ).to_numpy(dtype=float),
                unite_vitesse,
            )

        if power_col is not None:
            df_base["hasPower"] = pd.to_numeric(
                df_brut[power_col],
                errors="coerce",
            ).to_numpy(dtype=float)

            if accel_col is not None:
                df_base["hasAcceleration"] = pd.to_numeric(
                    df_brut[accel_col],
                    errors="coerce",
                ).to_numpy(dtype=float)

            elif speed_col is not None:
                dv = np.diff(
                    df_base["speed"].to_numpy()
                )

                dt_arr = np.diff(
                    temps_base
                )

                acc = np.divide(
                    dv,
                    dt_arr,
                    out=np.zeros_like(dv),
                    where=dt_arr != 0,
                )

                df_base["hasAcceleration"] = np.concatenate(
                    ([0.0], acc)
                )

            else:
                df_base["hasAcceleration"] = 0.0

        else:
            forces = core.compute_forces_and_power(
                df_base["speed"].to_numpy(),
                temps_base,
                mass=masse,
                cx=cx,
                frontal_area=surface_frontale,
                c0=c0,
                c1=c1,
                slope_rad=math.radians(pente_deg),
                rho=rho,
                gravity=core.GRAVITY_MS2,
            )

            for col, valeurs in forces.items():
                df_base[col] = valeurs

        # ----------------------------------------------------
        # Répétition du cycle déjà calculé,
        # sans nouvelle dérivation
        # ----------------------------------------------------

        n_rep = int(nombre_repetitions)

        if n_rep > 1:
            df_cycle = pd.concat(
                [
                    df_base.copy()
                    for _ in range(n_rep)
                ],
                ignore_index=True,
            )

        else:
            df_cycle = df_base.copy()

        df_cycle.insert(
            0,
            "time",
            np.arange(
                len(df_cycle),
                dtype=float,
            ) * pas_de_temps_s,
        )

    st.session_state["cycle_pret"] = df_cycle

    st.session_state["prep_meta"] = {
        "n_points": len(df_cycle),
        "n_repetitions": n_rep,
        "puissance_source": (
            "colonne fournie"
            if power_col is not None
            else "calculée à partir de la dynamique du véhicule"
        ),
    }

    st.success(
        f"Données préparées : {len(df_cycle)} points au total."
    )

if "cycle_pret" not in st.session_state:
    st.stop()

df_cycle = st.session_state["cycle_pret"]

meta = st.session_state.get(
    "prep_meta",
    {},
)

st.divider()

# ============================================================
# 6. Formules utilisées (dynamique du véhicule + architecture batteries)
# ============================================================

st.subheader("6. Formules utilisées")

with st.expander("Récapitulatif des formules (dynamique du véhicule et packs batteries)"):
    st.markdown(
        """
**Dynamique longitudinale du véhicule** (si la puissance n'est pas fournie directement) :

- Accélération : `a = dv/dt`
- Force aérodynamique : `F_aero = 0.5 * rho * S * Cx * v²`
- Force de roulement : `F_roulement = m * g * (C0 + C1 * v²)`
- Force de gravité : `F_gravite = m * g * sin(pente)`
- Force d'accélération : `F_acceleration = m * a`
- Force totale : `F_totale = F_aero + F_roulement + F_gravite + F_acceleration`
- Puissance demandée : `hasPower = F_totale * v`

**Architecture des packs batteries** (bloc 7 → bloc 8) :

- Tension : `Tension = V_cellule * n_serie`
- Masse : `Masse = masse_cellule * n_serie * n_parallele`
- Puissance décharge max : `P_decharge = I_decharge_cellule * Tension * n_parallele`
- Puissance recharge max : `P_recharge = I_recharge_cellule * Tension * n_parallele`
- Énergie : `Energie = DE * Masse`

*Formule Énergie = DE * Masse confirmée correcte par l'encadrant.*
"""
    )

# ============================================================
# 7. Architecture des batteries -- nombre de cellules, masses
# ============================================================

st.subheader("7. Architecture des batteries")

st.write(
    "Paramètres cellule et architecture (nombre de cellules en série / en "
    "parallèle) pour chacune des deux batteries. Valeurs par défaut reprises "
    "du cadrage du projet -- modifiables ci-dessous."
)

col_eb, col_pb = st.columns(2)

with col_eb:
    st.markdown("**Batterie d'énergie (EB)**")
    eb_n_serie = st.number_input("Cellules en série (EB)", min_value=1, value=core.CELL_EB_N_SERIE, step=1, key="eb_n_serie")
    eb_n_parallele = st.number_input("Cellules en parallèle (EB)", min_value=1, value=core.CELL_EB_N_PARALLELE, step=1, key="eb_n_parallele")
    eb_masse_cellule = st.number_input("Masse par cellule (kg, EB)", min_value=0.0, value=core.CELL_EB_MASSE_KG, step=0.001, format="%.3f", key="eb_masse_cellule")
    eb_v_cellule = st.number_input("Tension par cellule (V, EB)", min_value=0.0, value=core.CELL_EB_V_CELLULE, step=0.1, key="eb_v_cellule")
    eb_i_decharge = st.number_input("Courant décharge par cellule (A, EB)", value=core.CELL_EB_I_DECHARGE_A, step=0.1, key="eb_i_decharge")
    eb_i_recharge = st.number_input("Courant recharge par cellule (A, EB)", value=core.CELL_EB_I_RECHARGE_A, step=0.1, key="eb_i_recharge")
    eb_de = st.number_input("Densité d'énergie DE (Wh/kg, EB)", min_value=0.0, value=core.CELL_EB_DE_WH_KG, step=1.0, key="eb_de")
    eb_capacite = st.number_input("Capacité par cellule (Ah, EB)", min_value=0.0, value=core.CELL_EB_CAPACITE_AH, step=0.1, key="eb_capacite")
    eb_rint = st.number_input("Résistance interne par cellule (Ohm, EB)", min_value=0.0, value=core.CELL_EB_RINT_OHM, step=0.001, format="%.4f", key="eb_rint")

with col_pb:
    st.markdown("**Batterie de puissance (PB)**")
    pb_n_serie = st.number_input("Cellules en série (PB)", min_value=1, value=core.CELL_PB_N_SERIE, step=1, key="pb_n_serie")
    pb_n_parallele = st.number_input("Cellules en parallèle (PB)", min_value=1, value=core.CELL_PB_N_PARALLELE, step=1, key="pb_n_parallele")
    pb_masse_cellule = st.number_input("Masse par cellule (kg, PB)", min_value=0.0, value=core.CELL_PB_MASSE_KG, step=0.001, format="%.3f", key="pb_masse_cellule")
    pb_v_cellule = st.number_input("Tension par cellule (V, PB)", min_value=0.0, value=core.CELL_PB_V_CELLULE, step=0.1, key="pb_v_cellule")
    pb_i_decharge = st.number_input("Courant décharge par cellule (A, PB)", value=core.CELL_PB_I_DECHARGE_A, step=0.1, key="pb_i_decharge")
    pb_i_recharge = st.number_input("Courant recharge par cellule (A, PB)", value=core.CELL_PB_I_RECHARGE_A, step=0.1, key="pb_i_recharge")
    pb_de = st.number_input("Densité d'énergie DE (Wh/kg, PB)", min_value=0.0, value=core.CELL_PB_DE_WH_KG, step=1.0, key="pb_de")
    pb_capacite = st.number_input("Capacité par cellule (Ah, PB)", min_value=0.0, value=core.CELL_PB_CAPACITE_AH, step=0.1, key="pb_capacite")
    pb_rint = st.number_input("Résistance interne par cellule (Ohm, PB)", min_value=0.0, value=core.CELL_PB_RINT_OHM, step=0.001, format="%.4f", key="pb_rint")

st.caption(
    f"Nombre total de composants : EB = {int(eb_n_serie * eb_n_parallele)} cellules "
    f"({eb_n_serie} en série × {eb_n_parallele} en parallèle), "
    f"PB = {int(pb_n_serie * pb_n_parallele)} cellules "
    f"({pb_n_serie} en série × {pb_n_parallele} en parallèle)."
)

# ============================================================
# 8. Résultats -- caractéristiques calculées des packs
# ============================================================

st.subheader("8. Résultats des packs batteries")

resultats_eb = core.compute_pack_characteristics(
    eb_v_cellule, eb_i_decharge, eb_i_recharge, eb_masse_cellule, eb_de, eb_n_serie, eb_n_parallele,
    capacite_cellule_ah=eb_capacite,
)
resultats_pb = core.compute_pack_characteristics(
    pb_v_cellule, pb_i_decharge, pb_i_recharge, pb_masse_cellule, pb_de, pb_n_serie, pb_n_parallele,
    capacite_cellule_ah=pb_capacite,
)

# Applique reellement ces caracteristiques au moteur de simulation (comme pour
# le convertisseur au bloc 9) : V_EB_PACK_NOM, P_EB_MIN_W, P_EB_MAX_W,
# CAPACITY_EB_AH et ENERGY_EB_WH (et les memes pour PB) sont mis a jour pour
# toutes les simulations qui suivent, pas seulement pour cet affichage.
core.set_battery_pack_parameters("EB", resultats_eb)
core.set_battery_pack_parameters("PB", resultats_pb)

tableau_resultats = pd.DataFrame([
    {
        "Batterie": "Énergie (EB)",
        "Tension (V)": round(resultats_eb["tension_V"], 2),
        "Masse (kg)": round(resultats_eb["masse_kg"], 2),
        "Capacité (Ah)": round(resultats_eb.get("capacite_Ah", float("nan")), 3),
        "P décharge max (W)": round(resultats_eb["puissance_decharge_W"], 2),
        "P recharge max (W)": round(resultats_eb["puissance_recharge_W"], 2),
        "Énergie (Wh)": round(resultats_eb["energie_Wh"], 2),
    },
    {
        "Batterie": "Puissance (PB)",
        "Tension (V)": round(resultats_pb["tension_V"], 2),
        "Masse (kg)": round(resultats_pb["masse_kg"], 2),
        "Capacité (Ah)": round(resultats_pb.get("capacite_Ah", float("nan")), 3),
        "P décharge max (W)": round(resultats_pb["puissance_decharge_W"], 2),
        "P recharge max (W)": round(resultats_pb["puissance_recharge_W"], 2),
        "Énergie (Wh)": round(resultats_pb["energie_Wh"], 2),
    },
    {
        "Batterie": "Couple de batteries",
        "Tension (V)": round(resultats_pb["tension_V"], 2),
        "Masse (kg)": round(resultats_eb["masse_kg"] + resultats_pb["masse_kg"], 2),
        "Capacité (Ah)": None,
        "P décharge max (W)": round(resultats_eb["puissance_decharge_W"] + resultats_pb["puissance_decharge_W"], 2),
        "P recharge max (W)": round(resultats_eb["puissance_recharge_W"] + resultats_pb["puissance_recharge_W"], 2),
        "Énergie (Wh)": round(resultats_eb["energie_Wh"] + resultats_pb["energie_Wh"], 2),
    },
])

st.dataframe(tableau_resultats, use_container_width=True, hide_index=True)

st.session_state["architecture_batteries"] = {
    "EB": {"n_serie": eb_n_serie, "n_parallele": eb_n_parallele, "masse_cellule": eb_masse_cellule,
           "v_cellule": eb_v_cellule, "i_decharge": eb_i_decharge, "i_recharge": eb_i_recharge,
           "de": eb_de, "capacite": eb_capacite, "rint": eb_rint, **resultats_eb},
    "PB": {"n_serie": pb_n_serie, "n_parallele": pb_n_parallele, "masse_cellule": pb_masse_cellule,
           "v_cellule": pb_v_cellule, "i_decharge": pb_i_decharge, "i_recharge": pb_i_recharge,
           "de": pb_de, "capacite": pb_capacite, "rint": pb_rint, **resultats_pb},
}

# ============================================================
# 9. Convertisseur -- nombre de composants et puissance associee
# ============================================================

st.divider()
st.subheader("9. Convertisseur")

st.write(
    "La puissance du convertisseur n'est pas une valeur fixe : elle dépend du "
    "nombre de composants (modules) installés en parallèle. Les valeurs par "
    "défaut ci-dessous (1 composant) reproduisent les limites utilisées jusqu'ici "
    "(1520 W en décharge, −760 W en recharge) -- à ajuster si l'architecture "
    "réelle comporte plusieurs composants."
)

col_conv1, col_conv2, col_conv3 = st.columns(3)
with col_conv1:
    conv_n_composants = st.number_input(
        "Nombre de composants du convertisseur", min_value=1,
        value=core.CONVERTER_N_COMPOSANTS, step=1, key="conv_n_composants",
    )
with col_conv2:
    conv_p_decharge = st.number_input(
        "Puissance décharge max par composant (W)", min_value=0.0,
        value=core.CONVERTER_P_DECHARGE_PAR_COMPOSANT_W, step=10.0, key="conv_p_decharge",
    )
with col_conv3:
    conv_p_recharge = st.number_input(
        "Puissance recharge max par composant (W)",
        value=core.CONVERTER_P_RECHARGE_PAR_COMPOSANT_W, step=10.0, key="conv_p_recharge",
    )

resultats_conv = core.compute_converter_characteristics(conv_n_composants, conv_p_decharge, conv_p_recharge)

# Applique reellement ces limites au moteur de simulation (pas seulement un
# affichage) : candidate_metrics et resoudre_decision_physique liront ces
# valeurs a chaque appel, pour toutes les strategies simulees ensuite.
core.set_converter_power_limits(resultats_conv["p_decharge_W"], resultats_conv["p_recharge_W"])

col_res1, col_res2 = st.columns(2)
col_res1.metric("Puissance totale décharge", f"{resultats_conv['p_decharge_W']:.0f} W")
col_res2.metric("Puissance totale recharge", f"{resultats_conv['p_recharge_W']:.0f} W")

st.caption(
    f"Ces {int(conv_n_composants)} composant(s) définissent les bornes "
    f"P_CONV_MIN_W = {resultats_conv['p_recharge_W']:.0f} W et "
    f"P_CONV_MAX_W = {resultats_conv['p_decharge_W']:.0f} W réellement utilisées "
    f"par le filtre de sécurité physique dans toutes les simulations qui suivent."
)

st.session_state["architecture_convertisseur"] = {
    "n_composants": conv_n_composants,
    "p_decharge_par_composant": conv_p_decharge,
    "p_recharge_par_composant": conv_p_recharge,
    **resultats_conv,
}

# ============================================================
# À partir d'ici : visualisation et export, une fois toute la
# configuration (colonnes, véhicule, batteries bloc 7-8, convertisseur
# bloc 9) terminée. Placé après le bloc 9 pour que l'aperçu SOC utilise
# bien les paramètres batteries/convertisseur tels que configurés
# ci-dessus, plutôt que les valeurs par défaut.
# ============================================================

st.divider()

m1, m2, m3 = st.columns(3)

m1.metric(
    "Points totaux",
    meta.get(
        "n_points",
        len(df_cycle),
    ),
)

m2.metric(
    "Répétitions",
    meta.get(
        "n_repetitions",
        1,
    ),
)

m3.metric(
    "Source de la puissance",
    meta.get(
        "puissance_source",
        "?",
    ),
)

p = df_cycle["hasPower"]

if abs(p.mean()) < 50 and p.max() < 200:
    st.warning(
        "La puissance moyenne calculée est presque nulle (< 50 W). "
        "Vérifie l'unité de vitesse, la colonne sélectionnée et la fréquence "
        "d'échantillonnage lorsque le fichier ne contient pas de colonne de temps. "
        "Dans le cas contraire, le SOC variera très peu pendant la simulation."
    )

st.subheader("Téléchargement des données préparées")

st.write(
    "Le fichier contient le temps, la vitesse (si disponible), et toutes les "
    "colonnes calculées lors de la préparation : accélération, forces "
    "(aérodynamique, roulement, gravité, accélération, totale) et puissance "
    "demandée, lorsque celles-ci ont été calculées via la dynamique du véhicule."
)

colonnes_export = [c for c in [
    "time", "speed", "hasAcceleration", "hasAeroForce", "hasRollingForce",
    "hasGravityForce", "hasAccelerationForce", "hasTotalForce", "hasPower",
] if c in df_cycle.columns]
colonnes_export += [c for c in df_cycle.columns if c not in colonnes_export]

format_export = st.radio(
    "Format du fichier",
    ["Excel (.xlsx)", "CSV (.csv)"],
    horizontal=True,
    key="format_export_cycle",
    help="Excel place chaque colonne dans une vraie colonne de feuille de calcul "
         "(A = temps, B = vitesse, etc.). Le CSV utilise le point-virgule comme "
         "séparateur, compatible avec Excel en français.",
)

if format_export == "Excel (.xlsx)":
    try:
        buffer_excel = io.BytesIO()
        with pd.ExcelWriter(buffer_excel, engine="xlsxwriter") as writer:
            df_cycle[colonnes_export].to_excel(writer, index=False, sheet_name="Cycle")
        st.download_button(
            "Télécharger les données préparées (Excel)",
            data=buffer_excel.getvalue(),
            file_name="cycle_prepare.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="download_cycle_prepare_xlsx",
        )
    except ImportError:
        st.error(
            "Export Excel indisponible : installe le module xlsxwriter "
            "(`pip install xlsxwriter`) pour l'activer, ou choisis le format CSV."
        )
else:
    csv_cycle = df_cycle[colonnes_export].to_csv(index=False, sep=";").encode("utf-8-sig")
    st.download_button(
        "Télécharger les données préparées (CSV)",
        data=csv_cycle,
        file_name="cycle_prepare.csv",
        mime="text/csv",
        key="download_cycle_prepare_csv",
    )

st.subheader("Aperçu général du cycle")

fig_brut, axes_brut = plt.subplots(
    1,
    2,
    figsize=(12, 4),
)

if "speed" in df_cycle.columns:
    axes_brut[0].plot(
        df_cycle["time"],
        df_cycle["speed"],
    )

    axes_brut[0].set_title(
        "Vitesse (m/s)"
    )

else:
    axes_brut[0].plot(
        df_cycle["time"],
        df_cycle["hasAcceleration"],
    )

    axes_brut[0].set_title(
        "Accélération"
    )

axes_brut[0].set_xlabel(
    "Temps (s)"
)

axes_brut[1].plot(
    df_cycle["time"],
    df_cycle["hasPower"],
)

axes_brut[1].set_title(
    "Puissance demandée (W)"
)

axes_brut[1].set_xlabel(
    "Temps (s)"
)

plt.tight_layout()

st.pyplot(
    fig_brut
)

plt.close(
    fig_brut
)

st.subheader(
    "Aperçu indicatif — SOC, puissances et courants "
    "(référence : EMS_power_limitation)"
)

st.write(
    "Ce calcul utilise la stratégie EMS_power_limitation, qui reste disponible "
    "sans poids entraînés, et reflète les paramètres batteries et convertisseur "
    "tels que configurés ci-dessus (blocs 7 à 9). Il fournit un premier aperçu "
    "du comportement du système sur ce cycle. La comparaison complète des sept "
    "stratégies est présentée sur la page « Simulation globale »."
)

col_soc0_1, col_soc0_2 = st.columns(2)

soc_eb0_apercu = col_soc0_1.slider(
    "SOC_EB initial pour l'aperçu",
    0.20,
    1.0,
    1.0,
    0.01,
    key="apercu_soc_eb0",
)

soc_pb0_apercu = col_soc0_2.slider(
    "SOC_PB initial pour l'aperçu",
    0.20,
    1.0,
    1.0,
    0.01,
    key="apercu_soc_pb0",
)

traj_apercu = simuler_strategie_deterministe(
    df_cycle,
    soc_eb0_apercu,
    soc_pb0_apercu,
    proposer_alpha=(
        lambda t, ligne, soc_eb, soc_pb, alpha_prev:
        eb_priority_alpha_single(
            ligne["hasPower"],
            soc_eb,
        )
    ),
)

fig_apercu, axes_apercu = plt.subplots(
    3,
    1,
    figsize=(11, 9),
    sharex=True,
)

axes_apercu[0].plot(
    df_cycle["time"],
    traj_apercu["SOC_EB"][:-1],
    label="SOC_EB",
)

axes_apercu[0].plot(
    df_cycle["time"],
    traj_apercu["SOC_PB"][:-1],
    label="SOC_PB",
)

axes_apercu[0].set_ylabel(
    "SOC"
)

axes_apercu[0].legend()

axes_apercu[0].grid(
    True,
    alpha=0.3,
)

axes_apercu[1].plot(
    df_cycle["time"],
    traj_apercu["P_EB"],
    label="P_EB",
)

axes_apercu[1].plot(
    df_cycle["time"],
    traj_apercu["P_PB"],
    label="P_PB",
)

axes_apercu[1].set_ylabel(
    "Puissance (W)"
)

axes_apercu[1].legend()

axes_apercu[1].grid(
    True,
    alpha=0.3,
)

axes_apercu[2].plot(
    df_cycle["time"],
    traj_apercu["I_EB"],
    label="I_EB",
)

axes_apercu[2].plot(
    df_cycle["time"],
    traj_apercu["I_PB"],
    label="I_PB",
)

axes_apercu[2].set_ylabel(
    "Courant (A)"
)

axes_apercu[2].set_xlabel(
    "Temps (s)"
)

axes_apercu[2].legend()

axes_apercu[2].grid(
    True,
    alpha=0.3,
)

plt.tight_layout()

st.pyplot(
    fig_apercu
)

plt.close(
    fig_apercu
)

st.session_state["soc_eb0"] = soc_eb0_apercu
st.session_state["soc_pb0"] = soc_pb0_apercu

st.divider()
if st.button("Passer à la simulation globale", type="primary"):
    st.switch_page("pages/3_Simulation_globale.py")