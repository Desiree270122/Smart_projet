"""
Préparer une simulation : importer un cycle de conduite, choisir ses colonnes,
calculer la puissance demandée si besoin, et régler les batteries et le
convertisseur. Le cycle préparé est ensuite simulé depuis « Lancer une simulation ».
"""

import io
import math
import sys
from pathlib import Path

DOSSIER_PROJET = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DOSSIER_PROJET))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import ems_core as core
from ems_core import eb_priority_alpha_single, simuler_strategie_deterministe
from core.format import nombre, separateurs_plotly
from core.i18n import tr
from core.navigation import pied_navigation
from core.style import COULEUR_CONVERTISSEUR, COULEUR_DEMANDE, COULEUR_EB, COULEUR_PB, COULEUR_SECONDAIRE, COULEUR_VIOLATION


AUCUNE = "__aucune__"   # choix « pas de colonne » dans les listes de colonnes

st.title(tr("📂 Préparer une simulation", "📂 Prepare a simulation"))
st.write(tr(
    "Importez un fichier de cycle de conduite au format CSV, TXT, TSV ou Excel, avec ou sans "
    "ligne d'en-tête. Les colonnes de temps, de vitesse, de puissance et d'accélération sont "
    "reconnues d'après leur nom, en français ou en anglais ; vous pouvez ensuite corriger ces choix.",
    "Import a driving cycle file in CSV, TXT, TSV or Excel format, with or without a header "
    "row. The time, speed, power and acceleration columns are recognised from their names, in "
    "French or English; you can then correct these choices.",
))


# 1. Importation du fichier

st.subheader(tr("1. Importation du fichier", "1. File import"))
sans_entete = st.checkbox(
    tr(
        "Ce fichier ne contient qu'une seule colonne : le profil de vitesse brut, sans ligne d'en-tête",
        "This file contains a single column: the raw speed profile, without a header row",
    ),
    key="sans_entete_checkbox",
)
fichier = st.file_uploader(
    tr("Fichier du cycle de conduite", "Driving cycle file"), type=["csv", "txt", "tsv", "xlsx", "xls"],
    key="fichier_cycle_uploader",
)
if fichier is None:
    st.info(tr("En attente d'un fichier.", "Waiting for a file."))
    st.stop()

est_excel = fichier.name.lower().endswith((".xlsx", ".xls"))

# Le sélecteur de feuille est créé ici, à la racine du script : il reste au même
# emplacement à chaque exécution, ce qui évite les comportements instables.
feuilles = None
if est_excel:
    try:
        fichier.seek(0)
        feuilles = pd.ExcelFile(fichier).sheet_names
    except Exception as exc:  # noqa: BLE001
        st.error(tr("Impossible de lire ce fichier Excel ({e}).", "This Excel file cannot be read ({e}).", e=exc))
        st.stop()

feuille_choisie = feuilles[0] if feuilles else None
if feuilles and len(feuilles) > 1:
    feuille_choisie = st.selectbox(tr("Feuille à utiliser", "Sheet to use"), feuilles, key="feuille_select")


def _charger_fichier(f, sans_entete: bool, feuille) -> pd.DataFrame:
    """Charge un fichier CSV, TXT, TSV ou Excel, sans créer de widget."""
    entete = None if sans_entete else 0
    f.seek(0)
    if feuille is not None:
        return pd.read_excel(f, sheet_name=feuille, header=entete)
    return pd.read_csv(f, sep=None, engine="python", header=entete)


try:
    df_brut = _charger_fichier(fichier, sans_entete, feuille_choisie)
except Exception as exc:  # noqa: BLE001
    st.error(tr("Impossible de lire ce fichier ({e}).", "This file cannot be read ({e}).", e=exc))
    st.stop()

if sans_entete:
    df_brut.columns = [f"col_{i}" for i in range(df_brut.shape[1])]
    if df_brut.shape[1] == 1:
        df_brut.columns = ["speed"]
else:
    df_brut.columns = [str(c).strip() for c in df_brut.columns]
    if all(str(c).isdigit() for c in df_brut.columns):
        st.warning(tr(
            "Les colonnes de ce fichier n'ont pas de nom : l'en-tête est absent ou n'a pas été "
            "reconnu, elles sont donc numérotées 0, 1, 2… S'il s'agit d'un profil de vitesse brut à "
            "une seule colonne, cochez la case ci-dessus.",
            "The columns of this file have no name: the header is missing or was not recognised, "
            "so they are numbered 0, 1, 2… If it is a raw single-column speed profile, tick the "
            "box above.",
        ))

st.success(tr(
    "Fichier chargé : {l} lignes et {c} colonnes.", "File loaded: {l} rows and {c} columns.",
    l=df_brut.shape[0], c=df_brut.shape[1],
))
with st.expander(tr("Aperçu des données brutes", "Raw data preview")):
    st.dataframe(df_brut.head(20), width="stretch")


# 2 à 5. Colonnes, unités, répétitions et véhicule, dans un formulaire

st.caption(tr(
    "Les réglages ci-dessous sont regroupés : la page ne se met à jour qu'après un clic sur "
    "« Préparer les données ».",
    "The settings below are grouped: the page only updates after clicking “Prepare the data”.",
))

colonnes = list(df_brut.columns)
guess_time = core.guess_column(colonnes, "time")
guess_speed = core.guess_column(colonnes, "speed")
guess_power = core.guess_column(colonnes, "power")
guess_accel = core.guess_column(colonnes, "acceleration")


def _choix_colonne(libelle, devine, texte_aucune, cle):
    options = [AUCUNE] + colonnes
    choix = st.selectbox(
        libelle, options, index=options.index(devine) if devine in colonnes else 0,
        format_func=lambda c: texte_aucune if c == AUCUNE else c, key=cle,
    )
    return None if choix == AUCUNE else choix


with st.form(key="prep_form"):
    st.subheader(tr("2. Sélection des colonnes", "2. Column selection"))
    c1, c2 = st.columns(2)
    with c1:
        speed_col = _choix_colonne(tr("Colonne de vitesse", "Speed column"), guess_speed,
                                   tr("(aucune)", "(none)"), "speed_col_select")
    with c2:
        power_col = _choix_colonne(
            tr("Colonne de puissance demandée", "Power demand column"), guess_power,
            tr("(aucune — calculer à partir de la dynamique du véhicule)", "(none — compute from the vehicle dynamics)"),
            "power_col_select",
        )
    c3, c4 = st.columns(2)
    with c3:
        accel_col = _choix_colonne(
            tr("Colonne d'accélération (facultative)", "Acceleration column (optional)"), guess_accel,
            tr("(aucune — considérée comme nulle)", "(none — taken as zero)"), "accel_col_select",
        )
    with c4:
        sans_colonne_temps = st.checkbox(
            tr("Aucune colonne de temps (échantillonnage à fréquence constante)",
               "No time column (constant sampling rate)"),
            value=df_brut.shape[1] == 1, key="sans_temps_checkbox",
        )

    tc1, tc2 = st.columns(2)
    with tc1:
        frequence_hz = st.number_input(
            tr("Fréquence d'échantillonnage (Hz), utilisée en l'absence de colonne de temps",
               "Sampling rate (Hz), used when there is no time column"),
            min_value=0.01, max_value=1000.0, value=1.0, step=0.1,
            help=tr("1 Hz = un point par seconde (cas des cycles WLTC et Artemis).",
                    "1 Hz = one point per second (as for the WLTC and Artemis cycles)."),
            key="freq_input",
        )
    with tc2:
        time_col_choice = st.selectbox(
            tr("Colonne de temps, utilisée si le fichier en contient une", "Time column, used if the file has one"),
            colonnes, index=colonnes.index(guess_time) if guess_time in colonnes else 0, key="time_col_select",
        )
    time_col = None if sans_colonne_temps else time_col_choice

    st.subheader(tr("3. Unité de la vitesse", "3. Speed unit"))
    if speed_col is not None:
        valeurs_vitesse = pd.to_numeric(df_brut[speed_col], errors="coerce").to_numpy(dtype=float)
        vitesse_valide = not pd.isna(valeurs_vitesse).all()
        if vitesse_valide:
            unite_detectee = core.detect_speed_unit(valeurs_vitesse)
            st.caption(tr(
                "Unité reconnue automatiquement : **{u}** (vitesse maximale ≈ {v} dans l'unité d'origine).",
                "Automatically recognised unit: **{u}** (maximum speed ≈ {v} in the original unit).",
                u=unite_detectee, v=nombre(pd.Series(valeurs_vitesse).abs().max(), 1),
            ))
        else:
            st.error(tr("La colonne de vitesse « {c} » ne contient pas de nombres.",
                        "The speed column “{c}” contains no numbers.", c=speed_col))
            unite_detectee = "km/h"
        unite_vitesse = st.radio(
            tr("Unité de la colonne de vitesse", "Unit of the speed column"), ["km/h", "m/s"],
            index=["km/h", "m/s"].index(unite_detectee), horizontal=True, key="unite_radio",
        )
    else:
        vitesse_valide = False
        unite_vitesse = "km/h"
        st.caption(tr("Aucune colonne de vitesse sélectionnée : cette section est ignorée.",
                      "No speed column selected: this section is skipped."))

    st.subheader(tr("4. Répétition du cycle", "4. Cycle repetition"))
    if vitesse_valide and speed_col is not None:
        repetition_auto = core.detect_repetition(core.convert_speed_to_ms(valeurs_vitesse, unite_vitesse))
        st.caption(tr(
            "Détection automatique, à titre indicatif : le motif semble répété {n} fois.",
            "Automatic detection, for information: the pattern seems to be repeated {n} times.",
            n=repetition_auto,
        ))
    else:
        repetition_auto = 1
    nombre_repetitions = st.number_input(
        tr("Nombre de répétitions du cycle à appliquer", "Number of cycle repetitions to apply"),
        min_value=1, max_value=50, value=int(repetition_auto), step=1,
        help=tr(
            "Indiquez la valeur exacte connue pour votre fichier plutôt que de vous fier "
            "uniquement à la détection automatique.",
            "Enter the exact value known for your file rather than relying only on the "
            "automatic detection.",
        ),
        key="repetitions_input",
    )

    st.subheader(tr(
        "5. Paramètres du véhicule (utilisés seulement sans colonne de puissance)",
        "5. Vehicle parameters (used only without a power column)",
    ))
    st.caption(tr(
        "La puissance est calculée à partir des forces aérodynamique, de roulement, de gravité et "
        "d'inertie. Les valeurs par défaut sont celles du véhicule de l'article de référence.",
        "Power is computed from the aerodynamic, rolling, gravity and inertia forces. The default "
        "values are those of the vehicle in the reference paper.",
    ))
    vc1, vc2, vc3 = st.columns(3)
    with vc1:
        masse = st.number_input(tr("Masse (kg)", "Mass (kg)"), min_value=1.0, value=core.VEHICLE_MASS_KG,
                                step=10.0, key="masse_input")
        cx = st.number_input(tr("Coefficient de traînée Cx", "Drag coefficient Cx"), min_value=0.0,
                             value=core.DRAG_COEFFICIENT_CX, step=0.01, key="cx_input")
    with vc2:
        surface_frontale = st.number_input(tr("Surface frontale S (m²)", "Frontal area S (m²)"), min_value=0.0,
                                           value=core.FRONTAL_AREA_M2, step=0.05, key="surface_input")
        c0 = st.number_input(tr("Coefficient de roulement C0", "Rolling coefficient C0"), min_value=0.0,
                             value=core.ROLLING_C0, step=0.001, format="%.4f", key="c0_input")
    with vc3:
        c1 = st.number_input(tr("Coefficient de roulement C1", "Rolling coefficient C1"), min_value=0.0,
                             value=core.ROLLING_C1, step=1e-7, format="%.2e", key="c1_input")
        pente_deg = st.number_input(tr("Pente de la route (°)", "Road slope (°)"), value=0.0, step=0.5,
                                    key="pente_input")
    rho = st.number_input(tr("Masse volumique de l'air (kg/m³)", "Air density (kg/m³)"), min_value=0.0,
                          value=core.AIR_DENSITY_KG_M3, step=0.005, format="%.3f", key="rho_input")

    valide = st.form_submit_button(tr("Préparer les données", "Prepare the data"), type="primary")

if not valide and "cycle_prepare" not in st.session_state:
    st.stop()

if valide:
    if speed_col is None and power_col is None:
        st.error(tr("Sélectionnez au moins une colonne de vitesse ou une colonne de puissance.",
                    "Select at least a speed column or a power column."))
        st.stop()
    if speed_col is not None and not vitesse_valide:
        st.error(tr("Corrigez la colonne de vitesse avant de préparer les données.",
                    "Correct the speed column before preparing the data."))
        st.stop()

    with st.spinner(tr("Préparation en cours…", "Preparing…")):
        # Temps et vitesse du cycle de base, avant répétition
        if time_col is not None:
            temps_brut = pd.to_numeric(df_brut[time_col], errors="coerce").to_numpy(dtype=float)
            pas_de_temps_s = float(np.median(np.diff(temps_brut))) if len(temps_brut) > 1 else 1.0
            if pas_de_temps_s <= 0 or np.isnan(pas_de_temps_s):
                pas_de_temps_s = 1.0
        else:
            pas_de_temps_s = 1.0 / frequence_hz

        n_base = len(df_brut)
        temps_base = np.arange(n_base) * pas_de_temps_s
        df_base = pd.DataFrame()

        if speed_col is not None:
            df_base["speed"] = core.convert_speed_to_ms(
                pd.to_numeric(df_brut[speed_col], errors="coerce").to_numpy(dtype=float), unite_vitesse,
            )

        if power_col is not None:
            df_base["hasPower"] = pd.to_numeric(df_brut[power_col], errors="coerce").to_numpy(dtype=float)
            if accel_col is not None:
                df_base["hasAcceleration"] = pd.to_numeric(df_brut[accel_col], errors="coerce").to_numpy(dtype=float)
            elif speed_col is not None:
                dv, dt_arr = np.diff(df_base["speed"].to_numpy()), np.diff(temps_base)
                acc = np.divide(dv, dt_arr, out=np.zeros_like(dv), where=dt_arr != 0)
                df_base["hasAcceleration"] = np.concatenate(([0.0], acc))
            else:
                df_base["hasAcceleration"] = 0.0
        else:
            forces = core.compute_forces_and_power(
                df_base["speed"].to_numpy(), temps_base, mass=masse, cx=cx, frontal_area=surface_frontale,
                c0=c0, c1=c1, slope_rad=math.radians(pente_deg), rho=rho, gravity=core.GRAVITY_MS2,
            )
            for col, valeurs in forces.items():
                df_base[col] = valeurs

        # Répétition du cycle déjà calculé, sans nouvelle dérivation
        n_rep = int(nombre_repetitions)
        df_cycle = pd.concat([df_base.copy() for _ in range(n_rep)], ignore_index=True) if n_rep > 1 else df_base.copy()
        df_cycle.insert(0, "time", np.arange(len(df_cycle), dtype=float) * pas_de_temps_s)

    # Le cycle préparé reste à part : les résultats affichés ailleurs gardent
    # leur propre cycle tant qu'une simulation n'a pas été lancée sur celui-ci.
    st.session_state["cycle_prepare"] = df_cycle
    st.session_state["prep_meta"] = {
        "n_points": len(df_cycle),
        "n_repetitions": n_rep,
        "puissance_source": "colonne" if power_col is not None else "dynamique",
    }
    st.success(tr("Données préparées : {n} points au total.", "Data prepared: {n} points in total.", n=len(df_cycle)))

if "cycle_prepare" not in st.session_state:
    st.stop()

df_cycle = st.session_state["cycle_prepare"]
meta = st.session_state.get("prep_meta", {})

st.divider()


# 6. Formules utilisées

st.subheader(tr("6. Formules utilisées", "6. Formulas used"))
with st.expander(tr("Dynamique du véhicule et packs de batteries", "Vehicle dynamics and battery packs")):
    st.markdown(tr(
        "**Dynamique longitudinale du véhicule** (si la puissance n'est pas fournie) :",
        "**Longitudinal vehicle dynamics** (if the power is not provided):",
    ))
    st.latex(
        r"a = \frac{dv}{dt} \qquad F_{aero} = \tfrac{1}{2}\,\rho\,S\,C_x\,v^2 \qquad "
        r"F_{roul} = m\,g\,(C_0 + C_1 v^2) \qquad F_{grav} = m\,g\,\sin\theta"
    )
    st.latex(r"F_{tot} = F_{aero} + F_{roul} + F_{grav} + m\,a \qquad P_{dem} = F_{tot}\,v")
    st.markdown(tr(
        "**Packs de batteries** (à partir des cellules, sections 7 et 8) :",
        "**Battery packs** (from the cells, sections 7 and 8):",
    ))
    st.latex(
        r"V = V_{cell}\,n_s \qquad M = m_{cell}\,n_s\,n_p \qquad P_{max} = I_{cell}\,V\,n_p "
        r"\qquad E = DE \times M \qquad C = C_{cell}\,n_p"
    )


# 7. Architecture des batteries

st.subheader(tr("7. Architecture des batteries", "7. Battery architecture"))
st.write(tr(
    "Caractéristiques d'une cellule et nombre de cellules en série et en parallèle, pour chacune "
    "des deux batteries. Les valeurs par défaut sont celles du projet ; elles sont modifiables.",
    "Characteristics of one cell and number of cells in series and in parallel, for each of the "
    "two batteries. The default values are those of the project; they can be changed.",
))

col_eb, col_pb = st.columns(2)


def _cellules(prefixe, defauts):
    """Champs de saisie d'une batterie ; retourne les valeurs saisies."""
    (n_s, n_p, m, v, i_d, i_r, de, cap, r) = defauts
    return (
        st.number_input(tr("Cellules en série", "Cells in series"), min_value=1, value=n_s, step=1, key=f"{prefixe}_n_serie"),
        st.number_input(tr("Cellules en parallèle", "Cells in parallel"), min_value=1, value=n_p, step=1, key=f"{prefixe}_n_parallele"),
        st.number_input(tr("Masse d'une cellule (kg)", "Mass of one cell (kg)"), min_value=0.0, value=m, step=0.001,
                        format="%.3f", key=f"{prefixe}_masse_cellule"),
        st.number_input(tr("Tension d'une cellule (V)", "Voltage of one cell (V)"), min_value=0.0, value=v, step=0.1,
                        key=f"{prefixe}_v_cellule"),
        st.number_input(tr("Courant de décharge d'une cellule (A)", "Discharge current of one cell (A)"), value=i_d,
                        step=0.1, key=f"{prefixe}_i_decharge"),
        st.number_input(tr("Courant de recharge d'une cellule (A)", "Charge current of one cell (A)"), value=i_r,
                        step=0.1, key=f"{prefixe}_i_recharge"),
        st.number_input(tr("Densité d'énergie DE (Wh/kg)", "Energy density DE (Wh/kg)"), min_value=0.0, value=de,
                        step=1.0, key=f"{prefixe}_de"),
        st.number_input(tr("Capacité d'une cellule (Ah)", "Capacity of one cell (Ah)"), min_value=0.0, value=cap,
                        step=0.1, key=f"{prefixe}_capacite"),
        st.number_input(tr("Résistance interne d'une cellule (Ω)", "Internal resistance of one cell (Ω)"), min_value=0.0,
                        value=r, step=0.001, format="%.4f", key=f"{prefixe}_rint"),
    )


with col_eb:
    st.markdown(tr("**Batterie Énergie (EB)**", "**Energy battery (EB)**"))
    (eb_n_serie, eb_n_parallele, eb_masse_cellule, eb_v_cellule, eb_i_decharge, eb_i_recharge,
     eb_de, eb_capacite, eb_rint) = _cellules("eb", (
        core.CELL_EB_N_SERIE, core.CELL_EB_N_PARALLELE, core.CELL_EB_MASSE_KG, core.CELL_EB_V_CELLULE,
        core.CELL_EB_I_DECHARGE_A, core.CELL_EB_I_RECHARGE_A, core.CELL_EB_DE_WH_KG, core.CELL_EB_CAPACITE_AH,
        core.CELL_EB_RINT_OHM,
    ))
with col_pb:
    st.markdown(tr("**Batterie Puissance (PB)**", "**Power battery (PB)**"))
    (pb_n_serie, pb_n_parallele, pb_masse_cellule, pb_v_cellule, pb_i_decharge, pb_i_recharge,
     pb_de, pb_capacite, pb_rint) = _cellules("pb", (
        core.CELL_PB_N_SERIE, core.CELL_PB_N_PARALLELE, core.CELL_PB_MASSE_KG, core.CELL_PB_V_CELLULE,
        core.CELL_PB_I_DECHARGE_A, core.CELL_PB_I_RECHARGE_A, core.CELL_PB_DE_WH_KG, core.CELL_PB_CAPACITE_AH,
        core.CELL_PB_RINT_OHM,
    ))

st.caption(tr(
    "Nombre total de cellules : EB = {neb} ({sEB} en série × {pEB} en parallèle), PB = {npb} ({sPB} en série × {pPB} en parallèle).",
    "Total number of cells: EB = {neb} ({sEB} in series × {pEB} in parallel), PB = {npb} ({sPB} in series × {pPB} in parallel).",
    neb=int(eb_n_serie * eb_n_parallele), sEB=eb_n_serie, pEB=eb_n_parallele,
    npb=int(pb_n_serie * pb_n_parallele), sPB=pb_n_serie, pPB=pb_n_parallele,
))


# 8. Caractéristiques calculées des packs

st.subheader(tr("8. Caractéristiques des packs", "8. Pack characteristics"))
resultats_eb = core.compute_pack_characteristics(
    eb_v_cellule, eb_i_decharge, eb_i_recharge, eb_masse_cellule, eb_de, eb_n_serie, eb_n_parallele,
    capacite_cellule_ah=eb_capacite,
)
resultats_pb = core.compute_pack_characteristics(
    pb_v_cellule, pb_i_decharge, pb_i_recharge, pb_masse_cellule, pb_de, pb_n_serie, pb_n_parallele,
    capacite_cellule_ah=pb_capacite,
)
# Ces caractéristiques sont réellement appliquées au moteur de simulation, pour
# toutes les simulations qui suivent, et pas seulement à cet affichage.
core.set_battery_pack_parameters("EB", resultats_eb)
core.set_battery_pack_parameters("PB", resultats_pb)

col_v, col_m, col_c = tr("Tension (V)", "Voltage (V)"), tr("Masse (kg)", "Mass (kg)"), tr("Capacité (Ah)", "Capacity (Ah)")
col_pd, col_pr = tr("P décharge max (W)", "Max discharge P (W)"), tr("P recharge max (W)", "Max charge P (W)")
col_e = tr("Énergie (Wh)", "Energy (Wh)")
st.dataframe(
    pd.DataFrame([
        {
            tr("Batterie", "Battery"): tr("Énergie (EB)", "Energy (EB)"),
            col_v: round(resultats_eb["tension_V"], 2), col_m: round(resultats_eb["masse_kg"], 2),
            col_c: round(resultats_eb.get("capacite_Ah", float("nan")), 3),
            col_pd: round(resultats_eb["puissance_decharge_W"], 2), col_pr: round(resultats_eb["puissance_recharge_W"], 2),
            col_e: round(resultats_eb["energie_Wh"], 2),
        },
        {
            tr("Batterie", "Battery"): tr("Puissance (PB)", "Power (PB)"),
            col_v: round(resultats_pb["tension_V"], 2), col_m: round(resultats_pb["masse_kg"], 2),
            col_c: round(resultats_pb.get("capacite_Ah", float("nan")), 3),
            col_pd: round(resultats_pb["puissance_decharge_W"], 2), col_pr: round(resultats_pb["puissance_recharge_W"], 2),
            col_e: round(resultats_pb["energie_Wh"], 2),
        },
        {
            tr("Batterie", "Battery"): tr("Les deux batteries", "Both batteries"),
            col_v: round(resultats_pb["tension_V"], 2), col_m: round(resultats_eb["masse_kg"] + resultats_pb["masse_kg"], 2),
            col_c: None,
            col_pd: round(resultats_eb["puissance_decharge_W"] + resultats_pb["puissance_decharge_W"], 2),
            col_pr: round(resultats_eb["puissance_recharge_W"] + resultats_pb["puissance_recharge_W"], 2),
            col_e: round(resultats_eb["energie_Wh"] + resultats_pb["energie_Wh"], 2),
        },
    ]),
    width="stretch", hide_index=True,
)

st.session_state["architecture_batteries"] = {
    "EB": {"n_serie": eb_n_serie, "n_parallele": eb_n_parallele, "masse_cellule": eb_masse_cellule,
           "v_cellule": eb_v_cellule, "i_decharge": eb_i_decharge, "i_recharge": eb_i_recharge,
           "de": eb_de, "capacite": eb_capacite, "rint": eb_rint, **resultats_eb},
    "PB": {"n_serie": pb_n_serie, "n_parallele": pb_n_parallele, "masse_cellule": pb_masse_cellule,
           "v_cellule": pb_v_cellule, "i_decharge": pb_i_decharge, "i_recharge": pb_i_recharge,
           "de": pb_de, "capacite": pb_capacite, "rint": pb_rint, **resultats_pb},
}


# 9. Convertisseur

st.divider()
st.subheader(tr("9. Convertisseur", "9. Converter"))
st.write(tr(
    "La puissance du convertisseur dépend du nombre de modules installés en parallèle. Les "
    "valeurs par défaut (un module) donnent 1 520 W en décharge et −760 W en recharge.",
    "The converter power depends on the number of modules installed in parallel. The default "
    "values (one module) give 1,520 W in discharge and −760 W in charge.",
))
col_conv1, col_conv2, col_conv3 = st.columns(3)
with col_conv1:
    conv_n_composants = st.number_input(tr("Nombre de modules", "Number of modules"), min_value=1,
                                        value=core.CONVERTER_N_COMPOSANTS, step=1, key="conv_n_composants")
with col_conv2:
    conv_p_decharge = st.number_input(tr("Puissance de décharge max par module (W)", "Max discharge power per module (W)"),
                                      min_value=0.0, value=core.CONVERTER_P_DECHARGE_PAR_COMPOSANT_W, step=10.0,
                                      key="conv_p_decharge")
with col_conv3:
    conv_p_recharge = st.number_input(tr("Puissance de recharge max par module (W)", "Max charge power per module (W)"),
                                      value=core.CONVERTER_P_RECHARGE_PAR_COMPOSANT_W, step=10.0, key="conv_p_recharge")

resultats_conv = core.compute_converter_characteristics(conv_n_composants, conv_p_decharge, conv_p_recharge)
# Ces limites sont réellement appliquées par le filtre de sécurité de toutes les simulations suivantes.
core.set_converter_power_limits(resultats_conv["p_decharge_W"], resultats_conv["p_recharge_W"])

col_res1, col_res2 = st.columns(2)
col_res1.metric(tr("Puissance totale de décharge", "Total discharge power"), f"{nombre(resultats_conv['p_decharge_W'], 0)} W")
col_res2.metric(tr("Puissance totale de recharge", "Total charge power"), f"{nombre(resultats_conv['p_recharge_W'], 0)} W")
st.caption(tr(
    "Le filtre de sécurité limite la puissance traitée par le convertisseur entre {pmin} W et "
    "{pmax} W dans toutes les simulations qui suivent.",
    "The safety filter keeps the power processed by the converter between {pmin} W and {pmax} W "
    "in all the following simulations.",
    pmin=nombre(resultats_conv["p_recharge_W"], 0), pmax=nombre(resultats_conv["p_decharge_W"], 0),
))
st.session_state["architecture_convertisseur"] = {
    "n_composants": conv_n_composants,
    "p_decharge_par_composant": conv_p_decharge,
    "p_recharge_par_composant": conv_p_recharge,
    **resultats_conv,
}


# Résumé, export et aperçu, une fois toute la configuration terminée (l'aperçu
# utilise ainsi les batteries et le convertisseur réglés ci-dessus).

st.divider()
m1, m2, m3 = st.columns(3)
m1.metric(tr("Nombre de points", "Number of points"), meta.get("n_points", len(df_cycle)))
m2.metric(tr("Répétitions", "Repetitions"), meta.get("n_repetitions", 1))
m3.metric(
    tr("Origine de la puissance", "Origin of the power"),
    tr("colonne du fichier", "file column") if meta.get("puissance_source") == "colonne"
    else tr("calculée (dynamique du véhicule)", "computed (vehicle dynamics)"),
)

p = df_cycle["hasPower"]
if abs(p.mean()) < 50 and p.max() < 200:
    st.warning(tr(
        "La puissance moyenne calculée est presque nulle (< 50 W). Vérifiez l'unité de vitesse, la "
        "colonne choisie et la fréquence d'échantillonnage ; sinon, le SOC variera très peu.",
        "The computed mean power is almost zero (< 50 W). Check the speed unit, the selected column "
        "and the sampling rate; otherwise the SOC will barely change.",
    ))

st.subheader(tr("Télécharger les données préparées", "Download the prepared data"))
st.write(tr(
    "Le fichier contient le temps, la vitesse, l'accélération, les forces et la puissance demandée.",
    "The file contains the time, speed, acceleration, forces and power demand.",
))
colonnes_export = [c for c in [
    "time", "speed", "hasAcceleration", "hasAeroForce", "hasRollingForce",
    "hasGravityForce", "hasAccelerationForce", "hasTotalForce", "hasPower",
] if c in df_cycle.columns]
colonnes_export += [c for c in df_cycle.columns if c not in colonnes_export]

format_export = st.radio(
    tr("Format du fichier", "File format"), ["xlsx", "csv"],
    format_func={"xlsx": "Excel (.xlsx)", "csv": "CSV (.csv)"}.get, horizontal=True, key="format_export_cycle",
    help=tr(
        "Le CSV utilise le point-virgule comme séparateur, compatible avec Excel en français.",
        "The CSV uses a semicolon as separator, compatible with French-language Excel.",
    ),
)
if format_export == "xlsx":
    try:
        buffer_excel = io.BytesIO()
        with pd.ExcelWriter(buffer_excel, engine="xlsxwriter") as writer:
            df_cycle[colonnes_export].to_excel(writer, index=False, sheet_name="Cycle")
        st.download_button(
            tr("Télécharger (Excel)", "Download (Excel)"), data=buffer_excel.getvalue(), file_name="cycle_prepare.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="download_cycle_prepare_xlsx",
        )
    except ImportError:
        st.error(tr("L'export Excel n'est pas disponible ici : choisissez le format CSV.",
                    "Excel export is not available here: choose the CSV format."))
else:
    st.download_button(
        tr("Télécharger (CSV)", "Download (CSV)"),
        data=df_cycle[colonnes_export].to_csv(index=False, sep=";").encode("utf-8-sig"),
        file_name="cycle_prepare.csv", mime="text/csv", key="download_cycle_prepare_csv",
    )

st.subheader(tr("Aperçu du cycle", "Cycle overview"))
temps_min_cycle = df_cycle["time"].to_numpy(dtype=float) / 60.0
fig_brut = make_subplots(
    rows=1, cols=2, horizontal_spacing=0.08,
    subplot_titles=(
        tr("Vitesse (km/h)", "Speed (km/h)") if "speed" in df_cycle.columns else tr("Accélération (m/s²)", "Acceleration (m/s²)"),
        tr("Puissance demandée (kW)", "Power demand (kW)"),
    ),
)
fig_brut.add_trace(
    go.Scattergl(
        x=temps_min_cycle, y=df_cycle["speed"] * 3.6 if "speed" in df_cycle.columns else df_cycle["hasAcceleration"],
        line=dict(color=COULEUR_DEMANDE, width=1.2), hovertemplate="%{y:.1f}<extra></extra>",
    ),
    row=1, col=1,
)
fig_brut.add_trace(
    go.Scattergl(x=temps_min_cycle, y=df_cycle["hasPower"] / 1000.0, line=dict(color=COULEUR_DEMANDE, width=1.2),
                 hovertemplate="%{y:.1f} kW<extra></extra>"),
    row=1, col=2,
)
fig_brut.update_xaxes(title_text=tr("Temps (min)", "Time (min)"))
fig_brut.update_layout(separators=separateurs_plotly(), height=320, showlegend=False,
                       margin=dict(t=40, b=40, l=10, r=10), hovermode="x unified")
st.plotly_chart(fig_brut, width="stretch")

st.subheader(tr("Premier aperçu avec le modèle physique", "First look with the physical model"))
st.write(tr(
    "Simulation rapide avec le modèle physique (la batterie Énergie d'abord, la batterie Puissance "
    "pour le surplus), qui ne demande aucun apprentissage, avec les batteries et le convertisseur "
    "réglés ci-dessus. La comparaison des sept stratégies se lance depuis « Lancer une simulation ».",
    "Quick simulation with the physical model (the Energy battery first, the Power battery for the "
    "surplus), which needs no training, with the batteries and converter set above. The comparison "
    "of the seven strategies is run from “Run a simulation”.",
))
col_soc0_1, col_soc0_2 = st.columns(2)
soc_eb0_apercu = col_soc0_1.slider(tr("SOC initial de l'EB", "Initial EB SOC"), 0.20, 1.0, 1.0, 0.01, key="apercu_soc_eb0")
soc_pb0_apercu = col_soc0_2.slider(tr("SOC initial de la PB", "Initial PB SOC"), 0.20, 1.0, 1.0, 0.01, key="apercu_soc_pb0")

traj_apercu = simuler_strategie_deterministe(
    df_cycle, soc_eb0_apercu, soc_pb0_apercu,
    proposer_alpha=lambda t, ligne, soc_eb, soc_pb, alpha_prev: eb_priority_alpha_single(ligne["hasPower"], soc_eb),
)

fig_apercu = make_subplots(
    rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05,
    subplot_titles=(
        tr("États de charge (%)", "States of charge (%)"),
        tr("Puissance des batteries (kW) : décharge au-dessus de zéro, recharge en dessous",
           "Battery power (kW): discharge above zero, charge below"),
        tr("Courants (A)", "Currents (A)"),
        tr("Puissance traitée par le convertisseur (kW)", "Power processed by the converter (kW)"),
    ),
)
eb_nom, pb_nom = tr("Batterie Énergie", "Energy battery"), tr("Batterie Puissance", "Power battery")
for cle_eb, cle_pb, echelle, ligne in (("SOC_EB", "SOC_PB", 100.0, 1), ("P_EB", "P_PB", 1 / 1000.0, 2), ("I_EB", "I_PB", 1.0, 3)):
    for cle, nom_batterie, coul in ((cle_eb, eb_nom, COULEUR_EB), (cle_pb, pb_nom, COULEUR_PB)):
        fig_apercu.add_trace(
            go.Scattergl(
                x=temps_min_cycle, y=np.asarray(traj_apercu[cle], dtype=float)[: len(df_cycle)] * echelle,
                name=nom_batterie, legendgroup=nom_batterie, showlegend=ligne == 1, line=dict(color=coul, width=1.2),
            ),
            row=ligne, col=1,
        )
fig_apercu.add_hline(y=core.SOC_EB_MIN * 100, line=dict(color=COULEUR_VIOLATION, dash="dot", width=1), row=1, col=1)

# Le convertisseur ne traite que la différence de tension entre les deux batteries.
p_conv_apercu = (core.V_EB_PACK_NOM - core.V_PB_PACK_NOM) * np.asarray(traj_apercu["I_EB"], dtype=float)[: len(df_cycle)]
fig_apercu.add_trace(
    go.Scattergl(x=temps_min_cycle, y=p_conv_apercu / 1000.0, name=tr("Convertisseur", "Converter"),
                 line=dict(color=COULEUR_CONVERTISSEUR, width=1.2)),
    row=4, col=1,
)
for limite, texte, position in (
    (core.P_CONV_MAX_W, tr("limite de décharge", "discharge limit"), "top left"),
    (core.P_CONV_MIN_W, tr("limite de recharge", "charge limit"), "bottom left"),
):
    fig_apercu.add_hline(
        y=limite / 1000.0, line=dict(color=COULEUR_SECONDAIRE, dash="dot", width=1), row=4, col=1,
        annotation_text=f"{texte} {nombre(limite / 1000.0, 2)} kW", annotation_position=position, annotation_font_size=10,
    )
fig_apercu.add_hline(y=core.P_EB_MAX_W / 1000.0, line=dict(color=COULEUR_SECONDAIRE, dash="dot", width=1), row=2, col=1,
                     annotation_text=tr("limite de la batterie Énergie", "Energy battery limit"),
                     annotation_position="top left", annotation_font_size=10)
fig_apercu.update_xaxes(title_text=tr("Temps (min)", "Time (min)"), row=4, col=1)
fig_apercu.update_layout(separators=separateurs_plotly(), height=820, margin=dict(t=40, b=40, l=10, r=10),
                         hovermode="x unified", legend=dict(orientation="h", y=1.05, x=0))
st.plotly_chart(fig_apercu, width="stretch")

st.session_state["soc_eb0_prepare"] = soc_eb0_apercu
st.session_state["soc_pb0_prepare"] = soc_pb0_apercu

pied_navigation("vues/2_Preparation_donnees.py")
