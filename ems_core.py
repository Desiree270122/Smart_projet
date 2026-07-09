

from pathlib import Path
import time

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn

except ImportError as _exc:
    raise ImportError(
        "PyTorch (torch) est requis par ems_core.py, mais il n'est pas installé "
        "ou n'a pas pu être importé. Installe-le avec `pip install torch`."
    ) from _exc




TORCH_GEOMETRIC_AVAILABLE = None  # None = non testé ; True/False = résultat du premier test


def _import_torch_geometric():
    """
    Tente d'importer torch_geometric une seule fois, uniquement lorsqu'EMS_GNN
    est réellement sollicité. Le résultat est mémorisé dans
    TORCH_GEOMETRIC_AVAILABLE.
    """
    global TORCH_GEOMETRIC_AVAILABLE, GCNConv, global_mean_pool

    if TORCH_GEOMETRIC_AVAILABLE is not None:
        return TORCH_GEOMETRIC_AVAILABLE

    try:
        from torch_geometric.nn import (
            GCNConv as _GCNConv,
            global_mean_pool as _global_mean_pool,
        )

        GCNConv = _GCNConv
        global_mean_pool = _global_mean_pool
        TORCH_GEOMETRIC_AVAILABLE = True

    except ImportError:
        TORCH_GEOMETRIC_AVAILABLE = False

    return TORCH_GEOMETRIC_AVAILABLE




def _resolve_root_dir():
    """Détermine le dossier racine du projet (celui qui contient models/,
    data/, results/, ...).

    On part du dossier où se trouve ems_core.py puis on remonte, et on
    retient le premier dossier qui contient réellement 'models' ou 'data'.
    Cette détection rend le chargement des modèles indépendant de la
    structure exacte : que ems_core.py soit à la racine du dépôt (cas du
    déploiement Streamlit Cloud, où le dossier code/ EST la racine) ou dans
    un sous-dossier code/ d'un projet plus large, les chemins restent
    corrects sans modification manuelle.
    """
    here = Path(__file__).resolve().parent
    for candidate in (here, here.parent, here.parent.parent):
        if (candidate / "models").is_dir() or (candidate / "data").is_dir():
            return candidate
    # Aucun dossier trouvé (première exécution sur une structure vide) :
    # on prend le dossier de ems_core.py, qui est la racine du dépôt déployé.
    return here


ROOT_DIR = _resolve_root_dir()

DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
RESULTS_DIR = ROOT_DIR / "results"
FIGURES_DIR = ROOT_DIR / "figures"
PREDICTIONS_DIR = ROOT_DIR / "predictions"
TABLES_DIR = ROOT_DIR / "tables"


for _directory in [
    DATA_DIR,
    PROCESSED_DIR,
    MODELS_DIR,
    CHECKPOINTS_DIR,
    RESULTS_DIR,
    FIGURES_DIR,
    PREDICTIONS_DIR,
    TABLES_DIR,
]:
    _directory.mkdir(
        parents=True,
        exist_ok=True,
    )



V_EB_PACK_NOM = 450.0
V_PB_PACK_NOM = 402.6  
V_HESS_BUS_NOM = 402.6  

CAPACITY_EB_AH = 30.4664
CAPACITY_PB_AH = 7.4196

SOC_EB_MIN, SOC_EB_MAX = 0.20, 1.0
SOC_PB_MIN, SOC_PB_MAX = 0.20, 1.0  
SOC_TOL = 5e-4
EPS_POWER_W = 100.0
DT_SECONDS = 1.0

P_EB_MIN_W, P_EB_MAX_W = -6300.0, 12600.0
P_PB_MIN_W, P_PB_MAX_W = -52338.0, 161040.0
P_CONV_MIN_W, P_CONV_MAX_W = -760.0, 1520.0

CONVERTER_RATIO_DISCHARGE = 9.493670886075948
CONVERTER_RATIO_CHARGE = 9.493670886075948

ENERGY_EB_WH = 13709.89
ENERGY_PB_WH = 2987.12

ALPHA_GRID_STEP = 0.001

ALPHA_GRID = np.arange(
    0.0,
    1.0 + ALPHA_GRID_STEP / 2.0,
    ALPHA_GRID_STEP,
    dtype=np.float64,
)


def set_alpha_grid_step(step):
    """Redéfinit la résolution de la grille alpha utilisée par le filtre
    physique (candidate_metrics, resoudre_decision_physique,
    optimiser_alpha_star_sequence).

    Cette grille est balayée à CHAQUE pas de temps et pour CHAQUE stratégie :
    c'est le coût partagé dominant de la simulation. Une grille plus grossière
    accélère fortement la simulation.

    - step = 0.001 (1001 points) : résolution de référence, la plus lente.
    - step = 0.005 (201 points)  : ~5x plus rapide sur le filtre.

    L'impact sur la comparaison des stratégies est négligeable : les écarts de
    coût entre stratégies (~0.02 à 0.05) sont bien plus grands qu'une
    quantification d'alpha à 0.005. Pour un résultat « qualité publication »,
    repasser à 0.001.
    """
    global ALPHA_GRID_STEP, ALPHA_GRID
    ALPHA_GRID_STEP = float(step)
    ALPHA_GRID = np.arange(
        0.0,
        1.0 + ALPHA_GRID_STEP / 2.0,
        ALPHA_GRID_STEP,
        dtype=np.float64,
    )


ALPHA_OPT_WEIGHTS = {
    "power_stress": 0.30,
    "energy_throughput": 0.20,
    "soc_risk": 0.25,
    "converter_stress": 0.10,
    "continuity": 0.15,
}

EXTENDED_COST_WEIGHTS = {
    "unserved": 1.0,
    "regen_curtailed": 1.0,
}

EXTENDED_COST_P_NORM = P_PB_MAX_W

ENERGY_TOTAL_WH = ENERGY_EB_WH + ENERGY_PB_WH
ENERGY_SHARE_EB = ENERGY_EB_WH / ENERGY_TOTAL_WH
ENERGY_SHARE_PB = ENERGY_PB_WH / ENERGY_TOTAL_WH

ENERGY_COST_NORMALIZER = max(
    1.0 / ENERGY_SHARE_EB**2,
    1.0 / ENERGY_SHARE_PB**2,
)

_P_EB_CONV_MAX = (
    P_CONV_MAX_W
    * CONVERTER_RATIO_DISCHARGE
)

_P_EB_CONV_MIN = (
    P_CONV_MIN_W
    * CONVERTER_RATIO_CHARGE
)




CONVERTER_N_COMPOSANTS = 1
CONVERTER_P_DECHARGE_PAR_COMPOSANT_W = 1520.0
CONVERTER_P_RECHARGE_PAR_COMPOSANT_W = -760.0


def compute_converter_characteristics(n_composants, p_decharge_par_composant, p_recharge_par_composant):
    """Calcule la puissance totale du convertisseur a partir du nombre de
    composants (modules en parallele) et de la puissance par composant."""
    return {
        "p_decharge_W": n_composants * p_decharge_par_composant,
        "p_recharge_W": n_composants * p_recharge_par_composant,
    }


def set_converter_power_limits(p_decharge_w, p_recharge_w):
    """Met a jour les limites de puissance du convertisseur reellement
    utilisees par le filtre de securite (candidate_metrics,
    resoudre_decision_physique, etc.), a partir des caracteristiques
    calculees au bloc 9 de la page Preparation des donnees. A appeler
    apres compute_converter_characteristics -- sans cet appel, le moteur
    de simulation continue d'utiliser les valeurs par defaut (1 composant)."""
    global P_CONV_MIN_W, P_CONV_MAX_W, _P_EB_CONV_MAX, _P_EB_CONV_MIN
    P_CONV_MIN_W = p_recharge_w
    P_CONV_MAX_W = p_decharge_w
    _P_EB_CONV_MAX = P_CONV_MAX_W * CONVERTER_RATIO_DISCHARGE
    _P_EB_CONV_MIN = P_CONV_MIN_W * CONVERTER_RATIO_CHARGE

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)




VEHICLE_MASS_KG = 1400.0
GRAVITY_MS2 = 9.81
FRONTAL_AREA_M2 = 2.75
DRAG_COEFFICIENT_CX = 0.30
ROLLING_C0 = 0.008
ROLLING_C1 = 1.6e-6
AIR_DENSITY_KG_M3 = 1.225
ROAD_SLOPE_RAD = 0.0
DEFAULT_SAMPLING_HZ = 1.0



CELL_EB_I_RECHARGE_A = -2.0
CELL_EB_I_DECHARGE_A = 4.0
CELL_EB_CAPACITE_AH = 4.0
CELL_EB_V_CELLULE = 3.6
CELL_EB_MASSE_KG = 0.063
CELL_EB_DE_WH_KG = 228.6
CELL_EB_DP_DECHARGE_W_KG = 228.6
CELL_EB_DP_RECHARGE_W_KG = -114.3
CELL_EB_RINT_OHM = 0.03
CELL_EB_N_SERIE = 125
CELL_EB_N_PARALLELE = 7

CELL_PB_I_RECHARGE_A = -65.0
CELL_PB_I_DECHARGE_A = 200.0
CELL_PB_CAPACITE_AH = 4.5
CELL_PB_V_CELLULE = 3.3
CELL_PB_MASSE_KG = 0.205
CELL_PB_DE_WH_KG = 72.44
CELL_PB_DP_DECHARGE_W_KG = 3219.5
CELL_PB_DP_RECHARGE_W_KG = -1046.3
CELL_PB_RINT_OHM = 0.0035
CELL_PB_N_SERIE = 122
CELL_PB_N_PARALLELE = 2


def compute_pack_characteristics(v_cellule, i_decharge_cellule, i_recharge_cellule,
                                  masse_cellule, de_wh_kg, n_serie, n_parallele,
                                  capacite_cellule_ah=None):
    """Calcule les caracteristiques d'un pack batterie a partir des parametres
    cellule et de l'architecture (nombre de cellules en serie/parallele).

    Formule Energie = DE * Masse CONFIRMEE correcte par l'encadrant (l'ecart
    precedemment observe avec les valeurs de reference venait d'une erreur de
    calcul de son cote, pas de la formule elle-meme).

    La capacite du pack (Ah) suit la regle standard d'association de cellules :
    elle est egale a la capacite d'une cellule multipliee par le nombre de
    branches en parallele (le nombre de cellules en serie ne change pas la
    capacite, seulement la tension)."""
    tension_v = v_cellule * n_serie
    masse_kg = masse_cellule * n_serie * n_parallele
    puissance_decharge_w = i_decharge_cellule * tension_v * n_parallele
    puissance_recharge_w = i_recharge_cellule * tension_v * n_parallele
    energie_wh = de_wh_kg * masse_kg
    resultat = {
        "tension_V": tension_v,
        "masse_kg": masse_kg,
        "puissance_decharge_W": puissance_decharge_w,
        "puissance_recharge_W": puissance_recharge_w,
        "energie_Wh": energie_wh,
    }
    if capacite_cellule_ah is not None:
        resultat["capacite_Ah"] = capacite_cellule_ah * n_parallele
    return resultat


def set_battery_pack_parameters(pack, caracteristiques):
    """Met a jour les constantes de la batterie 'EB' ou 'PB' reellement
    utilisees par le moteur de simulation (update_soc, candidate_metrics,
    resoudre_decision_physique, etc.), a partir des caracteristiques
    calculees au bloc 8 de la page Preparation des donnees. A appeler apres
    compute_pack_characteristics -- sans cet appel, le moteur continue
    d'utiliser les valeurs par defaut confirmees dans le cadrage du projet.

    pack doit valoir 'EB' ou 'PB'."""
    global V_EB_PACK_NOM, P_EB_MIN_W, P_EB_MAX_W, CAPACITY_EB_AH, ENERGY_EB_WH
    global V_PB_PACK_NOM, P_PB_MIN_W, P_PB_MAX_W, CAPACITY_PB_AH, ENERGY_PB_WH
    global ENERGY_TOTAL_WH, ENERGY_SHARE_EB, ENERGY_SHARE_PB, ENERGY_COST_NORMALIZER

    if pack == "EB":
        V_EB_PACK_NOM = caracteristiques["tension_V"]
        P_EB_MIN_W = caracteristiques["puissance_recharge_W"]
        P_EB_MAX_W = caracteristiques["puissance_decharge_W"]
        ENERGY_EB_WH = caracteristiques["energie_Wh"]
        if "capacite_Ah" in caracteristiques:
            CAPACITY_EB_AH = caracteristiques["capacite_Ah"]
    elif pack == "PB":
        V_PB_PACK_NOM = caracteristiques["tension_V"]
        P_PB_MIN_W = caracteristiques["puissance_recharge_W"]
        P_PB_MAX_W = caracteristiques["puissance_decharge_W"]
        ENERGY_PB_WH = caracteristiques["energie_Wh"]
        if "capacite_Ah" in caracteristiques:
            CAPACITY_PB_AH = caracteristiques["capacite_Ah"]
    else:
        raise ValueError("pack doit valoir 'EB' ou 'PB'")

    ENERGY_TOTAL_WH = ENERGY_EB_WH + ENERGY_PB_WH
    ENERGY_SHARE_EB = ENERGY_EB_WH / ENERGY_TOTAL_WH
    ENERGY_SHARE_PB = ENERGY_PB_WH / ENERGY_TOTAL_WH
    ENERGY_COST_NORMALIZER = max(1.0 / ENERGY_SHARE_EB**2, 1.0 / ENERGY_SHARE_PB**2)



_COLUMN_KEYWORDS = {
    "time": [
        "time",
        "temps",
        "t_",
        "instant",
        "horodatage",
        "date",
        "sec",
        "secondes",
        "duree",
    ],
    "speed": [
        "speed",
        "vitesse",
        "vit",
        "v_",
        "vel",
        "velocity",
        "km_h",
        "kmh",
        "m_s",
        "allure",
    ],
    "power": [
        "power",
        "puissance",
        "haspower",
        "has_power",
        "p_dem",
        "pdem",
        "p_load",
        "pload",
        "demande",
        "charge",
        "conso",
        "p_traction",
    ],
    "acceleration": [
        "acceleration",
        "accel",
        "acc",
        "hasacceleration",
    ],
}


def _normalize_colname(name: str) -> str:
    n = (
        str(name)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )

    for a, b in {
        "é": "e",
        "è": "e",
        "ê": "e",
        "à": "a",
        "ù": "u",
        "ô": "o",
        "î": "i",
        "ç": "c",
    }.items():
        n = n.replace(a, b)

    return n


def guess_column(
    colnames,
    category: str,
):
    """
    Détermine la colonne la plus probable pour 'time', 'speed', 'power'
    ou 'acceleration' à partir de mots-clés courants en français et en anglais.

    Retourne None lorsqu'aucune correspondance n'est trouvée, afin d'éviter
    d'imposer un choix arbitraire par défaut.
    """
    keywords = [
        _normalize_colname(k)
        for k in _COLUMN_KEYWORDS.get(category, [])
    ]

    for c in colnames:
        cl = _normalize_colname(c)

        if any(k in cl for k in keywords):
            return c

    return None


def detect_speed_unit(speed) -> str:
    """
    Estime l'unité de vitesse à partir du 95e percentile du signal :

    - 'km/h' lorsque les valeurs sont élevées ;
    - 'm/s' dans le cas contraire.

    Cette estimation doit être confirmée manuellement dans l'interface.
    """
    speed = np.asarray(
        speed,
        dtype=float,
    )

    p95 = np.nanpercentile(
        np.abs(speed),
        95,
    )

    return (
        "km/h"
        if p95 > 45
        else "m/s"
    )


def convert_speed_to_ms(
    speed,
    unit: str,
):
    speed = np.asarray(
        speed,
        dtype=float,
    )

    return (
        speed / 3.6
        if unit == "km/h"
        else speed
    )


def detect_repetition(
    speed_ms,
    max_candidates: int = 10,
) -> int:
    """
    Détecte si le signal contient un motif de base répété plusieurs fois
    à l'identique.

    Le résultat reste indicatif : une valeur confirmée manuellement
    doit être privilégiée.
    """
    speed_ms = np.asarray(
        speed_ms,
        dtype=float,
    )

    n_total = len(speed_ms)

    for n in range(
        max_candidates,
        1,
        -1,
    ):
        if n_total % n != 0:
            continue

        chunk_len = n_total // n
        base = speed_ms[:chunk_len]

        if all(
            np.allclose(
                base,
                speed_ms[
                    k * chunk_len:
                    (k + 1) * chunk_len
                ],
                atol=1e-2,
            )
            for k in range(1, n)
        ):
            return n

    return 1


def compute_forces_and_power(
    speed_ms,
    time_s,
    mass=VEHICLE_MASS_KG,
    cx=DRAG_COEFFICIENT_CX,
    frontal_area=FRONTAL_AREA_M2,
    c0=ROLLING_C0,
    c1=ROLLING_C1,
    slope_rad=ROAD_SLOPE_RAD,
    rho=AIR_DENSITY_KG_M3,
    gravity=GRAVITY_MS2,
):
    """
    Calcule l'accélération, les forces longitudinales — aérodynamique,
    roulement, gravité et accélération — ainsi que la puissance demandée
    à partir des signaux de vitesse (m/s) et de temps (s).

    L'équation standard de dynamique longitudinale utilise les constantes
    validées du véhicule : masse, Cx, S, C0 et C1.

    Cette fonction doit être appelée sur un seul cycle de base, avant
    répétition, afin d'éviter une discontinuité artificielle de
    l'accélération à la jonction entre deux répétitions.
    """
    v = np.asarray(
        speed_ms,
        dtype=float,
    )

    t = np.asarray(
        time_s,
        dtype=float,
    )

    dv = np.diff(v)
    dt_arr = np.diff(t)

    acc = np.divide(
        dv,
        dt_arr,
        out=np.zeros_like(dv),
        where=dt_arr != 0,
    )

    acc = np.concatenate(
        ([0.0], acc)
    )

    aero = (
        0.5
        * rho
        * frontal_area
        * cx
        * v**2
    )

    rolling = (
        mass
        * gravity
        * (
            c0
            + c1 * v**2
        )
    )

    gravity_force = np.full_like(
        v,
        mass
        * gravity
        * np.sin(slope_rad),
    )

    accel_force = (
        mass
        * acc
    )

    total_force = (
        aero
        + rolling
        + gravity_force
        + accel_force
    )

    power = (
        total_force
        * v
    )

    return {
        "hasAcceleration": acc,
        "hasAeroForce": aero,
        "hasRollingForce": rolling,
        "hasGravityForce": gravity_force,
        "hasAccelerationForce": accel_force,
        "hasTotalForce": total_force,
        "hasPower": power,
    }


# ============================================================
# Fonctions physiques de base
# ============================================================

def estimate_p_conv(p_eb):
    p_eb = np.asarray(
        p_eb,
        dtype=float,
    )

    return np.where(
        p_eb >= 0,
        p_eb / CONVERTER_RATIO_DISCHARGE,
        p_eb / CONVERTER_RATIO_CHARGE,
    )


def candidate_metrics(
    alpha,
    p_dem,
    soc_eb,
    soc_pb,
    alpha_prev,
):
    alpha = np.asarray(
        alpha,
        dtype=float,
    )

    p_pb = (
        alpha
        * p_dem
    )

    p_eb = (
        p_dem
        - p_pb
    )

    p_conv = estimate_p_conv(
        p_eb
    )

    i_eb = (
        p_eb
        / V_EB_PACK_NOM
    )

    i_pb = (
        p_pb
        / V_PB_PACK_NOM
    )

    soc_eb_next = (
        soc_eb
        - i_eb
        * DT_SECONDS
        / (
            3600.0
            * CAPACITY_EB_AH
        )
    )

    soc_pb_next = (
        soc_pb
        - i_pb
        * DT_SECONDS
        / (
            3600.0
            * CAPACITY_PB_AH
        )
    )

    feasible = (
        (alpha >= 0.0)
        & (alpha <= 1.0)
        & (p_eb >= P_EB_MIN_W)
        & (p_eb <= P_EB_MAX_W)
        & (p_pb >= P_PB_MIN_W)
        & (p_pb <= P_PB_MAX_W)
        & (p_conv >= P_CONV_MIN_W)
        & (p_conv <= P_CONV_MAX_W)
    )

    if soc_eb <= SOC_EB_MIN + SOC_TOL:
        feasible &= (
            p_eb
            <= EPS_POWER_W
        )

    if soc_pb <= SOC_PB_MIN + SOC_TOL:
        feasible &= (
            p_pb
            <= EPS_POWER_W
        )

    if soc_eb >= SOC_EB_MAX - SOC_TOL:
        feasible &= (
            p_eb
            >= -EPS_POWER_W
        )

    if soc_pb >= SOC_PB_MAX - SOC_TOL:
        feasible &= (
            p_pb
            >= -EPS_POWER_W
        )

    feasible &= (
        soc_eb_next
        >= SOC_EB_MIN - SOC_TOL
    )

    feasible &= (
        soc_pb_next
        >= SOC_PB_MIN - SOC_TOL
    )

    feasible &= (
        soc_eb_next
        <= SOC_EB_MAX + SOC_TOL
    )

    feasible &= (
        soc_pb_next
        <= SOC_PB_MAX + SOC_TOL
    )

    eb_dir = np.where(
        p_eb >= 0,
        P_EB_MAX_W,
        abs(P_EB_MIN_W),
    )

    pb_dir = np.where(
        p_pb >= 0,
        P_PB_MAX_W,
        abs(P_PB_MIN_W),
    )

    conv_dir = np.where(
        p_conv >= 0,
        P_CONV_MAX_W,
        abs(P_CONV_MIN_W),
    )

    eb_u = (
        np.abs(p_eb)
        / eb_dir
    )

    pb_u = (
        np.abs(p_pb)
        / pb_dir
    )

    conv_u = (
        np.abs(p_conv)
        / conv_dir
    )

    power_stress = (
        0.5
        * (
            eb_u**2
            + pb_u**2
        )
    )

    energy_throughput = (
        (
            (
                (1.0 - alpha)
                / ENERGY_SHARE_EB
            ) ** 2
            + (
                alpha
                / ENERGY_SHARE_PB
            ) ** 2
        )
        / ENERGY_COST_NORMALIZER
    )

    eb_soc_level = np.clip(
        (
            soc_eb
            - SOC_EB_MIN
        )
        / (
            SOC_EB_MAX
            - SOC_EB_MIN
        ),
        0.0,
        1.0,
    )

    pb_soc_level = np.clip(
        (
            soc_pb
            - SOC_PB_MIN
        )
        / (
            SOC_PB_MAX
            - SOC_PB_MIN
        ),
        0.0,
        1.0,
    )

    if p_dem > EPS_POWER_W:
        eb_risk = (
            1.0
            - eb_soc_level
        )

        pb_risk = (
            1.0
            - pb_soc_level
        )

    elif p_dem < -EPS_POWER_W:
        eb_risk = eb_soc_level
        pb_risk = pb_soc_level

    else:
        eb_risk = 0.0
        pb_risk = 0.0

    soc_risk = (
        (1.0 - alpha)
        * eb_risk
        + alpha
        * pb_risk
    )

    converter_stress = (
        conv_u**2
    )

    continuity_cost = (
        (alpha - alpha_prev) ** 2
        if (
            alpha_prev is not None
            and np.isfinite(alpha_prev)
        )
        else np.zeros_like(alpha)
    )

    total_cost = (
        ALPHA_OPT_WEIGHTS["power_stress"]
        * power_stress
        + ALPHA_OPT_WEIGHTS["energy_throughput"]
        * energy_throughput
        + ALPHA_OPT_WEIGHTS["soc_risk"]
        * soc_risk
        + ALPHA_OPT_WEIGHTS["converter_stress"]
        * converter_stress
        + ALPHA_OPT_WEIGHTS["continuity"]
        * continuity_cost
    )

    return {
        "feasible": feasible,
        "total_cost": total_cost,
        "P_EB": p_eb,
        "P_PB": p_pb,
        "P_conv": p_conv,
        "SOC_EB_next": soc_eb_next,
        "SOC_PB_next": soc_pb_next,
    }


def eb_priority_alpha_single(
    p_dem,
    soc_eb,
):
    if abs(p_dem) <= EPS_POWER_W:
        return 0.5

    if p_dem < P_EB_MIN_W:
        p_eb = P_EB_MIN_W

    elif p_dem < 0:
        p_eb = p_dem

    elif soc_eb <= SOC_EB_MIN:
        p_eb = 0.0

    elif p_dem <= P_EB_MAX_W:
        p_eb = p_dem

    else:
        p_eb = P_EB_MAX_W

    p_pb = (
        p_dem
        - p_eb
    )

    return float(
        np.clip(
            p_pb / p_dem,
            0.0,
            1.0,
        )
    )


def diagnose_violation(
    alpha_req,
    p_dem,
    soc_eb,
    soc_pb,
):
    reasons = []

    if alpha_req < 0 or alpha_req > 1:
        reasons.append(
            "alpha_hors_0_1"
        )

    p_pb = (
        alpha_req
        * p_dem
    )

    p_eb = (
        p_dem
        - p_pb
    )

    p_conv = estimate_p_conv(
        np.array([p_eb])
    )[0]

    if (
        p_eb < P_EB_MIN_W
        or p_eb > P_EB_MAX_W
    ):
        reasons.append(
            "P_EB_hors_limites"
        )

    if (
        p_pb < P_PB_MIN_W
        or p_pb > P_PB_MAX_W
    ):
        reasons.append(
            "P_PB_hors_limites"
        )

    if (
        p_conv < P_CONV_MIN_W
        or p_conv > P_CONV_MAX_W
    ):
        reasons.append(
            "P_conv_hors_limites"
        )

    if (
        soc_eb <= SOC_EB_MIN + SOC_TOL
        and p_eb > EPS_POWER_W
    ):
        reasons.append(
            "EB_protection_SOC_bas"
        )

    if (
        soc_pb <= SOC_PB_MIN + SOC_TOL
        and p_pb > EPS_POWER_W
    ):
        reasons.append(
            "PB_protection_SOC_bas"
        )

    i_eb = (
        p_eb
        / V_EB_PACK_NOM
    )

    i_pb = (
        p_pb
        / V_PB_PACK_NOM
    )

    soc_eb_next = (
        soc_eb
        - i_eb
        * DT_SECONDS
        / (
            3600.0
            * CAPACITY_EB_AH
        )
    )

    soc_pb_next = (
        soc_pb
        - i_pb
        * DT_SECONDS
        / (
            3600.0
            * CAPACITY_PB_AH
        )
    )

    if (
        soc_eb_next
        < SOC_EB_MIN - SOC_TOL
        or soc_eb_next
        > SOC_EB_MAX + SOC_TOL
    ):
        reasons.append(
            "SOC_EB_projete_hors_bornes"
        )

    if (
        soc_pb_next
        < SOC_PB_MIN - SOC_TOL
        or soc_pb_next
        > SOC_PB_MAX + SOC_TOL
    ):
        reasons.append(
            "SOC_PB_projete_hors_bornes"
        )

    return reasons


def resoudre_decision_physique(
    alpha_requested,
    p_dem,
    soc_eb,
    soc_pb,
    alpha_prev=None,
):
    """
    Fonction physique centrale et source unique de alpha_final, P_EB_final,
    P_PB_final, P_unserved et P_regen_curtailed.

    Toutes ces valeurs sont calculées de manière cohérente au cours
    d'un même passage.
    """

    if abs(p_dem) <= EPS_POWER_W:
        alpha_final = (
            alpha_prev
            if alpha_prev is not None
            else 0.5
        )

        return {
            "alpha_requested": alpha_requested,
            "alpha_final": alpha_final,
            "P_EB_final": 0.0,
            "P_PB_final": 0.0,
            "P_unserved": 0.0,
            "P_regen_curtailed": 0.0,
            "feasible": True,
            "correction_applied": False,
            "violated_constraint": "aucune",
            "explanation": (
                "Demande nulle — alpha conservé uniquement "
                "pour assurer la continuité."
            ),
        }

    metrics = candidate_metrics(
        ALPHA_GRID,
        p_dem,
        soc_eb,
        soc_pb,
        alpha_prev,
    )

    feasible_idx = np.flatnonzero(
        metrics["feasible"]
    )

    if len(feasible_idx) > 0:
        feasible_alphas = ALPHA_GRID[
            feasible_idx
        ]

        nearest = feasible_idx[
            np.argmin(
                np.abs(
                    feasible_alphas
                    - alpha_requested
                )
            )
        ]

        alpha_final = float(
            ALPHA_GRID[nearest]
        )

        p_eb_final = float(
            metrics["P_EB"][nearest]
        )

        p_pb_final = float(
            metrics["P_PB"][nearest]
        )

        corrected = not np.isclose(
            alpha_final,
            alpha_requested,
            atol=ALPHA_GRID_STEP * 1.1,
        )

        reasons = (
            diagnose_violation(
                alpha_requested,
                p_dem,
                soc_eb,
                soc_pb,
            )
            if corrected
            else []
        )

        return {
            "alpha_requested": alpha_requested,
            "alpha_final": alpha_final,
            "P_EB_final": p_eb_final,
            "P_PB_final": p_pb_final,
            "P_unserved": 0.0,
            "P_regen_curtailed": 0.0,
            "feasible": True,
            "correction_applied": corrected,
            "violated_constraint": (
                ",".join(reasons)
                if reasons
                else "aucune"
            ),
            "explanation": (
                "aucune correction nécessaire"
                if not corrected
                else (
                    "Projection vers la décision réalisable "
                    f"la plus proche (causes : {', '.join(reasons)})"
                )
            ),
        }

    eb_charge_limit_soc = (
        max(
            P_EB_MIN_W,
            -(
                (
                    SOC_EB_MAX
                    - soc_eb
                )
                * 3600.0
                * CAPACITY_EB_AH
                * V_EB_PACK_NOM
                / DT_SECONDS
            ),
        )
        if soc_eb < SOC_EB_MAX
        else 0.0
    )

    pb_charge_limit = (
        max(
            P_PB_MIN_W,
            -(
                (
                    SOC_PB_MAX
                    - soc_pb
                )
                * 3600.0
                * CAPACITY_PB_AH
                * V_PB_PACK_NOM
                / DT_SECONDS
            ),
        )
        if soc_pb < SOC_PB_MAX
        else 0.0
    )

    eb_charge_limit = max(
        eb_charge_limit_soc,
        _P_EB_CONV_MIN,
    )

    eb_discharge_limit = (
        0.0
        if soc_eb <= SOC_EB_MIN + SOC_TOL
        else min(
            P_EB_MAX_W,
            _P_EB_CONV_MAX,
        )
    )

    pb_discharge_limit = (
        0.0
        if soc_pb <= SOC_PB_MIN + SOC_TOL
        else P_PB_MAX_W
    )

    if p_dem > 0:
        p_eb_final = float(
            np.clip(
                p_dem,
                0.0,
                eb_discharge_limit,
            )
        )

        p_pb_final = float(
            np.clip(
                p_dem - p_eb_final,
                0.0,
                pb_discharge_limit,
            )
        )

    else:
        p_eb_final = float(
            np.clip(
                p_dem,
                eb_charge_limit,
                0.0,
            )
        )

        p_pb_final = float(
            np.clip(
                p_dem - p_eb_final,
                pb_charge_limit,
                0.0,
            )
        )

    residual = (
        p_dem
        - (
            p_eb_final
            + p_pb_final
        )
    )

    p_unserved = max(
        0.0,
        residual,
    )

    p_regen_curtailed = max(
        0.0,
        -residual,
    )

    alpha_final = (
        float(
            np.clip(
                p_pb_final / p_dem,
                0.0,
                1.0,
            )
        )
        if abs(p_dem) > EPS_POWER_W
        else (
            alpha_prev
            if alpha_prev is not None
            else 0.5
        )
    )

    return {
        "alpha_requested": alpha_requested,
        "alpha_final": alpha_final,
        "P_EB_final": p_eb_final,
        "P_PB_final": p_pb_final,
        "P_unserved": p_unserved,
        "P_regen_curtailed": p_regen_curtailed,
        "feasible": False,
        "correction_applied": True,
        "violated_constraint": "aucune_solution_faisable",
        "explanation": (
            "Aucune répartition électrique entièrement réalisable — "
            f"{'puissance non servie' if p_unserved > 0 else 'freinage mécanique requis'} "
            f"de {max(p_unserved, p_regen_curtailed):.0f} W."
        ),
    }


def analyser_capacites_hess(p_dem, soc_eb, soc_pb):
    """Analyse des capacités instantanées du HESS avant toute décision EMS.

    Détermine, pour la demande p_dem (W) et l'état SOC courant, la puissance de
    TRACTION (décharge) maximale que chaque batterie peut fournir — bornée par
    le SOC (une batterie au SOC minimal ne peut plus se décharger) et, pour l'EB,
    par le convertisseur à puissance partielle. La faisabilité et la répartition
    physique réellement réalisable proviennent du solveur `resoudre_decision_physique`
    (source unique), afin d'éviter toute divergence avec le moteur de simulation.

    Ce module agit comme un filtre physique : l'EMS ne choisit sa répartition que
    parmi ce que les batteries et le convertisseur peuvent réellement fournir.
    """
    eb_dispo_decharge = (
        0.0
        if soc_eb <= SOC_EB_MIN + SOC_TOL
        else float(min(P_EB_MAX_W, _P_EB_CONV_MAX))
    )
    pb_dispo_decharge = (
        0.0
        if soc_pb <= SOC_PB_MIN + SOC_TOL
        else float(P_PB_MAX_W)
    )
    hess_dispo_decharge = eb_dispo_decharge + pb_dispo_decharge

    decision = resoudre_decision_physique(0.5, p_dem, soc_eb, soc_pb)

    return {
        "p_dem_W": float(p_dem),
        "soc_eb": float(soc_eb),
        "soc_pb": float(soc_pb),
        "eb_dispo_max_W": eb_dispo_decharge,
        "pb_dispo_max_W": pb_dispo_decharge,
        "hess_dispo_max_W": hess_dispo_decharge,
        "marge_eb_W": eb_dispo_decharge - max(0.0, decision["P_EB_final"]),
        "marge_pb_W": pb_dispo_decharge - max(0.0, decision["P_PB_final"]),
        "faisable": bool(decision["feasible"]),
        "P_EB_reparti_W": float(decision["P_EB_final"]),
        "P_PB_reparti_W": float(decision["P_PB_final"]),
        "P_non_servie_W": float(decision["P_unserved"]),
        "P_regen_rejetee_W": float(decision["P_regen_curtailed"]),
        "alpha": float(decision["alpha_final"]),
        "explication": decision["explanation"],
    }


def optimiser_alpha_star_sequence(
    df,
    soc_eb0,
    soc_pb0,
):
    """
    Calcule la séquence alpha_star, qui représente l'optimum physique
    hors ligne, dans une boucle fermée.

    Cette séquence sert uniquement de référence de comparaison et ne
    constitue jamais une stratégie déployable en temps réel.
    """
    n = len(df)

    soc_eb = soc_eb0
    soc_pb = soc_pb0
    alpha_prev = None

    p_seq = df[
        "hasPower"
    ].to_numpy(
        dtype=float
    )

    traj = {
        "SOC_EB": np.zeros(n + 1),
        "SOC_PB": np.zeros(n + 1),
    }

    traj["SOC_EB"][0] = soc_eb0
    traj["SOC_PB"][0] = soc_pb0

    for key in [
        "P_EB",
        "P_PB",
        "alpha_final",
        "cost",
        "P_unserved",
        "P_regen_curtailed",
    ]:
        traj[key] = np.zeros(n)

    traj["feasible"] = np.zeros(
        n,
        dtype=bool,
    )

    for t in range(n):
        p_dem = p_seq[t]

        if abs(p_dem) <= EPS_POWER_W:
            alpha_opt = (
                alpha_prev
                if alpha_prev is not None
                else 0.5
            )

            p_eb_opt = 0.0
            p_pb_opt = 0.0
            cost_opt = 0.0
            feasible_opt = True
            p_unserved = 0.0
            p_regen_curtailed = 0.0

        else:
            metrics = candidate_metrics(
                ALPHA_GRID,
                p_dem,
                soc_eb,
                soc_pb,
                alpha_prev,
            )

            feasible_idx = np.flatnonzero(
                metrics["feasible"]
            )

            if len(feasible_idx) > 0:
                best = feasible_idx[
                    np.argmin(
                        metrics["total_cost"][
                            feasible_idx
                        ]
                    )
                ]

                alpha_opt = float(
                    ALPHA_GRID[best]
                )

                p_eb_opt = float(
                    metrics["P_EB"][best]
                )

                p_pb_opt = float(
                    metrics["P_PB"][best]
                )

                cost_opt = float(
                    metrics["total_cost"][best]
                )

                feasible_opt = True
                p_unserved = 0.0
                p_regen_curtailed = 0.0

            else:
                decision_repli = resoudre_decision_physique(
                    0.5,
                    p_dem,
                    soc_eb,
                    soc_pb,
                    alpha_prev,
                )

                alpha_opt = decision_repli[
                    "alpha_final"
                ]

                p_eb_opt = decision_repli[
                    "P_EB_final"
                ]

                p_pb_opt = decision_repli[
                    "P_PB_final"
                ]

                p_unserved = decision_repli[
                    "P_unserved"
                ]

                p_regen_curtailed = decision_repli[
                    "P_regen_curtailed"
                ]

                cost_opt = np.nan
                feasible_opt = False

        traj["P_EB"][t] = p_eb_opt
        traj["P_PB"][t] = p_pb_opt
        traj["alpha_final"][t] = alpha_opt
        traj["cost"][t] = cost_opt
        traj["feasible"][t] = feasible_opt
        traj["P_unserved"][t] = p_unserved
        traj["P_regen_curtailed"][t] = p_regen_curtailed

        soc_eb, soc_pb, _ = update_soc(
            soc_eb,
            soc_pb,
            p_eb_opt,
            p_pb_opt,
        )

        traj["SOC_EB"][t + 1] = soc_eb
        traj["SOC_PB"][t + 1] = soc_pb

        if abs(p_dem) > EPS_POWER_W:
            alpha_prev = alpha_opt

    traj["SOC_EB_final"] = soc_eb
    traj["SOC_PB_final"] = soc_pb

    return traj


def cout_etendu(
    p_unserved,
    p_regen_curtailed,
):
    return (
        EXTENDED_COST_WEIGHTS["unserved"]
        * (
            p_unserved
            / EXTENDED_COST_P_NORM
        ) ** 2
        + EXTENDED_COST_WEIGHTS["regen_curtailed"]
        * (
            p_regen_curtailed
            / EXTENDED_COST_P_NORM
        ) ** 2
    )


def update_soc(
    soc_eb,
    soc_pb,
    p_eb,
    p_pb,
):
    i_eb = (
        p_eb
        / V_EB_PACK_NOM
    )

    i_pb = (
        p_pb
        / V_PB_PACK_NOM
    )

    soc_eb_raw = (
        soc_eb
        - i_eb
        * DT_SECONDS
        / (
            3600.0
            * CAPACITY_EB_AH
        )
    )

    soc_pb_raw = (
        soc_pb
        - i_pb
        * DT_SECONDS
        / (
            3600.0
            * CAPACITY_PB_AH
        )
    )

    violation = (
        not (
            SOC_EB_MIN - SOC_TOL
            <= soc_eb_raw
            <= SOC_EB_MAX + SOC_TOL
        )
        or not (
            SOC_PB_MIN - SOC_TOL
            <= soc_pb_raw
            <= SOC_PB_MAX + SOC_TOL
        )
    )

    return (
        float(
            np.clip(
                soc_eb_raw,
                SOC_EB_MIN,
                SOC_EB_MAX,
            )
        ),
        float(
            np.clip(
                soc_pb_raw,
                SOC_PB_MIN,
                SOC_PB_MAX,
            )
        ),
        violation,
    )


def current_from_power(
    p_eb,
    p_pb,
):
    """
    Calcule les courants instantanés I_EB et I_PB, en ampères,
    à partir des puissances P_EB et P_PB exprimées en watts.
    """
    return (
        p_eb / V_EB_PACK_NOM,
        p_pb / V_PB_PACK_NOM,
    )


# ============================================================
# Logique floue — copie fidèle de 09_EMS_fuzzy_logic.ipynb
# Cellules 2, 4, 6 et 8
#
# SOC_LOW_THRESHOLD = 0.30 CONFIRME dans 01_configuration.ipynb
# (n'etait qu'une valeur supposee auparavant).
# ============================================================

SOC_LOW_THRESHOLD = 0.30  # confirme (01_configuration.ipynb)

SOC_LOW_FULL_EB = (
    SOC_EB_MIN
    + SOC_LOW_THRESHOLD
) / 2.0

SOC_LOW_FULL_PB = (
    SOC_PB_MIN
    + SOC_LOW_THRESHOLD
) / 2.0

FUZZY_DEFAULT_ALPHA = 0.30

FUZZY_RULE_NAMES = [
    "R1_PB_low_traction",
    "R2_EB_low_PB_available",
    "R3_strong_traction",
    "R4_zero_demand",
    "R5_regenerative_braking",
    "R5b_PB_high_recharge",
    "R7_two_low_SOC",
]

FUZZY_RULE_CONSEQUENTS = np.array(
    [
        0.20,
        0.75,
        0.75,
        0.20,
        0.75,
        0.20,
        0.50,
    ],
    dtype=float,
)


def left_shoulder(
    x,
    full_until,
    zero_from,
):
    x = np.asarray(
        x,
        dtype=float,
    )

    return np.where(
        x <= full_until,
        1.0,
        np.where(
            x >= zero_from,
            0.0,
            (
                zero_from
                - x
            )
            / (
                zero_from
                - full_until
            ),
        ),
    )


def right_shoulder(
    x,
    zero_until,
    full_from,
):
    x = np.asarray(
        x,
        dtype=float,
    )

    return np.where(
        x <= zero_until,
        0.0,
        np.where(
            x >= full_from,
            1.0,
            (
                x
                - zero_until
            )
            / (
                full_from
                - zero_until
            ),
        ),
    )


def trapmf(
    x,
    a,
    b,
    c,
    d,
):
    x = np.asarray(
        x,
        dtype=float,
    )

    y = np.zeros_like(
        x,
        dtype=float,
    )

    left = (
        (x > a)
        & (x < b)
    )

    plateau = (
        (x >= b)
        & (x <= c)
    )

    right = (
        (x > c)
        & (x < d)
    )

    y[left] = (
        x[left]
        - a
    ) / (
        b
        - a
    )

    y[plateau] = 1.0

    y[right] = (
        d
        - x[right]
    ) / (
        d
        - c
    )

    return y


def soc_eb_low(value):
    return left_shoulder(
        value,
        SOC_LOW_FULL_EB,
        SOC_LOW_THRESHOLD,
    )


def soc_pb_low(value):
    return left_shoulder(
        value,
        SOC_LOW_FULL_PB,
        SOC_LOW_THRESHOLD,
    )


def soc_medium(value):
    return trapmf(
        value,
        0.25,
        0.35,
        0.65,
        0.75,
    )


def soc_high(value):
    return right_shoulder(
        value,
        0.70,
        0.80,
    )


def power_strong_recharge(value):
    return left_shoulder(
        value,
        P_EB_MIN_W - 2000.0,
        P_EB_MIN_W,
    )


def power_moderate_recharge(value):
    return trapmf(
        value,
        P_EB_MIN_W - 1000.0,
        P_EB_MIN_W,
        -2.0 * EPS_POWER_W,
        -EPS_POWER_W,
    )


def power_zero(value):
    return trapmf(
        value,
        -2.0 * EPS_POWER_W,
        -EPS_POWER_W,
        EPS_POWER_W,
        2.0 * EPS_POWER_W,
    )


def power_moderate_traction(value):
    return trapmf(
        value,
        EPS_POWER_W,
        2.0 * EPS_POWER_W,
        0.60 * P_EB_MAX_W,
        P_EB_MAX_W,
    )


def power_strong_traction(value):
    return right_shoulder(
        value,
        0.60 * P_EB_MAX_W,
        P_EB_MAX_W,
    )


def acceleration_braking(value):
    return left_shoulder(
        value,
        -1.20,
        -0.20,
    )


def acceleration_stable(value):
    return trapmf(
        value,
        -0.40,
        -0.10,
        0.10,
        0.40,
    )


def alpha_fuzzy_calc(
    soc_eb,
    soc_pb,
    p_dem,
    acceleration,
):
    soc_eb = np.asarray(
        soc_eb,
        dtype=float,
    )

    soc_pb = np.asarray(
        soc_pb,
        dtype=float,
    )

    p_dem = np.asarray(
        p_dem,
        dtype=float,
    )

    acceleration = np.asarray(
        acceleration,
        dtype=float,
    )

    eb_low = soc_eb_low(
        soc_eb
    )

    pb_low = soc_pb_low(
        soc_pb
    )

    pb_medium = soc_medium(
        soc_pb
    )

    pb_high = soc_high(
        soc_pb
    )

    strong_recharge = power_strong_recharge(
        p_dem
    )

    moderate_recharge = power_moderate_recharge(
        p_dem
    )

    zero_demand = power_zero(
        p_dem
    )

    moderate_traction = power_moderate_traction(
        p_dem
    )

    strong_traction = power_strong_traction(
        p_dem
    )

    stable = acceleration_stable(
        acceleration
    )

    traction = np.maximum(
        moderate_traction,
        strong_traction,
    )

    recharge_power = np.maximum(
        strong_recharge,
        moderate_recharge,
    )

    recharge = np.where(
        p_dem < -EPS_POWER_W,
        recharge_power,
        0.0,
    )

    pb_available = np.maximum(
        pb_medium,
        pb_high,
    )

    pb_rechargeable = np.maximum(
        pb_low,
        pb_medium,
    )

    w1 = np.minimum(
        pb_low,
        traction,
    )

    w2 = np.minimum.reduce(
        [
            eb_low,
            pb_available,
            traction,
        ]
    )

    w3 = np.minimum(
        strong_traction,
        pb_available,
    )

    w4 = np.minimum(
        zero_demand,
        stable,
    )

    w5 = np.minimum(
        recharge,
        pb_rechargeable,
    )

    w5b = np.minimum(
        recharge,
        pb_high,
    )

    w7 = np.minimum(
        eb_low,
        pb_low,
    )

    strengths = np.stack(
        [
            w1,
            w2,
            w3,
            w4,
            w5,
            w5b,
            w7,
        ],
        axis=-1,
    )

    strength_sum = strengths.sum(
        axis=-1
    )

    coverage = (
        strength_sum
        > 1e-9
    )

    weighted_sum = (
        strengths
        * FUZZY_RULE_CONSEQUENTS
    ).sum(
        axis=-1
    )

    alpha = np.full(
        strength_sum.shape,
        FUZZY_DEFAULT_ALPHA,
        dtype=float,
    )

    np.divide(
        weighted_sum,
        strength_sum,
        out=alpha,
        where=coverage,
    )

    alpha = np.clip(
        alpha,
        0.0,
        1.0,
    )

    dominant_index = np.argmax(
        strengths,
        axis=-1,
    )

    rule_names = np.asarray(
        FUZZY_RULE_NAMES,
        dtype=object,
    )

    dominant_rule = np.where(
        coverage,
        rule_names[dominant_index],
        "DEFAULT",
    )

    return {
        "alpha": alpha,
        "strengths": strengths,
        "coverage": coverage,
        "dominant_rule": dominant_rule,
    }


# ============================================================
# États symboliques et libellés des règles
# Pages Explicabilité et Ontologie
#
# ATTENTION : compute_symbolic_states fournit une lecture
# simplifiée à seuils fixes. Cette fonction N'EST PAS un
# moteur d'inférence OWL/SWRL exécuté en direct.
# ============================================================

SYMBOLIC_HIGH_POWER_RATIO = 0.60  # cohérent avec le seuil de power_strong_traction


HIGH_POWER_THRESHOLD_W = 20000.0  # confirme (01_configuration.ipynb / 07_EMS_symbolic_states.ipynb)
CONVERTER_RISK_THRESHOLD = 0.85  # confirme (07_EMS_symbolic_states.ipynb)


def compute_symbolic_states(
    p_dem,
    soc_eb,
    soc_pb,
    p_eb=None,
):
    """Reproduit exactement add_symbolic_states de 07_EMS_symbolic_states.ipynb.

    p_eb est necessaire pour converter_risk (utilisation du convertisseur) ;
    s'il n'est pas fourni (p.ex. avant qu'une decision d'alpha n'ait ete prise),
    on estime la puissance convertisseur a partir de p_dem seul comme
    approximation raisonnable -- moins precis que d'utiliser le vrai P_EB
    du pas precedent."""
    p_conv_estime = estimate_p_conv(np.array([p_eb if p_eb is not None else p_dem]))[0]
    utilisation = (
        p_conv_estime / P_CONV_MAX_W
        if p_conv_estime >= 0
        else abs(p_conv_estime) / abs(P_CONV_MIN_W)
    )

    return {
        "EB_available": bool(soc_eb > SOC_EB_MIN),
        "PB_available": bool(soc_pb > SOC_PB_MIN),
        "EB_low_SOC": bool(soc_eb <= SOC_LOW_THRESHOLD),
        "PB_low_SOC": bool(soc_pb <= SOC_LOW_THRESHOLD),
        "high_power_demand": bool(abs(p_dem) >= HIGH_POWER_THRESHOLD_W),
        "regenerative_braking": bool(p_dem < -EPS_POWER_W),
        "zero_power_demand": bool(abs(p_dem) <= EPS_POWER_W),
        "converter_risk": bool(utilisation >= CONVERTER_RISK_THRESHOLD),
    }


RULE_LABELS_FR = {
    "R1_PB_low_traction": (
        "limiter la PB, car elle est presque déchargée, "
        "même en phase de traction"
    ),
    "R2_EB_low_PB_available": (
        "faire porter l'effort sur la PB, car l'EB est faible "
        "tandis que la PB reste disponible"
    ),
    "R3_strong_traction": (
        "solliciter davantage la PB en raison d'une forte "
        "demande de traction"
    ),
    "R4_zero_demand": (
        "maintenir une répartition stable, car la demande "
        "est presque nulle"
    ),
    "R5_regenerative_braking": (
        "orienter la récupération d'énergie vers la PB"
    ),
    "R5b_PB_high_recharge": (
        "limiter la recharge de la PB, car elle est déjà "
        "fortement chargée"
    ),
    "R7_two_low_SOC": (
        "adopter une répartition prudente, car les deux "
        "batteries présentent un SOC faible"
    ),
    "DEFAULT": (
        "appliquer la répartition par défaut, car aucune "
        "règle ne domine clairement"
    ),
}


# ============================================================
# Classes des modèles
# ============================================================

class MLPSimpleModel(nn.Module):

    def __init__(
        self,
        n_in,
        h1,
        h2,
        dropout,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(
                n_in,
                h1,
            ),
            nn.ReLU(),
            nn.Dropout(
                dropout
            ),
            nn.Linear(
                h1,
                h2,
            ),
            nn.ReLU(),
            nn.Linear(
                h2,
                1,
            ),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x,
    ):
        return self.net(
            x
        ).squeeze(-1)


class DirectAlphaMLP(nn.Module):

    def __init__(
        self,
        n_in,
        h1,
        h2,
        dropout,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                n_in,
                h1,
            ),
            nn.ReLU(),
            nn.Dropout(
                dropout
            ),
            nn.Linear(
                h1,
                h2,
            ),
            nn.ReLU(),
            nn.Linear(
                h2,
                1,
            ),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x,
    ):
        return self.network(
            x
        ).squeeze(-1)


class MLPNeuroSymbolique(nn.Module):

    def __init__(
        self,
        n_in,
        hidden1,
        hidden2,
        dropout,
        max_delta,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(
                n_in,
                hidden1,
            ),
            nn.ReLU(),
            nn.Dropout(
                dropout
            ),
            nn.Linear(
                hidden1,
                hidden2,
            ),
            nn.ReLU(),
            nn.Linear(
                hidden2,
                1,
            ),
        )

        self.max_delta = max_delta
        self.tanh = nn.Tanh()
        self.clip = nn.Hardtanh(
            0.0,
            1.0,
        )

    def forward(
        self,
        x,
        alpha_fuzzy,
    ):
        z = self.net(
            x
        ).squeeze(-1)

        delta_alpha = (
            self.max_delta
            * self.tanh(z)
        )

        alpha_unclamped = (
            alpha_fuzzy
            + delta_alpha
        )

        alpha = self.clip(
            alpha_unclamped
        )

        return (
            alpha,
            delta_alpha,
            alpha_unclamped,
        )


class LSTMSeul(nn.Module):

    def __init__(
        self,
        n_features,
        hidden_size,
        num_layers,
        dropout,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            n_features,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=(
                dropout
                if num_layers > 1
                else 0.0
            ),
        )

        self.fc = nn.Sequential(
            nn.Linear(
                hidden_size,
                32,
            ),
            nn.ReLU(),
            nn.Linear(
                32,
                3,
            ),
        )

    def forward(
        self,
        x,
    ):
        out, _ = self.lstm(
            x
        )

        return self.fc(
            out[:, -1, :]
        )


class LSTMNeuroSymbolique(nn.Module):

    def __init__(
        self,
        n_features,
        hidden_size,
        num_layers,
        dropout,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            n_features,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=(
                dropout
                if num_layers > 1
                else 0.0
            ),
        )

        self.head = nn.Sequential(
            nn.Linear(
                hidden_size,
                32,
            ),
            nn.ReLU(),
            nn.Linear(
                32,
                3,
            ),
        )

    def forward(
        self,
        x,
    ):
        out, _ = self.lstm(
            x
        )

        return self.head(
            out[:, -1, :]
        )


class GNNSimple(nn.Module):
    """
    Architecture confirmée dans 06_EMS_GNN.ipynb :

    - couches GCNConv empilées ;
    - agrégation global_mean_pool ;
    - tête de sortie avec activation Sigmoid.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_layers,
        dropout,
    ):
        super().__init__()

        if not _import_torch_geometric():
            raise ImportError(
                "torch_geometric est requis pour EMS_GNN "
                "(pip install torch_geometric)."
            )

        if num_layers < 1:
            raise ValueError(
                "GNN_NUM_LAYERS doit être supérieur ou égal à 1."
            )

        self.convs = nn.ModuleList()

        self.convs.append(
            GCNConv(
                input_dim,
                hidden_dim,
            )
        )

        for _ in range(
            num_layers - 1
        ):
            self.convs.append(
                GCNConv(
                    hidden_dim,
                    hidden_dim,
                )
            )

        self.dropout = nn.Dropout(
            dropout
        )

        self.head = nn.Sequential(
            nn.Linear(
                hidden_dim,
                16,
            ),
            nn.ReLU(),
            nn.Dropout(
                dropout
            ),
            nn.Linear(
                16,
                1,
            ),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x,
        edge_index,
        batch,
    ):
        for conv in self.convs:
            x = conv(
                x,
                edge_index,
            )

            x = torch.relu(
                x
            )

            x = self.dropout(
                x
            )

        graph_embedding = global_mean_pool(
            x,
            batch,
        )

        return self.head(
            graph_embedding
        ).squeeze(-1)


# ============================================================
# Configuration des sept stratégies
# Chemins, hyperparamètres et colonnes d'entrée
# ============================================================

# --- EMS_MLP -------------------------------------------------------------

MLP_CHECKPOINT = (
    CHECKPOINTS_DIR
    / "EMS_MLP.pt"
)

MLP_SCALER_FILE = MODELS_DIR / "EMS_MLP_scalers.npz"  # confirme par l'utilisateur

# CONFIRME empiriquement (chargement reussi du checkpoint reel) :
# 5 colonnes suffisent pour EMS_MLP.

MLP_INPUT_COLS = [
    "SOC_EB",
    "SOC_PB",
    "hasPower",
    "speed",
    "hasAcceleration",
]

MLP_HIDDEN_1 = 64
MLP_HIDDEN_2 = 32
MLP_DROPOUT = 0.10  # À CONFIRMER


# --- EMS_MLP_neurosymbolic -----------------------------------------------

MLP_NS_CHECKPOINT = (
    CHECKPOINTS_DIR
    / "EMS_MLP_neurosymbolic.pt"
)

MLP_NS_SCALER_FILE = MODELS_DIR / "mlp_ns_scalers.npz"  # A CONFIRMER si le nom differe

# CONFIRME dans 01_configuration.ipynb (liste et ordre exacts). Note importante :
# les 3 colonnes Pdem_pred_ns / delta_soc_eb_pred_ns / delta_soc_pb_pred_ns sont
# les SORTIES BRUTES d'EMS_LSTM -- ce modele depend donc d'EMS_LSTM charge en
# amont, en plus de la logique floue (alpha_ems_fuzzy_logic). Voir
# construire_entree_mlp_neurosymbolic.
MLP_NS_INPUT_COLS = [
    "SOC_EB", "SOC_PB", "hasPower", "speed", "hasAcceleration",
    "Pdem_pred_ns", "delta_soc_eb_pred_ns", "delta_soc_pb_pred_ns",
    "alpha_ems_fuzzy_logic",
    "EB_available", "PB_available", "EB_low_SOC", "PB_low_SOC",
    "high_power_demand", "regenerative_braking", "zero_power_demand",
    "converter_risk",
]  # confirme -- necessite EMS_LSTM charge + converter_risk (non calcule
   # actuellement dans compute_symbolic_states, voir avertissement associe)

MLP_NS_HIDDEN_1 = 64
MLP_NS_HIDDEN_2 = 32
MLP_NS_DROPOUT = 0.10  # À CONFIRMER

MLP_NS_MAX_DELTA = 0.20  # confirme (01_configuration.ipynb : MAX_DELTA_ALPHA, etait 0.30 avant)
GAMMA_FUSION = 0.30  # confirme (01_configuration.ipynb) -- role exact dans la fusion alpha non confirme, non utilise pour l'instant


# --- EMS_LSTM / EMS_LSTM_neurosymbolic ----------------------------------

LSTM_CHECKPOINT = (
    CHECKPOINTS_DIR
    / "EMS_LSTM.pt"
)

LSTM_SCALER_FILE = MODELS_DIR / "EMS_LSTM_scalers.npz"  # A CONFIRMER si le nom differe

LSTM_NS_CHECKPOINT = (
    CHECKPOINTS_DIR
    / "EMS_LSTM_neurosymbolic.pt"
)

LSTM_NS_SCALER_FILE = MODELS_DIR / "EMS_LSTM_neurosymbolic_scalers.npz"  # A CONFIRMER si le nom differe

# CONFIRME dans 01_configuration.ipynb (liste et ordre exacts) :
LSTM_FEATURE_COLS = [
    "speed", "hasPower", "hasAcceleration", "hasTotalForce",
    "SOC_EB", "SOC_PB", "I_EB",
] 


LSTM_NS_FEATURE_COLS = [
    "speed", "hasPower", "hasAcceleration", "hasTotalForce",
    "SOC_EB", "SOC_PB", "I_EB",
    "high_power_demand", "regenerative_braking", "zero_power_demand", "converter_risk",
] 



LSTM_OUTPUT_NAMES = ["Pdem_future", "delta_SOC_EB", "delta_SOC_PB"]  

LSTM_WINDOW = 20 
LSTM_HIDDEN_SIZE = 64
LSTM_NUM_LAYERS = 2
LSTM_DROPOUT = 0.20  


# --- EMS_GNN (confirmé) --------------------------------------------------

GNN_NODE_NAMES = [
    "energy_battery",
    "power_battery",
    "converter",
    "motor",
    "vehicle",
]

GNN_CONTINUOUS_FEATURE_NAMES = [
    "soc",
    "current_discharge_max_a",
    "capacity_ah",
    "power_discharge_max_w",
    "power_charge_max_abs_w",
    "power_demand_w",
    "acceleration",
]

GNN_INPUT_DIM = (
    len(GNN_CONTINUOUS_FEATURE_NAMES)
    + len(GNN_NODE_NAMES)
)

GNN_HIDDEN_SIZE = 32
GNN_NUM_LAYERS = 2
GNN_DROPOUT = 0.10
GNN_TARGET = "alpha_historical"

GNN_CHECKPOINT = (
    CHECKPOINTS_DIR
    / "EMS_GNN.pt"
)

GNN_SCALER_FILE = (
    MODELS_DIR
    / "gnn_node_scalers.npz"
)

GNN_GRAPHS_FILE = (
    PROCESSED_DIR
    / "hess_graphs.pt"
)


# ============================================================
# Description des stratégies
# Source commune aux pages 7 (résumé) et 8 (tableau)
# ============================================================

MODEL_ORDER = [
    "EMS_power_limitation",
    "EMS_fuzzy_logic",
    "EMS_MLP",
    "EMS_MLP_neurosymbolic",
    "EMS_LSTM",
    "EMS_LSTM_neurosymbolic",
    "EMS_GNN",
]


MODEL_DISPLAY_NAMES = {
    "EMS_power_limitation": "EMS limitation de puissance (priorité EB)",
    "EMS_fuzzy_logic": "EMS logique floue",
    "EMS_MLP": "EMS MLP",
    "EMS_MLP_neurosymbolic": "EMS MLP neurosymbolique",
    "EMS_LSTM": "EMS LSTM",
    "EMS_LSTM_neurosymbolic": "EMS LSTM neurosymbolique",
    "EMS_GNN": "EMS GNN",
}


MODEL_CONSTRUCTION_SUMMARY = {
    "EMS_power_limitation": (
        "Règle déterministe de type if-then : l'EB est prioritaire dans "
        "la limite de sa puissance maximale, puis la PB fournit le complément. "
        "Cette stratégie reste toujours calculable et sert de solution de secours."
    ),
    "EMS_fuzzy_logic": (
        "Système d'inférence floue de type Mamdani fondé sur sept règles pondérées. "
        "Les concepts utilisés, comme le SOC faible ou la forte traction, ont été "
        "formalisés hors ligne à l'aide de l'ontologie OntoHESS."
    ),
    "EMS_MLP": (
        "Réseau de neurones tabulaire de type MLP qui prédit directement alpha "
        "à partir de l'état instantané du système."
    ),
    "EMS_MLP_neurosymbolic": (
        "MLP entraîné à prédire une correction résiduelle bornée par tanh et clip. "
        "Cette correction est ajoutée à la sortie de la logique floue au lieu de "
        "prédire directement alpha."
    ),
    "EMS_LSTM": (
        "Réseau récurrent LSTM entraîné à prédire les variations de SOC plutôt "
        "que les valeurs absolues, afin de limiter le décalage entre l'entraînement "
        "et le test. Il joue également un second rôle en proposant directement alpha."
    ),
    "EMS_LSTM_neurosymbolic": (
        "Variante qui combine la prédiction temporelle du LSTM avec une correction "
        "symbolique issue de la logique floue."
    ),
    "EMS_GNN": (
        "Réseau de neurones sur graphe GNNSimple. Le système est représenté par "
        "cinq nœuds — EB, PB, convertisseur, moteur et véhicule — portant des "
        "caractéristiques physiques. Une agrégation global_mean_pool est suivie "
        "d'une tête de régression."
    ),
}


MODEL_CONSTRUCTION_DETAILED = [
    {
        "modele": "EMS_power_limitation",
        "type": "Déterministe",
        "donnees_entree": "hasPower, SOC_EB",
        "role": (
            "Stratégie de secours toujours disponible et calculable."
        ),
    },
    {
        "modele": "EMS_fuzzy_logic",
        "type": "Logique floue (Mamdani)",
        "donnees_entree": (
            "SOC_EB, SOC_PB, hasPower, acceleration"
        ),
        "role": (
            "Base de règles réutilisée par les variantes "
            "neurosymboliques."
        ),
    },
    {
        "modele": "EMS_MLP",
        "type": "Supervisé (tabulaire)",
        "donnees_entree": (
            ", ".join(MLP_INPUT_COLS)
        ),
        "role": (
            "Référence neuronale sans composante symbolique. "
            "5 entrées confirmées empiriquement."
        ),
    },
    {
        "modele": "EMS_MLP_neurosymbolic",
        "type": "Neurosymbolique",
        "donnees_entree": (
            "17 entrées attendues par le checkpoint reel -- "
            "liste exacte a confirmer (voir avertissement dans le code)"
        ),
        "role": (
            "Applique une correction résiduelle bornée à la "
            "sortie de la logique floue."
        ),
    },
    {
        "modele": "EMS_LSTM",
        "type": "Supervisé (temporel)",
        "donnees_entree": (
            "7 entrées attendues par le checkpoint reel -- "
            "liste exacte a confirmer (voir avertissement dans le code)"
        ),
        "role": (
            "Fournit également des informations au modèle "
            "EMS_LSTM_neurosymbolic."
        ),
    },
    {
        "modele": "EMS_LSTM_neurosymbolic",
        "type": "Neurosymbolique temporel",
        "donnees_entree": (
            "11 entrées attendues par le checkpoint reel -- "
            "liste exacte a confirmer (voir avertissement dans le code)"
        ),
        "role": (
            "Combine les prédictions du LSTM avec la "
            "connaissance symbolique."
        ),
    },
    {
        "modele": "EMS_GNN",
        "type": "Supervisé (graphe)",
        "donnees_entree": (
            ", ".join(
                GNN_CONTINUOUS_FEATURE_NAMES
            )
        ),
        "role": (
            "Représente le système sous la forme d'un "
            "graphe à cinq nœuds."
        ),
    },
]


# ============================================================
# Chargement des modèles entraînés
# ============================================================

def _resumer_erreur_chargement(exc: Exception, nom_modele: str) -> str:
    """Traduit/simplifie l'erreur (souvent en anglais) de
    model.load_state_dict en un message francais lisible, en particulier
    pour les erreurs de type 'size mismatch' (nombre d'entrees incorrect)."""
    message = str(exc)
    if "size mismatch" in message.lower():
        import re
        mismatches = re.findall(
            r"size mismatch for ([^\s:]+): copying a param with shape "
            r"torch\.Size\(\[([^\]]+)\]\) from checkpoint, the shape in "
            r"current model is torch\.Size\(\[([^\]]+)\]\)",
            message,
        )
        if mismatches:
            details = "; ".join(
                f"{nom} : le fichier de poids attend {forme_checkpoint}, "
                f"le modele construit ici utilise {forme_modele}"
                for nom, forme_checkpoint, forme_modele in mismatches
            )
            return (
                f"{nom_modele} : incompatibilite de dimensions entre le "
                f"checkpoint et l'architecture definie dans ems_core.py "
                f"({details}). Les hyperparametres ou colonnes d'entree "
                f"utilises ici ne correspondent pas a ceux du modele "
                f"reellement entraine -- a corriger dans ems_core.py."
            )
    return f"{nom_modele} : {message}"


def _load_state_dict(
    checkpoint_path,
):
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Poids introuvables : {checkpoint_path}"
        )

    try:
        return torch.load(
            checkpoint_path,
            map_location=DEVICE,
            weights_only=True,
        )

    except TypeError:
        return torch.load(
            checkpoint_path,
            map_location=DEVICE,
        )


def verifier_colonnes_scaler(scaler, colonnes_attendues, nom_modele):
    """Si le scaler contient 'input_cols' (confirme present dans les scalers
    reels du projet), compare a la liste de colonnes utilisee ici et retourne
    un avertissement en cas d'ecart -- source de verite superieure a nos
    constantes deduites/supposees, puisque directement sauvegardee a
    l'entrainement."""
    if scaler is None or "input_cols" not in scaler.files:
        return None
    reelles = [str(c) for c in scaler["input_cols"].tolist()]
    if reelles != list(colonnes_attendues):
        return (
            f"{nom_modele} : les colonnes utilisees ici ne correspondent pas a "
            f"celles sauvegardees dans le scaler -- attendu {reelles}, "
            f"utilise {list(colonnes_attendues)}."
        )
    return None


def charger_scaler(scaler_path):
    """Charge un fichier de normalisation .npz si present, sinon retourne None
    sans lever d'erreur (la normalisation est alors simplement desactivee pour
    ce modele, avec un avertissement affiche a l'appelant)."""
    if not scaler_path.exists():
        return None
    return np.load(scaler_path)


def appliquer_scaler(valeurs, scaler):
    """Normalise un vecteur de features brutes (1D, meme ordre qu'a
    l'entrainement) en detectant automatiquement le format du scaler parmi
    les conventions les plus courantes (StandardScaler, MinMaxScaler, sklearn
    ou artisanal). Leve ValueError si aucun format reconnu n'est trouve --
    l'appelant doit alors retomber sur les valeurs brutes avec un avertissement."""
    cles = set(scaler.files)
    valeurs = np.asarray(valeurs, dtype=np.float64)

    if {"x_mean", "x_std"}.issubset(cles):  # confirme (05_EMS_graph_construction, format GNN)
        ecart = scaler["x_std"]
        return (valeurs - scaler["x_mean"]) / np.where(ecart == 0, 1.0, ecart)
    if {"mean", "scale"}.issubset(cles):
        echelle = scaler["scale"]
        return (valeurs - scaler["mean"]) / np.where(echelle == 0, 1.0, echelle)
    if {"mean_", "scale_"}.issubset(cles):
        echelle = scaler["scale_"]
        return (valeurs - scaler["mean_"]) / np.where(echelle == 0, 1.0, echelle)
    if {"mean", "std"}.issubset(cles):
        ecart = scaler["std"]
        return (valeurs - scaler["mean"]) / np.where(ecart == 0, 1.0, ecart)
    if {"data_min_", "data_max_"}.issubset(cles):
        plage = scaler["data_max_"] - scaler["data_min_"]
        return (valeurs - scaler["data_min_"]) / np.where(plage == 0, 1.0, plage)
    if {"min", "max"}.issubset(cles):
        plage = scaler["max"] - scaler["min"]
        return (valeurs - scaler["min"]) / np.where(plage == 0, 1.0, plage)

    raise ValueError(
        f"Format de scaler non reconnu -- cles disponibles : {sorted(cles)}. "
        "Formats geres : mean/scale, mean_/scale_, mean/std, "
        "data_min_/data_max_, min/max."
    )


def load_mlp_simple():
    model = MLPSimpleModel(
        len(MLP_INPUT_COLS),
        MLP_HIDDEN_1,
        MLP_HIDDEN_2,
        MLP_DROPOUT,
    ).to(DEVICE)

    try:
        model.load_state_dict(
            _load_state_dict(
                MLP_CHECKPOINT
            )
        )
    except RuntimeError as exc:
        raise RuntimeError(_resumer_erreur_chargement(exc, "EMS_MLP")) from exc

    model.eval()

    return model


def load_mlp_neurosymbolic():
    model = MLPNeuroSymbolique(
        len(MLP_NS_INPUT_COLS),
        MLP_NS_HIDDEN_1,
        MLP_NS_HIDDEN_2,
        MLP_NS_DROPOUT,
        MLP_NS_MAX_DELTA,
    ).to(DEVICE)

    try:
        model.load_state_dict(
            _load_state_dict(
                MLP_NS_CHECKPOINT
            )
        )
    except RuntimeError as exc:
        raise RuntimeError(_resumer_erreur_chargement(exc, "EMS_MLP_neurosymbolic")) from exc

    model.eval()

    return model


def load_lstm_seul():
    model = LSTMSeul(
        len(LSTM_FEATURE_COLS),
        LSTM_HIDDEN_SIZE,
        LSTM_NUM_LAYERS,
        LSTM_DROPOUT,
    ).to(DEVICE)

    try:
        model.load_state_dict(
            _load_state_dict(
                LSTM_CHECKPOINT
            )
        )
    except RuntimeError as exc:
        raise RuntimeError(_resumer_erreur_chargement(exc, "EMS_LSTM")) from exc

    model.eval()

    return model


def load_lstm_neurosymbolic():
    model = LSTMNeuroSymbolique(
        len(LSTM_NS_FEATURE_COLS),
        LSTM_HIDDEN_SIZE,
        LSTM_NUM_LAYERS,
        LSTM_DROPOUT,
    ).to(DEVICE)

    try:
        model.load_state_dict(
            _load_state_dict(
                LSTM_NS_CHECKPOINT
            )
        )
    except RuntimeError as exc:
        raise RuntimeError(_resumer_erreur_chargement(exc, "EMS_LSTM_neurosymbolic")) from exc

    model.eval()

    return model


def load_gnn_simple():
    if not _import_torch_geometric():
        raise ImportError(
            "torch_geometric n'est pas installé et reste "
            "nécessaire pour utiliser EMS_GNN."
        )

    model = GNNSimple(
        GNN_INPUT_DIM,
        GNN_HIDDEN_SIZE,
        GNN_NUM_LAYERS,
        GNN_DROPOUT,
    ).to(DEVICE)

    try:
        model.load_state_dict(
            _load_state_dict(
                GNN_CHECKPOINT
            )
        )
    except RuntimeError as exc:
        raise RuntimeError(_resumer_erreur_chargement(exc, "EMS_GNN")) from exc

    model.eval()

    scaler = None

    if GNN_SCALER_FILE.exists():
        scaler = np.load(
            GNN_SCALER_FILE
        )

    return (
        model,
        scaler,
    )


def load_gnn_test_graphs():
    """
    Charge les graphes de test préconstruits pour les ensembles
    d'entraînement, de validation et de test.

    Ils sont utilisés pour l'évaluation hors ligne et pour GNNExplainer,
    mais pas pour une simulation en boucle fermée sur un nouveau cycle.

    Voir construire_graphe_instant.
    """
    if not GNN_GRAPHS_FILE.exists():
        raise FileNotFoundError(
            f"Fichier de graphes absent : {GNN_GRAPHS_FILE}"
        )

    try:
        graph_data = torch.load(
            GNN_GRAPHS_FILE,
            map_location="cpu",
            weights_only=False,
        )

    except TypeError:
        graph_data = torch.load(
            GNN_GRAPHS_FILE,
            map_location="cpu",
        )

    return graph_data


def construire_graphe_instant(
    p_dem,
    soc_eb,
    soc_pb,
    acceleration,
    scaler=None,
):
    """
    CONFIRME -- reproduit build_raw_node_features + build_graph de
    05_EMS_graph_construction.ipynb.

    Aretes exactes (bidirectionnelles) : energy_battery-converter,
    power_battery-converter, converter-motor, energy_battery-motor (liaison
    directe confirmee, absente d'une hypothese precedente), motor-vehicle.

    p_dem et acceleration sont dupliques sur les 5 noeuds (confirme -- pas
    seulement moteur/vehicule comme suppose auparavant). SOC uniquement sur
    EB/PB. Le convertisseur porte un courant max de decharge egal a
    P_CONV_MAX_W / V_HESS_BUS_NOM (confirme), pas zero.

    Si un scaler (x_mean/x_std, cf. gnn_node_scalers.npz) est fourni, les 7
    features continues sont normalisees avant d'etre concatenees a
    l'encodage one-hot du type de noeud -- exactement comme a l'entrainement.
    """
    if not _import_torch_geometric():
        raise ImportError(
            "torch_geometric est requis pour construire_graphe_instant."
        )

    idx = {nom: i for i, nom in enumerate(GNN_CONTINUOUS_FEATURE_NAMES)}
    i_eb, i_pb, i_conv, i_motor, i_veh = 0, 1, 2, 3, 4

    features = np.zeros((5, len(GNN_CONTINUOUS_FEATURE_NAMES)), dtype=np.float32)

    # SOC (uniquement EB/PB, 0 ailleurs -- confirme)
    features[i_eb, idx["soc"]] = soc_eb
    features[i_pb, idx["soc"]] = soc_pb

    # Caracteristiques statiques par noeud (confirme, STATIC_FEATURES)
    features[i_eb, idx["current_discharge_max_a"]] = P_EB_MAX_W / V_EB_PACK_NOM
    features[i_eb, idx["capacity_ah"]] = CAPACITY_EB_AH
    features[i_eb, idx["power_discharge_max_w"]] = P_EB_MAX_W
    features[i_eb, idx["power_charge_max_abs_w"]] = abs(P_EB_MIN_W)

    features[i_pb, idx["current_discharge_max_a"]] = P_PB_MAX_W / V_PB_PACK_NOM
    features[i_pb, idx["capacity_ah"]] = CAPACITY_PB_AH
    features[i_pb, idx["power_discharge_max_w"]] = P_PB_MAX_W
    features[i_pb, idx["power_charge_max_abs_w"]] = abs(P_PB_MIN_W)

    features[i_conv, idx["current_discharge_max_a"]] = P_CONV_MAX_W / V_HESS_BUS_NOM
    features[i_conv, idx["capacity_ah"]] = 0.0
    features[i_conv, idx["power_discharge_max_w"]] = P_CONV_MAX_W
    features[i_conv, idx["power_charge_max_abs_w"]] = abs(P_CONV_MIN_W)

    # motor / vehicle : caracteristiques statiques nulles (confirme)

    # p_dem et acceleration dupliques sur TOUS les noeuds (confirme)
    features[:, idx["power_demand_w"]] = p_dem
    features[:, idx["acceleration"]] = acceleration

    if scaler is not None:
        try:
            for n in range(features.shape[0]):
                features[n, :] = appliquer_scaler(features[n, :], scaler)
        except ValueError:
            pass  # format non reconnu -- on continue sans normaliser

    node_type_onehot = np.eye(5, dtype=np.float32)
    x = np.concatenate([features, node_type_onehot], axis=1)

    # Aretes confirmees (05_EMS_graph_construction.ipynb, UNDIRECTED_EDGES)
    aretes = np.array([
        [i_eb, i_conv], [i_conv, i_eb],
        [i_pb, i_conv], [i_conv, i_pb],
        [i_conv, i_motor], [i_motor, i_conv],
        [i_eb, i_motor], [i_motor, i_eb],
        [i_motor, i_veh], [i_veh, i_motor],
    ], dtype=np.int64).T

    return (
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(aretes, dtype=torch.long),
    )


def deriver_alpha_depuis_sortie_lstm(
    sortie_lstm,
    p_dem,
):
    """
    Convertit les 3 sorties du LSTM -- Pdem, delta_SOC_EB, delta_SOC_PB
    (cf. 19_EMS_explainability.ipynb, dans cet ordre suppose) -- en une
    valeur d'alpha proposee.

    ATTENTION -- INTERPRETATION PHYSIQUE, PAS CONFIRMEE DEPUIS LE NOTEBOOK :
    cette fonction n'utilise aucune formule tiree de 04_EMS_LSTM.ipynb ou
    08_EMS_LSTM_neurosymbolic.ipynb (non fournis). Elle applique en sens
    inverse l'equation de mise a jour du SOC deja confirmee ailleurs dans
    ce fichier (update_soc) :

        delta_SOC = -(P / V) * DT_SECONDS / (3600 * Capacite)
        => P = -delta_SOC * V * 3600 * Capacite / DT_SECONDS

    En appliquant cette inversion aux deux deltas de SOC predits, on obtient
    une puissance EB et PB "impliquee" par la prediction du modele, dont on
    deduit alpha = P_PB / p_dem. C'est une interpretation raisonnable mais
    non confirmee -- a valider ou remplacer si la vraie formule du projet
    est retrouvee.

    Retourne (alpha, diagnostic) ou diagnostic contient P_EB_implique,
    P_PB_implique et l'ecart entre Pdem predit par le modele et p_dem reel
    (un grand ecart suggere que cette interpretation est peu fiable a cet
    instant).
    """
    pdem_pred, delta_soc_eb_pred, delta_soc_pb_pred = (
        float(sortie_lstm[0]),
        float(sortie_lstm[1]),
        float(sortie_lstm[2]),
    )

    p_eb_implique = (
        -delta_soc_eb_pred * V_EB_PACK_NOM * 3600.0 * CAPACITY_EB_AH / DT_SECONDS
    )
    p_pb_implique = (
        -delta_soc_pb_pred * V_PB_PACK_NOM * 3600.0 * CAPACITY_PB_AH / DT_SECONDS
    )

    if abs(p_dem) > EPS_POWER_W:
        alpha = float(np.clip(p_pb_implique / p_dem, 0.0, 1.0))
    else:
        alpha = 0.5

    diagnostic = {
        "Pdem_predit": pdem_pred,
        "P_EB_implique": p_eb_implique,
        "P_PB_implique": p_pb_implique,
        "ecart_Pdem": pdem_pred - p_dem,
    }
    return alpha, diagnostic


# ============================================================
# Simulation en boucle fermée des stratégies déployables
# ============================================================

def _trajectoire_vide(n):
    traj = {
        "SOC_EB": np.zeros(n + 1),
        "SOC_PB": np.zeros(n + 1),
    }

    for key in [
        "P_EB",
        "P_PB",
        "I_EB",
        "I_PB",
        "alpha_requested",
        "alpha_final",
        "cost",
        "P_unserved",
        "P_regen_curtailed",
    ]:
        traj[key] = np.zeros(n)

    traj["correction_applied"] = np.zeros(
        n,
        dtype=bool,
    )

    traj["feasible"] = np.zeros(
        n,
        dtype=bool,
    )

    traj["soc_violation"] = np.zeros(
        n,
        dtype=bool,
    )

    return traj


def simuler_strategie_deterministe(
    df,
    soc_eb0,
    soc_pb0,
    proposer_alpha,
):
    """
    Exécute une boucle de simulation générique.

    La fonction proposer_alpha(
        t,
        ligne,
        soc_eb,
        soc_pb,
        alpha_prev,
    )

    doit retourner une valeur d'alpha proposée.

    Le filtre de sécurité physique est ensuite appliqué
    à chaque pas de simulation.
    """
    n = len(df)

    traj = _trajectoire_vide(
        n
    )

    traj["SOC_EB"][0] = soc_eb0
    traj["SOC_PB"][0] = soc_pb0

    soc_eb = soc_eb0
    soc_pb = soc_pb0
    alpha_prev = None

    p_seq = df[
        "hasPower"
    ].to_numpy(
        dtype=float
    )

    # Pré-extraction de toutes les lignes en dictionnaires natifs, une seule
    # fois. Remplace un df.iloc[t] pandas par pas de temps (très lent) par un
    # simple accès O(1) dans une liste. Résultat strictement identique :
    # lignes[t] == df.iloc[t].to_dict().
    lignes = df.to_dict("records")

    for t in range(n):
        p_dem = p_seq[t]

        alpha_req = proposer_alpha(
            t,
            lignes[t],
            soc_eb,
            soc_pb,
            alpha_prev,
        )

        decision = resoudre_decision_physique(
            alpha_req,
            p_dem,
            soc_eb,
            soc_pb,
            alpha_prev,
        )

        traj["alpha_requested"][t] = decision[
            "alpha_requested"
        ]

        traj["alpha_final"][t] = decision[
            "alpha_final"
        ]

        traj["P_EB"][t] = decision[
            "P_EB_final"
        ]

        traj["P_PB"][t] = decision[
            "P_PB_final"
        ]

        (
            traj["I_EB"][t],
            traj["I_PB"][t],
        ) = current_from_power(
            decision["P_EB_final"],
            decision["P_PB_final"],
        )

        traj["P_unserved"][t] = decision[
            "P_unserved"
        ]

        traj["P_regen_curtailed"][t] = decision[
            "P_regen_curtailed"
        ]

        traj["correction_applied"][t] = decision[
            "correction_applied"
        ]

        traj["feasible"][t] = decision[
            "feasible"
        ]

        traj["cost"][t] = cout_etendu(
            decision["P_unserved"],
            decision["P_regen_curtailed"],
        )

        (
            soc_eb,
            soc_pb,
            violation,
        ) = update_soc(
            soc_eb,
            soc_pb,
            decision["P_EB_final"],
            decision["P_PB_final"],
        )

        traj["SOC_EB"][t + 1] = soc_eb
        traj["SOC_PB"][t + 1] = soc_pb
        traj["soc_violation"][t] = violation

        if abs(p_dem) > EPS_POWER_W:
            alpha_prev = decision[
                "alpha_final"
            ]

    traj["SOC_EB_final"] = soc_eb
    traj["SOC_PB_final"] = soc_pb

    return traj


def simuler_toutes_strategies(
    df,
    soc_eb0,
    soc_pb0,
    modeles_charges=None,
):
    """
    Simule les stratégies EMS déployables sur le cycle fourni.

    Le paramètre modeles_charges est un dictionnaire facultatif contenant
    les modèles déjà chargés, par exemple :

    {
        "EMS_MLP": model,
        "EMS_MLP_neurosymbolic": model,
        ...
    }

    Sur la page 3, les fonctions load_* sont mises en cache avec
    st.cache_resource.

    La fonction retourne deux éléments :

    - resultats : dictionnaire associant chaque stratégie à sa trajectoire ;
    - avertissements : liste de messages concernant les stratégies qui
      n'ont pas pu être simulées en boucle fermée, notamment le GNN
      et les modèles LSTM.
    """
    modeles_charges = (
        modeles_charges
        or {}
    )

    resultats = {}
    avertissements = []


    # --------------------------------------------------------
    # EMS_power_limitation
    # Cette stratégie reste toujours calculable
    # --------------------------------------------------------

    resultats["EMS_power_limitation"] = (
        simuler_strategie_deterministe(
            df,
            soc_eb0,
            soc_pb0,
            proposer_alpha=(
                lambda t, ligne, soc_eb, soc_pb, alpha_prev:
                eb_priority_alpha_single(
                    ligne["hasPower"],
                    soc_eb,
                )
            ),
        )
    )


    # --------------------------------------------------------
    # EMS_fuzzy_logic
    # --------------------------------------------------------

    def _alpha_fuzzy_instant(
        t,
        ligne,
        soc_eb,
        soc_pb,
        alpha_prev,
    ):
        accel = (
            float(
                ligne["hasAcceleration"]
            )
            if "hasAcceleration" in df.columns
            else 0.0
        )

        res = alpha_fuzzy_calc(
            np.array([soc_eb]),
            np.array([soc_pb]),
            np.array(
                [ligne["hasPower"]]
            ),
            np.array([accel]),
        )

        return float(
            res["alpha"][0]
        )


    resultats["EMS_fuzzy_logic"] = (
        simuler_strategie_deterministe(
            df,
            soc_eb0,
            soc_pb0,
            _alpha_fuzzy_instant,
        )
    )


    # --------------------------------------------------------
    # EMS_MLP
    # --------------------------------------------------------

    if "EMS_MLP" in modeles_charges:
        model = modeles_charges[
            "EMS_MLP"
        ]

        scaler_mlp = charger_scaler(MLP_SCALER_FILE)
        if scaler_mlp is None:
            avertissements.append(
                f"EMS_MLP : aucun fichier de normalisation trouve a {MLP_SCALER_FILE} -- "
                "entrees non normalisees, resultats potentiellement peu fiables."
            )
        else:
            avert_colonnes = verifier_colonnes_scaler(scaler_mlp, MLP_INPUT_COLS, "EMS_MLP")
            if avert_colonnes:
                avertissements.append(avert_colonnes)
        _avert_scaler_mlp_deja_signale = [False]

        def _alpha_mlp(
            t,
            ligne,
            soc_eb,
            soc_pb,
            alpha_prev,
        ):
            valeurs = {
                **ligne,
                "SOC_EB": soc_eb,
                "SOC_PB": soc_pb,
            }

            brut = np.array(
                [valeurs[c] for c in MLP_INPUT_COLS],
                dtype=np.float64,
            )

            if scaler_mlp is not None:
                try:
                    brut = appliquer_scaler(brut, scaler_mlp)
                except ValueError as exc:
                    if not _avert_scaler_mlp_deja_signale[0]:
                        avertissements.append(f"EMS_MLP : {exc}")
                        _avert_scaler_mlp_deja_signale[0] = True

            x = torch.tensor(
                [brut.tolist()],
                dtype=torch.float32,
                device=DEVICE,
            )

            with torch.no_grad():
                return float(
                    model(x).item()
                )

        try:
            _t0 = time.time()
            resultats["EMS_MLP"] = (
                simuler_strategie_deterministe(
                    df,
                    soc_eb0,
                    soc_pb0,
                    _alpha_mlp,
                )
            )
            avertissements.append(f"[timing] EMS_MLP : {time.time() - _t0:.1f} s")

        except KeyError as exc:
            avertissements.append(
                "EMS_MLP non simulé : une colonne d'entrée "
                f"est absente ({exc}). Vérifie MLP_INPUT_COLS."
            )




    def _construire_predicteur_lstm_brut(model, feature_cols, window, scaler=None):
        """Retourne une fonction (t, ligne, soc_eb, soc_pb) -> sortie brute
        (numpy, 3 valeurs), en maintenant l'historique de SOC_EB/SOC_PB et en
        reconstruisant I_EB a partir de la variation de SOC_EB d'un pas a
        l'autre (I = -delta_SOC * 3600 * Capacite / DT_SECONDS -- inversion
        de update_soc, deja confirmee ailleurs dans ce fichier). Necessaire
        car I_EB n'est pas une colonne du cycle brut : c'est un etat qui
        depend des decisions prises au fil de la simulation.

        Gere aussi les 4 indicateurs symboliques confirmes pour EMS_LSTM_neurosymbolic
        (high_power_demand, regenerative_braking, zero_power_demand, converter_risk),
        calcules a chaque pas via compute_symbolic_states et suivis dans leur propre
        historique pour construire la fenetre.

        Si un scaler est fourni, chaque pas de la fenetre est normalise
        (meme moyenne/echelle appliquee a chaque instant de la sequence)."""
        colonnes_symboliques = {
            "EB_available", "PB_available", "EB_low_SOC", "PB_low_SOC",
            "high_power_demand", "regenerative_braking", "zero_power_demand", "converter_risk",
        }
        colonnes_uniques = list(dict.fromkeys(feature_cols))
        valeurs_statiques = {
            col: df[col].to_numpy(dtype=np.float32)
            for col in colonnes_uniques
            if col not in ("SOC_EB", "SOC_PB", "I_EB") and col not in colonnes_symboliques
        }
        colonnes_symboliques_utilisees = [c for c in colonnes_uniques if c in colonnes_symboliques]
        # Puissance instantanee pre-extraite une fois (evite un df["hasPower"].iloc[t]
        # pandas a chaque pas dans predire()).
        puissance_instantanee = df["hasPower"].to_numpy(dtype=np.float64)
        historique_soc_eb, historique_soc_pb, historique_i_eb = [], [], []
        historique_symboliques = {c: [] for c in colonnes_symboliques_utilisees}
        avert_scaler_signale = [False]

        def predire(t, soc_eb, soc_pb):
            historique_soc_eb.append(soc_eb)
            historique_soc_pb.append(soc_pb)
            if len(historique_soc_eb) >= 2:
                delta_soc_eb = historique_soc_eb[-2] - historique_soc_eb[-1]
                i_eb_realise = delta_soc_eb * 3600.0 * CAPACITY_EB_AH / DT_SECONDS
            else:
                i_eb_realise = 0.0
            historique_i_eb.append(i_eb_realise)

            if colonnes_symboliques_utilisees:
                p_dem_instant = float(puissance_instantanee[t])
                etats = compute_symbolic_states(
                    p_dem_instant, soc_eb, soc_pb, p_eb=i_eb_realise * V_EB_PACK_NOM,
                )
                for c in colonnes_symboliques_utilisees:
                    historique_symboliques[c].append(float(etats[c]))

            debut = max(0, t - window + 1)
            n_manquant = window - (t - debut + 1)

            fenetre = np.empty((window, len(feature_cols)), dtype=np.float32)
            for j, col in enumerate(feature_cols):
                if col == "SOC_EB":
                    vals = np.asarray(historique_soc_eb[debut:t + 1], dtype=np.float32)
                elif col == "SOC_PB":
                    vals = np.asarray(historique_soc_pb[debut:t + 1], dtype=np.float32)
                elif col == "I_EB":
                    vals = np.asarray(historique_i_eb[debut:t + 1], dtype=np.float32)
                elif col in colonnes_symboliques:
                    vals = np.asarray(historique_symboliques[col][debut:t + 1], dtype=np.float32)
                else:
                    vals = valeurs_statiques[col][debut:t + 1]
                if n_manquant > 0:
                    vals = np.concatenate(
                        [np.full(n_manquant, vals[0], dtype=np.float32), vals]
                    )
                fenetre[:, j] = vals

            if scaler is not None:
                try:
                    # appliquer_scaler diffuse moyenne/echelle sur le dernier axe :
                    # l'appliquer a toute la fenetre (window, n_features) d'un coup
                    # donne exactement le meme resultat que la boucle ligne par ligne,
                    # mais en un seul appel vectorise.
                    fenetre = appliquer_scaler(fenetre, scaler).astype(np.float32)
                except ValueError as exc:
                    if not avert_scaler_signale[0]:
                        avertissements.append(f"Scaler LSTM non applique : {exc}")
                        avert_scaler_signale[0] = True

            x = torch.from_numpy(fenetre[np.newaxis, ...]).to(dtype=torch.float32, device=DEVICE)
            with torch.no_grad():
                sortie_brute = model(x)[0].cpu().numpy()

            
            if scaler is not None and {"y_mean", "y_std"}.issubset(set(scaler.files)):
                sortie_brute = sortie_brute * scaler["y_std"] + scaler["y_mean"]

            return sortie_brute

        return predire

    def _construire_proposeur_lstm(model, feature_cols, window, scaler=None):
        predire = _construire_predicteur_lstm_brut(model, feature_cols, window, scaler)

        def proposer_alpha(t, ligne, soc_eb, soc_pb, alpha_prev):
            sortie = predire(t, soc_eb, soc_pb)
            alpha, _diagnostic = deriver_alpha_depuis_sortie_lstm(sortie, ligne["hasPower"])
            return alpha

        return proposer_alpha

    if "EMS_LSTM" in modeles_charges:
        scaler_lstm = charger_scaler(LSTM_SCALER_FILE)
        if scaler_lstm is None:
            avertissements.append(
                f"EMS_LSTM : aucun fichier de normalisation trouve a {LSTM_SCALER_FILE} -- "
                "entrees non normalisees, resultats potentiellement peu fiables."
            )
        else:
            avert_colonnes = verifier_colonnes_scaler(scaler_lstm, LSTM_FEATURE_COLS, "EMS_LSTM")
            if avert_colonnes:
                avertissements.append(avert_colonnes)
        try:
            proposeur = _construire_proposeur_lstm(
                modeles_charges["EMS_LSTM"], LSTM_FEATURE_COLS, LSTM_WINDOW, scaler_lstm,
            )
            _t0 = time.time()
            resultats["EMS_LSTM"] = simuler_strategie_deterministe(df, soc_eb0, soc_pb0, proposeur)
            avertissements.append(f"[timing] EMS_LSTM : {time.time() - _t0:.1f} s")
            avertissements.append(
                "EMS_LSTM simule en boucle fermee via une interpretation physique non "
                "confirmee de sa sortie (deriver_alpha_depuis_sortie_lstm). Colonnes "
                "d'entree desormais confirmees (01_configuration.ipynb) ; I_EB est "
                "reconstruit a partir de la variation de SOC_EB (non mesure directement)."
            )
        except (KeyError, RuntimeError) as exc:
            avertissements.append(f"EMS_LSTM non simule : {exc}")

    if "EMS_LSTM_neurosymbolic" in modeles_charges:
        scaler_lstm_ns = charger_scaler(LSTM_NS_SCALER_FILE)
        if scaler_lstm_ns is None:
            avertissements.append(
                f"EMS_LSTM_neurosymbolic : aucun fichier de normalisation trouve a "
                f"{LSTM_NS_SCALER_FILE} -- entrees non normalisees si le modele est simule."
            )
        try:
            proposeur = _construire_proposeur_lstm(
                modeles_charges["EMS_LSTM_neurosymbolic"], LSTM_NS_FEATURE_COLS, LSTM_WINDOW, scaler_lstm_ns,
            )
            _t0 = time.time()
            resultats["EMS_LSTM_neurosymbolic"] = simuler_strategie_deterministe(df, soc_eb0, soc_pb0, proposeur)
            avertissements.append(f"[timing] EMS_LSTM_neurosymbolic : {time.time() - _t0:.1f} s")
            avertissements.append(
                "EMS_LSTM_neurosymbolic simulé en boucle fermée avec les 11 colonnes "
                "confirmées (sortie réelle de 08_EMS_LSTM_neurosymbolic.ipynb : 7 variables "
                "physiques + high_power_demand, regenerative_braking, zero_power_demand, "
                "converter_risk). L'interprétation physique de sa sortie en alpha reste "
                "néanmoins non confirmée (deriver_alpha_depuis_sortie_lstm)."
            )
        except (KeyError, RuntimeError) as exc:
            avertissements.append(f"EMS_LSTM_neurosymbolic non simule : {exc}")



    if "EMS_MLP_neurosymbolic" in modeles_charges:
        model = modeles_charges["EMS_MLP_neurosymbolic"]

        if "EMS_LSTM" not in modeles_charges:
            avertissements.append(
                "EMS_MLP_neurosymbolic non simulé : nécessite EMS_LSTM chargé pour "
                "produire Pdem_pred_ns, delta_soc_eb_pred_ns, delta_soc_pb_pred_ns "
                "(confirmé dans 01_configuration.ipynb : MLP_NS_INPUT_COLS en dépend)."
            )
        else:
            scaler_lstm_pour_ns = charger_scaler(LSTM_SCALER_FILE)
            scaler_mlp_ns = charger_scaler(MLP_NS_SCALER_FILE)
            if scaler_mlp_ns is None:
                avertissements.append(
                    f"EMS_MLP_neurosymbolic : aucun fichier de normalisation trouvé à "
                    f"{MLP_NS_SCALER_FILE} -- entrées non normalisées, résultats "
                    "potentiellement peu fiables."
                )
            else:
                avert_colonnes = verifier_colonnes_scaler(scaler_mlp_ns, MLP_NS_INPUT_COLS, "EMS_MLP_neurosymbolic")
                if avert_colonnes:
                    avertissements.append(avert_colonnes)


            predire_lstm_pour_ns = _construire_predicteur_lstm_brut(
                modeles_charges["EMS_LSTM"], LSTM_FEATURE_COLS, LSTM_WINDOW, scaler_lstm_pour_ns,
            )
            avert_scaler_mlp_ns_signale = [False]

            def _alpha_mlp_ns(t, ligne, soc_eb, soc_pb, alpha_prev):
                accel = float(ligne["hasAcceleration"]) if "hasAcceleration" in df.columns else 0.0
                p_dem = float(ligne["hasPower"])

                alpha_fuzzy_val = float(
                    alpha_fuzzy_calc(
                        np.array([soc_eb]), np.array([soc_pb]),
                        np.array([p_dem]), np.array([accel]),
                    )["alpha"][0]
                )

                sortie_lstm = predire_lstm_pour_ns(t, soc_eb, soc_pb)
                etats = compute_symbolic_states(p_dem, soc_eb, soc_pb)

                valeurs = {
                    "SOC_EB": soc_eb,
                    "SOC_PB": soc_pb,
                    "hasPower": p_dem,
                    "speed": float(ligne["speed"]) if "speed" in df.columns else 0.0,
                    "hasAcceleration": accel,
                    "Pdem_pred_ns": float(sortie_lstm[0]),
                    "delta_soc_eb_pred_ns": float(sortie_lstm[1]),
                    "delta_soc_pb_pred_ns": float(sortie_lstm[2]),
                    "alpha_ems_fuzzy_logic": alpha_fuzzy_val,
                    **{cle: float(valeur) for cle, valeur in etats.items()},
                }

                brut = np.array([valeurs[c] for c in MLP_NS_INPUT_COLS], dtype=np.float64)
                if scaler_mlp_ns is not None:
                    try:
                        brut = appliquer_scaler(brut, scaler_mlp_ns)
                    except ValueError as exc:
                        if not avert_scaler_mlp_ns_signale[0]:
                            avertissements.append(f"EMS_MLP_neurosymbolic : {exc}")
                            avert_scaler_mlp_ns_signale[0] = True

                x = torch.tensor(
                    [brut.tolist()],
                    dtype=torch.float32, device=DEVICE,
                )
                af = torch.tensor([alpha_fuzzy_val], dtype=torch.float32, device=DEVICE)

                with torch.no_grad():
                    alpha, _, _ = model(x, af)

                return float(alpha.item())

            try:
                _t0 = time.time()
                resultats["EMS_MLP_neurosymbolic"] = simuler_strategie_deterministe(
                    df, soc_eb0, soc_pb0, _alpha_mlp_ns,
                )
                avertissements.append(f"[timing] EMS_MLP_neurosymbolic : {time.time() - _t0:.1f} s")
                avertissements.append(
                    "EMS_MLP_neurosymbolic simulé avec les 17 colonnes confirmées "
                    "(01_configuration.ipynb), y compris les 3 sorties d'EMS_LSTM "
                    "(recalculées en interne, ce qui double le coût de calcul du LSTM)."
                )
            except KeyError as exc:
                avertissements.append(
                    f"EMS_MLP_neurosymbolic non simulé : une colonne d'entrée est absente ({exc})."
                )


    # --------------------------------------------------------
    # EMS_GNN
    #
    # Simule en boucle fermee via construire_graphe_instant, desormais
    # CONFIRME (aretes, features par noeud et normalisation reproduisent
    # exactement 05_EMS_graph_construction.ipynb).
    # --------------------------------------------------------

    if "EMS_GNN" in modeles_charges:
        model = modeles_charges["EMS_GNN"]
        scaler_gnn = charger_scaler(GNN_SCALER_FILE)
        if scaler_gnn is None:
            avertissements.append(
                f"EMS_GNN : aucun fichier de normalisation trouvé à {GNN_SCALER_FILE} -- "
                "graphes non normalisés, résultats potentiellement peu fiables."
            )

        def _alpha_gnn(t, ligne, soc_eb, soc_pb, alpha_prev):
            accel = (
                float(ligne["hasAcceleration"])
                if "hasAcceleration" in df.columns
                else 0.0
            )
            x, edge_index = construire_graphe_instant(
                ligne["hasPower"], soc_eb, soc_pb, accel, scaler_gnn,
            )
            x, edge_index = x.to(DEVICE), edge_index.to(DEVICE)
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=DEVICE)
            with torch.no_grad():
                alpha = model(x, edge_index, batch)
            return float(np.clip(alpha.item(), 0.0, 1.0))

        try:
            _t0 = time.time()
            resultats["EMS_GNN"] = simuler_strategie_deterministe(df, soc_eb0, soc_pb0, _alpha_gnn)
            avertissements.append(f"[timing] EMS_GNN : {time.time() - _t0:.1f} s")
            avertissements.append(
                "EMS_GNN simulé en boucle fermée avec la construction de graphe "
                "confirmée (05_EMS_graph_construction.ipynb) : arêtes exactes "
                "(y compris la liaison directe EB-moteur), caractéristiques par "
                "nœud et normalisation via le scaler réel si présent."
            )
        except Exception as exc:
            avertissements.append(f"EMS_GNN non simulé : {exc}")

    return (
        resultats,
        avertissements,
    )