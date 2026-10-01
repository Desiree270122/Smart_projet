"""
scripts/generer_cycles.py — Régénère les jeux de données des cycles de conduite
avec les paramètres de batteries d'ems_core (packs calculés à partir des cellules).

Pour chaque cycle : cinématique (vitesse, forces, puissance demandée), puis EMS
de référence de l'article (« power limitation », fig. 10), calculée au niveau
des courants comme dans le notebook Calcul_SOC.ipynb :
    I_EB = I_charge · V_PB / V_EB, saturé aux limites de l'EB, nul si SOC_EB ≤ 20 %
    I'   = I_EB · (V_EB − V_PB) / V_PB          (courant d'entrée du convertisseur)
    I_PB = I_charge − I_EB − I'                 (saturé aux limites de la PB)
    P_conv = I_EB · (V_EB − V_PB)              (éq. 8 de l'article)

Sorties :
- data/Artemis.csv : Artemis urbain + routier, 6 répétitions (12 456 s). Avec les
  packs calculés à partir des cellules, 7 répétitions ne sont pas réalisables
  (il manquerait environ 334 Wh, même sans pertes).
- data/wltc.csv : WLTC classe 3 (23,3 km), 4 répétitions, jamais vu à l'entraînement.

Usage : python scripts/generer_cycles.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

DOSSIER_PROJET = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DOSSIER_PROJET))

import ems_core as core  # noqa: E402

DONNEES = DOSSIER_PROJET / "data"
PERIODE_ARTEMIS_S = 2076          # une répétition urbain + routier
REPETITIONS_ARTEMIS = 6
REPETITIONS_WLTC = 4             # 2 556 Wh par cycle : 5 ne laisseraient que 200 Wh de marge
P_FREINAGE_MIN_W = -50000.0       # écrêtage de la puissance de freinage (article)

COLONNES = [
    "time", "speed", "hasAeroForce", "hasRollingForce", "hasGravityForce",
    "hasAcceleration", "hasAccelerationForce", "hasTotalForce", "hasPower",
    "P_EB", "P_PB", "P_conv", "I_EB", "I_PB", "I_load", "I_prime",
    "SOC_EB", "SOC_PB", "SOC_HESS",
]


def cinematique(vitesse, temps):
    """Forces et puissance demandée à partir de la vitesse (m/s), comme dans
    calcul_wltc.ipynb : accélération avant, véhicule de l'article."""
    v = np.asarray(vitesse, dtype=float)
    t = np.asarray(temps, dtype=float)
    dv, dt = np.diff(v), np.diff(t)
    acc = np.concatenate(([0.0], np.divide(dv, dt, out=np.zeros_like(dv), where=dt != 0)))
    m, g = core.VEHICLE_MASS_KG, core.GRAVITY_MS2
    df = pd.DataFrame({"time": t, "speed": v})
    df["hasAeroForce"] = (0.5 * core.AIR_DENSITY_KG_M3 * core.FRONTAL_AREA_M2 * core.DRAG_COEFFICIENT_CX * v**2).round(2)
    df["hasRollingForce"] = (m * g * (core.ROLLING_C0 + core.ROLLING_C1 * v**2)).round(2)
    df["hasGravityForce"] = round(float(m * g * np.sin(core.ROAD_SLOPE_RAD)), 2)
    df["hasAcceleration"] = acc.round(2)
    df["hasAccelerationForce"] = (m * acc).round(2)
    df["hasTotalForce"] = (
        df["hasAeroForce"] + df["hasRollingForce"] + df["hasGravityForce"] + df["hasAccelerationForce"]
    ).round(2)
    df["hasPower"] = (df["hasTotalForce"] * df["speed"]).round(2).clip(lower=P_FREINAGE_MIN_W)
    return df


def ems_reference(df):
    """EMS « power limitation » de l'article au niveau des courants (Calcul_SOC.ipynb)."""
    v_eb, v_pb = core.V_EB_PACK_NOM, core.V_PB_PACK_NOM
    e_eb_j, e_pb_j = core.ENERGY_EB_WH * 3600.0, core.ENERGY_PB_WH * 3600.0
    ibe_max, ibe_min = core.P_EB_MAX_W / v_eb, core.P_EB_MIN_W / v_eb
    ibp_max, ibp_min = core.P_PB_MAX_W / v_pb, core.P_PB_MIN_W / v_pb
    dt = core.DT_SECONDS

    p_load = df["hasPower"].to_numpy(dtype=float)
    i_charge = p_load / v_pb
    n = len(df)
    soc_eb, soc_pb = np.ones(n), np.ones(n)
    i_eb, i_pb, i_prime = np.zeros(n), np.zeros(n), np.zeros(n)

    for i in range(n):
        ibe = float(np.clip(i_charge[i] * v_pb / v_eb, ibe_min, ibe_max))
        if soc_eb[i] <= core.SOC_EB_MIN and i_charge[i] > 0:
            ibe = 0.0
        ip = ibe * (v_eb - v_pb) / v_pb
        ibp = float(np.clip(i_charge[i] - ibe - ip, ibp_min, ibp_max))
        i_eb[i], i_pb[i], i_prime[i] = ibe, ibp, ip
        if i < n - 1:
            soc_eb[i + 1] = np.clip(soc_eb[i] - ibe * v_eb * dt / e_eb_j, 0.0, 1.0)
            soc_pb[i + 1] = np.clip(soc_pb[i] - ibp * v_pb * dt / e_pb_j, 0.0, 1.0)

    sortie = df.copy()
    sortie["P_EB"] = np.round(i_eb * v_eb, 2)
    sortie["P_PB"] = np.round(i_pb * v_pb, 2)
    sortie["P_conv"] = np.round(i_eb * (v_eb - v_pb), 4)
    sortie["I_EB"] = np.round(i_eb, 4)
    sortie["I_PB"] = np.round(i_pb, 4)
    sortie["I_load"] = np.round(i_charge, 4)
    sortie["I_prime"] = np.round(i_prime, 4)
    sortie["SOC_EB"] = np.round(soc_eb, 6)
    sortie["SOC_PB"] = np.round(soc_pb, 6)
    sortie["SOC_HESS"] = np.round(
        (soc_eb * core.ENERGY_EB_WH + soc_pb * core.ENERGY_PB_WH) / (core.ENERGY_EB_WH + core.ENERGY_PB_WH), 6
    )
    return sortie[COLONNES]


def bilan(nom, df):
    p = df["hasPower"].to_numpy(dtype=float)
    print(
        f"{nom} : {len(df)} s, {np.sum(df['speed']) / 1000:.1f} km, énergie nette "
        f"{p.sum() / 3600:.0f} Wh ; SOC final EB {df['SOC_EB'].iloc[-1] * 100:.1f} %, "
        f"PB {df['SOC_PB'].iloc[-1] * 100:.1f} % ; SOC min EB {df['SOC_EB'].min() * 100:.1f} %, "
        f"PB {df['SOC_PB'].min() * 100:.1f} %"
    )


def main():
    # Artemis : la cinématique d'origine (colonnes jusqu'à hasPower) est conservée
    # telle quelle ; seules les répétitions et la partie batteries changent.
    source = DONNEES / "Artemis.csv"
    artemis = pd.read_csv(source)
    cine = artemis[COLONNES[:9]].iloc[: PERIODE_ARTEMIS_S * REPETITIONS_ARTEMIS].reset_index(drop=True)
    artemis_6 = ems_reference(cine)
    artemis_6.to_csv(source, index=False)
    bilan("Artemis x6", artemis_6)

    # WLTC : un cycle de 1 801 s (vitesse en m/s), répété REPETITIONS_WLTC fois.
    vitesse = pd.read_excel(DONNEES / "wltc.xlsx", sheet_name="Feuil1", header=None).iloc[:, 0].to_numpy(dtype=float)
    vitesse = np.tile(vitesse, REPETITIONS_WLTC)
    wltc = ems_reference(cinematique(vitesse, np.arange(len(vitesse), dtype=float)))
    wltc.to_csv(DONNEES / "wltc.csv", index=False)
    bilan(f"WLTC x{REPETITIONS_WLTC}", wltc)


if __name__ == "__main__":
    main()
