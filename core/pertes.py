"""
core/pertes.py — Bilan de pertes estimé du HESS, par stratégie.

Architecture « CCS cascade » de Fonseca de Freitas et al. (IEEE Access 2024,
fig. 8, éq. (9)–(16)) : la batterie Énergie (Source 1, V_EB > V_PB) débite le
courant de sortie du convertisseur, qui ne traite que la différence de
tension, P_conv = (V_EB − V_PB)·I_EB, soit environ 10,5 % de P_EB. La
batterie Puissance est directement sur le bus DC.

Pertes estimées à chaque pas de temps :
- batterie Énergie : R_EB·I_EB², avec R_EB = r_cellule × n_série / n_parallèle ;
- batterie Puissance : R_PB·I_PB² ;
- convertisseur : (1 − η)·|P_conv|.

Limite assumée, affichée à l'utilisateur : les trajectoires ont été simulées
SANS pertes. Ce bilan est calculé après coup ; il sert à comparer les
stratégies, pas à dire si le pack terminerait le cycle une fois les pertes
prises en compte.
"""

import numpy as np

import ems_core as core


# Rendement du convertisseur PSFB mesuré à puissance nominale ([1], fig. 33).
# Il est plus faible à charge partielle (environ 91,5 % à 1,2 kW).
RENDEMENT_CONVERTISSEUR = 0.955


def resistances_packs():
    """Résistances internes des packs (Ω), à partir des données par cellule."""
    r_eb = core.CELL_EB_RINT_OHM * core.CELL_EB_N_SERIE / core.CELL_EB_N_PARALLELE
    r_pb = core.CELL_PB_RINT_OHM * core.CELL_PB_N_SERIE / core.CELL_PB_N_PARALLELE
    return r_eb, r_pb


def pertes_par_pas(traj, rendement=RENDEMENT_CONVERTISSEUR):
    """Puissances perdues (W) à chaque pas : batterie Énergie, batterie Puissance,
    convertisseur."""
    r_eb, r_pb = resistances_packs()
    i_eb = np.asarray(traj["I_EB"], dtype=float)
    i_pb = np.asarray(traj["I_PB"], dtype=float)
    p_conv = (core.V_EB_PACK_NOM - core.V_PB_PACK_NOM) * i_eb
    return {
        "eb": r_eb * i_eb ** 2,
        "pb": r_pb * i_pb ** 2,
        "convertisseur": (1.0 - rendement) * np.abs(p_conv),
    }


def bilan_pertes(traj, rendement=RENDEMENT_CONVERTISSEUR):
    """Pertes estimées (Wh) d'une trajectoire : EB, PB, convertisseur, total,
    et leur part dans l'énergie de traction fournie au bus."""
    pas = pertes_par_pas(traj, rendement)
    h = core.DT_SECONDS / 3600.0

    eb = float(np.sum(pas["eb"])) * h
    pb = float(np.sum(pas["pb"])) * h
    conv = float(np.sum(pas["convertisseur"])) * h
    total = eb + pb + conv

    p_bus = np.asarray(traj["P_EB"], dtype=float) + np.asarray(traj["P_PB"], dtype=float)
    traction = float(np.sum(np.clip(p_bus, 0.0, None))) * h
    return {
        "eb_wh": eb,
        "pb_wh": pb,
        "convertisseur_wh": conv,
        "total_wh": total,
        "part_traction": total / traction if traction > 0 else float("nan"),
    }
