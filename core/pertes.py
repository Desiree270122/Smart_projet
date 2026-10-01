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
- convertisseur : (1 − η(P_conv))·|P_conv|, η suivant la courbe mesurée (fig. 33).

Par défaut, les trajectoires sont simulées SANS pertes, comme dans l'article :
ce bilan est alors calculé après coup. L'option « inclure les pertes dans le
SOC » de la page « Lancer une simulation » les intègre à la simulation elle-même
(ems_core.set_pertes), avec le même modèle.
"""

import numpy as np

import ems_core as core


# Rendement du convertisseur PSFB mesuré à puissance nominale ([1], fig. 33).
# Les pertes utilisent la courbe mesurée complète (ems_core.rendement_convertisseur),
# plus basse à charge partielle : environ 91,5 % à 1,2 kW.
RENDEMENT_CONVERTISSEUR = 0.955


def resistances_packs():
    """Résistances internes des packs (Ω), à partir des données par cellule."""
    return core.R_EB_PACK_OHM, core.R_PB_PACK_OHM


def pertes_par_pas(traj):
    """Puissances perdues (W) à chaque pas : batterie Énergie, batterie Puissance,
    convertisseur (même modèle que les pertes incluses dans le SOC)."""
    eb, pb, conv = core.pertes_instantanees(traj["P_EB"], traj["P_PB"])
    return {"eb": eb, "pb": pb, "convertisseur": conv}


def bilan_pertes(traj):
    """Pertes estimées (Wh) d'une trajectoire : EB, PB, convertisseur, total,
    et leur part dans l'énergie de traction fournie au bus."""
    pas = pertes_par_pas(traj)
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
