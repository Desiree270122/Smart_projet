"""
api/simulation.py — Préparer un cycle de conduite et lancer une simulation.

Préparer : lire un fichier (CSV, TXT, TSV, Excel), choisir ses colonnes, calculer
la puissance demandée si elle manque, régler les batteries et le convertisseur.
Lancer : simuler les stratégies choisies sur un cycle, en tâche de fond ; le
résultat devient un cycle analysable comme les cycles de référence.
"""

import io
import math
import re
import threading
import time
from functools import lru_cache

import numpy as np
import pandas as pd
import torch

import ems_core as core
from api import donnees
from api.donnees import liste, nombre_json
from core import ontology_explainer as ox
from core.i18n import tr
from core.resultats import CYCLES_REFERENCE, libelle_cycle

REFERENCES = ["EMS_power_limitation", "EMS_fuzzy_logic"]   # toujours simulées
# Précision du calcul : pas de recherche de la répartition par le filtre de sécurité, facteur de durée.
PRECISIONS = {"rapide": (0.005, 1.0), "normale": (0.002, 1.5), "fine": (0.001, 2.5)}
# Temps de calcul relatif des stratégies à apprentissage.
COUT_CALCUL = {
    "EMS_MLP": ("rapide", 1.0), "EMS_MLP_neurosymbolic": ("moyen", 1.5), "EMS_LSTM": ("moyen", 1.5),
    "EMS_LSTM_neurosymbolic": ("moyen", 1.5), "EMS_GNN": ("long", 3.0),
}
COLONNES_EXPORT = [
    "time", "speed", "hasAcceleration", "hasAeroForce", "hasRollingForce",
    "hasGravityForce", "hasAccelerationForce", "hasTotalForce", "hasPower",
]


# Fichier du cycle

def _lire(contenu: bytes, nom: str, sans_entete: bool, feuille):
    entete = None if sans_entete else 0
    if nom.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(contenu), sheet_name=feuille or 0, header=entete)
    return pd.read_csv(io.BytesIO(contenu), sep=None, engine="python", header=entete)


def _tableau(fichier, sans_entete, feuille):
    df = _lire(fichier["contenu"], fichier["nom"], sans_entete, feuille)
    if sans_entete:
        df.columns = ["speed"] if df.shape[1] == 1 else [f"col_{i}" for i in range(df.shape[1])]
    else:
        df.columns = [str(c).strip() for c in df.columns]
    return df


def _vitesse(df, colonne):
    valeurs = pd.to_numeric(df[colonne], errors="coerce").to_numpy(dtype=float)
    return None if pd.isna(valeurs).all() else valeurs


def analyser_fichier(nom: str, contenu: bytes, sans_entete: bool = False, feuille=None, identifiant=None) -> dict:
    """Lit le fichier importé et propose les colonnes à utiliser."""
    feuilles = None
    if nom.lower().endswith((".xlsx", ".xls")):
        feuilles = pd.ExcelFile(io.BytesIO(contenu)).sheet_names
        feuille = feuille if feuille in feuilles else feuilles[0]
    fichier = {"nom": nom, "contenu": contenu}
    df = _tableau(fichier, sans_entete, feuille)
    identifiant = identifiant or donnees.nouvel_identifiant()
    donnees.enregistrer(donnees.fichiers, identifiant, fichier)

    colonnes = list(df.columns)
    devine = {c: core.guess_column(colonnes, c) for c in ("time", "speed", "power", "acceleration")}
    devine = {c: (v if v in colonnes else None) for c, v in devine.items()}
    return {
        "id": identifiant, "nom": nom, "nb_lignes": int(df.shape[0]), "colonnes": colonnes,
        "feuilles": feuilles, "feuille": feuille, "sans_entete": sans_entete,
        "colonnes_sans_nom": (not sans_entete) and all(str(c).isdigit() for c in colonnes),
        "apercu": df.head(20).astype(object).where(pd.notna(df.head(20)), None).to_dict("records"),
        "devine": devine,
        "vitesse": analyser_vitesse(identifiant, devine["speed"], sans_entete, feuille) if devine["speed"] else None,
    }


def analyser_vitesse(id_fichier: str, colonne: str, sans_entete: bool = False, feuille=None, unite=None) -> dict:
    """Unité probable de la colonne de vitesse, et nombre de répétitions du motif."""
    fichier = donnees.fichiers.get(id_fichier)
    if fichier is None:
        raise donnees.Introuvable(id_fichier)
    valeurs = _vitesse(_tableau(fichier, sans_entete, feuille), colonne)
    if valeurs is None:
        return {"numerique": False}
    unite_detectee = core.detect_speed_unit(valeurs)
    return {
        "numerique": True, "unite": unite_detectee, "maximum": nombre_json(pd.Series(valeurs).abs().max(), 2),
        "repetitions": int(core.detect_repetition(core.convert_speed_to_ms(valeurs, unite or unite_detectee))),
    }


def vehicule_par_defaut() -> dict:
    return {
        "masse": core.VEHICLE_MASS_KG, "cx": core.DRAG_COEFFICIENT_CX, "surface": core.FRONTAL_AREA_M2,
        "c0": core.ROLLING_C0, "c1": core.ROLLING_C1, "pente_deg": 0.0, "rho": core.AIR_DENSITY_KG_M3,
    }


def preparer_cycle(p: dict) -> dict:
    """Construit le cycle de conduite à partir du fichier et des réglages choisis."""
    fichier = donnees.fichiers.get(p["id_fichier"])
    if fichier is None:
        raise donnees.Introuvable(p["id_fichier"])
    brut = _tableau(fichier, bool(p.get("sans_entete")), p.get("feuille"))
    col_vitesse, col_puissance = p.get("col_vitesse"), p.get("col_puissance")
    col_acceleration, col_temps = p.get("col_acceleration"), p.get("col_temps")
    if not col_vitesse and not col_puissance:
        raise ValueError(tr("Sélectionnez au moins une colonne de vitesse ou une colonne de puissance.",
                            "Select at least a speed column or a power column."))
    if col_vitesse and _vitesse(brut, col_vitesse) is None:
        raise ValueError(tr("La colonne de vitesse « {c} » ne contient pas de nombres.",
                            "The speed column “{c}” contains no numbers.", c=col_vitesse))

    # Temps et vitesse du cycle de base, avant répétition
    if col_temps:
        temps_brut = pd.to_numeric(brut[col_temps], errors="coerce").to_numpy(dtype=float)
        pas = float(np.median(np.diff(temps_brut))) if len(temps_brut) > 1 else 1.0
        if pas <= 0 or np.isnan(pas):
            pas = 1.0
    else:
        pas = 1.0 / float(p.get("frequence_hz") or 1.0)
    temps_base = np.arange(len(brut)) * pas
    base = pd.DataFrame()

    if col_vitesse:
        base["speed"] = core.convert_speed_to_ms(_vitesse(brut, col_vitesse), p.get("unite_vitesse") or "km/h")
    if col_puissance:
        base["hasPower"] = pd.to_numeric(brut[col_puissance], errors="coerce").to_numpy(dtype=float)
        if col_acceleration:
            base["hasAcceleration"] = pd.to_numeric(brut[col_acceleration], errors="coerce").to_numpy(dtype=float)
        elif col_vitesse:
            dv, dt = np.diff(base["speed"].to_numpy()), np.diff(temps_base)
            base["hasAcceleration"] = np.concatenate(([0.0], np.divide(dv, dt, out=np.zeros_like(dv), where=dt != 0)))
        else:
            base["hasAcceleration"] = 0.0
    else:
        v = {**vehicule_par_defaut(), **(p.get("vehicule") or {})}
        forces = core.compute_forces_and_power(
            base["speed"].to_numpy(), temps_base, mass=v["masse"], cx=v["cx"], frontal_area=v["surface"],
            c0=v["c0"], c1=v["c1"], slope_rad=math.radians(v["pente_deg"]), rho=v["rho"], gravity=core.GRAVITY_MS2,
        )
        for colonne, valeurs in forces.items():
            base[colonne] = valeurs

    repetitions = max(1, int(p.get("repetitions") or 1))
    cycle = pd.concat([base.copy() for _ in range(repetitions)], ignore_index=True) if repetitions > 1 else base.copy()
    cycle.insert(0, "time", np.arange(len(cycle), dtype=float) * pas)

    identifiant = donnees.PREFIXE_PREPARE + donnees.nouvel_identifiant()
    meta = {"nb_points": len(cycle), "repetitions": repetitions, "puissance_calculee": not col_puissance}
    donnees.enregistrer(donnees.cycles_prepares, identifiant, {"df": cycle, "meta": meta})
    puissance = cycle["hasPower"]
    return {
        "id": identifiant, **meta,
        "puissance_presque_nulle": bool(abs(puissance.mean()) < 50 and puissance.max() < 200),
        "apercu": {
            "temps_min": liste(cycle["time"].to_numpy(dtype=float) / 60.0, 4),
            "vitesse_kmh": liste(cycle["speed"].to_numpy(dtype=float) * 3.6, 2) if "speed" in cycle.columns else None,
            "acceleration": liste(cycle["hasAcceleration"].to_numpy(dtype=float), 3) if "speed" not in cycle.columns else None,
            "puissance_kw": liste(puissance.to_numpy(dtype=float) / 1000.0),
        },
    }


def exporter_cycle(id_prepare: str, format_fichier: str):
    """(contenu, type, nom de fichier) du cycle préparé, en CSV ou en Excel."""
    prepare = donnees.cycles_prepares.get(id_prepare)
    if prepare is None:
        raise donnees.Introuvable(id_prepare)
    df = prepare["df"]
    colonnes = [c for c in COLONNES_EXPORT if c in df.columns]
    colonnes += [c for c in df.columns if c not in colonnes]
    if format_fichier == "xlsx":
        tampon = io.BytesIO()
        with pd.ExcelWriter(tampon, engine="xlsxwriter") as ecrivain:
            df[colonnes].to_excel(ecrivain, index=False, sheet_name="Cycle")
        return tampon.getvalue(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "cycle_prepare.xlsx"
    return df[colonnes].to_csv(index=False, sep=";").encode("utf-8-sig"), "text/csv", "cycle_prepare.csv"


# Batteries et convertisseur

def materiel_par_defaut() -> dict:
    """Cellules, architecture des packs et convertisseur du projet."""
    def cellule(prefixe):
        return {
            "n_serie": getattr(core, f"CELL_{prefixe}_N_SERIE"), "n_parallele": getattr(core, f"CELL_{prefixe}_N_PARALLELE"),
            "masse_cellule": getattr(core, f"CELL_{prefixe}_MASSE_KG"), "v_cellule": getattr(core, f"CELL_{prefixe}_V_CELLULE"),
            "i_decharge": getattr(core, f"CELL_{prefixe}_I_DECHARGE_A"), "i_recharge": getattr(core, f"CELL_{prefixe}_I_RECHARGE_A"),
            "de": getattr(core, f"CELL_{prefixe}_DE_WH_KG"), "capacite": getattr(core, f"CELL_{prefixe}_CAPACITE_AH"),
            "rint": getattr(core, f"CELL_{prefixe}_RINT_OHM"),
        }

    return {
        "eb": cellule("EB"), "pb": cellule("PB"),
        "convertisseur": {
            "n_modules": core.CONVERTER_N_COMPOSANTS, "p_decharge": core.CONVERTER_P_DECHARGE_PAR_COMPOSANT_W,
            "p_recharge": core.CONVERTER_P_RECHARGE_PAR_COMPOSANT_W,
        },
    }


def _pack(c):
    return core.compute_pack_characteristics(
        c["v_cellule"], c["i_decharge"], c["i_recharge"], c["masse_cellule"], c["de"], int(c["n_serie"]), int(c["n_parallele"]),
        capacite_cellule_ah=c["capacite"],
    )


def materiel_du_moteur(materiel):
    """Réglages du moteur de simulation pour un matériel donné ; None s'il est celui par défaut
    (le moteur garde alors ses valeurs d'origine)."""
    defaut = materiel_par_defaut()
    if not materiel:
        return None
    complet = {
        "eb": {**defaut["eb"], **(materiel.get("eb") or {})}, "pb": {**defaut["pb"], **(materiel.get("pb") or {})},
        "convertisseur": {**defaut["convertisseur"], **(materiel.get("convertisseur") or {})},
    }
    if all(math.isclose(float(complet[k][c]), float(defaut[k][c]), rel_tol=1e-9) for k in defaut for c in defaut[k]):
        return None
    conv = core.compute_converter_characteristics(
        int(complet["convertisseur"]["n_modules"]), complet["convertisseur"]["p_decharge"], complet["convertisseur"]["p_recharge"],
    )
    return {"eb": _pack(complet["eb"]), "pb": _pack(complet["pb"]), "convertisseur": conv, "cellules": complet}


def caracteristiques_materiel(materiel=None) -> dict:
    """Caractéristiques des deux packs et du convertisseur, calculées à partir des cellules."""
    defaut = materiel_par_defaut()
    materiel = materiel or {}
    cellules = {
        "eb": {**defaut["eb"], **(materiel.get("eb") or {})}, "pb": {**defaut["pb"], **(materiel.get("pb") or {})},
        "convertisseur": {**defaut["convertisseur"], **(materiel.get("convertisseur") or {})},
    }
    eb, pb = _pack(cellules["eb"]), _pack(cellules["pb"])
    conv = core.compute_converter_characteristics(
        int(cellules["convertisseur"]["n_modules"]), cellules["convertisseur"]["p_decharge"], cellules["convertisseur"]["p_recharge"],
    )

    def ligne(cle, c):
        return {"cle": cle, "tension_v": c["tension_V"], "masse_kg": c["masse_kg"], "capacite_ah": c.get("capacite_Ah"),
                "p_decharge_w": c["puissance_decharge_W"], "p_recharge_w": c["puissance_recharge_W"], "energie_wh": c["energie_Wh"]}

    return {
        "cellules": cellules,
        "packs": [
            ligne("eb", eb), ligne("pb", pb),
            {"cle": "total", "tension_v": pb["tension_V"], "masse_kg": eb["masse_kg"] + pb["masse_kg"], "capacite_ah": None,
             "p_decharge_w": eb["puissance_decharge_W"] + pb["puissance_decharge_W"],
             "p_recharge_w": eb["puissance_recharge_W"] + pb["puissance_recharge_W"], "energie_wh": eb["energie_Wh"] + pb["energie_Wh"]},
        ],
        "nb_cellules": {
            "eb": int(cellules["eb"]["n_serie"]) * int(cellules["eb"]["n_parallele"]),
            "pb": int(cellules["pb"]["n_serie"]) * int(cellules["pb"]["n_parallele"]),
        },
        "convertisseur": {"p_decharge_w": conv["p_decharge_W"], "p_recharge_w": conv["p_recharge_W"]},
    }


def apercu_modele_physique(identifiant: str, soc_eb0: float, soc_pb0: float, materiel=None) -> dict:
    """Simulation rapide du cycle avec le modèle physique (aucun apprentissage nécessaire)."""
    df, _ = donnees.cycle_a_simuler(identifiant)
    with donnees.moteur(materiel_du_moteur(materiel)):
        traj = core.simuler_strategie_deterministe(
            df, soc_eb0, soc_pb0,
            lambda t, ligne, soc_eb, soc_pb, alpha_prev: core.eb_priority_alpha_single(ligne["hasPower"], soc_eb),
        )
        n = len(df)
        p_conv = (core.V_EB_PACK_NOM - core.V_PB_PACK_NOM) * np.asarray(traj["I_EB"], dtype=float)[:n]
        return {
            "temps_min": liste(df["time"].to_numpy(dtype=float) / 60.0, 4),
            "soc_eb_pct": liste(np.asarray(traj["SOC_EB"], float)[:n] * 100, 2),
            "soc_pb_pct": liste(np.asarray(traj["SOC_PB"], float)[:n] * 100, 2),
            "eb_kw": liste(np.asarray(traj["P_EB"], float)[:n] / 1000.0), "pb_kw": liste(np.asarray(traj["P_PB"], float)[:n] / 1000.0),
            "eb_a": liste(np.asarray(traj["I_EB"], float)[:n], 2), "pb_a": liste(np.asarray(traj["I_PB"], float)[:n], 2),
            "conv_kw": liste(p_conv / 1000.0),
            "limites": {"p_eb_max_kw": core.P_EB_MAX_W / 1000.0, "p_conv_max_kw": core.P_CONV_MAX_W / 1000.0,
                        "p_conv_min_kw": core.P_CONV_MIN_W / 1000.0, "soc_min_pct": core.SOC_EB_MIN * 100},
        }


# Lancer une simulation

def cycles_simulables() -> list:
    """Cycles sur lesquels une simulation peut être lancée : les cycles de référence."""
    return [{"id": c, "nom": libelle_cycle(c)} for c in CYCLES_REFERENCE if CYCLES_REFERENCE[c][1].exists()]


def resume_cycle(identifiant: str, materiel=None) -> dict:
    """Ce que demande un cycle, comparé à l'énergie disponible dans le HESS."""
    df, _ = donnees.cycle_a_simuler(identifiant)
    n = len(df)
    p = df["hasPower"].to_numpy(dtype=float) if "hasPower" in df.columns else np.zeros(n)
    duree_s = float(df["time"].iloc[-1]) if "time" in df.columns and n else n * core.DT_SECONDS
    with donnees.moteur(materiel_du_moteur(materiel)):
        energie_utile = ((1 - core.SOC_EB_MIN) * core.ENERGY_EB_WH + (1 - core.SOC_PB_MIN) * core.ENERGY_PB_WH) / 1000.0
    energie_nette = float(p.sum() * core.DT_SECONDS / 3.6e6)
    return {
        "id": identifiant,
        "nom": libelle_cycle(identifiant) if identifiant in CYCLES_REFERENCE else tr("Le cycle que vous avez préparé", "The cycle you prepared"),
        "duree_min": duree_s / 60.0,
        "distance_km": nombre_json(df["speed"].sum() * core.DT_SECONDS / 1000.0, 2) if "speed" in df.columns else None,
        "demande_max_kw": float(p.max(initial=0.0) / 1000.0), "freinage_max_kw": float(-p.min(initial=0.0) / 1000.0),
        "energie_nette_kwh": energie_nette, "energie_utile_kwh": energie_utile, "trop_exigeant": energie_nette > energie_utile,
    }


def diagnostic(soc_eb0: float, soc_pb0: float, nb_strategies: int) -> dict:
    return ox.diagnostic_configuration(soc_eb0, soc_pb0, nb_strategies)


def options_de_simulation() -> dict:
    """Ce que propose la page « Lancer une simulation »."""
    from api.analyse import fiche_strategie
    from core.resultats import FAMILLES
    from core.i18n import lib

    return {
        "familles": [
            {"nom": lib(nom), "strategies": [
                {**fiche_strategie(c), "reference": c in REFERENCES, "poids": COUT_CALCUL.get(c, ("", 0.0))[1],
                 "temps": COUT_CALCUL.get(c, (None, 0.0))[0]}
                for c in membres
            ]}
            for nom, membres in FAMILLES.items()
        ],
        "precisions": [{"cle": k, "pas": v[0], "facteur": v[1]} for k, v in PRECISIONS.items()],
        "hypotheses": ox.hypotheses(),
        "pertes": {
            "r_eb_mohm": core.R_EB_PACK_OHM_DEFAUT * 1000, "r_pb_mohm": core.R_PB_PACK_OHM_DEFAUT * 1000,
            "r_cellule_eb_mohm": core.CELL_EB_RINT_OHM * 1000, "eb_serie": core.CELL_EB_N_SERIE, "eb_parallele": core.CELL_EB_N_PARALLELE,
            "r_cellule_pb_mohm": core.CELL_PB_RINT_OHM * 1000, "pb_serie": core.CELL_PB_N_SERIE, "pb_parallele": core.CELL_PB_N_PARALLELE,
        },
        "cycles": cycles_simulables(),
    }


@lru_cache(maxsize=None)
def _modele(cle):
    chargeurs = {
        "EMS_MLP": core.load_mlp_simple, "EMS_MLP_neurosymbolic": core.load_mlp_neurosymbolic,
        "EMS_LSTM": core.load_lstm_seul, "EMS_LSTM_neurosymbolic": core.load_lstm_neurosymbolic,
    }
    if cle == "EMS_GNN":
        modele, _ = core.load_gnn_simple()
    else:
        modele = chargeurs[cle]()
    if hasattr(modele, "eval"):
        modele.eval()
    return modele


def _simuler(identifiant, df, strategies, soc_eb0, soc_pb0, pas_alpha, pertes, materiel, meta):
    simulation = donnees.simulations[identifiant]
    debut = time.time()
    try:
        modeles, erreurs = {}, []
        for cle in strategies:
            try:
                modeles[cle] = _modele(cle)
            except Exception:  # noqa: BLE001 - une stratégie impossible à charger n'empêche pas les autres
                erreurs.append(cle)
        with donnees.moteur(materiel):
            core.set_alpha_grid_step(pas_alpha)
            core.set_pertes(*pertes)
            with torch.inference_mode():
                resultats, messages = core.simuler_toutes_strategies(df, soc_eb0, soc_pb0, modeles)
        durees = {}
        for message in messages:
            m = re.match(r"\[timing\]\s*(\S+)\s*:\s*([\d.]+)", message)
            if m:
                durees[m.group(1)] = float(m.group(2))
        simulation.update({
            "etat": "terminee", "erreurs": erreurs, "durees": durees, "duree_s": time.time() - debut,
            "donnees": {"resultats": resultats, "cycle_df": df, "coherence": {}, "meta": meta, "materiel": materiel},
        })
    except Exception as exc:  # noqa: BLE001 - l'échec est rapporté à l'interface
        simulation.update({"etat": "echec", "message": str(exc), "duree_s": time.time() - debut})


def lancer(p: dict) -> dict:
    """Lance une simulation en tâche de fond ; retourne son identifiant."""
    df, _ = donnees.cycle_a_simuler(p["cycle"])
    strategies = [c for c in (p.get("strategies") or []) if c in COUT_CALCUL]
    if "EMS_MLP_neurosymbolic" in strategies and "EMS_LSTM_neurosymbolic" not in strategies:
        strategies.append("EMS_LSTM_neurosymbolic")   # NS-MLP utilise les prévisions de NS-LSTM
    soc_eb0 = float(np.clip(p.get("soc_eb0", 1.0), 0.2, 1.0))
    soc_pb0 = float(np.clip(p.get("soc_pb0", 1.0), 0.2, 1.0))
    if "SOC_EB" in df.columns and len(df):
        df.loc[df.index[0], "SOC_EB"] = soc_eb0
    if "SOC_PB" in df.columns and len(df):
        df.loc[df.index[0], "SOC_PB"] = soc_pb0
    pas_alpha = PRECISIONS.get(p.get("precision"), PRECISIONS["normale"])[0]
    reglage = p.get("pertes") or {}
    avec_pertes = bool(reglage.get("activer"))
    pertes = (
        avec_pertes, reglage.get("rendement_constant") if avec_pertes else None,
        float(reglage["r_eb_mohm"]) / 1000.0 if avec_pertes and reglage.get("r_eb_mohm") else None,
        float(reglage["r_pb_mohm"]) / 1000.0 if avec_pertes and reglage.get("r_pb_mohm") else None,
    )
    meta = {"cycle_origine": p["cycle"], "soc_eb0": soc_eb0, "soc_pb0": soc_pb0, "pas_alpha": pas_alpha,
            "pertes_dans_soc": avec_pertes}

    identifiant = donnees.PREFIXE_SIMULATION + donnees.nouvel_identifiant()
    donnees.enregistrer(donnees.simulations, identifiant, {"etat": "en_cours", "debut": time.time(), "nb_strategies": len(strategies) + len(REFERENCES)})
    threading.Thread(
        target=_simuler, daemon=True,
        args=(identifiant, df, strategies, soc_eb0, soc_pb0, pas_alpha, pertes, materiel_du_moteur(p.get("materiel")), meta),
    ).start()
    return {"id": identifiant, "etat": "en_cours"}


def etat_simulation(identifiant: str) -> dict:
    from core.resultats import nom_affichage

    simulation = donnees.simulations.get(identifiant)
    if simulation is None:
        raise donnees.Introuvable(identifiant)
    sortie = {"id": identifiant, "etat": simulation["etat"], "nb_strategies": simulation.get("nb_strategies")}
    if simulation["etat"] == "en_cours":
        sortie["ecoule_s"] = time.time() - simulation["debut"]
    elif simulation["etat"] == "terminee":
        meta = simulation["donnees"]["meta"]
        sortie.update({
            "duree_s": simulation["duree_s"], "nb_strategies": len(simulation["donnees"]["resultats"]),
            "cycle": donnees.nom_du_cycle(meta["cycle_origine"]) if meta["cycle_origine"] in CYCLES_REFERENCE
            else tr("Le cycle que vous avez préparé", "The cycle you prepared"),
            "pertes_dans_soc": meta["pertes_dans_soc"],
            "non_chargees": [nom_affichage(c) for c in simulation["erreurs"]],
            "durees": [{"nom": nom_affichage(c), "duree_s": d} for c, d in sorted(simulation["durees"].items(), key=lambda kv: -kv[1])],
        })
    else:
        sortie["message"] = simulation.get("message")
    return sortie
