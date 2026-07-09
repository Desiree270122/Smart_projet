
import argparse
import json
import sys
import time
from pathlib import Path

# Rendre `import ems_core` possible quel que soit le dossier d'exécution :
# ce script est dans code/scripts/, ems_core.py est dans code/.
RACINE_CODE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE_CODE))

import numpy as np
import pandas as pd
import joblib

import ems_core as core


# SOC initiaux imposés à 100 %, identiques à la page 3 de l'application.
SOC_EB0 = 1.0
SOC_PB0 = 1.0

# Cycle de référence par défaut : le cycle Artemis déjà prétraité
# (contient speed, hasPower, hasAcceleration, hasTotalForce, SOC_EB, SOC_PB, I_EB).
CYCLE_DEFAUT = "data/processed/artemis_clean.csv"


def charger_cycle(chemin_cycle: Path) -> pd.DataFrame:
    """Charge le cycle de référence et impose les SOC initiaux exactement
    comme le fait l'application (première ligne SOC_EB/SOC_PB = 1.0)."""
    if not chemin_cycle.exists():
        raise FileNotFoundError(
            f"Cycle de référence introuvable : {chemin_cycle}. "
            "Indique un fichier existant via --cycle."
        )

    df = pd.read_csv(chemin_cycle)

    if "SOC_EB" in df.columns and len(df) > 0:
        df.loc[df.index[0], "SOC_EB"] = SOC_EB0
    if "SOC_PB" in df.columns and len(df) > 0:
        df.loc[df.index[0], "SOC_PB"] = SOC_PB0

    return df


def charger_modeles(avec_gnn: bool):
    """Charge les modèles entraînés via ems_core, en isolant les erreurs.
    Le GNN reste optionnel (torch_geometric peut être absent)."""
    modeles = {}
    erreurs = {}

    chargeurs = {
        "EMS_MLP": core.load_mlp_simple,
        "EMS_MLP_neurosymbolic": core.load_mlp_neurosymbolic,
        "EMS_LSTM": core.load_lstm_seul,
        "EMS_LSTM_neurosymbolic": core.load_lstm_neurosymbolic,
    }

    for nom, chargeur in chargeurs.items():
        try:
            modele = chargeur()
            if hasattr(modele, "eval"):
                modele.eval()
            modeles[nom] = modele
        except Exception as exc:  # noqa: BLE001 - on veut continuer malgré une erreur
            erreurs[nom] = str(exc)

    if avec_gnn:
        try:
            gnn_model, _gnn_scaler = core.load_gnn_simple()
            if hasattr(gnn_model, "eval"):
                gnn_model.eval()
            modeles["EMS_GNN"] = gnn_model
        except Exception as exc:  # noqa: BLE001
            erreurs["EMS_GNN"] = str(exc)

    return modeles, erreurs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cycle",
        default=CYCLE_DEFAUT,
        help=f"Chemin du cycle de référence (défaut : {CYCLE_DEFAUT}).",
    )
    parser.add_argument(
        "--pas-alpha",
        type=float,
        default=0.001,
        help="Résolution de la grille alpha (défaut : 0.001, qualité maximale hors-ligne).",
    )
    parser.add_argument(
        "--avec-gnn",
        action="store_true",
        help="Inclure EMS_GNN (nécessite torch_geometric).",
    )
    parser.add_argument(
        "--sortie",
        default=None,
        help="Dossier de sortie (défaut : results/precomputed/).",
    )
    args = parser.parse_args()

    chemin_cycle = (RACINE_CODE / args.cycle).resolve()
    dossier_sortie = (
        Path(args.sortie).resolve()
        if args.sortie
        else core.RESULTS_DIR / "precomputed"
    )
    dossier_sortie.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Résolution de la grille alpha : pas = {args.pas_alpha}")
    core.set_alpha_grid_step(args.pas_alpha)

    print(f"[2/4] Chargement du cycle : {chemin_cycle}")
    df = charger_cycle(chemin_cycle)
    print(f"      -> {len(df)} échantillons, colonnes : {list(df.columns)[:8]}...")

    print("[3/4] Chargement des modèles")
    modeles, erreurs = charger_modeles(args.avec_gnn)
    print(f"      -> chargés : {sorted(modeles.keys())}")
    if erreurs:
        for nom, err in erreurs.items():
            print(f"      -> NON chargé {nom} : {err}")

    print("[4/4] Simulation de toutes les stratégies (peut prendre plusieurs minutes)")
    debut = time.time()
    resultats, avertissements = core.simuler_toutes_strategies(
        df, SOC_EB0, SOC_PB0, modeles
    )
    duree = time.time() - debut
    print(f"      -> {len(resultats)} stratégies simulées en {duree:.1f} s")

    # Cycle COMPLET conservé dans le fichier, pour que le résultat soit
    # AUTO-SUFFISANT : les pages n'ont plus besoin du CSV d'origine (dont le
    # chemin absolu n'existe pas sur Streamlit). On garde tout le DataFrame
    # (toutes colonnes : hasPower, forces, speed, ...) afin que le « pont »
    # vers session_state["cycle_pret"] fournisse exactement ce que les pages
    # existantes attendent.
    cycle_df = df.reset_index(drop=True)

    # Sauvegarde principale : trajectoires + cycle complet + avertissements + méta.
    meta = {
        "cycle_nom": chemin_cycle.name,
        "nb_points": int(len(df)),
        "soc_eb0": SOC_EB0,
        "soc_pb0": SOC_PB0,
        "pas_alpha": args.pas_alpha,
        "strategies": sorted(resultats.keys()),
        "duree_simulation_s": duree,
    }
    chemin_joblib = dossier_sortie / "simulation_reference.joblib"
    joblib.dump(
        {
            "resultats": resultats,
            "cycle_df": cycle_df,
            "avertissements": avertissements,
            "meta": meta,
        },
        chemin_joblib,
    )
    print(f"      -> sauvegardé : {chemin_joblib}")

    # Résumé lisible : coût moyen par stratégie (contrôle rapide de cohérence).
    resume = {}
    for nom, traj in resultats.items():
        cout = traj.get("cost")
        if cout is not None:
            resume[nom] = float(np.nanmean(np.asarray(cout, dtype=float)))
    chemin_resume = dossier_sortie / "resume_couts.json"
    with open(chemin_resume, "w", encoding="utf-8") as f:
        json.dump(resume, f, indent=2, ensure_ascii=False)
    print(f"      -> résumé des coûts : {chemin_resume}")

    print("\nTerminé. L'application peut désormais charger ce résultat au lieu de recalculer.")


if __name__ == "__main__":
    main()
