"""
core/resultats.py — Source UNIQUE de chargement et d'exploitation des
résultats de simulation précalculés.

Toutes les pages « précalculées » (Dashboard, Comparaison, Résultats & Analyse,
Explicabilité, Moteur Neuro-Symbolique) passent par ce module — et non chacune
à sa manière. Elles ne relancent jamais la simulation lourde : elles lisent le
fichier produit hors-ligne par `scripts/run_simulations.py`.

Le fichier de référence est AUTO-SUFFISANT : il contient les trajectoires de
chaque stratégie, les signaux du cycle (dont hasPower), et des métadonnées.
Aucune dépendance au CSV d'origine n'est donc nécessaire.
"""

from pathlib import Path

import numpy as np

import ems_core as core


# Emplacement standard du résultat précalculé.
FICHIER_REFERENCE = core.RESULTS_DIR / "precomputed" / "simulation_reference.joblib"


# Noms lisibles des stratégies (clés internes -> libellés d'affichage), alignés
# sur ceux de tes tables de résultats (comparaison_finale_6_strategies.csv).
NOMS_AFFICHAGE = {
    "EMS_power_limitation": "EB-priority",
    "EMS_fuzzy_logic": "Fuzzy-v1",
    "EMS_MLP": "MLP simple",
    "EMS_MLP_neurosymbolic": "MLP neurosymbolique",
    "EMS_LSTM": "LSTM",
    "EMS_LSTM_neurosymbolic": "LSTM neurosymbolique",
    "EMS_GNN": "GNN simple",
}


def nom_affichage(cle: str) -> str:
    """Libellé d'une stratégie. On affiche le code interne (EMS_MLP, EMS_LSTM,
    ...) tel quel, sauf la stratégie déterministe de référence, renommée
    « Modèle physique » car elle repose sur la physique et non sur de l'IA."""
    if cle == "EMS_power_limitation":
        return "Modèle physique"
    return cle


def charger_reference(chemin=None) -> dict:
    """Charge le fichier de simulation précalculé.

    Retourne un dictionnaire {resultats, cycle, avertissements, meta}.
    Lève FileNotFoundError explicite si le précalcul n'a pas encore été lancé.
    """
    import joblib

    chemin = Path(chemin) if chemin else FICHIER_REFERENCE
    if not chemin.exists():
        raise FileNotFoundError(
            f"Résultats précalculés introuvables : {chemin}. "
            "Lance d'abord le précalcul :\n"
            "    python scripts/run_simulations.py"
        )
    return joblib.load(chemin)


def assurer_donnees_session(st, chemin=None) -> str:
    """« Pont » entre les résultats précalculés et les pages existantes.

    Les pages Résultats/Analyse/Explicabilité/Moteur ont été écrites pour lire
    `st.session_state["resultats_simulation"]` et `["cycle_pret"]`, remplis
    autrefois par une simulation live (lente). Cette fonction les remplit
    depuis le fichier précalculé si — et seulement si — ils sont absents.

    Ainsi :
    - par défaut, les pages affichent les résultats de RÉFÉRENCE, instantanément ;
    - si l'utilisateur a lancé une simulation sur un cycle personnalisé (page
      dédiée), ces clés existent déjà et ont la priorité : rien n'est écrasé.

    Retourne la source utilisée ("référence précalculée" ou "déjà en session").
    """
    if "resultats_simulation" in st.session_state and "cycle_pret" in st.session_state:
        return st.session_state.get("_source_donnees", "déjà en session")

    donnees = charger_reference(chemin)
    st.session_state["resultats_simulation"] = donnees["resultats"]
    st.session_state["cycle_pret"] = donnees["cycle_df"]
    st.session_state["avertissements_simulation"] = donnees.get("avertissements", [])
    st.session_state["soc_eb0"] = donnees["meta"].get("soc_eb0", 1.0)
    st.session_state["soc_pb0"] = donnees["meta"].get("soc_pb0", 1.0)
    st.session_state["_source_donnees"] = "référence précalculée"
    return "référence précalculée"


def recalculer_cout_physique(traj: dict, p_dem) -> np.ndarray:
    """Recalcule, pas à pas, le VRAI coût physique multi-objectif (total_cost du
    filtre) de la décision prise par une stratégie, à partir de sa trajectoire.

    On rejoue `candidate_metrics` avec l'alpha réellement appliqué (alpha_final)
    et l'état SOC de CETTE stratégie à chaque instant. La règle de mise à jour de
    alpha_prev (inchangé sur demande quasi nulle) reproduit celle du moteur, pour
    que le terme de continuité soit cohérent.

    Retourne un tableau de longueur n (NaN sur les pas à demande quasi nulle, où
    aucune répartition n'est calculée).
    """
    alpha = np.asarray(traj["alpha_final"], dtype=float)
    soc_eb = np.asarray(traj["SOC_EB"], dtype=float)  # longueur n+1 (état avant chaque pas)
    soc_pb = np.asarray(traj["SOC_PB"], dtype=float)
    p = np.asarray(p_dem, dtype=float)
    n = len(p)

    couts = np.full(n, np.nan, dtype=float)
    alpha_prev = None

    for t in range(n):
        p_t = p[t]
        if abs(p_t) <= core.EPS_POWER_W:
            continue
        m = core.candidate_metrics(
            np.array([alpha[t]]), p_t, soc_eb[t], soc_pb[t], alpha_prev
        )
        couts[t] = float(m["total_cost"][0])
        alpha_prev = alpha[t]

    return couts


def _somme_wh(puissance) -> float:
    """Intègre une puissance (W, pas de 1 s) en énergie (Wh)."""
    return float(np.nansum(np.asarray(puissance, dtype=float))) * core.DT_SECONDS / 3600.0


def calculer_metriques(donnees: dict) -> dict:
    """Calcule, pour chaque stratégie, un jeu complet de métriques comparables,
    à partir des trajectoires et du cycle précalculés.

    Retourne {cle_strategie: {metrique: valeur, ...}}.
    """
    resultats = donnees["resultats"]
    cycle_df = donnees.get("cycle_df")
    p_dem = (
        cycle_df["hasPower"].to_numpy(dtype=float)
        if cycle_df is not None and "hasPower" in cycle_df.columns
        else None
    )

    metriques = {}
    for nom, traj in resultats.items():
        soc_eb = np.asarray(traj["SOC_EB"], dtype=float)
        soc_pb = np.asarray(traj["SOC_PB"], dtype=float)
        m = min(len(soc_eb), len(soc_pb))
        desequilibre = np.abs(soc_eb[:m] - soc_pb[:m])

        infos = {
            "cout_etendu_moyen": float(np.nanmean(np.asarray(traj["cost"], dtype=float))),
            "nb_violations": int(np.nansum(np.asarray(traj["soc_violation"], dtype=float))),
            "nb_corrections": int(np.nansum(np.asarray(traj["correction_applied"], dtype=float))),
            "taux_faisabilite": float(np.nanmean(np.asarray(traj["feasible"], dtype=float))),
            "soc_eb_final": float(traj.get("SOC_EB_final", soc_eb[-1])),
            "soc_pb_final": float(traj.get("SOC_PB_final", soc_pb[-1])),
            "desequilibre_soc_moyen": float(np.nanmean(desequilibre)),
            "energie_non_servie_wh": _somme_wh(traj["P_unserved"]),
            "regen_rejetee_wh": _somme_wh(traj["P_regen_curtailed"]),
        }

        # Vrai coût physique (si le cycle est disponible dans le fichier).
        if p_dem is not None:
            couts_phys = recalculer_cout_physique(traj, p_dem)
            infos["cout_physique_moyen"] = float(np.nanmean(couts_phys))
        else:
            infos["cout_physique_moyen"] = float("nan")

        metriques[nom] = infos

    return metriques


def statistiques_detaillees(donnees: dict) -> dict:
    """Statistiques physiques synthétiques par stratégie (pour la page
    Résultats & Analyse) : plutôt que 14 000 points bruts, on résume chaque
    série temporelle par des indicateurs lisibles par un physicien.

    Retourne {cle_strategie: {indicateur: valeur, ...}}.
    """
    resultats = donnees["resultats"]
    dt = core.DT_SECONDS

    stats = {}
    for nom, traj in resultats.items():
        p_eb = np.asarray(traj["P_EB"], dtype=float)
        p_pb = np.asarray(traj["P_PB"], dtype=float)
        i_eb = np.asarray(traj["I_EB"], dtype=float)
        i_pb = np.asarray(traj["I_PB"], dtype=float)
        soc_eb = np.asarray(traj["SOC_EB"], dtype=float)
        soc_pb = np.asarray(traj["SOC_PB"], dtype=float)

        stats[nom] = {
            "soc_eb_final": float(traj.get("SOC_EB_final", soc_eb[-1])),
            "soc_pb_final": float(traj.get("SOC_PB_final", soc_pb[-1])),
            # Énergie délivrée (décharge = puissance positive), en Wh.
            "energie_eb_wh": float(np.sum(np.clip(p_eb, 0.0, None)) * dt / 3600.0),
            "energie_pb_wh": float(np.sum(np.clip(p_pb, 0.0, None)) * dt / 3600.0),
            "i_eb_rms": float(np.sqrt(np.mean(i_eb ** 2))),
            "i_pb_rms": float(np.sqrt(np.mean(i_pb ** 2))),
            "i_eb_max": float(np.max(np.abs(i_eb))),
            "i_pb_max": float(np.max(np.abs(i_pb))),
            "p_eb_max": float(np.max(np.abs(p_eb))),
            "p_pb_max": float(np.max(np.abs(p_pb))),
            "p_eb_moy": float(np.mean(np.abs(p_eb))),
            "p_pb_moy": float(np.mean(np.abs(p_pb))),
        }
    return stats


# Critères de sélection : libellé -> (métrique, sens). "min" = plus bas = mieux.
CRITERES = {
    "Sécurité physique": ("nb_violations", "min"),
    "Coût énergétique": ("cout_physique_moyen", "min"),
    "Préservation EB": ("soc_eb_final", "max"),
    "Préservation PB": ("soc_pb_final", "max"),
    "Équilibre EB/PB": ("desequilibre_soc_moyen", "min"),
    "Performance globale": ("cout_physique_moyen", "min"),
    "Explicabilité": ("nb_corrections", "min"),  # proxy : moins de corrections = décision plus alignée
}


def meilleure_strategie(metriques: dict, critere: str):
    """Retourne (cle_strategie, valeur) de la meilleure stratégie selon le
    critère choisi. Ignore les valeurs NaN.
    """
    if critere not in CRITERES:
        raise ValueError(f"Critère inconnu : {critere}. Choix : {list(CRITERES)}")

    metrique, sens = CRITERES[critere]
    candidats = [
        (nom, infos[metrique])
        for nom, infos in metriques.items()
        if metrique in infos and not np.isnan(infos[metrique])
    ]
    if not candidats:
        return None, float("nan")

    if sens == "min":
        return min(candidats, key=lambda kv: kv[1])
    return max(candidats, key=lambda kv: kv[1])
