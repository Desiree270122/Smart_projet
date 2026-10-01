"""
core/resultats.py — Source UNIQUE de chargement et d'exploitation des
résultats de simulation.

Toutes les pages d'analyse (Tableau de bord, Comparaison, Résultats de simulation,
Explicabilité, Base de connaissances) passent par ce module. Elles ne relancent
jamais la simulation complète : elles lisent les fichiers produits par
`scripts/run_simulations.py`, ou la dernière simulation lancée dans l'application.

Chaque fichier de référence est AUTO-SUFFISANT : il contient les trajectoires de
chaque stratégie, les signaux du cycle (dont hasPower), et des métadonnées.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np

import ems_core as core
from core.i18n import lib, tr
from core.pertes import bilan_pertes


# Emplacement standard du résultat de référence.
FICHIER_REFERENCE = core.RESULTS_DIR / "precomputed" / "simulation_reference.joblib"


# Noms affichés des stratégies (français, anglais).
LIBELLES = {
    "EMS_power_limitation": ("Modèle physique", "Physical model"),
    "EMS_fuzzy_logic": ("Logique floue", "Fuzzy logic"),
    "EMS_MLP": ("MLP", "MLP"),
    "EMS_LSTM": ("LSTM", "LSTM"),
    "EMS_GNN": ("GNN", "GNN"),
    "EMS_MLP_neurosymbolic": ("NS-MLP", "NS-MLP"),
    "EMS_LSTM_neurosymbolic": ("NS-LSTM", "NS-LSTM"),
}


def nom_affichage(cle: str) -> str:
    """Nom lisible d'une stratégie (clé interne sinon)."""
    return lib(LIBELLES[cle]) if cle in LIBELLES else cle


# Cycles de référence précalculés par scripts/run_simulations.py.
CYCLES_REFERENCE = {
    "artemis": (("Artemis urbain + routier (×6)", "Artemis urban + road (×6)"), FICHIER_REFERENCE),
    "wltc": (("WLTC classe 3 (×4)", "WLTC class 3 (×4)"), core.RESULTS_DIR / "precomputed" / "simulation_wltc.joblib"),
}
CYCLE_PERSONNALISE = "personnalise"

# Clés de session. « cycle_pret » est TOUJOURS le cycle des résultats affichés ;
# le cycle en cours de préparation (page 2) vit à part, dans « cycle_prepare ».
CLE_CYCLE = "_cycle_choisi"          # cycle affiché ; conservé d'une page à l'autre
_CLE_SELECTEUR = "selecteur_cycle"   # clé du widget, effacée par Streamlit hors de la page
CLE_PERSONNALISE = "_simulation_personnalisee"

SOURCES = {
    "reference": ("résultats calculés à l'avance", "pre-computed results"),
    CYCLE_PERSONNALISE: ("simulation lancée dans l'application", "simulation run in the app"),
}


def libelle_cycle(cle: str) -> str:
    """Nom d'un cycle de référence ou du cycle préparé."""
    if cle in CYCLES_REFERENCE:
        return lib(CYCLES_REFERENCE[cle][0])
    return tr("Le cycle que vous avez préparé", "The cycle you prepared")


def charger_reference(chemin=None) -> dict:
    """Charge un fichier de résultats de référence.

    Retourne un dictionnaire {resultats, cycle_df, avertissements, meta}.
    Lève FileNotFoundError si les résultats n'ont pas encore été calculés.
    """
    chemin = Path(chemin) if chemin else FICHIER_REFERENCE
    if not chemin.exists():
        raise FileNotFoundError(
            tr(
                "Les résultats de référence sont introuvables ({f}). Ils se calculent une fois "
                "avec la commande : python scripts/run_simulations.py",
                "The reference results cannot be found ({f}). They are computed once with "
                "the command: python scripts/run_simulations.py",
                f=chemin.name,
            )
        )
    return _charger_joblib(str(chemin), chemin.stat().st_mtime)


@lru_cache(maxsize=4)
def _charger_joblib(chemin, _mtime):
    import joblib

    return joblib.load(chemin)


def cycles_disponibles(st) -> dict:
    """{clé: libellé} des cycles dont les résultats peuvent être affichés."""
    options = {cle: lib(libelle) for cle, (libelle, chemin) in CYCLES_REFERENCE.items() if Path(chemin).exists()}
    if CLE_PERSONNALISE in st.session_state:
        options[CYCLE_PERSONNALISE] = tr("Ma dernière simulation", "My last simulation")
    return options


def afficher_simulation_personnalisee(st, donnees) -> None:
    """Enregistre une simulation lancée dans l'application et la fait afficher
    par toutes les pages d'analyse."""
    st.session_state[CLE_PERSONNALISE] = donnees
    st.session_state[CLE_CYCLE] = CYCLE_PERSONNALISE
    _installer(st, donnees, CYCLE_PERSONNALISE, CYCLE_PERSONNALISE)


def choisir_cycle(st, conteneur=None) -> str:
    """Affiche le sélecteur du cycle de conduite dont les résultats sont montrés
    (cycles de référence, et dernière simulation lancée) ; retourne le choix.

    Le choix vaut pour toutes les pages d'analyse. Il est enregistré avant la
    réexécution de la page : assurer_donnees_session le lit donc à jour, que le
    sélecteur soit placé avant ou après son appel.
    """
    options = cycles_disponibles(st)
    if not options:
        return "artemis"
    courant = st.session_state.get(CLE_CYCLE)
    if courant not in options:
        courant = next(iter(options))
    st.session_state[CLE_CYCLE] = courant
    st.session_state[_CLE_SELECTEUR] = courant

    def _memoriser():
        st.session_state[CLE_CYCLE] = st.session_state[_CLE_SELECTEUR]

    (conteneur or st).selectbox(
        tr("Cycle de conduite", "Driving cycle"), list(options), format_func=options.get,
        key=_CLE_SELECTEUR, on_change=_memoriser,
        help=tr(
            "Résultats affichés sur toutes les pages d'analyse. « WLTC » est un cycle que les "
            "stratégies à apprentissage n'ont jamais rencontré.",
            "Results shown on all analysis pages. “WLTC” is a cycle the learning-based strategies "
            "have never encountered.",
        ),
    )
    return courant


def _installer(st, donnees, choix, source):
    st.session_state["resultats_simulation"] = donnees["resultats"]
    st.session_state["cycle_pret"] = donnees["cycle_df"]
    st.session_state["avertissements_simulation"] = donnees.get("avertissements", [])
    st.session_state["pas_alpha"] = donnees["meta"].get("pas_alpha")
    st.session_state["meta_simulation"] = donnees["meta"]
    # Cohérence physique (E3) précalculée avec la référence, si disponible.
    st.session_state["coherence_simulation"] = donnees.get("coherence")
    st.session_state["_donnees_chargees"] = choix
    st.session_state["_source_donnees"] = source


def libelle_cycle_affiche(st) -> str:
    """Nom du cycle dont les résultats sont affichés."""
    choix = st.session_state.get("_donnees_chargees", "artemis")
    if choix == CYCLE_PERSONNALISE:
        meta = st.session_state.get(CLE_PERSONNALISE, {}).get("meta", {})
        return tr("Ma dernière simulation : {c}", "My last simulation: {c}", c=libelle_cycle(meta.get("cycle_cle", "")))
    return libelle_cycle(choix)


def assurer_donnees_session(st, chemin=None) -> str:
    """Met en session les résultats du cycle choisi dans la barre latérale :
    `st.session_state["resultats_simulation"]` et `["cycle_pret"]`, que lisent
    toutes les pages d'analyse.

    - Cycles de référence : lus dans leur fichier précalculé (instantané).
    - « Ma dernière simulation » : celle lancée dans l'application, conservée
      à part pour pouvoir passer d'un cycle à l'autre sans la perdre.

    Les résultats et leur cycle sont toujours installés ensemble : préparer un
    nouveau cycle (page 2) ne peut pas les désaccorder.

    Retourne la source des données, dans la langue choisie.
    """
    choix = st.session_state.get(CLE_CYCLE, "artemis")
    if choix == CYCLE_PERSONNALISE and CLE_PERSONNALISE not in st.session_state:
        choix = "artemis"
    deja = st.session_state.get("_donnees_chargees") == choix and "resultats_simulation" in st.session_state
    if not deja:
        if choix == CYCLE_PERSONNALISE:
            _installer(st, st.session_state[CLE_PERSONNALISE], choix, CYCLE_PERSONNALISE)
        else:
            _installer(st, charger_reference(chemin or CYCLES_REFERENCE[choix][1]), choix, "reference")
    return lib(SOURCES.get(st.session_state.get("_source_donnees"), SOURCES["reference"]))


def _somme_wh(puissance) -> float:
    """Intègre une puissance (W, pas de 1 s) en énergie (Wh)."""
    return float(np.nansum(np.asarray(puissance, dtype=float))) * core.DT_SECONDS / 3600.0


# Découpage temporel des notebooks (01_configuration.ipynb) : les réseaux sont
# entraînés sur les premiers 50 % du cycle Artemis, validés sur les 25 % suivants
# et testés sur les 25 % restants, qu'ils n'ont jamais vus.
PART_ENTRAINEMENT, PART_VALIDATION = 0.50, 0.25


def debut_partie_test(n: int) -> int:
    """Premier instant de la partie test du cycle Artemis (n instants)."""
    return int(n * PART_ENTRAINEMENT) + int(n * PART_VALIDATION)


def cycle_artemis_affiche(st) -> bool:
    """Vrai si les résultats affichés portent sur le cycle Artemis, le seul dont
    une partie a servi à l'entraînement."""
    choix = st.session_state.get("_donnees_chargees")
    if choix == CYCLE_PERSONNALISE:
        meta = st.session_state.get(CLE_PERSONNALISE, {}).get("meta", {})
        return meta.get("cycle_cle") == "artemis"
    return choix in (None, "artemis")


def restreindre(donnees: dict, debut: int, fin=None) -> dict:
    """Mêmes données {resultats, cycle_df}, limitées aux instants [debut, fin)."""
    df = donnees["cycle_df"]
    n = len(df)
    fin = n if fin is None else fin
    resultats = {}
    for nom, traj in donnees["resultats"].items():
        coupe = {}
        for cle, valeur in traj.items():
            v = np.asarray(valeur) if isinstance(valeur, (list, np.ndarray)) else None
            if v is not None and v.ndim == 1 and len(v) in (n, n + 1):
                coupe[cle] = v[debut: fin + (len(v) - n)]
            else:
                coupe[cle] = valeur
        resultats[nom] = coupe
    return {"resultats": resultats, "cycle_df": df.iloc[debut:fin].reset_index(drop=True)}


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

    # Distance du cycle, pour ramener l'énergie consommée au kilomètre (M1).
    distance_km = (
        float(np.nansum(cycle_df["speed"].to_numpy(dtype=float))) * core.DT_SECONDS / 1000.0
        if cycle_df is not None and "speed" in cycle_df.columns
        else float("nan")
    )
    i_eb_lim = (core.P_EB_MIN_W / core.V_EB_PACK_NOM, core.P_EB_MAX_W / core.V_EB_PACK_NOM)
    i_pb_lim = (core.P_PB_MIN_W / core.V_PB_PACK_NOM, core.P_PB_MAX_W / core.V_PB_PACK_NOM)

    metriques = {}
    for nom, traj in resultats.items():
        soc_eb = np.asarray(traj["SOC_EB"], dtype=float)
        soc_pb = np.asarray(traj["SOC_PB"], dtype=float)
        m = min(len(soc_eb), len(soc_pb))
        desequilibre = np.abs(soc_eb[:m] - soc_pb[:m])
        pertes = bilan_pertes(traj)
        p_hess = np.asarray(traj["P_EB"], dtype=float) + np.asarray(traj["P_PB"], dtype=float)
        e_traction = float(np.sum(np.clip(p_hess, 0.0, None))) * core.DT_SECONDS / 3600.0
        e_cons = _somme_wh(p_hess) + pertes["total_wh"]
        i_eb = np.asarray(traj["I_EB"], dtype=float)
        i_pb = np.asarray(traj["I_PB"], dtype=float)
        tol = 1e-6
        viol_courant = int(np.sum(
            (i_eb < i_eb_lim[0] - tol) | (i_eb > i_eb_lim[1] + tol)
            | (i_pb < i_pb_lim[0] - tol) | (i_pb > i_pb_lim[1] + tol)
        ))
        # Suivi de puissance (M6) : écart entre puissance fournie et demandée en
        # traction, c'est-à-dire la demande non servie à chaque instant.
        if p_dem is not None:
            n_p = min(len(p_dem), len(p_hess))
            ecart_p = np.where(p_dem[:n_p] > core.EPS_POWER_W, p_hess[:n_p] - p_dem[:n_p], 0.0)
        else:
            ecart_p = -np.asarray(traj["P_unserved"], dtype=float)

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
            "pertes_totales_wh": pertes["total_wh"],
            # Protocole M1 à M6 (page « Comparaison des stratégies EMS »).
            "energie_consommee_wh": e_cons,
            "energie_km_wh": e_cons / distance_km if distance_km > 0 else float("nan"),
            "rendement_hess": e_traction / (e_traction + pertes["total_wh"]) if e_traction > 0 else float("nan"),
            "rmse_delta_soc": float(np.sqrt(np.nanmean((soc_eb[:m] - soc_pb[:m]) ** 2))),
            "delta_soc_max": float(np.nanmax(desequilibre)),
            "delta_soc_final": float(desequilibre[-1]),
            "pertes_convertisseur_wh": pertes["convertisseur_wh"],
            "nb_violations_courant": viol_courant,
            "rmse_puissance_kw": float(np.sqrt(np.mean(ecart_p ** 2))) / 1000.0,
            "ecart_puissance_max_kw": float(np.max(np.abs(ecart_p))) / 1000.0 if len(ecart_p) else 0.0,
            # Courants efficaces : indicateurs de vieillissement des deux batteries (M7).
            "i_eb_rms": float(np.sqrt(np.nanmean(i_eb ** 2))),
            "i_pb_rms": float(np.sqrt(np.nanmean(i_pb ** 2))),
        }

        metriques[nom] = infos

    return metriques


# Les quatre familles comparées dans l'offre de stage 2SMART (objectif 5) :
# règles fixes, ontologie seule, apprentissage seul, approche hybride.
FAMILLES = {
    ("Règles fixes", "Fixed rules"): ("EMS_power_limitation",),
    ("Ontologie seule", "Ontology only"): ("EMS_fuzzy_logic",),
    ("Apprentissage seul", "Learning only"): ("EMS_MLP", "EMS_LSTM", "EMS_GNN"),
    ("Hybride neuro-symbolique", "Neuro-symbolic hybrid"): ("EMS_MLP_neurosymbolic", "EMS_LSTM_neurosymbolic"),
}


def famille(cle: str) -> str:
    """Famille d'une stratégie (clé interne) ; « — » si elle n'est pas classée."""
    return next((lib(f) for f, membres in FAMILLES.items() if cle in membres), "—")


# Même réseau, sans puis avec composante symbolique : c'est la comparaison qui
# isole l'apport du symbolique (étude d'ablation).
PAIRES_SYMBOLIQUE = (
    ("EMS_MLP", "EMS_MLP_neurosymbolic"),
    ("EMS_LSTM", "EMS_LSTM_neurosymbolic"),
)
