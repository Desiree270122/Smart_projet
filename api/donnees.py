"""
api/donnees.py — Données servies par le serveur de calcul.

Trois sortes de « cycles » peuvent être analysés :
- les cycles de référence (« artemis », « wltc »), lus dans leur fichier de résultats ;
- les simulations lancées depuis l'application (« sim-… »), gardées en mémoire ;
- les cycles préparés (« prep-… ») ne sont pas des résultats : ils attendent une simulation.

Le moteur de simulation (ems_core) garde ses paramètres dans des variables
globales. Un verrou unique protège donc tous les calculs : un seul à la fois, et
les paramètres par défaut sont remis en place après chaque calcul.
"""

import threading
import uuid
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path

import numpy as np

import ems_core as core
from core.resultats import CYCLES_REFERENCE, charger_reference, libelle_cycle

PREFIXE_SIMULATION = "sim-"
PREFIXE_PREPARE = "prep-"
NB_MAX_EN_MEMOIRE = 8          # simulations et cycles préparés gardés en mémoire
ATTENTE_VERROU_S = 20.0        # au-delà, le serveur répond « occupé »

_verrou_moteur = threading.RLock()
_verrou_registres = threading.Lock()
simulations = OrderedDict()     # id -> {"etat", "donnees", "erreurs", "duree_s", "durees", "message"}
cycles_prepares = OrderedDict() # id -> {"df", "meta", "materiel"}
fichiers = OrderedDict()        # id -> {"nom", "contenu"}

# Paramètres du moteur que la préparation d'un cycle peut modifier ; leur valeur
# d'origine est gardée pour être remise en place après chaque calcul.
_PARAMETRES_MOTEUR = [
    "V_EB_PACK_NOM", "P_EB_MIN_W", "P_EB_MAX_W", "CAPACITY_EB_AH", "ENERGY_EB_WH",
    "V_PB_PACK_NOM", "P_PB_MIN_W", "P_PB_MAX_W", "CAPACITY_PB_AH", "ENERGY_PB_WH",
    "ENERGY_TOTAL_WH", "ENERGY_SHARE_EB", "ENERGY_SHARE_PB", "ENERGY_COST_NORMALIZER",
    "P_CONV_MIN_W", "P_CONV_MAX_W", "_P_EB_CONV_MAX", "_P_EB_CONV_MIN",
]
_VALEURS_D_ORIGINE = {nom: getattr(core, nom) for nom in _PARAMETRES_MOTEUR}


class Occupe(Exception):
    """Le moteur est pris par une simulation en cours."""


class Introuvable(Exception):
    """Cycle, simulation ou fichier inconnu (ou sorti de la mémoire)."""


def nouvel_identifiant() -> str:
    return uuid.uuid4().hex[:12]


def enregistrer(registre, identifiant, valeur):
    with _verrou_registres:
        registre[identifiant] = valeur
        while len(registre) > NB_MAX_EN_MEMOIRE:
            registre.popitem(last=False)


def appliquer_materiel(materiel):
    """Règle le moteur avec les batteries et le convertisseur d'un cycle préparé.
    materiel : {"eb": caractéristiques du pack, "pb": …, "convertisseur": {p_decharge_W, p_recharge_W}}."""
    if not materiel:
        return
    core.set_battery_pack_parameters("EB", materiel["eb"])
    core.set_battery_pack_parameters("PB", materiel["pb"])
    core.set_converter_power_limits(materiel["convertisseur"]["p_decharge_W"], materiel["convertisseur"]["p_recharge_W"])


def retablir_moteur():
    for nom, valeur in _VALEURS_D_ORIGINE.items():
        setattr(core, nom, valeur)
    core.set_pertes(False)
    core.set_alpha_grid_step(core.ALPHA_GRID_STEP_DEFAUT)


@contextmanager
def moteur(materiel=None, attendre=True):
    """Réserve le moteur de simulation le temps d'un calcul, avec le matériel du
    cycle concerné ; remet ensuite les paramètres par défaut."""
    if not _verrou_moteur.acquire(timeout=ATTENTE_VERROU_S if attendre else -1):
        raise Occupe()
    try:
        appliquer_materiel(materiel)
        yield
    finally:
        retablir_moteur()
        _verrou_moteur.release()


def cycles_de_reference() -> list:
    """[(identifiant, nom)] des cycles de référence dont les résultats existent."""
    return [(cle, libelle_cycle(cle)) for cle, (_, chemin) in CYCLES_REFERENCE.items() if Path(chemin).exists()]


def nom_du_cycle(identifiant: str) -> str:
    """Nom affiché d'un cycle de référence, d'une simulation ou d'un cycle préparé."""
    from core.i18n import tr

    if identifiant in CYCLES_REFERENCE:
        return libelle_cycle(identifiant)
    if identifiant.startswith(PREFIXE_SIMULATION):
        origine = simulations.get(identifiant, {}).get("donnees", {}).get("meta", {}).get("cycle_origine", "")
        base = nom_du_cycle(origine) if origine else ""
        return tr("Ma dernière simulation", "My last simulation") + (f" · {base}" if base else "")
    return tr("Le cycle que vous avez préparé", "The cycle you prepared")


def obtenir(identifiant: str) -> dict:
    """Résultats d'un cycle : {resultats, cycle_df, coherence, meta, materiel}."""
    if identifiant in CYCLES_REFERENCE:
        chemin = CYCLES_REFERENCE[identifiant][1]
        if not Path(chemin).exists():
            raise Introuvable(identifiant)
        donnees = charger_reference(chemin)
        return {
            "resultats": donnees["resultats"],
            "cycle_df": donnees["cycle_df"],
            "coherence": donnees.get("coherence") or {},
            "meta": {**donnees.get("meta", {}), "pertes_dans_soc": False},
            "materiel": None,
        }
    simulation = simulations.get(identifiant)
    if simulation is None or simulation.get("etat") != "terminee":
        raise Introuvable(identifiant)
    return simulation["donnees"]


def cycle_a_simuler(identifiant: str):
    """(tableau du cycle, matériel) pour lancer une simulation sur un cycle de
    référence ou sur un cycle préparé."""
    if identifiant in CYCLES_REFERENCE:
        return obtenir(identifiant)["cycle_df"].copy(), None
    prepare = cycles_prepares.get(identifiant)
    if prepare is None:
        raise Introuvable(identifiant)
    return prepare["df"].copy(), prepare.get("materiel")


# Mise en forme pour le JSON

def liste(valeurs, decimales=3):
    """Tableau numérique -> liste JSON (arrondie ; None pour les valeurs manquantes)."""
    a = np.asarray(valeurs, dtype=float)
    a = np.round(a, decimales)
    if not np.isfinite(a).all():
        return [None if not np.isfinite(v) else v for v in a.tolist()]
    return a.tolist()


def nombre_json(x, decimales=6):
    """Nombre -> valeur JSON (None s'il manque)."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return round(x, decimales) if np.isfinite(x) else None


def textes_compacts(textes):
    """Liste de textes souvent répétés -> {"textes": distincts, "indices": un par instant}."""
    rangs, distincts, indices = {}, [], []
    for texte in textes:
        rang = rangs.get(texte)
        if rang is None:
            rang = rangs[texte] = len(distincts)
            distincts.append(texte)
        indices.append(rang)
    return {"textes": distincts, "indices": indices}
