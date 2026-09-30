"""
core/xai.py — Explication des décisions et critère M7 (explicabilité).

Trois outils, communs à toutes les pages :

1. Fonctions de décision : pour chaque stratégie, la part de la PB (alpha, avant
   le filtre de sécurité) qu'elle choisirait dans une situation donnée, en
   pouvant modifier une grandeur (SOC, puissance demandée).

2. Valeurs de Shapley EXACTES pour les réseaux opaques (MLP, LSTM, GNN), sans la
   bibliothèque shap : avec 4 à 11 entrées, on évalue le réseau sur toutes les
   combinaisons d'entrées ; une entrée absente prend sa valeur de référence
   (moyenne sur le cycle ; état inactif pour un état symbolique). Propriété
   garantie : alpha = alpha de référence + somme des contributions.

3. E3, cohérence physique : sur un échantillon d'instants de traction, la
   décision varie-t-elle dans le sens attendu par la physique du HESS ?
"""

import math
from functools import lru_cache

import numpy as np
import torch

import ems_core as core
from core import ontology_explainer as ox


ETATS_SYMBOLIQUES = set(ox.LIBELLES_SYMBOLIQUES)

LIBELLES_ENTREES = {
    "SOC_EB": "SOC EB",
    "SOC_PB": "SOC PB",
    "hasPower": "Puissance demandée",
    "speed": "Vitesse",
    "hasAcceleration": "Accélération",
    "hasTotalForce": "Force totale",
    "I_EB": "Courant EB",
    "high_power_demand": "Forte demande (symbolique)",
    "regenerative_braking": "Freinage (symbolique)",
    "zero_power_demand": "Demande nulle (symbolique)",
    "converter_risk": "Convertisseur chargé (symbolique)",
}

# E1 transparence et E2 traçabilité : propriétés de l'architecture, déclarées.
# 3 = directe, 2 = décomposable (règles + partie opaque bornée), 1 = indirecte.
TRANSPARENCE = {
    "EMS_power_limitation": (3, "directe : équations et seuils physiques", "native : la branche appliquée"),
    "EMS_fuzzy_logic": (3, "directe : règles SI … ALORS explicites", "native : contribution exacte de chaque règle"),
    "EMS_MLP_neurosymbolic": (2, "décomposable : règles floues + correction bornée", "exacte pour les règles ; la correction reste opaque"),
    "EMS_MLP": (1, "indirecte : réseau dense", "reconstruite : valeurs de Shapley"),
    "EMS_LSTM": (1, "indirecte : réseau récurrent", "reconstruite : valeurs de Shapley sur l'historique"),
    "EMS_LSTM_neurosymbolic": (1, "indirecte : réseau récurrent à entrées symboliques", "reconstruite : valeurs de Shapley, entrées symboliques nommées"),
    "EMS_GNN": (1, "indirecte : réseau sur graphe", "reconstruite : valeurs de Shapley"),
}

SHAPLEY_DISPONIBLE = {"EMS_MLP", "EMS_LSTM", "EMS_LSTM_neurosymbolic", "EMS_GNN"}


# Chargement des modèles (une fois par processus)

@lru_cache(maxsize=None)
def _modele(strategie):
    if strategie == "EMS_MLP":
        return core.load_mlp_simple().eval(), core.charger_scaler(core.MLP_SCALER_FILE)
    if strategie == "EMS_LSTM":
        return core.load_lstm_seul().eval(), core.charger_scaler(core.LSTM_SCALER_FILE)
    if strategie == "EMS_LSTM_neurosymbolic":
        return core.load_lstm_neurosymbolic().eval(), core.charger_scaler(core.LSTM_NS_SCALER_FILE)
    if strategie == "EMS_MLP_neurosymbolic":
        return core.load_mlp_neurosymbolic().eval(), core.charger_scaler(core.MLP_NS_SCALER_FILE)
    if strategie == "EMS_GNN":
        res = core.load_gnn_simple()
        return (res[0] if isinstance(res, tuple) else res).eval(), core.charger_scaler(core.GNN_SCALER_FILE)
    raise KeyError(strategie)


def _normaliser(x, scaler):
    if scaler is None:
        return np.asarray(x, dtype=np.float64)
    try:
        return core.appliquer_scaler(x, scaler)
    except ValueError:
        return np.asarray(x, dtype=np.float64)


# Situation à un instant, et fenêtre temporelle vue par les LSTM

def situation(df, traj, i):
    return {
        "SOC_EB": float(traj["SOC_EB"][i]),
        "SOC_PB": float(traj["SOC_PB"][i]),
        "hasPower": float(df["hasPower"].iloc[i]),
        "speed": float(df["speed"].iloc[i]) if "speed" in df.columns else 0.0,
        "hasAcceleration": float(df["hasAcceleration"].iloc[i]) if "hasAcceleration" in df.columns else 0.0,
    }


def fenetre_lstm(cols, i, df, traj, surcharges=None):
    """Fenêtre (LSTM_WINDOW × colonnes) de valeurs brutes vue par un LSTM à
    l'instant i ; les surcharges modifient la dernière ligne (instant courant)."""
    w = core.LSTM_WINDOW
    idx = [max(0, i - w + 1 + k) for k in range(w)]
    lignes = []
    for rang, j in enumerate(idx):
        base = situation(df, traj, j)
        # Comme dans la simulation : le courant EB connu à l'instant j est celui du
        # pas précédent, déduit de la variation de SOC (nul au premier pas).
        base["I_EB"] = float(traj["I_EB"][j - 1]) if j >= 1 else 0.0
        base["hasTotalForce"] = float(df["hasTotalForce"].iloc[j]) if "hasTotalForce" in df.columns else 0.0
        if surcharges and rang == w - 1:
            base.update(surcharges)
        if any(c in ETATS_SYMBOLIQUES for c in cols):
            base.update(
                core.compute_symbolic_states(
                    base["hasPower"], base["SOC_EB"], base["SOC_PB"], p_eb=base["I_EB"] * core.V_EB_PACK_NOM,
                )
            )
        lignes.append([float(base[c]) for c in cols])
    return np.asarray(lignes, dtype=np.float64)


def _alpha_depuis_lstm(sortie, p_dem):
    """Même interprétation que la simulation (deriver_alpha_depuis_sortie_lstm),
    vectorisée sur la troisième sortie (variation de SOC de la PB prédite)."""
    if abs(p_dem) <= core.EPS_POWER_W:
        return np.full(len(sortie), 0.5)
    p_pb = -np.asarray(sortie, dtype=float) * core.V_PB_PACK_NOM * 3600.0 * core.CAPACITY_PB_AH / core.DT_SECONDS
    return np.clip(p_pb / p_dem, 0.0, 1.0)


def _lstm_lot(strategie, fenetres, p_dem):
    modele, scaler = _modele(strategie)
    x = torch.tensor(_normaliser(fenetres, scaler), dtype=torch.float32, device=core.DEVICE)
    with torch.no_grad(), torch.backends.cudnn.flags(enabled=False):
        sortie = modele(x).cpu().numpy()
    # Comme dans la simulation : le LSTM prédit des cibles normalisées.
    if scaler is not None and {"y_mean", "y_std"}.issubset(set(scaler.files)):
        sortie = sortie * scaler["y_std"] + scaler["y_mean"]
    return sortie


# 1 — Fonction de décision de chaque stratégie

def alpha_decision(strategie, df, traj, i, surcharges=None):
    """Part de la PB (avant filtre) que choisirait la stratégie à l'instant i,
    avec d'éventuelles grandeurs modifiées. NaN si la demande est quasi nulle."""
    s = situation(df, traj, i)
    s.update(surcharges or {})
    p, se, sp, acc = s["hasPower"], s["SOC_EB"], s["SOC_PB"], s["hasAcceleration"]
    if abs(p) <= core.EPS_POWER_W:
        return float("nan")

    if strategie == "EMS_power_limitation":
        return float(ox.alpha_ontologie_vect([p], [se])[0])
    if strategie == "EMS_fuzzy_logic":
        return float(core.alpha_fuzzy_calc(np.array([se]), np.array([sp]), np.array([p]), np.array([acc]))["alpha"][0])
    if strategie == "EMS_MLP":
        modele, scaler = _modele(strategie)
        x = _normaliser(np.array([s[c] for c in core.MLP_INPUT_COLS]), scaler)
        with torch.no_grad():
            return float(modele(torch.tensor([x.tolist()], dtype=torch.float32, device=core.DEVICE)).item())
    if strategie == "EMS_GNN":
        modele, scaler = _modele(strategie)
        x_g, edge = core.construire_graphe_instant(p, se, sp, acc, scaler)
        with torch.no_grad():
            return float(
                modele(x_g.to(core.DEVICE), edge.to(core.DEVICE),
                       torch.zeros(x_g.shape[0], dtype=torch.long, device=core.DEVICE)).item()
            )
    if strategie in ("EMS_LSTM", "EMS_LSTM_neurosymbolic"):
        cols = core.LSTM_NS_FEATURE_COLS if strategie == "EMS_LSTM_neurosymbolic" else core.LSTM_FEATURE_COLS
        fen = fenetre_lstm(cols, i, df, traj, surcharges)
        return float(_alpha_depuis_lstm(_lstm_lot(strategie, fen[None], p)[:, 2], p)[0])
    if strategie == "EMS_MLP_neurosymbolic":
        # Reconstitution des 17 entrées : prédictions du LSTM sur la trajectoire de
        # la stratégie, sortie floue, états symboliques (comme dans la simulation).
        pred = _lstm_lot("EMS_LSTM", fenetre_lstm(core.LSTM_FEATURE_COLS, i, df, traj, surcharges)[None], p)[0]
        a_f = float(core.alpha_fuzzy_calc(np.array([se]), np.array([sp]), np.array([p]), np.array([acc]))["alpha"][0])
        valeurs = {
            **s, "Pdem_pred_ns": float(pred[0]), "delta_soc_eb_pred_ns": float(pred[1]),
            "delta_soc_pb_pred_ns": float(pred[2]), "alpha_ems_fuzzy_logic": a_f,
            **{k: float(v) for k, v in core.compute_symbolic_states(p, se, sp).items()},
        }
        modele, scaler = _modele(strategie)
        x = _normaliser(np.array([valeurs[c] for c in core.MLP_NS_INPUT_COLS]), scaler)
        with torch.no_grad():
            alpha, _, _ = modele(
                torch.tensor([x.tolist()], dtype=torch.float32, device=core.DEVICE),
                torch.tensor([a_f], dtype=torch.float32, device=core.DEVICE),
            )
        return float(alpha.item())
    raise KeyError(strategie)


# 2 — Valeurs de Shapley exactes

def _shapley(valeurs, n):
    """Valeurs de Shapley à partir des valeurs v[code] de toutes les coalitions
    (code binaire : bit j = entrée j présente)."""
    poids = [math.factorial(s) * math.factorial(n - s - 1) / math.factorial(n) for s in range(n)]
    phi = np.zeros(n)
    for code in range(2 ** n):
        taille = bin(code).count("1")
        for j in range(n):
            if not code >> j & 1:
                phi[j] += poids[taille] * (valeurs[code | 1 << j] - valeurs[code])
    return phi


def _masques(n):
    return np.array([[code >> j & 1 for j in range(n)] for code in range(2 ** n)], dtype=bool)


def shapley(strategie, df, traj, i):
    """(alpha de référence, [(entrée, contribution)], alpha expliqué) pour un
    réseau opaque à l'instant i. Référence : moyenne de chaque grandeur sur le
    cycle (0 pour un état symbolique, c'est-à-dire état inactif)."""
    s = situation(df, traj, i)
    p = s["hasPower"]
    n_pts = min(len(df), len(traj["SOC_EB"]))
    moyenne = {
        "SOC_EB": float(np.mean(traj["SOC_EB"][:n_pts])),
        "SOC_PB": float(np.mean(traj["SOC_PB"][:n_pts])),
        "hasPower": float(df["hasPower"].mean()),
        "speed": float(df["speed"].mean()) if "speed" in df.columns else 0.0,
        "hasAcceleration": float(df["hasAcceleration"].mean()) if "hasAcceleration" in df.columns else 0.0,
        "hasTotalForce": float(df["hasTotalForce"].mean()) if "hasTotalForce" in df.columns else 0.0,
        "I_EB": float(np.mean(traj["I_EB"][:n_pts])),
    }

    if strategie == "EMS_MLP":
        cols = list(core.MLP_INPUT_COLS)
        x = np.array([s[c] for c in cols])
        b = np.array([moyenne[c] for c in cols])
        m = _masques(len(cols))
        modele, scaler = _modele(strategie)
        lot = torch.tensor(_normaliser(np.where(m, x, b), scaler), dtype=torch.float32, device=core.DEVICE)
        with torch.no_grad():
            v = modele(lot).cpu().numpy().reshape(-1)
    elif strategie == "EMS_GNN":
        cols = ["hasPower", "SOC_EB", "SOC_PB", "hasAcceleration"]
        x = np.array([s[c] for c in cols])
        b = np.array([moyenne[c] for c in cols])
        m = _masques(len(cols))
        v = np.array([
            alpha_decision(strategie, df, traj, i, dict(zip(cols, np.where(mm, x, b)))) for mm in m
        ])
    elif strategie in ("EMS_LSTM", "EMS_LSTM_neurosymbolic"):
        cols = list(core.LSTM_NS_FEATURE_COLS if strategie == "EMS_LSTM_neurosymbolic" else core.LSTM_FEATURE_COLS)
        fen = fenetre_lstm(cols, i, df, traj)
        base = np.array([0.0 if c in ETATS_SYMBOLIQUES else moyenne[c] for c in cols])
        m = _masques(len(cols))
        lots = np.where(m[:, None, :], fen[None, :, :], base[None, None, :])
        v = _alpha_depuis_lstm(_lstm_lot(strategie, lots, p)[:, 2], p)
    else:
        raise KeyError(f"Pas de valeurs de Shapley pour {strategie}")

    phi = _shapley(v, len(cols))
    return float(v[0]), list(zip(cols, phi)), float(v[-1])


# 3 — E3 : cohérence physique des décisions

# (grandeur modifiée, variation, sens attendu de la puissance PB, énoncé).
# On juge la puissance fournie par la PB (alpha × demande) : si la demande monte
# et que la PB garde la même puissance, sa part baisse mécaniquement sans que la
# décision soit incohérente.
CONTRAINTES_PHYSIQUES = [
    ("SOC_PB", 0.05, +1, "plus de charge dans la PB ne doit pas réduire la puissance de la PB"),
    ("SOC_EB", 0.05, -1, "plus de charge dans l'EB ne doit pas augmenter la puissance de la PB"),
    ("hasPower", 1000.0, +1, "une demande de traction plus forte ne doit pas réduire la puissance de la PB"),
]
TOLERANCE = 0.01  # en part de la demande (1 %)


def coherence_physique(strategie, df, traj, n_instants=120):
    """Part des instants de traction échantillonnés où la décision respecte les
    trois sens de variation attendus, et part pour chaque contrainte."""
    p = df["hasPower"].to_numpy(dtype=float)
    n_pts = min(len(p), len(traj["SOC_EB"]))
    traction = np.flatnonzero(p[:n_pts] > 1000.0)
    traction = traction[traction >= core.LSTM_WINDOW]
    if len(traction) == 0:
        return float("nan"), {}
    instants = traction[np.linspace(0, len(traction) - 1, min(n_instants, len(traction))).astype(int)]

    p_pb0 = {i: alpha_decision(strategie, df, traj, i) * p[i] for i in instants}
    respect = np.ones(len(instants), dtype=bool)
    par_contrainte = {}
    for grandeur, pas, sens, enonce in CONTRAINTES_PHYSIQUES:
        ok = []
        for i in instants:
            valeur = situation(df, traj, i)[grandeur]
            haut = min(valeur + pas, 1.0) if grandeur.startswith("SOC") else valeur + pas
            p_haut = haut if grandeur == "hasPower" else p[i]
            p_pb1 = alpha_decision(strategie, df, traj, i, {grandeur: haut}) * p_haut
            ok.append(sens * (p_pb1 - p_pb0[i]) >= -TOLERANCE * p[i])
        ok = np.asarray(ok)
        respect &= ok
        par_contrainte[enonce] = float(ok.mean())
    return float(respect.mean()), par_contrainte
