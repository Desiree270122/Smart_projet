"""
core/verdict.py — Désigner le meilleur modèle en tenant compte de tous les critères.

Protocole de comparaison : mêmes conditions pour toutes les stratégies (même
cycle, mêmes batteries, même convertisseur, même filtre), puis sept métriques.

- M5 respect des contraintes et M6 suivi de puissance sont ÉLIMINATOIRES : une
  stratégie qui viole les limites ou ne fournit pas la puissance demandée
  paraîtrait sinon plus sobre (M1) et plus efficace (M2) qu'elle ne l'est.
- M1 énergie consommée, M2 rendement du HESS, M3 équilibrage des SOC, M4 pertes
  du convertisseur et M7 sollicitation des batteries (durée de vie) départagent
  les stratégies restantes. M7 n'était pas dans la proposition initiale à six
  métriques : sans lui, le protocole favoriserait les stratégies qui usent la PB.

Le verdict ne fusionne pas les métriques en un score arbitraire ; il suit
quatre étapes d'aide à la décision multicritère :
1. critères éliminatoires (M5, M6) ;
2. dominance : A domine B s'il est au moins aussi bon partout et meilleur au
   moins une fois — vrai quelle que soit la pondération ;
3. comparaison deux à deux avec seuils d'indifférence (PROMETHEE II) ;
4. robustesse (SMAA) : part des pondérations aléatoires où chaque stratégie est première.
"""

import numpy as np

from core.resultats import EXPLICABILITE


def _c(cle, libelle, objectif, sens, seuil, relatif, justification, unite="", echelle=1.0, fmt="{:.1f}"):
    return {
        "cle": cle, "libelle": libelle, "objectif": objectif, "sens": sens, "seuil": seuil,
        "relatif": relatif, "justification": justification, "unite": unite, "echelle": echelle, "fmt": fmt,
    }


# Métriques principales du protocole (M1 à M4, M7).
CRITERES_PRINCIPAUX = [
    _c("energie_km_wh", "M1 · Énergie consommée", "Rendement global", "min", 0.01, True,
       "1 % (énergie tirée des batteries, pertes estimées comprises)", "Wh/km"),
    _c("rendement_hess", "M2 · Rendement du HESS", "Rendement global", "max", 0.001, False,
       "0,1 point de rendement", "%", 100.0, "{:.2f}"),
    _c("rmse_delta_soc", "M3 · Équilibrage des SOC", "Équilibre des batteries", "min", 0.01, False,
       "1 point de SOC (écart quadratique moyen entre SOC_EB et SOC_PB)", "pts", 100.0),
    _c("pertes_convertisseur_wh", "M4 · Pertes du convertisseur", "Pertes du convertisseur", "min", 0.05, True,
       "5 % (pertes estimées avec un rendement constant)", "Wh", 1.0, "{:.0f}"),
    _c("i_eb_rms", "M7 · Courant efficace EB", "Durée de vie", "min", 0.5, False, "0,5 A", "A"),
    _c("i_pb_rms", "M7 · Courant efficace PB", "Durée de vie", "min", 0.5, False, "0,5 A", "A"),
]

# Critères complémentaires, proposés en option.
CRITERES_COMPLEMENTAIRES = [
    _c("nb_corrections", "Corrections du filtre", "Cohérence physique", "min", 0.05, True,
       "5 % du nombre de corrections", "", 1.0, "{:.0f}"),
    _c("cout_physique_moyen", "Coût physique", "Rendement global", "min", 0.01, True,
       "1 % du coût multicritère du filtre", "", 1.0, "{:.4f}"),
    _c("soc_eb_final", "SOC final EB", "Préservation des batteries", "max", 0.01, False, "1 point de SOC", "%", 100.0),
    _c("soc_pb_final", "SOC final PB", "Préservation des batteries", "max", 0.01, False, "1 point de SOC", "%", 100.0),
]

CRITERE_EXPLICABILITE = _c(
    "explicabilite", "Explicabilité (déclarée)", "Explicabilité", "max", 0.5, False,
    "un niveau d'écart (niveau déclaré, non mesuré)", "", 1.0, "{:.0f}",
)

# Métriques éliminatoires, affichées dans la matrice.
ELIMINATOIRES = [
    _c("nb_violations", "M5 · Violations de SOC", "Respect des contraintes", "min", 0, False, "doit valoir 0", "", 1.0, "{:.0f}"),
    _c("nb_violations_courant", "M5 · Violations de courant", "Respect des contraintes", "min", 0, False, "doit valoir 0", "", 1.0, "{:.0f}"),
    _c("rmse_puissance_kw", "M6 · Suivi de puissance (RMSE)", "Suivi de puissance", "min", 0, False, "doit valoir 0", "kW", 1.0, "{:.2f}"),
]

SEUIL_ROBUSTE = 0.70


def valeur(metriques, n, cle):
    if cle == "explicabilite":
        return float(EXPLICABILITE.get(n, (0, ""))[0])
    return float(metriques[n].get(cle, float("nan")))


def eliminer(metriques):
    """(retenues, {stratégie éliminée: raison}) selon M5 et M6."""
    exclues = {}
    for n, m in metriques.items():
        raisons = []
        if m.get("rmse_puissance_kw", 0.0) > 1e-3 or m.get("energie_non_servie_wh", 0.0) >= 1.0:
            raisons.append(
                f"M6, demande non suivie : {m.get('energie_non_servie_wh', 0.0):.0f} Wh non fournis, "
                f"jusqu'à {m.get('ecart_puissance_max_kw', 0.0):.1f} kW manquants à un instant"
            )
        viol = m.get("nb_violations", 0) + m.get("nb_violations_courant", 0)
        if viol > 0:
            raisons.append(f"M5, {viol} violation(s) de contrainte")
        if raisons:
            exclues[n] = " ; ".join(raisons)
    return [n for n in metriques if n not in exclues], exclues


def preference(metriques, a, b, crit):
    """+1 si a est meilleur que b au-delà du seuil, −1 s'il est moins bon, 0 sinon."""
    va, vb = valeur(metriques, a, crit["cle"]), valeur(metriques, b, crit["cle"])
    seuil = crit["seuil"] * abs(vb) if crit["relatif"] else crit["seuil"]
    ecart = (va - vb) if crit["sens"] == "max" else (vb - va)
    return 1 if ecart > seuil else (-1 if ecart < -seuil else 0)


def flux_par_critere(metriques, strategies, criteres):
    """Matrice (stratégies × critères) des flux nets : victoires moins défaites,
    divisées par le nombre d'adversaires. Entre −1 et +1."""
    n = len(strategies)
    if n < 2:
        return np.zeros((n, len(criteres)))
    return np.array(
        [
            [sum(preference(metriques, a, b, c) for b in strategies if b != a) / (n - 1) for c in criteres]
            for a in strategies
        ]
    )


def dominances(metriques, strategies, criteres):
    """[(a, b)] : a domine b."""
    return [
        (a, b)
        for a in strategies for b in strategies
        if a != b
        and all(preference(metriques, a, b, c) >= 0 for c in criteres)
        and any(preference(metriques, a, b, c) == 1 for c in criteres)
    ]


def bilan_paires(metriques, strategies, criteres):
    """{(a, b): (critères où a est meilleur, critères où a est moins bon)}."""
    return {
        (a, b): (
            [c["libelle"] for c in criteres if preference(metriques, a, b, c) == 1],
            [c["libelle"] for c in criteres if preference(metriques, a, b, c) == -1],
        )
        for a in strategies for b in strategies if a != b
    }


def robustesse(flux, n_tirages=5000, graine=0):
    """Part des pondérations aléatoires (uniformes sur le simplexe) où chaque
    stratégie est première, et son rang moyen."""
    n, k = flux.shape
    if n == 0 or k == 0:
        return np.zeros(n), np.zeros(n)
    poids = np.random.default_rng(graine).dirichlet(np.ones(k), n_tirages)
    scores = poids @ flux.T
    premiers = np.bincount(scores.argmax(axis=1), minlength=n) / n_tirages
    rangs = (-scores).argsort(axis=1).argsort(axis=1) + 1
    return premiers, rangs.mean(axis=0)
