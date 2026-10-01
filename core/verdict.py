"""
core/verdict.py — Quelle stratégie retenir, tous critères confondus ?

La conclusion ne fusionne pas les critères dans une note arbitraire. Elle suit
trois étapes, les mêmes pour toutes les stratégies :

1. Condition préalable (M5, M6) : une stratégie doit fournir toute la puissance
   demandée et respecter les limites. Sinon elle est écartée : ne pas servir la
   demande fait mécaniquement baisser l'énergie consommée et les pertes, ses
   autres chiffres seraient donc flattés.
2. Comparaison deux à deux sur chaque critère restant : une stratégie marque un
   point quand elle fait mieux qu'une autre d'un écart supérieur au seuil
   d'indifférence du critère, et en perd un quand elle fait moins bien. Son score
   est la moyenne de (comparaisons gagnées − perdues), entre −1 et +1.
3. Sensibilité aux poids : le score suppose des critères de même importance. On
   tire donc un grand nombre de jeux de poids au hasard et on compte dans quelle
   part d'entre eux chaque stratégie arrive en tête.

(Dans la littérature : comparaison par paires à seuils de type PROMETHEE II,
et analyse de robustesse de type SMAA.)
"""

import numpy as np

from core.i18n import lib
from core.xai import TRANSPARENCE


def _c(cle, libelle, famille, sens, seuil, relatif, seuil_txt):
    return {"cle": cle, "libelle": libelle, "famille": famille, "sens": sens,
            "seuil": seuil, "relatif": relatif, "seuil_txt": seuil_txt}


# Critères qui départagent les stratégies : (français, anglais) pour les textes.
CRITERES = [
    _c("energie_km_wh", ("M1 · Énergie consommée", "M1 · Energy consumed"), "performance", "min", 0.01, True,
       ("1 % de l'énergie consommée", "1 % of the energy consumed")),
    _c("rendement_hess", ("M2 · Rendement du HESS", "M2 · HESS efficiency"), "performance", "max", 0.001, False,
       ("0,1 point de rendement", "0.1 efficiency point")),
    _c("rmse_delta_soc", ("M3 · Écart entre les SOC", "M3 · Gap between SOCs"), "performance", "min", 0.01, False,
       ("1 point de SOC", "1 SOC point")),
    _c("pertes_convertisseur_wh", ("M4 · Pertes du convertisseur", "M4 · Converter losses"), "performance", "min", 0.05, True,
       ("5 % des pertes", "5 % of the losses")),
    _c("transparence", ("E1–E2 · Lisibilité de la décision", "E1–E2 · Readability of the decision"), "explicabilite",
       "max", 0.5, False, ("un niveau (directe, décomposable, indirecte)", "one level (direct, decomposable, indirect)")),
    _c("coherence_physique", ("E3 · Cohérence physique", "E3 · Physical consistency"), "explicabilite", "max", 0.05, False,
       ("5 points de pourcentage", "5 percentage points")),
]

NB_TIRAGES = 5000


def criteres(avec_explicabilite=True):
    return [c for c in CRITERES if avec_explicabilite or c["famille"] == "performance"]


def libelle(critere) -> str:
    return lib(critere["libelle"])


def completer(metriques, coherence=None):
    """Ajoute aux métriques les deux critères d'explicabilité : le niveau de
    lisibilité (propriété de l'architecture) et la cohérence physique mesurée."""
    for nom, m in metriques.items():
        m["transparence"] = float(TRANSPARENCE.get(nom, (0,))[0])
        mesure = (coherence or {}).get(nom)
        m["coherence_physique"] = float(mesure[0]) if mesure else float("nan")
    return metriques


def ecartees(metriques):
    """{stratégie: (énergie non fournie en Wh, nombre de dépassements de limite)}
    pour les stratégies qui ne remplissent pas la condition préalable."""
    sortie = {}
    for nom, m in metriques.items():
        non_fourni = float(m.get("energie_non_servie_wh", 0.0))
        depassements = int(m.get("nb_violations", 0) + m.get("nb_violations_courant", 0))
        if non_fourni >= 1.0 or m.get("rmse_puissance_kw", 0.0) > 1e-3 or depassements > 0:
            sortie[nom] = (non_fourni, depassements)
    return sortie


def preference(metriques, a, b, critere):
    """+1 si a fait mieux que b au-delà du seuil d'indifférence, −1 s'il fait
    moins bien, 0 sinon (y compris quand une valeur manque)."""
    va, vb = metriques[a].get(critere["cle"], float("nan")), metriques[b].get(critere["cle"], float("nan"))
    seuil = critere["seuil"] * abs(vb) if critere["relatif"] else critere["seuil"]
    ecart = (va - vb) if critere["sens"] == "max" else (vb - va)
    return 1 if ecart > seuil else (-1 if ecart < -seuil else 0)


def bilans(metriques, strategies, liste_criteres):
    """Matrice (stratégies × critères) : comparaisons gagnées moins perdues,
    divisées par le nombre d'adversaires. Entre −1 et +1."""
    n = len(strategies)
    if n < 2:
        return np.zeros((n, len(liste_criteres)))
    return np.array([
        [sum(preference(metriques, a, b, c) for b in strategies if b != a) / (n - 1) for c in liste_criteres]
        for a in strategies
    ])


def _poids(nb_criteres, graine=0):
    return np.random.default_rng(graine).dirichlet(np.ones(nb_criteres), NB_TIRAGES)


def _part_en_tete(scores_par_tirage):
    """Part des tirages où chaque stratégie a le meilleur score (les ex æquo se partagent le tirage)."""
    meilleurs = np.isclose(scores_par_tirage, scores_par_tirage.max(axis=1, keepdims=True))
    return (meilleurs / meilleurs.sum(axis=1, keepdims=True)).mean(axis=0)


def evaluer(metriques, liste_criteres, strategies=None):
    """Étapes 1 à 3 sur une évaluation (un cycle). `strategies` restreint la
    comparaison à une liste donnée ; sinon, toutes celles qui remplissent la
    condition préalable.

    Retourne {retenues, ecartees, bilans, score, en_tete} ; `score` et `en_tete`
    sont des dictionnaires {stratégie: valeur}.
    """
    exclues = ecartees(metriques)
    retenues = [n for n in metriques if n not in exclues] if strategies is None else list(strategies)
    matrice = bilans(metriques, retenues, liste_criteres)
    if len(retenues) == 0:
        return {"retenues": [], "ecartees": exclues, "bilans": matrice, "score": {}, "en_tete": {}}
    en_tete = _part_en_tete(_poids(len(liste_criteres)) @ matrice.T)
    return {
        "retenues": retenues,
        "ecartees": exclues,
        "bilans": matrice,
        "score": dict(zip(retenues, matrice.mean(axis=1))),
        "en_tete": dict(zip(retenues, en_tete)),
    }


def conclure(evaluations, liste_criteres):
    """Conclusion sur plusieurs évaluations {nom: métriques} (les cycles).

    Seules les stratégies qui remplissent la condition préalable sur TOUTES les
    évaluations sont comparées entre elles ; leur score est la moyenne de leurs
    scores sur chaque évaluation.

    Retourne {communes, score, en_tete, par_evaluation} ; `par_evaluation` donne,
    pour chaque évaluation, le résultat de `evaluer` restreint aux stratégies communes.
    """
    toutes = list(next(iter(evaluations.values()))) if evaluations else []
    communes = [n for n in toutes if all(n in m and n not in ecartees(m) for m in evaluations.values())]
    par_evaluation = {nom: evaluer(m, liste_criteres, communes) for nom, m in evaluations.items()}
    if not communes:
        return {"communes": [], "score": {}, "en_tete": {}, "par_evaluation": par_evaluation}
    moyenne = np.mean([r["bilans"] for r in par_evaluation.values()], axis=0)
    return {
        "communes": communes,
        "score": dict(zip(communes, moyenne.mean(axis=1))),
        "en_tete": dict(zip(communes, _part_en_tete(_poids(len(liste_criteres)) @ moyenne.T))),
        "par_evaluation": par_evaluation,
    }


def meilleures(score, tolerance=0.02):
    """Stratégies en tête d'un dictionnaire de scores (ex æquo à `tolerance` près), meilleure d'abord."""
    if not score:
        return []
    maxi = max(score.values())
    return [n for n, s in sorted(score.items(), key=lambda kv: -kv[1]) if s >= maxi - tolerance]


def points_forts(metriques, strategie, strategies, liste_criteres):
    """Critères sur lesquels `strategie` n'est battue par aucune autre et en bat au moins une."""
    autres = [n for n in strategies if n != strategie]
    return [
        c for c in liste_criteres
        if autres
        and all(preference(metriques, strategie, b, c) >= 0 for b in autres)
        and any(preference(metriques, strategie, b, c) == 1 for b in autres)
    ]
