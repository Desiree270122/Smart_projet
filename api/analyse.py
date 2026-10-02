"""
api/analyse.py — Données des pages d'analyse : tableau de bord, résultats de
simulation et comparaison des stratégies.

Les phrases rédigées (synthèse, conclusion, bulles des courbes) sont produites
ici, dans la langue de la requête ; les nombres bruts sont mis en forme par
l'interface.
"""

from pathlib import Path

import numpy as np

import ems_core as core
from api.donnees import liste, nom_du_cycle, nombre_json, obtenir, textes_compacts
from core import lecture, verdict, xai
from core.format import nombre
from core.i18n import lib, tr
from core.pertes import pertes_par_pas
from core.resultats import (
    CYCLES_REFERENCE, PAIRES_SYMBOLIQUE, calculer_metriques, charger_reference, famille, libelle_cycle, nom_affichage,
)
from core.style import couleur

NB_POINTS_PERTES = 2000   # la courbe des pertes cumulées est lisse : inutile d'en envoyer plus


def fiche_strategie(cle: str) -> dict:
    return {"cle": cle, "nom": nom_affichage(cle), "famille": famille(cle), "couleur": couleur(cle)}


def limites() -> dict:
    """Limites physiques et grandeurs du HESS utilisées par les graphiques."""
    return {
        "p_eb_max_w": core.P_EB_MAX_W, "p_eb_min_w": core.P_EB_MIN_W,
        "p_pb_max_w": core.P_PB_MAX_W, "p_pb_min_w": core.P_PB_MIN_W,
        "p_conv_max_w": core.P_CONV_MAX_W, "p_conv_min_w": core.P_CONV_MIN_W,
        "soc_eb_min": core.SOC_EB_MIN, "soc_pb_min": core.SOC_PB_MIN, "soc_max": core.SOC_EB_MAX,
        "v_eb": core.V_EB_PACK_NOM, "v_pb": core.V_PB_PACK_NOM,
        "energie_eb_wh": core.ENERGY_EB_WH, "energie_pb_wh": core.ENERGY_PB_WH,
        "pas_s": core.DT_SECONDS,
    }


def _temps(df, n):
    return df["time"].to_numpy(dtype=float)[:n] if "time" in df.columns else np.arange(n, dtype=float)


def infos_cycle(identifiant: str) -> dict:
    """Carte d'identité d'un cycle analysable : nom, durée, stratégies simulées."""
    donnees = obtenir(identifiant)
    df, resultats = donnees["cycle_df"], donnees["resultats"]
    n = min([len(df)] + [len(t["P_EB"]) for t in resultats.values()])
    temps = _temps(df, n)
    # Instant proposé pour l'explication : le plus proche du milieu du cycle où le véhicule
    # demande une puissance franche (le milieu lui-même tombe souvent sur un arrêt).
    p_dem = df["hasPower"].to_numpy(dtype=float)[:n]
    francs = np.flatnonzero(p_dem >= 0.3 * p_dem.max(initial=0.0)) if p_dem.max(initial=0.0) > 0 else np.array([], dtype=int)
    i_defaut = int(francs[np.abs(francs - n // 2).argmin()]) if len(francs) else n // 2
    return {
        "id": identifiant,
        "nom": nom_du_cycle(identifiant),
        "strategies": [fiche_strategie(c) for c in core.MODEL_ORDER if c in resultats],
        "nb_instants": n,
        "t_min": float(temps[0]), "t_max": float(temps[-1]), "t_defaut": float(temps[i_defaut]),
        "duree_min": float(n * core.DT_SECONDS / 60.0),
        "distance_km": nombre_json(df["speed"].to_numpy(dtype=float)[:n].sum() * core.DT_SECONDS / 1000.0, 2) if "speed" in df.columns else None,
        "pertes_dans_soc": bool(donnees["meta"].get("pertes_dans_soc")),
        "e3_disponible": all(c in donnees["coherence"] for c in resultats),
        "limites": limites(),
    }


def _trajectoire(identifiant, strategie):
    donnees = obtenir(identifiant)
    if strategie not in donnees["resultats"]:
        raise KeyError(strategie)
    traj, df = donnees["resultats"][strategie], donnees["cycle_df"]
    return donnees, df, traj, min(len(df), len(traj["P_EB"]))


def courbes(identifiant: str, strategie: str) -> dict:
    """Courbes d'une stratégie sur le cycle, avec la phrase affichée au survol de chaque instant."""
    donnees, df, traj, n = _trajectoire(identifiant, strategie)
    temps_min = _temps(df, n) / 60.0
    p_dem = df["hasPower"].to_numpy(dtype=float)[:n]
    p_eb = np.asarray(traj["P_EB"], dtype=float)[:n]
    p_pb = np.asarray(traj["P_PB"], dtype=float)[:n]
    p_conv = (core.V_EB_PACK_NOM - core.V_PB_PACK_NOM) * np.asarray(traj["I_EB"], dtype=float)[:n]
    soc_eb = np.asarray(traj["SOC_EB"], dtype=float)
    soc_pb = np.asarray(traj["SOC_PB"], dtype=float)

    pas = pertes_par_pas(traj)
    heures = core.DT_SECONDS / 3600.0
    saut = max(1, n // NB_POINTS_PERTES)
    cumul = {k: np.cumsum(v[:n]) * heures for k, v in pas.items()}
    total = cumul["eb"] + cumul["pb"] + cumul["convertisseur"]
    instantane = pas["eb"][:n] + pas["pb"][:n] + pas["convertisseur"][:n]

    return {
        "temps_min": liste(temps_min, 4),
        "demande_kw": liste(p_dem / 1000.0), "eb_kw": liste(p_eb / 1000.0), "pb_kw": liste(p_pb / 1000.0),
        "conv_kw": liste(p_conv / 1000.0),
        "soc_temps_min": liste(np.arange(len(soc_eb)) * core.DT_SECONDS / 60.0, 4),
        "soc_eb_pct": liste(soc_eb * 100.0, 2), "soc_pb_pct": liste(soc_pb * 100.0, 2),
        "pertes": {
            "temps_min": liste(temps_min[::saut], 4),
            "eb_wh": liste(cumul["eb"][::saut], 1), "pb_wh": liste(cumul["pb"][::saut], 1),
            "conv_wh": liste(cumul["convertisseur"][::saut], 1), "total_wh": liste(total[::saut], 1),
            "eb_w": liste(pas["eb"][:n][::saut], 0), "pb_w": liste(pas["pb"][:n][::saut], 0),
            "conv_w": liste(pas["convertisseur"][:n][::saut], 0), "total_w": liste(instantane[::saut], 0),
        },
        "lectures": {
            "repartition": textes_compacts(lecture.lire_repartition(p_dem, traj, n)),
            "soc": textes_compacts(lecture.lire_soc(traj)),
            "eb": textes_compacts(lecture.lire_composant(p_eb, core.P_EB_MAX_W, core.P_EB_MIN_W, core.EPS_POWER_W)),
            "pb": textes_compacts(lecture.lire_composant(p_pb, core.P_PB_MAX_W, core.P_PB_MIN_W, core.EPS_POWER_W)),
            "conv": textes_compacts(lecture.lire_composant(p_conv, core.P_CONV_MAX_W, core.P_CONV_MIN_W, core.EPS_POWER_W * 0.1)),
        },
    }


def indicateurs(identifiant: str, strategie: str) -> dict:
    """Métriques M1 à M6, synthèse rédigée, bilan de charge/décharge et pertes d'une stratégie."""
    donnees, df, traj, n = _trajectoire(identifiant, strategie)
    m = calculer_metriques({"resultats": {strategie: traj}, "cycle_df": df})[strategie]
    nom = nom_affichage(strategie)
    p_dem = df["hasPower"].to_numpy(dtype=float)[:n]
    p_eb = np.asarray(traj["P_EB"], dtype=float)[:n]
    p_pb = np.asarray(traj["P_PB"], dtype=float)[:n]
    p_conv = (core.V_EB_PACK_NOM - core.V_PB_PACK_NOM) * np.asarray(traj["I_EB"], dtype=float)[:n]
    soc_eb, soc_pb = np.asarray(traj["SOC_EB"], dtype=float), np.asarray(traj["SOC_PB"], dtype=float)
    traction = p_dem > core.EPS_POWER_W
    violations = int(m["nb_violations"] + m["nb_violations_courant"])
    non_fourni = float(m["energie_non_servie_wh"])
    respectees = violations == 0 and non_fourni < 1

    # Synthèse rédigée (tableau de bord)
    part_pb = float(np.sum(p_pb[traction])) / float(np.sum(p_dem[traction])) * 100 if traction.any() else 0.0
    temps_pb = float(np.mean(p_pb[traction] > 100.0)) * 100 if traction.any() else 0.0
    distance = float(np.sum(df["speed"].to_numpy(dtype=float)[:n])) * core.DT_SECONDS / 1000 if "speed" in df.columns else float("nan")
    synthese = [
        tr(
            "Sur ce cycle ({d} min, {km} km), la demande atteint {pmax} kW en traction et {pmin} kW au freinage.",
            "Over this cycle ({d} min, {km} km), demand reaches {pmax} kW in traction and {pmin} kW when braking.",
            d=nombre(n * core.DT_SECONDS / 60, 0), km=nombre(distance, 0),
            pmax=nombre(p_dem.max() / 1000, 1), pmin=nombre(p_dem.min() / 1000, 1),
        ),
        tr(
            "{s} confie **{part} %** de l'énergie de traction à la batterie Puissance, qui intervient "
            "pendant {t} % du temps de traction.",
            "{s} gives **{part} %** of the traction energy to the Power battery, which is used during "
            "{t} % of the traction time.",
            s=nom, part=nombre(part_pb, 0), t=nombre(temps_pb, 0),
        ),
        tr(
            "L'écart entre les deux SOC atteint au plus **{max} points** (écart quadratique moyen "
            "{rms} points) ; les SOC finaux sont de {eb} % (Énergie) et {pb} % (Puissance).",
            "The gap between the two SOCs reaches at most **{max} points** (root-mean-square gap "
            "{rms} points); the final SOCs are {eb} % (Energy) and {pb} % (Power).",
            max=nombre(m["delta_soc_max"] * 100, 1), rms=nombre(m["rmse_delta_soc"] * 100, 1),
            eb=nombre(soc_eb[-1] * 100, 1), pb=nombre(soc_pb[-1] * 100, 1),
        ),
        tr(
            "Aucun dépassement de SOC ni de courant, et toute la puissance demandée a été fournie.",
            "No SOC or current limit was exceeded, and all the demanded power was supplied.",
        ) if respectees else tr(
            "**Attention** : {v} dépassement(s) de limite et {e} Wh de demande non fournie ; les autres "
            "indicateurs de cette stratégie sont donc flattés.",
            "**Warning**: {v} limit violation(s) and {e} Wh of demand not supplied; the other "
            "indicators of this strategy are therefore flattering.",
            v=violations, e=nombre(non_fourni, 0),
        ),
        tr(
            "Rendement estimé du HESS : **{r} %** ({p} Wh de pertes estimées, dont {c} Wh dans le convertisseur).",
            "Estimated HESS efficiency: **{r} %** ({p} Wh of estimated losses, including {c} Wh in the converter).",
            r=nombre(m["rendement_hess"] * 100, 2), p=nombre(m["pertes_totales_wh"], 0), c=nombre(m["pertes_convertisseur_wh"], 0),
        ),
    ]

    avertissement = None
    if non_fourni >= 1.0:
        avertissement = tr(
            "{s} ne fournit pas toute la puissance demandée ({e} Wh manquants, jusqu'à {p} kW à un "
            "instant) : son énergie consommée et son rendement en sont flattés.",
            "{s} does not deliver all the demanded power ({e} Wh missing, up to {p} kW at one time "
            "step): its energy consumption and efficiency are flattered as a result.",
            s=nom, e=nombre(non_fourni, 0), p=nombre(m["ecart_puissance_max_kw"], 1),
        )

    # Charge et décharge de chaque composant
    heures = core.DT_SECONDS / 3600.0
    composants = []
    for cle, puissance, lim_dech, lim_rech, seuil in (
        ("eb", p_eb, core.P_EB_MAX_W, core.P_EB_MIN_W, core.EPS_POWER_W),
        ("pb", p_pb, core.P_PB_MAX_W, core.P_PB_MIN_W, core.EPS_POWER_W),
        ("conv", p_conv, core.P_CONV_MAX_W, core.P_CONV_MIN_W, core.EPS_POWER_W * 0.1),
    ):
        composants.append({
            "cle": cle,
            "energie_decharge_wh": float(np.clip(puissance, 0, None).sum() * heures),
            "energie_recharge_wh": float(-np.clip(puissance, None, 0).sum() * heures),
            "temps_decharge": float(np.mean(puissance > seuil)),
            "temps_recharge": float(np.mean(puissance < -seuil)),
            "p_max_kw": float(puissance.max() / 1000.0), "p_min_kw": float(puissance.min() / 1000.0),
            "limite_decharge_kw": lim_dech / 1000.0, "limite_recharge_kw": lim_rech / 1000.0,
        })

    # Pertes et réserve d'énergie en fin de cycle
    pas = pertes_par_pas(traj)
    pertes = {k: float(np.sum(v[:n]) * heures) for k, v in pas.items()}
    total = sum(pertes.values())
    reserve = (
        (float(soc_eb[-1]) - core.SOC_EB_MIN) * core.ENERGY_EB_WH + (float(soc_pb[-1]) - core.SOC_PB_MIN) * core.ENERGY_PB_WH
    )
    if donnees["meta"].get("pertes_dans_soc"):
        bilan = ("info", tr(
            "Les pertes sont comptées dans cette simulation : il reste {r} Wh utilisables en fin de cycle.",
            "Losses are counted in this simulation: {r} Wh remain usable at the end of the cycle.",
            r=nombre(reserve, 0),
        ))
    elif reserve - total >= 0:
        bilan = ("succes", tr(
            "Cette simulation ne compte pas les pertes dans les états de charge. Il reste {r} Wh "
            "utilisables en fin de cycle, plus que les {t} Wh de pertes estimées : en les comptant, la "
            "stratégie finirait le cycle ({m} Wh de marge).",
            "This simulation does not count the losses in the states of charge. {r} Wh remain usable at "
            "the end of the cycle, more than the {t} Wh of estimated losses: counting them, the strategy "
            "would still complete the cycle ({m} Wh margin).",
            r=nombre(reserve, 0), t=nombre(total, 0), m=nombre(reserve - total, 0),
        ))
    else:
        bilan = ("alerte", tr(
            "Cette simulation ne compte pas les pertes dans les états de charge. Il reste {r} Wh "
            "utilisables en fin de cycle, moins que les {t} Wh de pertes estimées : en les comptant, il "
            "manquerait environ {m} Wh. Pour le vérifier, relancez la simulation en incluant les pertes "
            "(page « Lancer une simulation »).",
            "This simulation does not count the losses in the states of charge. {r} Wh remain usable at "
            "the end of the cycle, less than the {t} Wh of estimated losses: counting them, about {m} Wh "
            "would be missing. To check, run the simulation again with losses included (“Run a "
            "simulation” page).",
            r=nombre(reserve, 0), t=nombre(total, 0), m=nombre(total - reserve, 0),
        ))

    return {
        "strategie": fiche_strategie(strategie),
        "metriques": {k: nombre_json(m[k]) for k in (
            "energie_km_wh", "energie_consommee_wh", "rendement_hess", "rmse_delta_soc", "delta_soc_max",
            "pertes_convertisseur_wh", "pertes_totales_wh", "nb_violations", "nb_violations_courant",
            "rmse_puissance_kw", "ecart_puissance_max_kw", "energie_non_servie_wh",
        )},
        "demande_max_kw": float(p_dem.max() / 1000.0),
        "demande_moyenne_traction_kw": float(p_dem[traction].mean() / 1000.0) if traction.any() else 0.0,
        "soc_eb_final": float(soc_eb[-1]), "soc_pb_final": float(soc_pb[-1]),
        "contraintes_respectees": respectees,
        "synthese": synthese,
        "avertissement": avertissement,
        "composants": composants,
        "pertes": {"eb_wh": pertes["eb"], "pb_wh": pertes["pb"], "conv_wh": pertes["convertisseur"], "total_wh": total,
                   "bilan": {"type": bilan[0], "texte": bilan[1]}},
        "hypotheses_pertes": {
            "r_eb_mohm": core.CELL_EB_RINT_OHM * 1000, "eb_serie": core.CELL_EB_N_SERIE, "eb_parallele": core.CELL_EB_N_PARALLELE,
            "r_pb_mohm": core.CELL_PB_RINT_OHM * 1000, "pb_serie": core.CELL_PB_N_SERIE, "pb_parallele": core.CELL_PB_N_PARALLELE,
        },
        "part_convertisseur": (core.V_EB_PACK_NOM - core.V_PB_PACK_NOM) / core.V_EB_PACK_NOM,
    }


# Comparaison des stratégies et conclusion

CLES_METRIQUES = [
    "energie_km_wh", "rendement_hess", "rmse_delta_soc", "pertes_convertisseur_wh", "violations_totales",
    "rmse_puissance_kw", "nb_violations", "nb_violations_courant", "ecart_puissance_max_kw", "energie_non_servie_wh",
]


def _metriques_du_cycle(donnees):
    m = calculer_metriques({"resultats": donnees["resultats"], "cycle_df": donnees["cycle_df"]})
    for v in m.values():
        v["violations_totales"] = v["nb_violations"] + v["nb_violations_courant"]
    return m


def _raison(non_fourni, depassements):
    morceaux = []
    if non_fourni >= 1.0:
        morceaux.append(tr("{e} Wh non fournis", "{e} Wh not delivered", e=nombre(non_fourni, 0)))
    if depassements:
        morceaux.append(tr("{d} dépassement(s) de limite", "{d} limit violation(s)", d=depassements))
    return ", ".join(morceaux) or tr("demande non suivie", "demand not followed")


def _enumerer(cles):
    noms = [nom_affichage(c) for c in cles]
    if len(noms) <= 1:
        return "".join(noms)
    return ", ".join(noms[:-1]) + tr(" et ", " and ") + noms[-1]


def _score(valeur):
    return nombre(valeur, 2, signe=True)


def _conclusion_generale(liste_criteres, avec_explicabilite):
    """Conclusion sur les deux cycles de référence ; None s'ils ne sont pas tous deux disponibles."""
    presents = [c for c, (_, chemin) in CYCLES_REFERENCE.items() if Path(chemin).exists()]
    if len(presents) < 2:
        return None
    evaluations = {}
    for c in presents:
        reference = charger_reference(CYCLES_REFERENCE[c][1])
        evaluations[c] = verdict.completer(_metriques_du_cycle(reference), reference.get("coherence"))
    bilan = verdict.conclure(evaluations, liste_criteres)
    communes = sorted(bilan["communes"], key=lambda n: -bilan["score"][n])
    toutes = [c for c in core.MODEL_ORDER if c in next(iter(evaluations.values()))]

    lignes = []
    for n in communes + [x for x in toutes if x not in communes]:
        cellules = []
        for c in presents:
            ecart = verdict.ecartees(evaluations[c]).get(n)
            if ecart:
                cellules.append(tr("écartée ({r})", "set aside ({r})", r=_raison(*ecart)))
            elif n in communes:
                cellules.append(_score(bilan["par_evaluation"][c]["score"][n]))
            else:
                cellules.append(tr("demande fournie", "demand met"))
        lignes.append({
            "strategie": fiche_strategie(n), "cellules": cellules,
            "score_moyen": nombre_json(bilan["score"][n]) if n in communes else None,
            "en_tete": nombre_json(bilan["en_tete"][n]) if n in communes else None,
        })

    if not communes:
        return {"cycles": [libelle_cycle(c) for c in presents], "lignes": lignes, "type": "alerte", "texte": tr(
            "Aucune stratégie ne fournit toute la demande sur les deux cycles : pas de conclusion générale.",
            "No strategy delivers the whole demand on both cycles: no overall conclusion.",
        )}

    en_tete = verdict.meilleures(bilan["score"])
    premiere = en_tete[0]
    part = bilan["en_tete"][premiere]
    phrases = []
    if len(en_tete) == 1:
        phrases.append(tr(
            "**{s} est la stratégie à retenir** sur l'ensemble des critères : parmi les {k} "
            "stratégies qui fournissent toute la demande sur les deux cycles ({l}), elle obtient "
            "le meilleur score moyen ({sc}) et arrive en tête dans {p} % des pondérations.",
            "**{s} is the strategy to choose** over all criteria: among the {k} strategies that "
            "deliver the whole demand on both cycles ({l}), it has the best mean score ({sc}) and "
            "comes first in {p} % of the weightings.",
            s=nom_affichage(premiere), k=len(communes), l=_enumerer(communes),
            sc=_score(bilan["score"][premiere]), p=nombre(part * 100, 0),
        ))
    else:
        phrases.append(tr(
            "**{l} arrivent à égalité** sur l'ensemble des critères (scores moyens {sc}), parmi "
            "les {k} stratégies qui fournissent toute la demande sur les deux cycles.",
            "**{l} are tied** over all criteria (mean scores {sc}), among the {k} strategies that "
            "deliver the whole demand on both cycles.",
            l=_enumerer(en_tete), sc=" / ".join(_score(bilan["score"][n]) for n in en_tete), k=len(communes),
        ))
    if len(communes) > 1 and len(en_tete) == 1:
        seconde = communes[1]
        phrases.append(tr(
            "{s} suit (score {sc}, en tête dans {p} % des pondérations).",
            "{s} comes next (score {sc}, first in {p} % of the weightings).",
            s=nom_affichage(seconde), sc=_score(bilan["score"][seconde]), p=nombre(bilan["en_tete"][seconde] * 100, 0),
        ))
        phrases.append(
            tr("Ce résultat est solide : il tient quelle que soit l'importance donnée à chaque critère, ou presque.",
               "This result is solid: it holds whatever importance is given to each criterion, or nearly so.")
            if part >= 0.70 else
            tr("Ce résultat dépend de l'importance donnée à chaque critère : l'écart avec la suivante est faible.",
               "This result depends on the importance given to each criterion: the gap with the next one is small.")
        )
    if avec_explicabilite:
        perf = verdict.conclure(evaluations, verdict.criteres(False))
        tete_perf = verdict.meilleures(perf["score"])
        phrases.append(tr(
            "Sur les seules performances (M1 à M4), {l} est en tête dans {p} % des pondérations.",
            "On performance alone (M1 to M4), {l} comes first in {p} % of the weightings.",
            l=_enumerer(tete_perf), p=nombre(sum(perf["en_tete"][n] for n in tete_perf) * 100, 0),
        ))
    hors = [n for n in toutes if n not in communes]
    if len(hors) > 1:
        phrases.append(tr(
            "{l} ne sont pas retenues : elles ne fournissent pas toute la demande sur au moins un des deux cycles.",
            "{l} are not retained: they do not deliver the whole demand on at least one of the two cycles.",
            l=_enumerer(hors),
        ))
    elif hors:
        phrases.append(tr(
            "{l} n'est pas retenue : elle ne fournit pas toute la demande sur au moins un des deux cycles.",
            "{l} is not retained: it does not deliver the whole demand on at least one of the two cycles.",
            l=_enumerer(hors),
        ))
    return {"cycles": [libelle_cycle(c) for c in presents], "lignes": lignes, "type": "succes", "texte": " ".join(phrases)}


def comparaison(identifiant: str, avec_explicabilite: bool = True) -> dict:
    """Métriques de toutes les stratégies, explicabilité, et conclusion tous critères confondus."""
    donnees = obtenir(identifiant)
    resultats = donnees["resultats"]
    noms = [c for c in core.MODEL_ORDER if c in resultats]
    metriques = _metriques_du_cycle(donnees)
    coherences = donnees["coherence"]
    liste_criteres = verdict.criteres(avec_explicabilite)

    sur_ce_cycle = verdict.evaluer(verdict.completer(metriques, coherences), liste_criteres)
    classement = sorted(sur_ce_cycle["retenues"], key=lambda n: -sur_ce_cycle["score"][n])

    return {
        "cycle": nom_du_cycle(identifiant),
        "strategies": [fiche_strategie(c) for c in noms],
        "metriques": {c: {k: nombre_json(metriques[c][k]) for k in CLES_METRIQUES} for c in noms},
        "paires": [[seul, ns] for seul, ns in PAIRES_SYMBOLIQUE if seul in metriques and ns in metriques],
        "explicabilite": {
            c: {
                "e1": lib(xai.TRANSPARENCE[c][1]) if c in xai.TRANSPARENCE else "—",
                "e2": lib(xai.TRANSPARENCE[c][2]) if c in xai.TRANSPARENCE else "—",
                "e3": nombre_json(coherences[c][0]) if c in coherences else None,
                "detail": [{"test": xai.libelle_contrainte(k), "part": nombre_json(v)} for k, v in coherences[c][1].items()]
                if c in coherences else [],
            }
            for c in noms
        },
        "e3_disponible": all(c in coherences for c in noms),
        "sur_ce_cycle": {
            "ecartees": [{"strategie": fiche_strategie(c), "raison": _raison(*r)} for c, r in sur_ce_cycle["ecartees"].items()],
            "classement": [
                {
                    "strategie": fiche_strategie(c),
                    "score": nombre_json(sur_ce_cycle["score"][c]),
                    "en_tete": nombre_json(sur_ce_cycle["en_tete"][c]),
                    "points_forts": ", ".join(
                        verdict.libelle(k).split(" · ")[0] for k in verdict.points_forts(metriques, c, classement, liste_criteres)
                    ) or "—",
                }
                for c in classement
            ],
        },
        "conclusion": _conclusion_generale(liste_criteres, avec_explicabilite),
        "criteres": [
            {"libelle": verdict.libelle(c), "sens": c["sens"], "seuil": lib(c["seuil_txt"])} for c in liste_criteres
        ],
    }


def mesurer_coherence(identifiant: str) -> dict:
    """Mesure la cohérence physique (E3) des stratégies d'une simulation, et la garde avec ses résultats."""
    donnees = obtenir(identifiant)
    for cle, traj in donnees["resultats"].items():
        if cle not in donnees["coherence"]:
            donnees["coherence"][cle] = xai.coherence_physique(cle, donnees["cycle_df"], traj)
    return {"e3": {c: nombre_json(v[0]) for c, v in donnees["coherence"].items()}}
