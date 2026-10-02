"""
api/connaissances.py — Données des pages « Fonctionnement des stratégies EMS » et
« Base de connaissances » (ontologie OntoHESS, ses règles, test d'une situation).
"""

import numpy as np

import ems_core as core
from api.analyse import fiche_strategie
from api.donnees import nombre_json, obtenir
from core import ontology_explainer as ox
from core import presentation
from core.format import nombre
from core.i18n import lib, tr
from core.resultats import CYCLES_REFERENCE


def _chaine(etapes):
    return [{"texte": texte, "role": role} for texte, role in etapes]


def strategies() -> dict:
    """Principe commun, fiche de chaque stratégie et tableau de synthèse."""
    fiches = presentation.fiches_strategies()
    synthese = presentation.synthese_strategies()
    return {
        "principe": _chaine(presentation.principe_commun()),
        "architecture": presentation.architecture_electrique(),
        "reference": presentation.REFERENCE_ARTICLE,
        "fiches": [
            {**fiche_strategie(cle), **{k: v for k, v in fiches[cle].items() if k != "flux"}, "flux": _chaine(fiches[cle]["flux"]),
             "neuro_symbolique": "neurosymbolic" in cle, "synthese": synthese[cle]}
            for cle in core.MODEL_ORDER if cle in fiches
        ],
    }


def _detail_regle(regle):
    type_regle, lecture = ox.lire_regle(regle)

    def sujet(s):
        return lib(presentation.SUJETS_REGLES[s]) if s in presentation.SUJETS_REGLES else s

    return {
        "id": regle["id"],
        "type": type_regle,
        "role": lib(presentation.TYPES_REGLES[type_regle]),
        "lecture": lecture,
        "phrase": ox.regle_en_phrase(regle),
        "sapplique_a": ", ".join(ox.nom_classe(c) for c in regle["classes"]) or "—",
        "grandeurs_lues": [
            f"`{args[-1]}` : {ox.nom_clair(p)} ({sujet(args[0])})" for p, args in regle["lectures"] if len(args) == 2
        ],
        "conditions": [f"`{c}`" for c in ox.conditions_en_clair(regle)],
        "calculs": [f"`{c}`" for c in ox.calculs_en_clair(regle)],
        "conclusions": [
            f"{ox.nom_clair(p)} ({sujet(args[0])}) = `{args[1]}`" for p, args in regle["affectations"] if len(args) == 2
        ],
        "appliquee": type_regle in ("mode", "repartition"),
    }


def ontologie() -> dict:
    """Contenu de l'ontologie OntoHESS, présenté dans l'ordre de la page."""
    regles = ox.charger_regles()
    relations, attributs, individus = ox.vocabulaire_ontologie()
    if not regles or not individus:
        return {"disponible": False}
    hierarchie = ox.hierarchie_classes()
    categories = presentation.categories_ontologie()
    details = [_detail_regle(r) for r in regles]

    # Les règles de répartition décrivent-elles le modèle physique ? (vérifié sur le cycle Artemis)
    verification = None
    try:
        reference = obtenir(next(iter(CYCLES_REFERENCE)))
        traj = reference["resultats"].get("EMS_power_limitation")
        if traj is not None:
            df = reference["cycle_df"]
            n = min(len(df), len(traj["alpha_requested"]))
            onto = ox.alpha_ontologie_vect(df["hasPower"].to_numpy(dtype=float)[:n], np.asarray(traj["SOC_EB"], float)[:n])
            m = ~np.isnan(onto)
            ecart = np.abs(np.asarray(traj["alpha_requested"], float)[:n][m] - onto[m])
            verification = tr(
                "Les règles de répartition R13 à R17 décrivent exactement le modèle physique : sur les {n} "
                "instants du cycle où elles s'appliquent, la répartition qu'elles prescrivent et celle du "
                "modèle physique diffèrent au plus de {e} point.",
                "The split rules R13 to R17 describe the physical model exactly: over the {n} time steps of "
                "the cycle where they apply, the split they prescribe and that of the physical model differ by "
                "at most {e} point.",
                n=nombre(m.sum(), 0), e=nombre(ecart.max() * 100, 2),
            )
    except Exception:  # noqa: BLE001 - résultats de référence absents : la page reste lisible
        verification = None

    objets = ox.individus_ontologie()
    proprietes_eb = dict(next((p for nom, _, p in objets if nom == "batteryE1"), []))
    i_max_onto = proprietes_eb.get("iEB_max_value")
    i_max_sim = core.P_EB_MAX_W / core.V_EB_PACK_NOM
    avertissement = None
    if i_max_onto is not None and abs(float(i_max_onto) - i_max_sim) > 0.01 * i_max_sim:
        avertissement = tr(
            "Écart entre l'ontologie et la simulation : le courant maximal de la batterie Énergie vaut "
            "{a} A dans l'ontologie, alors que la simulation la limite à {b} A ({p} kW).",
            "Mismatch between the ontology and the simulation: the Energy battery's maximum current is "
            "{a} A in the ontology, whereas the simulation limits it to {b} A ({p} kW).",
            a=nombre(float(i_max_onto), 2), b=nombre(i_max_sim, 1), p=nombre(core.P_EB_MAX_W / 1000, 1),
        )

    usages = presentation.usages_ontologie()
    return {
        "disponible": True,
        "compteurs": {
            "concepts": len(ox.classes_ontologie()), "relations": len(relations), "proprietes": len(attributs),
            "objets": len(individus), "regles": len(regles),
        },
        "categories": [
            {"cle": racine, "nom": categories.get(racine, (racine, ""))[0], "description": categories.get(racine, (racine, ""))[1],
             "concepts": [{"nom": c, "clair": ox.nom_classe(c) if c in ox.CLASSES else None} for c in enfants]}
            for racine, enfants in hierarchie.items()
        ],
        "objets": [
            {"nom": nom, "concepts": ", ".join(types), "proprietes": " ; ".join(f"{p} = {v}" for p, v in props)}
            for nom, types, props in objets
        ],
        "avertissement": avertissement,
        "relations": [{"nom": r, "de": ", ".join(d) or "—", "vers": ", ".join(p) or "—"} for r, d, p in ox.relations_ontologie()],
        "proprietes": list(attributs),
        "regles_lues": {
            "mode": [{"id": d["id"], "phrase": d["phrase"]} for d in details if d["type"] == "mode"],
            "repartition": [{"id": d["id"], "phrase": d["phrase"]} for d in details if d["type"] == "repartition"],
        },
        "autres_regles": [d["id"] for d in details if d["type"] not in ("mode", "repartition")],
        "verification": verification,
        "regles": details,
        "roles": [{"cle": cle, "nom": lib(nom)} for cle, nom in presentation.TYPES_REGLES.items()],
        "glossaire": [{"symbole": v, "sens": s} for v, s in ox.glossaire()],
        "regles_floues": [
            {**{k: r[k] for k in ("cle", "libelle", "si", "alpha", "sens")}, "concepts": [nom for nom, _ in r["concepts"]]}
            for r in ox.regles_floues()
        ],
        "termes_flous": [{"terme": terme, "definition": definition} for terme, definition in ox.termes_flous()],
        "alpha_par_defaut": core.FUZZY_DEFAULT_ALPHA,
        "usages": [{**fiche_strategie(cle), "usage": usage} for cle, usage in usages.items()],
        "chaines": [
            {"titre": titre, "chaine": _chaine(etapes), "legende": legende}
            for titre, etapes, legende in presentation.chaines_neuro_symboliques().values()
        ],
    }


def tester_regles(p_kw: float, soc_eb: float, soc_pb: float) -> dict:
    """Ce que l'ontologie déduit d'une situation choisie : état, règles appliquées, répartition de référence."""
    p = p_kw * 1000.0
    etat = ox.etat_fonctionnement(p)
    activees, non_activees, _ = ox.evaluer_regles(p, soc_eb, soc_pb)
    repartition = ox.repartition_ontologie(p, soc_eb, soc_pb)
    et = tr(" et ", " and ")

    if repartition is not None:
        a = repartition["alpha"]
        conclusion = {"type": "succes", "texte": tr(
            "Répartition de référence (règle {i}) : batterie Énergie {eb} kW, batterie Puissance {pb} kW (alpha = {a} %).",
            "Reference split (rule {i}): Energy battery {eb} kW, Power battery {pb} kW (alpha = {a} %).",
            i=repartition["regle"]["id"], eb=nombre(p * (1 - a) / 1000, 1), pb=nombre(p * a / 1000, 1), a=nombre(a * 100, 0),
        )}
    else:
        conclusion = {"type": "info", "texte": tr(
            "Demande quasi nulle : aucune règle de répartition ne s'applique.", "Near-zero demand: no split rule applies.",
        )}
    return {
        "etat": {"cle": etat, "libelle": ox.libelle_etat(etat)},
        "etats": [{"cle": c, "libelle": ox.libelle_etat(c, court=True)} for c in ox.ETATS_ONTOLOGIE],
        "regles": [
            tr("**{i}** — si {p}, alors {l}.", "**{i}** — if {p}, then {l}.",
               i=r["id"], p=et.join(f"`{d['texte']}`" for d in r["details"]), l=r["lecture"])
            for r in sorted((r for r in activees if r["type"] in ("mode", "repartition")), key=lambda r: r["type"])
        ],
        "conclusion": conclusion,
        "alpha": nombre_json(repartition["alpha"]) if repartition is not None else None,
        "non_appliquees": [
            tr("**{i}** ({l}) : `{e}` n'est pas vérifié", "**{i}** ({l}): `{e}` does not hold",
               i=r["id"], l=r["lecture"], e=" ; ".join(d["texte"] for d in r["details"] if d["ok"] is False))
            for r in non_activees if r["type"] in ("mode", "repartition")
        ],
    }
