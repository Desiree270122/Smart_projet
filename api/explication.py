"""
api/explication.py — Données de la page « Pourquoi cette décision ? » : pour un
instant du cycle et une stratégie, la décision, ses raisons en clair et le détail
de l'explication ; puis le bilan sur tout le cycle.

L'explication est exacte quand le calcul de la stratégie est lisible (règles),
reconstruite après coup quand il ne l'est pas (réseaux de neurones).
"""

import warnings
from functools import lru_cache

import numpy as np
import torch

import ems_core as core
from api.analyse import fiche_strategie
from api.donnees import liste, nombre_json, obtenir
from core import ontology_explainer as ox
from core import xai
from core.format import nombre
from core.i18n import lib, tr
from core.resultats import nom_affichage

# Nature de l'explication : (exacte ?, texte).
_RECONSTRUITE = (
    "Le calcul du réseau de neurones n'est pas lisible : l'explication est reconstruite après coup, "
    "en mesurant ce que chaque grandeur apporte à la décision.",
    "The neural network's calculation cannot be read: the explanation is reconstructed afterwards, "
    "by measuring what each quantity contributes to the decision.",
)
NATURE = {
    "EMS_power_limitation": (True, (
        "La règle appliquée est connue : l'explication est le calcul lui-même.",
        "The rule applied is known: the explanation is the calculation itself.",
    )),
    "EMS_fuzzy_logic": (True, (
        "Les règles actives et leur poids sont le calcul lui-même.",
        "The active rules and their weights are the calculation itself.",
    )),
    "EMS_MLP_neurosymbolic": (True, (
        "La décision se décompose exactement en règles + correction du réseau + garde-fou ; seule la "
        "correction, limitée à ±20 points, n'est pas lisible.",
        "The decision splits exactly into rules + network correction + safeguard; only the "
        "correction, limited to ±20 points, cannot be read.",
    )),
    "EMS_MLP": (False, _RECONSTRUITE),
    "EMS_LSTM": (False, _RECONSTRUITE),
    "EMS_GNN": (False, _RECONSTRUITE),
    "EMS_LSTM_neurosymbolic": (False, (
        _RECONSTRUITE[0] + " Les états déduits par l'ontologie y ont un sens physique.",
        _RECONSTRUITE[1] + " The states inferred by the ontology have a physical meaning in it.",
    )),
}

NOMS_COMPOSANTS = {
    "energy_battery": ("Batterie Énergie", "Energy battery"),
    "power_battery": ("Batterie Puissance", "Power battery"),
    "converter": ("Convertisseur", "Converter"),
    "motor": ("Moteur", "Motor"),
    "vehicle": ("Véhicule", "Vehicle"),
}

SCENARIOS = {
    ("SOC_PB", +1): ("SOC de la batterie Puissance + 5 points", "Power battery SOC + 5 points"),
    ("SOC_PB", -1): ("SOC de la batterie Puissance − 5 points", "Power battery SOC − 5 points"),
    ("SOC_EB", +1): ("SOC de la batterie Énergie + 5 points", "Energy battery SOC + 5 points"),
    ("SOC_EB", -1): ("SOC de la batterie Énergie − 5 points", "Energy battery SOC − 5 points"),
    ("hasPower", +1): ("Demande + 1 kW", "Demand + 1 kW"),
    ("hasPower", -1): ("Demande − 1 kW", "Demand − 1 kW"),
}
# Sens attendu de la puissance de la batterie Puissance quand la grandeur augmente.
SENS_ATTENDU = {"SOC_PB": +1, "SOC_EB": -1, "hasPower": +1}

# Rôle de chaque étape d'une chaîne de décision (l'interface en déduit la couleur).
REFERENCE, SECONDAIRE, DECISION = "reference", "secondaire", "decision"


def _kw(w):
    return f"{nombre(w / 1000.0, 1)} kW"


def _pct(x, decimales=1):
    return f"{nombre(x * 100, decimales)} %"


def _points(x):
    return tr("{v} points", "{v} points", v=nombre(x * 100, 1, signe=True))


def _regle_floue(cle):
    return ox.libelle_regle_floue(cle) if cle in ox.REGLES_FLOUES else tr("Répartition par défaut", "Default split")


def _grandeur(col):
    """Nom d'une grandeur dans une phrase (les sigles gardent leur casse)."""
    nom = xai.libelle_entree(col)
    return nom if nom[:2].isupper() else nom[0].lower() + nom[1:]


def index_de_temps(df, n, t):
    temps = df["time"].to_numpy(dtype=float)[:n] if "time" in df.columns else np.arange(n, dtype=float)
    i = int(np.abs(temps - t).argmin())
    return i, float(temps[i])


@lru_cache(maxsize=512)
def _contributions(identifiant, strategie, instant):
    donnees = obtenir(identifiant)
    return xai.shapley(strategie, donnees["cycle_df"], donnees["resultats"][strategie], instant)


def _decomposition_ns(donnees, instant):
    """(alpha des règles floues, alpha après correction du réseau, garde-fou actif) de NS-MLP."""
    traj = donnees["resultats"]["EMS_MLP_neurosymbolic"]
    if "alpha_reseau" in traj and np.isfinite(traj["alpha_reseau"][instant]):
        return float(traj["alpha_flou"][instant]), float(traj["alpha_reseau"][instant]), bool(traj["garde_fou"][instant])
    d = xai.decomposer_ns_mlp(donnees["cycle_df"], traj, instant)
    return d["alpha_flou"], d["alpha_reseau"], d["garde_fou"]


def _poids_composants_gnn(p_dem, soc_eb, soc_pb, accel):
    """Poids (%) de chaque composant du schéma du HESS dans la décision du GNN."""
    modele, scaler = xai._modele("EMS_GNN")
    x_g, liaisons = core.construire_graphe_instant(p_dem, soc_eb, soc_pb, accel, scaler)
    x_g = x_g.to(core.DEVICE).clone().requires_grad_(True)
    liaisons = liaisons.to(core.DEVICE)
    modele(x_g, liaisons, torch.zeros(x_g.shape[0], dtype=torch.long, device=core.DEVICE)).sum().backward()
    poids = np.abs((x_g.grad * x_g).detach().cpu().numpy()).sum(axis=1)
    poids = poids / poids.sum() * 100.0 if poids.sum() > 0 else poids
    paires = liaisons.detach().cpu().numpy()
    return {
        "composants": [
            {"nom": lib(NOMS_COMPOSANTS[c]) if c in NOMS_COMPOSANTS else c, "poids": nombre_json(p, 1)}
            for c, p in zip(core.GNN_NODE_NAMES, poids)
        ],
        "liaisons": [[int(paires[0, k]), int(paires[1, k])] for k in range(paires.shape[1])],
    }


def explication(identifiant: str, strategie: str, t: float) -> dict:
    """Tout ce que la page affiche pour un instant : décision, raisons, onglets."""
    donnees = obtenir(identifiant)
    resultats, df = donnees["resultats"], donnees["cycle_df"]
    if strategie not in resultats:
        raise KeyError(strategie)
    noms = [c for c in core.MODEL_ORDER if c in resultats]
    n = min([len(df)] + [len(x["P_EB"]) for x in resultats.values()])
    instant, t_sel = index_de_temps(df, n, t)

    traj = resultats[strategie]
    vitesse = float(df["speed"].iloc[instant]) if "speed" in df.columns else 0.0
    accel = float(df["hasAcceleration"].iloc[instant]) if "hasAcceleration" in df.columns else 0.0
    p_dem = float(df["hasPower"].iloc[instant])
    soc_eb, soc_pb = float(traj["SOC_EB"][instant]), float(traj["SOC_PB"][instant])
    p_eb, p_pb = float(traj["P_EB"][instant]), float(traj["P_PB"][instant])
    alpha_final = float(traj["alpha_final"][instant])
    alpha_req = float(traj["alpha_requested"][instant]) if "alpha_requested" in traj else alpha_final
    correction = bool(traj["correction_applied"][instant]) if "correction_applied" in traj else False
    demande_nulle = abs(p_dem) <= core.EPS_POWER_W

    etat = ox.etat_instant(p_dem, soc_eb, soc_pb, p_eb=p_eb)
    onto = ox.repartition_ontologie(p_dem, soc_eb, soc_pb)
    exacte, nature = NATURE.get(strategie, (False, ("", "")))

    res_flou = core.alpha_fuzzy_calc(np.array([soc_eb]), np.array([soc_pb]), np.array([p_dem]), np.array([accel]))
    alpha_flou = float(res_flou["alpha"][0])
    forces = np.asarray(res_flou["strengths"][0], dtype=float)
    contrib_regles = ox.contributions_floues(forces)
    dominante = str(res_flou["dominant_rule"][0])
    avec_contributions = strategie in xai.SHAPLEY_DISPONIBLE and not demande_nulle
    if avec_contributions:
        reference, contribs, alpha_explique = _contributions(identifiant, strategie, instant)
        principales = sorted(contribs, key=lambda kv: -abs(kv[1]))
    entrees_onto = {
        "EMS_MLP_neurosymbolic": core.MLP_NS_INPUT_COLS,
        "EMS_LSTM_neurosymbolic": core.LSTM_NS_FEATURE_COLS,
    }.get(strategie, [])
    etats_transmis = [c for c in ox.LIBELLES_SYMBOLIQUES if c in entrees_onto]

    if strategie == "EMS_MLP_neurosymbolic" and not demande_nulle:
        alpha_flou, alpha_reseau, garde_actif = _decomposition_ns(donnees, instant)
    else:
        alpha_reseau, garde_actif = alpha_req, False
    regle_par_defaut = dominante not in core.FUZZY_RULE_NAMES or float(forces.max(initial=0.0)) < 1e-3
    et = tr(" et ", " and ")

    # Raisons en clair (3 à 5)
    def raisons():
        if demande_nulle:
            return [tr(
                "La demande est quasi nulle : aucune batterie n'est sollicitée, il n'y a pas de répartition à expliquer.",
                "The demand is almost zero: no battery is in use, there is no split to explain.",
            )]
        r = []
        if p_dem > 0:
            r.append(tr(
                "La demande de traction est de **{p}**, {pos} la limite de la batterie Énergie ({l}).",
                "The traction demand is **{p}**, {pos} the Energy battery's limit ({l}).",
                p=_kw(p_dem), l=_kw(core.P_EB_MAX_W),
                pos=tr("au-delà de", "beyond") if p_dem > core.P_EB_MAX_W else tr("dans", "within"),
            ))
        else:
            r.append(tr(
                "Le véhicule freine : **{p}** sont à récupérer{plus} (limite {l}).",
                "The vehicle is braking: **{p}** can be recovered{plus} (limit {l}).",
                p=_kw(-p_dem), l=_kw(-core.P_EB_MIN_W),
                plus=tr(", plus que ce que la batterie Énergie peut absorber", ", more than the Energy battery can absorb")
                if p_dem < core.P_EB_MIN_W else "",
            ))
        ecart = (soc_eb - soc_pb) * 100
        r.append(tr(
            "Le SOC de la batterie Énergie ({a} %) est {c} celui de la batterie Puissance ({b} %).",
            "The Energy battery's SOC ({a} %) is {c} that of the Power battery ({b} %).",
            a=nombre(soc_eb * 100, 0), b=nombre(soc_pb * 100, 0),
            c=tr("supérieur à", "higher than") if ecart > 1 else tr("inférieur à", "lower than") if ecart < -1 else tr("proche de", "close to"),
        ))
        if strategie == "EMS_MLP_neurosymbolic":
            r.append(
                (tr("Aucune règle floue n'est nettement vraie : les règles proposent leur répartition par défaut, ",
                    "No fuzzy rule is clearly true: the rules propose their default split, ")
                 if regle_par_defaut else
                 tr("Les règles floues, menées par **{g}**, proposent ", "The fuzzy rules, led by **{g}**, propose ", g=_regle_floue(dominante)))
                + tr("**{a}** pour la batterie Puissance.", "**{a}** for the Power battery.", a=_pct(alpha_flou))
            )
            r.append(tr(
                "Le réseau de neurones y ajoute une correction limitée de **{c}** (au plus ±{m}).",
                "The neural network adds a limited correction of **{c}** (at most ±{m}).",
                c=_points(alpha_reseau - alpha_flou), m=nombre(core.MLP_NS_MAX_DELTA * 100, 0),
            ))
            if garde_actif:
                r.append(tr(
                    "L'ontologie déclare la batterie Puissance faible (SOC {s} % ≤ {l} %) : le **garde-fou** "
                    "applique les règles R14/R16 d'OntoHESS et ramène sa part à **{a}**.",
                    "The ontology declares the Power battery low (SOC {s} % ≤ {l} %): the **safeguard** "
                    "applies OntoHESS rules R14/R16 and brings its share back to **{a}**.",
                    s=nombre(soc_pb * 100, 0), l=nombre(core.MLP_NS_RESERVE_PB_SOC * 100, 0), a=_pct(alpha_req),
                ))
        elif strategie == "EMS_LSTM_neurosymbolic":
            vrais = [ox.libelle_symbolique(c).lower() for c in etats_transmis if etat["symboliques"].get(c)]
            total = sum(abs(v) for _, v in contribs) or 1.0
            part = sum(abs(v) for c, v in contribs if c in xai.ETATS_SYMBOLIQUES) / total * 100
            r.append(tr(
                "États déduits par l'ontologie et transmis au réseau : {e} ; ils pèsent {p} % de la décision.",
                "States inferred by the ontology and passed to the network: {e}; they account for {p} % of the decision.",
                e=", ".join(vrais) or tr("aucun n'est vrai", "none is true"), p=nombre(part, 0),
            ))
            principale = next(((c, v) for c, v in principales if c not in xai.ETATS_SYMBOLIQUES), ("—", 0.0))
            r.append(tr(
                "Le réseau, qui lit les {w} dernières secondes, s'appuie surtout sur **{g}** ({v}).",
                "The network, which reads the last {w} seconds, relies mostly on **{g}** ({v}).",
                w=core.LSTM_WINDOW, g=_grandeur(principale[0]), v=_points(principale[1]),
            ))
        else:
            r.append(tr("L'ontologie en déduit l'état « {e} ».", "The ontology infers the state “{e}”.", e=etat["libelle"]))
            if strategie == "EMS_power_limitation" and onto is not None:
                r.append(tr(
                    "La règle **{i}** d'OntoHESS s'applique : {l}.", "OntoHESS rule **{i}** applies: {l}.",
                    i=onto["regle"]["id"], l=onto["regle"]["lecture"],
                ))
            elif strategie == "EMS_fuzzy_logic" and regle_par_defaut:
                r.append(tr(
                    "Aucune règle floue n'est nettement vraie : la répartition par défaut s'applique ({a} pour la batterie Puissance).",
                    "No fuzzy rule is clearly true: the default split applies ({a} for the Power battery).",
                    a=_pct(alpha_flou, 0),
                ))
            elif strategie == "EMS_fuzzy_logic":
                i_dom = list(core.FUZZY_RULE_NAMES).index(dominante)
                r.append(tr(
                    "La règle floue qui pèse le plus est **{g}** (vraie à {f} %) : {s}.",
                    "The fuzzy rule that weighs most is **{g}** (true to {f} %): {s}.",
                    g=_regle_floue(dominante), f=nombre(forces[i_dom] * 100, 0), s=ox.sens_regle_floue(dominante),
                ))
            elif avec_contributions:
                fortes = [(_grandeur(c), v) for c, v in principales[:2] if abs(v) >= 5e-4]
                r.append(
                    tr("Le réseau s'appuie surtout sur ", "The network relies mostly on ")
                    + et.join(f"**{g}** ({_points(v)})" for g, v in fortes) + "."
                )
        r.append(
            tr("Le filtre de sécurité a **corrigé** la proposition pour respecter les limites physiques.",
               "The safety filter **corrected** the proposal to respect the physical limits.")
            if correction else
            tr("Le filtre de sécurité a **validé** la décision : les limites physiques sont respectées.",
               "The safety filter **validated** the decision: the physical limits are respected.")
        )
        if len(r) > 5:
            r.pop(1)  # la comparaison des SOC est la raison la moins informative
        return r[:5]

    sortie = {
        "strategie": fiche_strategie(strategie),
        "instant": {"index": instant, "t_s": t_sel, "vitesse_kmh": vitesse * 3.6, "etat": etat["libelle"]},
        "decision": {"demande_w": p_dem, "soc_eb": soc_eb, "soc_pb": soc_pb, "alpha": alpha_final, "p_eb_w": p_eb, "p_pb_w": p_pb},
        "demande_nulle": demande_nulle,
        "raisons": raisons(),
        "nature": {"exacte": exacte, "texte": lib(nature)},
    }

    # Onglet « Décision » : la décision décomposée, et les autres stratégies au même instant
    if not demande_nulle:
        if strategie in ("EMS_fuzzy_logic", "EMS_MLP_neurosymbolic"):
            etapes = (
                [(_regle_floue(core.FUZZY_RULE_NAMES[i]), float(contrib_regles[i]))
                 for i in np.argsort(contrib_regles)[::-1] if contrib_regles[i] > 5e-4]
                if contrib_regles is not None else [(tr("Répartition par défaut", "Default split"), alpha_flou)]
            )
            if strategie == "EMS_MLP_neurosymbolic":
                etapes += [(tr("Correction du réseau", "Network correction"), alpha_reseau - alpha_flou)]
                if garde_actif:
                    etapes += [(tr("Garde-fou de l'ontologie", "Ontology safeguard"), alpha_req - alpha_reseau)]
            elif abs(alpha_req - alpha_flou) > 1e-4:
                etapes += [(tr("Ajustement", "Adjustment"), alpha_req - alpha_flou)]
        elif strategie == "EMS_power_limitation":
            etapes = [(tr("Règle physique", "Physical rule"), alpha_req)]
        else:
            reste = sum(v for _, v in principales[6:])
            etapes = (
                [(tr("Situation moyenne du cycle", "Average situation of the cycle"), reference)]
                + [(xai.libelle_entree(c), v) for c, v in principales[:6] if abs(v) >= 5e-4]
                + ([(tr("Autres grandeurs", "Other quantities"), reste)] if abs(reste) >= 5e-4 else [])
                + ([(tr("Écart résiduel", "Residual gap"), alpha_req - alpha_explique)] if abs(alpha_req - alpha_explique) > 1e-3 else [])
            )
        etapes += [(tr("Filtre de sécurité", "Safety filter"), alpha_final - alpha_req)]

        note = None
        if correction:
            note = {"type": "alerte", "texte": tr(
                "Le filtre de sécurité a corrigé la répartition proposée ({a}) pour respecter les "
                "limites physiques des batteries et du convertisseur.",
                "The safety filter corrected the proposed split ({a}) to respect the physical limits of "
                "the batteries and the converter.",
                a=_pct(alpha_req),
            )}
        elif abs(alpha_final - alpha_req) > 1e-9:
            pas = donnees["meta"].get("pas_alpha") or core.ALPHA_GRID_STEP_DEFAUT
            note = {"type": "note", "texte": tr(
                "Proposé {a}, appliqué {b} : le filtre de sécurité règle la répartition par pas de "
                "{p} % ; un écart plus petit est un arrondi, pas une correction.",
                "Proposed {a}, applied {b}: the safety filter sets the split in steps of {p} %; a "
                "smaller gap is rounding, not a correction.",
                a=_pct(alpha_req), b=_pct(alpha_final), p=nombre(pas * 100, 1),
            )}
        sortie["cascade"] = {
            "etapes": [{"libelle": l, "valeur": nombre_json(v)} for l, v in etapes], "alpha_final": alpha_final, "note": note,
        }
        sortie["autres"] = [
            {**fiche_strategie(c), "alpha": float(resultats[c]["alpha_final"][instant])} for c in noms
        ]
        sortie["regle_onto"] = {"id": onto["regle"]["id"], "alpha": onto["alpha"]} if onto is not None else None

    demi = 150
    i0, i1 = max(0, instant - demi), min(n, instant + demi + 1)
    temps = df["time"].to_numpy(dtype=float) if "time" in df.columns else np.arange(len(df), dtype=float)
    sortie["contexte"] = {
        "t_s": liste(temps[i0:i1], 1),
        "demande_kw": liste(df["hasPower"].to_numpy(dtype=float)[i0:i1] / 1000.0),
        "eb_kw": liste(np.asarray(traj["P_EB"], float)[i0:i1] / 1000.0),
        "pb_kw": liste(np.asarray(traj["P_PB"], float)[i0:i1] / 1000.0),
    }

    # Onglet « Raisons »
    if demande_nulle:
        sortie["raisons_detail"] = {"type": "vide"}
    elif strategie == "EMS_power_limitation":
        phrase = None
        if onto is not None:
            premisses = et.join(f"`{x['texte']}`" for x in onto["regle"]["details"])
            phrase = tr(
                "La décision découle d'une seule règle : **{i}** — si {p}, alors {l}.",
                "The decision follows from a single rule: **{i}** — if {p}, then {l}.",
                i=onto["regle"]["id"], p=premisses, l=onto["regle"]["lecture"],
            )
        sortie["raisons_detail"] = {"type": "regle", "texte": phrase}
    elif strategie in ("EMS_fuzzy_logic", "EMS_MLP_neurosymbolic"):
        regles_f = {r["cle"]: r for r in ox.regles_floues()}
        sortie["raisons_detail"] = {
            "type": "regles_floues",
            "regles": [
                {
                    "regle": regles_f[core.FUZZY_RULE_NAMES[i]]["libelle"],
                    "si": regles_f[core.FUZZY_RULE_NAMES[i]]["si"],
                    "verite": float(forces[i]),
                    "propose": regles_f[core.FUZZY_RULE_NAMES[i]]["alpha"],
                    "contribution": nombre_json(contrib_regles[i]) if contrib_regles is not None else None,
                }
                for i in np.argsort(forces)[::-1] if forces[i] > 5e-4
            ],
        }
    else:
        sortie["raisons_detail"] = {
            "type": "contributions",
            "reference": reference,
            "contributions": [{"grandeur": xai.libelle_entree(c), "valeur": nombre_json(v)} for c, v in principales],
        }

    # Onglet « Connaissances expertes »
    connaissances = {"chaine": None, "titre": None}
    if strategie == "EMS_MLP_neurosymbolic" and not demande_nulle:
        connaissances["titre"] = tr(
            "NS-MLP : les règles décident, le réseau corrige, l'ontologie protège la réserve de la batterie Puissance",
            "NS-MLP: the rules decide, the network corrects, the ontology protects the Power battery's reserve",
        )
        connaissances["chaine"] = (
            [
                {"texte": tr("Règles floues", "Fuzzy rules"), "role": REFERENCE},
                {"texte": tr("alpha des règles {a}", "alpha from the rules {a}", a=_pct(alpha_flou)), "role": DECISION},
                {"texte": tr("correction du réseau {c} pts", "network correction {c} pts",
                             c=nombre((alpha_reseau - alpha_flou) * 100, 1, signe=True)), "role": SECONDAIRE},
            ]
            + ([{"texte": tr("garde-fou R14/R16 : {a}", "safeguard R14/R16: {a}", a=_pct(alpha_req)), "role": REFERENCE}] if garde_actif else [])
            + [{"texte": tr("alpha final {a}", "final alpha {a}", a=_pct(alpha_final)), "role": DECISION}]
        )
    elif strategie == "EMS_LSTM_neurosymbolic" and not demande_nulle:
        vrais = [ox.libelle_symbolique(c) for c in etats_transmis if etat["symboliques"].get(c)]
        connaissances["titre"] = tr("NS-LSTM : l'ontologie informe le réseau", "NS-LSTM: the ontology informs the network")
        connaissances["chaine"] = [
            {"texte": tr("{w} dernières secondes", "Last {w} seconds", w=core.LSTM_WINDOW), "role": REFERENCE},
            {"texte": tr("+ états déduits par l'ontologie : {e}", "+ states inferred by the ontology: {e}",
                         e=", ".join(vrais) or tr("aucun n'est vrai", "none is true")), "role": REFERENCE},
            {"texte": tr("Réseau LSTM", "LSTM network"), "role": SECONDAIRE},
            {"texte": f"alpha {_pct(alpha_final)}", "role": DECISION},
        ]

    oui, non = tr("oui", "yes"), tr("non", "no")
    connaissances["etats_transmis"] = None
    if etats_transmis:
        connaissances["etats_transmis"] = (
            tr("États déduits par l'ontologie et transmis au réseau : ", "States inferred by the ontology and passed to the network: ")
            + ", ".join(f"{ox.libelle_symbolique(c)} **{oui if etat['symboliques'][c] else non}**" for c in etats_transmis)
            + "."
        )
    elif strategie == "EMS_fuzzy_logic":
        concepts = ox.concepts_regle_floue(dominante)
        if concepts and not regle_par_defaut:
            connaissances["etats_transmis"] = (
                tr("Concepts de l'ontologie utilisés par la règle qui pèse le plus : ",
                   "Ontology concepts used by the rule that weighs most: ")
                + ", ".join(nom for nom, _ in concepts) + "."
            )
    connaissances["etat"] = tr("État de fonctionnement déduit : **{e}**.", "Inferred operating state: **{e}**.", e=etat["libelle"])
    activees, non_activees, _ = ox.evaluer_regles(p_dem, soc_eb, soc_pb)
    connaissances["regles"] = [
        tr("**{i}** — si {p}, alors {l}.", "**{i}** — if {p}, then {l}.",
           i=r["id"], p=et.join(f"`{x['texte']}`" for x in r["details"]), l=r["lecture"])
        for r in sorted((r for r in activees if r["type"] in ("mode", "repartition")), key=lambda r: r["type"])
    ]
    connaissances["reference"] = None
    if onto is not None and not demande_nulle:
        ecart = (alpha_final - onto["alpha"]) * 100
        position = (
            tr("identique à", "identical to") if abs(ecart) < 0.5
            else tr("{x} points au-dessus de", "{x} points above", x=nombre(ecart, 1)) if ecart > 0
            else tr("{x} points en dessous de", "{x} points below", x=nombre(-ecart, 1))
        )
        connaissances["reference"] = tr(
            "La règle de répartition {i} prescrit **{a}** pour la batterie Puissance ; la décision de la stratégie est {p} cette référence.",
            "The split rule {i} prescribes **{a}** for the Power battery; the strategy's decision is {p} this reference.",
            i=onto["regle"]["id"], a=_pct(onto["alpha"]), p=position,
        )

    connaissances["comparaison_ns"] = None
    paire = [c for c in ("EMS_MLP_neurosymbolic", "EMS_LSTM_neurosymbolic") if c in resultats]
    if len(paire) == 2 and not demande_nulle:
        cartes, decisions = [], {}
        for cle in paire:
            trj = resultats[cle]
            se, sp = float(trj["SOC_EB"][instant]), float(trj["SOC_PB"][instant])
            a_req_ns, a_fin_ns = float(trj["alpha_requested"][instant]), float(trj["alpha_final"][instant])
            pe, pp = float(trj["P_EB"][instant]), float(trj["P_PB"][instant])
            corr_ns = bool(trj["correction_applied"][instant])
            decisions[cle] = a_fin_ns
            if cle == "EMS_MLP_neurosymbolic":
                rf = core.alpha_fuzzy_calc(np.array([se]), np.array([sp]), np.array([p_dem]), np.array([accel]))
                a_f, a_r, garde_ns = _decomposition_ns(donnees, instant)
                lignes = [
                    tr("Règles floues (menées par {g}) : **{a}**", "Fuzzy rules (led by {g}): **{a}**",
                       g=_regle_floue(str(rf["dominant_rule"][0])), a=_pct(a_f)),
                    tr("Correction du réseau : **{c}**", "Network correction: **{c}**", c=_points(a_r - a_f)),
                ]
                if garde_ns:
                    lignes.append(tr("Garde-fou de l'ontologie : **{c}**", "Ontology safeguard: **{c}**", c=_points(a_req_ns - a_r)))
            else:
                _, contribs_ns, _ = _contributions(identifiant, cle, instant)
                total = sum(abs(v) for _, v in contribs_ns) or 1.0
                part = sum(abs(v) for c, v in contribs_ns if c in xai.ETATS_SYMBOLIQUES) / total * 100
                principale = max(
                    ((c, v) for c, v in contribs_ns if c not in xai.ETATS_SYMBOLIQUES),
                    key=lambda kv: abs(kv[1]), default=("—", 0.0),
                )
                lignes = [
                    tr("États déduits par l'ontologie : **{p} %** de la décision",
                       "States inferred by the ontology: **{p} %** of the decision", p=nombre(part, 0)),
                    tr("Grandeur qui pèse le plus : {g} ({v})", "Quantity that weighs most: {g} ({v})",
                       g=xai.libelle_entree(principale[0]), v=_points(principale[1])),
                ]
            lignes.append(tr(
                "Décision : **{a}** pour la batterie Puissance · EB {pe} · PB {pp}",
                "Decision: **{a}** for the Power battery · EB {pe} · PB {pp}",
                a=_pct(a_fin_ns), pe=_kw(pe), pp=_kw(pp),
            ))
            cartes.append({
                "nom": nom_affichage(cle), "lignes": lignes,
                "verifications": [
                    {"ok": not corr_ns, "texte": tr("acceptée par le filtre de sécurité", "accepted by the safety filter") if not corr_ns
                     else tr("corrigée par le filtre de sécurité", "corrected by the safety filter")},
                    {"ok": se >= core.SOC_EB_MIN and sp >= core.SOC_PB_MIN, "texte": tr("SOC dans leurs limites", "SOCs within their limits")},
                    {"ok": core.P_EB_MIN_W - 1 <= pe <= core.P_EB_MAX_W + 1,
                     "texte": tr("batterie Énergie dans ses limites de puissance", "Energy battery within its power limits")},
                    {"ok": float(trj["P_unserved"][instant]) < 1.0, "texte": tr("demande entièrement fournie", "demand fully met")},
                ],
            })
        connaissances["comparaison_ns"] = {
            "cartes": cartes,
            "ecart": tr(
                "Écart de {e} points entre les deux décisions : NS-MLP suit ses règles puis les corrige "
                "à l'instant présent ; NS-LSTM raisonne sur les {w} dernières secondes, où les états "
                "déduits par l'ontologie ne sont qu'une information parmi d'autres.",
                "Gap of {e} points between the two decisions: NS-MLP follows its rules then corrects "
                "them at the present time; NS-LSTM reasons over the last {w} seconds, where the states "
                "inferred by the ontology are only one piece of information among others.",
                e=nombre(abs(decisions[paire[0]] - decisions[paire[1]]) * 100, 1), w=core.LSTM_WINDOW,
            ),
        }
    sortie["connaissances"] = connaissances

    # Onglet « Réseau de neurones »
    if strategie in ("EMS_power_limitation", "EMS_fuzzy_logic"):
        sortie["reseau"] = {"type": "aucun"}
    elif demande_nulle:
        sortie["reseau"] = {"type": "vide"}
    elif strategie == "EMS_MLP_neurosymbolic":
        delta = alpha_reseau - alpha_flou
        sortie["reseau"] = {
            "type": "ns_mlp", "correction": delta, "part_permise": abs(delta) / core.MLP_NS_MAX_DELTA, "garde_fou": garde_actif,
            "correction_max": core.MLP_NS_MAX_DELTA, "reserve": core.MLP_NS_RESERVE_PB_SOC,
        }
    else:
        reseau = {
            "type": "contributions", "nb_grandeurs": len(contribs), "reference": reference, "alpha_explique": alpha_explique,
            "memoire_s": core.LSTM_WINDOW if strategie in ("EMS_LSTM", "EMS_LSTM_neurosymbolic") else None,
            "poids_ontologie": None, "schema": None,
        }
        if strategie == "EMS_LSTM_neurosymbolic":
            total = sum(abs(v) for _, v in contribs) or 1.0
            reseau["poids_ontologie"] = sum(abs(v) for c, v in contribs if c in xai.ETATS_SYMBOLIQUES) / total
        if strategie == "EMS_GNN":
            reseau["schema"] = _poids_composants_gnn(p_dem, soc_eb, soc_pb, accel)
        sortie["reseau"] = reseau

    # Onglet « Et si… ? »
    sortie["et_si"] = None
    if not demande_nulle:
        situation = xai.situation(df, traj, instant)
        base_a = xai.alpha_decision(strategie, df, traj, instant)
        base_p = base_a * situation["hasPower"]
        variantes = []
        for (grandeur, signe), libelle in SCENARIOS.items():
            valeur = situation[grandeur] + signe * (1000.0 if grandeur == "hasPower" else 0.05)
            if grandeur != "hasPower":
                valeur = float(np.clip(valeur, 0.0, 1.0))
            a = xai.alpha_decision(strategie, df, traj, instant, {grandeur: valeur})
            p = a * (valeur if grandeur == "hasPower" else situation["hasPower"])
            variantes.append({
                "libelle": lib(libelle), "alpha": nombre_json(a), "p_w": nombre_json(p), "variation_w": nombre_json(p - base_p),
                "conforme": bool(SENS_ATTENDU[grandeur] * signe * (p - base_p) >= -0.01 * abs(p_dem)),
            })
        sortie["et_si"] = {
            "base": {"alpha": nombre_json(base_a), "p_w": nombre_json(base_p)},
            "variantes": variantes,
            "contrefactuels": ox.contrefactuels(p_dem, soc_eb, soc_pb),
            "regles_non_appliquees": [
                f"**{r['id']}** ({r['lecture']}) : "
                + " ; ".join(tr("`{c}` n'est pas vérifié", "`{c}` does not hold", c=x["texte"]) for x in r["details"] if x["ok"] is False)
                for r in non_activees if r["type"] in ("mode", "repartition")
            ],
        }
    return sortie


def _par_minute(y, pas=60):
    k = len(y) // pas * pas
    if k == 0:
        return np.asarray([], dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(np.asarray(y[:k], dtype=float).reshape(-1, pas), axis=1)


def explication_cycle(identifiant: str, strategie: str) -> dict:
    """Bilan de l'explicabilité d'une stratégie sur tout le cycle."""
    donnees = obtenir(identifiant)
    resultats, df = donnees["resultats"], donnees["cycle_df"]
    if strategie not in resultats:
        raise KeyError(strategie)
    traj = resultats[strategie]
    n = min([len(df)] + [len(x["P_EB"]) for x in resultats.values()])

    p = df["hasPower"].to_numpy(dtype=float)[:n]
    accel = (df["hasAcceleration"].to_numpy(dtype=float) if "hasAcceleration" in df.columns else np.zeros(len(df)))[:n]
    soc_eb, soc_pb = np.asarray(traj["SOC_EB"], dtype=float)[:n], np.asarray(traj["SOC_PB"], dtype=float)[:n]
    a_req = np.asarray(traj["alpha_requested"] if "alpha_requested" in traj else traj["alpha_final"], dtype=float)[:n]
    a_fin = np.asarray(traj["alpha_final"], dtype=float)[:n]
    corr = np.asarray(traj["correction_applied"], dtype=float)[:n]
    actif = np.abs(p) > core.EPS_POWER_W
    flou = core.alpha_fuzzy_calc(soc_eb, soc_pb, p, accel)
    alpha_flou = np.asarray(flou["alpha"], dtype=float)
    forces = np.asarray(flou["strengths"], dtype=float)
    dominantes = np.asarray(flou["dominant_rule"]).astype(str)
    alpha_onto = ox.alpha_ontologie_vect(p, soc_eb)

    m = actif & ~np.isnan(alpha_onto)
    ecart_onto = np.abs(a_fin[m] - alpha_onto[m])
    if strategie not in donnees["coherence"]:
        donnees["coherence"][strategie] = xai.coherence_physique(strategie, df, traj)
    e3, e3_detail = donnees["coherence"][strategie]
    exacte, _ = NATURE.get(strategie, (False, None))

    sortie = {
        "exacte": exacte,
        "e3": nombre_json(e3),
        "accord_ontologie": nombre_json(np.mean(ecart_onto < 0.05)) if len(ecart_onto) else None,
        "part_corrigee": nombre_json(np.mean(corr[actif])) if actif.any() else None,
        "legende": tr(
            "Cohérence physique : part de 120 instants de traction où la décision évolue dans le sens "
            "attendu ({d}). Accord avec la règle de l'ontologie : part des instants où la décision reste à "
            "moins de 5 points de la répartition prescrite par OntoHESS (écart moyen : {e} points).",
            "Physical consistency: share of 120 traction time steps where the decision moves in the "
            "expected direction ({d}). Agreement with the ontology rule: share of time steps where the "
            "decision stays within 5 points of the split prescribed by OntoHESS (mean gap: {e} points).",
            d=" ; ".join(f"{xai.libelle_contrainte(k)} : {_pct(v, 0)}" for k, v in e3_detail.items()),
            e=nombre(np.mean(ecart_onto) * 100, 1) if len(ecart_onto) else "—",
        ),
        "regles": None,
    }

    avec_regles = strategie in ("EMS_fuzzy_logic", "EMS_MLP_neurosymbolic")
    if avec_regles:
        regles = {"vraies_par_decision": nombre_json(np.mean(np.sum(forces[actif] > 0.05, axis=1))), "correction_utilisee": None,
                  "garde_fou": None, "regles_seules": None, "reserve": core.MLP_NS_RESERVE_PB_SOC}
        if strategie == "EMS_MLP_neurosymbolic":
            a_reseau = np.asarray(traj.get("alpha_reseau", a_req), dtype=float)[:n]
            delta = np.abs(a_reseau[actif] - alpha_flou[actif])
            regles["correction_utilisee"] = nombre_json(np.nanmean(delta) / core.MLP_NS_MAX_DELTA)
            if "garde_fou" in traj:
                regles["garde_fou"] = nombre_json(np.mean(np.asarray(traj["garde_fou"], dtype=bool)[:n][actif]))
            else:
                regles["regles_seules"] = nombre_json(np.mean(delta < 0.02))
        cles, comptes = np.unique(dominantes[actif], return_counts=True)
        ordre = list(np.argsort(comptes)[::-1])
        regles["dominantes"] = [
            {"regle": _regle_floue(cles[i]), "part": nombre_json(comptes[i] / comptes.sum())} for i in ordre
        ]
        sortie["regles"] = regles

    temps = df["time"].to_numpy(dtype=float)[:n] if "time" in df.columns else np.arange(n, dtype=float)
    sortie["au_fil_du_cycle"] = {
        "temps_min": liste(_par_minute(temps) / 60.0, 2),
        "decision_pct": liste(_par_minute(np.where(actif, a_fin, np.nan)) * 100, 2),
        "ontologie_pct": liste(_par_minute(alpha_onto) * 100, 2),
        "regles_pct": liste(_par_minute(np.where(actif, alpha_flou, np.nan)) * 100, 2) if strategie == "EMS_MLP_neurosymbolic" else None,
    }
    return sortie
