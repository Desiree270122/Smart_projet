"""
core/lecture.py — Lecture en clair des courbes : la phrase qui s'affiche quand le
curseur passe sur un point d'un graphique.

Chaque fonction retourne une liste de textes, un par instant, dans la langue
choisie (core/i18n.py). Les textes sont pliés sur plusieurs lignes pour tenir
dans la bulle du graphique.

Un cycle compte plus de dix mille instants mais peu de situations différentes :
chaque phrase n'est rédigée qu'une fois par situation, puis réutilisée.
"""

import textwrap

import numpy as np
import plotly.graph_objects as go

import ems_core as core
from core.format import nombre
from core.i18n import tr

LARGEUR_BULLE = 58   # caractères par ligne dans la bulle
REPOS_W = 50.0       # en dessous, une batterie est considérée au repos


def bulle(nom: str, valeur: str) -> str:
    """Ligne de la bulle d'une courbe : son nom, puis sa valeur (gabarit Plotly)."""
    return f"{nom}{tr(' : ', ': ')}{valeur}<extra></extra>"


def courbe_explication(x, y, textes, acceleree=True):
    """Courbe invisible qui ajoute à la bulle la phrase d'explication de l'instant survolé."""
    Courbe = go.Scattergl if acceleree else go.Scatter
    return Courbe(
        x=x, y=y, name=tr("En clair", "In plain words"), customdata=textes, hovertemplate="%{customdata}<extra></extra>",
        mode="lines", line=dict(color="rgba(0,0,0,0)", width=0), showlegend=False,
    )


def aide_survol() -> str:
    return tr(
        "Passez le curseur sur une courbe : la bulle donne les valeurs et explique ce qui se passe à cet instant.",
        "Move the cursor over a curve: the tooltip gives the values and explains what is happening at that time.",
    )


def _plier(texte: str) -> str:
    return "<br>".join(textwrap.wrap(texte, LARGEUR_BULLE))


def _par_situation(situations, rediger) -> list:
    """Un texte par instant, rédigé une seule fois par situation distincte."""
    deja = {}
    textes = []
    for situation in situations:
        texte = deja.get(situation)
        if texte is None:
            texte = deja[situation] = rediger(situation)
        textes.append(texte)
    return textes


def lire_repartition(p_dem, traj, n) -> list:
    """Pourquoi la puissance est répartie ainsi à chaque instant : situation
    (traction, freinage, repos), limites atteintes, part de la batterie Puissance."""
    p_dem = np.asarray(p_dem, dtype=float)[:n]
    p_eb = np.asarray(traj["P_EB"], dtype=float)[:n]
    p_pb = np.asarray(traj["P_PB"], dtype=float)[:n]
    eb_vide = np.asarray(traj["SOC_EB"], dtype=float)[:n] <= core.SOC_EB_MIN + 5e-3
    pb_vide = np.asarray(traj["SOC_PB"], dtype=float)[:n] <= core.SOC_PB_MIN + 5e-3
    non_fourni = np.asarray(traj.get("P_unserved", np.zeros(n)), dtype=float)[:n]
    non_recupere = np.asarray(traj.get("P_regen_curtailed", np.zeros(n)), dtype=float)[:n]
    propose = np.asarray(traj.get("alpha_requested", traj["alpha_final"]), dtype=float)[:n]
    corrige = np.abs(np.asarray(traj["alpha_final"], dtype=float)[:n] - propose) > 0.02

    lim_dech = f"{nombre(core.P_EB_MAX_W / 1000.0, 1)} kW"
    lim_rech = f"{nombre(-core.P_EB_MIN_W / 1000.0, 1)} kW"
    # Gabarits : {v} est la part de la batterie Puissance (%) ou une puissance (kW).
    gabarits = {
        "repos": tr("Demande quasi nulle : les batteries sont au repos.", "Near-zero demand: the batteries are idle."),
        "manque_pb": tr(
            "⚠ {v} kW ne sont pas fournis : la batterie Puissance est à son SOC minimal et la demande dépasse la limite de la batterie Énergie ({l}).",
            "⚠ {v} kW are not delivered: the Power battery is at its minimum SOC and the demand exceeds the Energy battery's limit ({l}).",
            v="{v}", l=lim_dech),
        "manque": tr("⚠ {v} kW ne sont pas fournis : les limites des batteries sont atteintes.",
                     "⚠ {v} kW are not delivered: the batteries' limits are reached.", v="{v}"),
        "eb_vide": tr("Batterie Énergie à son SOC minimal : la batterie Puissance fournit la demande.",
                      "Energy battery at its minimum SOC: the Power battery supplies the demand."),
        "au_dela": tr(
            "Demande au-delà de la limite de la batterie Énergie ({l}) : elle fournit son maximum, la batterie Puissance le complément ({v} %).",
            "Demand beyond the Energy battery's limit ({l}): it supplies its maximum, the Power battery the rest ({v} %).",
            l=lim_dech, v="{v}"),
        "au_dela_plus": tr(
            "Demande au-delà de la limite de la batterie Énergie ({l}) : la batterie Puissance en fournit {v} %, plus que le strict complément.",
            "Demand beyond the Energy battery's limit ({l}): the Power battery supplies {v} % of it, more than the strict remainder.",
            l=lim_dech, v="{v}"),
        "eb_seule": tr("Demande dans la limite de la batterie Énergie : elle la fournit seule.",
                       "Demand within the Energy battery's limit: it supplies it alone."),
        "pb_seule": tr("La batterie Puissance fournit seule la demande, que la batterie Énergie aurait pu couvrir.",
                       "The Power battery supplies the demand alone, although the Energy battery could have covered it."),
        "partage": tr(
            "Demande dans la limite de la batterie Énergie ; la stratégie en confie tout de même {v} % à la batterie Puissance.",
            "Demand within the Energy battery's limit; the strategy still assigns {v} % of it to the Power battery.", v="{v}"),
        "frein_perdu": tr("⚠ Freinage : {v} kW ne sont pas récupérés (limites des batteries atteintes).",
                          "⚠ Braking: {v} kW are not recovered (battery limits reached).", v="{v}"),
        "frein_au_dela": tr(
            "Freinage au-delà de ce que la batterie Énergie peut absorber ({l}) : elle absorbe son maximum, la batterie Puissance le surplus ({v} %).",
            "Braking beyond what the Energy battery can absorb ({l}): it absorbs its maximum, the Power battery the surplus ({v} %).",
            l=lim_rech, v="{v}"),
        "frein_au_dela_plus": tr(
            "Freinage au-delà de ce que la batterie Énergie peut absorber ({l}) : la batterie Puissance en absorbe {v} %.",
            "Braking beyond what the Energy battery can absorb ({l}): the Power battery absorbs {v} % of it.",
            l=lim_rech, v="{v}"),
        "frein_eb": tr("Freinage : la batterie Énergie récupère toute l'énergie.", "Braking: the Energy battery recovers all the energy."),
        "frein_pb": tr("Freinage : la batterie Puissance récupère toute l'énergie.", "Braking: the Power battery recovers all the energy."),
        "frein_partage": tr("Freinage : l'énergie est récupérée par les deux batteries ({v} % pour la batterie Puissance).",
                            "Braking: the energy is recovered by both batteries ({v} % for the Power battery).", v="{v}"),
    }
    filtre = tr(" Le filtre de sécurité a corrigé la proposition de la stratégie.",
                " The safety filter corrected the strategy's proposal.")

    # Situation de chaque instant : (gabarit, valeur arrondie, décimales, correction du filtre).
    situations = []
    for t in range(n):
        p, eb, pb = p_dem[t], p_eb[t], p_pb[t]
        if abs(p) <= core.EPS_POWER_W:
            situations.append(("repos", 0, 0, False))
            continue
        part = int(round(100.0 * pb / p))
        if p > 0:
            if non_fourni[t] > 1.0:
                situations.append(("manque_pb" if pb_vide[t] else "manque", round(non_fourni[t] / 1000.0, 1), 1, False))
                continue
            if eb_vide[t] and eb < REPOS_W:
                cle = "eb_vide"
            elif p > core.P_EB_MAX_W:
                cle = "au_dela" if eb >= core.P_EB_MAX_W - 1.0 else "au_dela_plus"
            elif pb < REPOS_W:
                cle = "eb_seule"
            elif eb < REPOS_W:
                cle = "pb_seule"
            else:
                cle = "partage"
        else:
            if non_recupere[t] > 1.0:
                situations.append(("frein_perdu", round(non_recupere[t] / 1000.0, 1), 1, False))
                continue
            if p < core.P_EB_MIN_W:
                cle = "frein_au_dela" if eb <= core.P_EB_MIN_W + 1.0 else "frein_au_dela_plus"
            elif pb > -REPOS_W:
                cle = "frein_eb"
            elif eb > -REPOS_W:
                cle = "frein_pb"
            else:
                cle = "frein_partage"
        situations.append((cle, part, 0, bool(corrige[t])))

    def rediger(situation):
        cle, valeur, decimales, avec_filtre = situation
        return _plier(gabarits[cle].replace("{v}", nombre(valeur, decimales)) + (filtre if avec_filtre else ""))

    return _par_situation(situations, rediger)


def lire_soc(traj) -> list:
    """Ce que font les deux batteries à chaque instant de la courbe des états de
    charge : décharge, recharge ou repos ; écart entre les deux SOC ; seuils atteints."""
    soc_eb = np.asarray(traj["SOC_EB"], dtype=float)
    soc_pb = np.asarray(traj["SOC_PB"], dtype=float)
    m = min(len(soc_eb), len(soc_pb))
    # Le dernier point reprend la dernière puissance connue.
    indices = np.minimum(np.arange(m), len(traj["P_EB"]) - 1)
    sens_eb = np.sign(np.where(np.abs(np.asarray(traj["P_EB"], dtype=float)) > REPOS_W, traj["P_EB"], 0.0))[indices].astype(int)
    sens_pb = np.sign(np.where(np.abs(np.asarray(traj["P_PB"], dtype=float)) > REPOS_W, traj["P_PB"], 0.0))[indices].astype(int)
    ecart = np.round(np.abs(soc_eb[:m] - soc_pb[:m]) * 100.0).astype(int)
    eb_vide = soc_eb[:m] <= core.SOC_EB_MIN + 5e-3
    pb_vide = soc_pb[:m] <= core.SOC_PB_MIN + 5e-3

    etats = {1: tr("se décharge", "is discharging"), -1: tr("se recharge", "is charging"), 0: tr("est au repos", "is idle")}
    modele = tr("La batterie Énergie {a}, la batterie Puissance {b}. Écart entre les SOC : {e} points.",
                "The Energy battery {a}, the Power battery {b}. Gap between the SOCs: {e} points.", a="{a}", b="{b}", e="{e}")
    repos = tr("Les deux batteries sont au repos. Écart entre les SOC : {e} points.",
               "Both batteries are idle. Gap between the SOCs: {e} points.", e="{e}")
    eb_min = tr(" Batterie Énergie à son SOC minimal : elle ne fournit plus.", " Energy battery at its minimum SOC: it no longer supplies.")
    pb_min = tr(" Batterie Puissance à son SOC minimal : elle ne peut plus aider aux pics.",
                " Power battery at its minimum SOC: it can no longer help with peaks.")

    def rediger(situation):
        a, b, e, eb_au_min, pb_au_min = situation
        texte = repos.format(e=e) if a == 0 and b == 0 else modele.format(a=etats[a], b=etats[b], e=e)
        return _plier(texte + (eb_min if eb_au_min else "") + (pb_min if pb_au_min else ""))

    return _par_situation(zip(sens_eb.tolist(), sens_pb.tolist(), ecart.tolist(), eb_vide.tolist(), pb_vide.tolist()), rediger)


def lire_composant(puissance, limite_decharge, limite_recharge, seuil) -> list:
    """État d'un composant à chaque instant : décharge ou recharge, puissance (à
    10 W près) et part de sa limite utilisée."""
    decharge = tr("se décharge à {p} kW ({l} % de sa limite)", "discharging at {p} kW ({l} % of its limit)", p="{p}", l="{l}")
    recharge = tr("se recharge à {p} kW ({l} % de sa limite)", "charging at {p} kW ({l} % of its limit)", p="{p}", l="{l}")
    repos = tr("au repos", "idle")
    w = np.asarray(puissance, dtype=float)
    # Situation : puissance arrondie à 10 W, 0 quand le composant est au repos.
    situations = np.where(np.abs(w) > seuil, np.round(w / 10.0), 0.0).astype(int).tolist()

    def rediger(dizaines_de_watts):
        watts = 10.0 * dizaines_de_watts
        if watts > 0:
            return decharge.format(p=nombre(watts / 1000.0, 2), l=nombre(100.0 * watts / limite_decharge, 0))
        if watts < 0:
            return recharge.format(p=nombre(-watts / 1000.0, 2), l=nombre(100.0 * watts / limite_recharge, 0))
        return repos

    return _par_situation(situations, rediger)
