"""
Pourquoi cette décision ? Pour un instant du cycle et une stratégie : la
décision prise, trois à cinq raisons en clair, puis le détail par onglets
(décision décomposée, raisons, connaissances expertes, réseau de neurones,
« et si… ? », bilan sur tout le cycle).

L'explication est exacte quand le calcul de la stratégie est lisible (règles),
reconstruite après coup quand il ne l'est pas (réseaux de neurones).
"""

import sys
import warnings
from pathlib import Path

DOSSIER_PROJET = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DOSSIER_PROJET))

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch

from ems_core import (
    ALPHA_GRID_STEP,
    DEVICE,
    EPS_POWER_W,
    FUZZY_RULE_NAMES,
    GNN_NODE_NAMES,
    GNN_SCALER_FILE,
    LSTM_NS_FEATURE_COLS,
    LSTM_WINDOW,
    MLP_NS_INPUT_COLS,
    MLP_NS_MAX_DELTA,
    MLP_NS_RESERVE_PB_SOC,
    P_EB_MAX_W,
    P_EB_MIN_W,
    SOC_EB_MIN,
    SOC_PB_MIN,
    alpha_fuzzy_calc,
    charger_scaler,
    construire_graphe_instant,
    load_gnn_simple,
)
from core import ontology_explainer as ox
from core import xai
from core.format import nombre, separateurs_plotly
from core.i18n import lib, tr
from core.instant import choisir_instant
from core.navigation import pied_navigation
from core.resultats import assurer_donnees_session, choisir_cycle, nom_affichage
from core.style import (
    COULEUR_DECISION,
    COULEUR_DEMANDE,
    COULEUR_EB,
    COULEUR_PB,
    COULEUR_REFERENCE,
    COULEUR_SECONDAIRE,
    couleur,
    flux_html,
)


# Une couleur = une signification (core/style.py).
C_EB, C_PB, C_GRIS = COULEUR_EB, COULEUR_PB, COULEUR_SECONDAIRE

NOMS_COMPOSANTS = {
    "energy_battery": ("Batterie Énergie", "Energy battery"),
    "power_battery": ("Batterie Puissance", "Power battery"),
    "converter": ("Convertisseur", "Converter"),
    "motor": ("Moteur", "Motor"),
    "vehicle": ("Véhicule", "Vehicle"),
}

# Nature de l'explication : exacte quand elle décrit le calcul lui-même,
# reconstruite quand le calcul du réseau de neurones n'est pas lisible.
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


# Calculs longs, gardés en mémoire. La signature (empreinte de la trajectoire)
# les fait recalculer si les données changent.

def _signature(strategie):
    return float(np.nansum(resultats[strategie]["alpha_final"]))


@st.cache_data(show_spinner=False)
def _contributions(strategie, instant, signature):
    return xai.shapley(strategie, df, resultats[strategie], instant)


@st.cache_data(show_spinner=False)
def _mesurer_coherence(strategie, signature):
    return xai.coherence_physique(strategie, df, resultats[strategie])


def _coherence(strategie, signature):
    """E3, lue dans les résultats de référence quand elle y figure, mesurée sinon."""
    connue = st.session_state.get("coherence_simulation") or {}
    if strategie in connue:
        return connue[strategie]
    with st.spinner(tr("Mesure de la cohérence physique…", "Measuring physical consistency…")):
        return _mesurer_coherence(strategie, signature)


@st.cache_resource(show_spinner=False)
def _charger_gnn():
    res = load_gnn_simple()
    modele = res[0] if isinstance(res, tuple) else res
    modele.eval()
    return modele, charger_scaler(GNN_SCALER_FILE)


def _poids_composants_gnn(p_dem, soc_eb, soc_pb, accel):
    """Poids (%) de chaque composant du schéma du HESS dans la décision du GNN
    (sensibilité locale de la décision aux grandeurs du composant)."""
    modele, scaler = _charger_gnn()
    x_g, liaisons = construire_graphe_instant(p_dem, soc_eb, soc_pb, accel, scaler)
    x_g = x_g.to(DEVICE).clone().requires_grad_(True)
    liaisons = liaisons.to(DEVICE)
    modele(x_g, liaisons, torch.zeros(x_g.shape[0], dtype=torch.long, device=DEVICE)).sum().backward()
    poids = np.abs((x_g.grad * x_g).detach().cpu().numpy()).sum(axis=1)
    return (poids / poids.sum() * 100.0 if poids.sum() > 0 else poids), liaisons


# Graphiques

def _barres_h(etiquettes, valeurs, couleurs, titre_x, textes):
    fig = go.Figure(
        go.Bar(
            y=etiquettes[::-1], x=valeurs[::-1], orientation="h", marker_color=couleurs[::-1],
            text=textes[::-1], textposition="outside", cliponaxis=False,
            hovertemplate="%{y} : %{text}<extra></extra>",
        )
    )
    fig.update_layout(
        separators=separateurs_plotly(), height=36 * len(etiquettes) + 70, margin=dict(t=10, b=40, l=10, r=60),
        xaxis_title=titre_x, showlegend=False,
    )
    return fig


def _cascade(etapes, alpha_final):
    """Décomposition de alpha (en % confiés à la batterie Puissance), de la première
    étape à la décision appliquée. etapes : [(libellé, valeur)] — la première est un
    point de départ, les suivantes des ajustements."""
    x = [e[0] for e in etapes] + [tr("Décision appliquée", "Decision applied")]
    y = [e[1] * 100 for e in etapes] + [0.0]
    mesures = ["absolute"] + ["relative"] * (len(etapes) - 1) + ["total"]
    textes = (
        [f"{nombre(y[0], 1)} %"] + [f"{nombre(v, 1, signe=True)} pts" for v in y[1:-1]] + [f"{nombre(alpha_final * 100, 1)} %"]
    )
    fig = go.Figure(
        go.Waterfall(
            x=x, y=y, measure=mesures, text=textes, textposition="outside",
            connector=dict(line=dict(color=C_GRIS, width=1)),
            increasing=dict(marker=dict(color=C_PB)),
            decreasing=dict(marker=dict(color=C_EB)),
            totals=dict(marker=dict(color=COULEUR_DECISION)),
            hoverinfo="skip",
        )
    )
    fig.update_layout(
        separators=separateurs_plotly(), height=340, margin=dict(t=20, b=40, l=10, r=10), showlegend=False,
        yaxis=dict(title=tr("Part confiée à la batterie Puissance (%)", "Share assigned to the Power battery (%)"), range=[0, 110]),
    )
    return fig


def _schema_gnn(poids, liaisons):
    """Schéma du HESS, chaque composant coloré selon son poids dans la décision."""
    positions = [(0.0, 1.0), (0.0, -1.0), (1.2, 0.0), (2.4, 0.0), (3.6, 0.0)]
    paires = liaisons.detach().cpu().numpy()
    lx, ly = [], []
    for k in range(paires.shape[1]):
        a, b = int(paires[0, k]), int(paires[1, k])
        if a < len(positions) and b < len(positions):
            lx += [positions[a][0], positions[b][0], None]
            ly += [positions[a][1], positions[b][1], None]
    noms_c = [lib(NOMS_COMPOSANTS[c]) if c in NOMS_COMPOSANTS else c for c in GNN_NODE_NAMES]
    fig = go.Figure([
        go.Scatter(x=lx, y=ly, mode="lines", line=dict(color="#C7CCD6", width=2), hoverinfo="skip"),
        go.Scatter(
            x=[p[0] for p in positions], y=[p[1] for p in positions], mode="markers+text",
            marker=dict(
                size=[34 + p * 0.9 for p in poids], color=list(poids), colorscale="Blues",
                showscale=True, colorbar=dict(title="%"), line=dict(color="white", width=2),
            ),
            text=[f"{l}<br>{nombre(p, 0)} %" for l, p in zip(noms_c, poids)],
            textposition="bottom center", hoverinfo="text",
        ),
    ])
    fig.update_layout(
        separators=separateurs_plotly(), height=360, showlegend=False, margin=dict(t=20, b=20, l=10, r=10),
        xaxis=dict(visible=False), yaxis=dict(visible=False, range=[-1.8, 1.6]),
    )
    return fig


def kw(x):
    return f"{nombre(x / 1000.0, 1)} kW"


def _pct(x, decimales=1):
    return f"{nombre(x * 100, decimales)} %"


def _points(x):
    return tr("{v} points", "{v} points", v=nombre(x * 100, 1, signe=True))


def _regle_floue(cle):
    if cle in ox.REGLES_FLOUES:
        return ox.libelle_regle_floue(cle)
    return tr("Répartition par défaut", "Default split")


def _grandeur(col):
    """Nom d'une grandeur dans une phrase (les sigles gardent leur casse)."""
    nom = xai.libelle_entree(col)
    return nom if nom[:2].isupper() else nom[0].lower() + nom[1:]


# Interface

st.title(tr("💡 Pourquoi cette décision ?", "💡 Why this decision?"))
st.caption(tr(
    "Pour un instant du cycle et une stratégie : quelle décision a été prise, et pourquoi. "
    "L'explication est exacte quand le calcul de la stratégie est lisible, reconstruite après coup "
    "quand il ne l'est pas.",
    "For one time step of the cycle and one strategy: which decision was made, and why. The "
    "explanation is exact when the strategy's calculation can be read, reconstructed afterwards when "
    "it cannot.",
))

try:
    assurer_donnees_session(st)
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

resultats = st.session_state.get("resultats_simulation")
df = st.session_state.get("cycle_pret")
if not resultats or df is None:
    st.warning(tr("Aucune donnée disponible.", "No data available."))
    st.stop()

noms = list(resultats.keys())
n = min([len(df)] + [len(t["P_EB"]) for t in resultats.values()])

col_t, col_s, col_c = st.columns([2, 1, 1])
instant, t_sel = choisir_instant(df, n, col_t)
strategie = col_s.selectbox(tr("Stratégie", "Strategy"), noms, format_func=nom_affichage, key="strategie_explication")
choisir_cycle(st, col_c)

traj = resultats[strategie]
vitesse = float(df["speed"].iloc[instant]) if "speed" in df.columns else 0.0
accel = float(df["hasAcceleration"].iloc[instant]) if "hasAcceleration" in df.columns else 0.0
p_dem = float(df["hasPower"].iloc[instant])
soc_eb = float(traj["SOC_EB"][instant])
soc_pb = float(traj["SOC_PB"][instant])
p_eb = float(traj["P_EB"][instant])
p_pb = float(traj["P_PB"][instant])
alpha_final = float(traj["alpha_final"][instant])
alpha_req = float(traj["alpha_requested"][instant]) if "alpha_requested" in traj else alpha_final
correction = bool(traj["correction_applied"][instant]) if "correction_applied" in traj else False
demande_nulle = abs(p_dem) <= EPS_POWER_W

etat = ox.etat_instant(p_dem, soc_eb, soc_pb, p_eb=p_eb)
onto = ox.repartition_ontologie(p_dem, soc_eb, soc_pb)
exacte, nature = NATURE.get(strategie, (False, ("", "")))

# Ingrédients de l'explication, calculés une fois pour toute la page
res_flou = alpha_fuzzy_calc(np.array([soc_eb]), np.array([soc_pb]), np.array([p_dem]), np.array([accel]))
alpha_flou = float(res_flou["alpha"][0])
forces = np.asarray(res_flou["strengths"][0], dtype=float)
contrib_regles = ox.contributions_floues(forces)
dominante = str(res_flou["dominant_rule"][0])
avec_contributions = strategie in xai.SHAPLEY_DISPONIBLE and not demande_nulle
if avec_contributions:
    with st.spinner(tr("Calcul de la contribution de chaque grandeur…", "Computing the contribution of each quantity…")):
        reference, contribs, alpha_explique = _contributions(strategie, instant, _signature(strategie))
    principales = sorted(contribs, key=lambda kv: -abs(kv[1]))
entrees_onto = {
    "EMS_MLP_neurosymbolic": MLP_NS_INPUT_COLS,
    "EMS_LSTM_neurosymbolic": LSTM_NS_FEATURE_COLS,
}.get(strategie, [])
etats_transmis = [c for c in ox.LIBELLES_SYMBOLIQUES if c in entrees_onto]


@st.cache_data(show_spinner=False)
def _decomposer_ns(instant, signature):
    return xai.decomposer_ns_mlp(df, resultats["EMS_MLP_neurosymbolic"], instant)


def decomposition_ns(cle, i):
    """(alpha des règles floues, alpha après correction du réseau, garde-fou actif) de
    NS-MLP, lus dans la trajectoire si la simulation les a gardés, recalculés sinon."""
    trj = resultats[cle]
    if "alpha_reseau" in trj and np.isfinite(trj["alpha_reseau"][i]):
        return float(trj["alpha_flou"][i]), float(trj["alpha_reseau"][i]), bool(trj["garde_fou"][i])
    d = _decomposer_ns(i, _signature(cle))
    return d["alpha_flou"], d["alpha_reseau"], d["garde_fou"]


if strategie == "EMS_MLP_neurosymbolic" and not demande_nulle:
    alpha_flou, alpha_reseau, garde_actif = decomposition_ns(strategie, instant)
else:
    alpha_reseau, garde_actif = alpha_req, False

# Aucune règle floue n'est nettement vraie : la répartition par défaut s'applique.
regle_par_defaut = dominante not in FUZZY_RULE_NAMES or float(forces.max(initial=0.0)) < 1e-3

nom_eb, nom_pb = tr("batterie Énergie", "Energy battery"), tr("batterie Puissance", "Power battery")


# La décision, tout en haut

with st.container(border=True):
    etat_min = etat["libelle"][0].lower() + etat["libelle"][1:]
    st.markdown(tr(
        "**{s}** · instant t = {t} s · vitesse {v} km/h · état déduit par l'ontologie : {e}",
        "**{s}** · time t = {t} s · speed {v} km/h · state inferred by the ontology: {e}",
        s=nom_affichage(strategie), t=nombre(t_sel, 0), v=nombre(vitesse * 3.6, 0), e=etat_min,
    ))
    d = st.columns(6)
    d[0].metric(tr("Demande", "Demand"), kw(p_dem))
    d[1].metric("SOC EB", _pct(soc_eb))
    d[2].metric("SOC PB", _pct(soc_pb))
    d[3].metric(tr("alpha (part de la PB)", "alpha (PB share)"), _pct(alpha_final))
    d[4].metric(tr("Batterie Énergie", "Energy battery"), kw(p_eb))
    d[5].metric(tr("Batterie Puissance", "Power battery"), kw(p_pb))


# Pourquoi cette décision ? 3 à 5 raisons en clair

def _raisons():
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
            p=kw(p_dem), l=kw(P_EB_MAX_W),
            pos=tr("au-delà de", "beyond") if p_dem > P_EB_MAX_W else tr("dans", "within"),
        ))
    else:
        r.append(tr(
            "Le véhicule freine : **{p}** sont à récupérer{plus} (limite {l}).",
            "The vehicle is braking: **{p}** can be recovered{plus} (limit {l}).",
            p=kw(-p_dem), l=kw(-P_EB_MIN_W),
            plus=tr(", plus que ce que la batterie Énergie peut absorber", ", more than the Energy battery can absorb")
            if p_dem < P_EB_MIN_W else "",
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
            c=_points(alpha_reseau - alpha_flou), m=nombre(MLP_NS_MAX_DELTA * 100, 0),
        ))
        if garde_actif:
            r.append(tr(
                "L'ontologie déclare la batterie Puissance faible (SOC {s} % ≤ {l} %) : le **garde-fou** "
                "applique les règles R14/R16 d'OntoHESS et ramène sa part à **{a}**.",
                "The ontology declares the Power battery low (SOC {s} % ≤ {l} %): the **safeguard** "
                "applies OntoHESS rules R14/R16 and brings its share back to **{a}**.",
                s=nombre(soc_pb * 100, 0), l=nombre(MLP_NS_RESERVE_PB_SOC * 100, 0), a=_pct(alpha_req),
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
            w=LSTM_WINDOW, g=_grandeur(principale[0]), v=_points(principale[1]),
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
            i_dom = list(FUZZY_RULE_NAMES).index(dominante)
            r.append(tr(
                "La règle floue qui pèse le plus est **{g}** (vraie à {f} %) : {s}.",
                "The fuzzy rule that weighs most is **{g}** (true to {f} %): {s}.",
                g=_regle_floue(dominante), f=nombre(forces[i_dom] * 100, 0), s=ox.sens_regle_floue(dominante),
            ))
        elif avec_contributions:
            fortes = [(_grandeur(c), v) for c, v in principales[:2] if abs(v) >= 5e-4]
            r.append(
                tr("Le réseau s'appuie surtout sur ", "The network relies mostly on ")
                + tr(" et ", " and ").join(f"**{g}** ({_points(v)})" for g, v in fortes) + "."
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


st.subheader(tr("Pourquoi cette décision ?", "Why this decision?"))
st.markdown("\n".join(f"{i}. {r}" for i, r in enumerate(_raisons(), start=1)))
st.caption(
    (tr("Explication exacte : ", "Exact explanation: ") if exacte else tr("Explication reconstruite : ", "Reconstructed explanation: "))
    + lib(nature)
)

onglet_dec, onglet_rai, onglet_onto, onglet_res, onglet_etsi, onglet_cycle = st.tabs([
    tr("Décision", "Decision"), tr("Raisons", "Reasons"), tr("Connaissances expertes", "Expert knowledge"),
    tr("Réseau de neurones", "Neural network"), tr("Et si… ?", "What if…?"), tr("Sur tout le cycle", "Over the whole cycle"),
])
legende_couleurs = tr(
    "orange = pousse vers la batterie Puissance, vert = vers la batterie Énergie",
    "orange = pushes towards the Power battery, green = towards the Energy battery",
)


# Onglet « Décision » : la décision décomposée, et les autres stratégies au même instant

with onglet_dec:
    if demande_nulle:
        st.info(tr("Demande quasi nulle : pas de répartition à décomposer.", "Near-zero demand: no split to break down."))
    else:
        if strategie in ("EMS_fuzzy_logic", "EMS_MLP_neurosymbolic"):
            etapes = (
                [(_regle_floue(FUZZY_RULE_NAMES[i]), float(contrib_regles[i]))
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
        st.plotly_chart(_cascade(etapes, alpha_final), width="stretch")
        st.caption(tr(
            "Part confiée à la batterie Puissance, étape par étape : {l}, violet = décision appliquée.",
            "Share assigned to the Power battery, step by step: {l}, purple = decision applied.",
            l=legende_couleurs,
        ))
        if correction:
            st.warning(tr(
                "Le filtre de sécurité a corrigé la répartition proposée ({a}) pour respecter les "
                "limites physiques des batteries et du convertisseur.",
                "The safety filter corrected the proposed split ({a}) to respect the physical limits of "
                "the batteries and the converter.",
                a=_pct(alpha_req),
            ))
        elif abs(alpha_final - alpha_req) > 1e-9:
            pas = st.session_state.get("pas_alpha") or ALPHA_GRID_STEP
            st.caption(tr(
                "Proposé {a}, appliqué {b} : le filtre de sécurité règle la répartition par pas de "
                "{p} % ; un écart plus petit est un arrondi, pas une correction.",
                "Proposed {a}, applied {b}: the safety filter sets the split in steps of {p} %; a "
                "smaller gap is rounding, not a correction.",
                a=_pct(alpha_req), b=_pct(alpha_final), p=nombre(pas * 100, 1),
            ))

        st.markdown(tr("**Les autres stratégies au même instant**", "**The other strategies at the same time step**"))
        alphas = {nm: float(resultats[nm]["alpha_final"][instant]) * 100 for nm in noms}
        ordre_s = sorted(noms, key=lambda nm: alphas[nm], reverse=True)
        fig_a = go.Figure(
            go.Bar(
                y=[nom_affichage(nm) for nm in ordre_s][::-1], x=[alphas[nm] for nm in ordre_s][::-1], orientation="h",
                marker=dict(
                    color=[couleur(nm) for nm in ordre_s][::-1],
                    opacity=[1.0 if nm == strategie else 0.4 for nm in ordre_s][::-1],
                ),
                text=[f"{nombre(alphas[nm], 1)} %" for nm in ordre_s][::-1], textposition="outside", hoverinfo="skip",
            )
        )
        if onto is not None:
            fig_a.add_vline(
                x=onto["alpha"] * 100, line=dict(color=COULEUR_REFERENCE, dash="dash"),
                annotation_text=tr("règle {i} de l'ontologie", "ontology rule {i}", i=onto["regle"]["id"]),
                annotation_position="top",
            )
        fig_a.update_layout(
            separators=separateurs_plotly(), height=40 * len(noms) + 90, margin=dict(t=30, b=40, l=10, r=50),
            xaxis=dict(title=tr("Part confiée à la batterie Puissance (%)", "Share assigned to the Power battery (%)"), range=[0, 110]),
            showlegend=False,
        )
        st.plotly_chart(fig_a, width="stretch")
        st.caption(tr(
            "Chaque stratégie décide avec ses propres états de charge à cet instant.",
            "Each strategy decides with its own states of charge at this time step.",
        ))

    with st.expander(tr("Les puissances autour de cet instant", "The powers around this time step")):
        demi = 150
        i0, i1 = max(0, instant - demi), min(n, instant + demi + 1)
        xx = df["time"].to_numpy()[i0:i1] if "time" in df.columns else np.arange(i0, i1)
        fig_ctx = go.Figure([
            go.Scatter(x=xx, y=df["hasPower"].to_numpy()[i0:i1] / 1000.0, name=tr("Demande", "Demand"), line=dict(color=COULEUR_DEMANDE)),
            go.Scatter(x=xx, y=np.asarray(traj["P_EB"], float)[i0:i1] / 1000.0, name=tr("Batterie Énergie", "Energy battery"), line=dict(color=C_EB)),
            go.Scatter(x=xx, y=np.asarray(traj["P_PB"], float)[i0:i1] / 1000.0, name=tr("Batterie Puissance", "Power battery"), line=dict(color=C_PB)),
        ])
        fig_ctx.add_vline(x=t_sel, line=dict(color=COULEUR_REFERENCE, dash="dash"))
        fig_ctx.update_layout(
            separators=separateurs_plotly(), xaxis_title=tr("Temps (s)", "Time (s)"), yaxis_title=tr("Puissance (kW)", "Power (kW)"),
            height=320, margin=dict(t=20, b=40), hovermode="x unified", legend=dict(orientation="h", y=1.1),
        )
        st.plotly_chart(fig_ctx, width="stretch")


# Onglet « Raisons » : ce qui a contribué à la décision

with onglet_rai:
    if demande_nulle:
        st.info(tr("Demande quasi nulle : pas de décision à expliquer.", "Near-zero demand: no decision to explain."))
    elif strategie == "EMS_power_limitation":
        if onto is not None:
            premisses = tr(" et ", " and ").join(f"`{x['texte']}`" for x in onto["regle"]["details"])
            st.markdown(tr(
                "La décision découle d'une seule règle : **{i}** — si {p}, alors {l}.",
                "The decision follows from a single rule: **{i}** — if {p}, then {l}.",
                i=onto["regle"]["id"], p=premisses, l=onto["regle"]["lecture"],
            ))
        st.caption(tr(
            "Le modèle physique ne dépend que de la puissance demandée et du SOC de la batterie Énergie.",
            "The physical model depends only on the power demand and the Energy battery's SOC.",
        ))
    elif strategie in ("EMS_fuzzy_logic", "EMS_MLP_neurosymbolic"):
        regles_f = {r["cle"]: r for r in ox.regles_floues()}
        vraies = [i for i in np.argsort(forces)[::-1] if forces[i] > 5e-4]
        if vraies:
            st.dataframe(
                [
                    {
                        tr("Règle", "Rule"): regles_f[FUZZY_RULE_NAMES[i]]["libelle"],
                        tr("Si", "If"): regles_f[FUZZY_RULE_NAMES[i]]["si"],
                        tr("Degré de vérité", "Degree of truth"): _pct(forces[i], 0),
                        tr("Propose pour la batterie Puissance", "Proposes for the Power battery"): _pct(regles_f[FUZZY_RULE_NAMES[i]]["alpha"], 0),
                        tr("Contribution à la décision", "Contribution to the decision"):
                            f"{nombre(contrib_regles[i] * 100, 1)} pts" if contrib_regles is not None else "—",
                    }
                    for i in vraies
                ],
                hide_index=True, width="stretch",
            )
        st.caption(tr(
            "La décision des règles est la moyenne de leurs propositions, pondérée par leur degré de "
            "vérité : la contribution de chaque règle est donc exacte.",
            "The decision of the rules is the average of their proposals, weighted by their degree of "
            "truth: the contribution of each rule is therefore exact.",
        ))
    else:
        valeurs = [(xai.libelle_entree(c), v * 100) for c, v in principales]
        st.plotly_chart(
            _barres_h(
                [g for g, _ in valeurs], [v for _, v in valeurs], [C_PB if v >= 0 else C_EB for _, v in valeurs],
                tr("Contribution à la part de la batterie Puissance (points)", "Contribution to the Power battery's share (points)"),
                [f"{nombre(v, 1, signe=True)} pts" for _, v in valeurs],
            ),
            width="stretch",
        )
        st.caption(tr(
            "Par rapport à une situation moyenne du cycle, où le réseau confierait {a} à la batterie Puissance : {l}.",
            "Compared with an average situation of the cycle, where the network would assign {a} to the Power battery: {l}.",
            a=_pct(reference), l=legende_couleurs,
        ))


# Onglet « Connaissances expertes » : l'ontologie, et sa place dans la stratégie

def _coche(ok, texte):
    return f"{'✓' if ok else '✗'} {texte}"


oui, non = tr("oui", "yes"), tr("non", "no")

with onglet_onto:
    if strategie == "EMS_MLP_neurosymbolic" and not demande_nulle:
        st.markdown(tr(
            "**NS-MLP : les règles décident, le réseau corrige, l'ontologie protège la réserve de la batterie Puissance**",
            "**NS-MLP: the rules decide, the network corrects, the ontology protects the Power battery's reserve**",
        ))
        st.markdown(
            flux_html(
                [
                    (tr("Règles floues", "Fuzzy rules"), COULEUR_REFERENCE),
                    (tr("alpha des règles {a}", "alpha from the rules {a}", a=_pct(alpha_flou)), COULEUR_DECISION),
                    (tr("correction du réseau {c} pts", "network correction {c} pts",
                        c=nombre((alpha_reseau - alpha_flou) * 100, 1, signe=True)), COULEUR_SECONDAIRE),
                ]
                + ([(tr("garde-fou R14/R16 : {a}", "safeguard R14/R16: {a}", a=_pct(alpha_req)), COULEUR_REFERENCE)] if garde_actif else [])
                + [(tr("alpha final {a}", "final alpha {a}", a=_pct(alpha_final)), COULEUR_DECISION)]
            ),
            unsafe_allow_html=True,
        )
    elif strategie == "EMS_LSTM_neurosymbolic" and not demande_nulle:
        vrais = [ox.libelle_symbolique(c) for c in etats_transmis if etat["symboliques"].get(c)]
        st.markdown(tr("**NS-LSTM : l'ontologie informe le réseau**", "**NS-LSTM: the ontology informs the network**"))
        st.markdown(
            flux_html([
                (tr("{w} dernières secondes", "Last {w} seconds", w=LSTM_WINDOW), COULEUR_REFERENCE),
                (tr("+ états déduits par l'ontologie : {e}", "+ states inferred by the ontology: {e}",
                    e=", ".join(vrais) or tr("aucun n'est vrai", "none is true")), COULEUR_REFERENCE),
                (tr("Réseau LSTM", "LSTM network"), COULEUR_SECONDAIRE),
                (f"alpha {_pct(alpha_final)}", COULEUR_DECISION),
            ]),
            unsafe_allow_html=True,
        )

    if etats_transmis:
        st.markdown(
            tr("États déduits par l'ontologie et transmis au réseau : ", "States inferred by the ontology and passed to the network: ")
            + ", ".join(f"{ox.libelle_symbolique(c)} **{oui if etat['symboliques'][c] else non}**" for c in etats_transmis)
            + "."
        )
    elif strategie == "EMS_fuzzy_logic":
        concepts = ox.concepts_regle_floue(dominante)
        if concepts and not regle_par_defaut:
            st.markdown(
                tr("Concepts de l'ontologie utilisés par la règle qui pèse le plus : ",
                   "Ontology concepts used by the rule that weighs most: ")
                + ", ".join(nom for nom, _ in concepts) + "."
            )

    st.markdown(tr("**Ce qu'en dit l'ontologie OntoHESS**", "**What the OntoHESS ontology says**"))
    st.markdown(tr("État de fonctionnement déduit : **{e}**.", "Inferred operating state: **{e}**.", e=etat["libelle"]))
    activees, non_activees, _ = ox.evaluer_regles(p_dem, soc_eb, soc_pb)
    for r in sorted((r for r in activees if r["type"] in ("mode", "repartition")), key=lambda r: r["type"]):
        premisses = tr(" et ", " and ").join(f"`{x['texte']}`" for x in r["details"])
        st.markdown(tr("- **{i}** — si {p}, alors {l}.", "- **{i}** — if {p}, then {l}.", i=r["id"], p=premisses, l=r["lecture"]))
    if onto is not None and not demande_nulle:
        ecart = (alpha_final - onto["alpha"]) * 100
        position = (
            tr("identique à", "identical to") if abs(ecart) < 0.5
            else tr("{x} points au-dessus de", "{x} points above", x=nombre(ecart, 1)) if ecart > 0
            else tr("{x} points en dessous de", "{x} points below", x=nombre(-ecart, 1))
        )
        st.markdown(tr(
            "La règle de répartition {i} prescrit **{a}** pour la batterie Puissance ; la décision de la stratégie est {p} cette référence.",
            "The split rule {i} prescribes **{a}** for the Power battery; the strategy's decision is {p} this reference.",
            i=onto["regle"]["id"], a=_pct(onto["alpha"]), p=position,
        ))
        st.caption(tr(
            "Ces règles décrivent la conduite de référence (la batterie Énergie d'abord, dans ses "
            "limites) ; s'en écarter n'est pas une erreur. Toutes les règles sont dans « Base de connaissances ».",
            "These rules describe the reference behaviour (the Energy battery first, within its limits); "
            "departing from it is not an error. All the rules are in “Knowledge base”.",
        ))

    paire = [c for c in ("EMS_MLP_neurosymbolic", "EMS_LSTM_neurosymbolic") if c in resultats]
    if len(paire) == 2 and not demande_nulle:
        with st.expander(tr("Comparer NS-MLP et NS-LSTM au même instant", "Compare NS-MLP and NS-LSTM at the same time step")):
            colonnes_ns = st.columns(2)
            decisions = {}
            for col, cle in zip(colonnes_ns, paire):
                trj = resultats[cle]
                se, sp = float(trj["SOC_EB"][instant]), float(trj["SOC_PB"][instant])
                a_req_ns, a_fin_ns = float(trj["alpha_requested"][instant]), float(trj["alpha_final"][instant])
                pe, pp = float(trj["P_EB"][instant]), float(trj["P_PB"][instant])
                corr_ns = bool(trj["correction_applied"][instant])
                decisions[cle] = a_fin_ns
                with col:
                    with st.container(border=True):
                        st.markdown(f"#### {nom_affichage(cle)}")
                        if cle == "EMS_MLP_neurosymbolic":
                            rf = alpha_fuzzy_calc(np.array([se]), np.array([sp]), np.array([p_dem]), np.array([accel]))
                            a_f, a_r, garde_ns = decomposition_ns(cle, instant)
                            lignes = [
                                tr("Règles floues (menées par {g}) : **{a}**", "Fuzzy rules (led by {g}): **{a}**",
                                   g=_regle_floue(str(rf["dominant_rule"][0])), a=_pct(a_f)),
                                tr("Correction du réseau : **{c}**", "Network correction: **{c}**", c=_points(a_r - a_f)),
                            ]
                            if garde_ns:
                                lignes.append(tr("Garde-fou de l'ontologie : **{c}**", "Ontology safeguard: **{c}**", c=_points(a_req_ns - a_r)))
                            st.markdown("  \n".join(lignes))
                        else:
                            _, contribs_ns, _ = _contributions(cle, instant, _signature(cle))
                            total = sum(abs(v) for _, v in contribs_ns) or 1.0
                            part = sum(abs(v) for c, v in contribs_ns if c in xai.ETATS_SYMBOLIQUES) / total * 100
                            principale = max(
                                ((c, v) for c, v in contribs_ns if c not in xai.ETATS_SYMBOLIQUES),
                                key=lambda kv: abs(kv[1]), default=("—", 0.0),
                            )
                            st.markdown("  \n".join([
                                tr("États déduits par l'ontologie : **{p} %** de la décision",
                                   "States inferred by the ontology: **{p} %** of the decision", p=nombre(part, 0)),
                                tr("Grandeur qui pèse le plus : {g} ({v})", "Quantity that weighs most: {g} ({v})",
                                   g=xai.libelle_entree(principale[0]), v=_points(principale[1])),
                            ]))
                        st.markdown(tr(
                            "Décision : **{a}** pour la batterie Puissance · EB {pe} · PB {pp}",
                            "Decision: **{a}** for the Power battery · EB {pe} · PB {pp}",
                            a=_pct(a_fin_ns), pe=kw(pe), pp=kw(pp),
                        ))
                        st.markdown("  \n".join([
                            _coche(not corr_ns, tr("acceptée par le filtre de sécurité", "accepted by the safety filter") if not corr_ns
                                   else tr("corrigée par le filtre de sécurité", "corrected by the safety filter")),
                            _coche(se >= SOC_EB_MIN and sp >= SOC_PB_MIN, tr("SOC dans leurs limites", "SOCs within their limits")),
                            _coche(P_EB_MIN_W - 1 <= pe <= P_EB_MAX_W + 1,
                                   tr("batterie Énergie dans ses limites de puissance", "Energy battery within its power limits")),
                            _coche(float(trj["P_unserved"][instant]) < 1.0, tr("demande entièrement fournie", "demand fully met")),
                        ]))
            a1, a2 = decisions[paire[0]], decisions[paire[1]]
            st.caption(tr(
                "Écart de {e} points entre les deux décisions : NS-MLP suit ses règles puis les corrige "
                "à l'instant présent ; NS-LSTM raisonne sur les {w} dernières secondes, où les états "
                "déduits par l'ontologie ne sont qu'une information parmi d'autres.",
                "Gap of {e} points between the two decisions: NS-MLP follows its rules then corrects "
                "them at the present time; NS-LSTM reasons over the last {w} seconds, where the states "
                "inferred by the ontology are only one piece of information among others.",
                e=nombre(abs(a1 - a2) * 100, 1), w=LSTM_WINDOW,
            ))


# Onglet « Réseau de neurones » : la part apprise de la décision

with onglet_res:
    if strategie in ("EMS_power_limitation", "EMS_fuzzy_logic"):
        st.info(tr(
            "Cette stratégie n'a pas de réseau de neurones : sa décision est entièrement lisible "
            "(onglets Raisons et Connaissances expertes).",
            "This strategy has no neural network: its decision can be read in full (Reasons and Expert "
            "knowledge tabs).",
        ))
    elif demande_nulle:
        st.info(tr("Demande quasi nulle : pas de décision à expliquer.", "Near-zero demand: no decision to explain."))
    elif strategie == "EMS_MLP_neurosymbolic":
        delta = alpha_reseau - alpha_flou
        r1, r2, r3 = st.columns(3)
        r1.metric(tr("Correction du réseau", "Network correction"), f"{nombre(delta * 100, 1, signe=True)} pts")
        r2.metric(tr("Part de la correction permise", "Share of the allowed correction"), _pct(abs(delta) / MLP_NS_MAX_DELTA, 0))
        r3.metric(tr("Garde-fou de l'ontologie", "Ontology safeguard"),
                  tr("intervient", "steps in") if garde_actif else tr("n'intervient pas", "does not step in"))
        st.markdown(tr(
            "Le réseau ne décide pas seul : il ajoute aux règles floues une correction limitée à ±{m} "
            "points. Le calcul de cette correction n'est pas lisible, mais son poids dans la décision est "
            "connu exactement. Quand le SOC de la batterie Puissance passe sous {r} %, un garde-fou de "
            "l'ontologie (règles R14 et R16) limite en traction sa part à ce que la batterie Énergie ne "
            "peut pas fournir.",
            "The network does not decide alone: it adds to the fuzzy rules a correction limited to ±{m} "
            "points. The calculation of this correction cannot be read, but its weight in the decision "
            "is known exactly. When the Power battery's SOC drops below {r} %, an ontology safeguard "
            "(rules R14 and R16) limits its share in traction to what the Energy battery cannot supply.",
            m=nombre(MLP_NS_MAX_DELTA * 100, 0), r=nombre(MLP_NS_RESERVE_PB_SOC * 100, 0),
        ))
    else:
        st.markdown(tr(
            "Comment l'explication est reconstruite : on remplace tour à tour chacune des {k} grandeurs "
            "d'entrée par sa valeur moyenne sur le cycle, dans toutes les combinaisons possibles, et on "
            "mesure de combien la décision change (méthode de Shapley). La situation moyenne plus les "
            "contributions redonnent exactement la décision du réseau ({a} + contributions = {b}).",
            "How the explanation is reconstructed: each of the {k} input quantities is replaced in turn "
            "by its average value over the cycle, in all possible combinations, and we measure how much "
            "the decision changes (Shapley method). The average situation plus the contributions give "
            "back exactly the network's decision ({a} + contributions = {b}).",
            k=len(contribs), a=_pct(reference), b=_pct(alpha_explique),
        ))
        if strategie in ("EMS_LSTM", "EMS_LSTM_neurosymbolic"):
            st.caption(tr(
                "Chaque grandeur compte pour toute son évolution sur les {w} dernières secondes.",
                "Each quantity counts for its whole evolution over the last {w} seconds.", w=LSTM_WINDOW,
            ))
        if strategie == "EMS_LSTM_neurosymbolic":
            total = sum(abs(v) for _, v in contribs) or 1.0
            st.metric(tr("Poids des états déduits par l'ontologie", "Weight of the states inferred by the ontology"),
                      f"{nombre(sum(abs(v) for c, v in contribs if c in xai.ETATS_SYMBOLIQUES) / total * 100, 0)} %")
        if strategie == "EMS_GNN":
            st.markdown(tr(
                "**Poids de chaque composant du HESS dans la décision** (sensibilité de la décision à ses grandeurs)",
                "**Weight of each HESS component in the decision** (sensitivity of the decision to its quantities)",
            ))
            poids_c, liaisons = _poids_composants_gnn(p_dem, soc_eb, soc_pb, accel)
            st.plotly_chart(_schema_gnn(poids_c, liaisons), width="stretch")


# Onglet « Et si… ? » : que déciderait la stratégie si la situation changeait un peu ?

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


@st.cache_data(show_spinner=False)
def _et_si(strategie, instant, signature):
    s = xai.situation(df, resultats[strategie], instant)
    base = xai.alpha_decision(strategie, df, resultats[strategie], instant)
    sortie = []
    for grandeur, signe in SCENARIOS:
        pas = 1000.0 if grandeur == "hasPower" else 0.05
        valeur = s[grandeur] + signe * pas
        if grandeur != "hasPower":
            valeur = float(np.clip(valeur, 0.0, 1.0))
        a = xai.alpha_decision(strategie, df, resultats[strategie], instant, {grandeur: valeur})
        p = valeur if grandeur == "hasPower" else s["hasPower"]
        sortie.append((grandeur, signe, a, a * p))
    return base, base * s["hasPower"], sortie


with onglet_etsi:
    if demande_nulle:
        st.info(tr("Demande quasi nulle : choisissez un instant de traction ou de freinage.",
                   "Near-zero demand: choose a time step in traction or braking."))
    else:
        with st.spinner(tr("Calcul des variantes…", "Computing the variants…")):
            base_a, base_p, variantes = _et_si(strategie, instant, _signature(strategie))
        st.markdown(tr(
            "Que déciderait **{s}** si la situation changeait un peu ? Ici : **{a}** pour la batterie Puissance ({p}).",
            "What would **{s}** decide if the situation changed slightly? Here: **{a}** for the Power battery ({p}).",
            s=nom_affichage(strategie), a=_pct(base_a), p=kw(base_p),
        ))
        st.dataframe(
            [
                {
                    tr("Variante", "Variant"): lib(SCENARIOS[(grandeur, signe)]),
                    tr("Part de la batterie Puissance", "Power battery share"): _pct(a),
                    tr("Puissance de la batterie Puissance", "Power battery power"): kw(p),
                    tr("Variation de cette puissance", "Change in that power"): f"{nombre((p - base_p) / 1000, 2, signe=True)} kW",
                    tr("Sens conforme à la physique", "Direction consistent with physics"):
                        "✓" if SENS_ATTENDU[grandeur] * signe * (p - base_p) >= -0.01 * abs(p_dem) else "✗",
                }
                for grandeur, signe, a, p in variantes
            ],
            hide_index=True, width="stretch",
        )
        st.caption(tr(
            "Sens attendu : plus de charge dans la batterie Puissance, ou plus de demande, ne doit pas "
            "réduire la puissance qu'elle fournit ; plus de charge dans la batterie Énergie ne doit pas "
            "l'augmenter (à 1 % de la demande près). C'est le test de cohérence physique E3, appliqué à cet instant.",
            "Expected direction: more charge in the Power battery, or more demand, must not reduce the "
            "power it supplies; more charge in the Energy battery must not increase it (to within 1 % of "
            "the demand). This is the E3 physical-consistency test, applied at this time step.",
        ))
        st.markdown(tr("**D'après les seuils de l'ontologie**", "**According to the ontology's thresholds**"))
        st.markdown("\n".join(f"- {phrase}" for phrase in ox.contrefactuels(p_dem, soc_eb, soc_pb)))
        echecs = [
            f"**{r['id']}** ({r['lecture']}) : "
            + " ; ".join(tr("`{c}` n'est pas vérifié", "`{c}` does not hold", c=x["texte"]) for x in r["details"] if x["ok"] is False)
            for r in non_activees if r["type"] in ("mode", "repartition")
        ]
        if echecs:
            with st.expander(tr("Règles de l'ontologie qui ne s'appliquent pas, et pourquoi", "Ontology rules that do not apply, and why")):
                st.markdown("\n".join(f"- {e}" for e in echecs))


# Onglet « Sur tout le cycle » : bilan de l'explicabilité

@st.cache_data(show_spinner=False)
def _analyse_cycle(p, accel_c, soc_eb_c, soc_pb_c, a_req, a_fin, corr):
    en_service = np.abs(p) > EPS_POWER_W
    flou = alpha_fuzzy_calc(soc_eb_c, soc_pb_c, p, accel_c)
    return {
        "actif": en_service,
        "alpha_flou": np.asarray(flou["alpha"], dtype=float),
        "forces": np.asarray(flou["strengths"], dtype=float),
        "dominante": np.asarray(flou["dominant_rule"]).astype(str),
        "alpha_onto": ox.alpha_ontologie_vect(p, soc_eb_c),
        "a_req": a_req, "a_fin": a_fin, "corr": corr,
    }


def _par_minute(y, pas=60):
    k = len(y) // pas * pas
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(np.asarray(y[:k], dtype=float).reshape(-1, pas), axis=1)


with onglet_cycle:
    cy = _analyse_cycle(
        df["hasPower"].to_numpy(dtype=float)[:n],
        (df["hasAcceleration"].to_numpy(dtype=float) if "hasAcceleration" in df.columns else np.zeros(len(df)))[:n],
        np.asarray(traj["SOC_EB"], dtype=float)[:n],
        np.asarray(traj["SOC_PB"], dtype=float)[:n],
        np.asarray(traj["alpha_requested"] if "alpha_requested" in traj else traj["alpha_final"], dtype=float)[:n],
        np.asarray(traj["alpha_final"], dtype=float)[:n],
        np.asarray(traj["correction_applied"], dtype=float)[:n],
    )
    m = cy["actif"] & ~np.isnan(cy["alpha_onto"])
    ecart_onto = np.abs(cy["a_fin"][m] - cy["alpha_onto"][m])

    e3, e3_detail = _coherence(strategie, _signature(strategie))
    k1, k2, k3, k4 = st.columns(4)
    k1.metric(tr("Explication", "Explanation"), tr("exacte", "exact") if exacte else tr("reconstruite", "reconstructed"))
    k2.metric(tr("Cohérence physique (E3)", "Physical consistency (E3)"), _pct(e3, 0))
    k3.metric(tr("Accord avec la règle de l'ontologie", "Agreement with the ontology rule"), _pct(np.mean(ecart_onto < 0.05), 0))
    k4.metric(tr("Décisions corrigées par le filtre", "Decisions corrected by the filter"), _pct(np.mean(cy["corr"][cy["actif"]])))
    st.caption(tr(
        "Cohérence physique : part de 120 instants de traction où la décision évolue dans le sens "
        "attendu ({d}). Accord avec la règle de l'ontologie : part des instants où la décision reste à "
        "moins de 5 points de la répartition prescrite par OntoHESS (écart moyen : {e} points).",
        "Physical consistency: share of 120 traction time steps where the decision moves in the "
        "expected direction ({d}). Agreement with the ontology rule: share of time steps where the "
        "decision stays within 5 points of the split prescribed by OntoHESS (mean gap: {e} points).",
        d=" ; ".join(f"{xai.libelle_contrainte(k)} : {_pct(v, 0)}" for k, v in e3_detail.items()),
        e=nombre(np.mean(ecart_onto) * 100, 1),
    ))

    avec_regles = strategie in ("EMS_fuzzy_logic", "EMS_MLP_neurosymbolic")
    if avec_regles:
        a = cy["actif"]
        j1, j2, j3 = st.columns(3)
        j1.metric(tr("Règles floues vraies par décision", "Fuzzy rules true per decision"),
                  nombre(np.mean(np.sum(cy["forces"][a] > 0.05, axis=1)), 1))
        if strategie == "EMS_MLP_neurosymbolic":
            a_reseau = np.asarray(traj.get("alpha_reseau", cy["a_req"]), dtype=float)[:n]
            delta = np.abs(a_reseau[a] - cy["alpha_flou"][a])
            j2.metric(tr("Part de la correction permise utilisée", "Share of the allowed correction used"),
                      _pct(np.nanmean(delta) / MLP_NS_MAX_DELTA, 0))
            if "garde_fou" in traj:
                j3.metric(
                    tr("Décisions reprises par le garde-fou", "Decisions taken over by the safeguard"),
                    _pct(np.mean(np.asarray(traj["garde_fou"], dtype=bool)[:n][a])),
                    help=tr("Instants de traction où la batterie Puissance était sous {r} % de SOC.",
                            "Traction time steps where the Power battery was below {r} % SOC.",
                            r=nombre(MLP_NS_RESERVE_PB_SOC * 100, 0)),
                )
            else:
                j3.metric(tr("Décisions laissées aux règles seules", "Decisions left to the rules alone"), _pct(np.mean(delta < 0.02)))

    temps = df["time"].to_numpy(dtype=float)[:n] if "time" in df.columns else np.arange(n, dtype=float)
    xm = _par_minute(temps) / 60.0
    courbes = [
        go.Scatter(x=xm, y=_par_minute(np.where(cy["actif"], cy["a_fin"], np.nan)) * 100,
                   name=tr("Décision de {s}", "Decision of {s}", s=nom_affichage(strategie)), line=dict(color=COULEUR_DECISION, width=2.2)),
        go.Scatter(x=xm, y=_par_minute(cy["alpha_onto"]) * 100, name=tr("Règle de l'ontologie", "Ontology rule"),
                   line=dict(color=COULEUR_REFERENCE, width=1.6, dash="dash")),
    ]
    if strategie == "EMS_MLP_neurosymbolic":
        courbes.append(
            go.Scatter(x=xm, y=_par_minute(np.where(cy["actif"], cy["alpha_flou"], np.nan)) * 100,
                       name=tr("Règles floues", "Fuzzy rules"), line=dict(color=C_GRIS, width=1.6, dash="dot"))
        )
    fig_c = go.Figure(courbes)
    fig_c.update_layout(
        separators=separateurs_plotly(), height=340, margin=dict(t=20, b=40, l=10, r=10), hovermode="x unified",
        xaxis_title=tr("Temps (min)", "Time (min)"),
        yaxis_title=tr("Part confiée à la batterie Puissance (%, moyenne par minute)", "Share assigned to the Power battery (%, mean per minute)"),
        legend=dict(orientation="h", y=1.1),
    )
    st.markdown(tr("**Décision et règle de référence au fil du cycle**", "**Decision and reference rule over the cycle**"))
    st.plotly_chart(fig_c, width="stretch")

    if avec_regles:
        st.markdown(tr("**Règles floues qui pèsent le plus sur le cycle**", "**Fuzzy rules that weigh most over the cycle**"))
        dom = cy["dominante"][cy["actif"]]
        cles_regles, comptes = np.unique(dom, return_counts=True)
        ordre = list(np.argsort(comptes)[::-1])
        parts = [comptes[i] / comptes.sum() * 100 for i in ordre]
        st.plotly_chart(
            _barres_h(
                [_regle_floue(cles_regles[i]) for i in ordre], parts, [C_GRIS] * len(ordre),
                tr("Part des instants où la règle pèse le plus (%)", "Share of time steps where the rule weighs most (%)"),
                [f"{nombre(v, 0)} %" for v in parts],
            ),
            width="stretch",
        )


pied_navigation("vues/7_Explicabilite.py")
