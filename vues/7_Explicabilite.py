import sys
import warnings
from pathlib import Path

DOSSIER_PROJET = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DOSSIER_PROJET))

import numpy as np
import torch
import plotly.graph_objects as go
import streamlit as st
from core.format import SEPARATEURS_PLOTLY, nombre

from ems_core import (
    alpha_fuzzy_calc,
    construire_graphe_instant,
    load_gnn_simple,
    charger_scaler,
    FUZZY_RULE_NAMES,
    RULE_LABELS_FR,
    MLP_NS_INPUT_COLS,
    MLP_NS_MAX_DELTA,
    MLP_NS_RESERVE_PB_SOC,
    LSTM_NS_FEATURE_COLS,
    LSTM_WINDOW,
    GNN_SCALER_FILE,
    GNN_NODE_NAMES,
    DEVICE,
    ALPHA_GRID_STEP,
    EPS_POWER_W,
    P_EB_MAX_W,
    P_EB_MIN_W,
    SOC_EB_MIN,
    SOC_PB_MIN,
)
from core.resultats import assurer_donnees_session, nom_affichage
from core.navigation import pied_navigation
from core.instant import choisir_instant
from core.style import (
    couleur,
    flux_html,
    COULEUR_DECISION,
    COULEUR_DEMANDE,
    COULEUR_EB,
    COULEUR_PB,
    COULEUR_REFERENCE,
    COULEUR_SECONDAIRE,
)
from core import ontology_explainer as ox
from core import xai


# Palette commune (core/style.py) : une couleur = une signification.
C_EB, C_PB, C_GRIS = COULEUR_EB, COULEUR_PB, COULEUR_SECONDAIRE

LABELS_NOEUDS = {
    "energy_battery": "Batterie Énergie",
    "power_battery": "Batterie Puissance",
    "converter": "Convertisseur",
    "motor": "Moteur",
    "vehicle": "Véhicule",
}

# Nature de l'explication : exacte quand elle décrit le calcul lui-même,
# approchée quand elle est reconstruite après coup sur un réseau opaque.
NATURE = {
    "EMS_power_limitation": (True, "La règle appliquée est connue : l'explication est le calcul lui-même."),
    "EMS_fuzzy_logic": (True, "Les règles floues activées et leur force sont le calcul lui-même."),
    "EMS_MLP_neurosymbolic": (
        True,
        "La décision se décompose exactement en base floue + correction du réseau ; seule "
        "la correction, bornée à ±20 points, vient d'un calcul opaque.",
    ),
    "EMS_MLP": (False, "Réseau opaque : l'explication est reconstruite après coup par valeurs de Shapley exactes."),
    "EMS_LSTM": (False, "Réseau opaque : l'explication est reconstruite après coup par valeurs de Shapley exactes."),
    "EMS_LSTM_neurosymbolic": (
        False,
        "Réseau opaque : l'explication est reconstruite après coup par valeurs de Shapley "
        "exactes ; ses entrées symboliques ont un sens métier, ce qui la rend plus lisible.",
    ),
    "EMS_GNN": (False, "Réseau opaque : l'explication est reconstruite après coup par valeurs de Shapley exactes."),
}


# Explications des réseaux et cohérence physique (core/xai.py), mises en cache.
# La signature (empreinte de la trajectoire) invalide le cache si les données changent.

def _signature(strategie):
    return float(np.nansum(resultats[strategie]["alpha_final"]))


@st.cache_data(show_spinner="Calcul des valeurs de Shapley…")
def _shapley(strategie, instant, signature):
    return xai.shapley(strategie, df, resultats[strategie], instant)


@st.cache_data(show_spinner="Mesure de la cohérence physique (E3)…")
def _mesurer_coherence(strategie, signature):
    return xai.coherence_physique(strategie, df, resultats[strategie])


def _coherence(strategie, signature):
    """E3, lu dans les résultats précalculés quand il y figure, mesuré sinon."""
    precalcule = st.session_state.get("coherence_simulation") or {}
    if strategie in precalcule:
        return precalcule[strategie]
    return _mesurer_coherence(strategie, signature)


@st.cache_resource(show_spinner=False)
def _charger_gnn():
    res = load_gnn_simple()
    modele = res[0] if isinstance(res, tuple) else res
    modele.eval()
    return modele, charger_scaler(GNN_SCALER_FILE)


def _attribution_gnn(p_dem, soc_eb, soc_pb, accel):
    """Importance (%) de chaque nœud du graphe du HESS."""
    modele, scaler = _charger_gnn()
    x_g, edge = construire_graphe_instant(p_dem, soc_eb, soc_pb, accel, scaler)
    x_g = x_g.to(DEVICE).clone().requires_grad_(True)
    edge = edge.to(DEVICE)
    modele(x_g, edge, torch.zeros(x_g.shape[0], dtype=torch.long, device=DEVICE)).sum().backward()
    imp = np.abs((x_g.grad * x_g).detach().cpu().numpy()).sum(axis=1)
    return (imp / imp.sum() * 100.0 if imp.sum() > 0 else imp), edge


# Graphiques

def _barres_h(etiquettes, valeurs, couleurs, titre_x, hover):
    fig = go.Figure(
        go.Bar(
            y=etiquettes[::-1], x=valeurs[::-1], orientation="h",
            marker_color=couleurs[::-1], hovertemplate=hover,
        )
    )
    fig.update_layout(
        separators=SEPARATEURS_PLOTLY, height=36 * len(etiquettes) + 70, margin=dict(t=10, b=40, l=10, r=20),
        xaxis_title=titre_x, showlegend=False,
    )
    return fig


def _cascade(etapes, alpha_final):
    """Décomposition de alpha (en % confiés à la PB), de la première étape à la
    décision appliquée. etapes : [(libellé, valeur)] — la première est un point
    de départ, les suivantes des ajustements."""
    x = [e[0] for e in etapes] + ["Décision appliquée"]
    y = [e[1] * 100 for e in etapes] + [0.0]
    mesures = ["absolute"] + ["relative"] * (len(etapes) - 1) + ["total"]
    textes = [f"{nombre(y[0], 1)} %"] + [f"{nombre(v, 1, signe=True)} pts" for v in y[1:-1]] + [f"{nombre(alpha_final * 100, 1)} %"]
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
        separators=SEPARATEURS_PLOTLY, height=340, margin=dict(t=20, b=40, l=10, r=10), showlegend=False,
        yaxis=dict(title="Part confiée à la PB (%)", range=[0, 110]),
    )
    return fig


def _graphe_gnn(pct_g, edge):
    """Schéma du HESS, nœuds colorés selon leur importance dans la décision."""
    positions = [(0.0, 1.0), (0.0, -1.0), (1.2, 0.0), (2.4, 0.0), (3.6, 0.0)]
    ei = edge.detach().cpu().numpy()
    edge_x, edge_y = [], []
    for k in range(ei.shape[1]):
        a, b = int(ei[0, k]), int(ei[1, k])
        if a < len(positions) and b < len(positions):
            edge_x += [positions[a][0], positions[b][0], None]
            edge_y += [positions[a][1], positions[b][1], None]
    labels = [LABELS_NOEUDS.get(n, n) for n in GNN_NODE_NAMES]
    fig = go.Figure(
        [
            go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(color="#C7CCD6", width=2), hoverinfo="skip"),
            go.Scatter(
                x=[p[0] for p in positions], y=[p[1] for p in positions], mode="markers+text",
                marker=dict(
                    size=[34 + p * 0.9 for p in pct_g], color=list(pct_g), colorscale="Blues",
                    showscale=True, colorbar=dict(title="%"), line=dict(color="white", width=2),
                ),
                text=[f"{l}<br>{nombre(p, 0)} %" for l, p in zip(labels, pct_g)],
                textposition="bottom center", hoverinfo="text",
            ),
        ]
    )
    fig.update_layout(
        separators=SEPARATEURS_PLOTLY, height=360, showlegend=False, margin=dict(t=20, b=20, l=10, r=10),
        xaxis=dict(visible=False), yaxis=dict(visible=False, range=[-1.8, 1.6]),
    )
    return fig


def kw(x):
    return f"{nombre(x / 1000.0, 1)} kW"


# Interface

st.title("💡 Pourquoi cette décision ?")
st.caption(
    "Pour un instant du cycle et une stratégie : quelle décision a été prise, et pourquoi. "
    "L'explication s'adapte au modèle : exacte quand on connaît son calcul, approchée "
    "quand le modèle est opaque."
)

try:
    assurer_donnees_session(st)
except FileNotFoundError as exc:
    st.error(str(exc))
    st.info("Lancez une fois le précalcul :  `python scripts/run_simulations.py`")
    st.stop()

resultats = st.session_state.get("resultats_simulation")
df = st.session_state.get("cycle_pret")
if not resultats or df is None:
    st.warning("Aucune donnée disponible.")
    st.stop()

noms = list(resultats.keys())
n = min([len(df)] + [len(traj["P_EB"]) for traj in resultats.values()])

col_t, col_s = st.columns([2, 1])
instant, t_sel = choisir_instant(df, n, col_t)
with col_s:
    strategie = st.selectbox("Modèle", noms, format_func=nom_affichage)

traj = resultats[strategie]
speed = float(df["speed"].iloc[instant]) if "speed" in df.columns else 0.0
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
exacte, nature_txt = NATURE.get(strategie, (False, ""))

# Ingrédients de l'explication, calculés une fois pour toute la page
res_flou = alpha_fuzzy_calc(np.array([soc_eb]), np.array([soc_pb]), np.array([p_dem]), np.array([accel]))
alpha_flou = float(res_flou["alpha"][0])
forces = np.asarray(res_flou["strengths"][0], dtype=float)
contrib = ox.contributions_floues(forces)
dominante = str(res_flou["dominant_rule"][0])
avec_shapley = strategie in xai.SHAPLEY_DISPONIBLE and not demande_nulle
if avec_shapley:
    ref_shap, contribs, alpha_explique = _shapley(strategie, instant, _signature(strategie))
    principales = sorted(contribs, key=lambda kv: -abs(kv[1]))
entrees_symb = {
    "EMS_MLP_neurosymbolic": MLP_NS_INPUT_COLS,
    "EMS_LSTM_neurosymbolic": LSTM_NS_FEATURE_COLS,
}.get(strategie, [])
etats_transmis = [c for c in ox.LIBELLES_SYMBOLIQUES if c in entrees_symb]


@st.cache_data(show_spinner=False)
def _decomposer_ns(instant, signature):
    return xai.decomposer_ns_mlp(df, resultats["EMS_MLP_neurosymbolic"], instant)


def decomposition_ns(cle_traj, i):
    """(alpha base floue, alpha après correction du réseau, garde-fou actif) de NS-MLP,
    lus dans la trajectoire si la simulation les a gardés, recalculés sinon."""
    tr = resultats[cle_traj]
    if "alpha_reseau" in tr and np.isfinite(tr["alpha_reseau"][i]):
        return float(tr["alpha_flou"][i]), float(tr["alpha_reseau"][i]), bool(tr["garde_fou"][i])
    d = _decomposer_ns(i, _signature(cle_traj))
    return d["alpha_flou"], d["alpha_reseau"], d["garde_fou"]


if strategie == "EMS_MLP_neurosymbolic" and not demande_nulle:
    alpha_flou, alpha_reseau, garde_actif = decomposition_ns(strategie, instant)
else:
    alpha_reseau, garde_actif = alpha_req, False


def _lib_regle(cle):
    return ox.REGLES_FLOUES.get(cle, (cle,))[0]


def _entree(col):
    """Nom lisible d'une entrée de réseau, dans une phrase (les sigles gardent leur casse)."""
    lib = xai.LIBELLES_ENTREES.get(col, col)
    return lib if lib[:2].isupper() else lib[0].lower() + lib[1:]


# Aucune règle floue ne s'active nettement : le moteur applique sa répartition par défaut.
regle_par_defaut = dominante not in FUZZY_RULE_NAMES or float(forces.max(initial=0.0)) < 1e-3


# La décision, tout en haut

with st.container(border=True):
    st.markdown(
        f"**{nom_affichage(strategie)}** · instant t = {nombre(t_sel, 0)} s · "
        f"vitesse {nombre(speed * 3.6, 0)} km/h · état inféré : {etat['libelle'][0].lower() + etat['libelle'][1:]}"
    )
    d = st.columns(6)
    d[0].metric("Demande", kw(p_dem))
    d[1].metric("SOC EB", f"{nombre(soc_eb * 100, 1)} %")
    d[2].metric("SOC PB", f"{nombre(soc_pb * 100, 1)} %")
    d[3].metric("alpha (part PB)", f"{nombre(alpha_final * 100, 1)} %")
    d[4].metric("Batterie Énergie", kw(p_eb))
    d[5].metric("Batterie Puissance", kw(p_pb))


# Pourquoi cette décision ? 3 à 5 raisons lisibles

def _raisons():
    if demande_nulle:
        return ["La demande est quasi nulle : aucune batterie n'est sollicitée, il n'y a pas de répartition à expliquer."]
    r = []
    if p_dem > 0:
        r.append(
            f"La demande de traction est de **{kw(p_dem)}**, "
            + ("au-delà de" if p_dem > P_EB_MAX_W else "dans")
            + f" la limite de la batterie Énergie ({kw(P_EB_MAX_W)})."
        )
    else:
        r.append(
            f"Le véhicule freine : **{kw(-p_dem)}** sont à récupérer"
            + (", au-delà de ce que la batterie Énergie peut absorber" if p_dem < P_EB_MIN_W else "")
            + f" (limite {kw(-P_EB_MIN_W)})."
        )
    ecart = (soc_eb - soc_pb) * 100
    r.append(
        f"Le SOC de la batterie Énergie ({nombre(soc_eb * 100, 0)} %) est "
        + ("supérieur à" if ecart > 1 else "inférieur à" if ecart < -1 else "proche de")
        + f" celui de la batterie Puissance ({nombre(soc_pb * 100, 0)} %)."
    )
    if strategie == "EMS_MLP_neurosymbolic":
        r.append(
            (
                "Aucune règle floue ne s'active nettement : la base floue propose sa répartition par défaut, "
                if regle_par_defaut
                else f"La base floue, menée par la règle **{_lib_regle(dominante)}**, propose "
            )
            + f"**{nombre(alpha_flou * 100, 1)} %** pour la PB."
        )
        r.append(
            f"Le réseau applique une correction limitée de **{nombre((alpha_reseau - alpha_flou) * 100, 1, signe=True)} points** "
            f"(borne ±{nombre(MLP_NS_MAX_DELTA * 100, 0)})."
        )
        if garde_actif:
            r.append(
                f"L'ontologie déclare la batterie Puissance faible (SOC {nombre(soc_pb * 100, 0)} % ≤ "
                f"{nombre(MLP_NS_RESERVE_PB_SOC * 100, 0)} %) : le **garde-fou symbolique** applique les règles "
                f"R14/R16 d'OntoHESS et ramène sa part à **{nombre(alpha_req * 100, 1)} %**."
            )
    elif strategie == "EMS_LSTM_neurosymbolic":
        actifs = [ox.LIBELLES_SYMBOLIQUES[c].lower() for c in etats_transmis if etat["symboliques"].get(c)]
        total = sum(abs(v) for _, v in contribs) or 1.0
        part = sum(abs(v) for c, v in contribs if c in xai.ETATS_SYMBOLIQUES) / total * 100
        r.append(
            f"États symboliques transmis au réseau : {', '.join(actifs) or 'aucun actif'} ; "
            f"ils pèsent {nombre(part, 0)} % de la décision."
        )
        top = next(((c, v) for c, v in principales if c not in xai.ETATS_SYMBOLIQUES), ("—", 0.0))
        r.append(
            f"Le LSTM, qui lit les {LSTM_WINDOW} dernières secondes, s'appuie surtout sur "
            f"**{_entree(top[0])}** ({nombre(top[1] * 100, 1, signe=True)} points)."
        )
    else:
        r.append(f"L'ontologie infère l'état « {etat['libelle']} ».")
        if strategie == "EMS_power_limitation" and onto is not None:
            r.append(f"La règle **{onto['regle']['id']}** d'OntoHESS s'applique : {onto['regle']['lecture']}.")
        elif strategie == "EMS_fuzzy_logic" and regle_par_defaut:
            r.append(
                "Aucune règle floue ne s'active nettement : le moteur applique sa répartition par "
                f"défaut ({nombre(alpha_flou * 100, 0)} % pour la PB)."
            )
        elif strategie == "EMS_fuzzy_logic":
            i_dom = list(FUZZY_RULE_NAMES).index(dominante)
            r.append(
                f"La règle floue dominante est **{_lib_regle(dominante)}** "
                f"({nombre(forces[i_dom] * 100, 0)} % d'activation) : {RULE_LABELS_FR.get(dominante, '')}."
            )
        elif avec_shapley:
            fortes = [(_entree(c), v) for c, v in principales[:2] if abs(v) >= 5e-4]
            r.append(
                "Le réseau s'appuie surtout sur "
                + " et ".join(f"**{lib}** ({nombre(v * 100, 1, signe=True)} points)" for lib, v in fortes)
                + "."
            )
    r.append(
        "Le filtre de sécurité a **corrigé** la proposition pour respecter les limites physiques."
        if correction
        else "Le filtre de sécurité a **validé** la décision : les limites physiques sont respectées."
    )
    if len(r) > 5:
        r.pop(1)  # la comparaison des SOC est la raison la moins informative
    return r[:5]


st.subheader("Pourquoi cette décision ?")
st.markdown("\n".join(f"{i}. {r}" for i, r in enumerate(_raisons(), start=1)))
st.caption(f"Explication {'exacte' if exacte else 'approchée (après coup)'} : {nature_txt}")


onglet_dec, onglet_rai, onglet_sym, onglet_res, onglet_etsi, onglet_cycle = st.tabs(
    ["Décision", "Raisons", "Symbolique", "Réseau neuronal", "Et si… ?", "Sur tout le cycle"]
)


# Onglet « Décision » : la décision décomposée, et les autres stratégies au même instant

with onglet_dec:
    if demande_nulle:
        st.info("Demande quasi nulle : pas de répartition à décomposer.")
    else:
        if strategie in ("EMS_fuzzy_logic", "EMS_MLP_neurosymbolic"):
            etapes = (
                [(_lib_regle(FUZZY_RULE_NAMES[i]), float(contrib[i])) for i in np.argsort(contrib)[::-1] if contrib[i] > 5e-4]
                if contrib is not None else [("Répartition par défaut", alpha_flou)]
            )
            if strategie == "EMS_MLP_neurosymbolic":
                etapes += [("Correction du réseau", alpha_reseau - alpha_flou)]
                if garde_actif:
                    etapes += [("Garde-fou symbolique (réserve PB)", alpha_req - alpha_reseau)]
            elif abs(alpha_req - alpha_flou) > 1e-4:
                etapes += [("Ajustement du moteur", alpha_req - alpha_flou)]
        elif strategie == "EMS_power_limitation":
            etapes = [("Règle physique", alpha_req)]
        else:
            reste = sum(v for _, v in principales[6:])
            etapes = (
                [("Référence (situation moyenne)", ref_shap)]
                + [(xai.LIBELLES_ENTREES.get(c, c), v) for c, v in principales[:6] if abs(v) >= 5e-4]
                + ([("Autres entrées", reste)] if abs(reste) >= 5e-4 else [])
                + ([("Écart de reconstitution", alpha_req - alpha_explique)] if abs(alpha_req - alpha_explique) > 1e-3 else [])
            )
        etapes += [("Filtre de sécurité", alpha_final - alpha_req)]
        st.plotly_chart(_cascade(etapes, alpha_final), width="stretch")
        st.caption(
            "Part confiée à la PB, étape par étape : orange = pousse vers la batterie Puissance, "
            "vert = vers la batterie Énergie, violet = décision appliquée."
        )
        if correction:
            st.warning(
                f"Le filtre de sécurité a corrigé la répartition proposée ({nombre(alpha_req * 100, 1)} %) "
                "pour respecter les limites physiques des batteries et du convertisseur."
            )
        elif abs(alpha_final - alpha_req) > 1e-9:
            _pas = st.session_state.get("pas_alpha") or ALPHA_GRID_STEP
            st.caption(
                f"Proposé {nombre(alpha_req * 100, 1)} %, appliqué {nombre(alpha_final * 100, 1)} % : le filtre choisit "
                f"alpha sur une grille de pas {_pas:g} ; un écart inférieur à {nombre(_pas * 110, 2)} points est "
                "un arrondi à cette grille, pas une correction."
            )

        st.markdown("**Les autres stratégies au même instant**")
        alphas = {nm: float(resultats[nm]["alpha_final"][instant]) * 100 for nm in noms}
        ordre_s = sorted(noms, key=lambda nm: alphas[nm], reverse=True)
        fig_a = go.Figure(
            go.Bar(
                y=[nom_affichage(nm) for nm in ordre_s][::-1], x=[alphas[nm] for nm in ordre_s][::-1],
                orientation="h",
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
                annotation_text=f"règle {onto['regle']['id']} de l'ontologie", annotation_position="top",
            )
        fig_a.update_layout(
            separators=SEPARATEURS_PLOTLY, height=40 * len(noms) + 90, margin=dict(t=30, b=40, l=10, r=50),
            xaxis=dict(title="Part confiée à la PB (%)", range=[0, 110]), showlegend=False,
        )
        st.plotly_chart(fig_a, width="stretch")
        st.caption("Chaque stratégie est évaluée sur ses propres états de charge à cet instant.")

    with st.expander("Contexte : puissances autour de cet instant"):
        demi = 150
        i0, i1 = max(0, instant - demi), min(n, instant + demi + 1)
        xx = df["time"].to_numpy()[i0:i1] if "time" in df.columns else np.arange(i0, i1)
        fig_ctx = go.Figure(
            [
                go.Scatter(x=xx, y=df["hasPower"].to_numpy()[i0:i1] / 1000.0, name="Demande", line=dict(color=COULEUR_DEMANDE)),
                go.Scatter(x=xx, y=np.asarray(traj["P_EB"], float)[i0:i1] / 1000.0, name="Batterie Énergie", line=dict(color=C_EB)),
                go.Scatter(x=xx, y=np.asarray(traj["P_PB"], float)[i0:i1] / 1000.0, name="Batterie Puissance", line=dict(color=C_PB)),
            ]
        )
        fig_ctx.add_vline(x=t_sel, line=dict(color=COULEUR_REFERENCE, dash="dash"))
        fig_ctx.update_layout(
            separators=SEPARATEURS_PLOTLY, xaxis_title="Temps (s)", yaxis_title="Puissance (kW)", height=320,
            margin=dict(t=20, b=40), hovermode="x unified", legend=dict(orientation="h", y=1.1),
        )
        st.plotly_chart(fig_ctx, width="stretch")


# Onglet « Raisons » : les éléments qui ont contribué à la décision

with onglet_rai:
    if demande_nulle:
        st.info("Demande quasi nulle : pas de décision à expliquer.")
    elif strategie == "EMS_power_limitation":
        if onto is not None:
            premisses = " et ".join(f"`{d['texte']}`" for d in onto["regle"]["details"])
            st.markdown(f"La décision découle d'une seule règle : **{onto['regle']['id']}** — si {premisses}, alors {onto['regle']['lecture']}.")
        st.caption("Le modèle physique ne dépend que de la puissance demandée et du SOC de la batterie Énergie.")
    elif strategie in ("EMS_fuzzy_logic", "EMS_MLP_neurosymbolic"):
        regles_f = {r["cle"]: r for r in ox.regles_floues()}
        actives = [i for i in np.argsort(forces)[::-1] if forces[i] > 5e-4]
        if actives:
            st.dataframe(
                [
                    {
                        "Règle": regles_f[FUZZY_RULE_NAMES[i]]["libelle"],
                        "Si": regles_f[FUZZY_RULE_NAMES[i]]["si"],
                        "Activation": f"{nombre(forces[i] * 100, 0)} %",
                        "Conclusion (part de la PB)": f"{nombre(regles_f[FUZZY_RULE_NAMES[i]]['alpha'] * 100, 0)} %",
                        "Contribution": f"{nombre(contrib[i] * 100, 1)} pts" if contrib is not None else "—",
                    }
                    for i in actives
                ],
                hide_index=True, width="stretch",
            )
        st.caption(
            "Le moteur flou calcule alpha comme la moyenne des conclusions des règles, pondérée "
            "par leur activation : chaque contribution est exacte."
        )
    else:
        valeurs = [(xai.LIBELLES_ENTREES.get(c, c), v * 100) for c, v in principales]
        st.plotly_chart(
            _barres_h(
                [lib for lib, _ in valeurs], [v for _, v in valeurs],
                [C_PB if v >= 0 else C_EB for _, v in valeurs],
                "Contribution à la part de la PB (points)", "%{y} : %{x:+.1f} points<extra></extra>",
            ),
            width="stretch",
        )
        st.caption(
            f"Par rapport à une situation moyenne du cycle, où le réseau confierait {nombre(ref_shap * 100, 1)} % "
            "à la PB : orange = pousse vers la batterie Puissance, vert = vers la batterie Énergie."
        )


# Onglet « Symbolique » : l'ontologie et la place du symbolique dans le modèle

def _coche(ok, texte):
    return f"{'✓' if ok else '✗'} {texte}"


with onglet_sym:
    if strategie == "EMS_MLP_neurosymbolic" and not demande_nulle:
        st.markdown("**NS-MLP : les règles décident, le réseau corrige, l'ontologie protège la réserve de la PB**")
        st.markdown(
            flux_html(
                [
                    ("Base floue (règles)", COULEUR_REFERENCE),
                    (f"alpha symbolique {nombre(alpha_flou * 100, 1)} %", COULEUR_DECISION),
                    (f"correction MLP {nombre((alpha_reseau - alpha_flou) * 100, 1, signe=True)} pts", COULEUR_SECONDAIRE),
                ]
                + (
                    [(f"garde-fou R14/R16 : {nombre(alpha_req * 100, 1)} %", COULEUR_REFERENCE)]
                    if garde_actif else []
                )
                + [
                    (f"alpha final {nombre(alpha_final * 100, 1)} %", COULEUR_DECISION),
                ]
            ),
            unsafe_allow_html=True,
        )
    elif strategie == "EMS_LSTM_neurosymbolic" and not demande_nulle:
        actifs = [ox.LIBELLES_SYMBOLIQUES[c] for c in etats_transmis if etat["symboliques"].get(c)]
        st.markdown("**NS-LSTM : le symbolique informe le réseau**")
        st.markdown(
            flux_html(
                [
                    (f"Historique ({LSTM_WINDOW} s)", COULEUR_REFERENCE),
                    (f"+ états symboliques : {', '.join(actifs) or 'aucun actif'}", COULEUR_REFERENCE),
                    ("LSTM", COULEUR_SECONDAIRE),
                    (f"alpha {nombre(alpha_final * 100, 1)} %", COULEUR_DECISION),
                ]
            ),
            unsafe_allow_html=True,
        )

    if etats_transmis:
        st.markdown(
            "États symboliques transmis au réseau : "
            + ", ".join(
                f"{ox.LIBELLES_SYMBOLIQUES[c]} **{'oui' if etat['symboliques'][c] else 'non'}**" for c in etats_transmis
            )
            + "."
        )
    elif strategie == "EMS_fuzzy_logic":
        concepts = ox.REGLES_FLOUES.get(dominante, ("", []))[1]
        if concepts and not regle_par_defaut:
            st.markdown(
                "Concepts de l'ontologie mobilisés par la règle dominante : "
                + ", ".join(f"{lib} (`{cl}`)" for lib, cl in concepts) + "."
            )

    st.markdown("**Ce qu'en dit l'ontologie OntoHESS**")
    st.markdown(f"État de fonctionnement inféré : **{etat['libelle']}** (`{etat['fonctionnement']}`).")
    activees, non_activees, _ = ox.evaluer_regles(p_dem, soc_eb, soc_pb)
    lues = [r for r in activees if r["type"] in ("mode", "repartition")]
    for r in sorted(lues, key=lambda r: r["type"]):
        premisses = " et ".join(f"`{d['texte']}`" for d in r["details"])
        st.markdown(f"- **{r['id']}** — si {premisses}, alors {r['lecture']}.")
    if onto is not None and not demande_nulle:
        ecart = (alpha_final - onto["alpha"]) * 100
        verdict = (
            "identique à" if abs(ecart) < 0.5
            else f"{nombre(ecart, 1)} points au-dessus de" if ecart > 0
            else f"{nombre(-ecart, 1)} points en dessous de"
        )
        st.markdown(
            f"La règle de répartition {onto['regle']['id']} prescrit **{nombre(onto['alpha'] * 100, 1)} %** pour la "
            f"PB ; la décision de la stratégie est {verdict} cette référence."
        )
        st.caption(
            "Ces règles décrivent la conduite de référence (priorité à l'EB, dans ses limites) ; "
            "s'en écarter n'est pas une erreur. Toutes les règles sont dans « Base de connaissances »."
        )

    paire = [c for c in ("EMS_MLP_neurosymbolic", "EMS_LSTM_neurosymbolic") if c in resultats]
    if len(paire) == 2 and not demande_nulle:
        with st.expander("Comparer NS-MLP et NS-LSTM au même instant"):
            colonnes_ns = st.columns(2)
            decisions = {}
            for col, cle in zip(colonnes_ns, paire):
                tr = resultats[cle]
                se, sp = float(tr["SOC_EB"][instant]), float(tr["SOC_PB"][instant])
                a_req_ns, a_fin_ns = float(tr["alpha_requested"][instant]), float(tr["alpha_final"][instant])
                pe, pp = float(tr["P_EB"][instant]), float(tr["P_PB"][instant])
                corr_ns = bool(tr["correction_applied"][instant])
                decisions[cle] = a_fin_ns
                with col:
                    with st.container(border=True):
                        st.markdown(f"#### {nom_affichage(cle)}")
                        if cle == "EMS_MLP_neurosymbolic":
                            rf = alpha_fuzzy_calc(np.array([se]), np.array([sp]), np.array([p_dem]), np.array([accel]))
                            dom = str(rf["dominant_rule"][0])
                            a_f, a_r, garde_ns = decomposition_ns(cle, instant)
                            st.markdown(
                                f"Règles floues (menées par {_lib_regle(dom)}) : **{nombre(a_f * 100, 1)} %**  \n"
                                f"Correction du réseau : **{nombre((a_r - a_f) * 100, 1, signe=True)} points**"
                                + (f"  \nGarde-fou symbolique : **{nombre((a_req_ns - a_r) * 100, 1, signe=True)} points**" if garde_ns else "")
                            )
                        else:
                            ref_ns, contribs_ns, _ = _shapley(cle, instant, _signature(cle))
                            total = sum(abs(v) for _, v in contribs_ns) or 1.0
                            part = sum(abs(v) for c, v in contribs_ns if c in xai.ETATS_SYMBOLIQUES) / total * 100
                            top = max(
                                ((c, v) for c, v in contribs_ns if c not in xai.ETATS_SYMBOLIQUES),
                                key=lambda kv: abs(kv[1]), default=("—", 0.0),
                            )
                            st.markdown(
                                f"États symboliques : **{nombre(part, 0)} %** de la décision  \n"
                                f"Entrée la plus influente : {xai.LIBELLES_ENTREES.get(top[0], top[0])} "
                                f"({nombre(top[1] * 100, 1, signe=True)} points)"
                            )
                        st.markdown(f"Décision : **{nombre(a_fin_ns * 100, 1)} %** pour la PB · EB {kw(pe)} · PB {kw(pp)}")
                        st.markdown(
                            "  \n".join(
                                [
                                    _coche(not corr_ns, "acceptée par le filtre" if not corr_ns else "corrigée par le filtre"),
                                    _coche(se >= SOC_EB_MIN and sp >= SOC_PB_MIN, "SOC dans leurs limites"),
                                    _coche(P_EB_MIN_W - 1 <= pe <= P_EB_MAX_W + 1, "EB dans ses limites de puissance"),
                                    _coche(float(tr["P_unserved"][instant]) < 1.0, "demande entièrement fournie"),
                                ]
                            )
                        )
            a1, a2 = decisions[paire[0]], decisions[paire[1]]
            st.caption(
                f"Écart de {nombre(abs(a1 - a2) * 100, 1)} points entre les deux décisions : NS-MLP suit ses "
                f"règles puis les corrige à l'instant présent ; NS-LSTM raisonne sur les {LSTM_WINDOW} "
                "dernières secondes, où le symbolique n'est qu'une entrée parmi d'autres."
            )


# Onglet « Réseau neuronal » : la part apprise de la décision

with onglet_res:
    if strategie in ("EMS_power_limitation", "EMS_fuzzy_logic"):
        st.info("Ce modèle n'a pas de réseau neuronal : sa décision est entièrement lisible (onglets Raisons et Symbolique).")
    elif demande_nulle:
        st.info("Demande quasi nulle : pas de décision à expliquer.")
    elif strategie == "EMS_MLP_neurosymbolic":
        delta = alpha_reseau - alpha_flou
        r1, r2, r3 = st.columns(3)
        r1.metric("Correction du réseau", f"{nombre(delta * 100, 1, signe=True)} pts")
        r2.metric("Part de la marge utilisée", f"{nombre(abs(delta) / MLP_NS_MAX_DELTA * 100, 0)} %")
        r3.metric("Garde-fou symbolique", "actif" if garde_actif else "inactif")
        st.markdown(
            f"La partie neuronale ne décide pas seule : elle ajoute à la base floue une correction "
            f"bornée à ±{nombre(MLP_NS_MAX_DELTA * 100, 0)} points. Cette correction reste opaque, mais son "
            "poids dans la décision est connu exactement. Quand le SOC de la batterie Puissance passe "
            f"sous {nombre(MLP_NS_RESERVE_PB_SOC * 100, 0)} %, un garde-fou symbolique (règles R14 et R16 "
            "d'OntoHESS) limite en traction sa part à ce que l'EB ne peut pas fournir."
        )
    else:
        st.markdown(
            f"Valeurs de Shapley exactes : le réseau est évalué sur toutes les combinaisons de ses "
            f"{len(contribs)} entrées, une entrée absente prenant sa valeur moyenne sur le cycle (état "
            "inactif pour un état symbolique). La référence plus les contributions redonnent exactement "
            f"la décision du réseau ({nombre(ref_shap * 100, 1)} % + contributions = {nombre(alpha_explique * 100, 1)} %)."
        )
        if strategie in ("EMS_LSTM", "EMS_LSTM_neurosymbolic"):
            st.caption(f"Chaque entrée compte pour tout son historique sur les {LSTM_WINDOW} dernières secondes.")
        if strategie == "EMS_LSTM_neurosymbolic":
            total = sum(abs(v) for _, v in contribs) or 1.0
            st.metric("Poids des états symboliques", f"{nombre(sum(abs(v) for c, v in contribs if c in xai.ETATS_SYMBOLIQUES) / total * 100, 0)} %")
        if strategie == "EMS_GNN":
            st.markdown("**Lecture par composant du graphe** (gradient × entrée, estimation locale)")
            pct_g, edge = _attribution_gnn(p_dem, soc_eb, soc_pb, accel)
            st.plotly_chart(_graphe_gnn(pct_g, edge), width="stretch")


# Onglet « Et si… ? » : que déciderait le modèle si la situation changeait un peu ?

@st.cache_data(show_spinner="Calcul des scénarios « Et si… ? »…")
def _et_si(strategie, instant, signature):
    s = xai.situation(df, resultats[strategie], instant)
    base = xai.alpha_decision(strategie, df, resultats[strategie], instant)
    scenarios = [
        ("SOC de la PB + 5 points", "SOC_PB", min(s["SOC_PB"] + 0.05, 1.0), +1),
        ("SOC de la PB − 5 points", "SOC_PB", max(s["SOC_PB"] - 0.05, 0.0), -1),
        ("SOC de l'EB + 5 points", "SOC_EB", min(s["SOC_EB"] + 0.05, 1.0), -1),
        ("SOC de l'EB − 5 points", "SOC_EB", max(s["SOC_EB"] - 0.05, 0.0), +1),
        ("Demande + 1 kW", "hasPower", s["hasPower"] + 1000.0, +1),
        ("Demande − 1 kW", "hasPower", s["hasPower"] - 1000.0, -1),
    ]
    sortie = []
    for libelle, grandeur, valeur, sens in scenarios:
        a = xai.alpha_decision(strategie, df, resultats[strategie], instant, {grandeur: valeur})
        p = valeur if grandeur == "hasPower" else s["hasPower"]
        sortie.append((libelle, a, a * p, sens))
    return base, base * s["hasPower"], sortie


with onglet_etsi:
    if demande_nulle:
        st.info("Demande quasi nulle : choisissez un instant de traction ou de freinage.")
    else:
        base_a, base_p, scenarios = _et_si(strategie, instant, _signature(strategie))
        st.markdown(
            f"Que déciderait **{nom_affichage(strategie)}** si la situation changeait un peu ? "
            f"Aujourd'hui : **{nombre(base_a * 100, 1)} %** pour la PB ({kw(base_p)})."
        )
        st.dataframe(
            [
                {
                    "Scénario": libelle,
                    "Part de la PB": f"{nombre(a * 100, 1)} %",
                    "Puissance PB": kw(p),
                    "Variation de la puissance PB": f"{nombre((p - base_p) / 1000, 2, signe=True)} kW",
                    "Sens physiquement attendu": "✓" if sens * (p - base_p) >= -0.01 * abs(p_dem) else "✗",
                }
                for libelle, a, p, sens in scenarios
            ],
            hide_index=True, width="stretch",
        )
        st.caption(
            "Sens attendu : plus de charge dans la PB, ou plus de demande, ne doit pas réduire la "
            "puissance de la PB ; plus de charge dans l'EB ne doit pas l'augmenter (à 1 % de la "
            "demande près). C'est le test E3 de cohérence physique, appliqué à cet instant."
        )
        st.markdown("**D'après les seuils de l'ontologie**")
        for phrase in ox.contrefactuels(p_dem, soc_eb, soc_pb):
            st.markdown(f"- {phrase}")
        echecs = [
            f"**{r['id']}** ({r['lecture']}) : "
            + " ; ".join(f"`{d['texte']}` non vérifié" for d in r["details"] if d["ok"] is False)
            for r in non_activees if r["type"] in ("mode", "repartition")
        ]
        if echecs:
            with st.expander("Règles de l'ontologie non activées, et pourquoi"):
                st.markdown("\n".join(f"- {e}" for e in echecs))


# Onglet « Sur tout le cycle » : indicateurs d'explicabilité

@st.cache_data(show_spinner="Analyse du cycle…")
def _analyse_cycle(p, accel_c, soc_eb_c, soc_pb_c, a_req, a_fin, corr):
    actif = np.abs(p) > EPS_POWER_W
    flou = alpha_fuzzy_calc(soc_eb_c, soc_pb_c, p, accel_c)
    return {
        "actif": actif,
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
    k1.metric("Explication", "exacte" if exacte else "approchée")
    k2.metric("Cohérence physique (E3)", f"{nombre(e3 * 100, 0)} %")
    k3.metric("Accord avec la règle de l'ontologie", f"{nombre(np.mean(ecart_onto < 0.05) * 100, 0)} %")
    k4.metric("Décisions corrigées par le filtre", f"{nombre(np.mean(cy['corr'][cy['actif']]) * 100, 1)} %")
    st.caption(
        "Cohérence physique : part de 120 instants de traction où la décision varie dans le sens "
        "attendu (" + " ; ".join(f"{k} : {nombre(v * 100, 0)} %" for k, v in e3_detail.items()) + "). "
        "Accord avec la règle de l'ontologie : part des instants où la décision reste à moins de "
        f"5 points de la répartition prescrite par OntoHESS (écart moyen : {nombre(np.mean(ecart_onto) * 100, 1)} points)."
    )

    utilise_flou = strategie in ("EMS_fuzzy_logic", "EMS_MLP_neurosymbolic")
    if utilise_flou:
        a = cy["actif"]
        j1, j2, j3 = st.columns(3)
        j1.metric("Règles floues actives par décision", f"{nombre(np.mean(np.sum(cy['forces'][a] > 0.05, axis=1)), 1)}")
        if strategie == "EMS_MLP_neurosymbolic":
            a_reseau = np.asarray(traj.get("alpha_reseau", cy["a_req"]), dtype=float)[:n]
            delta = np.abs(a_reseau[a] - cy["alpha_flou"][a])
            j2.metric("Marge de correction utilisée", f"{nombre(np.nanmean(delta) / MLP_NS_MAX_DELTA * 100, 0)} %")
            if "garde_fou" in traj:
                j3.metric(
                    "Décisions reprises par le garde-fou",
                    f"{nombre(np.mean(np.asarray(traj['garde_fou'], dtype=bool)[:n][a]) * 100, 1)} %",
                    help=f"Instants de traction où la PB était sous {nombre(MLP_NS_RESERVE_PB_SOC * 100, 0)} % de SOC.",
                )
            else:
                j3.metric("Décisions laissées aux règles seules", f"{nombre(np.mean(delta < 0.02) * 100, 1)} %")

    temps = df["time"].to_numpy(dtype=float)[:n] if "time" in df.columns else np.arange(n, dtype=float)
    xm = _par_minute(temps) / 60.0
    traces = [
        go.Scatter(x=xm, y=_par_minute(np.where(cy["actif"], cy["a_fin"], np.nan)) * 100,
                   name=f"Décision de {nom_affichage(strategie)}", line=dict(color=COULEUR_DECISION, width=2.2)),
        go.Scatter(x=xm, y=_par_minute(cy["alpha_onto"]) * 100, name="Règle de l'ontologie",
                   line=dict(color=COULEUR_REFERENCE, width=1.6, dash="dash")),
    ]
    if strategie == "EMS_MLP_neurosymbolic":
        traces.append(
            go.Scatter(x=xm, y=_par_minute(np.where(cy["actif"], cy["alpha_flou"], np.nan)) * 100,
                       name="Base floue", line=dict(color=C_GRIS, width=1.6, dash="dot"))
        )
    fig_c = go.Figure(traces)
    fig_c.update_layout(
        separators=SEPARATEURS_PLOTLY, height=340, margin=dict(t=20, b=40, l=10, r=10), hovermode="x unified",
        xaxis_title="Temps (min)", yaxis_title="Part confiée à la PB (%, moyenne par minute)",
        legend=dict(orientation="h", y=1.1),
    )
    st.markdown("**Décision et règle de référence au fil du cycle**")
    st.plotly_chart(fig_c, width="stretch")

    if utilise_flou:
        st.markdown("**Règles floues dominantes sur le cycle**")
        dom = cy["dominante"][cy["actif"]]
        regles, comptes = np.unique(dom, return_counts=True)
        ordre = list(np.argsort(comptes)[::-1])
        st.plotly_chart(
            _barres_h(
                [_lib_regle(regles[i]) for i in ordre], [comptes[i] / comptes.sum() * 100 for i in ordre],
                [C_GRIS] * len(ordre), "Part des instants où la règle domine (%)", "%{y} : %{x:.0f} %<extra></extra>",
            ),
            width="stretch",
        )


pied_navigation("vues/7_Explicabilite.py")
