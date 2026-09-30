import sys
import warnings
from pathlib import Path

DOSSIER_PROJET = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DOSSIER_PROJET))

import numpy as np
import torch
import plotly.graph_objects as go
import streamlit as st

from ems_core import (
    alpha_fuzzy_calc,
    construire_graphe_instant,
    load_gnn_simple,
    charger_scaler,
    FUZZY_RULE_NAMES,
    RULE_LABELS_FR,
    MLP_NS_INPUT_COLS,
    MLP_NS_MAX_DELTA,
    LSTM_NS_FEATURE_COLS,
    LSTM_WINDOW,
    GNN_SCALER_FILE,
    GNN_NODE_NAMES,
    DEVICE,
    ALPHA_GRID_STEP,
    EPS_POWER_W,
)
from core.resultats import assurer_donnees_session, nom_affichage
from core.navigation import pied_navigation
from core.instant import choisir_instant
from core.style import couleur
from core import ontology_explainer as ox
from core import xai


# Batterie Énergie = bleu, batterie Puissance = vert, comme sur les autres pages.
C_EB = "#5B8DEF"
C_PB = "#30A46C"
C_GRIS = "#8B93A7"
C_REPERE = "#E0A030"

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
def _coherence(strategie, signature):
    return xai.coherence_physique(strategie, df, resultats[strategie])


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
        height=36 * len(etiquettes) + 70, margin=dict(t=10, b=40, l=10, r=20),
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
    textes = [f"{y[0]:.1f} %"] + [f"{v:+.1f} pts" for v in y[1:-1]] + [f"{alpha_final * 100:.1f} %"]
    fig = go.Figure(
        go.Waterfall(
            x=x, y=y, measure=mesures, text=textes, textposition="outside",
            connector=dict(line=dict(color=C_GRIS, width=1)),
            increasing=dict(marker=dict(color=C_PB)),
            decreasing=dict(marker=dict(color=C_EB)),
            totals=dict(marker=dict(color=C_REPERE)),
            hoverinfo="skip",
        )
    )
    fig.update_layout(
        height=340, margin=dict(t=20, b=40, l=10, r=10), showlegend=False,
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
                text=[f"{l}<br>{p:.0f} %" for l, p in zip(labels, pct_g)],
                textposition="bottom center", hoverinfo="text",
            ),
        ]
    )
    fig.update_layout(
        height=360, showlegend=False, margin=dict(t=20, b=20, l=10, r=10),
        xaxis=dict(visible=False), yaxis=dict(visible=False, range=[-1.8, 1.6]),
    )
    return fig


def kw(x):
    return f"{x / 1000.0:.1f} kW"


# Interface

st.title("💡 Pourquoi cette décision ?")
st.caption(
    "Comment une stratégie a-t-elle réparti la puissance entre les deux batteries, et "
    "que dit l'ontologie OntoHESS de cette décision ? L'explication s'adapte au modèle : "
    "exacte quand on connaît son calcul, approchée quand le modèle est opaque."
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
    strategie = st.selectbox("Stratégie", noms, format_func=nom_affichage)

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

tab_instant, tab_cycle = st.tabs(["À cet instant", "Sur tout le cycle"])


with tab_instant:

    # 1 — La situation

    st.subheader("1. La situation")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Temps", f"{t_sel:.0f} s")
    c2.metric("Vitesse", f"{speed * 3.6:.0f} km/h")
    c3.metric("Puissance demandée", kw(p_dem))
    c4.metric("SOC Énergie", f"{soc_eb * 100:.1f} %")
    c5.metric("SOC Puissance", f"{soc_pb * 100:.1f} %")
    st.markdown(
        f"État de fonctionnement inféré par l'ontologie : **{etat['libelle']}** "
        f"(`{etat['fonctionnement']}`)."
    )

    # 2 — La décision

    st.subheader("2. La décision")
    if demande_nulle:
        st.info("Demande quasi nulle : aucune batterie n'est sollicitée, il n'y a pas de répartition à expliquer.")
    else:
        d1, d2, d3 = st.columns(3)
        d1.metric("Part confiée à la PB (alpha)", f"{alpha_final * 100:.1f} %")
        d2.metric("Batterie Énergie", kw(p_eb))
        d3.metric("Batterie Puissance", kw(p_pb))
        if correction:
            st.warning(
                f"Le filtre de sécurité a corrigé la répartition proposée ({alpha_req * 100:.1f} %) "
                "pour respecter les limites physiques des batteries et du convertisseur."
            )
        elif abs(alpha_final - alpha_req) > 1e-9:
            _pas = st.session_state.get("pas_alpha") or ALPHA_GRID_STEP
            st.caption(
                f"Proposé {alpha_req * 100:.1f} %, appliqué {alpha_final * 100:.1f} % : le filtre "
                f"choisit alpha sur une grille de pas {_pas:g} ; un écart inférieur à "
                f"{_pas * 110:.2f} points est un arrondi à cette grille, pas une correction."
            )
        else:
            st.caption("Décision acceptée telle quelle par le filtre de sécurité.")

    # 3 — Comment la décision a été construite

    st.subheader("3. Comment la décision a été construite")
    with st.container(border=True):
        st.markdown(
            f"**Explication {'exacte' if exacte else 'approchée (après coup)'}** — {nature_txt}"
        )

    if not demande_nulle:
        res_flou = alpha_fuzzy_calc(np.array([soc_eb]), np.array([soc_pb]), np.array([p_dem]), np.array([accel]))
        alpha_flou = float(res_flou["alpha"][0])
        forces = np.asarray(res_flou["strengths"][0], dtype=float)
        contrib = ox.contributions_floues(forces)
        # Une barre par règle floue : alpha flou = somme exacte des contributions.
        etapes_floues = (
            [
                (ox.REGLES_FLOUES.get(FUZZY_RULE_NAMES[i], (FUZZY_RULE_NAMES[i],))[0], float(contrib[i]))
                for i in np.argsort(contrib)[::-1]
                if contrib[i] > 5e-4
            ]
            if contrib is not None
            else [("Répartition par défaut", alpha_flou)]
        )

        if strategie == "EMS_MLP_neurosymbolic":
            etapes = etapes_floues + [
                ("Correction du réseau", alpha_req - alpha_flou),
                ("Filtre de sécurité", alpha_final - alpha_req),
            ]
        elif strategie == "EMS_fuzzy_logic":
            etapes = etapes_floues
            if abs(alpha_req - alpha_flou) > 1e-4:
                etapes = etapes + [("Ajustement du moteur", alpha_req - alpha_flou)]
            etapes = etapes + [("Filtre de sécurité", alpha_final - alpha_req)]
        elif strategie == "EMS_power_limitation":
            etapes = [("Règle physique", alpha_req), ("Filtre de sécurité", alpha_final - alpha_req)]
        else:
            # Réseau opaque : valeurs de Shapley exactes, par rapport à une situation
            # moyenne du cycle. Référence + contributions = décision du réseau.
            ref_shap, contribs, alpha_explique = _shapley(strategie, instant, _signature(strategie))
            principales = sorted(contribs, key=lambda kv: -abs(kv[1]))
            reste = sum(v for _, v in principales[6:])
            etapes = (
                [("Référence (situation moyenne)", ref_shap)]
                + [(xai.LIBELLES_ENTREES.get(c, c), v) for c, v in principales[:6] if abs(v) >= 5e-4]
                + ([("Autres entrées", reste)] if abs(reste) >= 5e-4 else [])
                + ([("Écart de reconstitution", alpha_req - alpha_explique)] if abs(alpha_req - alpha_explique) > 1e-3 else [])
                + [("Filtre de sécurité", alpha_final - alpha_req)]
            )
        st.plotly_chart(_cascade(etapes, alpha_final), width="stretch")
        st.caption("Vert : pousse vers la batterie Puissance ; bleu : vers la batterie Énergie.")

        # Détail propre à chaque famille de modèle
        if strategie == "EMS_power_limitation":
            if onto is not None:
                st.markdown(
                    f"Branche appliquée : **{onto['regle']['lecture']}**. C'est la règle "
                    f"**{onto['regle']['id']}** d'OntoHESS (section 4) : sur tout le cycle, "
                    "cette stratégie prend exactement la décision que prescrivent les règles "
                    "de répartition de l'ontologie."
                )

        elif strategie in ("EMS_fuzzy_logic", "EMS_MLP_neurosymbolic"):
            dominante = str(res_flou["dominant_rule"][0])
            regles_f = {r["cle"]: r for r in ox.regles_floues()}
            actives = [i for i in np.argsort(forces)[::-1] if forces[i] > 5e-4]
            if actives:
                st.dataframe(
                    [
                        {
                            "Règle": regles_f[FUZZY_RULE_NAMES[i]]["libelle"],
                            "Si": regles_f[FUZZY_RULE_NAMES[i]]["si"],
                            "Activation": f"{forces[i] * 100:.0f} %",
                            "Conclusion (part de la PB)": f"{regles_f[FUZZY_RULE_NAMES[i]]['alpha'] * 100:.0f} %",
                            "Contribution": f"{contrib[i] * 100:.1f} pts" if contrib is not None else "—",
                        }
                        for i in actives
                    ],
                    hide_index=True,
                    width="stretch",
                )
            if dominante in RULE_LABELS_FR:
                concepts = ox.REGLES_FLOUES.get(dominante, ("", []))[1]
                st.markdown(
                    f"Règle dominante : **{ox.REGLES_FLOUES.get(dominante, (dominante,))[0]}** — "
                    f"{RULE_LABELS_FR[dominante]}."
                    + (
                        " Concepts de l'ontologie mobilisés : "
                        + ", ".join(f"{lib} (`{cl}`)" for lib, cl in concepts) + "."
                        if concepts else ""
                    )
                )
            st.caption(
                "Le moteur flou calcule alpha comme la moyenne des conclusions des règles, "
                "pondérée par leur activation : chaque contribution est exacte, et leur somme "
                "donne la sortie floue (barres de la cascade)."
            )
            if strategie == "EMS_MLP_neurosymbolic":
                delta = alpha_req - alpha_flou
                st.markdown(
                    f"Le réseau a corrigé la base floue de **{delta * 100:+.1f} points**, soit "
                    f"{abs(delta) / MLP_NS_MAX_DELTA * 100:.0f} % de sa marge maximale "
                    f"(±{MLP_NS_MAX_DELTA * 100:.0f} points)."
                )

        elif strategie in xai.SHAPLEY_DISPONIBLE:
            fortes = [(xai.LIBELLES_ENTREES.get(c, c), v) for c, v in principales[:2] if abs(v) >= 5e-4]
            if fortes:
                st.markdown(
                    f"Dans une situation moyenne du cycle, le réseau confierait "
                    f"**{ref_shap * 100:.1f} %** à la PB. À cet instant, "
                    + " ; ".join(
                        f"**{lib}** {'augmente' if v > 0 else 'réduit'} cette part de {abs(v) * 100:.1f} points"
                        for lib, v in fortes
                    )
                    + "."
                )
            if strategie == "EMS_LSTM_neurosymbolic":
                total = sum(abs(v) for _, v in contribs)
                symb = sum(abs(v) for c, v in contribs if c in xai.ETATS_SYMBOLIQUES)
                st.markdown(
                    f"Les quatre états symboliques de l'ontologie pèsent **{symb / total * 100:.0f} %** "
                    "de l'explication à cet instant." if total > 0 else ""
                )
            if strategie in ("EMS_LSTM", "EMS_LSTM_neurosymbolic"):
                st.caption(
                    f"Chaque entrée compte pour tout son historique sur les {LSTM_WINDOW} dernières "
                    "secondes vues par le réseau."
                )
            st.caption(
                "Valeurs de Shapley exactes : le réseau est évalué sur toutes les combinaisons "
                "d'entrées, une entrée absente prenant sa valeur moyenne sur le cycle (état "
                "inactif pour un état symbolique). La référence plus les contributions redonnent "
                "exactement la décision du réseau."
            )
            if strategie == "EMS_GNN":
                with st.expander("Lecture par composant du graphe (gradient × entrée)"):
                    pct_g, edge = _attribution_gnn(p_dem, soc_eb, soc_pb, accel)
                    st.plotly_chart(_graphe_gnn(pct_g, edge), width="stretch")
                    st.caption("Estimation locale approchée : importance de chaque nœud du graphe du HESS.")

    # 4 — Ce qu'en dit l'ontologie

    st.subheader("4. Ce qu'en dit l'ontologie")
    activees, non_activees, _ = ox.evaluer_regles(p_dem, soc_eb, soc_pb)
    lues = [r for r in activees if r["type"] in ("mode", "repartition")]
    if lues:
        st.markdown("Règles SWRL d'OntoHESS vérifiées à cet instant :")
        for r in sorted(lues, key=lambda r: r["type"]):
            premisses = " et ".join(f"`{d['texte']}`" for d in r["details"])
            st.markdown(f"- **{r['id']}** — si {premisses}, alors {r['lecture']}.")
    else:
        st.markdown("Aucune règle de mode ou de répartition ne s'applique (demande quasi nulle).")

    if onto is not None and not demande_nulle:
        ecart = (alpha_final - onto["alpha"]) * 100
        if abs(ecart) < 0.5:
            verdict = "la décision de la stratégie est **identique** à cette règle de référence"
        elif ecart > 0:
            verdict = f"la stratégie confie **{ecart:.1f} points de plus** à la PB que cette règle"
        else:
            verdict = f"la stratégie confie **{-ecart:.1f} points de moins** à la PB que cette règle"
        st.markdown(
            f"La règle de répartition {onto['regle']['id']} prescrit **{onto['alpha'] * 100:.1f} %** "
            f"pour la PB ; {verdict}."
        )
        st.caption(
            "Ces règles décrivent la conduite de référence (priorité à l'EB, dans ses limites). "
            "S'en écarter n'est pas une erreur : c'est ce que font les stratégies qui cherchent mieux."
        )

    entrees = {
        "EMS_MLP_neurosymbolic": MLP_NS_INPUT_COLS,
        "EMS_LSTM_neurosymbolic": LSTM_NS_FEATURE_COLS,
    }.get(strategie)
    if entrees:
        etats_modele = [c for c in ox.LIBELLES_SYMBOLIQUES if c in entrees]
        st.markdown(
            "États symboliques transmis au réseau : "
            + ", ".join(
                f"{ox.LIBELLES_SYMBOLIQUES[c]} **{'oui' if etat['symboliques'][c] else 'non'}**"
                for c in etats_modele
            )
            + "."
        )

    with st.expander("Et si… ? (d'après les seuils de l'ontologie)"):
        for phrase in ox.contrefactuels(p_dem, soc_eb, soc_pb):
            st.markdown(f"- {phrase}")
        echecs = [
            f"**{r['id']}** ({r['lecture']}) : " + " ; ".join(f"`{d['texte']}` non vérifié" for d in r["details"] if d["ok"] is False)
            for r in non_activees if r["type"] in ("mode", "repartition")
        ]
        if echecs:
            st.markdown("Règles non activées, et pourquoi :")
            for e in echecs:
                st.markdown(f"- {e}")
    st.caption("La structure de l'ontologie et toutes ses règles sont présentées dans « Base de connaissances ».")

    # 5 — Les autres stratégies au même instant

    if not demande_nulle:
        st.subheader("5. Les autres stratégies au même instant")
        alphas = {nm: float(resultats[nm]["alpha_final"][instant]) * 100 for nm in noms}
        ordre_s = sorted(noms, key=lambda nm: alphas[nm], reverse=True)
        fig_a = go.Figure(
            go.Bar(
                y=[nom_affichage(nm) for nm in ordre_s][::-1],
                x=[alphas[nm] for nm in ordre_s][::-1],
                orientation="h",
                marker=dict(
                    color=[couleur(nm) for nm in ordre_s][::-1],
                    opacity=[1.0 if nm == strategie else 0.4 for nm in ordre_s][::-1],
                ),
                text=[f"{alphas[nm]:.1f} %" for nm in ordre_s][::-1],
                textposition="outside",
                hoverinfo="skip",
            )
        )
        if onto is not None:
            fig_a.add_vline(
                x=onto["alpha"] * 100, line=dict(color=C_REPERE, dash="dash"),
                annotation_text=f"règle {onto['regle']['id']} de l'ontologie", annotation_position="top",
            )
        fig_a.update_layout(
            height=40 * len(noms) + 90, margin=dict(t=30, b=40, l=10, r=50),
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
                go.Scatter(x=xx, y=df["hasPower"].to_numpy()[i0:i1] / 1000.0, name="Demande", line=dict(color=C_GRIS)),
                go.Scatter(x=xx, y=np.asarray(traj["P_EB"], float)[i0:i1] / 1000.0, name="Batterie Énergie", line=dict(color=C_EB)),
                go.Scatter(x=xx, y=np.asarray(traj["P_PB"], float)[i0:i1] / 1000.0, name="Batterie Puissance", line=dict(color=C_PB)),
            ]
        )
        fig_ctx.add_vline(x=t_sel, line=dict(color=C_REPERE, dash="dash"))
        fig_ctx.update_layout(
            xaxis_title="Temps (s)", yaxis_title="Puissance (kW)", height=320,
            margin=dict(t=20, b=40), hovermode="x unified", legend=dict(orientation="h", y=1.1),
        )
        st.plotly_chart(fig_ctx, width="stretch")


# Sur tout le cycle : indicateurs d'explicabilité

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


with tab_cycle:
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

    st.subheader("Indicateurs d'explicabilité")
    e3, e3_detail = _coherence(strategie, _signature(strategie))
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Explication", "exacte" if exacte else "approchée")
    k2.metric("Cohérence physique (E3)", f"{e3 * 100:.0f} %")
    k3.metric("Accord avec la règle de l'ontologie", f"{np.mean(ecart_onto < 0.05) * 100:.0f} %")
    k4.metric("Décisions corrigées par le filtre", f"{np.mean(cy['corr'][cy['actif']]) * 100:.1f} %")
    st.caption(
        f"Cohérence physique : part de {120} instants de traction où la décision varie dans le "
        "sens attendu quand on augmente le SOC d'une batterie ou la demande "
        + " (" + " ; ".join(f"{k} : {v * 100:.0f} %" for k, v in e3_detail.items()) + "). "
        "Accord avec la règle de l'ontologie : part des instants où la décision reste à moins "
        f"de 5 points de la répartition prescrite par OntoHESS (écart moyen : "
        f"{np.mean(ecart_onto) * 100:.1f} points)."
    )

    utilise_flou = strategie in ("EMS_fuzzy_logic", "EMS_MLP_neurosymbolic")
    if utilise_flou:
        a = cy["actif"]
        nb_regles = np.mean(np.sum(cy["forces"][a] > 0.05, axis=1))
        j1, j2, j3 = st.columns(3)
        j1.metric("Règles floues actives par décision", f"{nb_regles:.1f}")
        if strategie == "EMS_MLP_neurosymbolic":
            delta = np.abs(cy["a_req"][a] - cy["alpha_flou"][a])
            j2.metric("Marge de correction utilisée", f"{np.mean(delta) / MLP_NS_MAX_DELTA * 100:.0f} %")
            j3.metric("Décisions laissées aux règles seules", f"{np.mean(delta < 0.02) * 100:.1f} %")
            st.caption(
                "Marge utilisée : correction moyenne du réseau rapportée à sa borne "
                f"(±{MLP_NS_MAX_DELTA * 100:.0f} points). Décisions laissées aux règles : part des "
                "instants où le réseau corrige de moins de 2 points. Une marge très utilisée "
                "signifie que la cible apprise s'éloigne nettement de la base floue."
            )
        else:
            st.caption("Nombre moyen de règles floues activées à plus de 5 % : plus il est faible, plus l'explication est concise.")

    # Évolution sur le cycle, en moyennes par minute
    temps = df["time"].to_numpy(dtype=float)[:n] if "time" in df.columns else np.arange(n, dtype=float)
    xm = _par_minute(temps) / 60.0
    traces = [
        go.Scatter(x=xm, y=_par_minute(np.where(cy["actif"], cy["a_fin"], np.nan)) * 100,
                   name=f"Décision de {nom_affichage(strategie)}", line=dict(color=couleur(strategie), width=2.2)),
        go.Scatter(x=xm, y=_par_minute(cy["alpha_onto"]) * 100, name="Règle de l'ontologie",
                   line=dict(color=C_REPERE, width=1.6, dash="dash")),
    ]
    if utilise_flou and strategie != "EMS_fuzzy_logic":
        traces.append(
            go.Scatter(x=xm, y=_par_minute(np.where(cy["actif"], cy["alpha_flou"], np.nan)) * 100,
                       name="Base floue", line=dict(color=C_GRIS, width=1.6, dash="dot"))
        )
    fig_c = go.Figure(traces)
    fig_c.update_layout(
        height=340, margin=dict(t=20, b=40, l=10, r=10), hovermode="x unified",
        xaxis_title="Temps (min)", yaxis_title="Part confiée à la PB (%, moyenne par minute)",
        legend=dict(orientation="h", y=1.1),
    )
    st.subheader("Décision et règle de référence au fil du cycle")
    st.plotly_chart(fig_c, width="stretch")

    if utilise_flou:
        st.subheader("Règles floues dominantes sur le cycle")
        dom = cy["dominante"][cy["actif"]]
        regles, comptes = np.unique(dom, return_counts=True)
        ordre = list(np.argsort(comptes)[::-1])
        st.plotly_chart(
            _barres_h(
                [ox.REGLES_FLOUES.get(regles[i], (regles[i],))[0] for i in ordre],
                [comptes[i] / comptes.sum() * 100 for i in ordre],
                [C_GRIS] * len(ordre),
                "Part des instants où la règle domine (%)",
                "%{y} : %{x:.0f} %<extra></extra>",
            ),
            width="stretch",
        )
    elif not exacte:
        st.caption(
            "Ce modèle n'a pas de règles internes : son explication reste approchée, instant "
            "par instant (onglet « À cet instant »)."
        )


pied_navigation("vues/7_Explicabilite.py")
