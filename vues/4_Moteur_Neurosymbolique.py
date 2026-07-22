import sys
from pathlib import Path

DOSSIER_PROJET = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DOSSIER_PROJET))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

import ems_core as core
from ems_core import alpha_fuzzy_calc, EPS_POWER_W
from core.resultats import assurer_donnees_session, nom_affichage
from core import ontology_explainer as ox


# Configuration de page gérée par le routeur Accueil.py.

C_EB = "#3B82F6"
C_PB = "#22C55E"
C_OK = "#22C55E"
C_ATTENTION = "#F59E0B"
C_GRIS = "#94A3B8"
MEDAILLES = ["🥇", "🥈", "🥉", "4e", "5e", "6e", "7e"]


def _scenario(p_dem, speed_kmh, accel):
    if abs(p_dem) <= EPS_POWER_W:
        return "Arrêt / roue libre"
    if p_dem < 0:
        return "Freinage / récupération"
    if accel > 0.8:
        return "Forte accélération"
    if speed_kmh > 90:
        return "Vitesse élevée"
    return "Traction régulière"


def _pensee(part_eb, part_pb, correction, p_dem):
    if abs(p_dem) <= EPS_POWER_W:
        texte = "Demande quasi nulle : je sollicite peu les batteries."
    elif p_dem < 0:
        cible = "Puissance" if part_pb >= part_eb else "Énergie"
        texte = f"Freinage : je dirige la récupération surtout vers la batterie {cible}."
    elif part_eb >= part_pb:
        texte = f"Je privilégie la batterie Énergie ({part_eb:.0f} %) car la demande reste gérable."
    else:
        texte = f"Je sollicite davantage la batterie Puissance ({part_pb:.0f} %) pour absorber le pic."
    if correction:
        texte += " Ma proposition a été ajustée par le filtre de sécurité."
    return texte


def _tag_decision(part_eb, part_pb, correction):
    if correction:
        return "décision corrigée par le filtre"
    if abs(part_eb - part_pb) <= 10:
        return "bon équilibre EB/PB"
    if part_eb > part_pb:
        return "préserve la batterie Puissance"
    return "soulage la batterie Énergie"


# Indice multicritère évalué à un instant donné. Chaque composante est calculée
# depuis la trajectoire ; les poids sont explicites et affichés à l'utilisateur.
# Le « temps de calcul » est volontairement absent : ce n'est pas une propriété
# de l'instant, et aucune mesure fiable par stratégie n'existe dans le fichier
# précalculé — l'inventer fausserait le classement.
CRITERES_INSTANT = [
    ("Respect des contraintes physiques", 25),
    ("Qualité de service (demande satisfaite)", 20),
    ("Coût énergétique", 20),
    ("Intervention du filtre de sécurité", 15),
    ("Préservation et équilibre des batteries", 10),
    ("Stabilité de la décision", 5),
    ("Cohérence avec OntoHESS", 5),
]


def _norm_min(valeurs):
    """Normalise un critère « plus bas = mieux » en sous-score 0..1 (1 = meilleur),
    relativement aux stratégies comparées à cet instant."""
    finis = [v for v in valeurs.values() if v == v]
    lo, hi = (min(finis), max(finis)) if finis else (0.0, 1.0)
    out = {}
    for nom, v in valeurs.items():
        if v != v:
            out[nom] = 0.0
        elif hi - lo < 1e-12:
            out[nom] = 1.0
        else:
            out[nom] = 1.0 - (v - lo) / (hi - lo)
    return out


def _evaluer(resultats, instant, p_dem, accel):
    """Évalue chaque stratégie à cet instant sur les sept critères pondérés.

    Retourne (points, total) où points[stratégie][critère] est le nombre de
    points obtenus sur le poids du critère.
    """
    brut = {}
    for nom, tr in resultats.items():
        se = float(tr["SOC_EB"][instant])
        sp = float(tr["SOC_PB"][instant])
        a = float(tr["alpha_final"][instant])
        a_req = float(tr["alpha_requested"][instant]) if "alpha_requested" in tr else a
        a_prev = float(tr["alpha_final"][instant - 1]) if instant > 0 else a

        try:
            m = core.candidate_metrics(np.array([a]), p_dem, se, sp, None)
            cout = float(m["total_cost"][0])
        except Exception:  # noqa: BLE001
            cout = float("nan")

        try:
            a_fuzzy = float(
                alpha_fuzzy_calc(
                    np.array([se]), np.array([sp]), np.array([p_dem]), np.array([accel])
                )["alpha"][0]
            )
        except Exception:  # noqa: BLE001
            a_fuzzy = a

        corrige = "correction_applied" in tr and bool(tr["correction_applied"][instant])
        brut[nom] = {
            "faisable": bool(tr["feasible"][instant]) if "feasible" in tr else True,
            "violation": bool(tr["soc_violation"][instant]) if "soc_violation" in tr else False,
            "service": abs(float(tr["P_unserved"][instant])) + abs(float(tr["P_regen_curtailed"][instant]))
            if ("P_unserved" in tr and "P_regen_curtailed" in tr)
            else 0.0,
            "cout": cout,
            "filtre": abs(a - a_req) + (1.0 if corrige else 0.0),
            "equilibre": abs(se - sp),
            "stabilite": abs(a - a_prev),
            "expert": abs(a - a_fuzzy),
        }

    s_service = _norm_min({n: b["service"] for n, b in brut.items()})
    s_cout = _norm_min({n: b["cout"] for n, b in brut.items()})
    s_filtre = _norm_min({n: b["filtre"] for n, b in brut.items()})
    s_equil = _norm_min({n: b["equilibre"] for n, b in brut.items()})
    s_stab = _norm_min({n: b["stabilite"] for n, b in brut.items()})
    s_exp = _norm_min({n: b["expert"] for n, b in brut.items()})

    points, total = {}, {}
    for nom, b in brut.items():
        # Critère de sécurité noté en ABSOLU : une stratégie ne mérite pas la
        # note maximale simplement parce que les autres font pire.
        if not b["faisable"]:
            p_phys = 0.0
        elif b["violation"]:
            p_phys = 25 * 0.5
        else:
            p_phys = 25.0

        points[nom] = {
            "Respect des contraintes physiques": p_phys,
            "Qualité de service (demande satisfaite)": 20 * s_service[nom],
            "Coût énergétique": 20 * s_cout[nom],
            "Intervention du filtre de sécurité": 15 * s_filtre[nom],
            "Préservation et équilibre des batteries": 10 * s_equil[nom],
            "Stabilité de la décision": 5 * s_stab[nom],
            # Écart à la proposition des règles floues, elles-mêmes fondées sur
            # les concepts et seuils déclarés dans l'ontologie OntoHESS.
            "Cohérence avec OntoHESS": 5 * s_exp[nom],
        }
        total[nom] = sum(points[nom].values())

    return points, total


st.title("📊 Analyse instantanée")
st.caption(
    "Comprenez comment chaque stratégie réagit à un instant précis du cycle de "
    "conduite. Ici on compare les approches ; pour explorer en profondeur une "
    "seule stratégie, voir « Pourquoi cette décision ? »."
)


try:
    _source = assurer_donnees_session(st)
except FileNotFoundError as exc:
    st.error(str(exc))
    st.info("Lance une fois le précalcul :  `python scripts/run_simulations.py`")
    st.stop()

resultats = st.session_state.get("resultats_simulation")
df = st.session_state.get("cycle_pret")
if not resultats or df is None:
    st.warning("Aucune donnée disponible.")
    st.stop()

st.caption(f"Source des données : {_source}")

n = min([len(df)] + [len(tr["P_EB"]) for tr in resultats.values()])
instant = st.slider("Instant du cycle à analyser", 0, n - 1, n // 2)

ligne = df.iloc[instant]
t_sel = float(ligne["time"]) if "time" in df.columns else float(instant)
speed = float(ligne["speed"]) if "speed" in df.columns else 0.0
speed_kmh = speed * 3.6
accel = float(ligne["hasAcceleration"]) if "hasAcceleration" in df.columns else 0.0
p_dem = float(ligne["hasPower"])
scenario = _scenario(p_dem, speed_kmh, accel)


# Situation actuelle

st.subheader("Situation actuelle")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Vitesse", f"{speed_kmh:.0f} km/h")
c2.metric("Puissance demandée", f"{p_dem / 1000:.1f} kW")
c3.metric("Accélération", f"{accel:+.1f} m/s²")
c4.metric("Situation", scenario)
st.caption(f"Instant analysé : t = {t_sel:.0f} s.")


# Décisions des modèles (cartes, plus de dataframe brut)

st.subheader("📊 Décisions des modèles")

noms = list(resultats.keys())
par_ligne = 4
for i0 in range(0, len(noms), par_ligne):
    bloc = noms[i0 : i0 + par_ligne]
    cols = st.columns(par_ligne)
    for col, nom in zip(cols, bloc):
        tr = resultats[nom]
        p_eb = float(tr["P_EB"][instant])
        p_pb = float(tr["P_PB"][instant])
        alpha = float(tr["alpha_final"][instant])
        corr = bool(tr["correction_applied"][instant]) if "correction_applied" in tr else False
        tot = abs(p_eb) + abs(p_pb)
        part_eb = 100.0 * abs(p_eb) / tot if tot > 1.0 else 0.0
        part_pb = 100.0 * abs(p_pb) / tot if tot > 1.0 else 0.0
        with col:
            with st.container(border=True):
                st.markdown(f"**{nom_affichage(nom)}**")
                st.markdown(
                    f"<span style='color:{C_EB};font-weight:700'>EB {part_eb:.0f} %</span> · "
                    f"<span style='color:{C_PB};font-weight:700'>PB {part_pb:.0f} %</span>",
                    unsafe_allow_html=True,
                )
                st.caption(f"alpha = {alpha:.2f}")
                coul = C_ATTENTION if corr else C_OK
                mot = "corrigée" if corr else "acceptée"
                st.markdown(
                    f"<span style='color:{coul}'>&#9679; {mot}</span>",
                    unsafe_allow_html=True,
                )


# Évolution autour de cet instant (une seule figure, stratégie au choix)

st.subheader("📈 Évolution autour de cet instant")

strategie_focus = st.selectbox(
    "Stratégie détaillée sur les courbes",
    noms,
    format_func=nom_affichage,
)
tr_f = resultats[strategie_focus]

demi = 150
i0, i1 = max(0, instant - demi), min(n, instant + demi + 1)
xx = df["time"].to_numpy()[i0:i1]

fig = make_subplots(
    rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
    subplot_titles=("Puissances (kW)", "États de charge (%)"),
)
fig.add_trace(go.Scatter(x=xx, y=df["hasPower"].to_numpy()[i0:i1] / 1000.0, name="P_dem", line=dict(color=C_GRIS)), row=1, col=1)
fig.add_trace(go.Scatter(x=xx, y=np.asarray(tr_f["P_EB"], float)[i0:i1] / 1000.0, name="P_EB", line=dict(color=C_EB)), row=1, col=1)
fig.add_trace(go.Scatter(x=xx, y=np.asarray(tr_f["P_PB"], float)[i0:i1] / 1000.0, name="P_PB", line=dict(color=C_PB)), row=1, col=1)
fig.add_trace(go.Scatter(x=xx, y=np.asarray(tr_f["SOC_EB"], float)[i0:i1] * 100.0, name="SOC_EB", line=dict(color=C_EB, dash="dot")), row=2, col=1)
fig.add_trace(go.Scatter(x=xx, y=np.asarray(tr_f["SOC_PB"], float)[i0:i1] * 100.0, name="SOC_PB", line=dict(color=C_PB, dash="dot")), row=2, col=1)
fig.add_vline(x=t_sel, line=dict(color="#EF4444", dash="dash"))
fig.update_layout(height=460, margin=dict(t=50, b=30), hovermode="x unified", legend=dict(orientation="h", y=1.12))
st.plotly_chart(fig, use_container_width=True)


# Construction de la décision (variante neuro-symbolique)

if "EMS_MLP_neurosymbolic" in resultats:
    st.subheader("🧠 Construction de la décision (MLP neuro-symbolique)")
    tr_ns = resultats["EMS_MLP_neurosymbolic"]
    soc_eb_ns = float(tr_ns["SOC_EB"][instant])
    soc_pb_ns = float(tr_ns["SOC_PB"][instant])
    alpha_final_ns = float(tr_ns["alpha_final"][instant])
    alpha_req_ns = float(tr_ns["alpha_requested"][instant]) if "alpha_requested" in tr_ns else alpha_final_ns
    alpha_fuzzy = float(
        alpha_fuzzy_calc(np.array([soc_eb_ns]), np.array([soc_pb_ns]), np.array([p_dem]), np.array([accel]))["alpha"][0]
    )
    delta = alpha_req_ns - alpha_fuzzy

    _etats_ns = core.compute_symbolic_states(p_dem, soc_eb_ns, soc_pb_ns)
    _libelles_etats = {
        "high_power_demand": "Forte demande de puissance",
        "EB_low_SOC": "SOC batterie Énergie faible",
        "EB_available": "Batterie Énergie disponible",
        "PB_available": "Batterie Puissance disponible",
        "regenerative_braking": "Freinage régénératif",
        "converter_risk": "Convertisseur proche de sa limite",
    }
    _detectes = [lib for cle, lib in _libelles_etats.items() if _etats_ns.get(cle)]

    def _fleche():
        st.markdown(f"<div style='color:{C_GRIS};font-size:1.1em'>&#8595;</div>", unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown("**Ontologie OntoHESS**")
        st.caption("Concepts et seuils décrivant les composants et les états du HESS.")
        _fleche()
        st.markdown("**Détection des états symboliques**")
        if _detectes:
            for _etat in _detectes:
                st.markdown(f"- {_etat}")
        else:
            st.caption("Aucun état particulier détecté : situation nominale.")
        _fleche()
        st.markdown(f"**Règles expertes floues** — {alpha_fuzzy * 100:.0f} % pour la PB")
        _fleche()
        st.markdown(f"**Correction du réseau neuronal** — {delta * 100:+.0f} %")
        _fleche()
        st.markdown("**Filtre physique de sécurité**")
        _fleche()
        st.markdown(f"**Décision finale** — {alpha_final_ns * 100:.0f} % pour la PB")

    st.caption(
        "La décision ne sort pas directement du réseau : elle passe d'abord par les "
        "connaissances métier de l'ontologie, puis par les règles expertes. Le réseau "
        "n'apporte qu'une correction bornée."
    )


# Raisonnement ontologique : les concepts métier actifs à cet instant

st.subheader("🧩 Raisonnement ontologique")

# L'état de charge dépend de la stratégie : on prend celle sélectionnée plus haut
# pour les courbes, afin que le raisonnement porte sur un état cohérent.
soc_eb_ref = float(tr_f["SOC_EB"][instant])
soc_pb_ref = float(tr_f["SOC_PB"][instant])

with st.container(border=True):
    st.markdown(
        "L'ontologie **OntoHESS** décrit les composants du système (batterie Énergie, "
        "batterie Puissance, convertisseur, charge) ainsi que les seuils et les états "
        "de fonctionnement. À partir des mesures physiques de cet instant, elle permet "
        "d'identifier automatiquement les concepts métier actifs."
    )
    st.caption(
        f"États de charge pris comme référence : ceux de **{nom_affichage(strategie_focus)}** "
        "(la stratégie choisie pour les courbes ci-dessus)."
    )

    _interp = ox.interpretation_ontologique(p_dem, soc_eb_ref, soc_pb_ref)
    st.markdown(
        f"**État de fonctionnement inféré** : {ox.ETATS_ONTOLOGIE[_interp['etat']]} "
        f"(`{_interp['etat']}`)"
    )

    _concepts = ox.concepts_actifs(p_dem, soc_eb_ref, soc_pb_ref)
    _actifs = [c for c in _concepts if c["actif"]]
    if _actifs:
        st.markdown("**États détectés**")
        for _c in _actifs:
            st.markdown(f"- {_c['libelle']}")
    else:
        st.caption("Aucun concept particulier détecté : la situation est nominale.")


# Raisonnement de chaque modèle (langage naturel)

st.subheader("💬 Raisonnement de chaque modèle")
for nom in noms:
    tr = resultats[nom]
    p_eb = float(tr["P_EB"][instant])
    p_pb = float(tr["P_PB"][instant])
    corr = bool(tr["correction_applied"][instant]) if "correction_applied" in tr else False
    tot = abs(p_eb) + abs(p_pb)
    part_eb = 100.0 * abs(p_eb) / tot if tot > 1.0 else 0.0
    part_pb = 100.0 * abs(p_pb) / tot if tot > 1.0 else 0.0
    st.markdown(f"- **{nom_affichage(nom)}** : « {_pensee(part_eb, part_pb, corr, p_dem)} »")


# Classement à cet instant

st.subheader("🏆 Quelle stratégie est la plus adaptée à cet instant ?")
st.caption(
    "Indice multicritère pondéré, calculé sur l'état du système à cet instant. "
    "Sauf le critère de sécurité (noté en absolu), chaque critère est normalisé "
    "entre les stratégies comparées."
)

if abs(p_dem) <= EPS_POWER_W:
    st.info("Demande quasi nulle à cet instant : l'évaluation n'est pas significative.")
    classement = []
    points_crit = {}
else:
    points_crit, totaux = _evaluer(resultats, instant, p_dem, accel)
    classement = sorted(totaux.items(), key=lambda kv: (-kv[1], kv[0]))

    cols = st.columns(len(classement))
    for col, (nom_c, score_c), medaille in zip(cols, classement, MEDAILLES):
        tr = resultats[nom_c]
        p_eb = float(tr["P_EB"][instant])
        p_pb = float(tr["P_PB"][instant])
        corr = bool(tr["correction_applied"][instant]) if "correction_applied" in tr else False
        tot = abs(p_eb) + abs(p_pb)
        part_eb = 100.0 * abs(p_eb) / tot if tot > 1.0 else 0.0
        part_pb = 100.0 * abs(p_pb) / tot if tot > 1.0 else 0.0
        with col:
            with st.container(border=True):
                st.markdown(
                    f"<div style='text-align:center;font-size:22px'>{medaille}</div>"
                    f"<div style='text-align:center;font-weight:700'>{nom_affichage(nom_c)}</div>"
                    f"<div style='text-align:center;color:{C_GRIS};font-size:.85em'>{score_c:.0f}/100</div>"
                    f"<div style='text-align:center;color:{C_GRIS};font-size:.8em'>{_tag_decision(part_eb, part_pb, corr)}</div>",
                    unsafe_allow_html=True,
                )

    # D'où vient le score : le détail par critère, pour toutes les stratégies.
    st.markdown("**D'où vient le score ?**")
    lignes_score = []
    for libelle, poids in CRITERES_INSTANT:
        ligne = {"Critère (poids)": f"{libelle} ({poids})"}
        for nom_c in resultats:
            ligne[nom_affichage(nom_c)] = f"{points_crit[nom_c][libelle]:.0f}"
        lignes_score.append(ligne)
    ligne_finale = {"Critère (poids)": "Score final (100)"}
    for nom_c in resultats:
        ligne_finale[nom_affichage(nom_c)] = f"{sum(points_crit[nom_c].values()):.0f}"
    lignes_score.append(ligne_finale)

    st.dataframe(pd.DataFrame(lignes_score).set_index("Critère (poids)"), use_container_width=True)
    st.caption(
        "Le temps de calcul n'entre pas dans l'indice : ce n'est pas une propriété de "
        "l'instant, et aucune mesure fiable par stratégie n'est disponible ici."
    )


# Verdict + à retenir

st.subheader("📋 Verdict")
if classement:
    nom_best, score_best = classement[0]
    detail_best = points_crit[nom_best]
    forts = [
        libelle
        for libelle, poids in CRITERES_INSTANT
        if poids > 0 and detail_best[libelle] >= 0.9 * poids
    ]
    faibles = [
        libelle
        for libelle, poids in CRITERES_INSTANT
        if poids > 0 and detail_best[libelle] <= 0.4 * poids
    ]

    with st.container(border=True):
        st.markdown(
            f"À cet instant, **{nom_affichage(nom_best)}** présente le **meilleur compromis** "
            f"selon les {len(CRITERES_INSTANT)} critères d'évaluation retenus "
            f"({score_best:.0f}/100)."
        )
        if forts:
            st.markdown("Points forts à cet instant :")
            for f in forts:
                st.markdown(f"- {f}")
        if faibles:
            st.markdown("Points faibles à cet instant :")
            for f in faibles:
                st.markdown(f"- {f}")
        st.caption(
            "Ce résultat est **local à cet instant du cycle** et ne préjuge pas des "
            "performances globales sur l'ensemble de la simulation. Pour le classement "
            "sur tout le cycle, voir « Comparer les méthodes »."
        )

    st.subheader("🎓 Ce qu'il faut retenir")
    alphas = np.array([float(resultats[nm]["alpha_final"][instant]) for nm in noms])
    ecart = float(alphas.max() - alphas.min())
    ns_noms = [nm for nm in noms if "neurosymbolic" in nm]
    corr_ns = sum(
        1 for nm in ns_noms if "correction_applied" in resultats[nm] and bool(resultats[nm]["correction_applied"][instant])
    )
    corr_autres = sum(
        1
        for nm in noms
        if nm not in ns_noms and "correction_applied" in resultats[nm] and bool(resultats[nm]["correction_applied"][instant])
    )
    if ecart < 0.10:
        phrase = "À cet instant, les modèles prennent des décisions très proches."
    else:
        phrase = f"À cet instant, les modèles divergent nettement (écart d'alpha de {ecart:.2f})."
    if ns_noms and corr_ns < corr_autres:
        phrase += (
            " Ici, les variantes neuro-symboliques nécessitent moins d'interventions du "
            "filtre de sécurité que les autres stratégies."
        )
    elif ns_noms and corr_ns > corr_autres:
        phrase += (
            " À cet instant, les variantes neuro-symboliques nécessitent davantage "
            "d'interventions du filtre que les autres stratégies."
        )
    st.info(phrase)

    st.caption(
        "Rappel méthodologique : les variantes neuro-symboliques exploitent les "
        "connaissances de l'ontologie OntoHESS (concepts, seuils et règles) **avant** "
        "la correction neuronale. Leur décision part donc d'une base déjà cohérente "
        "avec les contraintes physiques, que le réseau ne fait qu'ajuster à la marge."
    )


from core.navigation import pied_navigation

pied_navigation("vues/4_Moteur_Neurosymbolique.py")
