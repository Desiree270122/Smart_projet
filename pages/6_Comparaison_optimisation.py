
import sys
from pathlib import Path

DOSSIER_PROJET = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DOSSIER_PROJET))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from ems_core import optimiser_alpha_star_sequence, MODEL_DISPLAY_NAMES


# ============================================================
# Configuration de la page
# ============================================================

st.set_page_config(
    page_title="2SMART — Comparaison et optimisation",
    layout="wide",
)

st.title("Comparaison et optimisation")


# ============================================================
# Introduction
# ============================================================

st.write(
    "Cette page compare les stratégies EMS déjà simulées à partir d’indicateurs "
    "physiques et explicables. L’objectif n’est pas seulement de conserver un "
    "SOC final élevé, mais de vérifier si la stratégie satisfait correctement "
    "la demande du véhicule, respecte les contraintes physiques, limite les "
    "corrections du filtre de sécurité et préserve les deux batteries."
)

st.info(
    "La référence alpha-star est une borne théorique calculée hors ligne. "
    "Elle sert uniquement de point de comparaison et ne représente pas une "
    "stratégie directement déployable en temps réel."
)


# ============================================================
# Vérification des données disponibles
# ============================================================

if (
    "resultats_simulation" not in st.session_state
    or "cycle_pret" not in st.session_state
):
    st.warning(
        "Une simulation globale doit être réalisée avant de consulter cette page."
    )

    if st.button("Ouvrir la page de simulation globale"):
        st.switch_page("pages/3_Simulation_globale.py")

    st.stop()


resultats = st.session_state["resultats_simulation"]
df = st.session_state["cycle_pret"]

soc_eb0 = st.session_state.get("soc_eb0", 1.0)
soc_pb0 = st.session_state.get("soc_pb0", 1.0)

if not resultats:
    st.warning("Aucune stratégie n’a produit de résultat exploitable.")
    st.stop()


# ============================================================
# Fonctions utilitaires
# ============================================================

SOC_MIN = 0.20
SOC_MAX = 1.00
SOC_RANGE = SOC_MAX - SOC_MIN


def get_array(traj, key, n=None, default=0.0, dtype=float):
    """
    Récupère une variable de trajectoire sous forme de tableau numpy.
    Si la variable n’existe pas, retourne un tableau par défaut.
    """
    if key in traj:
        return np.asarray(traj[key], dtype=dtype)

    if n is None:
        return np.asarray([], dtype=dtype)

    return np.full(n, default, dtype=dtype)


def get_scalar(traj, key, default=np.nan):
    """Récupère une valeur scalaire de trajectoire."""
    value = traj.get(key, default)

    try:
        return float(value)
    except Exception:
        return default


def format_nombre(x, ndigits=3):
    """Formatage robuste des nombres."""
    if x is None or not np.isfinite(x):
        return "n/a"

    return f"{x:.{ndigits}f}"


def energie_positive_wh(p):
    """Énergie délivrée positive en Wh."""
    p = np.asarray(p, dtype=float)
    return float(np.maximum(p, 0.0).sum() / 3600.0)


def energie_negative_wh(p):
    """Énergie récupérée en Wh à partir d’une puissance négative."""
    p = np.asarray(p, dtype=float)
    return float((-np.minimum(p, 0.0)).sum() / 3600.0)


def calculer_demande_positive_wh(df):
    """Calcule la demande positive totale du cycle."""
    if "hasPower" not in df.columns:
        return 1.0

    p_dem = df["hasPower"].to_numpy(dtype=float)
    return max(1.0, float(np.maximum(p_dem, 0.0).sum() / 3600.0))


def calculer_metriques_strategie(
    nom,
    traj,
    demande_positive_wh,
    deployable=True,
):
    """
    Calcule les indicateurs principaux d'une stratégie.
    """
    p_eb = get_array(traj, "P_EB")
    n = len(p_eb)

    p_pb = get_array(traj, "P_PB", n=n)
    p_unserved = get_array(traj, "P_unserved", n=n)
    p_regen_curtailed = get_array(traj, "P_regen_curtailed", n=n)

    alpha_final = get_array(traj, "alpha_final", n=n)
    alpha_requested = get_array(traj, "alpha_requested", n=n)

    correction = get_array(
        traj,
        "correction_applied",
        n=n,
        default=False,
        dtype=bool,
    )

    soc_violation = get_array(
        traj,
        "soc_violation",
        n=n,
        default=False,
        dtype=bool,
    )

    feasible = get_array(
        traj,
        "feasible",
        n=n,
        default=True,
        dtype=bool,
    )

    soc_eb = get_array(traj, "SOC_EB")
    soc_pb = get_array(traj, "SOC_PB")

    soc_eb_final = get_scalar(
        traj,
        "SOC_EB_final",
        soc_eb[-1] if len(soc_eb) else np.nan,
    )

    soc_pb_final = get_scalar(
        traj,
        "SOC_PB_final",
        soc_pb[-1] if len(soc_pb) else np.nan,
    )

    soc_eb_min = float(np.nanmin(soc_eb)) if len(soc_eb) else np.nan
    soc_pb_min = float(np.nanmin(soc_pb)) if len(soc_pb) else np.nan

    energie_eb_wh = energie_positive_wh(p_eb)
    energie_pb_wh = energie_positive_wh(p_pb)
    energie_totale_batteries_wh = max(1.0, energie_eb_wh + energie_pb_wh)

    energie_non_servie_wh = float(np.maximum(p_unserved, 0.0).sum() / 3600.0)
    energie_regen_perdue_wh = float(np.maximum(p_regen_curtailed, 0.0).sum() / 3600.0)

    taux_correction = 100.0 * float(np.mean(correction)) if n > 0 else 0.0
    taux_faisabilite = 100.0 * float(np.mean(feasible)) if n > 0 else 0.0
    taux_respect_soc = 100.0 * (1.0 - float(np.mean(soc_violation))) if n > 0 else 100.0

    cout_moyen = (
        float(np.nanmean(traj["cost"]))
        if "cost" in traj and len(traj["cost"]) > 0
        else np.nan
    )

    cout_total = (
        float(np.nansum(traj["cost"]))
        if "cost" in traj and len(traj["cost"]) > 0
        else np.nan
    )

    part_energie_pb = 100.0 * energie_pb_wh / energie_totale_batteries_wh

    ecart_soc_final = (
        abs(soc_eb_final - soc_pb_final)
        if np.isfinite(soc_eb_final) and np.isfinite(soc_pb_final)
        else np.nan
    )

    alpha_moyen = float(np.nanmean(alpha_final)) if len(alpha_final) else np.nan
    alpha_max = float(np.nanmax(alpha_final)) if len(alpha_final) else np.nan

    taux_alpha_fort = (
        100.0 * float(np.mean(alpha_final > 0.50))
        if len(alpha_final)
        else np.nan
    )

    # ========================================================
    # Scores XAI normalisés entre 0 et 1
    # ========================================================

    score_demande = np.clip(
        1.0 - energie_non_servie_wh / demande_positive_wh,
        0.0,
        1.0,
    )

    marge_soc_min = min(
        soc_eb_min if np.isfinite(soc_eb_min) else SOC_MIN,
        soc_pb_min if np.isfinite(soc_pb_min) else SOC_MIN,
    )

    score_soc = np.clip(
        (marge_soc_min - SOC_MIN) / SOC_RANGE,
        0.0,
        1.0,
    )

    score_correction = np.clip(
        1.0 - taux_correction / 100.0,
        0.0,
        1.0,
    )

    score_equilibre = np.clip(
        1.0 - ecart_soc_final / SOC_RANGE,
        0.0,
        1.0,
    )

    score_pb = np.clip(
        (soc_pb_final - SOC_MIN) / SOC_RANGE,
        0.0,
        1.0,
    )

    poids_xai = {
        "Satisfaction de la demande": 0.35,
        "Respect des limites SOC": 0.25,
        "Peu de corrections": 0.20,
        "Équilibre final EB/PB": 0.10,
        "Préservation de la PB": 0.10,
    }

    scores_xai = {
        "Satisfaction de la demande": score_demande,
        "Respect des limites SOC": score_soc,
        "Peu de corrections": score_correction,
        "Équilibre final EB/PB": score_equilibre,
        "Préservation de la PB": score_pb,
    }

    score_global = 100.0 * sum(
        poids_xai[k] * scores_xai[k]
        for k in poids_xai
    )

    if not deployable:
        verdict = "Référence théorique"
    elif energie_non_servie_wh > 0.10 * demande_positive_wh:
        verdict = "Demande insuffisamment servie"
    elif taux_correction > 30.0:
        verdict = "Décisions souvent corrigées"
    elif soc_pb_final <= SOC_MIN + 0.03:
        verdict = "PB fortement sollicitée"
    elif score_global >= 80.0:
        verdict = "Robuste"
    elif score_global >= 65.0:
        verdict = "Bon compromis"
    else:
        verdict = "À améliorer"

    return {
        "Code": nom,
        "Stratégie": MODEL_DISPLAY_NAMES.get(nom, nom),
        "Déployable": "Oui" if deployable else "Non",
        "Score global XAI": score_global,
        "Verdict": verdict,
        "Faisabilité (%)": taux_faisabilite,
        "Respect SOC (%)": taux_respect_soc,
        "Corrections filtre (%)": taux_correction,
        "Déficit énergétique (Wh)": energie_non_servie_wh,
        "Énergie régénérative perdue (Wh)": energie_regen_perdue_wh,
        "Énergie EB délivrée (Wh)": energie_eb_wh,
        "Énergie PB délivrée (Wh)": energie_pb_wh,
        "Part énergie PB (%)": part_energie_pb,
        "SOC EB final": soc_eb_final,
        "SOC PB final": soc_pb_final,
        "SOC EB min": soc_eb_min,
        "SOC PB min": soc_pb_min,
        "Écart SOC final": ecart_soc_final,
        "Alpha moyen": alpha_moyen,
        "Alpha max": alpha_max,
        "Alpha > 0.5 (%)": taux_alpha_fort,
        "Coût total": cout_total,
        "Coût moyen": cout_moyen,
        "Scores XAI": scores_xai,
        "Poids XAI": poids_xai,
    }


def generer_explication_xai(row):
    """
    Génère une explication XAI courte et utile.
    """
    strategie = row["Stratégie"]
    verdict = row["Verdict"]

    texte = f"**{strategie}** — verdict : **{verdict}**. "

    energie_non_servie = row["Déficit énergétique (Wh)"]
    correction = row["Corrections filtre (%)"]
    soc_eb_final = row["SOC EB final"]
    soc_pb_final = row["SOC PB final"]
    part_pb = row["Part énergie PB (%)"]
    score = row["Score global XAI"]

    texte += (
        f"Le score global XAI est de **{score:.1f}/100**. "
    )

    if energie_non_servie > 1000:
        texte += (
            "La principale faiblesse est une déficit énergétique élevé, "
            "ce qui signifie que la stratégie préserve certaines batteries "
            "mais ne satisfait pas suffisamment la demande du véhicule. "
        )
    elif energie_non_servie > 100:
        texte += (
            "Une partie de la demande n’est pas servie, mais le niveau reste "
            "modéré par rapport aux stratégies les plus pénalisées. "
        )
    else:
        texte += (
            "La demande du véhicule est globalement bien satisfaite. "
        )

    if correction > 25:
        texte += (
            "Le filtre de sécurité intervient souvent, ce qui indique que "
            "les décisions proposées par la stratégie doivent régulièrement "
            "être corrigées pour respecter les contraintes physiques. "
        )
    elif correction > 5:
        texte += (
            "Le filtre de sécurité intervient de manière ponctuelle, ce qui "
            "montre que la stratégie reste globalement exploitable mais nécessite "
            "encore des corrections. "
        )
    else:
        texte += (
            "Les décisions nécessitent peu de corrections, ce qui traduit un "
            "comportement naturellement cohérent avec les contraintes du système. "
        )

    if soc_pb_final <= SOC_MIN + 0.03:
        texte += (
            "La PB termine très proche de son seuil minimal, ce qui montre "
            "qu’elle a été fortement sollicitée pendant le cycle. "
        )
    else:
        texte += (
            "La PB conserve une marge finale exploitable. "
        )

    if soc_eb_final > soc_pb_final + 0.15:
        texte += (
            "Le SOC final de l’EB est nettement supérieur à celui de la PB : "
            "la stratégie utilise donc fortement la batterie de puissance. "
        )
    elif soc_pb_final > soc_eb_final + 0.15:
        texte += (
            "Le SOC final de la PB est nettement supérieur à celui de l’EB : "
            "la stratégie donne bien la priorité à la batterie d’énergie. "
        )
    else:
        texte += (
            "Les SOC finaux restent relativement équilibrés. "
        )

    texte += (
        f"La PB fournit environ **{part_pb:.1f} %** de l’énergie positive "
        "délivrée par les batteries sur ce cycle."
    )

    return texte


def afficher_graphique_barres_horizontales(df_plot, colonne, titre, xlabel):
    """
    Affiche un graphique horizontal lisible.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    df_local = df_plot.sort_values(colonne, ascending=True)

    ax.barh(
        df_local["Stratégie courte"],
        df_local[colonne],
    )

    ax.set_title(titre)
    ax.set_xlabel(xlabel)
    ax.grid(True, axis="x", alpha=0.3)

    st.pyplot(fig)
    plt.close(fig)


def nom_court(nom):
    """
    Simplifie les noms pour les graphiques.
    """
    remplacements = {
        "EMS limitation de puissance (priorité EB)": "Limitation EB",
        "EMS logique floue": "Floue",
        "EMS MLP": "MLP",
        "EMS MLP neurosymbolique": "MLP NS",
        "EMS LSTM": "LSTM",
        "EMS LSTM neurosymbolique": "LSTM NS",
        "EMS GNN": "GNN",
        "Alpha-star (référence non déployable)": "Alpha-star",
    }

    return remplacements.get(nom, nom)


# ============================================================
# 1. Référence physique théorique
# ============================================================

st.subheader("1. Référence physique théorique")

if "alpha_star_reference" not in st.session_state:
    with st.spinner(
        "Calcul de la référence alpha-star en cours..."
    ):
        st.session_state["alpha_star_reference"] = optimiser_alpha_star_sequence(
            df,
            soc_eb0,
            soc_pb0,
        )

    st.success("La référence alpha-star a été calculée pour ce cycle.")

else:
    st.info(
        "La référence alpha-star a déjà été calculée pour ce cycle. "
        "Aucun nouveau calcul n’est nécessaire."
    )

alpha_star_traj = st.session_state["alpha_star_reference"]


# ============================================================
# 2. Construction du tableau de comparaison
# ============================================================

demande_positive_wh = calculer_demande_positive_wh(df)

lignes = []

for nom, traj in resultats.items():
    lignes.append(
        calculer_metriques_strategie(
            nom=nom,
            traj=traj,
            demande_positive_wh=demande_positive_wh,
            deployable=True,
        )
    )

ligne_alpha_star = calculer_metriques_strategie(
    nom="Alpha-star (référence non déployable)",
    traj=alpha_star_traj,
    demande_positive_wh=demande_positive_wh,
    deployable=False,
)

ligne_alpha_star["Stratégie"] = "Alpha-star (référence non déployable)"
ligne_alpha_star["Verdict"] = "Borne théorique non déployable"

lignes.append(ligne_alpha_star)

tableau = pd.DataFrame(lignes)

tableau["Stratégie courte"] = tableau["Stratégie"].apply(nom_court)

colonnes_affichage = [
    "Stratégie",
    "Déployable",
    "Score global XAI",
    "Verdict",
    "Faisabilité (%)",
    "Respect SOC (%)",
    "Corrections filtre (%)",
    "Déficit énergétique (Wh)",
    "Énergie régénérative perdue (Wh)",
    "Énergie EB délivrée (Wh)",
    "Énergie PB délivrée (Wh)",
    "Part énergie PB (%)",
    "SOC EB final",
    "SOC PB final",
    "Écart SOC final",
    "Alpha moyen",
]


# ============================================================
# 2. Synthèse comparative
# ============================================================

st.subheader("2. Synthèse comparative des stratégies")

deployables = tableau[tableau["Déployable"] == "Oui"].copy()

meilleure_score = deployables.sort_values(
    "Score global XAI",
    ascending=False,
).iloc[0]

meilleure_demande = deployables.sort_values(
    "Déficit énergétique (Wh)",
    ascending=True,
).iloc[0]

meilleure_correction = deployables.sort_values(
    "Corrections filtre (%)",
    ascending=True,
).iloc[0]

c1, c2, c3 = st.columns(3)

with c1:
    st.metric(
        "Meilleur score XAI",
        meilleure_score["Stratégie"],
        f"{meilleure_score['Score global XAI']:.1f}/100",
    )

with c2:
    st.metric(
        "Demande la mieux servie",
        meilleure_demande["Stratégie"],
        f"{meilleure_demande['Déficit énergétique (Wh)']:.1f} Wh de déficit",
    )

with c3:
    st.metric(
        "Moins de corrections",
        meilleure_correction["Stratégie"],
        f"{meilleure_correction['Corrections filtre (%)']:.1f} %",
    )

st.write(
    "Le score XAI combine plusieurs critères : satisfaction de la demande, "
    "respect des limites SOC, faible intervention du filtre de sécurité, "
    "équilibre final entre EB et PB, et préservation de la batterie de puissance. "
    "Ce score évite de choisir une stratégie uniquement parce qu’elle garde un "
    "SOC final élevé."
)


# ============================================================
# 3. Tableau détaillé
# ============================================================

st.subheader("3. Tableau détaillé des indicateurs")

critere = st.selectbox(
    "Critère de classement",
    [
        "Score global XAI",
        "Déficit énergétique (Wh)",
        "Corrections filtre (%)",
        "Faisabilité (%)",
        "Respect SOC (%)",
        "SOC PB final",
        "Écart SOC final",
    ],
)

ascendant = critere in [
    "Déficit énergétique (Wh)",
    "Corrections filtre (%)",
    "Écart SOC final",
]

tableau_classe = (
    tableau
    .sort_values(
        critere,
        ascending=ascendant,
    )
    .reset_index(drop=True)
)

st.dataframe(
    tableau_classe[colonnes_affichage].round(4),
    use_container_width=True,
    hide_index=True,
)

meilleure_deployable = (
    tableau_classe[tableau_classe["Déployable"] == "Oui"]
    .iloc[0]
)

st.success(
    "Selon le critère sélectionné, la meilleure stratégie déployable est "
    f"**{meilleure_deployable['Stratégie']}**. "
    "Ce résultat doit être interprété avec les autres indicateurs, surtout "
    "l’déficit énergétique, les corrections du filtre et l’état final de la PB."
)


# ============================================================
# 4. Synthèse XAI par stratégie
# ============================================================

st.subheader("4. Explication XAI par stratégie")

strategie_xai = st.selectbox(
    "Choisir une stratégie à expliquer",
    tableau["Stratégie"].tolist(),
)

ligne_xai = tableau[
    tableau["Stratégie"] == strategie_xai
].iloc[0]

with st.container(border=True):
    st.write(
        generer_explication_xai(ligne_xai)
    )

scores_xai = ligne_xai["Scores XAI"]
poids_xai = ligne_xai["Poids XAI"]

df_xai = pd.DataFrame(
    {
        "Facteur XAI": list(scores_xai.keys()),
        "Score facteur": [scores_xai[k] for k in scores_xai],
        "Poids": [poids_xai[k] for k in scores_xai],
        "Contribution pondérée": [
            100.0 * scores_xai[k] * poids_xai[k]
            for k in scores_xai
        ],
    }
)

st.write(
    "Lecture XAI : plus une contribution pondérée est élevée, plus ce facteur "
    "a contribué positivement au score global de la stratégie."
)

st.dataframe(
    df_xai.round(4),
    use_container_width=True,
    hide_index=True,
)

fig_xai, ax_xai = plt.subplots(figsize=(9, 4))

df_xai_plot = df_xai.sort_values(
    "Contribution pondérée",
    ascending=True,
)

ax_xai.barh(
    df_xai_plot["Facteur XAI"],
    df_xai_plot["Contribution pondérée"],
)

ax_xai.set_xlabel("Contribution au score global")
ax_xai.set_title(f"Explication XAI — {strategie_xai}")
ax_xai.grid(True, axis="x", alpha=0.3)

st.pyplot(fig_xai)
plt.close(fig_xai)


# ============================================================
# 5. Instants critiques explicables
# ============================================================

st.subheader("5. Instants critiques de la stratégie sélectionnée")

if ligne_xai["Déployable"] == "Non":
    st.info(
        "La référence alpha-star est une borne théorique. "
        "L’analyse des instants critiques est surtout utile pour les stratégies déployables."
    )

else:
    code_selectionne = ligne_xai["Code"]
    traj_selectionne = resultats[code_selectionne]

    p_eb = get_array(traj_selectionne, "P_EB")
    n = len(p_eb)

    p_pb = get_array(traj_selectionne, "P_PB", n=n)
    p_unserved = get_array(traj_selectionne, "P_unserved", n=n)
    alpha_req = get_array(traj_selectionne, "alpha_requested", n=n)
    alpha_final = get_array(traj_selectionne, "alpha_final", n=n)
    correction = get_array(
        traj_selectionne,
        "correction_applied",
        n=n,
        default=False,
        dtype=bool,
    )

    soc_eb = get_array(traj_selectionne, "SOC_EB")
    soc_pb = get_array(traj_selectionne, "SOC_PB")

    if "time" in df.columns:
        temps = df["time"].to_numpy()[:n]
    else:
        temps = np.arange(n)

    if "hasPower" in df.columns:
        p_dem = df["hasPower"].to_numpy(dtype=float)[:n]
    else:
        p_dem = p_eb + p_pb

    delta_alpha = np.abs(alpha_final - alpha_req)

    denom = max(1.0, float(np.nanmax(np.abs(p_dem))))

    score_critique = (
        delta_alpha
        + 2.0 * np.maximum(p_unserved, 0.0) / denom
        + 0.5 * correction.astype(float)
        + 0.5 * (soc_pb[:-1] <= SOC_MIN + 0.03).astype(float)
    )

    df_critiques = pd.DataFrame(
        {
            "Temps": temps,
            "P_dem (W)": p_dem,
            "SOC_EB": soc_eb[:-1],
            "SOC_PB": soc_pb[:-1],
            "Alpha demandé": alpha_req,
            "Alpha appliqué": alpha_final,
            "Correction alpha": delta_alpha,
            "P_EB (W)": p_eb,
            "P_PB (W)": p_pb,
            "Déficit instantané (W)": p_unserved,
            "Filtre corrigé": correction,
            "Score critique": score_critique,
        }
    )

    df_critiques = (
        df_critiques
        .sort_values("Score critique", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )

    st.write(
        "Ces instants correspondent aux moments où le filtre de sécurité intervient, "
        "où l’alpha demandé diffère fortement de l’alpha appliqué, ou encore où "
        "une partie de la demande n’est pas servie."
    )

    st.dataframe(
        df_critiques.round(4),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# 6. Déficit énergétique cumulé avec interprétation XAI
# ============================================================

st.subheader("6. Déficit énergétique cumulé")


def calculer_deficit_energetique_total(traj):
    """
    Calcule le déficit énergétique total en Wh.

    Le déficit énergétique correspond à la puissance demandée qui n’a pas pu
    être couverte par le HESS après application des contraintes physiques
    et du filtre de sécurité.
    """
    p_unserved = get_array(traj, "P_unserved")

    if len(p_unserved) == 0:
        return 0.0

    return float(
        np.sum(
            np.maximum(
                p_unserved,
                0.0,
            )
        )
        / 3600.0
    )


def generer_explication_deficit_energetique(
    nom_meilleur,
    deficit_min,
    nom_pire,
    deficit_max,
    deficit_reference,
):
    """
    Génère automatiquement une explication XAI du déficit énergétique cumulé.

    L’explication n’est pas écrite manuellement pour une stratégie précise.
    Elle est produite à partir des indicateurs calculés après simulation.
    """
    texte = (
        "Ce graphique présente le **déficit énergétique cumulé** associé à "
        "chaque stratégie EMS. Ce déficit représente la part de la demande de "
        "puissance qui n’a pas pu être couverte par le système HESS après "
        "application des contraintes physiques et du filtre de sécurité. "
    )

    if deficit_min < 1.0:
        texte += (
            f"La stratégie **{nom_meilleur}** couvre presque toute la demande "
            "du véhicule sur le cycle, avec un déficit énergétique quasi nul. "
        )

    elif deficit_min < 500.0:
        texte += (
            f"La stratégie **{nom_meilleur}** présente le déficit énergétique "
            f"le plus faible parmi les stratégies simulées, avec environ "
            f"**{deficit_min:.1f} Wh** non couverts. "
        )

    else:
        texte += (
            f"La stratégie **{nom_meilleur}** est la moins pénalisée selon cet "
            f"indicateur, mais elle conserve tout de même un déficit énergétique "
            f"notable d’environ **{deficit_min:.1f} Wh**. "
        )

    if deficit_max > 1000.0:
        texte += (
            f"À l’inverse, la stratégie **{nom_pire}** présente un déficit élevé "
            f"d’environ **{deficit_max:.1f} Wh**. Cela indique qu’une partie "
            "importante de la demande du véhicule n’a pas été satisfaite. "
            "Ce comportement peut provenir d’une saturation des limites physiques, "
            "d’une sollicitation excessive d’une batterie ou d’une décision EMS "
            "trop conservative. "
        )

    elif deficit_max > 100.0:
        texte += (
            f"La stratégie **{nom_pire}** présente le déficit le plus important, "
            f"mais celui-ci reste modéré, avec environ **{deficit_max:.1f} Wh** "
            "non couverts sur l’ensemble du cycle. "
        )

    else:
        texte += (
            "Toutes les stratégies présentent un déficit énergétique faible, "
            "ce qui indique que la demande du véhicule est globalement bien "
            "couverte par le système HESS. "
        )

    texte += (
        f"La référence **Alpha-star** présente un déficit d’environ "
        f"**{deficit_reference:.1f} Wh**. Cette référence reste une borne "
        "théorique calculée hors ligne ; elle ne doit pas être interprétée "
        "comme une stratégie directement déployable en temps réel."
    )

    return texte


# ------------------------------------------------------------
# Calcul des déficits énergétiques
# ------------------------------------------------------------

deficit_par_strategie = {}

for nom, traj in resultats.items():
    deficit_par_strategie[nom] = calculer_deficit_energetique_total(
        traj
    )


deficit_alpha_star = calculer_deficit_energetique_total(
    alpha_star_traj
)


# ------------------------------------------------------------
# Génération automatique de l’explication XAI
# ------------------------------------------------------------

if deficit_par_strategie:
    meilleur_code = min(
        deficit_par_strategie,
        key=deficit_par_strategie.get,
    )

    pire_code = max(
        deficit_par_strategie,
        key=deficit_par_strategie.get,
    )

    meilleur_nom = MODEL_DISPLAY_NAMES.get(
        meilleur_code,
        meilleur_code,
    )

    pire_nom = MODEL_DISPLAY_NAMES.get(
        pire_code,
        pire_code,
    )

    deficit_min = deficit_par_strategie[meilleur_code]
    deficit_max = deficit_par_strategie[pire_code]

    with st.container(border=True):
        st.write(
            generer_explication_deficit_energetique(
                meilleur_nom,
                deficit_min,
                pire_nom,
                deficit_max,
                deficit_alpha_star,
            )
        )


st.caption(
    "Lecture XAI : un déficit énergétique faible indique que la stratégie "
    "satisfait correctement la demande du véhicule tout en respectant les limites "
    "physiques du HESS. Un déficit élevé signifie que le filtre de sécurité ou "
    "les contraintes physiques empêchent la stratégie de fournir toute la puissance "
    "demandée."
)


# ------------------------------------------------------------
# Graphique du déficit énergétique cumulé
# ------------------------------------------------------------

fig_unserved, ax_unserved = plt.subplots(
    figsize=(11, 5)
)

for nom, traj in resultats.items():
    p_unserved = get_array(
        traj,
        "P_unserved",
    )

    energie_cumulee = (
        np.cumsum(
            np.maximum(
                p_unserved,
                0.0,
            )
        )
        / 3600.0
    )

    ax_unserved.plot(
        energie_cumulee,
        label=MODEL_DISPLAY_NAMES.get(nom, nom),
    )


p_unserved_star = get_array(
    alpha_star_traj,
    "P_unserved",
)

energie_cumulee_star = (
    np.cumsum(
        np.maximum(
            p_unserved_star,
            0.0,
        )
    )
    / 3600.0
)

ax_unserved.plot(
    energie_cumulee_star,
    label="Alpha-star (référence théorique)",
    linestyle="--",
)

ax_unserved.set_xlabel(
    "Pas de simulation"
)

ax_unserved.set_ylabel(
    "Déficit énergétique cumulé (Wh)"
)

ax_unserved.set_title(
    "Déficit énergétique cumulé sur le cycle"
)

ax_unserved.grid(
    True,
    alpha=0.3,
)

ax_unserved.legend(
    fontsize=8,
)

st.pyplot(fig_unserved)

plt.close(fig_unserved)


# ------------------------------------------------------------
# Tableau XAI associé au déficit énergétique
# ------------------------------------------------------------

st.markdown("**Résumé numérique du déficit énergétique**")

tableau_deficit = []

for nom, deficit in deficit_par_strategie.items():
    tableau_deficit.append(
        {
            "Stratégie": MODEL_DISPLAY_NAMES.get(nom, nom),
            "Déficit énergétique cumulé (Wh)": deficit,
            "Type": "Stratégie EMS simulée",
        }
    )

tableau_deficit.append(
    {
        "Stratégie": "Alpha-star",
        "Déficit énergétique cumulé (Wh)": deficit_alpha_star,
        "Type": "Référence théorique",
    }
)

df_deficit = pd.DataFrame(
    tableau_deficit
).sort_values(
    "Déficit énergétique cumulé (Wh)",
    ascending=True,
)

st.dataframe(
    df_deficit.round(3),
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# 7. Corrections cumulées du filtre de sécurité
# ============================================================

st.subheader("7. Corrections cumulées du filtre de sécurité")

st.write(
    "Une stratégie robuste doit proposer naturellement des décisions proches "
    "des contraintes physiques. Si le nombre de corrections augmente rapidement, "
    "cela signifie que le filtre de sécurité doit souvent modifier la décision "
    "initiale du modèle."
)

fig_corr, ax_corr = plt.subplots(figsize=(11, 5))

for nom, traj in resultats.items():
    correction = get_array(
        traj,
        "correction_applied",
        default=False,
        dtype=bool,
    )

    ax_corr.plot(
        np.cumsum(correction.astype(int)),
        label=MODEL_DISPLAY_NAMES.get(nom, nom),
    )

ax_corr.set_xlabel("Pas de simulation")
ax_corr.set_ylabel("Nombre cumulé de corrections")
ax_corr.set_title("Interventions cumulées du filtre de sécurité")
ax_corr.grid(True, alpha=0.3)
ax_corr.legend(fontsize=8)

st.pyplot(fig_corr)
plt.close(fig_corr)


# ============================================================
# 8. Répartition énergétique EB/PB
# ============================================================

st.subheader("8. Répartition énergétique entre EB et PB")

st.write(
    "Pour une stratégie cohérente avec la priorité EB, la batterie d’énergie "
    "doit fournir la majeure partie de l’énergie. La PB doit intervenir surtout "
    "en assistance, lors des pics de puissance ou lorsque l’EB atteint une limite."
)

df_energie = tableau[
    [
        "Stratégie",
        "Stratégie courte",
        "Énergie EB délivrée (Wh)",
        "Énergie PB délivrée (Wh)",
        "Part énergie PB (%)",
        "Déployable",
    ]
].copy()

df_energie = df_energie.sort_values(
    "Énergie EB délivrée (Wh)",
    ascending=True,
)

fig_energy, ax_energy = plt.subplots(figsize=(10, 6))

y = np.arange(len(df_energie))

ax_energy.barh(
    y,
    df_energie["Énergie EB délivrée (Wh)"],
    label="EB",
)

ax_energy.barh(
    y,
    df_energie["Énergie PB délivrée (Wh)"],
    left=df_energie["Énergie EB délivrée (Wh)"],
    label="PB",
)

ax_energy.set_yticks(y)
ax_energy.set_yticklabels(df_energie["Stratégie courte"])
ax_energy.set_xlabel("Énergie positive délivrée (Wh)")
ax_energy.set_title("Répartition de l’énergie délivrée par EB et PB")
ax_energy.grid(True, axis="x", alpha=0.3)
ax_energy.legend(fontsize=8)

st.pyplot(fig_energy)
plt.close(fig_energy)

st.dataframe(
    df_energie.round(3),
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# 9. Classement visuel selon les indicateurs clés
# ============================================================

st.subheader("9. Classements visuels")

df_deployables = tableau[
    tableau["Déployable"] == "Oui"
].copy()

tab1, tab2, tab3 = st.tabs(
    [
        "Score XAI",
        "Déficit énergétique",
        "Préservation PB",
    ]
)

with tab1:
    afficher_graphique_barres_horizontales(
        df_deployables,
        "Score global XAI",
        "Score global XAI des stratégies déployables",
        "Score / 100",
    )

with tab2:
    afficher_graphique_barres_horizontales(
        df_deployables,
        "Déficit énergétique (Wh)",
        "Déficit énergétique par stratégie",
        "Wh",
    )

with tab3:
    afficher_graphique_barres_horizontales(
        df_deployables,
        "SOC PB final",
        "SOC final de la batterie de puissance",
        "SOC final",
    )


# ============================================================
# 10. Conclusion comparative
# ============================================================

st.subheader("10. Conclusion comparative")

st.write(
    "La comparaison montre qu’une stratégie ne doit pas être jugée uniquement "
    "sur le SOC final. Un SOC élevé peut parfois signifier que la batterie a été "
    "préservée au prix d’une demande non satisfaite. Le classement XAI proposé "
    "combine donc plusieurs critères physiques : demande servie, respect des SOC, "
    "corrections du filtre, équilibre EB/PB et préservation de la PB."
)

st.write(
    "Les stratégies les plus intéressantes sont celles qui conservent une bonne "
    "faisabilité, limitent l’déficit énergétique, sollicitent la PB uniquement "
    "lorsque cela est nécessaire et restent interprétables pour une future "
    "analyse expérimentale."
)


# ============================================================
# Navigation
# ============================================================

st.divider()

col_nav1, col_nav2 = st.columns(2)

with col_nav1:
    if st.button(
        "Consulter l’explicabilité détaillée",
        type="primary",
    ):
        st.switch_page("pages/7_Explicabilite.py")

with col_nav2:
    if st.button("Retour à l’évolution du SOC"):
        st.switch_page("pages/4_Evolution_SOC.py")