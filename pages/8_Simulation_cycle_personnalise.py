
import os
import sys
import time
from pathlib import Path

DOSSIER_PROJET = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DOSSIER_PROJET))

import streamlit as st
import torch

from ems_core import (
    simuler_toutes_strategies,
    set_alpha_grid_step,
    MODEL_DISPLAY_NAMES,
    load_mlp_simple,
    load_mlp_neurosymbolic,
    load_lstm_seul,
    load_lstm_neurosymbolic,
    load_gnn_simple,
)


# ============================================================
# Configuration de la page
# ============================================================

st.set_page_config(
    page_title="2SMART — Simulation globale",
    layout="wide",
)

try:
    torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
except Exception:
    pass


# ============================================================
# Constantes de page
# ============================================================

STRATEGIES_ATTENDUES_COMPLETES = [
    "EMS_power_limitation",
    "EMS_fuzzy_logic",
    "EMS_MLP",
    "EMS_MLP_neurosymbolic",
    "EMS_LSTM",
    "EMS_LSTM_neurosymbolic",
    "EMS_GNN",
]

MODELES_ENTRAINES_ATTENDUS = [
    "EMS_MLP",
    "EMS_MLP_neurosymbolic",
    "EMS_LSTM",
    "EMS_LSTM_neurosymbolic",
    "EMS_GNN",
]

CLES_RESULTATS_A_SUPPRIMER = [
    "resultats_simulation",
    "avertissements_simulation",
    "signature_simulation",
    "modele_actif",
    "alpha_star_reference",
]


def nom_affiche(code_modele: str) -> str:
    """Retourne le nom lisible d'une stratégie."""
    return MODEL_DISPLAY_NAMES.get(code_modele, code_modele)


def supprimer_anciens_resultats():
    """Supprime les anciens résultats de simulation et de comparaison."""
    for cle in CLES_RESULTATS_A_SUPPRIMER:
        st.session_state.pop(cle, None)


# ============================================================
# Titre
# ============================================================

st.title("Simulation — cycle personnalisé")

st.warning(
    "⏱️ Cette page **relance une simulation complète** sur le cycle préparé. "
    "C'est **long** (plusieurs minutes sur le serveur gratuit) car chaque stratégie "
    "est simulée pas à pas. Pour la démonstration et l'analyse habituelle, utilise "
    "plutôt les pages **Comparaison / Résultats / Explicabilité**, qui affichent "
    "instantanément les résultats de référence précalculés. Cette page ne sert "
    "que si tu veux tester **ton propre cycle**."
)


# ============================================================
# Vérification de la présence d'un cycle préparé
# ============================================================

if "cycle_pret" not in st.session_state:
    st.warning("Aucun cycle de conduite n’a encore été préparé.")

    if st.button("Ouvrir la page de préparation des données"):
        st.switch_page("pages/2_Preparation_donnees.py")

    st.stop()


df = st.session_state["cycle_pret"].copy()


# ============================================================
# SOC initiaux imposés à 100 %
# ============================================================

soc_eb0 = 1.0
soc_pb0 = 1.0

st.session_state["soc_eb0"] = soc_eb0
st.session_state["soc_pb0"] = soc_pb0

if "SOC_EB" in df.columns and len(df) > 0:
    df.loc[df.index[0], "SOC_EB"] = soc_eb0

if "SOC_PB" in df.columns and len(df) > 0:
    df.loc[df.index[0], "SOC_PB"] = soc_pb0


nb_points = len(df)
nb_points_affiche = f"{nb_points:,}".replace(",", " ")


# ============================================================
# Introduction
# ============================================================

st.write(
    f"Cycle de conduite prêt : **{nb_points_affiche} échantillons** "
    "disponibles pour la simulation. Les SOC initiaux sont fixés à "
    "**100 %** pour les deux batteries. Les stratégies EMS déterministes "
    "et les modèles entraînés sont prêts à être utilisés pour comparer "
    "la répartition de puissance entre la batterie d’énergie EB et la "
    "batterie de puissance PB."
)

info1, info2, info3 = st.columns(3)

with info1:
    st.metric("Nombre d’échantillons", nb_points_affiche)

with info2:
    st.metric("SOC initial EB", "100 %")

with info3:
    st.metric("SOC initial PB", "100 %")


# ============================================================
# 1. Initialisation des stratégies EMS
# ============================================================

st.subheader("1. Initialisation des stratégies EMS")

st.write(
    "Cette étape vérifie la disponibilité des stratégies EMS utilisées "
    "pendant la simulation globale."
)

col_det, col_ml = st.columns(2)

with col_det:
    with st.container(border=True):
        st.markdown("**Stratégies déterministes disponibles**")
        st.write("- EMS power limitation")
        st.write("- EMS fuzzy logic")

with col_ml:
    with st.container(border=True):
        st.markdown("**Modèles entraînés à charger**")
        st.write("- EMS MLP")
        st.write("- EMS MLP neurosymbolique")
        st.write("- EMS LSTM")
        st.write("- EMS LSTM neurosymbolique")
        st.write("- EMS GNN")


charger_gnn = st.checkbox(
    "Inclure le modèle EMS GNN",
    value=False,
    help=(
        "Le modèle GNN nécessite torch_geometric, dont l'import est lent "
        "(plusieurs secondes) et ralentit fortement le démarrage de l'app. "
        "Laisse cette option décochée pour un chargement rapide (6 stratégies) ; "
        "coche-la seulement quand tu veux la 7ᵉ stratégie GNN."
    ),
)

strategies_attendues = (
    STRATEGIES_ATTENDUES_COMPLETES
    if charger_gnn
    else [s for s in STRATEGIES_ATTENDUES_COMPLETES if s != "EMS_GNN"]
)


@st.cache_resource(show_spinner=False)
def _charger_modeles_deterministes():
    """
    Charge les 4 modèles neuronaux déterministes (MLP, MLP-NS, LSTM, LSTM-NS)
    et conserve séparément les éventuelles erreurs de chargement.

    Ce cache ne dépend PAS de la case GNN : cocher/décocher le GNN ne
    provoque plus le rechargement de ces modèles.
    """
    modeles = {}
    erreurs = {}

    chargeurs = {
        "EMS_MLP": load_mlp_simple,
        "EMS_MLP_neurosymbolic": load_mlp_neurosymbolic,
        "EMS_LSTM": load_lstm_seul,
        "EMS_LSTM_neurosymbolic": load_lstm_neurosymbolic,
    }

    for nom, chargeur in chargeurs.items():
        try:
            modele = chargeur()

            if hasattr(modele, "eval"):
                modele.eval()

            modeles[nom] = modele

        except Exception as exc:
            erreurs[nom] = str(exc)

    return modeles, erreurs


@st.cache_resource(show_spinner=False)
def _charger_gnn():
    """
    Charge le modèle GNN dans un cache séparé. torch_geometric (import lent)
    n'est sollicité que lorsque cette fonction est réellement appelée, donc
    uniquement quand l'utilisateur coche « Inclure le modèle EMS GNN ».
    """
    gnn_model, gnn_scaler = load_gnn_simple()

    if hasattr(gnn_model, "eval"):
        gnn_model.eval()

    return gnn_model, gnn_scaler


def _charger_modeles(charger_gnn: bool):
    """
    Assemble les modèles à partir des deux caches (déterministes + GNN
    optionnel), sans jamais muter les objets mis en cache.
    """
    modeles_det, erreurs_det = _charger_modeles_deterministes()

    modeles = dict(modeles_det)
    erreurs = dict(erreurs_det)
    gnn_scaler = None

    if charger_gnn:
        try:
            gnn_model, gnn_scaler = _charger_gnn()
            modeles["EMS_GNN"] = gnn_model

        except Exception as exc:
            erreurs["EMS_GNN"] = str(exc)

    return modeles, erreurs, gnn_scaler


# ============================================================
# Réinitialisation complète
# ============================================================

with st.expander("Réinitialisation et diagnostic"):
    st.write(
        "Utilise cette action si Streamlit affiche encore d’anciens résultats "
        "ou si une ancienne simulation à 6 stratégies reste en mémoire."
    )

    if st.button("Réinitialiser les résultats et les caches"):
        supprimer_anciens_resultats()
        st.session_state.pop("modeles_charges_cache", None)
        st.session_state.pop("gnn_scaler", None)

        try:
            st.cache_resource.clear()
        except Exception:
            pass

        try:
            st.cache_data.clear()
        except Exception:
            pass

        st.success("Résultats et caches réinitialisés.")
        st.rerun()


with st.spinner("Initialisation des modèles entraînés..."):
    modeles_charges, erreurs_chargement, gnn_scaler = _charger_modeles(charger_gnn)

st.session_state["modeles_charges_cache"] = modeles_charges

if gnn_scaler is not None:
    st.session_state["gnn_scaler"] = gnn_scaler


# ============================================================
# Affichage de l'état de chargement
# ============================================================

if modeles_charges:
    st.success(
        "Initialisation réussie : les stratégies déterministes et les modèles "
        "chargés sont prêts pour la simulation."
    )
else:
    st.error(
        "Aucun modèle entraîné n’a pu être chargé. Les stratégies déterministes "
        "peuvent rester disponibles selon la configuration de ems_core.py."
    )

col_ok, col_ko = st.columns(2)

with col_ok:
    with st.container(border=True):
        st.markdown("**Modèles chargés avec succès**")

        if modeles_charges:
            for nom in MODELES_ENTRAINES_ATTENDUS:
                if nom in modeles_charges:
                    st.write(f"- {nom_affiche(nom)}")
        else:
            st.write("Aucun modèle entraîné chargé.")

with col_ko:
    with st.container(border=True):
        st.markdown("**Modèles non disponibles**")

        modeles_non_disponibles = [
            nom
            for nom in MODELES_ENTRAINES_ATTENDUS
            if nom not in modeles_charges
        ]

        if modeles_non_disponibles:
            for nom in modeles_non_disponibles:
                if nom in erreurs_chargement:
                    st.write(f"- {nom_affiche(nom)} : {erreurs_chargement[nom]}")
                elif nom == "EMS_GNN" and not charger_gnn:
                    st.write(f"- {nom_affiche(nom)} : chargement désactivé")
                else:
                    st.write(f"- {nom_affiche(nom)} : non chargé")
        else:
            st.write("Tous les modèles sélectionnés ont été chargés sans erreur.")


#with st.expander("Notes techniques"):
#    st.write(
#        "Les deux stratégies déterministes sont directement disponibles : "
#        "`EMS_power_limitation` et `EMS_fuzzy_logic`."
#    )

#    st.write(
 #       "Les modèles PyTorch sont placés en mode évaluation. La simulation "
  #      "est exécutée avec `torch.inference_mode()` pour accélérer l’inférence."
   # )

  #  st.write(
  #      "Si une stratégie chargée n’apparaît pas dans les résultats finaux, "
  #      "cela signifie qu’elle a échoué pendant la simulation. Dans ce cas, "
#      "la cause apparaît dans les avertissements techniques."
 #   )

  #  st.write("Stratégies attendues :", [nom_affiche(n) for n in strategies_attendues])
  #  st.write("Modèles chargés :", [nom_affiche(n) for n in modeles_charges.keys()])

#    if erreurs_chargement:
#        st.write("Détails des erreurs de chargement :")
#        for nom, err in erreurs_chargement.items():
#            st.code(f"{nom_affiche(nom)} : {err}")


# ============================================================
# 2. Simulation des stratégies EMS
# ============================================================

st.subheader("2. Simulation")

st.write(
    "Cette section lance la simulation globale des stratégies EMS sur "
    "l’ensemble du cycle de conduite. Pour chaque instant, l’application "
    "calcule le coefficient de répartition `alpha`, les puissances `P_EB` "
    "et `P_PB`, puis vérifie le respect des contraintes physiques du système HESS."
)

st.write(
    "Les résultats obtenus seront ensuite utilisés dans les pages suivantes "
    "pour analyser l’évolution des SOC, comparer les modèles et expliquer "
    "les décisions prises par les stratégies EMS."
)

st.info(
    "À chaque nouveau lancement, les anciens résultats sont supprimés afin "
    "d’éviter de conserver une ancienne simulation incomplète."
)


@st.cache_data(show_spinner=False)
def _simuler_en_cache(df, soc_eb0, soc_pb0, signature_modeles, pas_alpha, _modeles_charges):
    """
    Enveloppe mise en cache de la simulation globale.

    Tant que le cycle (df), les SOC initiaux, l'ensemble des modèles chargés
    (signature_modeles) et la précision de grille (pas_alpha) ne changent pas,
    le résultat est réutilisé tel quel : la simulation lourde n'est calculée
    qu'UNE seule fois, puis reste instantanée sur les relances suivantes.

    _modeles_charges est préfixé par « _ » afin que Streamlit ne tente pas de
    le hacher (les modèles PyTorch ne sont pas hachables) ; ce sont les noms
    de modèles (signature_modeles) qui servent de clé de cache à leur place.

    pas_alpha règle la résolution de la grille alpha du filtre physique : plus
    il est grossier, plus la simulation est rapide (impact négligeable sur la
    comparaison des stratégies).

    Le résultat (dictionnaires de trajectoires numpy + liste d'avertissements)
    est sérialisable, donc compatible avec st.cache_data.
    """
    set_alpha_grid_step(pas_alpha)
    with torch.inference_mode():
        return simuler_toutes_strategies(df, soc_eb0, soc_pb0, _modeles_charges)


pas_alpha = st.select_slider(
    "Précision de la grille alpha (vitesse ⇄ précision)",
    options=[0.001, 0.002, 0.005],
    value=0.005,
    format_func=lambda v: {
        0.001: "0.001 — précis (lent, qualité publication)",
        0.002: "0.002 — équilibré",
        0.005: "0.005 — rapide (recommandé pour l'exploration)",
    }[v],
    help=(
        "Pas de balayage du coefficient de répartition alpha dans le filtre "
        "physique, appliqué à chaque pas de temps et chaque stratégie. "
        "0.005 est environ 5× plus rapide que 0.001, avec un impact négligeable "
        "sur la comparaison des stratégies. Repasse à 0.001 pour ton run final."
    ),
)


if st.button(
    "Lancer la simulation globale",
    type="primary",
):
    supprimer_anciens_resultats()

    debut = time.time()

    with st.spinner(
        "Simulation des stratégies EMS en cours... "
        "Tous les modèles disponibles sont conservés."
    ):
        resultats, avertissements = _simuler_en_cache(
            df,
            soc_eb0,
            soc_pb0,
            tuple(sorted(modeles_charges.keys())),
            pas_alpha,
            modeles_charges,
        )

    duree = time.time() - debut

    st.session_state["resultats_simulation"] = resultats
    st.session_state["avertissements_simulation"] = avertissements
    st.session_state["signature_simulation"] = {
        "nb_points": nb_points,
        "soc_eb0": soc_eb0,
        "soc_pb0": soc_pb0,
        "modeles": sorted(list(modeles_charges.keys())),
        "strategies_attendues": strategies_attendues,
    }
    st.session_state["modele_actif"] = "Toutes les stratégies EMS"

    strategies_manquantes = [
        nom
        for nom in strategies_attendues
        if nom not in resultats
    ]

    if strategies_manquantes:
        st.warning(
            f"{len(resultats)} stratégie(s) simulée(s) sur "
            f"{len(strategies_attendues)} attendue(s), en {duree:.2f} secondes. "
            "Stratégie(s) manquante(s) : "
            + ", ".join(nom_affiche(nom) for nom in strategies_manquantes)
        )
    else:
        st.success(
            f"Les {len(strategies_attendues)} stratégies EMS attendues ont été "
            f"simulées sur un cycle de {nb_points_affiche} échantillons "
            f"en {duree:.2f} secondes."
        )


# ============================================================
# 3. Résultats disponibles
# ============================================================

if "resultats_simulation" in st.session_state:
    st.subheader("3. Résultats disponibles")

    resultats = st.session_state["resultats_simulation"]

    strategies_manquantes = [
        nom
        for nom in strategies_attendues
        if nom not in resultats
    ]

    col_res1, col_res2, col_res3 = st.columns(3)

    with col_res1:
        st.metric(
            "Stratégies simulées",
            f"{len(resultats)} / {len(strategies_attendues)}",
        )

    with col_res2:
        st.metric(
            "Échantillons par stratégie",
            nb_points_affiche,
        )

    with col_res3:
        st.metric(
            "Stratégies manquantes",
            len(strategies_manquantes),
        )

    with st.container(border=True):
        st.markdown("**Stratégies disponibles pour l’analyse**")

        for nom in strategies_attendues:
            if nom in resultats:
                st.write(f"- {nom_affiche(nom)}")
            else:
                st.write(f"- {nom_affiche(nom)} — non simulée")

    if strategies_manquantes:
        st.error(
            "Toutes les stratégies n’ont pas été simulées. "
            "Consulte les avertissements techniques ci-dessous pour identifier "
            "la cause exacte."
        )

    avertissements = st.session_state.get(
        "avertissements_simulation",
        [],
    )

    if avertissements:
        with st.expander("Avertissements techniques de simulation", expanded=True):
            for msg in avertissements:
                st.warning(msg)
    else:
        st.success("Aucun avertissement technique n’a été retourné par la simulation.")

    st.divider()

    col_btn1, col_btn2 = st.columns(2)

    with col_btn1:
        if st.button(
            "Afficher les résultats et l’évolution du SOC",
            type="primary",
        ):
            st.switch_page("pages/6_Resultats_et_Analyse.py")

    with col_btn2:
        if st.button("Comparer les stratégies"):
            st.switch_page("pages/5_Comparaison_des_strategies.py")
