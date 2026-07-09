
import sys
from pathlib import Path

DOSSIER_PROJET = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DOSSIER_PROJET))

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from ems_core import MODEL_DISPLAY_NAMES


st.set_page_config(
    page_title="Analyse instantanée",
    layout="wide",
)

st.title("Analyse instantanée")


# ============================================================
# Vérification de la présence des résultats de simulation
# ============================================================

if (
    "resultats_simulation" not in st.session_state
    or "cycle_pret" not in st.session_state
):
    st.warning(
        "Une simulation doit être réalisée avant de consulter cette page."
    )

    if st.button("Ouvrir la page de simulation globale"):
        st.switch_page("pages/3_Simulation_globale.py")

    st.stop()


resultats = st.session_state["resultats_simulation"]
df = st.session_state["cycle_pret"]


if not resultats:
    st.warning(
        "Aucune stratégie n'a produit de résultat exploitable."
    )
    st.stop()


# ============================================================
# Sélection de l'instant à analyser
# ============================================================

nombre_points_max = min(
    len(traj["P_EB"])
    for traj in resultats.values()
)

instant_choisi = st.slider(
    "Instant du cycle à analyser",
    0,
    nombre_points_max - 1,
    nombre_points_max // 2,
)


ligne = df.iloc[instant_choisi]


st.write(
    f"**Temps : {float(ligne['time']):.0f} s** — "
    f"Puissance demandée : {float(ligne['hasPower']) / 1000:.2f} kW"
)


# ============================================================
# Courbe de la puissance demandée sur tout le cycle
# ============================================================

st.subheader("Puissance demandée sur le cycle")

st.caption(
    "La valeur affichée ci-dessus est celle de l'instant sélectionné (repère "
    "rouge) -- elle change avec le curseur. Ce graphique montre comment la "
    "puissance demandée évolue sur l'ensemble du cycle, pour situer cet "
    "instant dans son contexte."
)

fig_puissance, ax_puissance = plt.subplots(figsize=(11, 3.5))
ax_puissance.plot(
    df["time"].to_numpy()[:nombre_points_max],
    df["hasPower"].to_numpy()[:nombre_points_max] / 1000.0,
    color="#5B8DEF", linewidth=1,
)
ax_puissance.axvline(
    float(ligne["time"]), color="#E5484D", linestyle="--", linewidth=1.5,
    label=f"Instant sélectionné ({float(ligne['time']):.0f} s)",
)
ax_puissance.scatter(
    [float(ligne["time"])], [float(ligne["hasPower"]) / 1000.0],
    color="#E5484D", zorder=5, s=40,
)
ax_puissance.set_xlabel("Temps (s)")
ax_puissance.set_ylabel("Puissance demandée (kW)")
ax_puissance.legend(loc="upper right", fontsize=8)
ax_puissance.grid(True, alpha=0.3)
plt.tight_layout()
st.pyplot(fig_puissance)
plt.close(fig_puissance)


# ============================================================
# Construction du tableau comparatif
# ============================================================

st.subheader("Détail par stratégie à cet instant")

lignes_tableau = []

for nom, traj in resultats.items():
    lignes_tableau.append(
        {
            "Stratégie": MODEL_DISPLAY_NAMES.get(nom, nom),
            "Alpha appliqué": float(
                traj["alpha_final"][instant_choisi]
            ),
            "P_EB (kW)": float(
                traj["P_EB"][instant_choisi]
            ) / 1000,
            "P_PB (kW)": float(
                traj["P_PB"][instant_choisi]
            ) / 1000,
            "I_EB (A)": float(
                traj["I_EB"][instant_choisi]
            ),
            "I_PB (A)": float(
                traj["I_PB"][instant_choisi]
            ),
            "SOC_EB": float(
                traj["SOC_EB"][instant_choisi]
            ),
            "SOC_PB": float(
                traj["SOC_PB"][instant_choisi]
            ),
            "Correction appliquée": (
                "Oui"
                if bool(
                    traj["correction_applied"][instant_choisi]
                )
                else "Non"
            ),
        }
    )


st.dataframe(
    pd.DataFrame(lignes_tableau).round(4),
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# Navigation vers la comparaison des stratégies
# ============================================================

st.divider()

if st.button(
    "Consulter la comparaison et l'optimisation",
    type="primary",
):
    st.switch_page(
        "pages/6_Comparaison_optimisation.py"
    )