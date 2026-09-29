"""
core/style.py — Couleurs partagées des stratégies EMS.

Une stratégie garde la même couleur sur toutes les pages (Comparaison,
Résultats & Analyse, ...) : les pages importent la palette d'ici au lieu de
définir chacune la leur.
"""


# Couleur de chaque stratégie (clés internes, comme dans core.resultats).
PALETTE = {
    "EMS_power_limitation": "#6B7280",
    "EMS_fuzzy_logic": "#C9A227",
    "EMS_MLP": "#6FB1E8",
    "EMS_LSTM": "#1F6FB2",
    "EMS_GNN": "#8E6FD0",
    "EMS_MLP_neurosymbolic": "#E8734A",
    "EMS_LSTM_neurosymbolic": "#2E9E6B",
}

# Stratégie inconnue, ou mise en retrait quand une page en met d'autres en
# avant. Distincte du gris du modèle physique pour ne pas les confondre.
COULEUR_NEUTRE = "#9AA0A6"


def couleur(cle: str) -> str:
    """Couleur d'une stratégie à partir de sa clé interne."""
    return PALETTE.get(cle, COULEUR_NEUTRE)
