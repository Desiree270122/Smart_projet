"""
core/style.py — Couleurs partagées des stratégies EMS.

Une stratégie garde la même couleur sur toutes les pages (Comparaison,
Résultats & Analyse, ...) : les pages importent la palette d'ici au lieu de
définir chacune la leur.
"""


# Couleur de chaque stratégie (clés internes, comme dans core.resultats).
# Réglée pour le thème sombre : couleurs claires et vives, et luminosités
# alternées pour que les stratégies voisines dans la légende, et les paires
# comparées (MLP / MLP neuro, LSTM / LSTM neuro, modèle physique / GNN),
# restent distinctes y compris pour un lecteur daltonien.
PALETTE = {
    "EMS_power_limitation": "#BCC5D1",
    "EMS_fuzzy_logic": "#E2B000",
    "EMS_MLP": "#6DBFFF",
    "EMS_LSTM": "#ED76B3",
    "EMS_GNN": "#B58BF9",
    "EMS_MLP_neurosymbolic": "#E06B39",
    "EMS_LSTM_neurosymbolic": "#5FD37F",
}

# Stratégie inconnue, ou mise en retrait quand une page en met d'autres en
# avant. Plus sombre que le gris clair du modèle physique pour ne pas les confondre.
COULEUR_NEUTRE = "#7A828E"


def couleur(cle: str) -> str:
    """Couleur d'une stratégie à partir de sa clé interne."""
    return PALETTE.get(cle, COULEUR_NEUTRE)
