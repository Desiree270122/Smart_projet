"""
core/style.py — Couleurs partagées des stratégies EMS.

Une stratégie garde la même couleur sur toutes les pages (Comparaison,
Résultats de simulation, ...) : les pages importent la palette d'ici au lieu de
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


# Grandeurs physiques : une couleur = une signification, sur toutes les pages.
# Validé sur fond sombre : les couleurs d'un même graphique restent distinctes,
# y compris pour un lecteur daltonien. Deux règles d'usage :
# - la palette des stratégies (ci-dessus) et celle des grandeurs ne se mélangent
#   jamais dans un même graphique ;
# - le bleu (demande) et le violet (décision) ne figurent pas ensemble.
COULEUR_DEMANDE = "#5B8DEF"        # demande de puissance, véhicule
COULEUR_EB = "#3DBE7A"             # batterie Énergie (batterie 1)
COULEUR_PB = "#F0913A"             # batterie Puissance (batterie 2)
COULEUR_CONVERTISSEUR = "#D66BC8"
COULEUR_DECISION = "#A98BF5"       # décision de l'EMS (alpha)
COULEUR_VIOLATION = "#EF5350"
COULEUR_REFERENCE = "#C5CAD3"      # référence : règle de l'ontologie, situation moyenne
COULEUR_SECONDAIRE = "#8B93A7"     # repère secondaire (base floue, arêtes, axes)


def couleur(cle: str) -> str:
    """Couleur d'une stratégie à partir de sa clé interne."""
    return PALETTE.get(cle, COULEUR_NEUTRE)


def flux_html(etapes) -> str:
    """Chaîne d'étapes reliées par des flèches, à afficher avec
    st.markdown(..., unsafe_allow_html=True). etapes : [(texte, couleur)]."""
    html = "<div style='display:flex;align-items:center;flex-wrap:wrap;gap:6px;margin:.4rem 0'>"
    for i, (texte, coul) in enumerate(etapes):
        html += (
            f"<span style='border:1px solid {coul};color:{coul};border-radius:9px;"
            f"padding:5px 10px;font-weight:600;font-size:.86rem'>{texte}</span>"
        )
        if i < len(etapes) - 1:
            html += "<span style='color:#94A3B8;font-weight:800'>&#8594;</span>"
    return html + "</div>"
