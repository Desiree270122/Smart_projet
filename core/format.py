"""
core/format.py — Écriture des nombres à la française dans l'interface :
virgule décimale et espace fine insécable entre les milliers (12 601,6).

Les graphiques Plotly font de même avec layout.separators = SEPARATEURS_PLOTLY.
"""

import math

SEPARATEURS_PLOTLY = ", "   # décimale, puis séparateur des milliers


def nombre(x, decimales=1, signe=False):
    """Nombre formaté à la française ; « — » si la valeur manque."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return str(x)
    if math.isnan(x):
        return "—"
    if math.isinf(x):
        return "∞" if x > 0 else "-∞"
    texte = f"{x:{'+' if signe else ''},.{decimales}f}"
    return texte.replace(",", " ").replace(".", ",")
