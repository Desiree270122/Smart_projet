"""
core/format.py — Écriture des nombres selon la langue de l'interface :
12 601,6 en français (virgule décimale, espace fine entre les milliers),
12,601.6 en anglais. Les graphiques Plotly font de même avec separateurs_plotly().
"""

import math

from core.i18n import langue


def separateurs_plotly() -> str:
    """Valeur de layout.separators : décimale, puis séparateur des milliers."""
    return ".," if langue() == "en" else ", "


def nombre(x, decimales=1, signe=False):
    """Nombre formaté selon la langue ; « — » si la valeur manque."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return str(x)
    if math.isnan(x):
        return "—"
    if math.isinf(x):
        return "∞" if x > 0 else "-∞"
    texte = f"{x:{'+' if signe else ''},.{decimales}f}"
    if langue() == "en":
        return texte
    return texte.replace(",", " ").replace(".", ",")
