"""
core/i18n.py — Langue de l'interface (français ou anglais).

Chaque texte affiché est écrit dans les deux langues, là où il est utilisé :

    tr("Texte en français", "English text")
    tr("La demande vaut {p}.", "Demand is {p}.", p=kw(p_dem))

Deux interfaces partagent ces textes :
- l'application Streamlit, où la langue se choisit dans la barre latérale (par
  défaut, celle du navigateur) ;
- le serveur de calcul de l'application web (dossier api/), qui fixe la langue de
  chaque requête avec `utiliser_langue`.

Pour ajouter une langue : l'ajouter à LANGUES, puis donner sa traduction en
argument nommé, par exemple tr(fr, en, es="…") ; à défaut, l'anglais est utilisé.
"""

from contextlib import contextmanager
from contextvars import ContextVar

LANGUES = {"fr": "Français", "en": "English"}
CLE_LANGUE = "langue"

# Langue imposée pour le traitement en cours (serveur de calcul, tests).
_langue_imposee = ContextVar("langue_imposee", default=None)


@contextmanager
def utiliser_langue(code: str):
    """Fixe la langue des textes pendant le bloc `with`, hors de Streamlit."""
    jeton = _langue_imposee.set(code if code in LANGUES else "fr")
    try:
        yield
    finally:
        _langue_imposee.reset(jeton)


def langue_du_navigateur() -> str:
    """« fr » si le navigateur est en français (ou inconnu), « en » sinon."""
    try:
        import streamlit as st

        locale = st.context.locale or ""
    except Exception:  # noqa: BLE001 - hors serveur Streamlit
        locale = ""
    return "fr" if not locale or locale.lower().startswith("fr") else "en"


def langue() -> str:
    imposee = _langue_imposee.get()
    if imposee is not None:
        return imposee
    try:
        import streamlit as st

        return st.session_state.get(CLE_LANGUE) or "fr"
    except Exception:  # noqa: BLE001 - hors serveur Streamlit
        return "fr"


def tr(fr: str, en: str, **valeurs) -> str:
    """Texte dans la langue choisie ; les {champs} sont remplis par `valeurs`."""
    texte = en if langue() == "en" else fr
    return texte.format(**valeurs) if valeurs else texte


def lib(paire) -> str:
    """Élément d'une paire (français, anglais) dans la langue choisie."""
    if isinstance(paire, str):
        return paire
    return paire[1] if langue() == "en" else paire[0]
