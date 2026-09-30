"""
core/instant.py — Choix de l'instant analysé, commun à toutes les pages.

« Analyse instantanée », « Pourquoi cette décision ? » et « Base de
connaissances » passent toutes par ici : même curseur (en secondes du cycle),
même conversion temps -> index. Sans cela, une page choisissait un temps (qui
commence à 1 s) et les autres un index (qui commence à 0) : le « même » instant
affichait des valeurs décalées d'un pas.

L'instant choisi est aussi conservé d'une page à l'autre.
"""

import numpy as np
import streamlit as st


_CLE_INSTANT = "_instant_t"      # survit au changement de page
_CLE_CURSEUR = "curseur_instant"  # clé du widget, nettoyée par Streamlit hors page


def temps_du_cycle(df, n):
    """Axe des temps des n premiers points : la colonne « time » si elle existe,
    sinon le numéro d'échantillon."""
    if "time" in df.columns:
        return df["time"].to_numpy(dtype=float)[:n]
    return np.arange(n, dtype=float)


def index_de_temps(temps, t):
    """Index du point du cycle le plus proche du temps t."""
    return int(np.abs(temps - t).argmin())


def _memoriser():
    st.session_state[_CLE_INSTANT] = st.session_state[_CLE_CURSEUR]


def choisir_instant(df, n, conteneur=st):
    """Affiche le curseur d'instant et retourne (index, temps en s).

    n : nombre de points exploitables (min des longueurs du cycle et des
    trajectoires). Le curseur reprend l'instant choisi sur une autre page,
    ramené dans les bornes du cycle courant.
    """
    temps = temps_du_cycle(df, n)
    t_min, t_max = int(temps[0]), int(temps[-1])
    t_defaut = int(temps[len(temps) // 2])

    st.session_state[_CLE_CURSEUR] = int(
        np.clip(st.session_state.get(_CLE_INSTANT, t_defaut), t_min, t_max)
    )
    t_choisi = conteneur.slider(
        "Instant du cycle (s)" if "time" in df.columns else "Échantillon",
        t_min,
        t_max,
        key=_CLE_CURSEUR,
        on_change=_memoriser,
    )
    index = index_de_temps(temps, t_choisi)
    return index, float(temps[index])
