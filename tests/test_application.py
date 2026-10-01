"""
Tests de l'application : chaque page doit s'exécuter sans erreur.

À lancer avant chaque push :
    python -m pytest tests

Ils tournent aussi automatiquement sur GitHub à chaque push (.github/workflows/tests.yml) :
une croix rouge sur le commit signale une page cassée avant qu'on la découvre en ligne.
"""

import py_compile
import sys
from pathlib import Path

import pytest

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE))

from streamlit.testing.v1 import AppTest  # noqa: E402

PAGES = sorted(
    str(p.relative_to(RACINE)).replace("\\", "/")
    for p in (RACINE / "vues").glob("[0-9]_*.py")
    if p.name != "4_Moteur_Neurosymbolique.py"  # page retirée du menu
)


@pytest.fixture(autouse=True)
def _dossier_projet(monkeypatch):
    monkeypatch.chdir(RACINE)


def _executer(page, session=None, delai=600):
    at = AppTest.from_file(str(RACINE / page), default_timeout=delai)
    for cle, valeur in (session or {}).items():
        at.session_state[cle] = valeur
    at.run()
    return at


def test_syntaxe_de_tous_les_fichiers():
    """Une faute de frappe dans n'importe quel fichier Python fait échouer ce test."""
    fichiers = [RACINE / "Accueil.py", RACINE / "ems_core.py", *RACINE.glob("core/*.py"),
                *RACINE.glob("vues/*.py"), *RACINE.glob("scripts/*.py")]
    for f in fichiers:
        py_compile.compile(str(f), doraise=True)


def test_parametres_des_packs_coherents_avec_les_cellules():
    import ems_core as core  # le chargement vérifie déjà la cohérence (assert)

    assert core.ENERGY_EB_WH == pytest.approx(12601.6, abs=1.0)
    assert core.ENERGY_PB_WH == pytest.approx(3623.4, abs=1.0)
    assert core.CAPACITY_EB_AH == pytest.approx(28.0, abs=0.01)


@pytest.mark.parametrize("page", PAGES)
def test_page_sans_erreur(page):
    at = _executer(page)
    assert not at.exception, [e.value for e in at.exception]


@pytest.mark.parametrize("page", ["vues/5_Comparaison_des_strategies.py", "vues/6_Resultats_et_Analyse.py"])
def test_pages_sur_le_cycle_wltc(page):
    at = _executer(page, {"_cycle_choisi": "wltc"})
    assert not at.exception, [e.value for e in at.exception]


def test_comparaison_sur_la_partie_test():
    at = _executer("vues/5_Comparaison_des_strategies.py")
    at.radio[0].set_value(at.radio[0].options[1]).run()
    assert not at.exception, [e.value for e in at.exception]


def test_routeur_et_selecteur_de_cycle():
    at = AppTest.from_file(str(RACINE / "Accueil.py"), default_timeout=600)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert any(s.label == "Cycle étudié" for s in at.sidebar.selectbox)
