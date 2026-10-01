"""
Tests de l'application : chaque page doit s'exécuter sans erreur, en français
et en anglais.

À lancer avant chaque push :
    python -m pytest tests

Ils tournent aussi automatiquement sur GitHub à chaque push (.github/workflows/tests.yml) :
une croix rouge sur le commit signale une page cassée avant qu'on la découvre en ligne.
"""

import ast
import io
import py_compile
import string
import sys
from pathlib import Path

import pandas as pd
import pytest

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE))

from streamlit.testing.v1 import AppTest  # noqa: E402

PAGES = sorted(str(p.relative_to(RACINE)).replace("\\", "/") for p in (RACINE / "vues").glob("[0-9]_*.py"))
LANGUES = ["fr", "en"]
SOURCES = [RACINE / "Accueil.py", *RACINE.glob("core/*.py"), *RACINE.glob("vues/*.py")]


@pytest.fixture(autouse=True)
def _dossier_projet(monkeypatch):
    monkeypatch.chdir(RACINE)


def _executer(page, session=None, delai=600):
    at = AppTest.from_file(str(RACINE / page), default_timeout=delai)
    for cle, valeur in (session or {}).items():
        at.session_state[cle] = valeur
    at.run()
    return at


def _erreurs(at):
    return [e.value for e in at.exception]


def test_syntaxe_de_tous_les_fichiers():
    """Une faute de frappe dans n'importe quel fichier Python fait échouer ce test."""
    for f in [*SOURCES, RACINE / "ems_core.py", *RACINE.glob("scripts/*.py")]:
        py_compile.compile(str(f), doraise=True)


def test_parametres_des_packs_coherents_avec_les_cellules():
    import ems_core as core  # le chargement vérifie déjà la cohérence (assert)

    assert core.ENERGY_EB_WH == pytest.approx(12601.6, abs=1.0)
    assert core.ENERGY_PB_WH == pytest.approx(3623.4, abs=1.0)
    assert core.CAPACITY_EB_AH == pytest.approx(28.0, abs=0.01)


def test_chaque_texte_existe_dans_les_deux_langues():
    """Chaque tr(français, anglais) doit avoir ses deux textes, avec les mêmes {champs}."""
    def champs(texte):
        return {nom for _, nom, _, _ in string.Formatter().parse(texte) if nom}

    defauts = []
    for fichier in SOURCES:
        for noeud in ast.walk(ast.parse(fichier.read_text(encoding="utf-8"))):
            if not (isinstance(noeud, ast.Call) and getattr(noeud.func, "id", "") == "tr"):
                continue
            textes = [a.value for a in noeud.args if isinstance(a, ast.Constant) and isinstance(a.value, str)]
            ou = f"{fichier.name}:{noeud.lineno}"
            if len(noeud.args) != 2 or len(textes) != 2 or not all(t.strip() for t in textes):
                defauts.append(f"{ou} : il faut un texte français et un texte anglais")
            elif champs(textes[0]) != champs(textes[1]):
                defauts.append(f"{ou} : champs différents entre les deux langues {champs(textes[0]) ^ champs(textes[1])}")
            elif noeud.keywords and champs(textes[0]) != {k.arg for k in noeud.keywords}:
                defauts.append(f"{ou} : valeurs fournies différentes des champs du texte")
    assert not defauts, "\n".join(defauts)


@pytest.mark.parametrize("langue", LANGUES)
@pytest.mark.parametrize("page", PAGES)
def test_page_sans_erreur(page, langue):
    at = _executer(page, {"langue": langue})
    assert not at.exception, _erreurs(at)


@pytest.mark.parametrize("page", [
    "vues/1_Accueil.py", "vues/5_Comparaison_des_strategies.py", "vues/6_Resultats_et_Analyse.py", "vues/7_Explicabilite.py",
])
def test_pages_sur_le_cycle_wltc(page):
    at = _executer(page, {"_cycle_choisi": "wltc", "langue": "en"})
    assert not at.exception, _erreurs(at)


def test_comparaison_conclusion_avec_et_sans_explicabilite():
    at = _executer("vues/5_Comparaison_des_strategies.py")
    assert at.success, "la conclusion générale doit s'afficher"
    at.checkbox[0].uncheck().run()
    assert not at.exception, _erreurs(at)
    assert at.success, "la conclusion générale doit s'afficher"


@pytest.mark.parametrize("langue", LANGUES)
def test_bulles_d_explication_des_courbes(langue, monkeypatch):
    """Chaque instant du cycle a sa phrase d'explication, pour chaque stratégie."""
    from core import format as format_nombres
    from core import i18n, lecture
    from core.resultats import charger_reference

    monkeypatch.setattr(i18n, "langue", lambda: langue)
    monkeypatch.setattr(format_nombres, "langue", lambda: langue)
    donnees = charger_reference()
    p_dem = donnees["cycle_df"]["hasPower"].to_numpy(dtype=float)
    for nom, traj in donnees["resultats"].items():
        n = min(len(p_dem), len(traj["P_EB"]))
        textes = lecture.lire_repartition(p_dem, traj, n)
        assert len(textes) == n and all(textes), nom
        assert len(lecture.lire_soc(traj)) == len(traj["SOC_EB"]), nom
    # Une stratégie qui ne fournit pas toute la demande doit le dire dans ses bulles.
    assert any("⚠" in t for t in lecture.lire_repartition(p_dem, donnees["resultats"]["EMS_MLP"], n))


@pytest.mark.parametrize("strategie", ["EMS_MLP_neurosymbolic", "EMS_LSTM_neurosymbolic", "EMS_fuzzy_logic"])
def test_explication_a_un_instant_de_forte_traction(strategie):
    at = _executer("vues/7_Explicabilite.py", {"_instant_t": 1249, "strategie_explication": strategie})
    assert not at.exception, _erreurs(at)
    assert any(m.value.startswith("1.") for m in at.markdown), "les raisons en clair doivent s'afficher"


def test_conclusion_tous_criteres():
    """La conclusion ne retient que des stratégies qui fournissent toute la demande sur les deux cycles."""
    from core import verdict
    from core.resultats import CYCLES_REFERENCE, calculer_metriques, charger_reference

    evaluations = {}
    for cle, (_, chemin) in CYCLES_REFERENCE.items():
        donnees = charger_reference(chemin)
        evaluations[cle] = verdict.completer(calculer_metriques(donnees), donnees.get("coherence"))
    bilan = verdict.conclure(evaluations, verdict.criteres(True))

    assert bilan["communes"], "au moins une stratégie doit fournir la demande sur les deux cycles"
    for metriques in evaluations.values():
        for nom in bilan["communes"]:
            assert metriques[nom]["energie_non_servie_wh"] < 1.0
    assert verdict.meilleures(bilan["score"])[0] in bilan["communes"]
    assert sum(bilan["en_tete"].values()) == pytest.approx(1.0)


@pytest.mark.parametrize("langue", LANGUES)
def test_preparation_d_un_cycle(langue, monkeypatch):
    """Import d'un fichier de vitesse, préparation, aperçu : les résultats affichés ailleurs ne changent pas."""
    import streamlit as st

    contenu = pd.read_csv(RACINE / "data" / "wltc.csv")[["time", "speed"]].iloc[:600].to_csv(index=False).encode()

    def faux_fichier(*args, **kwargs):
        f = io.BytesIO(contenu)
        f.name = "wltc.csv"
        return f

    monkeypatch.setattr(st, "file_uploader", faux_fichier)
    at = _executer("vues/2_Preparation_donnees.py", {"langue": langue})
    assert not at.exception, _erreurs(at)
    at.button[0].click().run()
    assert not at.exception, _erreurs(at)
    assert "cycle_prepare" in at.session_state
    assert "resultats_simulation" not in at.session_state


def test_lancer_une_simulation_des_deux_references():
    cycle = pd.read_csv(RACINE / "data" / "wltc.csv").iloc[:600]
    at = _executer("vues/8_Simulation_cycle_personnalise.py", {"cycle_prepare": cycle}, delai=900)
    assert not at.exception, _erreurs(at)
    for case in at.checkbox:
        if not case.disabled and case.key and case.key.startswith("sim_"):
            case.uncheck()
    at.run()
    next(b for b in at.button if "Lancer" in b.label).click().run()
    assert not at.exception, _erreurs(at)
    assert at.session_state["_cycle_choisi"] == "personnalise"
    assert set(at.session_state["resultats_simulation"]) == {"EMS_power_limitation", "EMS_fuzzy_logic"}


def test_routeur_langue_dans_la_barre_laterale():
    at = AppTest.from_file(str(RACINE / "Accueil.py"), default_timeout=600)
    at.run()
    assert not at.exception, _erreurs(at)
    assert len(at.sidebar.radio) == 1, "le choix de la langue est dans la barre latérale"
    assert not at.sidebar.selectbox, "le choix du cycle n'est plus dans la barre latérale"
