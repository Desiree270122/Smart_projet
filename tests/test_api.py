"""
Tests du serveur de calcul de l'application web (dossier api/) : chaque route
doit répondre, en français et en anglais.

    python -m pytest tests/test_api.py
"""

import sys
import time
from pathlib import Path

import pandas as pd
import pytest

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE))

from fastapi.testclient import TestClient  # noqa: E402

from api.main import app  # noqa: E402

client = TestClient(app)
LANGUES = ["fr", "en"]
STRATEGIES = [
    "EMS_power_limitation", "EMS_fuzzy_logic", "EMS_MLP", "EMS_MLP_neurosymbolic", "EMS_LSTM",
    "EMS_LSTM_neurosymbolic", "EMS_GNN",
]


def lire(url, **parametres):
    reponse = client.get(url, params=parametres)
    assert reponse.status_code == 200, (url, reponse.status_code, reponse.text[:300])
    return reponse.json()


@pytest.mark.parametrize("langue", LANGUES)
def test_meta_et_cycles(langue):
    meta = lire("/api/meta", lang=langue)
    assert [c["id"] for c in meta["cycles"]] == ["artemis", "wltc"]
    for cycle in meta["cycles"]:
        infos = lire(f"/api/cycles/{cycle['id']}", lang=langue)
        assert len(infos["strategies"]) == 7 and infos["e3_disponible"]
    assert meta["cycles"][0]["nom"].startswith("Artemis ur" + ("bain" if langue == "fr" else "ban"))


@pytest.mark.parametrize("strategie", ["EMS_power_limitation", "EMS_MLP_neurosymbolic"])
def test_courbes_et_indicateurs(strategie):
    courbes = lire(f"/api/cycles/artemis/strategies/{strategie}/courbes")
    n = len(courbes["temps_min"])
    assert n == len(courbes["demande_kw"]) == len(courbes["eb_kw"]) == len(courbes["lectures"]["repartition"]["indices"])
    assert len(courbes["soc_eb_pct"]) == len(courbes["lectures"]["soc"]["indices"]) == n + 1
    assert max(courbes["lectures"]["repartition"]["indices"]) < len(courbes["lectures"]["repartition"]["textes"])
    indicateurs = lire(f"/api/cycles/artemis/strategies/{strategie}/indicateurs", lang="en")
    assert len(indicateurs["synthese"]) == 5 and len(indicateurs["composants"]) == 3
    assert 0.9 < indicateurs["metriques"]["rendement_hess"] < 1.0


def test_strategie_inconnue_et_cycle_inconnu():
    assert client.get("/api/cycles/artemis/strategies/EMS_inconnue/courbes").status_code == 404
    assert client.get("/api/cycles/sim-inconnu").status_code == 404


@pytest.mark.parametrize("langue", LANGUES)
@pytest.mark.parametrize("cycle", ["artemis", "wltc"])
def test_comparaison_et_conclusion(langue, cycle):
    c = lire(f"/api/cycles/{cycle}/comparaison", lang=langue)
    assert len(c["strategies"]) == 7 and len(c["criteres"]) == 6
    assert c["conclusion"]["type"] == "succes" and "NS-LSTM" in c["conclusion"]["texte"]
    assert all(l["strategie"]["cle"] in STRATEGIES for l in c["conclusion"]["lignes"])
    sans = lire(f"/api/cycles/{cycle}/comparaison", lang=langue, explicabilite="false")
    assert len(sans["criteres"]) == 4


@pytest.mark.parametrize("strategie", STRATEGIES)
def test_explication_a_un_instant(strategie):
    e = lire("/api/cycles/artemis/explication", strategie=strategie, t=1249, lang="en")
    assert 3 <= len(e["raisons"]) <= 5 and not e["demande_nulle"]
    assert e["cascade"]["etapes"] and len(e["autres"]) == 7 and len(e["et_si"]["variantes"]) == 6
    assert e["raisons_detail"]["type"] in ("regle", "regles_floues", "contributions")


def test_explication_demande_nulle_et_garde_fou():
    nulle = lire("/api/cycles/artemis/explication", strategie="EMS_MLP", t=1)
    assert nulle["demande_nulle"] and len(nulle["raisons"]) == 1 and nulle["et_si"] is None
    garde = lire("/api/cycles/artemis/explication", strategie="EMS_MLP_neurosymbolic", t=9560)
    assert garde["reseau"]["garde_fou"] and any("garde-fou" in r for r in garde["raisons"])


@pytest.mark.parametrize("strategie", ["EMS_MLP_neurosymbolic", "EMS_LSTM", "EMS_power_limitation"])
def test_explication_sur_le_cycle(strategie):
    e = lire("/api/cycles/wltc/explication-cycle", strategie=strategie)
    assert 0 <= e["e3"] <= 1 and len(e["au_fil_du_cycle"]["temps_min"]) == len(e["au_fil_du_cycle"]["decision_pct"])
    assert (e["regles"] is not None) == (strategie == "EMS_MLP_neurosymbolic")


@pytest.mark.parametrize("langue", LANGUES)
def test_strategies_et_ontologie(langue):
    s = lire("/api/strategies", lang=langue)
    assert len(s["fiches"]) == 7 and all(f["flux"] and f["fonctionnement"] for f in s["fiches"])
    o = lire("/api/ontologie", lang=langue)
    assert o["disponible"] and o["compteurs"]["regles"] == 22 and len(o["regles_floues"]) == 7
    assert len(o["regles_lues"]["mode"]) == 4 and len(o["regles_lues"]["repartition"]) == 5
    test = lire("/api/ontologie/test", p_kw=-30, soc_eb=0.6, soc_pb=0.8, lang=langue)
    assert test["etat"]["cle"] == "state_Overload_Low" and test["alpha"] == pytest.approx(0.79, abs=0.01)


def test_preparer_un_cycle_puis_le_simuler():
    """Import d'un fichier, préparation, aperçu, export, puis simulation des deux références."""
    contenu = pd.read_csv(RACINE / "data" / "wltc.csv")[["time", "speed"]].iloc[:400].to_csv(index=False).encode()
    fichier = client.post("/api/fichiers", files={"fichier": ("wltc.csv", contenu, "text/csv")}).json()
    assert fichier["devine"]["speed"] == "speed" and fichier["devine"]["time"] == "time"
    assert fichier["vitesse"]["unite"] == "m/s"

    prepare = client.post("/api/cycles-prepares", json={
        "id_fichier": fichier["id"], "col_vitesse": "speed", "col_temps": "time", "unite_vitesse": "m/s", "repetitions": 2,
    }).json()
    assert prepare["nb_points"] == 800 and prepare["puissance_calculee"]

    apercu = client.post(f"/api/cycles-prepares/{prepare['id']}/apercu", json={"soc_eb0": 0.9, "soc_pb0": 0.8}).json()
    assert len(apercu["soc_eb_pct"]) == 800 and apercu["soc_eb_pct"][0] == pytest.approx(90.0)
    assert client.get(f"/api/cycles-prepares/{prepare['id']}/export", params={"format": "csv"}).content.startswith(b"\xef\xbb\xbftime;")

    resume = client.post("/api/simulation/resume", json={"cycle": prepare["id"]}).json()
    assert resume["duree_min"] > 13 and not resume["trop_exigeant"]

    lancee = client.post("/api/simulations", json={"cycle": prepare["id"], "strategies": [], "precision": "rapide"}).json()
    for _ in range(120):
        etat = client.get(f"/api/simulations/{lancee['id']}").json()
        if etat["etat"] != "en_cours":
            break
        time.sleep(0.5)
    assert etat["etat"] == "terminee" and etat["nb_strategies"] == 2

    infos = lire(f"/api/cycles/{lancee['id']}")
    assert [s["cle"] for s in infos["strategies"]] == ["EMS_power_limitation", "EMS_fuzzy_logic"] and not infos["e3_disponible"]
    assert lire(f"/api/cycles/{lancee['id']}/comparaison")["sur_ce_cycle"]["classement"]
    assert client.post(f"/api/cycles/{lancee['id']}/coherence").json()["e3"]["EMS_power_limitation"] == pytest.approx(1.0)


def test_materiel_modifie_puis_moteur_retabli():
    """Un matériel différent change les limites le temps du calcul, puis le moteur reprend ses valeurs."""
    import ems_core as core

    defauts = lire("/api/preparation/defauts")
    assert defauts["materiel"]["packs"][0]["energie_wh"] == pytest.approx(12601.6, abs=1)
    double = client.post("/api/materiel", json={"eb": {"n_parallele": 14}}).json()
    assert double["packs"][0]["energie_wh"] == pytest.approx(2 * 12601.6, abs=2)
    assert core.ENERGY_EB_WH == pytest.approx(12601.6, abs=1) and core.P_EB_MAX_W == 12600.0


def test_options_de_simulation_et_diagnostic():
    options = lire("/api/simulation/options", lang="en")
    assert sum(len(f["strategies"]) for f in options["familles"]) == 7 and len(options["precisions"]) == 3
    assert lire("/api/simulation/diagnostic", soc_eb0=0.3, soc_pb0=1.0, nb_strategies=7)["alertes"]


def test_instant_propose_pour_l_explication():
    """L'instant proposé par défaut correspond à une demande franche, pas à un arrêt du véhicule."""
    for cycle in ("artemis", "wltc"):
        infos = lire(f"/api/cycles/{cycle}")
        assert infos["t_min"] <= infos["t_defaut"] <= infos["t_max"]
        assert not lire(f"/api/cycles/{cycle}/explication", strategie="EMS_power_limitation", t=infos["t_defaut"])["demande_nulle"]


def test_textes_de_l_interface_dans_les_deux_langues():
    """Dans l'interface web, chaque tr("français", "anglais") a ses deux textes, avec les mêmes {champs}."""
    import re

    appel = re.compile(r'\btr\(\s*"((?:[^"\\]|\\.)*)"\s*,\s*"((?:[^"\\]|\\.)*)"')
    champs = re.compile(r"\{(\w+)\}")
    fichiers = sorted((RACINE / "web" / "src").rglob("*.jsx"))
    assert fichiers, "interface web introuvable"
    defauts, nombre = [], 0
    for fichier in fichiers:
        for francais, anglais in appel.findall(fichier.read_text(encoding="utf-8")):
            nombre += 1
            if not francais.strip() or not anglais.strip():
                defauts.append(f"{fichier.name} : texte vide ({francais[:40]!r} / {anglais[:40]!r})")
            elif set(champs.findall(francais)) != set(champs.findall(anglais)):
                defauts.append(f"{fichier.name} : champs différents entre les deux langues ({francais[:50]!r})")
    assert nombre > 300 and not defauts, "\n".join(defauts)
