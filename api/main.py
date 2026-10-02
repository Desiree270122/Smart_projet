"""
api/main.py — Routes du serveur de calcul, et service de l'interface web.

Lancement, depuis le dossier code/ :
    uvicorn api.main:app --port 8000
puis ouvrir http://localhost:8000 (l'interface doit avoir été construite :
cd web && npm install && npm run build). La documentation des routes est à
http://localhost:8000/docs.

Chaque route accepte ?lang=fr ou ?lang=en : les phrases rédigées sont renvoyées
dans cette langue.
"""

from pathlib import Path
from typing import Optional

from fastapi import Body, FastAPI, File, Form, Query, Request, UploadFile
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from api import analyse, connaissances, donnees, explication, simulation
from core import style
from core.i18n import LANGUES, tr, utiliser_langue

DOSSIER_INTERFACE = Path(__file__).resolve().parent.parent / "web" / "dist"

app = FastAPI(
    title="2SMART — serveur de calcul",
    description="Données des pages de l'application web 2SMART (gestion d'énergie d'un HESS).",
    version="2.0",
)
app.add_middleware(GZipMiddleware, minimum_size=2000)


def _langue(request: Request) -> str:
    demandee = (request.query_params.get("lang") or "").lower()
    if demandee in LANGUES:
        return demandee
    return "en" if (request.headers.get("accept-language") or "fr").lower().startswith("en") else "fr"


def _repondre(request: Request, calcul, cycle: Optional[str] = None):
    """Exécute un calcul dans la langue de la requête ; si un cycle est donné, le moteur
    est réservé et réglé avec le matériel de ce cycle pendant le calcul."""
    with utiliser_langue(_langue(request)):
        try:
            if cycle is None:
                return calcul()
            with donnees.moteur(donnees.obtenir(cycle)["materiel"]):
                return calcul()
        except donnees.Introuvable:
            return JSONResponse(status_code=404, content={"erreur": tr(
                "Ces résultats ne sont plus disponibles (le serveur a été relancé ou la simulation est trop ancienne).",
                "These results are no longer available (the server was restarted or the simulation is too old).",
            )})
        except KeyError:
            return JSONResponse(status_code=404, content={"erreur": tr(
                "Cette stratégie n'a pas été simulée sur ce cycle.", "This strategy was not simulated on this cycle.",
            )})
        except donnees.Occupe:
            return JSONResponse(status_code=503, content={"erreur": tr(
                "Une simulation est en cours : les calculs reprendront dès qu'elle sera terminée.",
                "A simulation is running: calculations will resume as soon as it is finished.",
            )})
        except ValueError as exc:
            return JSONResponse(status_code=400, content={"erreur": str(exc)})


# Général

@app.get("/api/meta")
def meta(request: Request):
    """Langues, cycles de référence et couleurs communes."""
    def calcul():
        return {
            "langues": LANGUES,
            "cycles": [{"id": c, "nom": nom} for c, nom in donnees.cycles_de_reference()],
            "couleurs": {
                "demande": style.COULEUR_DEMANDE, "eb": style.COULEUR_EB, "pb": style.COULEUR_PB,
                "convertisseur": style.COULEUR_CONVERTISSEUR, "decision": style.COULEUR_DECISION,
                "violation": style.COULEUR_VIOLATION, "reference": style.COULEUR_REFERENCE,
                "secondaire": style.COULEUR_SECONDAIRE,
            },
            "limites": analyse.limites(),
        }
    return _repondre(request, calcul)


# Analyse d'un cycle

@app.get("/api/cycles/{cycle}")
def infos_cycle(request: Request, cycle: str):
    return _repondre(request, lambda: analyse.infos_cycle(cycle), cycle)


@app.get("/api/cycles/{cycle}/strategies/{strategie}/courbes")
def courbes(request: Request, cycle: str, strategie: str):
    return _repondre(request, lambda: analyse.courbes(cycle, strategie), cycle)


@app.get("/api/cycles/{cycle}/strategies/{strategie}/indicateurs")
def indicateurs(request: Request, cycle: str, strategie: str):
    return _repondre(request, lambda: analyse.indicateurs(cycle, strategie), cycle)


@app.get("/api/cycles/{cycle}/comparaison")
def comparaison(request: Request, cycle: str, explicabilite: bool = True):
    return _repondre(request, lambda: analyse.comparaison(cycle, explicabilite), cycle)


@app.post("/api/cycles/{cycle}/coherence")
def coherence(request: Request, cycle: str):
    """Mesure la cohérence physique (E3) des stratégies d'une simulation (environ 30 s)."""
    return _repondre(request, lambda: analyse.mesurer_coherence(cycle), cycle)


@app.get("/api/cycles/{cycle}/explication")
def expliquer(request: Request, cycle: str, strategie: str, t: float = Query(..., description="Instant du cycle, en secondes")):
    return _repondre(request, lambda: explication.explication(cycle, strategie, t), cycle)


@app.get("/api/cycles/{cycle}/explication-cycle")
def expliquer_cycle(request: Request, cycle: str, strategie: str):
    return _repondre(request, lambda: explication.explication_cycle(cycle, strategie), cycle)


# Stratégies et ontologie

@app.get("/api/strategies")
def strategies(request: Request):
    return _repondre(request, connaissances.strategies)


@app.get("/api/ontologie")
def ontologie(request: Request):
    return _repondre(request, connaissances.ontologie)


@app.get("/api/ontologie/test")
def tester_ontologie(request: Request, p_kw: float, soc_eb: float, soc_pb: float):
    return _repondre(request, lambda: connaissances.tester_regles(p_kw, soc_eb, soc_pb))


# Préparer un cycle

@app.get("/api/preparation/defauts")
def defauts_preparation(request: Request):
    return _repondre(request, lambda: {
        "vehicule": simulation.vehicule_par_defaut(), "materiel": simulation.caracteristiques_materiel(),
    })


@app.post("/api/fichiers")
async def importer_fichier(
    request: Request, fichier: UploadFile = File(...), sans_entete: bool = Form(False), feuille: Optional[str] = Form(None),
):
    contenu = await fichier.read()

    def calcul():
        try:
            return simulation.analyser_fichier(fichier.filename or "cycle.csv", contenu, sans_entete, feuille)
        except Exception as exc:  # noqa: BLE001 - fichier illisible : message clair plutôt qu'une erreur brute
            raise ValueError(tr("Impossible de lire ce fichier ({e}).", "This file cannot be read ({e}).", e=exc)) from exc
    return _repondre(request, calcul)


@app.post("/api/fichiers/{id_fichier}/relire")
def relire_fichier(request: Request, id_fichier: str, options: dict = Body(...)):
    """Relit un fichier déjà importé avec d'autres options (sans en-tête, autre feuille)."""
    def calcul():
        fichier = donnees.fichiers.get(id_fichier)
        if fichier is None:
            raise donnees.Introuvable(id_fichier)
        return simulation.analyser_fichier(
            fichier["nom"], fichier["contenu"], bool(options.get("sans_entete")), options.get("feuille"), identifiant=id_fichier,
        )
    return _repondre(request, calcul)


@app.get("/api/fichiers/{id_fichier}/vitesse")
def analyser_vitesse(
    request: Request, id_fichier: str, colonne: str, sans_entete: bool = False, feuille: Optional[str] = None,
    unite: Optional[str] = None,
):
    return _repondre(request, lambda: simulation.analyser_vitesse(id_fichier, colonne, sans_entete, feuille, unite))


@app.post("/api/materiel")
def materiel(request: Request, reglages: dict = Body(default={})):
    """Caractéristiques des packs et du convertisseur pour les cellules données."""
    return _repondre(request, lambda: simulation.caracteristiques_materiel(reglages))


@app.post("/api/cycles-prepares")
def preparer(request: Request, reglages: dict = Body(...)):
    return _repondre(request, lambda: simulation.preparer_cycle(reglages))


@app.post("/api/cycles-prepares/{id_prepare}/apercu")
def apercu(request: Request, id_prepare: str, reglages: dict = Body(default={})):
    return _repondre(request, lambda: simulation.apercu_modele_physique(
        id_prepare, float(reglages.get("soc_eb0", 1.0)), float(reglages.get("soc_pb0", 1.0)), reglages.get("materiel"),
    ))


@app.get("/api/cycles-prepares/{id_prepare}/export")
def exporter(request: Request, id_prepare: str, format: str = "csv"):
    with utiliser_langue(_langue(request)):
        try:
            contenu, type_fichier, nom = simulation.exporter_cycle(id_prepare, format)
        except donnees.Introuvable:
            return JSONResponse(status_code=404, content={"erreur": tr("Cycle préparé introuvable.", "Prepared cycle not found.")})
    return Response(content=contenu, media_type=type_fichier, headers={"Content-Disposition": f'attachment; filename="{nom}"'})


# Lancer une simulation

@app.get("/api/simulation/options")
def options_simulation(request: Request):
    return _repondre(request, simulation.options_de_simulation)


@app.post("/api/simulation/resume")
def resume_cycle(request: Request, reglages: dict = Body(...)):
    return _repondre(request, lambda: simulation.resume_cycle(reglages["cycle"], reglages.get("materiel")))


@app.get("/api/simulation/diagnostic")
def diagnostic(request: Request, soc_eb0: float = 1.0, soc_pb0: float = 1.0, nb_strategies: int = 7):
    return _repondre(request, lambda: simulation.diagnostic(soc_eb0, soc_pb0, nb_strategies))


@app.post("/api/simulations")
def lancer_simulation(request: Request, reglages: dict = Body(...)):
    return _repondre(request, lambda: simulation.lancer(reglages))


@app.get("/api/simulations/{id_simulation}")
def etat_simulation(request: Request, id_simulation: str):
    return _repondre(request, lambda: simulation.etat_simulation(id_simulation))


# Interface web (construite par « npm run build » dans web/)

if DOSSIER_INTERFACE.exists():
    app.mount("/assets", StaticFiles(directory=DOSSIER_INTERFACE / "assets"), name="assets")

    @app.get("/{chemin:path}", include_in_schema=False)
    def interface(chemin: str):
        fichier = DOSSIER_INTERFACE / chemin
        if chemin and fichier.is_file() and DOSSIER_INTERFACE in fichier.resolve().parents:
            return FileResponse(fichier)
        return FileResponse(DOSSIER_INTERFACE / "index.html")
else:
    @app.get("/", include_in_schema=False)
    def interface_absente():
        return JSONResponse({"message": "Interface non construite : lancez « npm install » puis « npm run build » dans le dossier web/."})
