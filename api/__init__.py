"""
api/ — Serveur de calcul de l'application web 2SMART.

Il expose, sous forme de données JSON, ce que les pages affichent : courbes,
métriques, comparaison, explications, ontologie, préparation d'un cycle et
lancement d'une simulation. Les calculs sont ceux d'ems_core.py et du dossier
core/, partagés avec l'application Streamlit.

Lancement :  uvicorn api.main:app --port 8000
"""
