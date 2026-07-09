# 2SMART — Semi-Supervised MAnagement of dual batteRy elecTric vehicles

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-app-FF4B4B)
![PyTorch](https://img.shields.io/badge/PyTorch-mod%C3%A8les-EE4C2C)
![Statut](https://img.shields.io/badge/statut-7%2F7%20strat%C3%A9gies%20op%C3%A9rationnelles-brightgreen)

Application interactive de gestion d'énergie pour un système de stockage hybride (HESS) en véhicule électrique : répartition de la puissance entre une batterie d'énergie (EB) et une batterie de puissance (PB), comparaison de 7 stratégies EMS (déterministes, apprises, neurosymboliques), filtre de sécurité physique, explicabilité et export des résultats.

Projet réalisé dans le cadre d'un stage de recherche .

---

## Sommaire

1. [Lancement](#1-lancement)
2. [Fichiers nécessaires](#2-fichiers-nécessaires-au-fonctionnement-des-modèles)
3. [Fonctionnalités opérationnelles](#3-fonctionnalités-opérationnelles)
4. [Points encore ouverts](#4-points-encore-ouverts)
5. [Organisation des fichiers](#5-organisation-des-fichiers)
6. [État actuel](#6-état-actuel)

---

## 1. Lancement

```bash
pip install -r requirements.txt
streamlit run Accueil.py
```

`torch_geometric` (nécessaire uniquement pour `EMS_GNN`) est importé de façon paresseuse : son absence ne bloque pas les 6 autres stratégies.

---

## 2. Fichiers nécessaires au fonctionnement des modèles

### Poids entraînés — `models/checkpoints/`

```text
EMS_MLP.pt
EMS_MLP_neurosymbolic.pt
EMS_LSTM.pt
EMS_LSTM_neurosymbolic.pt
EMS_GNN.pt
```

### Fichiers de normalisation (scalers) — `models/`

```text
EMS_MLP_scalers.npz
EMS_LSTM_scalers.npz
EMS_LSTM_neurosymbolic_scalers.npz
mlp_ns_scalers.npz          # nom different de la convention EMS_<nom>_scalers.npz
gnn_node_scalers.npz
```

Sans ces fichiers, chaque modèle concerné tourne quand même (avec un avertissement explicite dans l'application), mais ses entrées ne sont pas normalisées et ses résultats sont peu fiables.

### Graphes de test préconstruits (évaluation hors ligne du GNN) — `data/processed/`

```text
hess_graphs.pt
```

### Cycle de conduite de référence — `data/`

```text
Artemis.csv
```

---

## 3. Fonctionnalités opérationnelles

* filtre de sécurité physique (SOC, puissance, courant, convertisseur) ;
* préparation et import de cycles de conduite (tout format, avec ou sans en-tête) ;
* configuration de l'architecture des batteries et du convertisseur, appliquée en direct à la simulation ;
* **les 7 stratégies EMS simulées en boucle fermée** ;
* comparaison à la référence hors ligne `alpha-star` ;
* visualisation de l'évolution des SOC, puissances et courants ;
* analyse instantanée des décisions ;
* explicabilité par stratégie ;
* consultation des concepts symboliques et de la base de règles floues ;
* relecture en pseudo-temps réel ;
* export des résultats (CSV / Excel), par stratégie ou combiné.

### Les 7 stratégies

| Stratégie | Type | Statut |
|---|---|---|
| `EMS_power_limitation` | Déterministe | Fiable |
| `EMS_fuzzy_logic` | Déterministe (7 règles) | Fiable |
| `EMS_MLP` | Appris (tabulaire) | Fiable (5 entrées confirmées) |
| `EMS_MLP_neurosymbolic` | Hybride | Fonctionnel — cible d'entraînement non confirmée |
| `EMS_LSTM` | Appris (temporel) | Fonctionnel — conversion sortie→alpha non confirmée |
| `EMS_LSTM_neurosymbolic` | Hybride | Fonctionnel — cible d'entraînement non confirmée |
| `EMS_GNN` | Appris (relationnel) | Fonctionnel — construction du graphe en boucle fermée non confirmée |

---

## 4. Points encore ouverts

La plupart des zones d'ombre initiales ont été résolues en confrontant le code aux notebooks source (01 à 12). Il reste :

### 4.1 Conversion de la sortie LSTM en `alpha`

`EMS_LSTM` et `EMS_LSTM_neurosymbolic` sortent `Pdem`, `ΔSOC_EB`, `ΔSOC_PB` — jamais `alpha` directement. La fonction `deriver_alpha_depuis_sortie_lstm` inverse l'équation confirmée de mise à jour du SOC pour en déduire un `alpha` exploitable. **C'est une interprétation physique de notre part, pas une formule confirmée** dans un notebook source.

### 4.2 Construction du graphe GNN en boucle fermée

`construire_graphe_instant` reproduit la topologie confirmée (5 nœuds, arêtes exactes, liaison directe EB-moteur incluse) de `05_EMS_graph_construction.ipynb`, mais sans appliquer la normalisation du scaler pendant la construction elle-même. À vérifier avant de considérer les résultats comme fiables.

### 4.3 Colonnes de `EMS_LSTM_neurosymbolic`

Les 11 colonnes confirmées (`08_EMS_LSTM_neurosymbolic.ipynb`) sont câblées, mais dépendent d'un seuil de variance calculé sur le jeu d'entraînement d'origine (indicateurs symboliques jugés « constants » donc exclus) — ce choix n'est pas garanti de rester valide sur un autre cycle.

### 4.4 Cibles d'entraînement des modèles LSTM

Confirmées pour `EMS_MLP`/`EMS_MLP_neurosymbolic` (`alpha_ems_alpha_star`) et `EMS_GNN` (`alpha_historical`). **Non confirmées** pour `EMS_LSTM` et `EMS_LSTM_neurosymbolic`.

### 4.5 Contenu exact de la fonction de perte neurosymbolique

La composition (target/rules/physics/balance/continuité/convergence pour MLP-NS ; data/bounds/physics/rules pour LSTM-NS) est confirmée par les logs d'entraînement, mais la composante `rules` y apparaît systématiquement à 0,0000 — signal à investiguer, pas encore expliqué.

> Ces points sont signalés directement dans l'application par des avertissements explicites lors de la simulation, et dans `ems_core.py` par des commentaires précisant la source de chaque hypothèse.

---

## 5. Organisation des fichiers

```text
Accueil.py
ems_core.py
requirements.txt
.gitignore

pages/
├── 2_Preparation_donnees.py
├── 3_Simulation_globale.py
├── 4_Evolution_SOC.py
├── 5_Analyse_instantanee.py
├── 6_Comparaison_optimisation.py
├── 7_Explicabilite.py
├── 8_Ontologie_regles.py
└── 9_Temps_reel_export.py

models/
├── checkpoints/
│   ├── EMS_MLP.pt
│   ├── EMS_MLP_neurosymbolic.pt
│   ├── EMS_LSTM.pt
│   ├── EMS_LSTM_neurosymbolic.pt
│   └── EMS_GNN.pt
├── EMS_MLP_scalers.npz
├── EMS_LSTM_scalers.npz
├── EMS_LSTM_neurosymbolic_scalers.npz
├── mlp_ns_scalers.npz
└── gnn_node_scalers.npz

data/
├── Artemis.csv
└── processed/
    └── hess_graphs.pt

ontologies/
└── OntoHESS.owl
```

---

## 6. État actuel

L'application couvre l'intégralité du pipeline : import et préparation d'un cycle de conduite, configuration de l'architecture physique (batteries, convertisseur), simulation des 7 stratégies EMS en boucle fermée, comparaison à la référence `alpha-star`, explicabilité, consultation de l'ontologie et des règles, export des résultats.

La batterie d'énergie (EB) fournit l'essentiel de l'énergie sur la durée du trajet ; la batterie de puissance (PB) intervient en assistance lors des pics de demande — comportement confirmé empiriquement (`alpha_ems_eb_priority` moyen ≈ 0,11 sur les données d'entraînement).

Les points encore ouverts (section 4) concernent des interprétations physiques assumées comme telles plutôt que des blocages fonctionnels : les 7 stratégies tournent et produisent des résultats exploitables, avec un niveau de confiance clairement différencié et signalé dans l'interface.