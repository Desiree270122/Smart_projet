# Application 2SMART — Guide de lancement et état d’avancement

## 1. Présentation générale

L’application 2SMART permet de préparer un cycle de conduite, de simuler plusieurs stratégies de gestion d’énergie, de comparer leurs performances et d’analyser les décisions prises par les différents modèles EMS.

Elle intègre également des fonctions de visualisation, d’explicabilité, de consultation des règles symboliques et d’exportation des résultats.

---

## 2. Lancement de l’application

Commence par installer les dépendances nécessaires :

```bash
pip install -r requirements.txt
```

Lance ensuite l’application Streamlit avec la commande suivante :

```bash
streamlit run Accueil.py
```

---

## 3. Fichiers nécessaires au fonctionnement des modèles

Les poids entraînés des modèles doivent être placés dans le dossier suivant :

```text
models/checkpoints/
```

Les fichiers attendus sont :

```text
EMS_MLP.pt
EMS_MLP_neurosymbolic.pt
EMS_LSTM.pt
EMS_LSTM_neurosymbolic.pt
EMS_GNN.pt
```

Le fichier de normalisation du modèle GNN doit être placé ici :

```text
models/gnn_node_scalers.npz
```

Les graphes préconstruits utilisés pour l’évaluation hors ligne du GNN doivent être enregistrés dans :

```text
data/processed/hess_graphs.pt
```

---

## 4. Fonctionnalités déjà opérationnelles

Les éléments suivants sont actuellement intégrés et utilisables dans l’application :

* le filtre de sécurité physique ;
* la stratégie fondée sur la limitation de puissance ;
* la logique floue ;
* la référence physique alpha-star ;
* la simulation en boucle fermée de plusieurs stratégies EMS ;
* la préparation et l’importation de cycles de conduite ;
* la visualisation de l’évolution des SOC ;
* l’analyse instantanée des décisions ;
* la comparaison des stratégies ;
* l’explicabilité des modèles ;
* la consultation des concepts symboliques et des règles ;
* la relecture des résultats en pseudo-temps réel ;
* l’exportation des résultats aux formats CSV et Excel.

Les stratégies actuellement simulées en boucle fermée sont :

```text
EMS_power_limitation
EMS_fuzzy_logic
EMS_MLP
EMS_MLP_neurosymbolic
```

L’application comprend également l’ensemble des pages de navigation, de la préparation des données jusqu’à l’exportation des résultats.

---

## 5. Points à vérifier avant une utilisation définitive

Certains éléments doivent encore être confirmés à partir des notebooks d’entraînement et du cahier des charges.

Ces points sont volontairement signalés dans le code par la mention :

```text
À CONFIRMER
```

Des avertissements sont également affichés dans l’interface lorsque l’application rencontre une fonctionnalité qui n’est pas encore complètement validée.

### 5.1 Colonnes d’entrée des modèles MLP

Les constantes suivantes sont définies dans `ems_core.py` :

```text
MLP_INPUT_COLS
MLP_NS_INPUT_COLS
```

La liste actuellement utilisée contient cinq variables connues, alors que neuf variables ont été évoquées pendant le développement.

La liste définitive doit donc être vérifiée dans le notebook :

```text
11_EMS_MLP.ipynb
```

L’ordre des variables doit également être identique à celui utilisé pendant l’entraînement.

---

### 5.2 Hyperparamètres des modèles MLP et LSTM

Certains hyperparamètres doivent encore être confirmés, notamment :

* le nombre de neurones dans les couches cachées ;
* le nombre de couches ;
* le taux de dropout ;
* la taille des séquences du LSTM ;
* les dimensions exactes des entrées et des sorties.

Les valeurs actuellement présentes dans `ems_core.py` sont cohérentes avec l’architecture supposée, mais elles doivent être comparées aux notebooks d’entraînement.

Lorsqu’un fichier `.pt` ne se charge pas à cause d’une erreur de dimension, il faut vérifier que l’architecture déclarée dans l’application correspond exactement à celle du modèle sauvegardé.

---

### 5.3 Simulation des modèles LSTM

Les modèles suivants peuvent être chargés :

```text
EMS_LSTM
EMS_LSTM_neurosymbolic
```

Cependant, ils ne sont pas encore utilisés directement dans la simulation en boucle fermée.

Le modèle LSTM produit actuellement trois sorties :

```text
Pdem
delta_SOC_EB
delta_SOC_PB
```

Il ne produit donc pas directement une valeur d’alpha.

La méthode permettant de convertir ces sorties en une décision de répartition n’a pas encore été confirmée.

La fonction suivante reste donc volontairement non implémentée :

```python
deriver_alpha_depuis_sortie_lstm
```

Tant que cette conversion n’est pas définie et validée, les modèles LSTM restent disponibles pour la prédiction et l’explicabilité, mais pas pour la décision directe en boucle fermée.

---

### 5.4 Simulation du modèle GNN

Le modèle EMS_GNN peut être chargé et utilisé pour l’évaluation hors ligne sur les graphes de test déjà construits.

Il peut également être analysé avec GNNExplainer depuis la page consacrée à l’explicabilité.

En revanche, son utilisation sur un nouveau cycle nécessite de reconstruire un graphe à chaque instant de la simulation.

La procédure exacte doit respecter :

* la structure des cinq nœuds ;
* les relations entre les nœuds ;
* les caractéristiques de chaque nœud ;
* la normalisation utilisée pendant l’entraînement ;
* l’ordre exact des variables.

La fonction suivante reste donc à compléter :

```python
construire_graphe_instant
```

Tant que cette fonction n’est pas validée, EMS_GNN reste réservé à l’évaluation hors ligne.

---

### 5.5 Constantes physiques à confirmer

Certaines constantes doivent encore être vérifiées dans le cahier des charges ou dans les documents techniques du système.

Les principaux points concernés sont :

```text
SOC_PB_MIN : 0.15 ou 0.20
V_PB : 400 V ou 402,60 V
SOC_LOW_THRESHOLD
```

Ces valeurs influencent directement les contraintes physiques, les règles floues et l’évolution des SOC.

Elles doivent donc être définitivement fixées avant la validation finale de l’application.

---

### 5.6 Calcul de la puissance à partir de la vitesse

L’application permet de calculer la puissance demandée à partir du profil de vitesse lorsque le fichier importé ne contient pas directement de colonne de puissance.

Ce calcul repose sur la dynamique longitudinale du véhicule et prend notamment en compte :

* la force aérodynamique ;
* la résistance au roulement ;
* la gravité ;
* l’accélération ;
* la masse du véhicule ;
* la surface frontale ;
* le coefficient de traînée ;
* la pente de la route.

Les paramètres utilisés doivent néanmoins être vérifiés pour s’assurer qu’ils correspondent exactement au véhicule étudié dans le projet 2SMART.

---

## 6. Organisation des fichiers

La structure recommandée du projet est la suivante :

```text
Accueil.py
ems_core.py
requirements.txt

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
└── gnn_node_scalers.npz

data/
└── processed/
    └── hess_graphs.pt
```

---

## 7. Recommandations avant la mise en production

Avant de considérer l’application comme complètement finalisée, il est recommandé de :

1. vérifier les colonnes d’entrée exactes des modèles ;
2. confirmer les hyperparamètres utilisés pendant l’entraînement ;
3. charger les mêmes scalers que ceux utilisés dans les notebooks ;
4. valider la conversion des sorties LSTM vers une décision EMS ;
5. implémenter la construction dynamique des graphes GNN ;
6. confirmer les constantes physiques encore incertaines ;
7. vérifier les paramètres du véhicule utilisés pour calculer la puissance ;
8. tester l’application sur plusieurs cycles de conduite ;
9. vérifier la cohérence des SOC, des puissances et des courants ;
10. documenter les versions finales des modèles utilisés.

---

## 8. État actuel de l’application

L’application constitue déjà une base fonctionnelle pour :

* importer et préparer un cycle de conduite ;
* simuler les stratégies EMS actuellement déployables ;
* comparer leurs performances ;
* analyser la répartition de l’énergie entre l’EB et la PB ;
* expliquer les décisions produites ;
* consulter les règles symboliques ;
* exporter les résultats.

La batterie d’énergie fournit principalement l’énergie nécessaire au véhicule sur la durée, tandis que la batterie de puissance intervient pour répondre aux pics de puissance et aux sollicitations rapides.

Les éléments restant à finaliser concernent principalement l’intégration complète des modèles LSTM et GNN, ainsi que la validation définitive de certains paramètres physiques et de certaines configurations d’entraînement.
#   S m a r t _ p r o j e t  
 