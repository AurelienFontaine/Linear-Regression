
# Projet de Régression Linéaire

Ce projet implémente une **régression linéaire simple** pour prédire le prix d'une voiture en fonction de son kilométrage, à l'aide de la descente de gradient. Il inclut la normalisation des données, des visualisations avancées, des métriques d'évaluation complètes et une interface utilisateur interactive.

## Sommaire
- [Présentation](#présentation)
- [Organisation du projet](#organisation-du-projet)
- [Principe de la régression linéaire](#principe-de-la-régression-linéaire)
- [Scripts et fonctionnalités](#scripts-et-fonctionnalités)
- [Métriques d'évaluation](#métriques-dévaluation)
- [Visualisations](#visualisations)
- [Utilisation](#utilisation)
- [Exemple de données](#exemple-de-données)

## Présentation

L'objectif est de modéliser la relation entre le kilométrage (`km`) et le prix (`price`) d'une voiture d'occasion. Le modèle apprend à partir d'un jeu de données réel, puis permet de prédire le prix d'une voiture à partir de son kilométrage.

## Organisation du projet

```
srcs/
	├── data.csv              # Jeu de données (km, price)
	├── train.py              # Entraînement du modèle
	├── prediction.py         # Prédiction interactive
	├── evaluate.py           # Évaluation et visualisation des performances
	└── model_parameters.pkl  # Paramètres sauvegardés (généré)
```

## Principe de la régression linéaire

La régression linéaire cherche à approximer la relation entre une variable indépendante $x$ (ici, le kilométrage) et une variable dépendante $y$ (le prix) par une droite :

$$ y = \theta_0 + \theta_1 x $$

Le modèle est entraîné par **descente de gradient** pour minimiser l'erreur quadratique moyenne (MSE) entre les prix réels et prédits. Les données sont **normalisées** pour accélérer la convergence.

## Scripts et fonctionnalités

### 1. Entraînement (`train.py`)
- Lecture et normalisation des données
- Descente de gradient pour ajuster $\theta_0$ et $\theta_1$
- Affichage de la progression (coût, R²)
- Sauvegarde des paramètres et des bornes de normalisation
- Génération de graphiques :
	- Données + droite de régression
	- Courbe de convergence du coût
	- Graphique des résidus

### 2. Prédiction (`prediction.py`)
- Chargement automatique du modèle entraîné
- Interface interactive en ligne de commande
- Prédiction du prix à partir d'un kilométrage saisi
- Gestion des erreurs et possibilité de faire plusieurs prédictions

### 3. Évaluation (`evaluate.py`)
- Calcul de métriques avancées :
	- MSE, RMSE, MAE, MAPE, R²
- Analyse et visualisation des résidus
- Graphiques de performance détaillés
- Interprétation automatique des résultats

## Métriques d'évaluation

- **MSE** (Mean Squared Error) : Erreur quadratique moyenne
- **RMSE** (Root Mean Squared Error) : Racine carrée de la MSE
- **MAE** (Mean Absolute Error) : Erreur absolue moyenne
- **MAPE** (Mean Absolute Percentage Error) : Erreur absolue moyenne en pourcentage
- **R²** (Score de détermination) : Proportion de la variance expliquée par le modèle

## Visualisations

Le projet génère automatiquement plusieurs graphiques :
- Données et droite de régression
- Courbe de convergence du coût
- Graphique des résidus
- Histogramme des résidus
- Erreur en fonction du kilométrage
- Analyse de l'erreur par tranche de prix
- Résumé visuel des performances

## Utilisation

1. **Entraîner le modèle** :
	 ```bash
	 cd srcs
	 python3 train.py
	 ```
	 → Génère `model_parameters.pkl` et `training_results.png`

2. **Faire des prédictions** :
	 ```bash
	 python3 prediction.py
	 ```
	 → Saisir un kilométrage pour obtenir une estimation du prix

3. **Évaluer le modèle** :
	 ```bash
	 python3 evaluate.py
	 ```
	 → Affiche les métriques et génère `model_evaluation.png`

## Exemple de données (`data.csv`)

| km     | price |
|--------|-------|
| 240000 | 3650  |
| 139800 | 3800  |
| 150500 | 4400  |
| ...    | ...   |

## Remarques
- Le projet est conçu pour être facilement extensible (autres variables, modèles, etc.)
- Les scripts sont robustes et gèrent les erreurs d'entrée
- Les visualisations sont sauvegardées automatiquement

---
*Auteur : Aurelien Fontaine — Octobre 2025*
