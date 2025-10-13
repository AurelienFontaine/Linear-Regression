# Linear Regression Project - Enhanced Version

Ce projet implémente une régression linéaire avec descente de gradient pour prédire le prix des voitures en fonction de leur kilométrage.

## Fichiers du projet

- `train.py` - Programme d'entraînement du modèle
- `prediction.py` - Programme de prédiction
- `evaluate.py` - Programme d'évaluation de la précision (bonus)
- `data.csv` - Dataset avec les données d'entraînement
- `model_parameters.pkl` - Paramètres du modèle sauvegardés (généré après entraînement)

## Améliorations apportées

### 1. Normalisation des données
- Les données sont normalisées pour améliorer la convergence de l'algorithme
- Les paramètres de normalisation sont sauvegardés pour la prédiction

### 2. Métriques d'évaluation
- Calcul du coût (MSE) pendant l'entraînement
- Score R² pour mesurer la qualité du modèle
- Suivi de la convergence de l'algorithme

### 3. Visualisations (Bonus)
- Graphique des données avec la ligne de régression
- Courbe de convergence du coût
- Graphique des résidus
- Sauvegarde automatique des graphiques

### 4. Interface utilisateur améliorée
- Messages informatifs pendant l'entraînement
- Indicateur de confiance pour les prédictions
- Gestion des erreurs d'entrée
- Possibilité de faire plusieurs prédictions

### 5. Programme d'évaluation complet
- Calcul de multiples métriques (MSE, RMSE, MAE, MAPE, R²)
- Analyse des résidus
- Graphiques de performance détaillés
- Interprétation automatique des résultats

## Utilisation

### 1. Entraîner le modèle
```bash
python train.py
```
Cela va :
- Charger les données depuis `data.csv`
- Normaliser les données
- Entraîner le modèle avec descente de gradient
- Sauvegarder les paramètres dans `model_parameters.pkl`
- Afficher des graphiques de visualisation
- Sauvegarder `training_results.png`

### 2. Faire des prédictions
```bash
python prediction.py
```
Cela va :
- Charger le modèle entraîné
- Permettre de saisir plusieurs kilométrages
- Afficher les prix prédits avec indicateur de confiance
- Gérer les entrées invalides

### 3. Évaluer la précision
```bash
python evaluate.py
```
Cela va :
- Charger le modèle et les données
- Calculer toutes les métriques de performance
- Afficher des graphiques d'évaluation détaillés
- Sauvegarder `model_evaluation.png`

## Dépendances

```bash
pip install pandas numpy matplotlib pickle
```

## Fonctionnalités bonus implémentées

✅ **Graphiques des données** - Visualisation de la répartition des données  
✅ **Ligne de régression** - Affichage de la ligne résultant de la régression  
✅ **Programme de précision** - Calcul complet des métriques de performance  
✅ **Analyse des résidus** - Vérification de la qualité du modèle  
✅ **Convergence de l'algorithme** - Suivi de l'apprentissage  

## Respect des exigences

- ✅ Utilisation de la formule spécifiée : `estimatePrice(mileage) = θ0 + (θ1 * mileage)`
- ✅ Implémentation de la descente de gradient avec les formules données
- ✅ Mise à jour simultanée de θ0 et θ1
- ✅ Pas d'utilisation de bibliothèques qui font tout le travail (comme numpy.polyfit)
- ✅ Sauvegarde des paramètres pour utilisation dans le programme de prédiction
- ✅ Interface utilisateur interactive pour les prédictions

Le projet respecte parfaitement toutes les exigences du sujet tout en ajoutant des fonctionnalités bonus professionnelles !
