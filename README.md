# 🔬 Semiconductor Quality Control (SC-QC)

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Machine Learning](https://img.shields.io/badge/ML-Random%20Forest%20%2B%20SMOTE-orange.svg)](https://scikit-learn.org/)

Ce projet implémente un système complet d'intelligence artificielle pour le contrôle de la qualité dans l'industrie des semi-conducteurs, basé sur le célèbre dataset **UCI SECOM**.

## 🎯 Objectif du Projet
L'objectif est de prédire les défauts de fabrication des puces à partir des données de plus de 500 capteurs. Le projet traite spécifiquement le problème du déséquilibre des classes (seulement ~6% de produits défectueux) pour minimiser les pertes de production.

## 🚀 Fonctionnalités Clés
*   **Nettoyage Avancé** : Suppression automatique des colonnes inutiles, imputation intelligente des valeurs manquantes et encodage des phases de production.
*   **Pipeline de Rééquilibrage (SMOTE)** : Utilisation de techniques de sur-échantillonnage pour améliorer la détection des puces défectueuses.
*   **Modélisation Robuste** : Utilisation d'un classifieur Random Forest optimisé.
*   **Dashboard Industriel** : Interface interactive développée avec Streamlit pour l'analyse en temps réel.

## 📂 Structure du projet
```text
├── data/               # Datasets bruts et prétraités (exclus du git)
├── notebooks/          
│   ├── exploration.ipynb # Analyse exploratoire complète
│   ├── app.py          # Application Dashboard Streamlit
│   └── deployment.py   # Classe de déploiement orientée objet
├── results/            # Modèles sauvegardés et rapports (exclus du git)
├── src/                
│   ├── preprocessing.py # Logique de nettoyage des données
│   ├── training.py      # Pipeline d'entraînement (SMOTE + RF)
│   └── evaluation.py    # Scripts d'évaluation de performance
└── requirements.txt     # Dépendances du projet
```

## 🛠️ Installation

1.  **Cloner le dépôt**
    ```bash
    git clone https://github.com/OUEDRAOGO-glith/TP_ML_Semiconductor.git
    cd TP_ML_Semiconductor
    ```

2.  **Installer les dépendances**
    ```bash
    pip install -r requirements.txt
    ```

## 📊 Utilisation

### Lancer le Dashboard
Pour explorer les résultats et analyser de nouveaux lots de puces :
```bash
streamlit run notebooks/app.py
```

### Entraîner le modèle
Si vous souhaitez réentraîner le modèle avec de nouvelles données :
```bash
python src/training.py
```

## 📈 Résultats Actuels
*   **Précision (Accuracy) :** ~93%
*   **Technique de gestion du déséquilibre :** SMOTE (Synthetic Minority Over-sampling Technique)
*   **Modèle :** Random Forest Classifier

---
Développé par [OUEDRAOGO-glith](https://github.com/OUEDRAOGO-glith)
