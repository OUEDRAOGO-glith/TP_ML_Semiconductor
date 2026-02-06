import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE

def train_and_save_model():
    print("🚀 Démarrage de l'entraînement du modèle...")
    
    # 1. Chargement des données
    data_path = 'data/secom_preprocessed.csv'
    if not os.path.exists(data_path):
        print(f"❌ Erreur: {data_path} introuvable. Lancez d'abord le prétraitement.")
        return

    df = pd.read_csv(data_path)
    X = df.drop('Target', axis=1)
    y = df['Target']
    
    print(f"✅ Données chargées : {X.shape[0]} lignes, {X.shape[1]} features")

    # 2. Séparation Train/Test
    # On garde 20% pour le test, stratifié pour préserver le ratio de défauts
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 3. Rééquilibrage avec SMOTE sur le set d'entraînement
    print("⚖️ Application du SMOTE pour rééquilibrer les classes...")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    print(f"📈 Volume après rééquilibrage : {len(y_train_res)} échantillons")
    print(f"   Distribution : {pd.Series(y_train_res).value_counts().to_dict()}")

    # 4. Entraînement du modèle
    print("🤖 Entraînement du Random Forest...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train_res, y_train_res)

    # 5. Évaluation sur le set de test
    y_pred = model.predict(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }

    print("\n✅ Performances sur le set de test :")
    print(f"   Accuracy  : {metrics['accuracy']:.4f}")
    print(f"   Precision : {metrics['precision']:.4f}")
    print(f"   Recall    : {metrics['recall']:.4f}")
    print(f"   F1-Score  : {metrics['f1_score']:.4f}")

    # 6. Sauvegarde
    if not os.path.exists('results'):
        os.makedirs('results')
        
    joblib.dump(model, 'results/final_model_smote_rf.pkl')
    joblib.dump(metrics, 'results/final_metrics.pkl')
    # Sauvegarde du test set pour l'évaluation ultérieure si besoin
    joblib.dump((X_test, y_test), 'results/test_dataset.pkl')
    
    print("\n💾 Modèle et métriques sauvegardés dans le dossier 'results/'")
    print("✨ Entraînement terminé avec succès !")

if __name__ == "__main__":
    train_and_save_model()
