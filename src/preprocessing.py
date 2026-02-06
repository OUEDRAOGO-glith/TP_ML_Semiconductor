import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import os

def load_and_preprocess(file_path):
    """
    Charge, nettoie et prépare le dataset SECOM.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Le fichier {file_path} est introuvable.")
        
    print(f"--- Chargement de {file_path} ---")
    df = pd.read_csv(file_path)
    
    # 1. Suppression de la colonne Time
    if 'Time' in df.columns:
        df = df.drop('Time', axis=1)
        
    # 2. Gestion de la cible
    target_name = 'Pass/Fail'
    if target_name not in df.columns:
        # Si déjà renommée ou déplacée
        target_name = df.columns[-1]
        
    y = df[target_name].replace({-1: 0, 1: 1})
    X = df.drop(target_name, axis=1)
    
    # 3. Encodage de la colonne Phase
    if 'Phase' in X.columns:
        X = pd.get_dummies(X, columns=['Phase'], prefix='Phase', drop_first=True)
        
    # 4. Suppression des colonnes avec trop de vides (> 50%)
    missing_ratio = X.isnull().mean()
    cols_to_keep = missing_ratio[missing_ratio <= 0.5].index
    X = X[cols_to_keep]
    
    # 5. Imputation par la médiane
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    # 6. Recombinaison
    df_clean = X_imputed.copy()
    df_clean['Target'] = y.values
    
    print(f"Prétraitement terminé : {df_clean.shape[1]} colonnes conservées.")
    return df_clean

if __name__ == "__main__":
    # Test du script
    raw_data_path = "data/uci-secom.csv"
    if os.path.exists(raw_data_path):
        processed_df = load_and_preprocess(raw_data_path)
        processed_df.to_csv("data/secom_preprocessed.csv", index=False)
        print("Fichier de sortie généré : data/secom_preprocessed.csv")
