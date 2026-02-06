# deployment.py
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

class SemiconductorQualityControl:
    """Classe pour déployer et utiliser le modèle de contrôle qualité"""
    
    def __init__(self, model_path=None, metrics_path=None):
        """
        Initialisation du système de contrôle qualité
        
        Args:
            model_path: Chemin vers le modèle sauvegardé
            metrics_path: Chemin vers les métriques sauvegardées
        """
        print("🔧 Initialisation du système de contrôle qualité...")
        
        # Gestion intelligente des chemins par défaut
        if model_path is None:
            # On teste les deux localisations classiques (depuis notebooks/ ou depuis racine)
            paths_to_test = ['results/final_model_smote_rf.pkl', '../results/final_model_smote_rf.pkl']
            model_path = next((p for p in paths_to_test if os.path.exists(p)), paths_to_test[0])
            
        if metrics_path is None:
            paths_to_test = ['results/final_metrics.pkl', '../results/final_metrics.pkl']
            metrics_path = next((p for p in paths_to_test if os.path.exists(p)), paths_to_test[0])
        
        # Charger le modèle
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
            print(f"✅ Modèle chargé depuis : {model_path}")
        else:
            print(f"⚠️  Modèle non trouvé : {model_path}")
            print("   Création d'un modèle fictif pour le test...")
            self.model = None
        
        # Charger les métriques de référence
        if os.path.exists(metrics_path):
            self.metrics = joblib.load(metrics_path)
            print(f"✅ Métriques de référence chargées depuis : {metrics_path}")
        else:
            self.metrics = None
            print(f"⚠️  Métriques de référence non trouvées à : {metrics_path}")
    
    def prepare_data(self, new_data):
        """
        Prépare les nouvelles données pour la prédiction
        
        Args:
            new_data: DataFrame pandas avec les nouvelles données
        
        Returns:
            DataFrame préparé
        """
        print("📊 Préparation des données...")
        
        # Vérifier que les données ont la bonne forme
        expected_features = 567  # Nombre de features attendues
        if new_data.shape[1] != expected_features:
            print(f"⚠️  Attention: {new_data.shape[1]} features au lieu de {expected_features}")
        
        # Vérifier les valeurs manquantes
        missing_values = new_data.isnull().sum().sum()
        if missing_values > 0:
            print(f"⚠️  {missing_values} valeurs manquantes détectées")
            # Imputation simple (médiane)
            imputer = SimpleImputer(strategy='median')
            new_data = pd.DataFrame(imputer.fit_transform(new_data), 
                                   columns=new_data.columns)
            print("✅ Valeurs manquantes imputées")
        
        return new_data
    
    def predict(self, new_data, threshold=None):
        """
        Fait des prédictions sur de nouvelles données
        
        Args:
            new_data: DataFrame pandas
            threshold: Seuil de décision personnalisé (optionnel)
        
        Returns:
            predictions: Prédictions (0: OK, 1: Défectueux)
            probabilities: Probabilités de la classe 1
        """
        print("🤖 Prédiction en cours...")
        
        if self.model is None:
            print("❌ Aucun modèle disponible pour la prédiction")
            return None, None
        
        # Préparer les données
        prepared_data = self.prepare_data(new_data)
        
        # Faire des prédictions
        try:
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(prepared_data)[:, 1]
                
                # Appliquer un seuil personnalisé si spécifié
                if threshold is not None:
                    predictions = (probabilities >= threshold).astype(int)
                else:
                    predictions = self.model.predict(prepared_data)
            else:
                predictions = self.model.predict(prepared_data)
                probabilities = None
            
            print(f"✅ {len(predictions)} prédictions effectuées")
            
            # Statistiques des prédictions
            defect_count = np.sum(predictions == 1)
            ok_count = np.sum(predictions == 0)
            if len(predictions) > 0:
                defect_rate = defect_count / len(predictions) * 100
            else:
                defect_rate = 0
            
            print(f"📊 Résumé: {defect_count} défectueux ({defect_rate:.1f}%), {ok_count} OK")
            
            return predictions, probabilities
            
        except Exception as e:
            print(f"❌ Erreur lors de la prédiction: {e}")
            return None, None
    
    def evaluate_performance(self, X_test, y_true):
        """
        Évalue la performance du modèle sur un jeu de test
        
        Args:
            X_test: Features de test
            y_true: Labels réels
        
        Returns:
            metrics_dict: Dictionnaire des métriques
        """
        print("📈 Évaluation des performances...")
        
        if self.model is None:
            print("❌ Aucun modèle disponible pour l'évaluation")
            return None
        
        # Faire des prédictions
        y_pred, _ = self.predict(X_test)
        
        if y_pred is None:
            return None
        
        # Calculer les métriques
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        cm = confusion_matrix(y_true, y_pred)
        
        # Afficher les résultats
        print("\n" + "="*60)
        print("📊 PERFORMANCES DU MODÈLE")
        print("="*60)
        print(f"Accuracy:    {accuracy:.4f}")
        print(f"Précision:   {precision:.4f}")
        print(f"Recall:      {recall:.4f}")
        print(f"F1-score:    {f1:.4f}")
        
        print(f"\n📋 Matrice de confusion:")
        print(f"    | Prédit OK | Prédit Défectueux |")
        print(f"    |-----------|-------------------|")
        print(f"Vrai OK | {cm[0,0]:^10} | {cm[0,1]:^17} |")
        print(f"Vrai Déf| {cm[1,0]:^10} | {cm[1,1]:^17} |")
        
        # Comparer avec les métriques de référence
        if self.metrics:
            print(f"\n📊 Comparaison avec les métriques d'entraînement:")
            print(f"    | Entraînement | Test      | Différence |")
            print(f"    |--------------|-----------|------------|")
            print(f"Accuracy  | {self.metrics['accuracy']:.4f}     | {accuracy:.4f}  | {accuracy-self.metrics['accuracy']:+.4f}    |")
            print(f"Precision | {self.metrics['precision']:.4f}     | {precision:.4f}  | {precision-self.metrics['precision']:+.4f}    |")
            print(f"Recall    | {self.metrics['recall']:.4f}     | {recall:.4f}  | {recall-self.metrics['recall']:+.4f}    |")
            print(f"F1-score  | {self.metrics['f1_score']:.4f}     | {f1:.4f}  | {f1-self.metrics['f1_score']:+.4f}    |")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm
        }

if __name__ == "__main__":
    # Test du système de déploiement
    qc_system = SemiconductorQualityControl()
    
    # 1. Charger les données pour le test
    data_path = '../data/secom_preprocessed.csv'
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        X = df.drop('Target', axis=1)
        y = df['Target']
        
        # Simuler l'arrivée de nouvelles données (5 premières lignes)
        print("\n--- TEST: Simulation de nouvelles données ---")
        new_samples = X.head(5)
        predictions, probs = qc_system.predict(new_samples)
        
        # Affichage détaillé
        for i, (pred, prob) in enumerate(zip(predictions, probs)):
            status = "DÉFECTUEUX ❌" if pred == 1 else "OK ✅"
            print(f"Échantillon {i+1} : {status} (Probabilité: {prob:.4f})")
            
        # 2. Évaluer les performances globales
        print("\n--- TEST: Évaluation globale du modèle ---")
        qc_system.evaluate_performance(X, y)
    else:
        print(f"❌ Erreur: Fichier de données {data_path} non trouvé.")
