import os
import joblib
import json
import logging
import torch
import numpy as np

# Need to import Autoencoder from train_anomaly so we can load its weights
from ml.training.train_anomaly import Autoencoder

logging.basicConfig(level=logging.INFO, format="%(message)s")

class HybridDetectionEngine:
    def __init__(self, models_dir="models/"):
        self.models_dir = models_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load Preprocessor and LabelEncoder
        self.preprocessor = joblib.load(os.path.join(models_dir, "preprocessor.joblib"))
        self.le = joblib.load(os.path.join(models_dir, "label_encoder.joblib"))
        
        # Load Supervised Model (XGBoost is our primary tree-based model)
        self.classifier = joblib.load(os.path.join(models_dir, "xgboost.joblib"))
        
        # Load Anomaly Detector
        anomaly_meta_path = os.path.join(models_dir, "anomaly", "anomaly_metadata.json")
        with open(anomaly_meta_path, 'r') as f:
            self.anomaly_meta = json.load(f)
            
        self.threshold = self.anomaly_meta['threshold']
        
        # The input_dim is the number of features after preprocessing
        input_dim = len(self.preprocessor['features'])
        
        self.autoencoder = Autoencoder(input_dim).to(self.device)
        self.autoencoder.load_state_dict(torch.load(os.path.join(models_dir, "anomaly", "autoencoder_best.pt"), map_location=self.device, weights_only=True))
        self.autoencoder.eval()
        
    def predict(self, df):
        """
        Takes a raw dataframe (single row or batch), preprocesses it, 
        and runs hybrid detection.
        """
        # Preprocess (this also drops leakage columns and handles scaling)
        X_processed = self.preprocessor['pipeline'].transform(df)
        
        # 1. Supervised Classification
        clf_preds = self.classifier.predict(X_processed)
        clf_probs = self.classifier.predict_proba(X_processed)
        clf_confidences = np.max(clf_probs, axis=1)
        clf_classes = self.le.inverse_transform(clf_preds)
        
        # 2. Anomaly Detection
        X_tensor = torch.FloatTensor(X_processed).to(self.device)
        with torch.no_grad():
            reconstructed = self.autoencoder(X_tensor)
            mse = torch.mean((X_tensor - reconstructed) ** 2, dim=1).cpu().numpy()
            
        is_anomaly = mse > self.threshold
        
        results = []
        for i in range(len(df)):
            confidence = float(clf_confidences[i])
            predicted_class = clf_classes[i]
            anomaly_score = float(mse[i])
            anomaly_flag = bool(is_anomaly[i])
            
            # Hybrid Decision Logic
            # If supervised says Benign, but Anomaly says True -> Flag as Unseen Attack / Anomaly
            # If supervised says Attack -> Trust supervised class
            
            final_class = predicted_class
            severity = "LOW"
            
            if "BENIGN" in predicted_class.upper():
                if anomaly_flag:
                    final_class = "Unknown Anomaly"
                    severity = "MEDIUM"
                    confidence = 1.0 - (self.threshold / (anomaly_score + 1e-9)) # Rough heuristic for anomaly confidence
                else:
                    severity = "INFO"
            else:
                # It's a known attack
                if anomaly_flag:
                    # Both models agree it's abnormal
                    severity = "CRITICAL"
                else:
                    # Only supervised flagged it
                    severity = "HIGH"
                    
            # Bound confidence between 0 and 1
            confidence = max(0.0, min(1.0, confidence))
            
            result = {
                "prediction": final_class,
                "confidence": confidence,
                "anomaly_score": anomaly_score,
                "is_anomaly": anomaly_flag,
                "severity": severity
            }
            results.append(result)
            
        return results

if __name__ == "__main__":
    import pandas as pd
    engine = HybridDetectionEngine()
    
    # Test with a dummy single row from the raw dataset
    df = pd.read_csv("data/raw/dataset.csv", nrows=5)
    res = engine.predict(df)
    for i, r in enumerate(res):
        logging.info(f"Row {i}: {r}")
