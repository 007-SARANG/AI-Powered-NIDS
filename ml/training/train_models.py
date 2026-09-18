import pandas as pd
import numpy as np
import logging
import os
import joblib
import json
from datetime import datetime
import time

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

from ml.preprocessing.pipeline import NIDSPreprocessor, split_data

logging.basicConfig(level=logging.INFO, format="%(message)s")

class ModelTrainer:
    def __init__(self, data_path="data/raw/dataset.csv", models_dir="models/"):
        self.data_path = data_path
        self.models_dir = models_dir
        self.experiment_log = os.path.join(models_dir, "experiment_log.json")
        os.makedirs(models_dir, exist_ok=True)
        
        # Label Encoder for targets (needed for XGBoost etc)
        self.le = LabelEncoder()

    def load_and_preprocess(self):
        preprocessor = NIDSPreprocessor()
        df = preprocessor.load_data(self.data_path)
        
        # Encode string labels to integers
        y = df['Label'].values
        y_encoded = self.le.fit_transform(y)
        df['Label'] = y_encoded
        
        # Save LabelEncoder
        joblib.dump(self.le, os.path.join(self.models_dir, "label_encoder.joblib"))
        
        self.n_classes = len(self.le.classes_)
        self.class_names = self.le.classes_.tolist()
        logging.info(f"Classes: {self.class_names}")
        
        # Split first
        from ml.preprocessing.pipeline import prepare_data
        X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test = prepare_data(df)
        
        # Fit on train
        X_train = preprocessor.fit_transform(X_train_raw)
        X_val = preprocessor.transform(X_val_raw)
        X_test = preprocessor.transform(X_test_raw)
        
        preprocessor.save(os.path.join(self.models_dir, "preprocessor.joblib"))
        
        return X_train, X_val, X_test, y_train.values, y_val.values, y_test.values
        
    def evaluate(self, model, X, y_true, model_name):
        start_time = time.time()
        y_pred = model.predict(X)
        inf_time = (time.time() - start_time) / len(X)
        
        y_prob = None
        roc_auc = None
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X)
            try:
                if self.n_classes > 2:
                    roc_auc = roc_auc_score(y_true, y_prob, multi_class='ovr')
                else:
                    roc_auc = roc_auc_score(y_true, y_prob[:, 1])
            except ValueError:
                roc_auc = None # Might happen if a class is missing in val/test
                
        cm = confusion_matrix(y_true, y_pred)
        # Calculate FPR and FNR per class or generally
        # For simplicity in metrics dict, we log the raw confusion matrix
        
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision_macro": float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
            "recall_macro": float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
            "f1_macro": float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
            "f1_weighted": float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
            "roc_auc": float(roc_auc) if roc_auc else None,
            "inference_latency_ms": inf_time * 1000,
            "confusion_matrix": cm.tolist()
        }
        
        return metrics
        
    def log_experiment(self, name, train_time, metrics, params):
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "model": name,
            "hyperparameters": params,
            "train_time_s": train_time,
            "metrics": metrics
        }
        
        logs = []
        if os.path.exists(self.experiment_log):
            with open(self.experiment_log, 'r') as f:
                logs = json.load(f)
                
        logs.append(log_entry)
        
        with open(self.experiment_log, 'w') as f:
            json.dump(logs, f, indent=4)
            
        logging.info(f"Logged experiment: {name} -> F1 Macro: {metrics['f1_macro']:.4f}")
        
    def run_all(self):
        logging.info("Starting training pipeline...")
        X_train, X_val, X_test, y_train, y_val, y_test = self.load_and_preprocess()
        
        models = {
            "LogisticRegression": LogisticRegression(max_iter=1000, n_jobs=-1, class_weight='balanced'),
            "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=15, n_jobs=-1, class_weight='balanced'),
            "XGBoost": XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, n_jobs=-1)
        }
        
        for name, model in models.items():
            logging.info(f"\n--- Training {name} ---")
            
            start_train = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - start_train
            
            logging.info(f"Training completed in {train_time:.2f}s")
            
            val_metrics = self.evaluate(model, X_val, y_val, name)
            test_metrics = self.evaluate(model, X_test, y_test, name)
            
            logging.info(f"Validation F1 Macro: {val_metrics['f1_macro']:.4f}")
            logging.info(f"Test F1 Macro: {test_metrics['f1_macro']:.4f}")
            
            # Save model
            joblib.dump(model, os.path.join(self.models_dir, f"{name.lower()}.joblib"))
            
            self.log_experiment(name, train_time, test_metrics, str(model.get_params()))
            
        logging.info("Training pipeline finished.")

if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.run_all()
