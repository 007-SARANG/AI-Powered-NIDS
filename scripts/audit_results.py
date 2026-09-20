import os
import json
import logging
import joblib
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score

from ml.preprocessing.pipeline import NIDSPreprocessor, prepare_data
from ml.training.train_dl import NIDS_MLP
from ml.training.train_anomaly import Autoencoder

logging.basicConfig(level=logging.INFO, format="%(message)s")

def audit_results(dataset_path="data/raw/dataset.csv", models_dir="models/"):
    logging.info("Starting Results Audit...")
    
    # 1. Load Preprocessor & Data
    preprocessor = NIDSPreprocessor()
    preprocessor.load(os.path.join(models_dir, "preprocessor.joblib"))
    
    df = preprocessor.load_data(dataset_path)
    # Using holdout_attack='DoS Hulk' to exactly match the training split
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data(df, holdout_attack="DoS Hulk")
    
    # Transform test set
    X_test_processed = preprocessor.transform(X_test)
    
    # Load Label Encoder
    le = joblib.load(os.path.join(models_dir, "label_encoder.joblib"))
    y_test_encoded = le.transform(y_test)
    
    classes = le.classes_
    n_classes = len(classes)
    
    def evaluate_supervised(model_name, y_true, y_pred, y_prob=None):
        logging.info(f"\n=======================================================")
        logging.info(f" SUPERVISED AUDIT: {model_name}")
        logging.info(f"=======================================================")
        
        # We explicitly verify this is the test set
        logging.info(f"VERIFICATION: Evaluated on final untouched test set (N={len(y_true)})")
        
        report = classification_report(y_true, y_pred, target_names=classes, digits=4, zero_division=0)
        logging.info("\nClassification Report:\n" + report)
        
        cm = confusion_matrix(y_true, y_pred)
        logging.info("\nConfusion Matrix:\n" + str(cm))
        
        # Calculate FPR and FNR per class
        logging.info("\nPer-Class FPR and FNR:")
        for i, class_name in enumerate(classes):
            # True Positives
            tp = cm[i, i]
            # False Negatives (sum of row i excluding tp)
            fn = np.sum(cm[i, :]) - tp
            # False Positives (sum of column i excluding tp)
            fp = np.sum(cm[:, i]) - tp
            # True Negatives (total excluding row i and column i)
            tn = np.sum(cm) - tp - fp - fn
            
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
            
            logging.info(f"  {class_name:<25} FPR: {fpr:.6f}   FNR: {fnr:.6f}")
            
        if y_prob is not None:
            try:
                # ROC-AUC
                roc_auc = roc_auc_score(y_true, y_prob, multi_class='ovr')
                logging.info(f"\nROC-AUC (Macro): {roc_auc:.6f}")
                
                # PR-AUC
                y_true_bin = np.zeros((len(y_true), n_classes))
                for i in range(len(y_true)):
                    y_true_bin[i, y_true[i]] = 1
                pr_auc = average_precision_score(y_true_bin, y_prob, average="macro")
                logging.info(f"PR-AUC (Macro): {pr_auc:.6f}")
            except Exception as e:
                logging.warning(f"Could not calculate ROC/PR AUC: {e}")
                
    # 2. Evaluate Random Forest
    rf_path = os.path.join(models_dir, "randomforest.joblib")
    if os.path.exists(rf_path):
        rf = joblib.load(rf_path)
        y_pred = rf.predict(X_test_processed)
        y_prob = rf.predict_proba(X_test_processed)
        evaluate_supervised("Random Forest", y_test_encoded, y_pred, y_prob)
    else:
        logging.warning("Random Forest model not found.")
        
    # 3. Evaluate XGBoost
    xgb_path = os.path.join(models_dir, "xgboost.joblib")
    if os.path.exists(xgb_path):
        xgb = joblib.load(xgb_path)
        y_pred = xgb.predict(X_test_processed)
        y_prob = xgb.predict_proba(X_test_processed)
        evaluate_supervised("XGBoost", y_test_encoded, y_pred, y_prob)
    else:
        logging.warning("XGBoost model not found.")
        
    # 4. Evaluate PyTorch MLP
    mlp_path = os.path.join(models_dir, "dl_model", "best_mlp.pt")
    if os.path.exists(mlp_path):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_dim = X_test_processed.shape[1]
        mlp = NIDS_MLP(input_dim, n_classes).to(device)
        mlp.load_state_dict(torch.load(mlp_path, map_location=device, weights_only=True))
        mlp.eval()
        
        X_tensor = torch.FloatTensor(X_test_processed).to(device)
        with torch.no_grad():
            outputs = mlp(X_tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            
        evaluate_supervised("PyTorch MLP", y_test_encoded, preds, probs)
    else:
        logging.warning("PyTorch MLP model not found.")
        
    # 5. Evaluate Autoencoder
    ae_path = os.path.join(models_dir, "anomaly", "autoencoder_best.pt")
    meta_path = os.path.join(models_dir, "anomaly", "anomaly_metadata.json")
    if os.path.exists(ae_path) and os.path.exists(meta_path):
        logging.info(f"\n=======================================================")
        logging.info(f" UNSUPERVISED AUDIT: Autoencoder")
        logging.info(f"=======================================================")
        
        with open(meta_path, "r") as f:
            ae_meta = json.load(f)
            
        threshold = ae_meta['threshold']
        logging.info(f"Loaded Threshold (95th percentile benign val): {threshold:.6f}")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_dim = X_test_processed.shape[1]
        ae = Autoencoder(input_dim).to(device)
        ae.load_state_dict(torch.load(ae_path, map_location=device, weights_only=True))
        ae.eval()
        
        X_tensor = torch.FloatTensor(X_test_processed).to(device)
        with torch.no_grad():
            reconstructed = ae(X_tensor)
            mse = torch.mean((X_tensor - reconstructed) ** 2, dim=1).cpu().numpy()
            
        y_pred_anomaly = (mse > threshold).astype(int)
        
        # 1 means anomaly (attack), 0 means benign
        benign_idx = -1
        for i, c in enumerate(classes):
            if "BENIGN" in c.upper() or "NORMAL" in c.upper():
                benign_idx = i
                break
                
        if benign_idx != -1:
            y_true_anomaly = (y_test_encoded != benign_idx).astype(int)
            
            # Extract Benign Test FPR
            benign_mask = (y_test_encoded == benign_idx)
            benign_support = np.sum(benign_mask)
            fp = np.sum(y_pred_anomaly[benign_mask] == 1)
            fpr = fp / benign_support if benign_support > 0 else 0.0
            
            # Extract Holdout DoS Hulk Detection Rate
            holdout_idx = -1
            if "DoS Hulk" in classes:
                holdout_idx = list(classes).index("DoS Hulk")
                
            if holdout_idx != -1:
                holdout_mask = (y_test_encoded == holdout_idx)
                holdout_support = np.sum(holdout_mask)
                holdout_tp = np.sum(y_pred_anomaly[holdout_mask] == 1)
                holdout_dr = holdout_tp / holdout_support if holdout_support > 0 else 0.0
                
                logging.info(f"Benign Test FPR: {fpr:.6f} (Support: {benign_support})")
                logging.info(f"Held-out DoS Hulk Detection Rate: {holdout_dr:.6f} (Support: {holdout_support})")
                logging.info(f"Attack Support (Total non-benign): {np.sum(~benign_mask)}")
            else:
                logging.warning("DoS Hulk class not found.")
        else:
            logging.warning("Benign class not found for Autoencoder audit.")
            
    else:
        logging.warning("Autoencoder model or metadata not found.")
        
if __name__ == "__main__":
    audit_results()
