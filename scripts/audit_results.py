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
    
    preprocessor = NIDSPreprocessor()
    preprocessor.load(os.path.join(models_dir, "preprocessor.joblib"))
    df = preprocessor.load_data(dataset_path)
    
    # --- Split 1: Supervised (No Holdout) ---
    logging.info("\nPreparing data for Supervised Audit (No Holdout)...")
    _, _, X_test_sup_raw, _, _, y_test_sup = prepare_data(df, holdout_attack=None)
    X_test_sup = preprocessor.transform(X_test_sup_raw)
    
    # --- Split 2: Unsupervised (Holdout DoS Hulk) ---
    logging.info("\nPreparing data for Unsupervised Audit (Holdout DoS Hulk)...")
    _, _, X_test_unsup_raw, _, _, y_test_unsup = prepare_data(df, holdout_attack="DoS Hulk")
    X_test_unsup = preprocessor.transform(X_test_unsup_raw)
    
    # Load Label Encoder (fitted during train_models without holdout)
    le = joblib.load(os.path.join(models_dir, "label_encoder.joblib"))
    y_test_sup_encoded = le.transform(y_test_sup)
    y_test_unsup_encoded = le.transform(y_test_unsup)
    
    classes = le.classes_
    n_classes = len(classes)
    
    def evaluate_supervised(model_name, y_true, y_pred, y_prob=None):
        logging.info(f"\n=======================================================")
        logging.info(f" SUPERVISED AUDIT: {model_name}")
        logging.info(f"=======================================================")
        
        logging.info(f"VERIFICATION: Evaluated on final untouched test set (N={len(y_true)})")
        
        report = classification_report(y_true, y_pred, target_names=classes, digits=4, zero_division=0)
        logging.info("\nClassification Report:\n" + report)
        
        cm = confusion_matrix(y_true, y_pred)
        logging.info("\nConfusion Matrix:\n" + str(cm))
        
        logging.info("\nPer-Class FPR and FNR:")
        for i, class_name in enumerate(classes):
            tp = cm[i, i]
            fn = np.sum(cm[i, :]) - tp
            fp = np.sum(cm[:, i]) - tp
            tn = np.sum(cm) - tp - fp - fn
            
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
            
            logging.info(f"  {class_name:<25} FPR: {fpr:.6f}   FNR: {fnr:.6f}")
            
        if y_prob is not None:
            try:
                roc_auc = roc_auc_score(y_true, y_prob, multi_class='ovr')
                logging.info(f"\nROC-AUC (Macro): {roc_auc:.6f}")
                
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
        y_pred = rf.predict(X_test_sup)
        y_prob = rf.predict_proba(X_test_sup)
        evaluate_supervised("Random Forest", y_test_sup_encoded, y_pred, y_prob)
    else:
        logging.warning("Random Forest model not found.")
        
    # 3. Evaluate XGBoost
    xgb_path = os.path.join(models_dir, "xgboost.joblib")
    if os.path.exists(xgb_path):
        xgb = joblib.load(xgb_path)
        y_pred = xgb.predict(X_test_sup)
        y_prob = xgb.predict_proba(X_test_sup)
        evaluate_supervised("XGBoost", y_test_sup_encoded, y_pred, y_prob)
    else:
        logging.warning("XGBoost model not found.")
        
    # 4. Evaluate PyTorch MLP
    mlp_path = os.path.join(models_dir, "dl_model", "mlp_best.pt")
    if os.path.exists(mlp_path):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_dim = X_test_sup.shape[1]
        mlp = NIDS_MLP(input_dim, n_classes).to(device)
        mlp.load_state_dict(torch.load(mlp_path, map_location=device, weights_only=True))
        mlp.eval()
        
        X_tensor = torch.FloatTensor(X_test_sup).to(device)
        with torch.no_grad():
            outputs = mlp(X_tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            
        evaluate_supervised("PyTorch MLP", y_test_sup_encoded, preds, probs)
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
        input_dim = X_test_unsup.shape[1]
        ae = Autoencoder(input_dim).to(device)
        ae.load_state_dict(torch.load(ae_path, map_location=device, weights_only=True))
        ae.eval()
        
        # We also need to evaluate using the RECONSTRUCTED isolated preprocessor
        # that was used during training but never saved.
        logging.info("\n--- Evaluation A: Using saved Supervised Preprocessor (Production Reality) ---")
        X_tensor_prod = torch.FloatTensor(X_test_unsup).to(device)
        with torch.no_grad():
            reconstructed_prod = ae(X_tensor_prod)
            mse_prod = torch.mean((X_tensor_prod - reconstructed_prod) ** 2, dim=1).cpu().numpy()
            
        y_pred_anomaly_prod = (mse_prod > threshold).astype(int)
        
        # Reconstruct isolated training preprocessor
        logging.info("\n--- Evaluation B: Using reconstructed Isolated Preprocessor (Training Reality) ---")
        X_train_unsup_raw, _, _, _, _, _ = prepare_data(df, holdout_attack="DoS Hulk")
        isolated_preprocessor = NIDSPreprocessor()
        isolated_preprocessor.fit_transform(X_train_unsup_raw)
        X_test_unsup_isolated = isolated_preprocessor.transform(X_test_unsup_raw)
        
        X_tensor_iso = torch.FloatTensor(X_test_unsup_isolated).to(device)
        with torch.no_grad():
            reconstructed_iso = ae(X_tensor_iso)
            mse_iso = torch.mean((X_tensor_iso - reconstructed_iso) ** 2, dim=1).cpu().numpy()
            
        y_pred_anomaly_iso = (mse_iso > threshold).astype(int)
        
        benign_idx = -1
        for i, c in enumerate(classes):
            if "BENIGN" in c.upper() or "NORMAL" in c.upper():
                benign_idx = i
                break
                
        if benign_idx != -1:
            benign_mask = (y_test_unsup_encoded == benign_idx)
            benign_support = np.sum(benign_mask)
            
            fp_prod = np.sum(y_pred_anomaly_prod[benign_mask] == 1)
            fpr_prod = fp_prod / benign_support if benign_support > 0 else 0.0
            
            fp_iso = np.sum(y_pred_anomaly_iso[benign_mask] == 1)
            fpr_iso = fp_iso / benign_support if benign_support > 0 else 0.0
            
            holdout_idx = -1
            if "DoS Hulk" in classes:
                holdout_idx = list(classes).index("DoS Hulk")
                
            if holdout_idx != -1:
                holdout_mask = (y_test_unsup_encoded == holdout_idx)
                holdout_support = np.sum(holdout_mask)
                
                holdout_tp_prod = np.sum(y_pred_anomaly_prod[holdout_mask] == 1)
                holdout_dr_prod = holdout_tp_prod / holdout_support if holdout_support > 0 else 0.0
                
                holdout_tp_iso = np.sum(y_pred_anomaly_iso[holdout_mask] == 1)
                holdout_dr_iso = holdout_tp_iso / holdout_support if holdout_support > 0 else 0.0
                
                logging.info(f"Production (Supervised Scaler) -> Benign Test FPR: {fpr_prod:.6f} | DoS Hulk DR: {holdout_dr_prod:.6f}")
                logging.info(f"Training (Isolated Scaler)   -> Benign Test FPR: {fpr_iso:.6f} | DoS Hulk DR: {holdout_dr_iso:.6f}")
            else:
                logging.warning("DoS Hulk class not found.")
        else:
            logging.warning("Benign class not found for Autoencoder audit.")
            
    else:
        logging.warning("Autoencoder model or metadata not found.")
        
if __name__ == "__main__":
    audit_results()
