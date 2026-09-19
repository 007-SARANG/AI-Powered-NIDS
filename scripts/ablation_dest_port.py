import os
import logging
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score

from ml.preprocessing.pipeline import NIDSPreprocessor, prepare_data

logging.basicConfig(level=logging.INFO, format="%(message)s")

def run_ablation(dataset_path="data/raw/dataset.csv", sample_frac=0.1):
    if not os.path.exists(dataset_path):
        logging.error(f"Dataset not found at {dataset_path}")
        return

    logging.info(f"Loading dataset from {dataset_path}...")
    df = pd.read_csv(dataset_path, low_memory=False)
    
    if sample_frac < 1.0:
        logging.info(f"Sampling {sample_frac*100}% of the dataset for quick ablation...")
        # Stratified sampling to maintain class balance in the quick run
        df = df.groupby('Label', group_keys=False).apply(lambda x: x.sample(frac=sample_frac, random_state=42))
        
    # Split first
    X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test = prepare_data(df)
    
    # -------------------------------------------------------------------
    # Experiment A: Keep Destination Port
    # -------------------------------------------------------------------
    logging.info("\n=======================================================")
    logging.info(" EXPERIMENT A: WITH Destination Port (drop_dest_port=False)")
    logging.info("=======================================================")
    
    prep_A = NIDSPreprocessor(drop_dest_port=False)
    X_train_A = prep_A.fit_transform(X_train_raw)
    X_test_A = prep_A.transform(X_test_raw)
    
    rf_A = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    rf_A.fit(X_train_A, y_train)
    
    preds_A = rf_A.predict(X_test_A)
    f1_A_macro = f1_score(y_test, preds_A, average='macro')
    logging.info("\nClassification Report (With Dest Port):")
    logging.info("\n" + classification_report(y_test, preds_A))
    
    # -------------------------------------------------------------------
    # Experiment B: Drop Destination Port
    # -------------------------------------------------------------------
    logging.info("\n=======================================================")
    logging.info(" EXPERIMENT B: WITHOUT Destination Port (drop_dest_port=True)")
    logging.info("=======================================================")
    
    prep_B = NIDSPreprocessor(drop_dest_port=True)
    X_train_B = prep_B.fit_transform(X_train_raw)
    X_test_B = prep_B.transform(X_test_raw)
    
    rf_B = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    rf_B.fit(X_train_B, y_train)
    
    preds_B = rf_B.predict(X_test_B)
    f1_B_macro = f1_score(y_test, preds_B, average='macro')
    logging.info("\nClassification Report (Without Dest Port):")
    logging.info("\n" + classification_report(y_test, preds_B))
    
    # -------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------
    logging.info("\n=======================================================")
    logging.info(" ABLATION SUMMARY")
    logging.info("=======================================================")
    logging.info(f"Macro F1 Score WITH Dest Port:    {f1_A_macro:.4f}")
    logging.info(f"Macro F1 Score WITHOUT Dest Port: {f1_B_macro:.4f}")
    
    diff = f1_A_macro - f1_B_macro
    if diff > 0.05:
        logging.warning("WARNING: Significant performance drop when removing Destination Port.")
        logging.warning("This indicates the model was heavily memorizing specific ports rather than general traffic behaviors.")
    else:
        logging.info("The model generalizes well even without Destination Port.")

if __name__ == "__main__":
    run_ablation()
