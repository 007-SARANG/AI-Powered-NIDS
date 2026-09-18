import pandas as pd
import numpy as np
import logging
import os
from collections import Counter

logging.basicConfig(level=logging.INFO, format="%(message)s")

def perform_eda(csv_path="data/raw/dataset.csv"):
    if not os.path.exists(csv_path):
        logging.error(f"Dataset not found at {csv_path}")
        return

    logging.info("="*50)
    logging.info(f"EXPLORATORY DATA ANALYSIS: {csv_path}")
    logging.info("="*50)
    
    df = pd.read_csv(csv_path)
    
    # Basic Stats
    logging.info(f"\n1. DATASET SHAPE: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Target Labels
    if 'Label' in df.columns:
        logging.info("\n2. CLASS DISTRIBUTION:")
        counts = df['Label'].value_counts()
        for label, count in counts.items():
            logging.info(f"   - {label}: {count} ({count/len(df)*100:.2f}%)")
    else:
        logging.info("\n2. TARGET LABEL: 'Label' column not found!")

    # Missing and Inf values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    
    logging.info("\n3. MISSING & INFINITE VALUES (after replacing inf with nan):")
    if len(missing) == 0:
        logging.info("   - None found.")
    else:
        for col, count in missing.items():
            logging.info(f"   - {col}: {count} missing ({count/len(df)*100:.2f}%)")
            
    # Duplicates
    duplicates = df.duplicated().sum()
    logging.info(f"\n4. DUPLICATE ROWS: {duplicates} ({duplicates/len(df)*100:.2f}%)")

    # Data Types
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    logging.info(f"\n5. FEATURE TYPES:")
    logging.info(f"   - Numerical Features: {len(num_cols)}")
    logging.info(f"   - Categorical Features: {len(cat_cols)} {cat_cols}")

    # Data Leakage & Feature Selection
    logging.info("\n6. POTENTIAL DATA LEAKAGE & EXCLUDED FEATURES:")
    leakage_candidates = ['Flow ID', 'Source IP', 'Destination IP', 'Timestamp', 'SimillarHTTP']
    excluded = []
    for col in df.columns:
        if col.strip() in leakage_candidates:
            excluded.append(col)
    
    if excluded:
        logging.info(f"   - Identified non-predictive/leaky features: {excluded}")
    else:
        logging.info("   - No obvious identity/timestamp features found in this subset.")
        
    logging.info("="*50)

if __name__ == "__main__":
    perform_eda()
