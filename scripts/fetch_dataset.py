import os
import sys
import pandas as pd
import numpy as np
import requests
import zipfile
import logging
from io import BytesIO

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

URL = "https://cicresearch.ca/CICDataset/CIC-IDS-2017/Dataset/MachineLearningCSV.zip"
TARGET_DIR = "data/raw"
OUTPUT_FILE = "dataset.csv"

def generate_synthetic_data(num_samples=10000):
    """Fallback method if download fails to allow pipeline development."""
    logging.warning("Generating synthetic data as fallback...")
    
    columns = [
        "Destination Port", "Flow Duration", "Total Fwd Packets", 
        "Total Backward Packets", "Total Length of Fwd Packets", 
        "Total Length of Bwd Packets", "Fwd Packet Length Max", 
        "Fwd Packet Length Min", "Fwd Packet Length Mean", 
        "Fwd Packet Length Std", "Bwd Packet Length Max", 
        "Bwd Packet Length Min", "Bwd Packet Length Mean", 
        "Bwd Packet Length Std", "Flow Bytes/s", "Flow Packets/s", 
        "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
        "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", 
        "Fwd IAT Min", "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std", 
        "Bwd IAT Max", "Bwd IAT Min", "Fwd PSH Flags", "Bwd PSH Flags", 
        "Fwd URG Flags", "Bwd URG Flags", "Fwd Header Length", 
        "Bwd Header Length", "Fwd Packets/s", "Bwd Packets/s", 
        "Min Packet Length", "Max Packet Length", "Packet Length Mean", 
        "Packet Length Std", "Packet Length Variance", "FIN Flag Count", 
        "SYN Flag Count", "RST Flag Count", "PSH Flag Count", 
        "ACK Flag Count", "URG Flag Count", "CWE Flag Count", 
        "ECE Flag Count", "Down/Up Ratio", "Average Packet Size", 
        "Avg Fwd Segment Size", "Avg Bwd Segment Size", 
        "Fwd Header Length.1", "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", 
        "Fwd Avg Bulk Rate", "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", 
        "Bwd Avg Bulk Rate", "Subflow Fwd Packets", "Subflow Fwd Bytes", 
        "Subflow Bwd Packets", "Subflow Bwd Bytes", "Init_Win_bytes_forward", 
        "Init_Win_bytes_backward", "act_data_pkt_fwd", "min_seg_size_forward", 
        "Active Mean", "Active Std", "Active Max", "Active Min", 
        "Idle Mean", "Idle Std", "Idle Max", "Idle Min", "Label"
    ]
    
    np.random.seed(42)
    df = pd.DataFrame(np.random.randn(num_samples, len(columns) - 1), columns=columns[:-1])
    
    # Introduce some realistic ranges
    df["Destination Port"] = np.random.choice([80, 443, 22, 53, 8080], size=num_samples)
    df["Flow Duration"] = np.random.randint(100, 10000000, size=num_samples)
    
    labels = ["BENIGN"] * int(num_samples * 0.8) + ["DoS Hulk"] * int(num_samples * 0.1) + ["PortScan"] * int(num_samples * 0.05) + ["DDoS"] * int(num_samples * 0.05)
    df["Label"] = labels
    
    # Create some missing and inf values to test preprocessing
    df.loc[10:20, "Flow Bytes/s"] = np.nan
    df.loc[30:40, "Flow Packets/s"] = np.inf
    
    return df

def fetch_data(synthetic=False):
    os.makedirs(TARGET_DIR, exist_ok=True)
    target_path = os.path.join(TARGET_DIR, OUTPUT_FILE)
    
    if synthetic:
        logging.info("Explicitly generating synthetic data for development/testing...")
        df = generate_synthetic_data()
        df.to_csv(target_path, index=False)
        logging.info(f"Saved synthetic dataset to {target_path}")
        return
        
    try:
        logging.info(f"Attempting to download dataset from {URL}...")
        response = requests.get(URL, stream=True, timeout=10)
        response.raise_for_status()
        
        logging.info("Download successful. Extracting...")
        with zipfile.ZipFile(BytesIO(response.content)) as z:
            csv_files = [f for f in z.namelist() if f.endswith('.csv')]
            if not csv_files:
                raise ValueError("No CSV files found in the archive.")
            
            target_csv = csv_files[0]
            for f in csv_files:
                if "DDos" in f or "PortScan" in f:
                    target_csv = f
                    break
                    
            logging.info(f"Extracting {target_csv}...")
            with z.open(target_csv) as f_in:
                df = pd.read_csv(f_in, encoding="cp1252")
                
            logging.info(f"Loaded dataframe of shape {df.shape}")
            df.columns = df.columns.str.strip()
            
            if len(df) > 50000:
                logging.info("Sampling 50,000 rows for rapid development...")
                df = df.sample(n=50000, random_state=42, weights=df.groupby('Label')['Label'].transform('count'))
                
            df.to_csv(target_path, index=False)
            logging.info(f"Saved to {target_path}")
            
    except Exception as e:
        logging.error(f"Failed to fetch real data: {e}")
        logging.error("ERROR: Silent synthetic fallback is disabled. Please ensure the dataset is accessible, or use --synthetic for development.")
        sys.exit(1)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fetch or generate NIDS dataset")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic data for testing instead of downloading")
    args = parser.parse_args()
    
    fetch_data(synthetic=args.synthetic)
