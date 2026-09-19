import os
import sys
import glob
import logging
import pandas as pd
from huggingface_hub import snapshot_download

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

REPO_ID = "c01dsnap/CIC-IDS2017"
RAW_DIR = "data/raw/cicids2017"
OUTPUT_FILE = "data/raw/dataset.csv"

def fetch_and_merge():
    # 1. Download from Hugging Face (Idempotent)
    csv_files = glob.glob(os.path.join(RAW_DIR, "*.csv"))
    if len(csv_files) >= 8:
        logging.info("[1/3] Raw CSV files ready (Already downloaded)")
    else:
        logging.info(f"Downloading CIC-IDS2017 CSVs from Hugging Face ({REPO_ID})...")
        try:
            snapshot_download(
                repo_id=REPO_ID,
                repo_type="dataset",
                allow_patterns="*.csv",
                local_dir=RAW_DIR,
                local_dir_use_symlinks=False
            )
            csv_files = glob.glob(os.path.join(RAW_DIR, "*.csv"))
            logging.info("[1/3] Raw CSV files ready")
        except Exception as e:
            logging.error(f"Download failed: {e}")
            logging.error("ERROR: Failed to download the required dataset files. Cannot proceed.")
            sys.exit(1)

    if not csv_files:
        logging.error(f"No CSV files found in {RAW_DIR}")
        sys.exit(1)

    logging.info(f"[2/3] Processing {len(csv_files)} CSV files...")

    all_dfs = []
    base_columns = None
    total_rows = 0
    invalid_timestamps = 0

    # Process each CSV
    for filepath in csv_files:
        filename = os.path.basename(filepath)
        
        try:
            df = pd.read_csv(filepath, encoding="cp1252", low_memory=False)
        except Exception as e:
            logging.error(f"Failed to read {filename}: {e}")
            sys.exit(1)
            
        rows_before = len(df)
        
        # Clean column names (strip whitespace)
        df.columns = df.columns.str.strip()
        
        # Validate schema compatibility
        if base_columns is None:
            base_columns = list(df.columns)
        else:
            if list(df.columns) != base_columns:
                logging.error(f"Schema mismatch in {filename}!")
                sys.exit(1)
                
        # Remove embedded repeated headers
        df = df[df['Destination Port'] != 'Destination Port'].copy()
        
        # Parse Timestamp
        if 'Timestamp' in df.columns:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], format="mixed", errors="coerce")
            na_ts = df['Timestamp'].isna().sum()
            invalid_timestamps += na_ts
        else:
            logging.warning(f"'Timestamp' column missing in {filename}! Proceeding without it.")
            
        if 'Label' not in df.columns:
            logging.error(f"'Label' column missing in {filename}!")
            sys.exit(1)
            
        rows_after = len(df)
        total_rows += rows_after
        logging.info(f"  -> {filename}: {rows_after:,} rows")
        
        all_dfs.append(df)

    # 4. Merge all dataframes
    logging.info("[3/3] Creating merged dataset...")
    merged_df = pd.concat(all_dfs, ignore_index=True)
    
    # 5. Sort chronologically if Timestamp exists
    if 'Timestamp' in merged_df.columns:
        logging.info("[3/3] Creating merged dataset...")
        merged_df = merged_df.sort_values(by="Timestamp").reset_index(drop=True)
    else:
        logging.info("[3/3] Creating merged dataset (Timestamp missing, skipping sort)...")
    
    # 6. Save to CSV
    merged_df.to_csv(OUTPUT_FILE, index=False)
    
    # 7. Print EDA Metrics
    logging.info("=========================================")
    logging.info("      DATASET INGESTION COMPLETE         ")
    logging.info("=========================================")
    logging.info(f"Source Files Processed: {len(csv_files)}")
    logging.info(f"Total Rows:             {len(merged_df):,}")
    logging.info(f"Number of Columns:      {len(merged_df.columns)}")
    
    if 'Timestamp' in merged_df.columns:
        ts_min = merged_df['Timestamp'].min()
        ts_max = merged_df['Timestamp'].max()
        logging.info(f"Timestamp Range:        {ts_min} to {ts_max}")
        logging.info(f"Invalid Timestamps:     {invalid_timestamps:,}")
    else:
        logging.info("Timestamp Range:        N/A (Column missing)")
        logging.info("Invalid Timestamps:     N/A")
    
    duplicates = merged_df.duplicated().sum()
    logging.info(f"Exact Duplicate Rows:   {duplicates:,}")
    
    missing_vals = merged_df.isna().sum().sum()
    logging.info(f"Total Missing Values:   {missing_vals:,}")
    
    file_size_gb = os.path.getsize(OUTPUT_FILE) / (1024**3)
    logging.info(f"Final Dataset Size:     {file_size_gb:.2f} GB")
    
    logging.info("--- Label Distribution ---")
    dist = merged_df['Label'].value_counts()
    for label, count in dist.items():
        logging.info(f"  {label:<25}: {count:,}")
    logging.info("=========================================")

if __name__ == "__main__":
    fetch_and_merge()
