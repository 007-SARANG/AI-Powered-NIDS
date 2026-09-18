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
    # 1. Download from Hugging Face
    logging.info(f"Downloading CIC-IDS2017 CSVs from Hugging Face ({REPO_ID})...")
    try:
        snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            allow_patterns="*.csv",
            local_dir=RAW_DIR,
            local_dir_use_symlinks=False
        )
    except Exception as e:
        logging.error(f"Download failed: {e}")
        logging.error("ERROR: Failed to download the required dataset files. Cannot proceed.")
        sys.exit(1)

    # 2. Find CSVs
    csv_files = glob.glob(os.path.join(RAW_DIR, "*.csv"))
    if not csv_files:
        logging.error(f"No CSV files found in {RAW_DIR}")
        sys.exit(1)

    logging.info(f"Found {len(csv_files)} source files. Processing...")

    all_dfs = []
    base_columns = None
    
    total_rows = 0

    # 3. Process each CSV
    for filepath in csv_files:
        filename = os.path.basename(filepath)
        logging.info(f"Loading {filename}...")
        
        # Read the CSV (encoding handles some weird characters in CIC-IDS2017)
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
            logging.info(f"Base schema established with {len(base_columns)} columns.")
        else:
            if list(df.columns) != base_columns:
                logging.error(f"Schema mismatch in {filename}!")
                logging.error(f"Expected: {base_columns}")
                logging.error(f"Found: {list(df.columns)}")
                sys.exit(1)
                
        # Remove embedded repeated headers
        # CIC-IDS2017 sometimes concatenates files poorly, leaving "Destination Port" in the data rows.
        df = df[df['Destination Port'] != 'Destination Port']
        
        # Parse Timestamp
        if 'Timestamp' in df.columns:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], format="mixed", errors="coerce")
        else:
            logging.error(f"'Timestamp' column missing in {filename}!")
            sys.exit(1)
            
        if 'Label' not in df.columns:
            logging.error(f"'Label' column missing in {filename}!")
            sys.exit(1)
            
        rows_after = len(df)
        total_rows += rows_after
        logging.info(f"  -> Rows: {rows_before} (Cleaned: {rows_after})")
        
        all_dfs.append(df)

    # 4. Merge all dataframes
    logging.info("Concatenating all flow files...")
    merged_df = pd.concat(all_dfs, ignore_index=True)
    
    # 5. Sort chronologically
    logging.info("Sorting merged dataset chronologically by Timestamp...")
    merged_df = merged_df.sort_values(by="Timestamp").reset_index(drop=True)
    
    # 6. Save to CSV
    logging.info(f"Saving merged dataset to {OUTPUT_FILE}...")
    merged_df.to_csv(OUTPUT_FILE, index=False)
    
    # 7. Print EDA Metrics
    logging.info("=========================================")
    logging.info("      DATASET INGESTION COMPLETE         ")
    logging.info("=========================================")
    logging.info(f"Source Files Processed: {len(csv_files)}")
    logging.info(f"Total Rows:             {len(merged_df):,}")
    
    ts_min = merged_df['Timestamp'].min()
    ts_max = merged_df['Timestamp'].max()
    logging.info(f"Timestamp Range:        {ts_min} to {ts_max}")
    
    duplicates = merged_df.duplicated().sum()
    logging.info(f"Exact Duplicate Rows:   {duplicates:,}")
    
    missing_vals = merged_df.isna().sum().sum()
    logging.info(f"Total Missing Values:   {missing_vals:,}")
    
    logging.info("--- Class Distribution ---")
    dist = merged_df['Label'].value_counts()
    for label, count in dist.items():
        logging.info(f"  {label:<25}: {count:,}")
    logging.info("=========================================")

if __name__ == "__main__":
    fetch_and_merge()
