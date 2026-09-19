import pandas as pd
import numpy as np
import logging
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

logging.basicConfig(level=logging.INFO, format="%(message)s")

class NIDSPreprocessor:
    def __init__(self, target_col='Label', drop_dest_port=False):
        self.target_col = target_col
        self.drop_dest_port = drop_dest_port
        self.pipeline = None
        self.features = None
        # CIC-IDS2017 typically contains these leaky/metadata columns if they exist
        self.leakage_cols = ['Flow ID', 'Source IP', 'Destination IP', 'Timestamp', 'SimillarHTTP']
        if self.drop_dest_port:
            self.leakage_cols.append('Destination Port')
        
    def load_data(self, filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset not found at {filepath}")
            
        logging.info(f"Loading data from {filepath}...")
        df = pd.read_csv(filepath)
        
        # Clean column names
        df.columns = df.columns.str.strip()
            
        # Replace inf with nan so imputer can handle it
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        return df
        
    def fit_transform(self, X):
        X_clean = X.copy()
        drop_cols = [c for c in self.leakage_cols if c in X_clean.columns]
        if drop_cols:
            logging.info(f"Dropping leaky/metadata columns in training: {drop_cols}")
            X_clean.drop(columns=drop_cols, inplace=True)
            
        self.features = X_clean.columns.tolist()
        
        num_cols = X_clean.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = X_clean.select_dtypes(exclude=[np.number]).columns.tolist()
        
        logging.info(f"Numerical features: {len(num_cols)}, Categorical features: {len(cat_cols)}")
        
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, num_cols),
                ('cat', categorical_transformer, cat_cols)
            ])
            
        self.pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
        
        logging.info("Fitting preprocessing pipeline...")
        X_processed = self.pipeline.fit_transform(X_clean)
        return X_processed
        
    def transform(self, df):
        if self.pipeline is None:
            raise ValueError("Pipeline not fitted yet!")
            
        df_clean = df.copy()
        if self.target_col in df_clean.columns:
            df_clean = df_clean.drop(columns=[self.target_col])
            
        drop_cols = [c for c in self.leakage_cols if c in df_clean.columns]
        if drop_cols:
            df_clean.drop(columns=drop_cols, inplace=True)
            
        df_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        missing_cols = set(self.features) - set(df_clean.columns)
        if missing_cols:
            raise ValueError(f"Missing columns in input data: {missing_cols}")
            
        df_clean = df_clean[self.features]
        return self.pipeline.transform(df_clean)
        
    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({'pipeline': self.pipeline, 'features': self.features}, filepath)
        logging.info(f"Pipeline saved to {filepath}")
        
    def load(self, filepath):
        data = joblib.load(filepath)
        self.pipeline = data['pipeline']
        self.features = data['features']
        logging.info(f"Pipeline loaded from {filepath}")
        
def prepare_data(df, target_col='Label', holdout_attack=None, test_size=0.15, val_size=0.15):
    """
    Perform a Stratified Random Split of the feature dataset into train, val, test.
    Drops exact duplicates globally prior to splitting to prevent train/test crossover.
    Optionally holds out a specific attack class entirely for the test set.
    """
    logging.info(f"Initial dataset shape: {df.shape}")
    
    # 1. Global Exact Duplicate Removal
    dups = df.duplicated().sum()
    if dups > 0:
        logging.info(f"Dropping {dups} exact duplicate rows to prevent train/test leakage...")
        df = df.drop_duplicates().reset_index(drop=True)
        logging.info(f"New dataset shape: {df.shape}")
        
    # 2. Holdout Attack Extraction
    if holdout_attack:
        logging.info(f"Holding out attack '{holdout_attack}' for unseen-attack evaluation.")
        df_holdout = df[df[target_col] == holdout_attack]
        df_train_val_test = df[df[target_col] != holdout_attack]
    else:
        df_holdout = pd.DataFrame(columns=df.columns)
        df_train_val_test = df
        
    y = df_train_val_test[target_col]
    X = df_train_val_test.drop(columns=[target_col])
    
    # 3. Stratified Split (Train / Val / Test)
    # First split off the test set
    val_test_ratio = val_size + test_size
    test_ratio_of_valtest = test_size / val_test_ratio
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=val_test_ratio, stratify=y, random_state=42
    )
    
    # Then split val and test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=test_ratio_of_valtest, stratify=y_temp, random_state=42
    )
    
    # 4. Append holdout back to the test set
    if not df_holdout.empty:
        X_test = pd.concat([X_test, df_holdout.drop(columns=[target_col])])
        y_test = pd.concat([y_test, df_holdout[target_col]])
        
    logging.info(f"Train set: {X_train.shape[0]} samples")
    logging.info(f"Val set:   {X_val.shape[0]} samples")
    logging.info(f"Test set:  {X_test.shape[0]} samples")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

if __name__ == "__main__":
    dataset_path = "data/raw/dataset.csv"
    if os.path.exists(dataset_path):
        preprocessor = NIDSPreprocessor()
        df = preprocessor.load_data(dataset_path)
        
        # Split first!
        X_train, X_val, X_test, y_train, y_val, y_test = prepare_data(df)
        
        # Fit on train only
        X_train_processed = preprocessor.fit_transform(X_train)
        
        # Transform val/test
        X_val_processed = preprocessor.transform(X_val)
        X_test_processed = preprocessor.transform(X_test)
        
        preprocessor.save("models/preprocessor.joblib")
    else:
        logging.warning(f"{dataset_path} not found. Skip testing.")
