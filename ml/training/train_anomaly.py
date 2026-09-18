import os
import time
import json
import logging
import joblib
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from ml.preprocessing.pipeline import NIDSPreprocessor, split_data

logging.basicConfig(level=logging.INFO, format="%(message)s")

class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class AnomalyDetectorTrainer:
    def __init__(self, data_path="data/raw/dataset.csv", models_dir="models/anomaly/"):
        self.data_path = data_path
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Anomaly Detection Training on: {self.device}")
        
    def load_data(self):
        preprocessor = NIDSPreprocessor()
        df = preprocessor.load_data(self.data_path)
        
        # We need the original labels to filter benign traffic
        original_labels = df['Label'].values
        
        # Use existing label encoder or create one
        le_path = "models/label_encoder.joblib"
        if os.path.exists(le_path):
            self.le = joblib.load(le_path)
            y_encoded = self.le.transform(df['Label'].values)
        else:
            from sklearn.preprocessing import LabelEncoder
            self.le = LabelEncoder()
            y_encoded = self.le.fit_transform(df['Label'].values)
            os.makedirs("models", exist_ok=True)
            joblib.dump(self.le, le_path)
        
        df['Label'] = y_encoded
        
        # Determine holdout attack based on argparse (we'll add this argument next)
        # For now, default to None, or we can pass it via self.holdout_attack
        holdout_encoded = None
        if hasattr(self, 'holdout_attack') and self.holdout_attack:
            try:
                holdout_encoded = self.le.transform([self.holdout_attack])[0]
            except ValueError:
                logging.warning(f"Holdout attack '{self.holdout_attack}' not found in labels.")
        
        # SPLIT FIRST
        from ml.preprocessing.pipeline import prepare_data
        X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test = prepare_data(df, holdout_attack=holdout_encoded)
        
        # FIT ON TRAIN
        X_train = preprocessor.fit_transform(X_train_raw)
        X_val = preprocessor.transform(X_val_raw)
        X_test = preprocessor.transform(X_test_raw)
        
        # y arrays
        y_train = y_train.values
        y_val = y_val.values
        y_test = y_test.values
        self.input_dim = X_train.shape[1]
        
        # Identify the BENIGN class index
        # Usually it's 'BENIGN' or 'Normal'
        benign_idx = None
        for i, class_name in enumerate(self.le.classes_):
            if "BENIGN" in class_name.upper() or "NORMAL" in class_name.upper():
                benign_idx = i
                break
                
        if benign_idx is None:
            raise ValueError(f"Could not find BENIGN class in {self.le.classes_}")
            
        logging.info(f"BENIGN class mapped to index {benign_idx}")
        
        # Filter training data to ONLY benign
        benign_mask_train = (y_train == benign_idx)
        X_train_benign = X_train[benign_mask_train]
        logging.info(f"Training Autoencoder on {X_train_benign.shape[0]} benign samples")
        
        self.train_dataset = TensorDataset(torch.FloatTensor(X_train_benign))
        self.val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
        self.test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
        self.benign_idx = benign_idx
        
    def train(self, batch_size=256, epochs=50, lr=0.001, patience=5):
        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        # Using subset of val for early stopping (only benign)
        val_loader = DataLoader(self.val_dataset, batch_size=batch_size)
        
        model = Autoencoder(self.input_dim).to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        best_loss = float('inf')
        patience_counter = 0
        best_model_path = os.path.join(self.models_dir, "autoencoder_best.pt")
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            for (X_batch,) in train_loader:
                X_batch = X_batch.to(self.device)
                optimizer.zero_grad()
                reconstructed = model(X_batch)
                loss = criterion(reconstructed, X_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * X_batch.size(0)
                
            train_loss /= len(self.train_dataset)
            
            # Validation loss (on ALL val data)
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, _ in val_loader:
                    X_batch = X_batch.to(self.device)
                    reconstructed = model(X_batch)
                    loss = criterion(reconstructed, X_batch)
                    val_loss += loss.item() * X_batch.size(0)
                    
            val_loss /= len(self.val_dataset)
            
            logging.info(f"Epoch {epoch+1}/{epochs} - Train MSE: {train_loss:.4f} - Val MSE (overall): {val_loss:.4f}")
            
            # Using train_loss for early stopping since we only want to perfectly reconstruct benign
            if train_loss < best_loss:
                best_loss = train_loss
                torch.save(model.state_dict(), best_model_path)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logging.info("Early stopping triggered")
                    break
                    
        model.load_state_dict(torch.load(best_model_path))
        self.determine_threshold(model)
        
    def determine_threshold(self, model):
        """Determine threshold using Validation set Benign traffic only"""
        model.eval()
        val_loader = DataLoader(self.val_dataset, batch_size=256)
        
        errors = []
        labels = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.device)
                reconstructed = model(X_batch)
                # MSE per sample
                mse = torch.mean((X_batch - reconstructed) ** 2, dim=1).cpu().numpy()
                errors.extend(mse)
                labels.extend(y_batch.numpy())
                
        errors = np.array(errors)
        labels = np.array(labels)
        
        # Get errors for benign val traffic
        benign_errors = errors[labels == self.benign_idx]
        
        # Set threshold at 95th percentile of benign validation errors
        threshold = float(np.percentile(benign_errors, 95))
        logging.info(f"Calculated Anomaly Threshold (95th percentile of Benign Val): {threshold:.6f}")
        
        # Evaluate on Test Set
        test_loader = DataLoader(self.test_dataset, batch_size=256)
        test_errors = []
        test_labels = []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(self.device)
                reconstructed = model(X_batch)
                mse = torch.mean((X_batch - reconstructed) ** 2, dim=1).cpu().numpy()
                test_errors.extend(mse)
                test_labels.extend(y_batch.numpy())
                
        test_errors = np.array(test_errors)
        test_labels = np.array(test_labels)
        
        # True is anomaly (malicious), False is normal (benign)
        y_true = (test_labels != self.benign_idx).astype(int)
        y_pred = (test_errors > threshold).astype(int)
        
        from sklearn.metrics import classification_report, f1_score
        rep = classification_report(y_true, y_pred, target_names=["Benign", "Anomaly"], zero_division=0)
        logging.info(f"\nAnomaly Detection Test Results:\n{rep}")
        
        # False Positive Rate on Benign Test Traffic
        benign_test_mask = (test_labels == self.benign_idx)
        fpr_benign = 0.0
        if np.sum(benign_test_mask) > 0:
            fpr_benign = np.mean(y_pred[benign_test_mask])
            logging.info(f"False Positive Rate on Benign Test Traffic: {fpr_benign:.4f}")
            
        # Detection rate on Holdout Attack
        dr_holdout = None
        if hasattr(self, 'holdout_attack') and self.holdout_attack:
            try:
                holdout_encoded = self.le.transform([self.holdout_attack])[0]
                holdout_mask = (test_labels == holdout_encoded)
                if np.sum(holdout_mask) > 0:
                    dr_holdout = float(np.mean(y_pred[holdout_mask]))
                    logging.info(f"Detection Rate on Holdout Attack ('{self.holdout_attack}'): {dr_holdout:.4f}")
                else:
                    logging.info(f"No samples of holdout attack '{self.holdout_attack}' found in test set.")
            except ValueError:
                pass
        
        metadata = {
            "model": "Autoencoder",
            "threshold": threshold,
            "test_f1_macro": float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
            "fpr_benign": float(fpr_benign),
            "dr_holdout": float(dr_holdout) if dr_holdout is not None else None,
            "date": datetime.now().isoformat()
        }
        
        with open(os.path.join(self.models_dir, "anomaly_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=4)
            
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Anomaly Detector")
    parser.add_argument("--holdout-attack", type=str, default=None, help="Holdout attack class to test unseen attack detection")
    args = parser.parse_args()
    
    trainer = AnomalyDetectorTrainer()
    trainer.holdout_attack = args.holdout_attack
    trainer.load_data()
    trainer.train()
