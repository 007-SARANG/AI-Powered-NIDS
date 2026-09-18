import os
import time
import json
import logging
import joblib
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from ml.preprocessing.pipeline import NIDSPreprocessor, split_data

logging.basicConfig(level=logging.INFO, format="%(message)s")

class NIDS_MLP(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dims=[256, 128, 64], dropout_rate=0.3):
        super(NIDS_MLP, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = h_dim
            
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

class DeepLearningTrainer:
    def __init__(self, data_path="data/raw/dataset.csv", models_dir="models/dl_model/"):
        self.data_path = data_path
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        
        # Check for CUDA
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if self.device.type == "cuda":
            logging.info(f"Training on GPU: {torch.cuda.get_device_name(0)}")
            logging.info(f"CUDA Available: True")
        else:
            logging.info("Training on CPU (CUDA not available)")
            
    def load_data(self):
        preprocessor = NIDSPreprocessor()
        df = preprocessor.load_data(self.data_path)
        
        le_path = "models/label_encoder.joblib"
        if os.path.exists(le_path):
            self.le = joblib.load(le_path)
            y_encoded = self.le.transform(df['Label'].values)
        else:
            from sklearn.preprocessing import LabelEncoder
            self.le = LabelEncoder()
            y_encoded = self.le.fit_transform(df['Label'].values)
            joblib.dump(self.le, le_path)
            
        df['Label'] = y_encoded
        self.n_classes = len(self.le.classes_)
        
        from ml.preprocessing.pipeline import prepare_data
        X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test = prepare_data(df)
        
        X_train = preprocessor.fit_transform(X_train_raw)
        X_val = preprocessor.transform(X_val_raw)
        X_test = preprocessor.transform(X_test_raw)
        
        self.train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train.values))
        self.val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val.values))
        self.test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test.values))
        self.input_dim = X_train.shape[1]
        
    def train(self, batch_size=256, epochs=50, lr=0.001, patience=5):
        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(self.val_dataset, batch_size=batch_size)
        
        model = NIDS_MLP(self.input_dim, self.n_classes).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        scaler = torch.amp.GradScaler('cuda' if self.device.type == 'cuda' else 'cpu') if self.device.type == 'cuda' else None
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_path = os.path.join(self.models_dir, "mlp_best.pt")
        
        start_train_time = time.time()
        
        for epoch in range(epochs):
            epoch_start = time.time()
            model.train()
            train_loss = 0.0
            
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                optimizer.zero_grad()
                
                if scaler:
                    with torch.amp.autocast('cuda'):
                        outputs = model(X_batch)
                        loss = criterion(outputs, y_batch)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    loss.backward()
                    optimizer.step()
                    
                train_loss += loss.item() * X_batch.size(0)
                
            train_loss /= len(train_loader.dataset)
            
            model.eval()
            val_loss = 0.0
            y_true, y_pred = [], []
            
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    val_loss += loss.item() * X_batch.size(0)
                    
                    _, preds = torch.max(outputs, 1)
                    y_true.extend(y_batch.cpu().numpy())
                    y_pred.extend(preds.cpu().numpy())
                    
            val_loss /= len(val_loader.dataset)
            val_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
            epoch_time = time.time() - epoch_start
            
            logging.info(f"Epoch {epoch+1}/{epochs} ({epoch_time:.2f}s) - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val F1: {val_f1:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_model_path)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logging.info(f"Early stopping triggered at epoch {epoch+1}")
                    break
                    
        train_time = time.time() - start_train_time
        logging.info(f"Training completed in {train_time:.2f}s")
        
        # Evaluate on Test
        model.load_state_dict(torch.load(best_model_path))
        self.evaluate(model, train_time)
        
    def evaluate(self, model, train_time):
        test_loader = DataLoader(self.test_dataset, batch_size=256)
        model.eval()
        
        y_true, y_pred, y_prob = [], [], []
        start_inf = time.time()
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(self.device)
                outputs = model(X_batch)
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                
                y_true.extend(y_batch.numpy())
                y_pred.extend(preds.cpu().numpy())
                y_prob.extend(probs.cpu().numpy())
                
        inf_time = (time.time() - start_inf) / len(self.test_dataset)
        
        try:
            if self.n_classes > 2:
                roc_auc = roc_auc_score(y_true, y_prob, multi_class='ovr')
            else:
                roc_auc = roc_auc_score(y_true, [p[1] for p in y_prob])
        except ValueError:
            roc_auc = None
            
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1_macro": float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
            "roc_auc": float(roc_auc) if roc_auc else None,
            "inference_latency_ms": inf_time * 1000
        }
        
        logging.info(f"Test Metrics: {metrics}")
        
        # Save metadata
        metadata = {
            "model_name": "PyTorch_MLP",
            "device": self.device.type,
            "training_time_s": train_time,
            "metrics": metrics,
            "date": datetime.now().isoformat()
        }
        with open(os.path.join(self.models_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=4)

if __name__ == "__main__":
    trainer = DeepLearningTrainer()
    trainer.load_data()
    trainer.train()
