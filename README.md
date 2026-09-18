# AI-Powered Network Intrusion Detection & Threat Monitoring Platform

## Project Overview
This project is an end-to-end defensive cybersecurity platform designed to monitor network traffic, detect intrusions using machine learning, and provide security operations center (SOC) analysts with actionable, explainable alerts.

## Motivation (Why Intrusion Detection Matters)
Modern cyber threats often bypass traditional signature-based firewalls. An AI-powered NIDS leverages behavioral analysis and anomaly detection to catch zero-day attacks, advanced persistent threats (APTs), and sophisticated scanning activities that do not have known signatures.

## Architecture
The system consists of:
1. **Packet Ingestion Layer**: NFStream extracts flow-level features from PCAP files.
2. **Preprocessing Pipeline**: Scikit-Learn pipeline for imputation and scaling.
3. **Detection Engine**: 
   - **XGBoost Classifier**: Supervised model for known threats (DDoS, PortScan, etc.).
   - **PyTorch Autoencoder**: Unsupervised anomaly detection for zero-day threats.
4. **Explainability Module**: SHAP-based TreeExplainer provides human-readable context for ML decisions.
5. **Backend**: FastAPI providing robust REST endpoints for integration.
6. **Dashboard**: Streamlit SOC UI for monitoring and investigation.

*See `docs/architecture.md` for a detailed breakdown.*

## Dataset & Feature Engineering
We utilize the **CIC-IDS2017** dataset, processed locally.
The preprocessing handles numeric scaling (StandardScaler), missing value imputation, and class imbalance. Explicit data leakage features (like Source/Destination IPs and Timestamps) are completely removed before model training to prevent the model from artificially learning network topology rather than attack behavior.

## Models
1. **Baseline**: Logistic Regression
2. **Ensembles**: Random Forest, XGBoost
3. **Deep Learning**: PyTorch Multi-Layer Perceptron (A100 GPU accelerated)
4. **Anomaly Detection**: PyTorch Autoencoder trained solely on benign traffic.

## Hybrid Engine
The Hybrid Engine combines the Supervised and Unsupervised models.
- If XGBoost predicts an attack -> High Severity.
- If XGBoost predicts Benign but Autoencoder flags high reconstruction error -> Unknown Anomaly (Zero-Day).
- If both agree it's malicious -> Critical Severity.

## SHAP Explainability
Alerts triggered by the XGBoost engine are processed through a SHAP TreeExplainer, providing the Top 5 contributing features for every attack to assist human analysts.

## Setup & Running Locally

**1. Create virtual environment and install dependencies:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**2. Fetch the Dataset:**
```bash
python scripts/fetch_dataset.py
```

**3. Train the Models:**
```bash
PYTHONPATH=. python ml/training/train_models.py
PYTHONPATH=. python ml/training/train_dl.py
PYTHONPATH=. python ml/training/train_anomaly.py
```

**4. Start the API:**
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

**5. Start the Dashboard:**
```bash
streamlit run dashboard/app.py
```

## Running with Docker
```bash
docker-compose up --build
```

## Running PCAP Ingestion
```bash
python nids/packet/pcap_processor.py data/samples/traffic.pcap
```

## GPU & A100 Setup (Future Training)
To train on the NVIDIA A100 cluster with the full 2.5 million row CIC-IDS2017 dataset:
1. Clone this repository onto the A100 server.
2. Download and place the real CIC-IDS2017 dataset into `data/raw/dataset.csv`.
3. Open `notebooks/A100_Training.ipynb`.
4. Run the cells sequentially. The notebook will automatically utilize the A100 GPU for the PyTorch Multi-Layer Perceptron and Autoencoder.

*Note: The final performance metrics are pending and will be natively calculated and displayed within the `A100_Training.ipynb` notebook upon execution.*

## Limitations & Future Work
- **Limitations**: Offline PCAP extraction introduces latency; not built for raw inline packet dropping (IPS).
- **Future Work**: Integration with live interface sniffing (e.g., AF_PACKET), streaming ML architectures, and Kafka message queuing.

## Security Considerations
See `docs/security.md` for a complete breakdown of adversarial considerations, deployment safety, and ML limitations.
