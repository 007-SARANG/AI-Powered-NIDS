# Model Evaluation & Experiment Tracking

## Dual Evaluation Strategy

This project enforces strict academic rigor by decoupling evaluation into two distinct pipelines based on data availability.

### 1. Feature-Classification Pipeline (Current)
This evaluation utilizes the Hugging Face `c01dsnap/CIC-IDS2017` feature-only CSV dataset. 
- **Constraint:** This dataset contains NO `Timestamp` or session identifiers.
- **Methodology:** Global exact duplicates are dropped. A **Stratified Random Split** (70/15/15) is used.
- **Limitation:** Makes no claims regarding chronological session boundaries. 
- **Unseen-Attack Evaluation:** A specific attack (e.g., DoS Hulk) is dropped entirely from Train/Validation sets and appended exclusively to the Test set to test zero-day generalizability.

### 2. Temporal / Scenario-Aware Pipeline (Future / Separate)
- **Constraint:** Requires raw PCAP processing or the original UNB ZIP that includes Timestamp and Flow IDs.
- **Methodology:** Sorts chronologically, tracks session boundaries, and guarantees zero temporal crossover.

## Supervised Model Comparison (Feature-Classification)

| Model | F1 Macro (Test) | Precision | Recall | ROC-AUC | Latency (ms) |
|---|---|---|---|---|---|
| Logistic Regression (Baseline) | *Pending A100 Run* | *Pending* | *Pending* | *Pending* | *Pending* |
| Random Forest | *Pending A100 Run* | *Pending* | *Pending* | *Pending* | *Pending* |
| XGBoost | *Pending A100 Run* | *Pending* | *Pending* | *Pending* | *Pending* |
| PyTorch MLP | *Pending A100 Run* | *Pending* | *Pending* | *Pending* | *Pending* |

*Note: Metrics will be populated natively upon execution on the A100 environment.*

## Anomaly Detection (Autoencoder)
The anomaly threshold is evaluated strictly under the following unseen-attack methodology:
- **Training:** Trained exclusively on benign training traffic.
- **Threshold Selection:** Calculated using the 95th percentile of reconstruction error on the benign validation set.
- **Evaluation (FPR):** False Positive Rate calculated on untouched benign test traffic.
- **Evaluation (Recall):** Detection rate calculated against the held-out/unseen attack class in the test set.

- **Calculated Threshold**: *Pending A100 Run*
- **Benign FPR**: *Pending A100 Run*
- **Unseen Attack Recall**: *Pending A100 Run*

## Feature Ablation Study

| Setup | F1 Macro | Minority Recall |
|---|---|---|
| A: Raw Features | *Pending A100 Run* | *Pending* |
| B: Engineered (Cleaned/Scaled) | *Pending A100 Run* | *Pending* |
| C: Engineered + Class Balancing | *Pending A100 Run* | *Pending* |
| D: Setup C + Tuned XGBoost | *Pending A100 Run* | *Pending* |

## CPU vs A100 GPU Experiment (PyTorch)

| Device | Batch Size | Epoch Time | Total Train Time | Val F1 Macro |
|---|---|---|---|---|
| CPU (Local) | 256 | *Pending* | *Pending* | *Pending* |
| A100 | 256 | *Pending* | *Pending* | *Pending* |

## Explanation of False Positives & Negatives
In an IDS context:
- **False Positives (Type I)**: Flagging benign traffic as an attack. This causes alert fatigue for SOC analysts.
- **False Negatives (Type II)**: Missing actual malicious traffic. This is critical as it results in a successful breach.
Because False Negatives are devastating, the hybrid model architecture intentionally trades a slight increase in False Positives (via the Anomaly detector) to minimize False Negatives.
