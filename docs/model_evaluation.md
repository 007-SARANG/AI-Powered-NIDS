# Model Evaluation & Experiment Tracking

## Dual Evaluation Strategy

This project enforces strict academic rigor by decoupling evaluation into two distinct pipelines based on data availability.

### 1. Feature-Classification Pipeline (Current)
This evaluation utilizes the Hugging Face `c01dsnap/CIC-IDS2017` feature-only CSV dataset. 
- **Constraint:** This dataset contains NO `Timestamp` or session identifiers.
- **Methodology:** Global exact duplicates are dropped. A **Stratified Random Split** (70/15/15) is used.
- **Limitation:** Makes no claims regarding chronological session boundaries. 
- **Unseen-Attack Evaluation:** A specific attack (e.g., DoS Hulk) is dropped entirely from Train/Validation sets and appended exclusively to the Test set to test unseen-attack generalizability.

### 2. Temporal / Scenario-Aware Pipeline (Future / Separate)
- **Constraint:** Requires raw PCAP processing or the original UNB ZIP that includes Timestamp and Flow IDs.
- **Methodology:** Sorts chronologically, tracks session boundaries, and guarantees zero temporal crossover.

## Supervised Model Comparison (Feature-Classification)

| Model | F1 Macro (Test) | ROC-AUC (Test) | Train Time (s) | Notes |
|---|---|---|---|---|
| Logistic Regression (Baseline) | 0.5001 | *Pending* | 1300.07 | *Incomplete (LBFGS failed to converge)* |
| Random Forest | 0.8256 | *Pending* | 50.59 | |
| XGBoost | **0.8800** | *Pending* | 49.29 | |
| PyTorch MLP | 0.7093 | **0.9973** | 561.41 | |

*Note: The high ROC-AUC on the PyTorch MLP does not necessarily guarantee excellent generalization due to class imbalances. We rely on Macro-F1 for a balanced assessment.*

## Anomaly Detection (Autoencoder)
The anomaly threshold is evaluated strictly under the following unseen-attack methodology:
- **Training:** Trained exclusively on benign training traffic.
- **Threshold Selection:** Calculated using the 95th percentile of reconstruction error on the benign validation set.
- **Evaluation (FPR):** False Positive Rate calculated on untouched benign test traffic.
- **Evaluation (Recall):** Detection rate calculated against the held-out/unseen attack class in the test set.

- **Holdout Attack Used**: DoS Hulk
- **Calculated Threshold**: 0.009342
- **Benign FPR**: 0.0498 (4.98%)
- **Unseen Attack Recall**: 0.9008 (90.08% detection of true zero-day/unseen attacks)

## Destination Port Ablation Study

| Setup | F1 Macro |
|---|---|
| A: WITH Destination Port | 0.7368 |
| B: WITHOUT Destination Port | **0.7704** |

*Note: This specific ablation ensures the model relies on true traffic behavioral patterns rather than trivially memorizing specific port numbers (e.g., assuming Port 80 = Attack). Performance slightly improved without it, indicating robust behavioral generalization.*

## CPU vs A100 GPU Experiment (PyTorch)

| Device | Batch Size | Epoch Time | Total Train Time | Val F1 Macro |
|---|---|---|---|---|
| CPU (Local) | 256 | *Pending* | *Pending* | *Pending* |
| A100 | 256 | *Pending* | 561.41s | 0.6766 |

## Explanation of False Positives & Negatives
In an IDS context:
- **False Positives (Type I)**: Flagging benign traffic as an attack. This causes alert fatigue for SOC analysts.
- **False Negatives (Type II)**: Missing actual malicious traffic. This is critical as it results in a successful breach.
Because False Negatives are devastating, the hybrid model architecture intentionally trades a slight increase in False Positives (via the Anomaly detector) to minimize False Negatives.
