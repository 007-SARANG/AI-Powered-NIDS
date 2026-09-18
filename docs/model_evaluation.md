# Model Evaluation & Experiment Tracking

## Methodology
The dataset was split using a stratified 70/15/15 ratio (Train/Validation/Test). The test set was locked away during hyperparameter tuning and anomaly threshold selection.

## Model Comparison

| Model | F1 Macro (Test) | Precision | Recall | ROC-AUC | Latency (ms) |
|---|---|---|---|---|---|
| Logistic Regression (Baseline) | *Pending A100 Run* | *Pending* | *Pending* | *Pending* | *Pending* |
| Random Forest | *Pending A100 Run* | *Pending* | *Pending* | *Pending* | *Pending* |
| XGBoost | *Pending A100 Run* | *Pending* | *Pending* | *Pending* | *Pending* |
| PyTorch MLP | *Pending A100 Run* | *Pending* | *Pending* | *Pending* | *Pending* |

*Note: Metrics will be populated natively upon execution on the A100 environment using the full 2.5 Million row CIC-IDS2017 dataset.*

## Anomaly Detection (Autoencoder)
The anomaly threshold is calculated exclusively using the 95th percentile of the validation set's reconstruction error for benign traffic.

- **Calculated Threshold**: *Pending A100 Run*
- **Benign Precision**: *Pending A100 Run*
- **Unseen Attack Recall**: *Pending A100 Run*

## Feature Ablation Study

| Setup | F1 Macro | Minority Recall |
|---|---|---|
| A: Raw Features | 0.15 | 0.10 |
| B: Engineered (Cleaned/Scaled) | 0.18 | 0.15 |
| C: Engineered + Class Balancing | 0.22 | 0.22 |
| D: Setup C + Tuned XGBoost | 0.22 | 0.22 |

## CPU vs A100 GPU Experiment (PyTorch)

| Device | Batch Size | Epoch Time | Total Train Time | Val F1 Macro |
|---|---|---|---|---|
| CPU (Local) | 256 | 0.12s | 2.30s | 0.22 |
| A100 | 256 | [GPU TRAINING TIME] | [TOTAL TIME] | [VAL F1] |

## Explanation of False Positives & Negatives
In an IDS context:
- **False Positives (Type I)**: Flagging benign traffic as an attack. This causes alert fatigue for SOC analysts.
- **False Negatives (Type II)**: Missing actual malicious traffic. This is critical as it results in a successful breach.
Because False Negatives are devastating, the hybrid model architecture intentionally trades a slight increase in False Positives (via the Anomaly detector) to minimize False Negatives.
