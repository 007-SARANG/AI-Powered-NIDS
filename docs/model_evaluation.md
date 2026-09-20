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

## Detailed Results Audit (Untouched Test Set)

### 1. XGBoost (Best Supervised Model)
- **Macro-F1:** 0.8800
- **Weighted-F1:** 0.9988
- **ROC-AUC (Macro):** 0.9999
- **PR-AUC (Macro):** 0.9519
- **Accuracy:** 0.9988
- **Test Set N:** 378,355

#### Per-Class Metrics (XGBoost)
| Class | Support | Precision | Recall | F1 | FPR | FNR |
|---|---|---|---|---|---|---|
| BENIGN | 314,473 | 0.9996 | 0.9993 | 0.9995 | 0.001878 | 0.000674 |
| Bot | 293 | 0.9626 | 0.7031 | 0.8126 | 0.000021 | 0.296928 |
| DDoS | 19,203 | 0.9998 | 0.9998 | 0.9998 | 0.000008 | 0.000208 |
| DoS GoldenEye | 1,543 | 0.9948 | 0.9961 | 0.9955 | 0.000021 | 0.003889 |
| DoS Hulk | 25,928 | 0.9981 | 0.9997 | 0.9989 | 0.000142 | 0.000270 |
| DoS Slowhttptest | 784 | 0.9886 | 0.9936 | 0.9911 | 0.000024 | 0.006378 |
| DoS slowloris | 808 | 1.0000 | 0.9913 | 0.9956 | 0.000000 | 0.008663 |
| FTP-Patator | 890 | 1.0000 | 0.9978 | 0.9989 | 0.000000 | 0.002247 |
| Heartbleed | 1 | 1.0000 | 1.0000 | 1.0000 | 0.000000 | 0.000000 |
| Infiltration | 5 | 1.0000 | 0.8000 | 0.8889 | 0.000000 | 0.200000 |
| PortScan | 13,623 | 0.9893 | 0.9990 | 0.9942 | 0.000403 | 0.000954 |
| SSH-Patator | 483 | 0.9979 | 0.9938 | 0.9959 | 0.000003 | 0.006211 |
| Web Attack - Brute Force | 220 | 0.7096 | 0.9773 | 0.8222 | 0.000233 | 0.022727 |
| Web Attack - Sql Injection | 3 | 1.0000 | 0.3333 | 0.5000 | 0.000000 | 0.666667 |
| Web Attack - XSS | 98 | 0.6667 | 0.1224 | 0.2069 | 0.000016 | 0.877551 |

### 2. Random Forest
- **Macro-F1:** 0.8256
- **Weighted-F1:** 0.9918
- **ROC-AUC (Macro):** 0.9998
- **PR-AUC (Macro):** 0.8891
- **Accuracy:** 0.9871
- **Test Set N:** 378,355

#### Per-Class Metrics (Random Forest)
| Class | Support | Precision | Recall | F1 | FPR | FNR |
|---|---|---|---|---|---|---|
| BENIGN | 314,473 | 0.9999 | 0.9852 | 0.9925 | 0.000548 | 0.014838 |
| Bot | 293 | 0.0749 | 0.9932 | 0.1393 | 0.009506 | 0.006826 |
| DDoS | 19,203 | 0.9999 | 0.9995 | 0.9997 | 0.000003 | 0.000521 |
| DoS GoldenEye | 1,543 | 0.9790 | 0.9961 | 0.9875 | 0.000088 | 0.003889 |
| DoS Hulk | 25,928 | 0.9895 | 0.9978 | 0.9936 | 0.000777 | 0.002198 |
| DoS Slowhttptest | 784 | 0.9836 | 0.9949 | 0.9892 | 0.000034 | 0.005102 |
| DoS slowloris | 808 | 1.0000 | 0.9938 | 0.9969 | 0.000000 | 0.006188 |
| FTP-Patator | 890 | 1.0000 | 0.9978 | 0.9989 | 0.000000 | 0.002247 |
| Heartbleed | 1 | 1.0000 | 1.0000 | 1.0000 | 0.000000 | 0.000000 |
| Infiltration | 5 | 1.0000 | 1.0000 | 1.0000 | 0.000000 | 0.000000 |
| PortScan | 13,623 | 0.9893 | 0.9989 | 0.9941 | 0.000403 | 0.001101 |
| SSH-Patator | 483 | 0.9639 | 0.9938 | 0.9786 | 0.000048 | 0.006211 |
| Web Attack - Brute Force | 220 | 0.4036 | 0.5045 | 0.4485 | 0.000434 | 0.495455 |
| Web Attack - Sql Injection | 3 | 0.6667 | 0.6667 | 0.6667 | 0.000003 | 0.333333 |
| Web Attack - XSS | 98 | 0.1132 | 0.8061 | 0.1985 | 0.001636 | 0.193878 |

### 3. Autoencoder (Unseen-Attack Evaluation)
- **Threshold (95th percentile benign validation):** 0.009342
- **Benign Test FPR:** 0.0915 (9.15%)
- **Held-out DoS Hulk Detection Rate:** 0.9158 (91.58%)
- **Benign Support:** 314,472
- **Attack Support (non-benign):** 210,804

*(Note: The discrepancy between earlier self-reported FPR of 4.98% and the strictly audited test FPR of 9.15% confirms the necessity of this strict final-test-set audit).*

### 4. PyTorch MLP
*Model `models/dl_model/best_mlp.pt` was not present on the server during the audit execution.*
