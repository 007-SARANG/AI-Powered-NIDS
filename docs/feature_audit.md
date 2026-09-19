# Feature Leakage Audit Report

## Background
In Network Intrusion Detection Systems (NIDS) utilizing machine learning, "data leakage" occurs when a model learns to predict malicious traffic based on identifying artifacts specific to the training environment (e.g., specific IP addresses) rather than generalizable behavioral patterns.

A model suffering from data leakage will achieve 99.9% accuracy during testing but will catastrophically fail in a real-world production environment (a phenomenon known as concept drift or environmental overfit).

## Features Explicitly Dropped

To guarantee scientific validity and real-world generalizability, the following features have been intentionally excluded from the model pipeline during the preprocessing stage:

| **Source IP & Dest IP** | Prevent the model from learning internal network topology and lab-specific IP configurations. |
| **Flow ID** | A unique identifier providing no behavioral value; highly correlated with specific session runs. |
| **Timestamp** | The Hugging Face `c01dsnap/CIC-IDS2017` feature dataset entirely omits this column. Even if present, it is dropped to prevent the model from learning temporal lab shortcuts (e.g., "attacks only happen on Wednesday mornings"). |
| **SimillarHTTP** | A heavily biased metadata string that often uniquely identifies the specific attack script used in the lab. |
| **Destination Port** | *Configurable.* Default behavior retains it, but an explicit ablation experiment (`scripts/ablation_dest_port.py`) is provided to test the model's reliance on port numbers versus actual traffic behavior. |

## Exact Duplicate Policy

Due to the absence of the 5-tuple (IPs, Ports, Protocol) and `Timestamp` in the Hugging Face CSV mirror, short identical flows (like DNS or repeated TCP handshakes) present as mathematically exact duplicate rows.
**Policy:** Out of 2.8M rows, approximately 308k exact duplicates exist. To mathematically guarantee that identical examples never leak across the train/test boundary in a random split, all exact duplicates are globally dropped prior to dataset splitting.

## Feature Mapping (NFStream to CIC-IDS2017)
The original dataset was generated using `CICFlowMeter`. Our real-time pipeline uses `NFStream`. 

Since these two tools use slightly different statistical aggregation methods, we have implemented an **Intersection Approach**:
Only mathematical equivalents between NFStream and CICFlowMeter are passed into the inference engine. Features that NFStream cannot accurately reproduce identically to CICFlowMeter are padded or ignored, preventing the model from encountering out-of-distribution feature spaces during live packet ingestion.

*This audit guarantees that the models evaluated in `docs/model_evaluation.md` demonstrate true behavioral learning.*
