# Feature Leakage Audit Report

## Background
In Network Intrusion Detection Systems (NIDS) utilizing machine learning, "data leakage" occurs when a model learns to predict malicious traffic based on identifying artifacts specific to the training environment (e.g., specific IP addresses) rather than generalizable behavioral patterns.

A model suffering from data leakage will achieve 99.9% accuracy during testing but will catastrophically fail in a real-world production environment (a phenomenon known as concept drift or environmental overfit).

## Features Explicitly Dropped

To guarantee scientific validity and real-world generalizability, the following features have been intentionally excluded from the model pipeline during the preprocessing stage:

| Feature | Reason for Exclusion |
|---------|----------------------|
| **Source IP** | The model would memorize attacker IPs from the lab environment, making it useless against real attackers using different IPs. |
| **Destination IP** | The model would learn the internal topology of the test network (e.g., "traffic to 192.168.10.50 is always an attack"). |
| **Flow ID** | A unique identifier providing no behavioral value; highly correlated with specific session runs. |
| **Timestamp** | The dataset was generated during specific hours. The model might learn that "attacks happen at 10:00 AM on Wednesday" rather than learning packet signatures. |
| **SimillarHTTP** | A heavily biased metadata string that often uniquely identifies the script used to launch the attack in the lab environment. |
| **Destination Port** | *(Dropping based on SOC audit consensus)* While some researchers keep port numbers, dropping it forces the model to learn the *behavior* of an attack (e.g., packet rate, size variance) rather than simply memorizing protocol conventions (e.g., "if Port = 22, it's brute force"). This prevents attackers from evading detection simply by running their exploit on a non-standard port. |

## Feature Mapping (NFStream to CIC-IDS2017)
The original dataset was generated using `CICFlowMeter`. Our real-time pipeline uses `NFStream`. 

Since these two tools use slightly different statistical aggregation methods, we have implemented an **Intersection Approach**:
Only mathematical equivalents between NFStream and CICFlowMeter are passed into the inference engine. Features that NFStream cannot accurately reproduce identically to CICFlowMeter are padded or ignored, preventing the model from encountering out-of-distribution feature spaces during live packet ingestion.

*This audit guarantees that the models evaluated in `docs/model_evaluation.md` demonstrate true behavioral learning.*
