# Security Architecture & Limitations

## Defensive Scope
This platform is purely defensive. It is designed to passively ingest, monitor, and classify network flows. It contains no modules for scanning, exploiting, or interacting with third-party networks.

## API Security Implementation
The FastAPI backend implements several defensive programming concepts:
1. **Input Validation**: Pydantic strictly enforces data types (e.g., rejecting string injections where ports are expected).
2. **Error Handling**: Custom exception handlers ensure stack traces are not leaked in the HTTP response.
3. **Authentication**: Administrative endpoints (like `/reload-model`) require an `X-API-Token`.
4. **Secrets Management**: Configuration is loaded from `.env` via `pydantic-settings`; no hardcoded secrets exist in the source repository.
5. **SQL Injection Prevention**: SQLAlchemy ORM is used, inherently parameterizing all database queries.

## ML Security Limitations
1. **Adversarial Examples**: Tree-based classifiers and Autoencoders can be evaded by adversaries slightly modifying packet timing or lengths to push the feature vectors below the anomaly threshold.
2. **Data Bias & Drift**: The model is trained on a static dataset. If enterprise traffic significantly deviates from the training baseline, false positive rates will spike.
3. **100% Accuracy is a Myth**: This system does not guarantee prevention. It is a detection layer meant to work in tandem with human analysts (hence the SHAP integrations).

## Live Monitoring Authorization
If the system is ever hooked into a live interface (e.g., via `tshark` or `NFStream` live capture), it must only be deployed on authorized interfaces where the operator has explicit consent to monitor the traffic.
