import pandas as pd
import logging
from typing import List, Dict, Any
from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.core.config import settings
from nids.detection.hybrid_engine import HybridDetectionEngine
from ml.explainability.shap_explainer import SHAPExplainer
from app.database import models

logging.basicConfig(level=logging.INFO, format="%(message)s")

class DetectionService:
    def __init__(self):
        logging.info("Initializing Detection Service...")
        try:
            self.engine = HybridDetectionEngine(models_dir=settings.MODEL_PATH)
            self.explainer = SHAPExplainer(models_dir=settings.MODEL_PATH)
        except Exception as e:
            logging.error(f"Failed to load models: {e}")
            raise RuntimeError("Model loading failed.")
            
    def predict_and_store(self, flows: List[Dict[str, Any]], db: Session):
        if not flows:
            raise HTTPException(status_code=400, detail="No flows provided")
            
        # Convert to DataFrame
        # Clean up keys: some might pass Destination_Port, we need "Destination Port" for CIC
        cleaned_flows = []
        metadata_list = []
        
        for flow in flows:
            # Extract metadata
            meta = {
                "source_ip": flow.pop("source_ip", None),
                "destination_ip": flow.pop("destination_ip", None),
                "protocol": flow.pop("protocol_str", None)
            }
            metadata_list.append(meta)
            
            # Fix column names by trying to match expected features
            cleaned = {}
            expected_features = self.engine.preprocessor['features']
            for k, v in flow.items():
                if k in expected_features:
                    cleaned[k] = v
                elif k.replace("_", " ") in expected_features:
                    cleaned[k.replace("_", " ")] = v
                else:
                    cleaned[k] = v
            cleaned_flows.append(cleaned)
            
        df = pd.DataFrame(cleaned_flows)
        
        try:
            predictions = self.engine.predict(df)
        except Exception as e:
            logging.error(f"Prediction failed: {e}")
            raise HTTPException(status_code=400, detail=f"Prediction failed. Ensure all features match CIC-IDS2017 schema.")
            
        alerts_created = []
        
        for i, res in enumerate(predictions):
            meta = metadata_list[i]
            
            # Generate SHAP explanation if it's not strictly benign with low severity
            explanation = None
            if res["severity"] in ["CRITICAL", "HIGH", "MEDIUM"]:
                # Explain single instance
                single_df = df.iloc[[i]]
                try:
                    exp = self.explainer.explain_instance(single_df, top_k=5)[0]
                    explanation = exp
                except Exception as e:
                    logging.warning(f"Failed to generate SHAP: {e}")
                    
            # Store in DB
            db_alert = models.Alert(
                source_ip=meta["source_ip"],
                destination_ip=meta["destination_ip"],
                protocol=meta["protocol"],
                prediction=res["prediction"],
                confidence=res["confidence"],
                anomaly_score=res["anomaly_score"],
                is_anomaly=res["is_anomaly"],
                severity=res["severity"],
                model_version=settings.MODEL_VERSION,
                explanation=explanation
            )
            db.add(db_alert)
            alerts_created.append(res)
            
        db.commit()
        return alerts_created
        
# Global instance
detection_service = None

def get_detection_service():
    global detection_service
    if detection_service is None:
        detection_service = DetectionService()
    return detection_service
