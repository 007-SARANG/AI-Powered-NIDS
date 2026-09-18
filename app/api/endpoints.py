from fastapi import APIRouter, Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader
from sqlalchemy.orm import Session
from typing import List

from app.core.config import settings
from app.database.database import get_db
from app.database import models
from app.models import schemas
from app.services.detection_service import get_detection_service, DetectionService

router = APIRouter()

api_key_header = APIKeyHeader(name="X-API-Token", auto_error=False)

def verify_token(api_key: str = Security(api_key_header)):
    if api_key != settings.API_AUTH_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Token",
        )
    return api_key

@router.get("/health")
def health_check():
    return {"status": "ok", "version": settings.VERSION}

@router.post("/predict", response_model=schemas.PredictionResponse)
def predict_flow(
    flow: schemas.NetworkFlow, 
    db: Session = Depends(get_db),
    service: DetectionService = Depends(get_detection_service)
):
    """Predict a single flow."""
    # Convert Pydantic model to dict, excluding None to allow fallback to default features if any
    flow_dict = flow.model_dump(exclude_unset=True)
    results = service.predict_and_store([flow_dict], db)
    return results[0]

@router.post("/predict/batch", response_model=List[schemas.PredictionResponse])
def predict_batch(
    flows: List[schemas.NetworkFlow], 
    db: Session = Depends(get_db),
    service: DetectionService = Depends(get_detection_service)
):
    """Predict a batch of flows."""
    if len(flows) > 1000:
        raise HTTPException(status_code=400, detail="Batch size limit is 1000")
        
    flow_dicts = [f.model_dump(exclude_unset=True) for f in flows]
    results = service.predict_and_store(flow_dicts, db)
    return results

@router.get("/alerts", response_model=List[schemas.AlertResponse])
def get_alerts(limit: int = 50, offset: int = 0, db: Session = Depends(get_db)):
    """Retrieve recent alerts."""
    alerts = db.query(models.Alert).order_by(models.Alert.timestamp.desc()).offset(offset).limit(limit).all()
    return alerts

@router.get("/alerts/{alert_id}", response_model=schemas.AlertDetail)
def get_alert(alert_id: int, db: Session = Depends(get_db)):
    """Retrieve a specific alert with SHAP explanation."""
    alert = db.query(models.Alert).filter(models.Alert.id == alert_id).first()
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    return alert

@router.get("/stats")
def get_stats(db: Session = Depends(get_db)):
    """Get system stats for the dashboard."""
    total = db.query(models.Alert).count()
    high_critical = db.query(models.Alert).filter(models.Alert.severity.in_(["HIGH", "CRITICAL"])).count()
    anomalies = db.query(models.Alert).filter(models.Alert.is_anomaly == True).count()
    
    return {
        "total_alerts": total,
        "high_critical_alerts": high_critical,
        "anomalies_detected": anomalies
    }

@router.post("/reload-model", dependencies=[Depends(verify_token)])
def reload_model():
    """Admin endpoint to reload ML models."""
    global detection_service
    from app.services.detection_service import detection_service as ds
    ds = None
    _ = get_detection_service() # Forces reload
    return {"status": "Models reloaded successfully"}
