from pydantic import BaseModel, ConfigDict
from typing import Dict, Any, List, Optional
from datetime import datetime

class NetworkFlow(BaseModel):
    model_config = ConfigDict(extra='allow')
    
    # Minimal fields for CIC-IDS2017 schema. Users can pass the rest as extra fields.
    Destination_Port: Optional[float] = None
    Flow_Duration: Optional[float] = None
    Total_Fwd_Packets: Optional[float] = None
    Total_Backward_Packets: Optional[float] = None
    
    # Metadata for the alert (optional, usually provided by PCAP extractor)
    source_ip: Optional[str] = None
    destination_ip: Optional[str] = None
    protocol_str: Optional[str] = None
    
class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    anomaly_score: float
    is_anomaly: bool
    severity: str
    
class SHAPFeature(BaseModel):
    feature: str
    original_value: Optional[float]
    processed_value: float
    shap_value: float

class SHAPExplanation(BaseModel):
    predicted_class_idx: int
    top_features: List[SHAPFeature]

class AlertResponse(BaseModel):
    id: int
    timestamp: datetime
    source_ip: Optional[str]
    destination_ip: Optional[str]
    prediction: str
    confidence: float
    severity: str
    status: str
    model_version: str
    
    model_config = ConfigDict(from_attributes=True)
    
class AlertDetail(AlertResponse):
    anomaly_score: float
    is_anomaly: bool
    source_port: Optional[int]
    destination_port: Optional[int]
    protocol: Optional[str]
    explanation: Optional[SHAPExplanation]
