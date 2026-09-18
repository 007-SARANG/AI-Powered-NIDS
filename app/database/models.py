from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, JSON
from datetime import datetime
from app.database.database import Base

class Alert(Base):
    __tablename__ = "alerts"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Network Info
    source_ip = Column(String, index=True, nullable=True)
    destination_ip = Column(String, index=True, nullable=True)
    source_port = Column(Integer, nullable=True)
    destination_port = Column(Integer, nullable=True)
    protocol = Column(String, nullable=True)
    
    # Prediction
    prediction = Column(String, index=True)
    confidence = Column(Float)
    anomaly_score = Column(Float)
    is_anomaly = Column(Boolean)
    
    # Alert Metadata
    severity = Column(String, index=True)
    status = Column(String, default="NEW", index=True) # NEW, INVESTIGATING, RESOLVED
    model_version = Column(String)
    
    # Detailed Explanation (SHAP)
    explanation = Column(JSON, nullable=True)
