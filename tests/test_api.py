import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import pandas as pd

from app.main import app
from app.database.database import Base, get_db
from app.core.config import settings

# Setup test DB
SQLALCHEMY_DATABASE_URL = "sqlite:///./test_nids.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db

@pytest.fixture(scope="module")
def client():
    Base.metadata.create_all(bind=engine)
    with TestClient(app) as c:
        yield c
    Base.metadata.drop_all(bind=engine)

def test_health_check(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": settings.VERSION}

def test_predict_endpoint(client):
    # Load one row from dataset
    df = pd.read_csv("data/raw/dataset.csv", nrows=1)
    # Convert to dict and replace spaces with underscores to match pydantic
    payload = df.iloc[0].to_dict()
    payload = {k.replace(" ", "_"): v for k, v in payload.items() if k != "Label"}
    
    # Add metadata
    payload["source_ip"] = "192.168.1.10"
    payload["destination_ip"] = "10.0.0.5"
    
    response = client.post("/api/v1/predict", json=payload)
    if response.status_code != 200:
        print("ERROR:", response.json())
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "severity" in data
    assert data["severity"] in ["INFO", "LOW", "MEDIUM", "HIGH", "CRITICAL"]

def test_get_alerts(client):
    response = client.get("/api/v1/alerts")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    # Should have at least the 1 alert from previous test
    assert len(data) >= 1

def test_get_stats(client):
    response = client.get("/api/v1/stats")
    assert response.status_code == 200
    data = response.json()
    assert "total_alerts" in data
    assert data["total_alerts"] >= 1

def test_auth_reload(client):
    # Missing token
    response = client.post("/api/v1/reload-model")
    assert response.status_code == 401
    
    # Valid token
    response = client.post("/api/v1/reload-model", headers={"X-API-Token": settings.API_AUTH_TOKEN})
    assert response.status_code == 200
