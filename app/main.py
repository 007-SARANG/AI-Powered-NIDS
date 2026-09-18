import uvicorn
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.core.config import settings
from app.api.endpoints import router as api_router
from app.database.database import engine
from app.database import models
from app.services.detection_service import get_detection_service

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logging.info("Starting up NIDS API...")
    # Create tables
    models.Base.metadata.create_all(bind=engine)
    # Preload ML models
    try:
        get_detection_service()
        logging.info("ML Models loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load ML models during startup: {e}")
    yield
    # Shutdown
    logging.info("Shutting down NIDS API...")

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    lifespan=lifespan
)

# Restrict CORS in production, but allow frontend dashboard for now
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Should be restricted to Streamlit IP in prod
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix=settings.API_PREFIX)

@app.get("/")
def root():
    return {"message": "NIDS API is running. See /docs for API documentation."}

if __name__ == "__main__":
    uvicorn.run("app.main:app", host=settings.API_HOST, port=settings.API_PORT, reload=True)
