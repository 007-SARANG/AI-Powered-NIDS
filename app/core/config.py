from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "NIDS API"
    VERSION: str = "1.0.0"
    API_PREFIX: str = "/api/v1"
    
    # Security
    API_AUTH_TOKEN: str = "your-secure-token-here"
    
    # Server
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    
    # Database
    DATABASE_URL: str = "sqlite:///./nids.db"
    
    # ML
    MODEL_PATH: str = "models/"
    MODEL_VERSION: str = "1.0.0"
    
    class Config:
        env_file = ".env"

settings = Settings()
