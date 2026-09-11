from enum import Enum
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict

class DatabaseMode(str, Enum):
    AUTO = "auto"
    POSTGRES = "postgres"
    SQLITE = "sqlite"

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # System & App Environment
    APP_NAME: str = "IBVAP-X"
    APP_ENV: str = "development"
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"
    SECRET_KEY: str = "dev_secret_key_ibvapx_2026"

    # Server Host & Ports
    API_HOST: str = "127.0.0.1"
    API_PORT: int = 8000
    DASHBOARD_PORT: int = 8501

    # Database Settings
    DATABASE_MODE: DatabaseMode = DatabaseMode.AUTO
    DB_URL_POSTGRES: str = "postgresql+psycopg://postgres:postgres@localhost:5432/ibvapx_dev"
    DB_URL_SQLITE: str = "sqlite:///./data/ibvapx_local.db"

    # Object Detection (YOLO)
    YOLO_MODEL_PATH: str = "data/models/yolov8n.pt"
    YOLO_CONFIDENCE_THRESHOLD: float = 0.50
    YOLO_DETECT_CLASSES: str = "person,car,motorcycle,bus,truck"

    # Object Tracking
    TRACK_EVICTION_FRAMES: int = 30
    TRACK_MIN_TRAJECTORY_FRAMES: int = 15

    # Context & Operational Hours
    DAY_START_HOUR: int = 6
    NIGHT_START_HOUR: int = 18
    LOITERING_THRESHOLD_SECONDS: int = 30
    LOITERING_COOLDOWN_SECONDS: int = 60
    DIRECTION_MIN_FRAMES: int = 10

    # Camera Reliability Thresholds & Weights
    BLUR_THRESHOLD: float = 50.0
    BRIGHTNESS_MIN: float = 40.0
    BRIGHTNESS_MAX: float = 220.0

    RELIABILITY_WEIGHT_BLUR: float = 0.35
    RELIABILITY_WEIGHT_BRIGHTNESS: float = 0.25
    RELIABILITY_WEIGHT_FRAME: float = 0.25
    RELIABILITY_WEIGHT_OBSTRUCTION: float = 0.15

    RELIABILITY_GOOD_THRESHOLD: float = 80.0
    RELIABILITY_DEGRADED_THRESHOLD: float = 50.0
    RELIABILITY_POOR_THRESHOLD: float = 20.0

    # Priority Scoring Policy Thresholds
    PRIORITY_LOW_MAX: int = 40
    PRIORITY_MEDIUM_MAX: int = 65
    PRIORITY_HIGH_MAX: int = 84

    # Actionability & Escalation
    ACK_TIMEOUT_SECONDS: int = 30
    PRE_EVENT_SECONDS: int = 10
    POST_EVENT_SECONDS: int = 10
    EVIDENCE_STORAGE_DIR: str = "data/evidence"

    # Cross-Camera Correlation
    CORRELATION_WINDOW_SECONDS: int = 40

    # Anomaly Engine (IsolationForest)
    ANOMALY_CONTAMINATION: float = 0.10
    ANOMALY_SCORE_THRESHOLD: float = 0.50
    ANOMALY_MODEL_PATH: str = "data/anomaly_models/ibvapx_anomaly_model.pkl"

    # Security & CORS
    CORS_ORIGINS: str = "http://localhost:8501,http://127.0.0.1:8501"

    @property
    def detect_classes_list(self) -> List[str]:
        return [c.strip() for c in self.YOLO_DETECT_CLASSES.split(",") if c.strip()]

    @property
    def cors_origins_list(self) -> List[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]

settings = Settings()
