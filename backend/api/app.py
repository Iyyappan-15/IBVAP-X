from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from backend.config.settings import settings
from backend.db.session import init_db, get_db, active_db_mode
from backend.db.repository import Repository

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup actions
    init_db()
    yield
    # Shutdown actions

app = FastAPI(
    title=settings.APP_NAME,
    version="3.0-final",
    description="Reliability-Aware Border Video Intelligence API",
    lifespan=lifespan
)

# Configure CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health", status_code=status.HTTP_200_OK)
def health_check(db: Session = Depends(get_db)):
    """API Health Check Endpoint"""
    return {
        "status": "ok",
        "app_name": settings.APP_NAME,
        "environment": settings.APP_ENV,
        "database_configured_mode": settings.DATABASE_MODE,
        "database_active_mode": active_db_mode,
        "version": "3.0-final"
    }

@app.get("/cameras", status_code=status.HTTP_200_OK)
def list_cameras(db: Session = Depends(get_db)):
    """List Active Cameras"""
    repo = Repository(db)
    cameras = repo.list_cameras()
    return {"cameras": [c.name for c in cameras]}
