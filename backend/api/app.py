from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from backend.config.settings import settings
from backend.db.session import init_db, get_db, active_db_mode
from backend.api.routes import cameras, alerts, evidence, coverage, analysis

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield

app = FastAPI(
    title=settings.APP_NAME,
    version="3.0-final",
    description="Reliability-Aware Border Video Intelligence API",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(cameras.router)
app.include_router(alerts.router)
app.include_router(evidence.router)
app.include_router(coverage.router)
app.include_router(analysis.router)

@app.get("/health", status_code=status.HTTP_200_OK)
def health_check(db: Session = Depends(get_db)):
    return {
        "status": "ok",
        "app_name": settings.APP_NAME,
        "environment": settings.APP_ENV,
        "database_configured_mode": settings.DATABASE_MODE,
        "database_active_mode": active_db_mode,
        "version": "3.0-final"
    }
