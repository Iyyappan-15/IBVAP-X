import logging
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from backend.config.settings import settings, DatabaseMode
from backend.db.models import Base

logger = logging.getLogger(__name__)

def create_db_engine():
    mode = settings.DATABASE_MODE
    pg_url = settings.DB_URL_POSTGRES
    sqlite_url = settings.DB_URL_SQLITE

    # Ensure parent directory for SQLite DB exists
    if sqlite_url.startswith("sqlite:///"):
        sqlite_path = sqlite_url.replace("sqlite:///", "")
        os.makedirs(os.path.dirname(os.path.abspath(sqlite_path)), exist_ok=True)

    if mode == DatabaseMode.POSTGRES:
        logger.info(f"[DB] MODE=postgres: Connecting to PostgreSQL at {pg_url}")
        engine = create_engine(pg_url, pool_pre_ping=True)
        return engine, DatabaseMode.POSTGRES

    elif mode == DatabaseMode.SQLITE:
        logger.info(f"[DB] MODE=sqlite: Connecting to SQLite at {sqlite_url}")
        engine = create_engine(sqlite_url, connect_args={"check_same_thread": False})
        return engine, DatabaseMode.SQLITE

    else:  # AUTO mode
        logger.info(f"[DB] MODE=auto: Attempting primary PostgreSQL connection...")
        try:
            pg_engine = create_engine(pg_url, pool_pre_ping=True)
            with pg_engine.connect() as conn:
                logger.info(f"[DB] PostgreSQL connection successful. Using PostgreSQL.")
            return pg_engine, DatabaseMode.POSTGRES
        except Exception as e:
            logger.warning(
                f"[DB] PostgreSQL connection failed ({e}). Falling back to SQLite at {sqlite_url}"
            )
            sqlite_engine = create_engine(sqlite_url, connect_args={"check_same_thread": False})
            return sqlite_engine, DatabaseMode.SQLITE

engine, active_db_mode = create_db_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    """Create all ORM database tables if they do not exist."""
    logger.info(f"[DB] Initializing database tables (Active mode: {active_db_mode})...")
    Base.metadata.create_all(bind=engine)
    logger.info("[DB] Database initialization complete.")

def get_db():
    """FastAPI Dependency for database session management."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
