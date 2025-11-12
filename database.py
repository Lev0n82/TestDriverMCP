"""
Database layer with SQLAlchemy for persistent storage.
Supports PostgreSQL for production and SQLite for testing.
"""

import os
from typing import Optional, AsyncGenerator
from contextlib import asynccontextmanager
from sqlalchemy import create_engine, Column, String, Float, Integer, Boolean, DateTime, JSON, Text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base, Session
from sqlalchemy.pool import NullPool
from datetime import datetime
import structlog

logger = structlog.get_logger()

Base = declarative_base()

class HealingEventDB(Base):
    """Database model for healing events."""
    __tablename__ = "healing_events"
    
    event_id = Column(String(255), primary_key=True)
    test_id = Column(String(255), nullable=False, index=True)
    execution_id = Column(String(255), nullable=False)
    element_id = Column(String(255), nullable=False, index=True)
    original_locator = Column(JSON, nullable=False)
    failure_reason = Column(Text, nullable=False)
    healing_strategy = Column(String(50), nullable=False)
    new_locator = Column(JSON, nullable=False)
    confidence_score = Column(Float, nullable=False)
    healing_successful = Column(Boolean, nullable=False)
    validation_method = Column(String(50), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    metadata_json = Column(JSON, nullable=True)

class TestExecutionDB(Base):
    """Database model for test executions."""
    __tablename__ = "test_executions"
    
    execution_id = Column(String(255), primary_key=True)
    test_id = Column(String(255), nullable=False, index=True)
    status = Column(String(50), nullable=False)
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    duration_ms = Column(Integer, nullable=True)
    elements_tested = Column(JSON, nullable=True)
    metadata_json = Column(JSON, nullable=True)

class ElementStabilityDB(Base):
    """Database model for element stability tracking."""
    __tablename__ = "element_stability"
    
    element_id = Column(String(255), primary_key=True)
    total_executions = Column(Integer, default=0)
    total_healings = Column(Integer, default=0)
    stability_score = Column(Float, default=1.0)
    last_updated = Column(DateTime, default=datetime.utcnow)
    metadata_json = Column(JSON, nullable=True)

class DatabaseManager:
    """Manages database connections and operations."""
    
    def __init__(self, database_url: Optional[str] = None):
        """
        Initialize database manager.
        
        Args:
            database_url: Database connection string (defaults to SQLite for testing)
        """
        if database_url is None:
            # Default to SQLite for testing
            database_url = "sqlite:///./testdriver.db"
            self.is_async = False
        else:
            # Check if async (PostgreSQL with asyncpg)
            self.is_async = database_url.startswith("postgresql+asyncpg://")
        
        self.database_url = database_url
        
        if self.is_async:
            self.engine = create_async_engine(
                database_url,
                echo=False,
                poolclass=NullPool
            )
            self.SessionLocal = async_sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
        else:
            # Synchronous engine for SQLite
            sync_url = database_url.replace("sqlite+aiosqlite://", "sqlite:///")
            self.engine = create_engine(sync_url, echo=False)
            self.SessionLocal = None
        
        logger.info("Database manager initialized", url=database_url, async_mode=self.is_async)
    
    async def create_tables(self):
        """Create all database tables."""
        if self.is_async:
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
        else:
            Base.metadata.create_all(self.engine)
        
        logger.info("Database tables created")
    
    async def drop_tables(self):
        """Drop all database tables (for testing)."""
        if self.is_async:
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.drop_all)
        else:
            Base.metadata.drop_all(self.engine)
        
        logger.info("Database tables dropped")
    
    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get database session (async context manager)."""
        if self.is_async:
            async with self.SessionLocal() as session:
                try:
                    yield session
                    await session.commit()
                except Exception:
                    await session.rollback()
                    raise
        else:
            # For SQLite, use synchronous session
            session = Session(self.engine)
            try:
                yield session
                session.commit()
            except Exception:
                session.rollback()
                raise
            finally:
                session.close()
    
    async def close(self):
        """Close database connections."""
        if self.is_async:
            await self.engine.dispose()
        else:
            self.engine.dispose()
        
        logger.info("Database connections closed")

# Global database manager instance
_db_manager: Optional[DatabaseManager] = None

def get_database_manager(database_url: Optional[str] = None) -> DatabaseManager:
    """Get or create global database manager instance."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager(database_url)
    return _db_manager

async def init_database(database_url: Optional[str] = None):
    """Initialize database and create tables."""
    db = get_database_manager(database_url)
    await db.create_tables()
    return db
