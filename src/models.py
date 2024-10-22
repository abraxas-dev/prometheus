from sqlalchemy import create_engine, Column, Integer, Float, DateTime, String
from sqlalchemy.orm import sessionmaker, DeclarativeBase, Mapped
from datetime import datetime

SQLALCHEMY_DATABASE_URL = "sqlite:///./prometheus.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, echo=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class Base(DeclarativeBase):
    pass

class QualityMetrics(Base):
    __tablename__ = "quality_metrics"
    id: Mapped[int] = Column(Integer, primary_key=True, index=True)
    timestamp: Mapped[datetime] = Column(DateTime, default=datetime.utcnow)
    temperature: Mapped[float] = Column(Float)
    pressure: Mapped[float] = Column(Float)
    vibration: Mapped[float] = Column(Float)
    humidity: Mapped[float] = Column(Float)
    error_rate: Mapped[float] = Column(Float)
    machine_id: Mapped[str] = Column(String)
    status: Mapped[str] = Column(String)

Base.metadata.create_all(bind=engine)