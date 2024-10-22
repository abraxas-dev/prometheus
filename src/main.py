from typing import List, Optional, Annotated, Generator
from datetime import datetime, timedelta
from pathlib import Path
import logging
from fastapi import FastAPI, HTTPException, Query, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select
from sqlalchemy.orm import Session
from pydantic import BaseModel, ConfigDict
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from prophet import Prophet
import pandas as pd
from models import QualityMetrics, SessionLocal

# Logging Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI App Initialization
app = FastAPI(
    title="Prometheus Quality Management System",
    description="Predictive Quality Management API with Anomaly Detection and Time Series Analysis",
    version="1.0.0"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"], #URL der Frontend-App
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic Models with updated config
class QualityMetricsBase(BaseModel):
    temperature: float
    pressure: float
    vibration: float
    humidity: float
    error_rate: float
    machine_id: str
    status: str

class QualityMetricsCreate(QualityMetricsBase):
    model_config = ConfigDict(from_attributes=True)

class QualityMetricsResponse(QualityMetricsBase):
    id: int
    timestamp: datetime
    model_config = ConfigDict(from_attributes=True)

class AnomalyResponse(BaseModel):
    timestamp: datetime
    is_anomaly: bool
    anomaly_score: float
    metric_type: str
    value: float
    model_config = ConfigDict(from_attributes=True)

class ForecastResponse(BaseModel):
    timestamp: datetime
    forecast_value: float
    lower_bound: float
    upper_bound: float
    metric_type: str
    model_config = ConfigDict(from_attributes=True)

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Helper Functions
def detect_anomalies(data: List[float]) -> List[bool]:
    clf = IsolationForest(contamination=0.1, random_state=42)
    return clf.fit_predict(np.array(data).reshape(-1, 1)) == -1

# API Routes
@app.post("/metrics/", response_model=QualityMetricsResponse, status_code=status.HTTP_201_CREATED)
def create_metric(metric: QualityMetricsCreate, db: Annotated[Session, Depends(get_db)]) -> QualityMetrics:
    db_metric = QualityMetrics(**metric.model_dump())
    db.add(db_metric)
    db.commit()
    db.refresh(db_metric)
    return db_metric

@app.get("/metrics/", response_model=List[QualityMetricsResponse])
def read_metrics(
    db: Annotated[Session, Depends(get_db)],
    skip: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=100)] = 100,
    machine_id: Optional[str] = None
) -> List[QualityMetrics]:
   
    try:
        query = select(QualityMetrics)
        if machine_id:
            query = query.where(QualityMetrics.machine_id == machine_id)
        query = query.offset(skip).limit(limit)
        result = db.execute(query)
        return list(result.scalars().all())
    except Exception as e:
        logger.error(f"Error reading metrics: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not retrieve metrics: {str(e)}"
        )

@app.get("/anomalies/{metric_type}", response_model=List[AnomalyResponse])
async def detect_metric_anomalies(
    metric_type: str,
    db: Annotated[Session, Depends(get_db)],
    machine_id: Optional[str] = None,
    hours: Annotated[int, Query(ge=1, le=168)] = 24
) -> List[AnomalyResponse]:
   
    try:
        time_threshold = datetime.utcnow() - timedelta(hours=hours)
        query = select(QualityMetrics).where(QualityMetrics.timestamp >= time_threshold)
        
        if machine_id:
            query = query.where(QualityMetrics.machine_id == machine_id)
            
        result = db.execute(query.order_by(QualityMetrics.timestamp))
        metrics = list(result.scalars().all())
        
        if not metrics:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No data found for the specified period"
            )
            
        values = [getattr(m, metric_type) for m in metrics]
        anomalies = detect_anomalies(values)
        
        return [
            AnomalyResponse(
                timestamp=m.timestamp,
                is_anomaly=is_anomaly,
                anomaly_score=abs(value - np.mean(values)) / (np.std(values) if np.std(values) != 0 else 1),
                metric_type=metric_type,
                value=value
            )
            for m, is_anomaly, value in zip(metrics, anomalies, values)
        ]
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error detecting anomalies: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Anomaly detection failed: {str(e)}"
        )

@app.get("/forecast/{metric_type}", response_model=List[ForecastResponse])
async def forecast_metric(
    metric_type: str,
    db: Annotated[Session, Depends(get_db)],
    machine_id: Optional[str] = None,
    hours: Annotated[int, Query(ge=1, le=168)] = 24,
    forecast_periods: Annotated[int, Query(ge=1, le=72)] = 10
) -> List[ForecastResponse]:
 
    try:
        time_threshold = datetime.utcnow() - timedelta(hours=hours)
        query = select(QualityMetrics).where(QualityMetrics.timestamp >= time_threshold)
        
        if machine_id:
            query = query.where(QualityMetrics.machine_id == machine_id)
            
        result = db.execute(query.order_by(QualityMetrics.timestamp))
        metrics = list(result.scalars().all())
        
        if not metrics:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No data found for the specified period"
            )
            
        values = [getattr(m, metric_type) for m in metrics]
        timestamps = [m.timestamp for m in metrics]
        
        df = pd.DataFrame({'ds': timestamps, 'y': values})
        model = Prophet()
        model.fit(df)
        
        future = model.make_future_dataframe(periods=forecast_periods, freq='H')
        forecast = model.predict(future)
        forecast = forecast.tail(forecast_periods)

        return [
            ForecastResponse(
                timestamp=row['ds'],
                forecast_value=row['yhat'],
                lower_bound=row['yhat_lower'],
                upper_bound=row['yhat_upper'],
                metric_type=metric_type
            )
            for _, row in forecast.iterrows()
        ]
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating forecast: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Forecast generation failed: {str(e)}"
        )

# Test data generation function
def generate_and_insert_test_data(num_records: int = 1000):
    from data_generator import TestDataGenerator
    generator = TestDataGenerator()
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(hours=num_records)
    
    data = generator.generate_data(num_records, start_date)
    
    db = SessionLocal()
    try:
        generator.insert_data(db, data)
        print(f"{num_records} Testdatensätze wurden erfolgreich eingefügt.")
    finally:
        db.close()

if __name__ == "__main__":
    generate_and_insert_test_data()