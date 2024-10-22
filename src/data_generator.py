import random
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import select
from models import QualityMetrics, SessionLocal

class TestDataGenerator:
    def __init__(self):
        self.machine_ids = ["M001", "M002", "M003", "M004", "M005"]
        self.statuses = ["Normal", "Warning", "Critical"]

    def generate_metric(self, timestamp: datetime) -> dict:
        return {
            "timestamp": timestamp,
            "temperature": random.uniform(20, 80),
            "pressure": random.uniform(0.8, 1.2),
            "vibration": random.uniform(0, 10),
            "humidity": random.uniform(30, 70),
            "error_rate": random.uniform(0, 5),
            "machine_id": random.choice(self.machine_ids),
            "status": random.choices(self.statuses, weights=[0.8, 0.15, 0.05])[0]
        }

    def generate_data(self, num_records: int, start_date: datetime) -> list:
        data = []
        for i in range(num_records):
            timestamp = start_date + timedelta(hours=i)
            data.append(self.generate_metric(timestamp))
        return data

    def insert_data(self, db: Session, data: list):
        for record in data:
            db_metric = QualityMetrics(**record)
            db.add(db_metric)
        db.commit()

def generate_and_insert_test_data(num_records: int = 1000):
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
