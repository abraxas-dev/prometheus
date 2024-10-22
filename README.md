# 🚀 Prometheus Quality Management System

## ✨ Features

- **Real-time Monitoring**: Track temperature, pressure, vibration, humidity, and error rates
- **Anomaly Detection**: Automatic identification of unusual patterns using Isolation Forest
- **Predictive Analytics**: Time series forecasting using Facebook Prophet
- **RESTful API**: Built with FastAPI for high performance and easy integration
- **Machine Learning**: Intelligent analysis of quality metrics
- **Test Data Generation**: Built-in test data generation for development

### Next Steps:
- 🎯 Frontend Development:
  - Angular-based web application
  - Real-time dashboard for metrics visualization
  - Interactive analytics interface
  - Machine status monitoring
  - Anomaly alerts visualization
  - Forecasting graphs

## 🛠 Tech Stack

- Python
- FastAPI
- SQLAlchemy
- scikit-learn
- Facebook Prophet
- SQLite

## 🔍 API Endpoints

- `POST /metrics/`: Create new quality metrics
- `GET /metrics/`: Retrieve quality metrics
- `GET /anomalies/{metric_type}`: Detect anomalies in specified metric
- `GET /forecast/{metric_type}`: Generate forecasts for specified metric

## 📦 Installation

1. Clone the repository
```bash
git clone UrlOfRep
cd prometheus-qms
```

2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Run the application
```bash
uvicorn main:app --reload
```

