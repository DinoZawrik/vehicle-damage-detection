# Quick Start Guide

## Option 1: Docker Compose (Recommended)

This is the easiest way to run the entire system.

```bash
# Clone the repository
git clone <your-repo-url>
cd vehicle-damage-detection

# Start all services
docker-compose up -d

# Check services status
docker-compose ps

# View logs
docker-compose logs -f api
```

Wait 30-60 seconds for all services to start, then access:
- **Web UI**: http://localhost:8501
- **API Docs**: http://localhost:8000/docs
- **MinIO Console**: http://localhost:9001 (admin/admin)

To stop:
```bash
docker-compose down
```

## Option 2: Local Development

### Prerequisites
- Python 3.10+
- PostgreSQL running on port 5432
- Redis running on port 6379
- MinIO running on port 9000

### Setup

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Run Services

**Terminal 1 - API:**
```bash
# Set environment variables
export DATABASE_URL="postgresql://postgres:password@localhost:5432/vehicle_damage_db"
export REDIS_URL="redis://localhost:6379/0"
export MINIO_ENDPOINT="localhost:9000"

# Run API
uvicorn src.api.main:app --reload --port 8000
```

**Terminal 2 - Streamlit UI:**
```bash
# Activate venv
venv\Scripts\activate

# Run UI
streamlit run src/ui/app.py
```

**Terminal 3 - Celery Worker (Optional):**
```bash
# Activate venv
venv\Scripts\activate

# Run worker
celery -A src.api.tasks.celery_app worker --loglevel=info
```

## Testing the System

### Via Web UI
1. Open http://localhost:8501
2. Upload a vehicle image
3. Click "Analyze Damage"
4. View results

### Via API

```python
import requests

# Test health
response = requests.get("http://localhost:8000/health")
print(response.json())

# Analyze image
with open("test_image.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/api/analyze",
        files={"file": f}
    )
    result = response.json()
    print(f"Severity: {result['classification']['severity']}")
    print(f"Cost: ${result['cost_estimate']['estimated_cost']}")
```

## Running Tests

```bash
# Activate venv
venv\Scripts\activate

# Run all tests
pytest

# Run with coverage
pytest --cov=src tests/
```

## Troubleshooting

**Port already in use:**
```bash
# Change ports in docker-compose.yml
# API: change 8000:8000 to 8001:8000
# UI: change 8501:8501 to 8502:8501
```

**Models not loading:**
```bash
# The system uses pretrained YOLO models that download automatically
# Make sure you have internet connection on first run
```

**Database connection errors:**
```bash
# Wait for PostgreSQL to fully start (30 seconds)
docker-compose ps
# Check logs
docker-compose logs postgres
```

**Memory issues:**
```bash
# Reduce batch size in model config
# Or allocate more RAM to Docker (4GB minimum)
```

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Check [API documentation](http://localhost:8000/docs) for all endpoints
- Explore the code in `src/` directory
- Run tests in `tests/` directory
