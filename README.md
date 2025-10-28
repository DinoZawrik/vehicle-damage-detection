# Vehicle Damage Detection System

AI-powered system for automated detection and assessment of vehicle damage. Built with YOLOv8, FastAPI, and Streamlit.

## Overview

This project implements an end-to-end solution for analyzing vehicle damage from images. It detects damaged areas, classifies severity, and estimates repair costs - all through a simple web interface backed by a REST API.

**Key Features:**
- Real-time damage detection using YOLOv8
- Automatic severity classification (minor/moderate/severe/critical)
- Cost estimation with confidence ranges
- RESTful API for integration
- Web UI for easy testing
- Async processing with Celery
- Object storage with MinIO
- Full Docker deployment

## Tech Stack

**Machine Learning:**
- PyTorch & Ultralytics YOLOv8 for object detection
- OpenCV for image processing
- Custom severity classifier and cost estimator

**Backend:**
- FastAPI for REST API
- PostgreSQL for data persistence
- SQLAlchemy ORM
- MinIO for object storage (S3-compatible)
- Celery + Redis for async task processing

**Frontend:**
- Streamlit for web interface
- Interactive visualizations

**DevOps:**
- Docker & Docker Compose
- Multi-container architecture
- Health checks and monitoring

## Project Structure

```
vehicle-damage-detection/
├── src/
│   ├── models/           # ML models and pipeline
│   │   ├── yolo_detector.py
│   │   ├── damage_classifier.py
│   │   ├── cost_estimator.py
│   │   └── pipeline.py
│   ├── api/              # FastAPI backend
│   │   ├── main.py
│   │   ├── models.py
│   │   ├── schemas.py
│   │   ├── storage.py
│   │   └── tasks.py
│   └── ui/               # Streamlit frontend
│       └── app.py
├── tests/                # Unit tests
├── data/                 # Data storage
├── docker-compose.yml
└── requirements.txt
```

## Quick Start

### Prerequisites
- Docker and Docker Compose
- 4GB+ RAM recommended

### Run with Docker Compose

```bash
# Clone the repository
git clone https://github.com/yourusername/vehicle-damage-detection.git
cd vehicle-damage-detection

# Start all services
docker-compose up -d

# Wait for services to be healthy (30-60 seconds)
docker-compose ps
```

**Access the application:**
- Web UI: http://localhost:8501
- API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- MinIO Console: http://localhost:9001

### Local Development

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run API server
uvicorn src.api.main:app --reload

# Run Streamlit (separate terminal)
streamlit run src/ui/app.py
```

## API Usage

### Analyze Image

```python
import requests

# Upload image for analysis
with open("car_image.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/api/analyze",
        files={"file": f}
    )

result = response.json()
print(f"Severity: {result['classification']['severity']}")
print(f"Estimated cost: ${result['cost_estimate']['estimated_cost']:.2f}")
```

### Get Analysis History

```python
response = requests.get("http://localhost:8000/api/history?limit=10")
history = response.json()
```

### API Endpoints

- `GET /` - Root endpoint
- `GET /health` - Health check
- `POST /api/analyze` - Analyze uploaded image
- `GET /api/results/{id}` - Get specific result
- `GET /api/history` - Get analysis history
- `DELETE /api/results/{id}` - Delete result

Full API documentation available at `/docs` when server is running.

## Architecture

The system uses a microservices architecture with the following components:

1. **API Service** - Handles HTTP requests, coordinates ML pipeline
2. **Worker Service** - Processes async tasks via Celery
3. **PostgreSQL** - Stores analysis results and metadata
4. **Redis** - Message broker for Celery
5. **MinIO** - Stores images and visualizations
6. **Streamlit UI** - User-facing web interface

```
┌─────────────┐      ┌──────────┐      ┌─────────────┐
│  Streamlit  │─────▶│   API    │─────▶│  ML Pipeline│
│     UI      │      │ (FastAPI)│      │   (YOLO)    │
└─────────────┘      └─────┬────┘      └─────────────┘
                            │
                ┌───────────┼───────────┐
                │           │           │
         ┌──────▼────┐ ┌───▼────┐ ┌───▼────┐
         │PostgreSQL │ │ MinIO  │ │ Redis  │
         └───────────┘ └────────┘ └────┬───┘
                                        │
                                  ┌─────▼─────┐
                                  │   Celery  │
                                  │   Worker  │
                                  └───────────┘
```

## ML Pipeline

### Detection
- Uses pre-trained YOLOv8 model
- Detects various damage types (scratches, dents, cracks, etc.)
- Returns bounding boxes with confidence scores

### Classification
- Analyzes detection results
- Calculates damage area ratio
- Considers damage count and distribution
- Outputs severity level

### Cost Estimation
- Rule-based pricing model
- Applies severity multipliers
- Includes labor cost estimation
- Provides confidence ranges

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/test_damage_classifier.py
```

## Configuration

Key configuration via environment variables:

```bash
# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/dbname

# Storage
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin

# Task Queue
REDIS_URL=redis://localhost:6379/0

# ML Model
MODEL_DEVICE=cpu  # or 'cuda' for GPU
```

## Performance

**Inference Time:**
- Detection: ~100-300ms (CPU)
- Full pipeline: ~200-500ms (CPU)
- Scales with image size

**Throughput:**
- Single API instance: ~5-10 req/sec
- Horizontal scaling supported via load balancer

## Limitations

- Uses pretrained YOLO model (not fine-tuned on vehicle damage dataset)
- Cost estimation is rule-based approximation
- Best results with clear, well-lit images
- Requires adequate hardware for real-time processing

## Future Improvements

- Fine-tune model on vehicle-specific damage dataset
- Implement ML-based cost prediction
- Add support for video processing
- Integrate with insurance APIs
- Mobile app interface
- Multi-language support

## License

MIT License - see LICENSE file for details

## Contact

For questions or collaboration opportunities, please open an issue or reach out via email.

---

**Note:** This project is designed for portfolio demonstration. For production use, additional security hardening, comprehensive testing, and performance optimization would be required.
