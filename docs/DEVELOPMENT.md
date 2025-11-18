# 👨‍💻 Development Guide

Руководство для разработчиков Vehicle Damage Detection System.

---

## Начало работы

### Настройка окружения разработчика

```bash
# 1. Клонировать репозиторий
git clone https://github.com/yourusername/vehicle-damage-detection.git
cd vehicle-damage-detection

# 2. Создать ветку для фичи
git checkout -b feature/your-feature-name

# 3. Настроить Python окружение
python -m venv venv
source venv/bin/activate  # Linux/Mac
# или
.\venv\Scripts\Activate.ps1  # Windows

# 4. Установить зависимости + dev tools
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### requirements-dev.txt

```
# Testing
pytest>=7.0.0
pytest-cov>=4.0.0
pytest-asyncio>=0.21.0

# Code quality
black>=23.0.0
flake8>=6.0.0
mypy>=1.0.0
isort>=5.12.0

# Development
ipython>=8.0.0
jupyter>=1.0.0
```

---

## Структура проекта

```
vehicle-damage-detection/
├── src/                          # Исходный код
│   ├── api/                     # FastAPI backend
│   │   ├── main.py             # API endpoints
│   │   ├── models.py           # Pydantic models
│   │   └── schemas.py          # Response schemas
│   ├── models/                 # AI models
│   │   ├── simple_pipeline.py  # Detection pipeline
│   │   └── damage_analyzer.py  # Damage analysis
│   └── utils/                  # Utilities
│       ├── image_utils.py      # Image processing
│       └── visualization.py    # Visualization helpers
├── web/                        # React frontend
│   ├── src/
│   │   ├── App.tsx            # Main component
│   │   ├── components/        # UI components
│   │   └── api.ts            # API client
│   └── package.json
├── tests/                     # Tests
│   ├── test_api.py           # API tests
│   ├── test_pipeline.py      # Pipeline tests
│   └── test_analyzer.py      # Analyzer tests
├── data/                      # Data files
│   ├── test_samples/         # Test images
│   └── uploads/              # User uploads
├── models/                    # Model weights
├── docs/                      # Documentation
├── docker-compose.yml         # Docker setup
├── requirements.txt           # Python dependencies
└── README.md                  # Main readme
```

---

## Backend Development

### Добавление нового endpoint

1. Определите Pydantic модель в `src/api/models.py`:

```python
from pydantic import BaseModel
from typing import List

class NewFeatureRequest(BaseModel):
    image_id: str
    params: dict

class NewFeatureResponse(BaseModel):
    result: str
    data: List[dict]
```

2. Добавьте endpoint в `src/api/main.py`:

```python
@app.post("/new-feature", response_model=NewFeatureResponse)
async def new_feature(request: NewFeatureRequest):
    """
    New feature endpoint.
    
    Args:
        request: Request with image_id and params
        
    Returns:
        NewFeatureResponse with results
    """
    try:
        # Your logic here
        result = process_new_feature(request.image_id, request.params)
        return NewFeatureResponse(
            result="success",
            data=result
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

3. Добавьте тесты в `tests/test_api.py`:

```python
def test_new_feature():
    """Test new feature endpoint."""
    response = client.post("/new-feature", json={
        "image_id": "test123",
        "params": {"key": "value"}
    })
    assert response.status_code == 200
    assert response.json()["result"] == "success"
```

### Модификация Detection Pipeline

Файл: `src/models/simple_pipeline.py`

```python
from ultralytics import YOLO
import numpy as np
from typing import List, Dict

class SimpleDetectionPipeline:
    """YOLO-only detection pipeline."""
    
    def __init__(self, model_path: str, conf: float = 0.35):
        self.model = YOLO(model_path)
        self.conf = conf
    
    def detect(self, image: np.ndarray) -> List[Dict]:
        """
        Detect damages in image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of detections with bbox, confidence, class
        """
        results = self.model(image, conf=self.conf)
        return self._parse_results(results)
    
    def _parse_results(self, results) -> List[Dict]:
        """Parse YOLO results to dict format."""
        detections = []
        for result in results:
            for box in result.boxes:
                detections.append({
                    'bbox': box.xyxy[0].tolist(),
                    'confidence': float(box.conf[0]),
                    'class': int(box.cls[0])
                })
        return detections
```

---

## Frontend Development

### Структура компонентов

```
web/src/
├── App.tsx                    # Main app
├── components/
│   ├── ImageUpload.tsx       # File upload
│   ├── LoadingSpinner.tsx    # Loading indicator
│   └── ResultsDisplay.tsx    # Results view
├── api.ts                    # API client
├── types.ts                  # TypeScript types
└── App.css                   # Styles
```

### Добавление нового компонента

1. Создайте файл `web/src/components/NewComponent.tsx`:

```typescript
import React from 'react';

interface NewComponentProps {
  data: string;
  onAction: () => void;
}

export const NewComponent: React.FC<NewComponentProps> = ({ data, onAction }) => {
  return (
    <div className="new-component">
      <h2>{data}</h2>
      <button onClick={onAction}>Action</button>
    </div>
  );
};
```

2. Импортируйте в `App.tsx`:

```typescript
import { NewComponent } from './components/NewComponent';

function App() {
  const handleAction = () => {
    console.log('Action triggered');
  };

  return (
    <div>
      <NewComponent data="Test" onAction={handleAction} />
    </div>
  );
}
```

### API Client

Файл: `web/src/api.ts`

```typescript
const API_BASE_URL = 'http://localhost:8000';

export interface DetectionResult {
  detections: Detection[];
  cost_estimate: CostEstimate;
  processing_time: number;
}

export const detectDamage = async (file: File): Promise<DetectionResult> => {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(`${API_BASE_URL}/detect`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new Error(`API error: ${response.status}`);
  }

  return await response.json();
};
```

---

## Testing

### Backend Tests

Запуск тестов:

```bash
# Все тесты
pytest

# С coverage
pytest --cov=src --cov-report=html

# Конкретный файл
pytest tests/test_api.py

# Конкретный тест
pytest tests/test_api.py::test_detect_endpoint
```

Пример теста:

```python
import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_health_endpoint():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()

def test_detect_with_valid_image():
    """Test detection with valid image."""
    with open("data/test_samples/car_01.jpg", "rb") as f:
        response = client.post(
            "/detect",
            files={"file": ("test.jpg", f, "image/jpeg")}
        )
    assert response.status_code == 200
    data = response.json()
    assert "detections" in data
    assert isinstance(data["detections"], list)

def test_detect_with_invalid_file():
    """Test detection with invalid file."""
    response = client.post(
        "/detect",
        files={"file": ("test.txt", b"not an image", "text/plain")}
    )
    assert response.status_code == 400
```

### Frontend Tests

```bash
cd web

# Run tests
npm test

# With coverage
npm test -- --coverage
```

---

## Code Quality

### Форматирование кода

```bash
# Black (Python formatter)
black src/ tests/

# isort (import sorting)
isort src/ tests/

# Prettier (JavaScript/TypeScript)
cd web
npm run format
```

### Linting

```bash
# flake8 (Python linter)
flake8 src/ tests/

# mypy (type checking)
mypy src/

# ESLint (JavaScript/TypeScript)
cd web
npm run lint
```

### Pre-commit hooks

Создайте `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.1.0
    hooks:
      - id: black

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort

  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
```

Установка:

```bash
pip install pre-commit
pre-commit install
```

---

## Debugging

### Backend

Используйте VS Code debugger. Создайте `.vscode/launch.json`:

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Python: FastAPI",
      "type": "python",
      "request": "launch",
      "module": "uvicorn",
      "args": [
        "src.api.main:app",
        "--reload",
        "--host",
        "0.0.0.0",
        "--port",
        "8000"
      ],
      "jinja": true,
      "justMyCode": false
    }
  ]
}
```

### Frontend

Chrome DevTools или VS Code debugger для React.

---

## Performance Optimization

### Backend

1. **Кэширование результатов:**

```python
from functools import lru_cache

@lru_cache(maxsize=100)
def get_model_predictions(image_hash: str):
    # Cache predictions by image hash
    pass
```

2. **Async operations:**

```python
import asyncio

async def process_multiple_images(images: List[str]):
    tasks = [detect_async(img) for img in images]
    results = await asyncio.gather(*tasks)
    return results
```

3. **Batch processing:**

```python
def detect_batch(images: List[np.ndarray]):
    # Process multiple images at once
    return model(images)
```

### Frontend

1. **Lazy loading:**

```typescript
import React, { lazy, Suspense } from 'react';

const HeavyComponent = lazy(() => import('./HeavyComponent'));

function App() {
  return (
    <Suspense fallback={<div>Loading...</div>}>
      <HeavyComponent />
    </Suspense>
  );
}
```

2. **Image optimization:**

```typescript
const optimizeImage = (file: File): Promise<Blob> => {
  return new Promise((resolve) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      const img = new Image();
      img.onload = () => {
        const canvas = document.createElement('canvas');
        const maxSize = 1920;
        
        let width = img.width;
        let height = img.height;
        
        if (width > height && width > maxSize) {
          height *= maxSize / width;
          width = maxSize;
        } else if (height > maxSize) {
          width *= maxSize / height;
          height = maxSize;
        }
        
        canvas.width = width;
        canvas.height = height;
        
        const ctx = canvas.getContext('2d')!;
        ctx.drawImage(img, 0, 0, width, height);
        
        canvas.toBlob((blob) => resolve(blob!), 'image/jpeg', 0.85);
      };
      img.src = e.target!.result as string;
    };
    reader.readAsDataURL(file);
  });
};
```

---

## Contribution Guidelines

1. **Создайте issue** перед началом работы
2. **Fork** репозиторий
3. **Создайте feature branch**: `git checkout -b feature/amazing-feature`
4. **Commit** изменения: `git commit -m 'Add amazing feature'`
5. **Push** в branch: `git push origin feature/amazing-feature`
6. **Откройте Pull Request**

### Требования к PR:

- [ ] Все тесты проходят
- [ ] Код отформатирован (black, prettier)
- [ ] Добавлены тесты для новой функциональности
- [ ] Документация обновлена
- [ ] Нет конфликтов с main branch

---

## Useful Commands

```bash
# Backend
uvicorn src.api.main:app --reload    # Dev server
pytest --cov=src                     # Tests with coverage
black src/                           # Format code
mypy src/                           # Type checking

# Frontend
npm run dev                         # Dev server
npm run build                       # Production build
npm test                           # Run tests
npm run lint                       # Lint code

# Docker
docker-compose up --build          # Build and run
docker-compose logs -f backend     # View logs
docker-compose exec backend bash   # Shell into container
```

---

## Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [React Documentation](https://react.dev/)
- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [TypeScript Handbook](https://www.typescriptlang.org/docs/)
- [pytest Documentation](https://docs.pytest.org/)

---

**Questions?** Open an [issue](https://github.com/yourusername/vehicle-damage-detection/issues) or join our [Discord](https://discord.gg/yourserver)
