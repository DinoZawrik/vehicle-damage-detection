# 🔌 API Documentation

Полная документация REST API для Vehicle Damage Detection System.

---

## Base URL

```
http://localhost:8000
```

В production замените на ваш домен.

---

## Аутентификация

В MVP версии аутентификация не требуется. Для production смотрите [DEPLOYMENT.md](DEPLOYMENT.md).

---

## Endpoints

### 1. Health Check

Проверка статуса API.

**Endpoint:** `GET /health`

**Response:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "timestamp": "2025-11-18T12:00:00.123456"
}
```

**Status Codes:**
- `200` - API работает
- `503` - Модель не загружена

**Example:**
```bash
curl http://localhost:8000/health
```

---

### 2. Detect Damage

Основной endpoint для обнаружения повреждений.

**Endpoint:** `POST /detect`

**Parameters:**
- `file` (required) - Изображение (multipart/form-data)
  - Formats: JPEG, PNG
  - Max size: 10MB

**Response:**
```json
{
  "detections": [
    {
      "type": "scratch",
      "confidence": 0.87,
      "bbox": [120, 340, 200, 380],
      "severity": "minor",
      "area": 4800
    },
    {
      "type": "dent",
      "confidence": 0.72,
      "bbox": [450, 200, 520, 260],
      "severity": "moderate",
      "area": 4200
    }
  ],
  "cost_estimate": {
    "min": 450,
    "max": 650,
    "currency": "USD"
  },
  "image_id": "abc123",
  "processing_time": 0.23,
  "timestamp": "2025-11-18T12:00:00"
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `detections` | Array | Список обнаруженных повреждений |
| `detections[].type` | String | Тип повреждения (scratch, dent, crack, etc.) |
| `detections[].confidence` | Float | Уверенность модели (0.0-1.0) |
| `detections[].bbox` | Array[4] | Bounding box [x1, y1, x2, y2] |
| `detections[].severity` | String | Серьезность (minor, moderate, severe) |
| `detections[].area` | Integer | Площадь повреждения в пикселях |
| `cost_estimate` | Object | Оценка стоимости ремонта |
| `cost_estimate.min` | Float | Минимальная стоимость |
| `cost_estimate.max` | Float | Максимальная стоимость |
| `cost_estimate.currency` | String | Валюта (USD, EUR, RUB) |
| `image_id` | String | Уникальный ID изображения |
| `processing_time` | Float | Время обработки в секундах |
| `timestamp` | String | ISO 8601 timestamp |

**Status Codes:**
- `200` - Успешная обработка
- `400` - Неверный формат файла
- `413` - Файл слишком большой
- `422` - Validation error
- `500` - Internal server error

**Examples:**

```bash
# Basic usage
curl -X POST "http://localhost:8000/detect" \
  -F "file=@car_damage.jpg"

# Python example
import requests

with open('car_damage.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/detect',
        files={'file': f}
    )
    result = response.json()
    print(result)

# JavaScript example
const formData = new FormData();
formData.append('file', fileInput.files[0]);

const response = await fetch('http://localhost:8000/detect', {
  method: 'POST',
  body: formData
});
const result = await response.json();
console.log(result);
```

---

### 3. Get Models Info

Получить информацию о доступных моделях.

**Endpoint:** `GET /models`

**Response:**
```json
{
  "models": [
    {
      "name": "yolov8n",
      "version": "8.0.0",
      "type": "detection",
      "classes": ["scratch", "dent", "crack", "broken_glass"],
      "loaded": true
    }
  ]
}
```

**Example:**
```bash
curl http://localhost:8000/models
```

---

### 4. Get Detection History

Получить историю детекций (опционально, если включена БД).

**Endpoint:** `GET /history`

**Query Parameters:**
- `limit` (optional) - Количество записей (default: 10)
- `offset` (optional) - Смещение (default: 0)

**Response:**
```json
{
  "total": 45,
  "items": [
    {
      "id": "abc123",
      "timestamp": "2025-11-18T12:00:00",
      "detections_count": 2,
      "cost_min": 450,
      "cost_max": 650
    }
  ]
}
```

**Example:**
```bash
curl "http://localhost:8000/history?limit=20&offset=0"
```

---

## Error Responses

Все ошибки возвращаются в формате:

```json
{
  "error": "Error type",
  "detail": "Detailed error message",
  "code": "ERROR_CODE"
}
```

### Common Error Codes:

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `INVALID_FILE_FORMAT` | 400 | Неподдерживаемый формат файла |
| `FILE_TOO_LARGE` | 413 | Файл превышает максимальный размер |
| `MODEL_NOT_LOADED` | 503 | YOLO модель не загружена |
| `PROCESSING_ERROR` | 500 | Ошибка обработки изображения |
| `VALIDATION_ERROR` | 422 | Ошибка валидации параметров |

**Example Error Response:**
```json
{
  "error": "Invalid file format",
  "detail": "Only JPEG and PNG images are supported",
  "code": "INVALID_FILE_FORMAT"
}
```

---

## Rate Limiting

В MVP версии rate limiting не реализован. Для production рекомендуется:

- Max 10 requests per minute per IP
- Max 100 requests per hour per IP

---

## Swagger UI

Интерактивная документация доступна по адресу:

```
http://localhost:8000/docs
```

ReDoc альтернатива:

```
http://localhost:8000/redoc
```

---

## SDKs

### Python Client

```python
import requests
from typing import Dict, Any

class DamageDetectionClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
    
    def detect(self, image_path: str) -> Dict[str, Any]:
        """Detect damage in image."""
        with open(image_path, 'rb') as f:
            response = requests.post(
                f"{self.base_url}/detect",
                files={'file': f}
            )
        response.raise_for_status()
        return response.json()
    
    def health(self) -> Dict[str, Any]:
        """Check API health."""
        response = requests.get(f"{self.base_url}/health")
        return response.json()

# Usage
client = DamageDetectionClient()
result = client.detect("car_damage.jpg")
print(result['detections'])
```

### JavaScript Client

```javascript
class DamageDetectionClient {
  constructor(baseUrl = 'http://localhost:8000') {
    this.baseUrl = baseUrl;
  }

  async detect(file) {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${this.baseUrl}/detect`, {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }

    return await response.json();
  }

  async health() {
    const response = await fetch(`${this.baseUrl}/health`);
    return await response.json();
  }
}

// Usage
const client = new DamageDetectionClient();
const result = await client.detect(fileInput.files[0]);
console.log(result.detections);
```

---

## WebSocket Support (Future)

В будущих версиях планируется WebSocket поддержка для real-time обновлений:

```
ws://localhost:8000/ws
```

---

## Versioning

API версия указывается в заголовке ответа:

```
X-API-Version: 1.0.0
```

---

## Best Practices

1. **Всегда проверяйте HTTP status codes**
2. **Обрабатывайте ошибки gracefully**
3. **Используйте retry логику для временных ошибок (5xx)**
4. **Кэшируйте результаты если возможно**
5. **Оптимизируйте изображения перед отправкой**
6. **Проверяйте health endpoint перед массовой обработкой**

---

## Performance Tips

- Изображения > 2000px по любой стороне автоматически ресайзятся
- JPEG качество 85% оптимально для баланса размер/качество
- Batch processing: используйте async/parallel requests

---

## Support

- 📖 [Installation Guide](INSTALLATION.md)
- 🚀 [Deployment Guide](DEPLOYMENT.md)
- 💻 [Development Guide](DEVELOPMENT.md)
- 🐛 [Report Issues](https://github.com/yourusername/vehicle-damage-detection/issues)

---

**API Version:** 1.0.0  
**Last Updated:** 2025-11-18
