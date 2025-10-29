# Vehicle Damage Detection API Documentation

## Overview

The Vehicle Damage Detection API provides RESTful endpoints for analyzing vehicle damage through computer vision and machine learning. The system can detect damage, classify severity, and estimate repair costs.

## Base URL

```
Development: http://localhost:8000
Production: https://your-domain.com
```

## Authentication

Currently, the API does not require authentication. For production deployment, implement JWT or API key authentication.

## Endpoints

### Health Check

#### GET /health

Check the health status of the API and its dependencies.

**Response:**
```json
{
  "status": "healthy|degraded",
  "version": "1.0.0",
  "model_loaded": true,
  "database_connected": true
}
```

**Status Codes:**
- `200` - Service is healthy
- `503` - Service is degraded (model not loaded or database not connected)

---

### Analyze Image

#### POST /api/analyze

Analyze an uploaded image for vehicle damage.

**Request:**
- `Content-Type: multipart/form-data`
- `file` (required): Image file (JPEG, PNG, BMP)
- `client_id` (optional): Client identifier for tracking
- `session_id` (optional): Session identifier for tracking

**Example:**
```bash
curl -X POST "http://localhost:8000/api/analyze" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@car_image.jpg" \
  -F "client_id=client123" \
  -F "session_id=session456"
```

**Response:**
```json
{
  "id": 1,
  "image_filename": "uuid123.jpg",
  "image_url": "http://localhost:9000/vehicle-images/uuid123.jpg",
  "detection": {
    "num_detections": 3,
    "detections": [
      {
        "bbox": [100, 50, 300, 200],
        "confidence": 0.85,
        "class": "scratch"
      }
    ],
    "inference_time": 0.25
  },
  "classification": {
    "severity": "moderate",
    "damage_count": 3,
    "total_damage_area": 4500,
    "area_ratio": 0.08,
    "avg_confidence": 0.82,
    "damage_types": {
      "scratch": 2,
      "dent": 1
    }
  },
  "cost_estimate": {
    "estimated_cost": 650.00,
    "min_cost": 487.50,
    "max_cost": 812.50,
    "currency": "USD",
    "breakdown": {
      "scratch": {
        "count": 2,
        "unit_cost": 150.0,
        "total": 300.0
      },
      "dent": {
        "count": 1,
        "unit_cost": 300.0,
        "total": 300.0
      }
    },
    "labor_cost": 300.00,
    "parts_cost": 600.00
  },
  "visualization_url": "http://localhost:9000/analysis-results/vis_uuid123.jpg",
  "total_processing_time": 0.45,
  "created_at": "2023-12-01T10:30:00Z"
}
```

**Status Codes:**
- `200` - Analysis completed successfully
- `400` - Invalid image format or corrupted file
- `413` - File too large (exceeds max size)
- `422` - Validation error (missing required fields)
- `503` - ML model not available
- `500` - Internal server error

---

### Get Analysis Result

#### GET /api/results/{id}

Retrieve a specific analysis result by ID.

**Parameters:**
- `id` (path, required): Analysis result ID

**Response:**
```json
{
  "id": 1,
  "image_filename": "uuid123.jpg",
  "image_url": "http://localhost:9000/vehicle-images/uuid123.jpg",
  "detection": {
    "num_detections": 3,
    "detections": [...],
    "inference_time": 0.25
  },
  "classification": {
    "severity": "moderate",
    "damage_count": 3,
    "area_ratio": 0.08,
    "avg_confidence": 0.82,
    "damage_types": {...}
  },
  "cost_estimate": {
    "estimated_cost": 650.00,
    "min_cost": 487.50,
    "max_cost": 812.50,
    "currency": "USD",
    "breakdown": {...},
    "labor_cost": 300.00,
    "parts_cost": 600.00
  },
  "visualization_url": "http://localhost:9000/analysis-results/vis_uuid123.jpg",
  "total_processing_time": 0.45,
  "created_at": "2023-12-01T10:30:00Z"
}
```

**Status Codes:**
- `200` - Result retrieved successfully
- `404` - Result not found

---

### Get Analysis History

#### GET /api/history

Get a list of previous analysis results.

**Query Parameters:**
- `limit` (query, optional): Maximum number of results to return (default: 50)
- `offset` (query, optional): Number of results to skip (default: 0)

**Example:**
```bash
curl "http://localhost:8000/api/history?limit=10&offset=0"
```

**Response:**
```json
[
  {
    "id": 1,
    "image_filename": "uuid123.jpg",
    "severity": "moderate",
    "damage_count": 3,
    "estimated_cost": 650.00,
    "currency": "USD",
    "created_at": "2023-12-01T10:30:00Z"
  }
]
```

**Status Codes:**
- `200` - History retrieved successfully

---

### Delete Analysis Result

#### DELETE /api/results/{id}

Delete an analysis result by ID.

**Parameters:**
- `id` (path, required): Analysis result ID

**Response:**
```json
{
  "message": "Result 1 deleted successfully"
}
```

**Status Codes:**
- `200` - Result deleted successfully
- `404` - Result not found

---

## Error Responses

All endpoints may return error responses in the following format:

```json
{
  "error": "Error type",
  "detail": "Detailed error message",
  "status_code": 400
}
```

**Common Error Types:**
- `validation_error` - Invalid input data
- `file_too_large` - Uploaded file exceeds size limit
- `invalid_image_format` - Unsupported image format
- `service_unavailable` - Required service is not available
- `internal_server_error` - Unexpected server error

## Rate Limiting

Currently, no rate limiting is implemented. For production:
- Limit requests per minute per client
- Implement exponential backoff for rate-limited requests
- Use Redis for distributed rate limiting

## File Upload Limits

- **Maximum file size**: 50MB
- **Supported formats**: JPEG, PNG, BMP
- **Recommended resolution**: 640x480 to 1920x1080
- **Aspect ratio**: No restrictions

## Response Time Expectations

- **Health check**: < 100ms
- **Image analysis**: 200-500ms (CPU), 100-200ms (GPU)
- **History retrieval**: < 50ms
- **Result retrieval**: < 30ms

## SDK Examples

### Python
```python
import requests

# Analyze image
with open("car_image.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/api/analyze",
        files={"file": f}
    )
    
if response.status_code == 200:
    result = response.json()
    severity = result["classification"]["severity"]
    cost = result["cost_estimate"]["estimated_cost"]
    print(f"Damage severity: {severity}")
    print(f"Estimated cost: ${cost}")
```

### JavaScript
```javascript
// Analyze image
const formData = new FormData();
formData.append('file', imageFile);

fetch('http://localhost:8000/api/analyze', {
  method: 'POST',
  body: formData
})
.then(response => response.json())
.then(result => {
  console.log('Severity:', result.classification.severity);
  console.log('Cost:', result.cost_estimate.estimated_cost);
});
```

### cURL
```bash
# Basic analysis
curl -X POST "http://localhost:8000/api/analyze" \
  -F "file=@car_image.jpg"

# With tracking IDs
curl -X POST "http://localhost:8000/api/analyze" \
  -F "file=@car_image.jpg" \
  -F "client_id=client123" \
  -F "session_id=session456"

# Get history
curl "http://localhost:8000/api/history?limit=10"

# Get specific result
curl "http://localhost:8000/api/results/1"
```

## Best Practices

### Image Upload
1. **Format**: Use JPEG for photos, PNG for graphics
2. **Quality**: Higher resolution improves detection accuracy
3. **Lighting**: Ensure good lighting conditions
4. **Angle**: Multiple angles provide better analysis
5. **Size**: Optimize file size while maintaining quality

### Error Handling
1. **Implement retry logic** for transient failures
2. **Handle timeouts** appropriately (60s for analysis)
3. **Validate file size** before upload
4. **Check response status codes** and handle errors gracefully

### Performance
1. **Use concurrent requests** for batch processing
2. **Cache results** for identical images
3. **Monitor processing time** and optimize accordingly
4. **Implement progress tracking** for long-running operations

## WebSocket (Future Enhancement)

Real-time updates for long-running analysis tasks:

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/analysis/progress');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Progress:', data.progress);
  console.log('Status:', data.status);
};
```

## API Versioning

Current version: `v1`

Future versions will be available at:
- `v2` - Enhanced models and additional damage types
- `v3` - Real-time analysis and video processing

Version negotiation via headers:
```http
Accept: application/vnd.vehicle-damage.v1+json
```

## Changelog

### v1.0.0 (Current)
- Initial API release
- Basic damage detection and classification
- Cost estimation functionality
- RESTful endpoints
- File upload and storage

---

For additional support or questions, please refer to the main README.md or create an issue in the repository.