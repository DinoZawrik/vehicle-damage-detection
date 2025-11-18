"""
Vehicle Damage Detection API - MVP Version

Simplified FastAPI backend with YOLO-only detection.
Clean, simple, and effective.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import io
import os
import time
from datetime import datetime
from typing import Optional, Dict, Any
import logging
import numpy as np

# Import our simple pipeline
from src.models.simple_pipeline import SimpleDetectionPipeline
from src.models.damage_analyzer import DamageAnalyzer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Vehicle Damage Detection API",
    description="Simple and effective vehicle damage detection using YOLOv8",
    version="1.0.0-mvp"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
UPLOAD_DIR = "./data/uploads"
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png'}

# Model configuration
MODEL_PATH = os.getenv("YOLO_MODEL", "yolov8n.pt")
CONF_THRESHOLD = float(os.getenv("YOLO_CONFIDENCE", "0.35"))
DEVICE = os.getenv("YOLO_DEVICE", "cpu")

# Create directories
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Global pipeline instance (singleton pattern)
_pipeline: Optional[SimpleDetectionPipeline] = None
_analyzer: Optional[DamageAnalyzer] = None


def get_pipeline() -> SimpleDetectionPipeline:
    """Get or create pipeline instance (singleton)."""
    global _pipeline
    if _pipeline is None:
        logger.info("🚀 Initializing detection pipeline...")
        _pipeline = SimpleDetectionPipeline(
            model_path=MODEL_PATH,
            conf_threshold=CONF_THRESHOLD,
            device=DEVICE
        )
        # Trigger model loading
        _pipeline.load_model()
    return _pipeline


def get_analyzer() -> DamageAnalyzer:
    """Get or create analyzer instance (singleton)."""
    global _analyzer
    if _analyzer is None:
        logger.info("🚀 Initializing damage analyzer...")
        _analyzer = DamageAnalyzer(currency="USD")
    return _analyzer


@app.on_event("startup")
async def startup_event():
    """Initialize app on startup."""
    logger.info("=" * 60)
    logger.info("🚗 Vehicle Damage Detection API - MVP")
    logger.info("=" * 60)
    logger.info(f"Model: {MODEL_PATH}")
    logger.info(f"Confidence: {CONF_THRESHOLD}")
    logger.info(f"Device: {DEVICE}")
    logger.info("=" * 60)
    
    # Pre-load model
    try:
        get_pipeline()
        get_analyzer()
        logger.info("✅ System ready!")
    except Exception as e:
        logger.error(f"❌ Startup error: {e}")


@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API info."""
    return {
        "service": "Vehicle Damage Detection API",
        "version": "1.0.0-mvp",
        "status": "running",
        "model": MODEL_PATH,
        "confidence_threshold": CONF_THRESHOLD,
        "device": DEVICE,
        "endpoints": {
            "detect": "POST /detect",
            "health": "GET /health",
            "models": "GET /models"
        },
        "docs": "/docs"
    }


@app.get("/health", tags=["General"])
async def health_check():
    """Health check endpoint."""
    try:
        pipeline = get_pipeline()
        return {
            "status": "ok",
            "model_loaded": pipeline.is_loaded,
            "model_path": pipeline.model_path,
            "device": pipeline.device,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "error",
                "message": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        )


@app.get("/models", tags=["Models"])
async def get_models_info():
    """Get information about loaded models."""
    try:
        pipeline = get_pipeline()
        return {
            "models": [
                {
                    "name": "yolov8n",
                    "path": pipeline.model_path,
                    "type": "object_detection",
                    "loaded": pipeline.is_loaded,
                    "confidence_threshold": pipeline.conf_threshold,
                    "iou_threshold": pipeline.iou_threshold,
                    "device": pipeline.device
                }
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/detect", tags=["Detection"])
async def detect_damage(
    file: UploadFile = File(...),
    confidence: Optional[float] = None
):
    """
    Detect damage in uploaded image.
    
    Args:
        file: Image file (JPEG, PNG)
        confidence: Override confidence threshold (optional)
        
    Returns:
        Detection results with damage analysis and cost estimate
    """
    start_time = time.time()
    
    # Validate file
    if not file:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Check file extension
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file format. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    # Read file
    try:
        contents = await file.read()
        
        # Check file size
        if len(contents) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Max size: {MAX_FILE_SIZE / 1024 / 1024}MB"
            )
        
        # Convert to image
        image = Image.open(io.BytesIO(contents))
        image_np = np.array(image.convert('RGB'))
        
        # Convert RGB to BGR for OpenCV
        image_bgr = image_np[:, :, ::-1].copy()
        
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to process image: {str(e)}"
        )
    
    # Run detection
    try:
        pipeline = get_pipeline()
        analyzer = get_analyzer()
        
        # Detect
        detection_result = pipeline.detect(
            image_bgr,
            conf=confidence
        )
        
        # Analyze
        analysis = analyzer.analyze(
            detection_result.detections,
            image_shape=(image_bgr.shape[0], image_bgr.shape[1])
        )
        
        # Add severity to detections
        detections_with_severity = analyzer.add_severity_to_detections(
            detection_result.detections.copy(),
            analysis.get('severity')
        )
        
        # Build response
        total_time = time.time() - start_time
        
        response = {
            "success": True,
            "timestamp": datetime.utcnow().isoformat(),
            "image_info": {
                "filename": file.filename,
                "width": image_bgr.shape[1],
                "height": image_bgr.shape[0]
            },
            "detections": detections_with_severity,
            "analysis": {
                "severity": analysis.get('severity'),
                "damage_count": analysis.get('damage_count'),
                "damage_types": analysis.get('damage_types'),
                "confidence_avg": analysis.get('avg_confidence')
            },
            "cost_estimate": analysis.get('cost_estimate'),
            "processing_time": round(total_time, 3),
            "model": {
                "name": "yolov8n",
                "confidence_threshold": confidence or pipeline.conf_threshold
            }
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Detection error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Detection failed: {str(e)}"
        )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "timestamp": datetime.utcnow().isoformat()
        }
    )


# Run server
if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
