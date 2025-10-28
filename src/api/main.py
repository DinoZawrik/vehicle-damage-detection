"""
FastAPI application for vehicle damage detection API.
"""

from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import cv2
import numpy as np
from typing import List
import logging
from datetime import datetime
import uuid

from . import models, schemas
from .database import engine, get_db, init_db
from .storage import StorageClient
from src.models.pipeline import DamageAnalysisPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create tables
models.Base.metadata.create_all(bind=engine)

# Initialize FastAPI app
app = FastAPI(
    title="Vehicle Damage Detection API",
    description="API for detecting and analyzing vehicle damage using ML",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
pipeline = None
storage = None


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global pipeline, storage

    logger.info("Starting up Vehicle Damage Detection API...")

    # Initialize ML pipeline
    try:
        pipeline = DamageAnalysisPipeline()
        logger.info("ML pipeline initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize ML pipeline: {e}")

    # Initialize storage client
    try:
        storage = StorageClient()
        logger.info("Storage client initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize storage: {e}")

    # Initialize database
    init_db()
    logger.info("Database initialized")


@app.get("/", tags=["Health"])
async def root():
    """Root endpoint."""
    return {
        "message": "Vehicle Damage Detection API",
        "version": "1.0.0",
        "status": "operational"
    }


@app.get("/health", response_model=schemas.HealthResponse, tags=["Health"])
async def health_check(db: Session = Depends(get_db)):
    """Health check endpoint."""
    model_loaded = pipeline is not None
    db_connected = False

    try:
        # Test database connection
        db.execute("SELECT 1")
        db_connected = True
    except Exception as e:
        logger.error(f"Database health check failed: {e}")

    return schemas.HealthResponse(
        status="healthy" if (model_loaded and db_connected) else "degraded",
        version="1.0.0",
        model_loaded=model_loaded,
        database_connected=db_connected
    )


@app.post("/api/analyze", response_model=schemas.AnalysisResponse, tags=["Analysis"])
async def analyze_image(
    file: UploadFile = File(...),
    client_id: str = None,
    session_id: str = None,
    db: Session = Depends(get_db)
):
    """
    Analyze an uploaded image for vehicle damage.

    Args:
        file: Uploaded image file
        client_id: Optional client identifier
        session_id: Optional session identifier
        db: Database session

    Returns:
        Complete analysis results
    """
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML pipeline not initialized"
        )

    try:
        # Read image file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid image format"
            )

        # Generate unique filename
        file_ext = file.filename.split(".")[-1]
        unique_filename = f"{uuid.uuid4()}.{file_ext}"

        # Upload original image to storage
        image_url = None
        if storage:
            try:
                image_url = storage.upload_image(contents, unique_filename)
            except Exception as e:
                logger.warning(f"Failed to upload image to storage: {e}")

        # Run analysis
        results = pipeline.analyze_image(image, visualize=True)

        # Upload visualization if available
        vis_url = None
        if storage and results.get("visualization") is not None:
            try:
                vis_filename = f"vis_{unique_filename}"
                _, vis_encoded = cv2.imencode(f".{file_ext}", results["visualization"])
                vis_url = storage.upload_result(vis_encoded.tobytes(), vis_filename)
            except Exception as e:
                logger.warning(f"Failed to upload visualization: {e}")

        # Store results in database
        db_result = models.AnalysisResult(
            image_filename=unique_filename,
            image_url=image_url,
            original_size=f"{image.shape[1]}x{image.shape[0]}",
            num_detections=results["detection"]["num_detections"],
            detections=results["detection"]["detections"],
            severity=results["classification"]["severity"],
            damage_count=results["classification"]["damage_count"],
            total_damage_area=results["classification"]["total_damage_area"],
            area_ratio=results["classification"]["area_ratio"],
            avg_confidence=results["classification"]["avg_confidence"],
            damage_types=results["classification"]["damage_types"],
            estimated_cost=results["cost_estimate"]["estimated_cost"],
            min_cost=results["cost_estimate"]["min_cost"],
            max_cost=results["cost_estimate"]["max_cost"],
            currency=results["cost_estimate"]["currency"],
            cost_breakdown=results["cost_estimate"]["breakdown"],
            inference_time=results["detection"]["inference_time"],
            total_processing_time=results["total_time"],
            model_version="yolov8n",
            visualization_url=vis_url,
            client_id=client_id,
            session_id=session_id
        )

        db.add(db_result)
        db.commit()
        db.refresh(db_result)

        # Format response
        response = schemas.AnalysisResponse(
            id=db_result.id,
            image_filename=db_result.image_filename,
            image_url=db_result.image_url,
            detection=schemas.DetectionResponse(
                num_detections=results["detection"]["num_detections"],
                detections=results["detection"]["detections"],
                inference_time=results["detection"]["inference_time"]
            ),
            classification=schemas.ClassificationResponse(**results["classification"]),
            cost_estimate=schemas.CostEstimateResponse(**results["cost_estimate"]),
            visualization_url=vis_url,
            total_processing_time=results["total_time"],
            created_at=db_result.created_at
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing image: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {str(e)}"
        )


@app.get("/api/results/{result_id}", response_model=schemas.AnalysisResponse, tags=["Analysis"])
async def get_result(result_id: int, db: Session = Depends(get_db)):
    """
    Retrieve analysis result by ID.

    Args:
        result_id: Analysis result ID
        db: Database session

    Returns:
        Analysis results
    """
    result = db.query(models.AnalysisResult).filter(models.AnalysisResult.id == result_id).first()

    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Result with ID {result_id} not found"
        )

    return schemas.AnalysisResponse(
        id=result.id,
        image_filename=result.image_filename,
        image_url=result.image_url,
        detection=schemas.DetectionResponse(
            num_detections=result.num_detections,
            detections=result.detections,
            inference_time=result.inference_time
        ),
        classification=schemas.ClassificationResponse(
            severity=result.severity,
            damage_count=result.damage_count,
            total_damage_area=result.total_damage_area,
            area_ratio=result.area_ratio,
            avg_confidence=result.avg_confidence,
            damage_types=result.damage_types
        ),
        cost_estimate=schemas.CostEstimateResponse(
            estimated_cost=result.estimated_cost,
            min_cost=result.min_cost,
            max_cost=result.max_cost,
            currency=result.currency,
            breakdown=result.cost_breakdown,
            labor_cost=result.cost_breakdown.get("labor", 0) if result.cost_breakdown else 0,
            parts_cost=result.cost_breakdown.get("parts", 0) if result.cost_breakdown else 0
        ),
        visualization_url=result.visualization_url,
        total_processing_time=result.total_processing_time,
        created_at=result.created_at
    )


@app.get("/api/history", response_model=List[schemas.AnalysisSummary], tags=["Analysis"])
async def get_history(
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db)
):
    """
    Get analysis history.

    Args:
        limit: Maximum number of results to return
        offset: Number of results to skip
        db: Database session

    Returns:
        List of analysis summaries
    """
    results = db.query(models.AnalysisResult)\
        .order_by(models.AnalysisResult.created_at.desc())\
        .limit(limit)\
        .offset(offset)\
        .all()

    return [
        schemas.AnalysisSummary(
            id=r.id,
            image_filename=r.image_filename,
            severity=r.severity,
            damage_count=r.damage_count,
            estimated_cost=r.estimated_cost,
            currency=r.currency,
            created_at=r.created_at
        )
        for r in results
    ]


@app.delete("/api/results/{result_id}", tags=["Analysis"])
async def delete_result(result_id: int, db: Session = Depends(get_db)):
    """
    Delete an analysis result.

    Args:
        result_id: Analysis result ID to delete
        db: Database session

    Returns:
        Success message
    """
    result = db.query(models.AnalysisResult).filter(models.AnalysisResult.id == result_id).first()

    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Result with ID {result_id} not found"
        )

    db.delete(result)
    db.commit()

    return {"message": f"Result {result_id} deleted successfully"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
