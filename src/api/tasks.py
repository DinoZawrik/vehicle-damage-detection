"""
Celery tasks for async processing of vehicle damage analysis.
"""

from celery import Celery
import os
import logging
import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Celery configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "vehicle_damage_tasks",
    broker=REDIS_URL,
    backend=REDIS_URL
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=300,  # 5 minutes max
)


@celery_app.task(name="analyze_image_async", bind=True)
def analyze_image_async(self, image_path: str):
    """
    Async task for analyzing vehicle damage in an image.

    Args:
        image_path: Path to the image file

    Returns:
        Analysis results dictionary
    """
    try:
        self.update_state(state="PROCESSING", meta={"status": "Loading image"})

        # Import here to avoid circular imports
        from src.models.pipeline import DamageAnalysisPipeline

        # Initialize pipeline
        pipeline = DamageAnalysisPipeline()

        self.update_state(state="PROCESSING", meta={"status": "Running analysis"})

        # Run analysis
        results = pipeline.analyze_image(image_path, visualize=False)

        self.update_state(state="PROCESSING", meta={"status": "Saving results"})

        return {
            "status": "completed",
            "results": {
                "num_detections": results["detection"]["num_detections"],
                "severity": results["classification"]["severity"],
                "estimated_cost": results["cost_estimate"]["estimated_cost"],
                "processing_time": results["total_time"]
            }
        }

    except Exception as e:
        logger.error(f"Error in async analysis: {e}")
        return {
            "status": "failed",
            "error": str(e)
        }


@celery_app.task(name="batch_analyze", bind=True)
def batch_analyze(self, image_paths: list):
    """
    Async task for batch processing multiple images.

    Args:
        image_paths: List of image file paths

    Returns:
        List of analysis results
    """
    try:
        from src.models.pipeline import DamageAnalysisPipeline

        pipeline = DamageAnalysisPipeline()
        results = []

        total = len(image_paths)
        for idx, img_path in enumerate(image_paths):
            self.update_state(
                state="PROCESSING",
                meta={"current": idx + 1, "total": total, "status": f"Processing image {idx + 1}/{total}"}
            )

            try:
                result = pipeline.analyze_image(img_path, visualize=False)
                results.append({
                    "image_path": img_path,
                    "status": "success",
                    "severity": result["classification"]["severity"],
                    "cost": result["cost_estimate"]["estimated_cost"]
                })
            except Exception as e:
                results.append({
                    "image_path": img_path,
                    "status": "error",
                    "error": str(e)
                })

        return {
            "status": "completed",
            "total_processed": total,
            "results": results
        }

    except Exception as e:
        logger.error(f"Error in batch analysis: {e}")
        return {
            "status": "failed",
            "error": str(e)
        }
