"""
SQLAlchemy database models for storing analysis results.
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, Text
from sqlalchemy.sql import func
from .database import Base


class AnalysisResult(Base):
    """
    Model for storing vehicle damage analysis results.
    """
    __tablename__ = "analysis_results"

    id = Column(Integer, primary_key=True, index=True)

    # Image information
    image_filename = Column(String(255), nullable=False)
    image_url = Column(String(512))  # MinIO URL
    original_size = Column(String(50))  # e.g., "1920x1080"

    # Detection results
    num_detections = Column(Integer, default=0)
    detections = Column(JSON)  # Store detection details as JSON

    # Classification results
    severity = Column(String(20))  # minor, moderate, severe, critical
    damage_count = Column(Integer, default=0)
    total_damage_area = Column(Integer, default=0)
    area_ratio = Column(Float, default=0.0)
    avg_confidence = Column(Float, default=0.0)
    damage_types = Column(JSON)  # Distribution of damage types

    # Cost estimation
    estimated_cost = Column(Float, default=0.0)
    min_cost = Column(Float, default=0.0)
    max_cost = Column(Float, default=0.0)
    currency = Column(String(10), default="USD")
    cost_breakdown = Column(JSON)

    # Processing metadata
    inference_time = Column(Float)  # in seconds
    total_processing_time = Column(Float)
    model_version = Column(String(50))

    # Visualization
    visualization_url = Column(String(512))  # URL to visualized image

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Optional: user/client tracking
    client_id = Column(String(100))
    session_id = Column(String(100))

    # Notes or additional info
    notes = Column(Text)

    def __repr__(self):
        return f"<AnalysisResult(id={self.id}, severity={self.severity}, cost={self.estimated_cost})>"
