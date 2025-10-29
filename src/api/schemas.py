"""
Pydantic schemas for API request/response validation.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any
from datetime import datetime


class DetectionResponse(BaseModel):
    """Schema for detection results."""
    class_name: str
    confidence: float
    bbox: List[int]  # [x1, y1, x2, y2]
    area: int


class YOLOv9DetectionSummary(BaseModel):
    """Schema for YOLOv9n detection summary."""
    detections: List[DetectionResponse]
    total_detections: int
    severity: str
    estimated_cost: float
    confidence: float
    processing_time: float


class HumanReadableReport(BaseModel):
    """Schema for LLM human-readable damage report."""
    summary: str = Field(..., description="Краткое описание ситуации")
    detailed_description: str = Field(..., description="Подробное описание повреждений")
    damage_areas: List[str] = Field(..., description="Список поврежденных областей")
    severity_level: str = Field(..., description="Уровень серьезности")
    estimated_cost_range: str = Field(..., description="Диапазон стоимости ремонта")
    recommendations: List[str] = Field(..., description="Рекомендации")
    confidence_score: float = Field(..., description="Уровень уверенности в анализе")


class LLMAnalysisResponse(BaseModel):
    """Schema for complete LLM analysis response."""
    raw_detection: YOLOv9DetectionSummary
    human_report: HumanReadableReport
    image_info: Dict[str, Any]
    model_info: Dict[str, Any]
    llm_analysis_time: float


class DetectionResponse(BaseModel):
    """Schema for detection results."""
    num_detections: int
    detections: List[Dict[str, Any]]
    inference_time: float


class ClassificationResponse(BaseModel):
    """Schema for damage classification results."""
    severity: Optional[str]
    damage_count: int
    total_damage_area: int
    area_ratio: float
    avg_confidence: float
    damage_types: Dict[str, int]


class CostEstimateResponse(BaseModel):
    """Schema for cost estimation results."""
    estimated_cost: float
    min_cost: float
    max_cost: float
    currency: str
    breakdown: Dict[str, Any]
    labor_cost: float
    parts_cost: float


class AnalysisResponse(BaseModel):
    """Complete analysis response schema."""
    id: int
    image_filename: str
    image_url: Optional[str]
    detection: DetectionResponse
    classification: ClassificationResponse
    cost_estimate: CostEstimateResponse
    visualization_url: Optional[str]
    total_processing_time: float
    created_at: datetime

    class Config:
        from_attributes = True


class AnalysisCreate(BaseModel):
    """Schema for creating new analysis."""
    image_filename: str
    client_id: Optional[str]
    session_id: Optional[str]
    notes: Optional[str]


class AnalysisSummary(BaseModel):
    """Lightweight summary of analysis results."""
    id: int
    image_filename: str
    severity: Optional[str]
    damage_count: int
    estimated_cost: float
    currency: str
    created_at: datetime

    class Config:
        from_attributes = True


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    model_loaded: bool
    database_connected: bool
    llm_analyzer_status: str = Field(..., description="Статус LLM анализатора")


class ErrorResponse(BaseModel):
    """Error response schema."""
    error: str
    detail: Optional[str]
    status_code: int
