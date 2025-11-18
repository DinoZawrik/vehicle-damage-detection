"""
Улучшенный YOLOv9n Vehicle Damage Detection API с мультимодальной системой.

Комбинирует YOLO, SAM, CLIP и LLM для SOTA анализа повреждений автомобилей.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import uvicorn
import sqlite3
import os
import shutil
from datetime import datetime
from typing import List, Dict, Any
import json
import logging
import torch
import cv2
import numpy as np
import asyncio

# Импорт мультимодального pipeline
from src.models.multi_modal_pipeline import MultiModalPipeline

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Создаем FastAPI приложение
app = FastAPI(
    title="Multi-Modal Vehicle Damage Detection API",
    description="SOTA система анализа повреждений автомобилей с YOLO + SAM + CLIP + LLM",
    version="4.0.0-multi-modal"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Конфигурация
UPLOAD_DIR = "./uploads"
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
DB_PATH = "./data/detection.db"
CONFIDENCE_THRESHOLD = 0.35
MODEL_PATH = "yolov8n.pt"

# Создаем директории
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs("./data", exist_ok=True)

# Глобальные переменные
pipeline = None

def init_db():
    """Инициализация SQLite базы данных"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            filepath TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            result TEXT,
            confidence REAL,
            processing_time REAL,
            model_version TEXT,
            modalities TEXT
        )
    ''')
    
    conn.commit()
    conn.close()

def init_pipeline():
    """Инициализация мультимодального pipeline"""
    global pipeline
    try:
        logger.info("Инициализация мультимодального pipeline...")
        
        pipeline = MultiModalPipeline(
            yolo_model_path=MODEL_PATH,
            yolo_conf_threshold=CONFIDENCE_THRESHOLD,
            enable_sam=True,
            enable_clip=True,
            enable_llm=True,
            device="auto"
        )
        
        logger.info("✅ Мультимодальный pipeline успешно инициализирован!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Ошибка инициализации pipeline: {e}")
        return False

@app.on_event("startup")
async def startup_event():
    """Событие запуска приложения"""
    global pipeline

    init_db()

    # Инициализируем pipeline
    logger.info("🚀 Запуск мультимодальной системы анализа повреждений...")
    if not init_pipeline():
        logger.warning("⚠️ Pipeline не инициализирован - приложение может работать нестабильно")

@app.get("/")
async def root():
    """Корневой эндпоинт"""
    pipeline_status = "loaded" if pipeline is not None else "loading"
    
    return {
        "service": "Multi-Modal Vehicle Damage Detection API",
        "version": "4.0.0-multi-modal",
        "status": "running",
        "pipeline_status": pipeline_status,
        "features": [
            "yolo_detection",
            "sam_segmentation",
            "clip_semantic_analysis",
            "llm_reporting",
            "ensemble_predictions",
            "precise_area_calculation",
            "professional_reports"
        ],
        "model_config": {
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "model_size": "nano",
            "modalities": ["yolo", "sam", "clip", "llm"]
        },
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
async def health_check():
    """Проверка здоровья системы"""
    pipeline_status = "ready" if pipeline is not None else "loading"
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "multi-modal-vehicle-damage-detection",
        "version": "4.0.0-multi-modal",
        "pipeline_status": pipeline_status,
        "torch_version": torch.__version__ if torch else "not_available",
        "cuda_available": torch.cuda.is_available() if torch else False,
        "modalities_available": {
            "yolo": pipeline is not None,
            "sam": pipeline is not None and hasattr(pipeline, 'sam_segmentor') and pipeline.sam_segmentor is not None,
            "clip": pipeline is not None and hasattr(pipeline, 'clip_analyzer') and pipeline.clip_analyzer is not None,
            "llm": pipeline is not None and hasattr(pipeline, 'llm_analyzer') and pipeline.llm_analyzer is not None
        }
    }

async def process_image(
    file: UploadFile,
    include_sam: bool = True,
    include_clip: bool = True,
    include_llm: bool = True,
    visualize: bool = True
) -> Dict[str, Any]:
    """Обработка изображения с настраиваемыми модальностями"""
    
    global pipeline
    
    # Проверка размера файла
    file.file.seek(0, os.SEEK_END)
    file_size = file.file.tell()
    file.file.seek(0)

    if file_size > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail=f"Файл слишком большой. Максимум {MAX_FILE_SIZE // (1024*1024)}MB")

    # Проверка формата файла
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Поддерживаются только изображения")

    # Гарантированная загрузка pipeline
    if pipeline is None:
        logger.warning("⚠️ Pipeline не загружен! Инициализирую сейчас...")
        if not init_pipeline():
            raise HTTPException(status_code=500, detail="Ошибка инициализации pipeline")
    
    try:
        # Сохраняем файл
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{file.filename}"
        filepath = os.path.join(UPLOAD_DIR, filename)

        with open(filepath, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Выполняем анализ
        logger.info(f"Анализ повреждений в файле {filename}...")
        start_time = datetime.now()

        # Загружаем изображение
        img = cv2.imread(filepath)
        if img is None:
            raise ValueError("Не удалось загрузить изображение")

        # Создаем временный pipeline с настройками модальностей
        temp_pipeline = MultiModalPipeline(
            yolo_model_path=MODEL_PATH,
            yolo_conf_threshold=CONFIDENCE_THRESHOLD,
            enable_sam=include_sam,
            enable_clip=include_clip,
            enable_llm=include_llm,
            device="auto"
        )

        # Выполняем анализ
        analysis_results = temp_pipeline.analyze_image(img, visualize=visualize)

        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        # Собираем информацию о модальностях
        modalities = []
        if include_sam:
            modalities.append("sam")
        if include_clip:
            modalities.append("clip")
        if include_llm:
            modalities.append("llm")
        
        # Формируем результат
        result = {
            "analysis_results": analysis_results,
            "timestamp": timestamp,
            "processing_time": processing_time,
            "modalities": modalities,
            "config": {
                "include_sam": include_sam,
                "include_clip": include_clip,
                "include_llm": include_llm,
                "visualize": visualize
            }
        }
        
        # Сохраняем результат в БД
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Получаем основную информацию для базы
        yolo_detections = analysis_results.get("yolo", {}).get("detections", [])
        avg_confidence = np.mean([d.get("confidence", 0.0) for d in yolo_detections]) if yolo_detections else 0.0
        
        cursor.execute('''
            INSERT INTO detections (filename, filepath, result, confidence, processing_time, model_version, modalities)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            filename,
            filepath,
            json.dumps(result),
            avg_confidence,
            processing_time,
            f"Multi-Modal v4.0 (SAM: {include_sam}, CLIP: {include_clip}, LLM: {include_llm})",
            json.dumps(modalities)
        ))
        
        conn.commit()
        conn.close()
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка обработки изображения: {str(e)}")

@app.post("/detect")
async def detect_damage(
    file: UploadFile = File(...),
    include_sam: bool = Query(True, description="Включить SAM сегментацию"),
    include_clip: bool = Query(True, description="Включить CLIP анализ"),
    include_llm: bool = Query(True, description="Включить LLM анализ"),
    visualize: bool = Query(True, description="Генерировать визуализацию")
):
    """Унифицированный endpoint для анализа повреждений с настраиваемыми модальностями"""
    
    return await process_image(file, include_sam, include_clip, include_llm, visualize)

@app.post("/detect/basic")
async def detect_damage_basic(file: UploadFile = File(...)):
    """Базовый анализ только с YOLO (быстрый)"""
    return await process_image(file, include_sam=False, include_clip=False, include_llm=False, visualize=True)

@app.post("/detect/segmented")
async def detect_damage_segmented(file: UploadFile = File(...)):
    """Анализ с YOLO + SAM сегментацией (точные маски)"""
    return await process_image(file, include_sam=True, include_clip=False, include_llm=False, visualize=True)

@app.post("/detect/semantic")
async def detect_damage_semantic(file: UploadFile = File(...)):
    """Анализ с YOLO + CLIP (семантическое понимание)"""
    return await process_image(file, include_sam=False, include_clip=True, include_llm=False, visualize=True)

@app.post("/detect/professional")
async def detect_damage_professional(file: UploadFile = File(...)):
    """Профессиональный анализ со всеми модальностями"""
    return await process_image(file, include_sam=True, include_clip=True, include_llm=True, visualize=True)

@app.get("/results")
async def get_results(limit: int = Query(10, description="Количество результатов")):
    """Получение последних результатов анализа"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, filename, timestamp, result, confidence, processing_time, model_version, modalities
            FROM detections
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        results = []
        for row in rows:
            result_data = {
                "id": row[0],
                "filename": row[1],
                "timestamp": row[2],
                "result": json.loads(row[3]) if row[3] else {},
                "confidence": row[4],
                "processing_time": row[5],
                "model_version": row[6],
                "modalities": json.loads(row[7]) if row[7] else []
            }
            results.append(result_data)
        
        return {"results": results, "total": len(results)}
        
    except Exception as e:
        logger.error(f"Error retrieving results: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка получения результатов: {str(e)}")

@app.get("/stats")
async def get_stats():
    """Получение статистики системы"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM detections')
        total_detections = cursor.fetchone()[0]
        
        cursor.execute('SELECT AVG(confidence) FROM detections WHERE confidence > 0')
        avg_confidence = cursor.fetchone()[0] or 0.0
        
        cursor.execute('SELECT AVG(processing_time) FROM detections WHERE processing_time > 0')
        avg_processing_time = cursor.fetchone()[0] or 0.0
        
        cursor.execute('SELECT model_version, COUNT(*) FROM detections GROUP BY model_version')
        model_usage = dict(cursor.fetchall())
        
        cursor.execute('SELECT modalities, COUNT(*) FROM detections GROUP BY modalities')
        modality_usage = dict(cursor.fetchall())
        
        conn.close()
        
        return {
            "total_detections": total_detections,
            "average_confidence": round(avg_confidence, 3),
            "average_processing_time": round(avg_processing_time, 3),
            "model_usage": model_usage,
            "modality_usage": modality_usage,
            "system_info": {
                "version": "4.0.0-multi-modal",
                "mode": "multi_modal_sota",
                "modalities": ["yolo", "sam", "clip", "llm"],
                "torch_version": torch.__version__ if torch else "not_available",
                "cuda_available": torch.cuda.is_available() if torch else False,
                "pipeline_available": pipeline is not None
            }
        }
        
    except Exception as e:
        logger.error(f"Error retrieving stats: {e}")
        return {"error": str(e)}

@app.get("/summary/{result_id}")
async def get_analysis_summary(result_id: int):
    """Получение текстового summary анализа"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('SELECT result FROM detections WHERE id = ?', (result_id,))
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            raise HTTPException(status_code=404, detail="Результат не найден")
        
        result_data = json.loads(row[0]) if row[0] else {}
        analysis_results = result_data.get("analysis_results", {})
        
        if not analysis_results:
            return {"summary": "Нет данных для генерации summary"}
        
        # Генерируем summary
        summary = pipeline.get_analysis_summary(analysis_results) if pipeline else "Pipeline недоступен"
        
        return {
            "result_id": result_id,
            "summary": summary,
            "analysis_results": analysis_results
        }
        
    except Exception as e:
        logger.error(f"Error generating summary: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка генерации summary: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        workers=1,
        log_level="info"
    )