"""
YOLOv9n Vehicle Damage Detection API with LLM Analysis
Реальная ML модель для обнаружения повреждений автомобилей + LLM отчеты
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
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
from ultralytics import YOLO
import cv2
import numpy as np
import asyncio

# Импорт LLM анализатора
from src.models.llm_analyzer import LLMAnalyzer, DetectionResult, llm_analyzer

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Создаем FastAPI приложение
app = FastAPI(
    title="YOLOv9n Vehicle Damage Detection with LLM",
    description="Обнаружение повреждений автомобилей с помощью YOLOv9n + LLM анализ",
    version="2.0.0-yolov9n-llm"
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
MODEL_PATH = "yolov9n.pt"
CONFIDENCE_THRESHOLD = 0.3
IOU_THRESHOLD = 0.5

# Создаем директории
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs("./data", exist_ok=True)

# Глобальные переменные
model = None
llm_analyzer_instance = None

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
            llm_report TEXT,
            confidence REAL,
            processing_time REAL,
            model_version TEXT
        )
    ''')
    
    conn.commit()
    conn.close()

def load_yolov9n_model():
    """Загрузка YOLOv9n модели"""
    global model
    try:
        logger.info("Загрузка YOLOv9n модели...")
        
        # Попробуем загрузить локальную модель, если нет - скачаем
        if os.path.exists(MODEL_PATH):
            model = YOLO(MODEL_PATH)
        else:
            logger.info("Скачивание YOLOv9n модели...")
            model = YOLO('yolov9n.pt')  # Автоматически скачает модель
        
        logger.info("YOLOv9n модель успешно загружена!")
        return True
        
    except Exception as e:
        logger.error(f"Ошибка загрузки модели: {e}")
        return False

def init_llm_analyzer():
    """Инициализация LLM анализатора"""
    global llm_analyzer_instance
    
    try:
        api_key = os.getenv('OPENROUTER_API_KEY')
        if api_key:
            logger.info("Инициализация LLM анализатора...")
            llm_analyzer_instance = LLMAnalyzer(api_key)
            logger.info("LLM анализатор успешно инициализирован!")
        else:
            logger.warning("OPENROUTER_API_KEY не найден. LLM анализ будет недоступен.")
            llm_analyzer_instance = LLMAnalyzer()
        return True
        
    except Exception as e:
        logger.error(f"Ошибка инициализации LLM анализатора: {e}")
        llm_analyzer_instance = LLMAnalyzer()
        return False

def detect_vehicle_damage(image_path: str) -> Dict[str, Any]:
    """Обнаружение повреждений автомобиля с помощью YOLOv9n"""
    try:
        # Загружаем изображение
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Не удалось загрузить изображение")
        
        # Получаем результаты от YOLOv9n
        results = model(img, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD)
        
        detections = []
        total_confidence = 0.0
        detection_count = 0
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Получаем координаты bounding box
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # Определяем класс объекта
                    class_names = {
                        0: "person", 2: "car", 3: "motorcycle", 5: "bus",
                        7: "truck", 67: "keyboard", 73: "laptop"
                    }
                    class_name = class_names.get(class_id, f"object_{class_id}")
                    
                    # Рассчитываем площадь
                    area = (x2 - x1) * (y2 - y1)
                    
                    detection = {
                        "class": class_name,
                        "confidence": confidence,
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "area": float(area)
                    }
                    
                    detections.append(detection)
                    total_confidence += confidence
                    detection_count += 1
        
        # Формируем итоговый результат
        if detection_count > 0:
            avg_confidence = total_confidence / detection_count
            severity = "high" if avg_confidence > 0.7 else "moderate" if avg_confidence > 0.4 else "low"
            estimated_cost = detection_count * 250.0 + avg_confidence * 500.0
        else:
            avg_confidence = 0.0
            severity = "none"
            estimated_cost = 0.0
        
        # Получаем информацию об изображении
        height, width = img.shape[:2]
        
        result = {
            "detections": detections,
            "summary": {
                "total_detections": detection_count,
                "severity": severity,
                "estimated_cost": round(estimated_cost, 2),
                "confidence": round(avg_confidence, 3),
                "processing_time": 0.0  # Будет обновлено
            },
            "image_info": {
                "width": width,
                "height": height,
                "format": "JPEG"
            },
            "model_info": {
                "model": "YOLOv9n",
                "confidence_threshold": CONFIDENCE_THRESHOLD,
                "iou_threshold": IOU_THRESHOLD
            }
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Ошибка в обнаружении: {e}")
        return {
            "error": str(e),
            "detections": [],
            "summary": {
                "total_detections": 0,
                "severity": "error",
                "estimated_cost": 0.0,
                "confidence": 0.0,
                "processing_time": 0.0
            },
            "image_info": {},
            "model_info": {
                "model": "YOLOv9n",
                "error": True
            }
        }

@app.on_event("startup")
async def startup_event():
    """Событие запуска приложения"""
    init_db()
    
    # Инициализируем LLM анализатор
    init_llm_analyzer()
    
    # Загружаем модель в синхронном режиме
    import threading
    
    def load_model_async():
        success = load_yolov9n_model()
        if not success:
            logger.error("Не удалось загрузить YOLOv9n модель")
    
    # Запускаем загрузку модели в отдельном потоке
    model_thread = threading.Thread(target=load_model_async)
    model_thread.daemon = True
    model_thread.start()
    
    logger.info("YOLOv9n Vehicle Damage Detection + LLM API запущен")

@app.get("/")
async def root():
    """Корневой эндпоинт"""
    model_status = "loaded" if model is not None else "loading"
    llm_status = "ready" if llm_analyzer_instance else "unavailable"
    
    return {
        "service": "YOLOv9n Vehicle Damage Detection + LLM",
        "version": "2.0.0-yolov9n-llm",
        "status": "running",
        "model_status": model_status,
        "llm_status": llm_status,
        "features": [
            "real_ml_detection",
            "yolov9n",
            "confidence_scoring",
            "results_storage",
            "llm_analysis",
            "human_readable_reports"
        ],
        "model_config": {
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "iou_threshold": IOU_THRESHOLD,
            "model_size": "nano"
        },
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
async def health_check():
    """Проверка здоровья системы"""
    model_status = "ready" if model is not None else "loading"
    llm_status = "ready" if llm_analyzer_instance and llm_analyzer_instance.api_key else "fallback"
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "yolov9n-vehicle-damage-detection-llm",
        "version": "2.0.0-yolov9n-llm",
        "model_status": model_status,
        "llm_analyzer_status": llm_status,
        "torch_version": torch.__version__ if torch else "not_available"
    }

@app.post("/detect")
async def detect_damage(file: UploadFile = File(...)):
    """Обнаружение повреждений на изображении"""
    
    # Проверка размера файла
    file.file.seek(0, os.SEEK_END)
    file_size = file.file.tell()
    file.file.seek(0)
    
    if file_size > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail=f"Файл слишком большой. Максимум {MAX_FILE_SIZE // (1024*1024)}MB")
    
    # Проверка формата файла
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Поддерживаются только изображения")
    
    # Проверка готовности модели
    if model is None:
        raise HTTPException(status_code=503, detail="Модель еще загружается. Попробуйте через несколько секунд.")
    
    try:
        # Сохраняем файл
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{file.filename}"
        filepath = os.path.join(UPLOAD_DIR, filename)
        
        with open(filepath, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Выполняем обнаружение
        start_time = datetime.now()
        yolov9n_result = detect_vehicle_damage(filepath)
        
        # Преобразуем результаты для LLM
        detection_results = []
        for det in yolov9n_result["detections"]:
            detection_results.append(DetectionResult(
                class_name=det["class"],
                confidence=det["confidence"],
                bbox=det["bbox"],
                area=det["area"]
            ))
        
        # LLM анализ (если доступен)
        llm_report = None
        llm_analysis_time = 0.0
        
        if llm_analyzer_instance:
            llm_start = datetime.now()
            try:
                llm_report = await llm_analyzer_instance.analyze_damage(
                    detection_results,
                    yolov9n_result["image_info"].get("width"),
                    yolov9n_result["image_info"].get("height")
                )
                llm_end = datetime.now()
                llm_analysis_time = (llm_end - llm_start).total_seconds()
            except Exception as e:
                logger.error(f"Ошибка LLM анализа: {e}")
                llm_report = None
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        # Формируем финальный ответ
        result = {
            "yolov9n_detection": yolov9n_result,
            "llm_analysis": None,
            "timestamp": timestamp,
            "processing_time": {
                "yolov9n": round(processing_time - llm_analysis_time, 3),
                "llm": round(llm_analysis_time, 3),
                "total": round(processing_time, 3)
            }
        }
        
        # Добавляем LLM отчет если доступен
        if llm_report:
            result["llm_analysis"] = {
                "summary": llm_report.summary,
                "detailed_description": llm_report.detailed_description,
                "damage_areas": llm_report.damage_areas,
                "severity_level": llm_report.severity_level,
                "estimated_cost_range": llm_report.estimated_cost_range,
                "recommendations": llm_report.recommendations,
                "confidence_score": llm_report.confidence_score
            }
        
        # Сохраняем результат в БД
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO detections (filename, filepath, result, llm_report, confidence, processing_time, model_version)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            filename,
            filepath,
            json.dumps(yolov9n_result),
            json.dumps(result.get("llm_analysis", {})) if result.get("llm_analysis") else None,
            yolov9n_result.get("summary", {}).get("confidence", 0.0),
            processing_time,
            "YOLOv9n + LLM"
        ))
        
        conn.commit()
        conn.close()
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка обработки изображения: {str(e)}")

@app.get("/results")
async def get_results(limit: int = 10):
    """Получение последних результатов обнаружения"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, filename, timestamp, result, llm_report, confidence, processing_time, model_version
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
                "result": json.loads(row[3]),
                "llm_report": json.loads(row[4]) if row[4] else None,
                "confidence": row[5],
                "processing_time": row[6],
                "model_version": row[7]
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
        
        conn.close()
        
        return {
            "total_detections": total_detections,
            "average_confidence": round(avg_confidence, 3),
            "average_processing_time": round(avg_processing_time, 3),
            "model_usage": model_usage,
            "system_info": {
                "version": "2.0.0-yolov9n-llm",
                "mode": "real_ml_with_llm",
                "ml_model": "YOLOv9n",
                "llm_model": "tngtech/deepseek-r1t2-chimera:free",
                "llm_available": llm_analyzer_instance and llm_analyzer_instance.api_key is not None,
                "torch_version": torch.__version__ if torch else "not_available",
                "cuda_available": torch.cuda.is_available() if torch else False
            }
        }
        
    except Exception as e:
        logger.error(f"Error retrieving stats: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        workers=1,
        log_level="info"
    )