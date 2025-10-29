"""
Ultra-light Vehicle Damage Detection API
Минимальная версия для быстрого тестирования
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

# Настройка логирования
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Создаем FastAPI приложение
app = FastAPI(
    title="Ultra-Light Vehicle Damage Detection",
    description="Минимальная система обнаружения повреждений автомобилей",
    version="1.0.0-ultra-light"
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
MAX_FILE_SIZE = 2 * 1024 * 1024  # 2MB
DB_PATH = "./data/detection.db"

# Создаем директории
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs("./data", exist_ok=True)

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
            processing_time REAL
        )
    ''')
    
    conn.commit()
    conn.close()

def fake_detection(image_path: str) -> Dict[str, Any]:
    """Fake функция обнаружения для ultra-light версии"""
    try:
        # Получаем размер изображения
        with Image.open(image_path) as img:
            width, height = img.size
            
        # Fake результат обнаружения
        fake_result = {
            "detections": [
                {
                    "class": "damage",
                    "confidence": 0.85,
                    "bbox": [100, 100, 200, 150],
                    "area": 5000
                },
                {
                    "class": "scratch", 
                    "confidence": 0.72,
                    "bbox": [250, 180, 320, 220],
                    "area": 1200
                }
            ],
            "summary": {
                "total_damages": 2,
                "severity": "moderate",
                "estimated_cost": 450.0,
                "processing_time": 0.15
            },
            "image_info": {
                "width": width,
                "height": height,
                "format": "JPEG"
            }
        }
        
        return fake_result
        
    except Exception as e:
        logger.error(f"Error in fake detection: {e}")
        return {
            "error": str(e),
            "detections": [],
            "summary": {
                "total_damages": 0,
                "severity": "unknown",
                "estimated_cost": 0.0,
                "processing_time": 0.0
            }
        }

@app.on_event("startup")
async def startup_event():
    """Событие запуска приложения"""
    init_db()
    logger.info("Ultra-light Vehicle Damage Detection API запущен")

@app.get("/")
async def root():
    """Корневой эндпоинт"""
    return {
        "service": "Ultra-Light Vehicle Damage Detection",
        "version": "1.0.0-ultra-light",
        "status": "running",
        "features": ["image_upload", "fake_detection", "results_storage"],
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
async def health_check():
    """Проверка здоровья системы"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "ultra-light-vehicle-damage-detection",
        "version": "1.0.0-ultra-light"
    }

@app.post("/detect")
async def detect_damage(file: UploadFile = File(...)):
    """Обнаружение повреждений на изображении"""
    
    # Проверка размера файла
    file.file.seek(0, os.SEEK_END)
    file_size = file.file.tell()
    file.file.seek(0)
    
    if file_size > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="Файл слишком большой")
    
    # Проверка формата файла
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Поддерживаются только изображения")
    
    try:
        # Сохраняем файл
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{file.filename}"
        filepath = os.path.join(UPLOAD_DIR, filename)
        
        with open(filepath, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Выполняем fake обнаружение
        start_time = datetime.now()
        result = fake_detection(filepath)
        end_time = datetime.now()
        
        processing_time = (end_time - start_time).total_seconds()
        
        # Сохраняем результат в БД
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO detections (filename, filepath, result, confidence, processing_time)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            filename,
            filepath,
            json.dumps(result),
            result.get("summary", {}).get("confidence", 0.0),
            processing_time
        ))
        
        conn.commit()
        conn.close()
        
        # Обновляем результат с временем обработки
        result["summary"]["processing_time"] = processing_time
        result["filename"] = filename
        result["timestamp"] = timestamp
        
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
            SELECT id, filename, timestamp, result, confidence, processing_time
            FROM detections
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        results = []
        for row in rows:
            results.append({
                "id": row[0],
                "filename": row[1],
                "timestamp": row[2],
                "result": json.loads(row[3]),
                "confidence": row[4],
                "processing_time": row[5]
            })
        
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
        
        conn.close()
        
        return {
            "total_detections": total_detections,
            "average_confidence": round(avg_confidence, 3),
            "average_processing_time": round(avg_processing_time, 3),
            "system_info": {
                "version": "1.0.0-ultra-light",
                "mode": "demo",
                "ml_model": "fake_detection"
            }
        }
        
    except Exception as e:
        logger.error(f"Error retrieving stats: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(
        "ultra_light_api:app",
        host="0.0.0.0",
        port=8080,
        workers=1,
        log_level="warning"
    )
