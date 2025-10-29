"""
LLM Analyzer для Vehicle Damage Detection
Интеграция с OpenRouter API для генерации человекочитаемых отчетов
"""

import os
import json
import httpx
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime

@dataclass
class DetectionResult:
    """Результат обнаружения объекта"""
    class_name: str
    confidence: float
    bbox: List[int]  # [x1, y1, x2, y2]
    area: int
    severity_assessment: Optional[str] = None

@dataclass
class HumanReadableReport:
    """Человекочитаемый отчет о повреждениях"""
    summary: str
    detailed_description: str
    damage_areas: List[str]
    severity_level: str
    estimated_cost_range: str
    recommendations: List[str]
    confidence_score: float

class LLMAnalyzer:
    """Анализатор повреждений с использованием LLM"""
    
    def __init__(self, openrouter_api_key: str = None):
        """
        Инициализация анализатора
        
        Args:
            openrouter_api_key: API ключ для OpenRouter
        """
        self.api_key = openrouter_api_key or os.getenv('OPENROUTER_API_KEY')
        self.base_url = "https://openrouter.ai/api/v1"
        self.model = "tngtech/deepseek-r1t2-chimera:free"
        
        if not self.api_key:
            print("⚠️ Предупреждение: OPENROUTER_API_KEY не установлен")
    
    async def analyze_damage(self, detections: List[DetectionResult], 
                           image_width: int = None, 
                           image_height: int = None) -> HumanReadableReport:
        """
        Анализ повреждений с использованием LLM
        
        Args:
            detections: Список обнаруженных объектов
            image_width: Ширина изображения
            image_height: Высота изображения
            
        Returns:
            HumanReadableReport: Человекочитаемый отчет
        """
        if not self.api_key:
            # Fallback к базовому анализу без LLM
            return self._fallback_analysis(detections)
        
        try:
            prompt = self._build_prompt(detections, image_width, image_height)
            response = await self._call_llm(prompt)
            return self._parse_llm_response(response, detections)
        except Exception as e:
            print(f"❌ Ошибка LLM анализа: {e}")
            return self._fallback_analysis(detections)
    
    def _build_prompt(self, detections: List[DetectionResult], 
                     image_width: int = None, 
                     image_height: int = None) -> str:
        """Построение промпта для LLM"""
        
        # Анализ обнаруженных объектов
        objects_info = []
        for det in detections:
            # Преобразование bbox в процентные координаты
            if image_width and image_height:
                x1_pct = round((det.bbox[0] / image_width) * 100, 1)
                y1_pct = round((det.bbox[1] / image_height) * 100, 1)
                x2_pct = round((det.bbox[2] / image_width) * 100, 1)
                y2_pct = round((det.bbox[3] / image_height) * 100, 1)
                location = f"область от ({x1_pct}%, {y1_pct}%) до ({x2_pct}%, {y2_pct}%)"
            else:
                location = f"область с координатами {det.bbox}"
            
            objects_info.append({
                "object": det.class_name,
                "confidence": round(det.confidence * 100, 1),
                "location": location,
                "area": det.area
            })
        
        # Определение потенциальных поврежденных объектов
        vehicle_objects = [obj for obj in objects_info if obj["object"] in ["car", "truck", "bus"]]
        other_objects = [obj for obj in objects_info if obj["object"] not in ["car", "truck", "bus"]]
        
        prompt = f"""
Ты - эксперт по оценке повреждений автомобилей. Проанализируй результаты компьютерного зрения и предоставь подробный отчет на русском языке.

ОБНАРУЖЕННЫЕ ОБЪЕКТЫ:
{json.dumps(objects_info, indent=2, ensure_ascii=False)}

ОБНАРУЖЕННЫЕ ТРАНСПОРТНЫЕ СРЕДСТВА:
{json.dumps(vehicle_objects, indent=2, ensure_ascii=False)}

ДРУГИЕ ОБЪЕКТЫ:
{json.dumps(other_objects, indent=2, ensure_ascii=False)}

Предоставь отчет в следующем формате (строго JSON):

{{
    "summary": "Краткое описание ситуации",
    "detailed_description": "Подробное описание повреждений с указанием местоположения",
    "damage_areas": ["список поврежденных областей"],
    "severity_level": "легкое|умеренное|тяжелое|критическое",
    "estimated_cost_range": "диапазон стоимости ремонта в рублях",
    "recommendations": ["список рекомендаций"],
    "confidence_score": 0.95
}}

Примеры интерпретации:
- Если обнаружен только автомобиль без других объектов: скорее всего повреждений нет или они минимальны
- Если обнаружены другие объекты рядом с автомобилем: возможны повреждения от столкновения
- Если несколько транспортных средств: возможно ДТП с множественными повреждениями
- Оцени серьезность по количеству и расположению обнаруженных объектов

Отвечай только в формате JSON, без дополнительного текста.
"""
        
        return prompt
    
    async def _call_llm(self, prompt: str) -> str:
        """Вызов LLM через OpenRouter API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://vehicle-damage-detection.com",
            "X-Title": "Vehicle Damage Detection"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system", 
                    "content": "Ты - эксперт по оценке повреждений автомобилей. Отвечай только в формате JSON."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": 1000,
            "temperature": 0.1
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload
            )
            
            if response.status_code != 200:
                raise Exception(f"API Error: {response.status_code}")
            
            data = response.json()
            return data["choices"][0]["message"]["content"]
    
    def _parse_llm_response(self, response: str, detections: List[DetectionResult]) -> HumanReadableReport:
        """Парсинг ответа LLM"""
        try:
            # Извлечение JSON из ответа
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx != -1 and end_idx != 0:
                json_str = response[start_idx:end_idx]
                llm_data = json.loads(json_str)
            else:
                raise ValueError("JSON не найден в ответе")
            
            # Создание отчета
            return HumanReadableReport(
                summary=llm_data.get("summary", "Анализ не удался"),
                detailed_description=llm_data.get("detailed_description", ""),
                damage_areas=llm_data.get("damage_areas", []),
                severity_level=llm_data.get("severity_level", "не определено"),
                estimated_cost_range=llm_data.get("estimated_cost_range", "не оценено"),
                recommendations=llm_data.get("recommendations", []),
                confidence_score=llm_data.get("confidence_score", 0.5)
            )
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"❌ Ошибка парсинга LLM ответа: {e}")
            return self._fallback_analysis(detections)
    
    def _fallback_analysis(self, detections: List[DetectionResult]) -> HumanReadableReport:
        """Fallback анализ без LLM"""
        vehicle_count = len([d for d in detections if d.class_name in ["car", "truck", "bus"]])
        other_objects = [d for d in detections if d.class_name not in ["car", "truck", "bus"]]
        
        # Базовый анализ
        if vehicle_count == 0:
            summary = "Транспортные средства не обнаружены на изображении"
            severity = "нет повреждений"
            cost_range = "0 - 0 рублей"
        elif len(other_objects) == 0:
            summary = "Обнаружен автомобиль(и) без явных признаков повреждений"
            severity = "минимальные повреждения"
            cost_range = "0 - 50,000 рублей"
        else:
            summary = f"Обнаружен автомобиль с {len(other_objects)} потенциально повреждающими объектами"
            severity = "умеренные повреждения"
            cost_range = "50,000 - 200,000 рублей"
        
        return HumanReadableReport(
            summary=summary,
            detailed_description=f"Система обнаружила {len(detections)} объектов: {', '.join([d.class_name for d in detections])}",
            damage_areas=[f"Область {d.class_name}" for d in detections],
            severity_level=severity,
            estimated_cost_range=cost_range,
            recommendations=["Рекомендуется профессиональная оценка повреждений"],
            confidence_score=0.6
        )

# Глобальный экземпляр анализатора
llm_analyzer = LLMAnalyzer()