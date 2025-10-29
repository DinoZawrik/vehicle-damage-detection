# Vehicle Damage Detection System - YOLOv9n
# Простая версия с реальной ML моделью

param(
    [string]$Action = "start",
    [switch]$Clean,
    [switch]$Logs
)

$ErrorActionPreference = "Stop"

Write-Host "=== Vehicle Damage Detection System (YOLOv9n) ===" -ForegroundColor Green
Write-Host "Версия: YOLOv9n (12MB модель)" -ForegroundColor Yellow
Write-Host "Ресурсы: 1GB RAM, 1 CPU core" -ForegroundColor Cyan

switch ($Action) {
    "start" {
        Write-Host "Запуск системы YOLOv9n..." -ForegroundColor Cyan
        
        # Создаем необходимые директории
        $dirs = @("data", "uploads", "models", "logs")
        foreach ($dir in $dirs) {
            if (!(Test-Path $dir)) {
                New-Item -ItemType Directory -Path $dir -Force | Out-Null
                Write-Host "Создана директория: $dir" -ForegroundColor Gray
            }
        }
        
        # Очистка старых контейнеров если нужно
        if ($Clean) {
            Write-Host "Очистка старых контейнеров..." -ForegroundColor Yellow
            try {
                docker-compose down --remove-orphans 2>$null
                docker system prune -f 2>$null
                Write-Host "Очистка завершена" -ForegroundColor Green
            }
            catch {
                Write-Host "Предупреждение при очистке: $_" -ForegroundColor Yellow
            }
        }
        
        # Запуск системы
        Write-Host "Запуск системы YOLOv9n (порт 8000)..." -ForegroundColor Cyan
        Write-Host "⚠️  Первая загрузка может занять 2-5 минут (скачивание модели ~12MB)" -ForegroundColor Yellow
        
        try {
            docker-compose up -d --build
            
            # Проверка статуса через 10 секунд
            Start-Sleep -Seconds 10
            $status = docker-compose ps --services --filter "status=running"
            
            if ($status -match "vehicle-damage-detector") {
                Write-Host ""
                Write-Host "✅ Система YOLOv9n запущена успешно!" -ForegroundColor Green
                Write-Host "🌐 API: http://localhost:8000" -ForegroundColor Cyan
                Write-Host "📖 Документация: http://localhost:8000/docs" -ForegroundColor Cyan
                Write-Host "💚 Health Check: http://localhost:8000/health" -ForegroundColor Cyan
                Write-Host ""
                Write-Host "Ресурсы:" -ForegroundColor Yellow
                Write-Host "- CPU: 1 core max" -ForegroundColor Gray
                Write-Host "- RAM: 1GB max" -ForegroundColor Gray
                Write-Host "- Модель: YOLOv9n (12MB)" -ForegroundColor Gray
                Write-Host "- Время загрузки модели: ~60 секунд" -ForegroundColor Yellow
            }
            else {
                Write-Host "❌ Ошибка запуска. Проверьте логи:" -ForegroundColor Red
                docker-compose logs --tail=20
            }
        }
        catch {
            Write-Host "❌ Ошибка при запуске: $_" -ForegroundColor Red
            Write-Host "Попробуйте: docker-compose down && docker system prune -f" -ForegroundColor Yellow
        }
    }
    
    "stop" {
        Write-Host "Остановка системы YOLOv9n..." -ForegroundColor Yellow
        docker-compose down
        Write-Host "✅ Система остановлена" -ForegroundColor Green
    }
    
    "logs" {
        Write-Host "Просмотр логов системы..." -ForegroundColor Cyan
        docker-compose logs -f
    }
    
    "status" {
        Write-Host "Статус системы:" -ForegroundColor Cyan
        docker-compose ps
    }
    
    "clean" {
        Write-Host "Полная очистка системы..." -ForegroundColor Yellow
        docker-compose down --volumes --remove-orphans
        docker system prune -af --volumes
        
        # Удаляем загруженную модель
        if (Test-Path "yolov9n.pt") {
            Remove-Item "yolov9n.pt" -Force
            Write-Host "Удалена локальная модель yolov9n.pt" -ForegroundColor Yellow
        }
        
        Write-Host "✅ Полная очистка завершена" -ForegroundColor Green
    }
    
    "test" {
        Write-Host "Тестирование API..." -ForegroundColor Cyan
        
        # Проверяем доступность API
        try {
            $response = Invoke-RestMethod -Uri "http://localhost:8000/health" -TimeoutSec 10
            Write-Host "✅ API доступен" -ForegroundColor Green
            Write-Host "Model Status: $($response.model_status)" -ForegroundColor $(if ($response.model_status -eq "ready") { "Green" } else { "Yellow" })
        }
        catch {
            Write-Host "❌ API недоступен. Запустите систему командой 'start'" -ForegroundColor Red
        }
    }
    
    default {
        Write-Host "Использование:" -ForegroundColor Yellow
        Write-Host "  .\start.ps1 start      - Запуск системы" -ForegroundColor Gray
        Write-Host "  .\start.ps1 stop       - Остановка системы" -ForegroundColor Gray
        Write-Host "  .\start.ps1 logs       - Просмотр логов" -ForegroundColor Gray
        Write-Host "  .\start.ps1 status     - Статус системы" -ForegroundColor Gray
        Write-Host "  .\start.ps1 test       - Тестирование API" -ForegroundColor Gray
        Write-Host "  .\start.ps1 clean      - Полная очистка" -ForegroundColor Gray
        Write-Host "  .\start.ps1 start -Clean - Запуск с очисткой" -ForegroundColor Gray
        Write-Host ""
        Write-Host "Пример использования API:" -ForegroundColor Yellow
        Write-Host "curl -X POST 'http://localhost:8000/detect' -F 'file=@image.jpg'" -ForegroundColor Cyan
    }
}