@echo off
REM Vehicle Damage Detection - Start Script
REM Запускает систему через Docker Compose

echo.
echo 🚗 Vehicle Damage Detection System
echo ===================================
echo.

REM Проверка Docker
docker --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker не установлен
    echo Установите Docker Desktop: https://www.docker.com/products/docker-desktop
    exit /b 1
)

docker-compose --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker Compose не установлен
    exit /b 1
)

echo ✅ Docker и Docker Compose установлены
echo.

REM Запуск системы
echo 🚀 Запускаю систему...
docker-compose up --build -d

echo.
echo ⏳ Ожидание запуска сервисов...
timeout /t 10 /nobreak >nul

REM Проверка статуса
echo.
echo 📊 Статус сервисов:
docker-compose ps

echo.
echo ✅ Система запущена!
echo.
echo 🌐 Доступные сервисы:
echo   - Frontend: http://localhost:3000
echo   - Backend API: http://localhost:8000
echo   - API Docs: http://localhost:8000/docs
echo.
echo 📝 Просмотр логов: docker-compose logs -f
echo 🛑 Остановка: docker-compose down
echo.
pause
