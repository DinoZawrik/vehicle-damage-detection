@echo off
REM Vehicle Damage Detection - Development Script
REM Запускает систему для локальной разработки

echo.
echo 🚗 Vehicle Damage Detection - Development Mode
echo ===============================================
echo.

REM Проверка Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python не установлен
    exit /b 1
)

REM Проверка Node.js
node --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Node.js не установлен
    exit /b 1
)

echo ✅ Python и Node.js установлены
echo.

REM Активация виртуального окружения
if not exist "venv\" (
    echo 📦 Создаю виртуальное окружение...
    python -m venv venv
)

echo 🔧 Активирую виртуальное окружение...
call venv\Scripts\activate.bat

REM Установка зависимостей
if not exist "venv\.deps_installed" (
    echo 📦 Устанавливаю Python зависимости...
    pip install --upgrade pip
    pip install -r requirements.txt
    type nul > venv\.deps_installed
)

REM Frontend зависимости
if not exist "web\node_modules\" (
    echo 📦 Устанавливаю Frontend зависимости...
    cd web
    call npm install
    cd ..
)

echo.
echo 🚀 Запускаю Backend на порту 8000...
start "Backend" cmd /k "venv\Scripts\activate.bat && uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000"

timeout /t 3 /nobreak >nul

echo 🚀 Запускаю Frontend на порту 3000...
start "Frontend" cmd /k "cd web && npm run dev"

echo.
echo ✅ Development сервера запущены!
echo.
echo 🌐 Доступные сервисы:
echo   - Frontend: http://localhost:3000
echo   - Backend API: http://localhost:8000
echo   - API Docs: http://localhost:8000/docs
echo.
echo 📝 Сервера запущены в отдельных окнах
echo 🛑 Закройте окна для остановки серверов
echo.
pause
