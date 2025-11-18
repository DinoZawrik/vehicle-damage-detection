#!/bin/bash

# Vehicle Damage Detection - Development Script
# Запускает систему для локальной разработки

set -e

echo "🚗 Vehicle Damage Detection - Development Mode"
echo "==============================================="
echo ""

# Проверка Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 не установлен"
    exit 1
fi

# Проверка Node.js
if ! command -v node &> /dev/null; then
    echo "❌ Node.js не установлен"
    exit 1
fi

echo "✅ Python и Node.js установлены"
echo ""

# Активация виртуального окружения
if [ ! -d "venv" ]; then
    echo "📦 Создаю виртуальное окружение..."
    python3 -m venv venv
fi

echo "🔧 Активирую виртуальное окружение..."
source venv/bin/activate

# Установка зависимостей
if [ ! -f "venv/.deps_installed" ]; then
    echo "📦 Устанавливаю Python зависимости..."
    pip install --upgrade pip
    pip install -r requirements.txt
    touch venv/.deps_installed
fi

# Frontend зависимости
if [ ! -d "web/node_modules" ]; then
    echo "📦 Устанавливаю Frontend зависимости..."
    cd web
    npm install
    cd ..
fi

echo ""
echo "🚀 Запускаю Backend на порту 8000..."
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

echo "🚀 Запускаю Frontend на порту 3000..."
cd web
npm run dev &
FRONTEND_PID=$!
cd ..

echo ""
echo "✅ Development сервера запущены!"
echo ""
echo "🌐 Доступные сервисы:"
echo "  - Frontend: http://localhost:3000"
echo "  - Backend API: http://localhost:8000"
echo "  - API Docs: http://localhost:8000/docs"
echo ""
echo "🛑 Для остановки нажмите Ctrl+C"

# Обработка остановки
trap "kill $BACKEND_PID $FRONTEND_PID; exit" INT TERM

wait
