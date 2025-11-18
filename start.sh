#!/bin/bash

# Vehicle Damage Detection - Start Script
# Запускает систему через Docker Compose

set -e

echo "🚗 Vehicle Damage Detection System"
echo "==================================="
echo ""

# Проверка Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker не установлен"
    echo "Установите Docker: https://docs.docker.com/get-docker/"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose не установлен"
    echo "Установите Docker Compose: https://docs.docker.com/compose/install/"
    exit 1
fi

echo "✅ Docker и Docker Compose установлены"
echo ""

# Запуск системы
echo "🚀 Запускаю систему..."
docker-compose up --build -d

echo ""
echo "⏳ Ожидание запуска сервисов..."
sleep 10

# Проверка статуса
echo ""
echo "📊 Статус сервисов:"
docker-compose ps

echo ""
echo "✅ Система запущена!"
echo ""
echo "🌐 Доступные сервисы:"
echo "  - Frontend: http://localhost:3000"
echo "  - Backend API: http://localhost:8000"
echo "  - API Docs: http://localhost:8000/docs"
echo ""
echo "📝 Просмотр логов: docker-compose logs -f"
echo "🛑 Остановка: docker-compose down"
