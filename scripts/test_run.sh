#!/bin/bash

# Vehicle Damage Detection System - Test Run Script
# Быстрый запуск для тестирования системы

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

echo "🚀 Vehicle Damage Detection System - Test Run"
echo "=============================================="
echo ""

# 1. Check system requirements
print_status "1. Проверка системных требований..."

# Check Docker
if ! command -v docker &> /dev/null; then
    print_error "Docker не установлен. Установите Docker сначала."
    echo "Инструкции: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check Docker Compose
if ! command -v docker-compose &> /dev/null; then
    print_error "Docker Compose не установлен. Установите Docker Compose сначала."
    echo "Инструкции: https://docs.docker.com/compose/install/"
    exit 1
fi

# Check Docker daemon
if ! docker info >/dev/null 2>&1; then
    print_error "Docker daemon не запущен. Запустите Docker Desktop или Docker service."
    exit 1
fi

print_success "Docker доступен"

# 2. Setup environment
print_status "2. Настройка окружения..."

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    if [ -f .env.example ]; then
        cp .env.example .env
        print_success "Создан .env файл из .env.example"
    else
        print_error ".env.example файл не найден"
        exit 1
    fi
else
    print_warning ".env файл уже существует, используем существующий"
fi

# Create necessary directories
mkdir -p data/{models,uploads,processed,raw,test_images} logs
print_success "Созданы необходимые директории"

# 3. Start test services
print_status "3. Запуск тестовых сервисов..."

# Stop any running services first
docker-compose -f docker-compose.test.yml down 2>/dev/null || true

# Build and start services
print_status "Сборка и запуск Docker образов (это займет несколько минут)..."
docker-compose -f docker-compose.test.yml up -d --build

# 4. Wait for services to be ready
print_status "4. Ожидание готовности сервисов..."

max_wait=120
elapsed=0

while [ $elapsed -lt $max_wait ]; do
    if curl -s http://localhost:8000/health >/dev/null 2>&1; then
        print_success "API готов к работе!"
        break
    fi
    
    echo -n "."
    sleep 5
    elapsed=$((elapsed + 5))
done

if [ $elapsed -ge $max_wait ]; then
    print_error "API не ответил в течение $max_wait секунд"
    print_status "Проверьте логи: docker-compose -f docker-compose.test.yml logs api"
    exit 1
fi

# 5. Run health checks
print_status "5. Проверка здоровья системы..."

# Test API health
if curl -s http://localhost:8000/health | jq . >/dev/null 2>&1; then
    health_data=$(curl -s http://localhost:8000/health)
    print_success "API health check:"
    echo "$health_data" | jq '.'
else
    print_error "API health check failed"
fi

# 6. Test image upload (create a simple test image)
print_status "6. Создание тестового изображения..."

# Create a simple test image using ImageMagick if available, otherwise use Python
if command -v convert &> /dev/null; then
    # Create a simple image with ImageMagick
    convert -size 640x480 xc:red -fill blue -draw "rectangle 100,100 300,200" -pointsize 30 -gravity center -annotate +0+0 "TEST CAR IMAGE" data/test_images/test_car.jpg
    print_success "Создано тестовое изображение с ImageMagick"
else
    # Create a simple image with Python
    python3 -c "
from PIL import Image, ImageDraw
import os

# Create a 640x480 red image
img = Image.new('RGB', (640, 480), color='red')
draw = ImageDraw.Draw(img)

# Draw a blue rectangle to simulate damage
draw.rectangle([100, 100, 300, 200], fill='blue')

# Save the image
os.makedirs('data/test_images', exist_ok=True)
img.save('data/test_images/test_car.jpg')
print('Test image created successfully')
"
    print_success "Создано тестовое изображение с Python"
fi

# 7. Test image analysis
print_status "7. Тестирование анализа изображения..."

if [ -f "data/test_images/test_car.jpg" ]; then
    echo "Анализируем тестовое изображение..."
    response=$(curl -s -X POST "http://localhost:8000/api/analyze" \
        -F "file=@data/test_images/test_car.jpg")
    
    if [ $? -eq 0 ]; then
        print_success "Анализ завершен! Результат:"
        echo "$response" | jq '.'
    else
        print_warning "Анализ не удался или модель еще загружается"
        print_status "Это нормально для первого запуска - модели загружаются при первом запросе"
    fi
else
    print_warning "Тестовое изображение не найдено"
fi

# 8. Show system status
print_status "8. Статус системы..."

docker-compose -f docker-compose.test.yml ps

# 9. Show access information
print_status "9. Информация о доступе:"
echo ""
echo "🌐 Доступные интерфейсы:"
echo "  📱 API:           http://localhost:8000"
echo "  📚 API Docs:      http://localhost:8000/docs"
echo "  ❤️  Health Check:  http://localhost:8000/health"
echo ""
echo "🐳 Docker команды:"
echo "  Просмотр логов:   docker-compose -f docker-compose.test.yml logs -f"
echo "  Остановка:        docker-compose -f docker-compose.test.yml down"
echo "  Статус сервисов:  docker-compose -f docker-compose.test.yml ps"
echo ""
echo "🧪 Тестирование:"
echo "  Загрузите изображение: http://localhost:8000/docs#/default/post_api_analyze"
echo "  Или используйте веб-интерфейс через браузер"
echo ""

# 10. Final recommendations
print_status "10. Рекомендации:"
echo ""
echo "Для полного тестирования рекомендуется:"
echo "  1. Загрузить реальное фото автомобиля через API или веб-интерфейс"
echo "  2. Проверить различные форматы изображений (JPG, PNG)"
echo "  3. Протестировать с изображениями разных размеров"
echo "  4. Проверить работу с поврежденными и чистыми автомобилями"
echo ""
echo "Для запуска полной системы (все сервисы):"
echo "  docker-compose up -d"
echo ""
echo "Для устранения проблем:"
echo "  docker-compose -f docker-compose.test.yml logs api"
echo ""

print_success "🎉 Система готова к тестированию!"
echo ""
echo "Система запущена в тестовом режиме с минимальными требованиями."
echo "Для production использования запустите полную конфигурацию: docker-compose up -d"