# 📦 Установка и настройка

Подробное руководство по установке Vehicle Damage Detection System.

---

## Системные требования

### Минимальные:
- **OS:** Windows 10/11, Ubuntu 20.04+, macOS 12+
- **RAM:** 2GB свободной памяти
- **Disk:** 5GB свободного места
- **Python:** 3.8 или выше
- **Node.js:** 16.x или выше (для frontend)

### Рекомендуемые:
- **RAM:** 4GB+
- **GPU:** CUDA-compatible (опционально, для ускорения)

---

## Способ 1: Docker (рекомендуется)

### Установка Docker

**Windows:**
1. Скачайте [Docker Desktop](https://www.docker.com/products/docker-desktop)
2. Установите и запустите
3. Убедитесь что Docker работает: `docker --version`

**Linux:**
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install docker.io docker-compose
sudo systemctl start docker
sudo systemctl enable docker

# Добавить пользователя в группу docker
sudo usermod -aG docker $USER
```

**macOS:**
1. Установите Docker Desktop для Mac
2. Запустите приложение

### Запуск проекта

```bash
# 1. Клонировать репозиторий
git clone https://github.com/yourusername/vehicle-damage-detection.git
cd vehicle-damage-detection

# 2. Запустить с Docker Compose
docker-compose up --build

# 3. Проверить что всё работает
curl http://localhost:8000/health
```

**Готово!** Система доступна:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

---

## Способ 2: Локальная установка

### Python Backend

#### Windows:

```powershell
# 1. Клонировать репозиторий
git clone https://github.com/yourusername/vehicle-damage-detection.git
cd vehicle-damage-detection

# 2. Создать виртуальное окружение
python -m venv venv
.\venv\Scripts\Activate.ps1

# 3. Установить зависимости
pip install --upgrade pip
pip install -r requirements.txt

# 4. Скачать YOLO модель (опционально, автоматически загрузится)
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# 5. Запустить backend
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

#### Linux/Mac:

```bash
# 1. Клонировать репозиторий
git clone https://github.com/yourusername/vehicle-damage-detection.git
cd vehicle-damage-detection

# 2. Создать виртуальное окружение
python3 -m venv venv
source venv/bin/activate

# 3. Установить зависимости
pip install --upgrade pip
pip install -r requirements.txt

# 4. Скачать YOLO модель
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# 5. Запустить backend
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

### React Frontend

В **отдельном терминале**:

```bash
# 1. Перейти в директорию web
cd web

# 2. Установить зависимости
npm install

# 3. Запустить dev сервер
npm run dev
```

Frontend будет доступен на http://localhost:3000

---

## Конфигурация

### Переменные окружения

Создайте `.env` файл в корне проекта:

```env
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=False

# YOLO Model
YOLO_MODEL=yolov8n.pt
YOLO_CONFIDENCE=0.35
YOLO_IOU=0.5
YOLO_DEVICE=cpu  # или 'cuda' для GPU

# Database
DATABASE_URL=sqlite:///./data/detection.db

# File Upload
MAX_IMAGE_SIZE=10485760  # 10MB в байтах
ALLOWED_EXTENSIONS=jpg,jpeg,png

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/app.log
```

### Настройки YOLO

Отредактируйте `src/api/main.py` для изменения параметров детекции:

```python
# Изменить уверенность детекции
CONF_THRESHOLD = 0.35  # от 0.0 до 1.0

# Изменить IoU threshold
IOU_THRESHOLD = 0.5

# Использовать GPU
DEVICE = 'cuda'  # вместо 'cpu'
```

---

## Проверка установки

### 1. Backend Health Check

```bash
curl http://localhost:8000/health
```

Ожидаемый ответ:
```json
{
  "status": "ok",
  "model_loaded": true,
  "timestamp": "2025-11-18T12:00:00"
}
```

### 2. API Docs

Откройте в браузере: http://localhost:8000/docs

Вы должны увидеть интерактивную Swagger документацию.

### 3. Frontend

Откройте: http://localhost:3000

Должен загрузиться web интерфейс.

### 4. Тест детекции

```bash
# Используйте тестовое изображение
curl -X POST "http://localhost:8000/detect" \
  -F "file=@data/test_samples/car_01.jpg"
```

---

## Решение проблем

### Проблема: Python версия < 3.8

**Решение:** Установите Python 3.8+ с [python.org](https://www.python.org/downloads/)

### Проблема: pip не найден

**Windows:**
```powershell
python -m ensurepip --upgrade
```

**Linux:**
```bash
sudo apt-get install python3-pip
```

### Проблема: Не устанавливается ultralytics

**Решение:** Обновите pip и setuptools:
```bash
pip install --upgrade pip setuptools wheel
pip install ultralytics
```

### Проблема: CUDA не найдена (для GPU)

**Решение:** 
1. Установите [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
2. Установите PyTorch с CUDA:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Проблема: Docker не запускается

**Windows:** Убедитесь что WSL2 включен:
```powershell
wsl --install
```

**Linux:** Проверьте что Docker daemon работает:
```bash
sudo systemctl status docker
```

### Проблема: Port 8000 уже занят

**Решение:** Измените порт в docker-compose.yml:
```yaml
ports:
  - "8001:8000"  # используйте 8001 вместо 8000
```

### Проблема: Frontend не подключается к API

**Решение:** Проверьте `web/src/api.ts`:
```typescript
const API_BASE_URL = 'http://localhost:8000';
```

Измените на правильный URL если нужно.

---

## Обновление системы

```bash
# Остановить систему
docker-compose down

# Обновить код
git pull origin main

# Пересобрать и запустить
docker-compose up --build
```

---

## Деинсталляция

### Docker версия:
```bash
# Остановить и удалить контейнеры
docker-compose down -v

# Удалить образы (опционально)
docker image prune -a
```

### Локальная версия:
```bash
# Деактивировать venv
deactivate

# Удалить директорию проекта
rm -rf vehicle-damage-detection
```

---

## Следующие шаги

После успешной установки:

1. 📖 Изучите [API документацию](API.md)
2. 🧪 Запустите тесты: `pytest tests/`
3. 🚀 Прочитайте [руководство по деплою](DEPLOYMENT.md)
4. 👨‍💻 Ознакомьтесь с [руководством разработчика](DEVELOPMENT.md)

---

**Возникли проблемы?** Создайте [issue на GitHub](https://github.com/yourusername/vehicle-damage-detection/issues)
