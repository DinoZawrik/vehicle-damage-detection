# 🚀 Deployment Guide

Руководство по деплою Vehicle Damage Detection System в production.

---

## Варианты деплоя

1. **Docker Compose** - простой деплой на VPS
2. **Kubernetes** - для больших нагрузок
3. **Cloud Services** - AWS, GCP, Azure

---

## 1. Docker Compose на VPS (рекомендуется для начала)

### Требования к серверу:

- **OS:** Ubuntu 20.04+ или Debian 11+
- **RAM:** Минимум 2GB, рекомендуется 4GB
- **CPU:** 2+ cores
- **Disk:** 10GB свободного места
- **Network:** Публичный IP адрес

### Шаг 1: Подготовка сервера

```bash
# Обновить систему
sudo apt-get update && sudo apt-get upgrade -y

# Установить Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Установить Docker Compose
sudo apt-get install docker-compose -y

# Добавить пользователя в группу docker
sudo usermod -aG docker $USER
```

### Шаг 2: Клонировать репозиторий

```bash
# Клонировать проект
git clone https://github.com/yourusername/vehicle-damage-detection.git
cd vehicle-damage-detection
```

### Шаг 3: Настроить окружение

Создайте `.env` файл:

```env
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=False

# YOLO Settings
YOLO_MODEL=yolov8n.pt
YOLO_CONFIDENCE=0.35
YOLO_DEVICE=cpu

# Security
API_KEYS=your-secret-api-key-1,your-secret-api-key-2

# Database
DATABASE_URL=sqlite:///./data/detection.db

# Logging
LOG_LEVEL=INFO
```

### Шаг 4: Production docker-compose.yml

```yaml
version: '3.8'

services:
  backend:
    build: .
    restart: always
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
      - ./data:/app/data
      - ./logs:/app/logs
    environment:
      - DEBUG=False
      - LOG_LEVEL=INFO
    deploy:
      resources:
        limits:
          memory: 2G
        reservations:
          memory: 1G
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  frontend:
    build: ./web
    restart: always
    ports:
      - "80:80"
    depends_on:
      - backend
    environment:
      - REACT_APP_API_URL=http://your-domain.com:8000

  nginx:
    image: nginx:alpine
    restart: always
    ports:
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - backend
      - frontend
```

### Шаг 5: Запустить систему

```bash
# Запустить в background
docker-compose up -d --build

# Проверить логи
docker-compose logs -f

# Проверить статус
docker-compose ps
```

---

## 2. Настройка Nginx + SSL

### Установка Certbot для Let's Encrypt

```bash
# Установить Certbot
sudo apt-get install certbot python3-certbot-nginx -y

# Получить SSL сертификат
sudo certbot --nginx -d your-domain.com
```

### nginx.conf

```nginx
upstream backend {
    server backend:8000;
}

upstream frontend {
    server frontend:3000;
}

# HTTP -> HTTPS redirect
server {
    listen 80;
    server_name your-domain.com;
    return 301 https://$server_name$request_uri;
}

# HTTPS
server {
    listen 443 ssl http2;
    server_name your-domain.com;

    ssl_certificate /etc/nginx/ssl/fullchain.pem;
    ssl_certificate_key /etc/nginx/ssl/privkey.pem;

    # Frontend
    location / {
        proxy_pass http://frontend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    # Backend API
    location /api/ {
        proxy_pass http://backend/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        
        # Увеличить timeout для больших изображений
        proxy_read_timeout 300;
        proxy_connect_timeout 300;
        proxy_send_timeout 300;
        
        # Увеличить max body size
        client_max_body_size 10M;
    }
}
```

---

## 3. База данных (PostgreSQL для production)

### Шаг 1: Добавить PostgreSQL в docker-compose.yml

```yaml
services:
  postgres:
    image: postgres:15-alpine
    restart: always
    environment:
      POSTGRES_DB: vehicle_damage
      POSTGRES_USER: dbuser
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  backend:
    # ... existing config
    environment:
      - DATABASE_URL=postgresql://dbuser:${DB_PASSWORD}@postgres:5432/vehicle_damage
    depends_on:
      - postgres

volumes:
  postgres_data:
```

### Шаг 2: Миграции (Alembic)

```bash
# Установить Alembic
pip install alembic psycopg2-binary

# Инициализировать
alembic init alembic

# Создать миграцию
alembic revision --autogenerate -m "Initial"

# Применить миграции
alembic upgrade head
```

---

## 4. Мониторинг и логирование

### Prometheus + Grafana

Добавьте в docker-compose.yml:

```yaml
services:
  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    ports:
      - "9090:9090"

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3001:3000"
    volumes:
      - grafana_data:/var/lib/grafana
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}

volumes:
  prometheus_data:
  grafana_data:
```

### prometheus.yml

```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'backend'
    static_configs:
      - targets: ['backend:8000']
```

---

## 5. Резервное копирование

### Автоматический backup скрипт

Создайте `backup.sh`:

```bash
#!/bin/bash

BACKUP_DIR="/backups"
DATE=$(date +%Y%m%d_%H%M%S)

# Backup database
docker exec postgres pg_dump -U dbuser vehicle_damage > "$BACKUP_DIR/db_$DATE.sql"

# Backup uploads
tar -czf "$BACKUP_DIR/uploads_$DATE.tar.gz" data/uploads/

# Удалить старые backups (>7 дней)
find $BACKUP_DIR -type f -mtime +7 -delete

echo "Backup completed: $DATE"
```

### Добавить в crontab

```bash
# Редактировать crontab
crontab -e

# Добавить строку (backup каждый день в 2 AM)
0 2 * * * /path/to/backup.sh
```

---

## 6. Масштабирование

### Горизонтальное масштабирование с Docker Swarm

```bash
# Инициализировать Swarm
docker swarm init

# Деплой stack
docker stack deploy -c docker-compose.yml vehicle-damage

# Масштабировать backend
docker service scale vehicle-damage_backend=3
```

### Load Balancing

```nginx
upstream backend {
    least_conn;
    server backend1:8000;
    server backend2:8000;
    server backend3:8000;
}
```

---

## 7. Безопасность

### Чеклист безопасности:

- [ ] SSL/TLS сертификат установлен
- [ ] API key аутентификация включена
- [ ] Firewall настроен (только 80, 443 порты открыты)
- [ ] Регулярные обновления системы
- [ ] Резервные копии настроены
- [ ] Rate limiting включен
- [ ] CORS настроен правильно
- [ ] Секреты в environment variables, не в коде
- [ ] Логирование включено
- [ ] Мониторинг настроен

### Настройка firewall (UFW)

```bash
# Разрешить SSH
sudo ufw allow 22/tcp

# Разрешить HTTP/HTTPS
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp

# Включить firewall
sudo ufw enable
```

### Rate Limiting (Nginx)

```nginx
# В nginx.conf
limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/m;

location /api/ {
    limit_req zone=api_limit burst=20 nodelay;
    # ... остальная конфигурация
}
```

---

## 8. CI/CD Pipeline (GitHub Actions)

Создайте `.github/workflows/deploy.yml`:

```yaml
name: Deploy to Production

on:
  push:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Deploy to server
        uses: appleboy/ssh-action@master
        with:
          host: ${{ secrets.SERVER_HOST }}
          username: ${{ secrets.SERVER_USER }}
          key: ${{ secrets.SSH_KEY }}
          script: |
            cd /path/to/vehicle-damage-detection
            git pull origin main
            docker-compose down
            docker-compose up -d --build
            docker-compose logs --tail=50
```

---

## 9. Мониторинг производительности

### Полезные команды

```bash
# Использование ресурсов контейнерами
docker stats

# Логи с tail
docker-compose logs -f --tail=100

# Перезапуск без downtime
docker-compose up -d --no-deps --build backend

# Проверка health
curl http://localhost:8000/health
```

---

## 10. Troubleshooting

### Высокое использование памяти

```bash
# Проверить память
free -h

# Перезапустить с лимитами
docker-compose down
docker-compose up -d
```

### Медленные запросы

1. Проверьте логи: `docker-compose logs backend`
2. Мониторьте метрики в Grafana
3. Оптимизируйте YOLO параметры
4. Добавьте кэширование

### База данных переполнена

```bash
# Очистить старые записи
docker exec backend python scripts/cleanup_old_data.py

# Или вручную в psql
docker exec -it postgres psql -U dbuser -d vehicle_damage
DELETE FROM detections WHERE created_at < NOW() - INTERVAL '30 days';
```

---

## 11. Cloud Deployment

### AWS EC2

1. Запустите EC2 instance (t2.medium или больше)
2. Настройте Security Group (порты 80, 443, 22)
3. Установите Docker
4. Следуйте инструкциям VPS deployment выше

### Google Cloud Platform

```bash
# Создать VM instance
gcloud compute instances create vehicle-damage-vm \
  --image-family=ubuntu-2004-lts \
  --image-project=ubuntu-os-cloud \
  --machine-type=e2-standard-2

# SSH в instance
gcloud compute ssh vehicle-damage-vm

# Установить Docker и deploy
```

### Azure Container Instances

```bash
# Создать container group
az container create \
  --resource-group myResourceGroup \
  --name vehicle-damage \
  --image yourregistry.azurecr.io/vehicle-damage:latest \
  --dns-name-label vehicle-damage \
  --ports 80 443
```

---

## Чеклист перед production

- [ ] Все тесты проходят (`pytest tests/`)
- [ ] SSL сертификат установлен
- [ ] Environment variables настроены
- [ ] Database backup настроен
- [ ] Мониторинг работает
- [ ] Логирование настроено
- [ ] Rate limiting включен
- [ ] Firewall настроен
- [ ] Документация обновлена
- [ ] Load testing выполнен

---

**Дополнительные ресурсы:**

- [Docker Documentation](https://docs.docker.com/)
- [Nginx Configuration](https://nginx.org/en/docs/)
- [Let's Encrypt](https://letsencrypt.org/)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)

---

**Нужна помощь с deployment?** Создайте [issue](https://github.com/yourusername/vehicle-damage-detection/issues)
