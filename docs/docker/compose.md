# 🐳 Docker Compose

Guia completo para orquestração de containers usando Docker Compose, incluindo configuração de múltiplos serviços, redes, volumes e ambientes de desenvolvimento/produção.

## 🎯 Visão Geral

Docker Compose permite definir e executar aplicações Docker com múltiplos containers de forma simples e reproduzível. Esta seção detalha como configurar todo o stack da aplicação.

## 📁 Estrutura dos Arquivos

```
docker/
├── docker-compose.yml          # Produção
├── docker-compose.dev.yml      # Desenvolvimento  
├── docker-compose.test.yml     # Testes
├── .env.example               # Variáveis de ambiente
├── nginx/
│   └── nginx.conf            # Configuração Nginx
└── scripts/
    ├── wait-for-it.sh        # Script para aguardar serviços
    └── init-db.sh           # Inicialização do banco
```

## 🚀 Docker Compose Principal

### 📋 docker-compose.yml (Produção)

```yaml
version: '3.8'

services:
  # API Principal
  api:
    build:
      context: .
      dockerfile: docker/api.Dockerfile
    container_name: flight_delay_api
    restart: unless-stopped
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=production
      - DATABASE_URL=mongodb://mongodb:27017/flight_predictions
      - REDIS_URL=redis://redis:6379/0
      - MODEL_PATH=/app/model/
      - LOG_LEVEL=INFO
    volumes:
      - ./model:/app/model:ro
      - ./logs:/app/logs
    depends_on:
      - mongodb
      - redis
    networks:
      - flight-network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  # Banco de Dados MongoDB
  mongodb:
    image: mongo:7.0
    container_name: flight_delay_mongodb
    restart: unless-stopped
    ports:
      - "27017:27017"
    environment:
      - MONGO_INITDB_ROOT_USERNAME=${MONGO_USERNAME:-admin}
      - MONGO_INITDB_ROOT_PASSWORD=${MONGO_PASSWORD:-password123}
      - MONGO_INITDB_DATABASE=flight_predictions
    volumes:
      - mongodb_data:/data/db
      - ./docker/scripts/init-db.sh:/docker-entrypoint-initdb.d/init-db.sh:ro
    networks:
      - flight-network
    healthcheck:
      test: echo 'db.runCommand("ping").ok' | mongosh localhost:27017/flight_predictions --quiet
      interval: 30s
      timeout: 10s
      retries: 3

  # Cache Redis
  redis:
    image: redis:7-alpine
    container_name: flight_delay_redis
    restart: unless-stopped
    ports:
      - "6379:6379"
    command: redis-server --appendonly yes --requirepass ${REDIS_PASSWORD:-redis123}
    volumes:
      - redis_data:/data
    networks:
      - flight-network
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Reverse Proxy Nginx
  nginx:
    image: nginx:alpine
    container_name: flight_delay_nginx
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./docker/nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./docker/nginx/ssl:/etc/nginx/ssl:ro
      - nginx_logs:/var/log/nginx
    depends_on:
      - api
    networks:
      - flight-network
    healthcheck:
      test: ["CMD", "wget", "--quiet", "--tries=1", "--spider", "http://localhost/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Worker para Processamento Assíncrono
  worker:
    build:
      context: .
      dockerfile: docker/api.Dockerfile
    container_name: flight_delay_worker
    restart: unless-stopped
    command: celery -A src.worker worker --loglevel=info --concurrency=4
    environment:
      - ENVIRONMENT=production
      - DATABASE_URL=mongodb://mongodb:27017/flight_predictions
      - REDIS_URL=redis://redis:6379/0
      - CELERY_BROKER_URL=redis://redis:6379/1
      - CELERY_RESULT_BACKEND=redis://redis:6379/1
    volumes:
      - ./model:/app/model:ro
      - ./logs:/app/logs
    depends_on:
      - mongodb
      - redis
    networks:
      - flight-network

  # Monitor Celery
  flower:
    build:
      context: .
      dockerfile: docker/api.Dockerfile
    container_name: flight_delay_flower
    restart: unless-stopped
    command: celery -A src.worker flower --port=5555
    ports:
      - "5555:5555"
    environment:
      - CELERY_BROKER_URL=redis://redis:6379/1
      - CELERY_RESULT_BACKEND=redis://redis:6379/1
    depends_on:
      - redis
      - worker
    networks:
      - flight-network

  # Monitoramento Prometheus
  prometheus:
    image: prom/prometheus:latest
    container_name: flight_delay_prometheus
    restart: unless-stopped
    ports:
      - "9090:9090"
    volumes:
      - ./docker/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
    networks:
      - flight-network

  # Dashboard Grafana
  grafana:
    image: grafana/grafana:latest
    container_name: flight_delay_grafana
    restart: unless-stopped
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD:-admin123}
    volumes:
      - grafana_data:/var/lib/grafana
      - ./docker/grafana/dashboards:/etc/grafana/provisioning/dashboards:ro
      - ./docker/grafana/datasources:/etc/grafana/provisioning/datasources:ro
    depends_on:
      - prometheus
    networks:
      - flight-network

# Volumes persistentes
volumes:
  mongodb_data:
    driver: local
  redis_data:
    driver: local
  prometheus_data:
    driver: local
  grafana_data:
    driver: local
  nginx_logs:
    driver: local

# Redes
networks:
  flight-network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
```

## 🔧 Ambiente de Desenvolvimento

### 📋 docker-compose.dev.yml

```yaml
version: '3.8'

services:
  # API com hot reload
  api-dev:
    build:
      context: .
      dockerfile: docker/Dockerfile.dev
    container_name: flight_delay_api_dev
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=development
      - DATABASE_URL=mongodb://mongodb-dev:27017/flight_predictions_dev
      - REDIS_URL=redis://redis-dev:6379/0
      - DEBUG=True
      - LOG_LEVEL=DEBUG
    volumes:
      - .:/app:delegated
      - /app/venv
      - dev_cache:/root/.cache
    depends_on:
      - mongodb-dev
      - redis-dev
    networks:
      - flight-dev-network
    command: uvicorn src.routers.main:app --host 0.0.0.0 --port 8000 --reload --log-level debug

  # MongoDB Desenvolvimento
  mongodb-dev:
    image: mongo:7.0
    container_name: flight_delay_mongodb_dev
    ports:
      - "27018:27017"
    environment:
      - MONGO_INITDB_ROOT_USERNAME=devuser
      - MONGO_INITDB_ROOT_PASSWORD=devpass
      - MONGO_INITDB_DATABASE=flight_predictions_dev
    volumes:
      - mongodb_dev_data:/data/db
      - ./data/sample:/docker-entrypoint-initdb.d:ro
    networks:
      - flight-dev-network

  # Redis Desenvolvimento
  redis-dev:
    image: redis:7-alpine
    container_name: flight_delay_redis_dev
    ports:
      - "6380:6379"
    command: redis-server --appendonly yes
    volumes:
      - redis_dev_data:/data
    networks:
      - flight-dev-network

  # Jupyter Lab para análise
  jupyter:
    build:
      context: .
      dockerfile: docker/Dockerfile.jupyter
    container_name: flight_delay_jupyter
    ports:
      - "8888:8888"
    environment:
      - JUPYTER_ENABLE_LAB=yes
      - JUPYTER_TOKEN=
      - DATABASE_URL=mongodb://mongodb-dev:27017/flight_predictions_dev
    volumes:
      - .:/app:delegated
      - jupyter_data:/home/jovyan/.jupyter
    depends_on:
      - mongodb-dev
    networks:
      - flight-dev-network
    command: start-notebook.sh --NotebookApp.token='' --NotebookApp.password=''

  # Adminer para gerenciar MongoDB
  mongo-express:
    image: mongo-express:latest
    container_name: flight_delay_mongo_express
    ports:
      - "8081:8081"
    environment:
      - ME_CONFIG_MONGODB_SERVER=mongodb-dev
      - ME_CONFIG_MONGODB_ADMINUSERNAME=devuser
      - ME_CONFIG_MONGODB_ADMINPASSWORD=devpass
      - ME_CONFIG_BASICAUTH_USERNAME=admin
      - ME_CONFIG_BASICAUTH_PASSWORD=admin123
    depends_on:
      - mongodb-dev
    networks:
      - flight-dev-network

  # Mailhog para testes de email
  mailhog:
    image: mailhog/mailhog:latest
    container_name: flight_delay_mailhog
    ports:
      - "1025:1025"  # SMTP
      - "8025:8025"  # Web UI
    networks:
      - flight-dev-network

volumes:
  mongodb_dev_data:
  redis_dev_data:
  jupyter_data:
  dev_cache:

networks:
  flight-dev-network:
    driver: bridge
```

## 🧪 Ambiente de Testes

### 📋 docker-compose.test.yml

```yaml
version: '3.8'

services:
  # API para testes
  api-test:
    build:
      context: .
      dockerfile: docker/api.Dockerfile
      target: test
    container_name: flight_delay_api_test
    environment:
      - ENVIRONMENT=testing
      - DATABASE_URL=mongodb://mongodb-test:27017/flight_predictions_test
      - REDIS_URL=redis://redis-test:6379/0
      - TESTING=True
    volumes:
      - .:/app:delegated
      - test_coverage:/app/htmlcov
    depends_on:
      - mongodb-test
      - redis-test
    networks:
      - flight-test-network
    command: >
      bash -c "
        python -m pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing --junit-xml=test-results.xml &&
        python -m pytest tests/integration/ -v --integration
      "

  # MongoDB Testes (em memória)
  mongodb-test:
    image: mongo:7.0
    container_name: flight_delay_mongodb_test
    environment:
      - MONGO_INITDB_DATABASE=flight_predictions_test
    tmpfs:
      - /data/db
    networks:
      - flight-test-network

  # Redis Testes
  redis-test:
    image: redis:7-alpine
    container_name: flight_delay_redis_test
    command: redis-server --save ""
    tmpfs:
      - /data
    networks:
      - flight-test-network

  # Testes de performance
  locust:
    build:
      context: .
      dockerfile: docker/Dockerfile.locust
    container_name: flight_delay_locust
    ports:
      - "8089:8089"
    environment:
      - LOCUST_HOST=http://api-test:8000
    volumes:
      - ./tests/performance:/app/tests
    depends_on:
      - api-test
    networks:
      - flight-test-network
    command: locust -f /app/tests/locustfile.py --host=http://api-test:8000

volumes:
  test_coverage:

networks:
  flight-test-network:
    driver: bridge
```

## ⚙️ Configurações Auxiliares

### 🌐 Nginx Configuration

```nginx
# docker/nginx/nginx.conf
events {
    worker_connections 1024;
}

http {
    upstream api_backend {
        server api:8000;
    }

    # Rate Limiting
    limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;
    limit_req_zone $binary_remote_addr zone=predict_limit:10m rate=2r/s;

    # Logging
    log_format main '$remote_addr - $remote_user [$time_local] "$request" '
                   '$status $body_bytes_sent "$http_referer" '
                   '"$http_user_agent" "$http_x_forwarded_for" '
                   'rt=$request_time uct="$upstream_connect_time" '
                   'uht="$upstream_header_time" urt="$upstream_response_time"';

    access_log /var/log/nginx/access.log main;
    error_log /var/log/nginx/error.log warn;

    server {
        listen 80;
        server_name localhost;

        # Security headers
        add_header X-Frame-Options DENY;
        add_header X-Content-Type-Options nosniff;
        add_header X-XSS-Protection "1; mode=block";
        add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;

        # Gzip compression
        gzip on;
        gzip_vary on;
        gzip_min_length 1024;
        gzip_types text/plain text/css application/json application/javascript text/xml application/xml;

        # Health check
        location /health {
            proxy_pass http://api_backend/health;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        # API endpoints with rate limiting
        location /api/ {
            limit_req zone=api_limit burst=20 nodelay;
            
            proxy_pass http://api_backend/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # Timeouts
            proxy_connect_timeout 5s;
            proxy_send_timeout 60s;
            proxy_read_timeout 60s;
        }

        # Predict endpoint with stricter rate limiting
        location /predict {
            limit_req zone=predict_limit burst=5 nodelay;
            
            proxy_pass http://api_backend/predict;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # Larger body size for batch predictions
            client_max_body_size 10M;
        }

        # Static files (documentação)
        location /docs {
            proxy_pass http://api_backend/docs;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
    }
}
```

### 📊 Prometheus Configuration

```yaml
# docker/prometheus/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "rules/*.yml"

scrape_configs:
  # API Metrics
  - job_name: 'flight-delay-api'
    static_configs:
      - targets: ['api:8000']
    metrics_path: '/metrics'
    scrape_interval: 10s

  # System Metrics
  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']

  # MongoDB Metrics
  - job_name: 'mongodb-exporter'
    static_configs:
      - targets: ['mongodb-exporter:9216']

  # Redis Metrics
  - job_name: 'redis-exporter'
    static_configs:
      - targets: ['redis-exporter:9121']

  # Nginx Metrics
  - job_name: 'nginx-exporter'
    static_configs:
      - targets: ['nginx-exporter:9113']

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

## 📝 Variáveis de Ambiente

### 🔧 .env.example

```bash
# Ambiente
ENVIRONMENT=production

# Banco de Dados
MONGO_USERNAME=admin
MONGO_PASSWORD=secure_password_here
DATABASE_URL=mongodb://mongodb:27017/flight_predictions

# Cache
REDIS_PASSWORD=redis_secure_password
REDIS_URL=redis://redis:6379/0

# Celery
CELERY_BROKER_URL=redis://redis:6379/1
CELERY_RESULT_BACKEND=redis://redis:6379/1

# Monitoring
GRAFANA_PASSWORD=grafana_admin_password

# API Keys (se necessário)
WEATHER_API_KEY=your_weather_api_key_here
SENTRY_DSN=your_sentry_dsn_here

# SSL (se usando HTTPS)
SSL_CERT_PATH=/etc/nginx/ssl/cert.pem
SSL_KEY_PATH=/etc/nginx/ssl/key.pem

# Recursos
API_WORKERS=4
API_MAX_REQUESTS=1000
API_MAX_REQUESTS_JITTER=100
```

## 🚀 Scripts de Gerenciamento

### 📋 Makefile

```makefile
# Makefile para gerenciar Docker Compose

.PHONY: help build up down logs ps clean test dev prod

# Variáveis
COMPOSE_FILE = docker-compose.yml
COMPOSE_DEV_FILE = docker-compose.dev.yml
COMPOSE_TEST_FILE = docker-compose.test.yml

help: ## Mostra esta ajuda
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

# Comandos de produção
build: ## Build das imagens para produção
	docker-compose -f $(COMPOSE_FILE) build --no-cache

up: ## Inicia todos os serviços em produção
	docker-compose -f $(COMPOSE_FILE) up -d

down: ## Para todos os serviços
	docker-compose -f $(COMPOSE_FILE) down

restart: ## Reinicia todos os serviços
	docker-compose -f $(COMPOSE_FILE) restart

logs: ## Mostra logs de todos os serviços
	docker-compose -f $(COMPOSE_FILE) logs -f

ps: ## Mostra status dos containers
	docker-compose -f $(COMPOSE_FILE) ps

# Comandos de desenvolvimento
dev-build: ## Build para desenvolvimento
	docker-compose -f $(COMPOSE_DEV_FILE) build

dev-up: ## Inicia ambiente de desenvolvimento
	docker-compose -f $(COMPOSE_DEV_FILE) up -d

dev-down: ## Para ambiente de desenvolvimento
	docker-compose -f $(COMPOSE_DEV_FILE) down

dev-logs: ## Logs do ambiente de desenvolvimento
	docker-compose -f $(COMPOSE_DEV_FILE) logs -f

# Comandos de teste
test-build: ## Build para testes
	docker-compose -f $(COMPOSE_TEST_FILE) build

test-up: ## Executa testes
	docker-compose -f $(COMPOSE_TEST_FILE) up --abort-on-container-exit

test-down: ## Para ambiente de teste
	docker-compose -f $(COMPOSE_TEST_FILE) down -v

# Comandos de limpeza
clean: ## Remove containers, networks e volumes não utilizados
	docker system prune -f
	docker volume prune -f
	docker network prune -f

clean-all: ## Remove tudo (CUIDADO!)
	docker-compose -f $(COMPOSE_FILE) down -v --rmi all
	docker-compose -f $(COMPOSE_DEV_FILE) down -v --rmi all
	docker-compose -f $(COMPOSE_TEST_FILE) down -v --rmi all

# Comandos de utilidade
shell-api: ## Acessa shell do container da API
	docker-compose -f $(COMPOSE_FILE) exec api bash

shell-db: ## Acessa shell do MongoDB
	docker-compose -f $(COMPOSE_FILE) exec mongodb mongosh

backup-db: ## Backup do banco de dados
	docker-compose -f $(COMPOSE_FILE) exec mongodb mongodump --host localhost --port 27017 --out /data/backup

restore-db: ## Restaura backup do banco
	docker-compose -f $(COMPOSE_FILE) exec mongodb mongorestore --host localhost --port 27017 /data/backup

# Monitoramento
stats: ## Estatísticas dos containers
	docker stats $(shell docker-compose -f $(COMPOSE_FILE) ps -q)

health: ## Verifica saúde dos serviços
	@echo "=== API Health ==="
	@curl -s http://localhost/health | jq .
	@echo "\n=== MongoDB Status ==="
	@docker-compose -f $(COMPOSE_FILE) exec mongodb mongosh --eval "db.adminCommand('ping')"
	@echo "\n=== Redis Status ==="
	@docker-compose -f $(COMPOSE_FILE) exec redis redis-cli ping
```

### 🔧 Scripts de Inicialização

```bash
#!/bin/bash
# docker/scripts/init-db.sh

set -e

echo "Inicializando banco de dados..."

# Aguardar MongoDB estar pronto
until mongosh --host localhost --port 27017 --eval "print(\"MongoDB is ready\")"; do
  echo "Aguardando MongoDB..."
  sleep 2
done

# Criar usuário da aplicação
mongosh --host localhost --port 27017 <<EOF
use flight_predictions;

// Criar usuário da aplicação
db.createUser({
  user: "app_user",
  pwd: "app_password_123",
  roles: [
    { role: "readWrite", db: "flight_predictions" }
  ]
});

// Criar índices
db.predictions.createIndex({ "flight_id": 1 });
db.predictions.createIndex({ "created_at": 1 });
db.predictions.createIndex({ "departure_airport": 1, "arrival_airport": 1 });

// Inserir dados de exemplo (opcional)
db.airports.insertMany([
  {
    "icao_code": "SBGR",
    "name": "São Paulo/Guarulhos",
    "city": "São Paulo",
    "country": "Brazil",
    "latitude": -23.4356,
    "longitude": -46.4731,
    "altitude": 750
  },
  {
    "icao_code": "SBRJ", 
    "name": "Rio de Janeiro/Santos Dumont",
    "city": "Rio de Janeiro",
    "country": "Brazil",
    "latitude": -22.9110,
    "longitude": -43.1631,
    "altitude": 3
  }
]);

print("Inicialização do banco concluída!");
EOF

echo "Banco de dados inicializado com sucesso!"
```

## 📊 Comandos Úteis

### 🚀 Comandos de Produção

```bash
# Inicializar stack completo
docker-compose up -d

# Verificar status
docker-compose ps
docker-compose logs -f api

# Escalar serviços
docker-compose up -d --scale worker=3

# Atualizar apenas a API
docker-compose up -d --build api

# Backup completo
docker-compose exec mongodb mongodump --archive | gzip > backup_$(date +%Y%m%d).gz

# Monitorar recursos
docker stats $(docker-compose ps -q)
```

### 🔧 Comandos de Desenvolvimento

```bash
# Ambiente de desenvolvimento
docker-compose -f docker-compose.dev.yml up -d

# Logs em tempo real
docker-compose -f docker-compose.dev.yml logs -f api-dev

# Executar testes
docker-compose -f docker-compose.test.yml up --abort-on-container-exit

# Acessar Jupyter
open http://localhost:8888

# Gerenciar MongoDB
open http://localhost:8081
```

### 🧹 Comandos de Limpeza

```bash
# Limpeza básica
docker-compose down --volumes
docker system prune -f

# Limpeza completa (CUIDADO!)
docker-compose down --rmi all --volumes --remove-orphans
docker system prune -a -f --volumes
```

## 🔗 Próximos Passos

1. **[🐳 Setup Docker](setup.md)** - Configuração básica do Docker
2. **[🚀 Deployment](deployment.md)** - Deploy em produção
3. **[🧪 Testes](../tests/integration.md)** - Testes de integração

---

## 📞 Referências

- 🏗️ **[Arquitetura](../architecture/overview.md)** - Visão geral do sistema
- ⚡ **[API](../api/endpoints.md)** - Endpoints da aplicação
- 🔧 **[Troubleshooting](../dev/troubleshooting.md)** - Solução de problemas