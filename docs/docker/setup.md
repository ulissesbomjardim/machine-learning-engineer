# 🐳 Setup Docker

Este guia apresenta como configurar e usar Docker para executar o projeto Machine Learning Engineer Challenge.

## 📋 Visão Geral

O projeto oferece duas opções de containerização:
- 🚀 **Docker simples** - Container único da API
- 🔄 **Docker Compose** - Orquestração completa (API + MongoDB)

## 🛠️ Pré-requisitos

### 📦 Instalação do Docker

#### Windows

**Opção 1: Docker Desktop (Recomendado)**
```bash
# Baixar: https://desktop.docker.com/win/main/amd64/Docker%20Desktop%20Installer.exe
# Instalar e reiniciar o sistema
```

**Opção 2: Chocolatey**
```powershell
choco install docker-desktop
```

**Opção 3: Winget**
```powershell
winget install Docker.DockerDesktop
```

#### Linux (Ubuntu/Debian)

```bash
# Instalar Docker
sudo apt update
sudo apt install -y docker.io docker-compose

# Iniciar serviço
sudo systemctl start docker
sudo systemctl enable docker

# Adicionar usuário ao grupo docker
sudo usermod -aG docker $USER
newgrp docker

# Verificar instalação
docker --version
docker-compose --version
```

#### macOS

**Homebrew:**
```bash
brew install --cask docker
```

**Manual:**
```bash
# Baixar: https://desktop.docker.com/mac/main/amd64/Docker.dmg
# Instalar via interface gráfica
```

### ✅ Verificar Instalação

```bash
# Verificar Docker
docker --version
# Esperado: Docker version 20.x.x

# Verificar Docker Compose
docker-compose --version
# Esperado: docker-compose version 1.x.x

# Teste básico
docker run hello-world
```

## 📁 Estrutura Docker

### 🗂️ Arquivos de Configuração

```
machine-learning-engineer/
├── 🐳 Dockerfile                    # Imagem da API
├── 🔄 docker-compose.yml           # Orquestração completa
├── 🚀 docker-compose.simple.yml    # Apenas API
├── 📋 .dockerignore                # Arquivos ignorados
└── 🔧 docker/                      # Configs específicas
    ├── api.Dockerfile              # Dockerfile otimizado
    ├── requirements.txt            # Dependências Docker
    └── mongo-init/
        └── init-db.js              # Setup MongoDB
```

### 📋 Dockerfile Principal

```dockerfile
# Dockerfile - Multi-stage build
FROM python:3.12-slim as base

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Configurar diretório de trabalho
WORKDIR /app

# Copiar e instalar dependências Python
COPY pyproject.toml poetry.lock ./
RUN pip install poetry && \
    poetry config virtualenvs.create false && \
    poetry install --without dev

# Copiar código fonte
COPY src/ ./src/
COPY model/ ./model/

# Expor porta
EXPOSE 8000

# Comando de inicialização
CMD ["uvicorn", "src.routers.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 🚀 Executando com Docker

### 1. 🏗️ Build da Imagem

```bash
# Build da imagem principal
docker build -t ml-engineer-api .

# Ou com tag específica
docker build -t ml-engineer-api:v1.0.0 .

# Build com cache limpo (se necessário)
docker build --no-cache -t ml-engineer-api .
```

### 2. ⚡ Executar Container

**Execução básica:**
```bash
# Executar em foreground
docker run -p 8000:8000 ml-engineer-api

# Executar em background
docker run -d -p 8000:8000 --name ml-api ml-engineer-api

# Com montagem de volume para modelos
docker run -d \
  -p 8000:8000 \
  -v $(pwd)/model:/app/model \
  --name ml-api \
  ml-engineer-api
```

**Com variáveis de ambiente:**
```bash
docker run -d \
  -p 8000:8000 \
  -e ENVIRONMENT=production \
  -e LOG_LEVEL=INFO \
  -v $(pwd)/model:/app/model \
  --name ml-api \
  ml-engineer-api
```

### 3. 🔍 Verificar Execução

```bash
# Verificar containers rodando
docker ps

# Verificar logs
docker logs ml-api

# Logs em tempo real
docker logs -f ml-api

# Entrar no container
docker exec -it ml-api bash
```

## 🔄 Docker Compose

### 📋 Configuração Completa

```yaml
# docker-compose.yml
version: '3.8'

services:
  api:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    volumes:
      - ./model:/app/model:ro
    environment:
      - ENVIRONMENT=development
      - DATABASE_URL=mongodb://mongodb:27017/flight_predictions
    depends_on:
      - mongodb
    networks:
      - ml-network

  mongodb:
    image: mongo:7.0
    ports:
      - "27017:27017"
    volumes:
      - mongodb_data:/data/db
      - ./docker/mongo-init:/docker-entrypoint-initdb.d:ro
    environment:
      - MONGO_INITDB_ROOT_USERNAME=admin
      - MONGO_INITDB_ROOT_PASSWORD=password
      - MONGO_INITDB_DATABASE=flight_predictions
    networks:
      - ml-network

volumes:
  mongodb_data:

networks:
  ml-network:
    driver: bridge
```

### 🚀 Comandos Docker Compose

**Execução completa:**
```bash
# Build e executar todos os serviços
docker-compose up --build

# Executar em background
docker-compose up -d

# Apenas build (sem executar)
docker-compose build

# Executar serviço específico
docker-compose up api

# Parar todos os serviços
docker-compose down

# Parar e remover volumes
docker-compose down --volumes
```

**Gerenciamento:**
```bash
# Ver status dos serviços
docker-compose ps

# Ver logs de todos os serviços
docker-compose logs

# Logs de serviço específico
docker-compose logs api

# Logs em tempo real
docker-compose logs -f api

# Executar comando no container
docker-compose exec api bash
docker-compose exec mongodb mongo
```

### 🎯 Compose Simplificado

Para usar apenas a API (sem MongoDB):

```bash
# Usar compose simplificado
docker-compose -f docker-compose.simple.yml up --build
```

**docker-compose.simple.yml:**
```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./model:/app/model:ro
    environment:
      - ENVIRONMENT=development
      - USE_MOCK_DB=true
```

## ⚙️ Configurações Avançadas

### 🔧 Variáveis de Ambiente

```bash
# .env (criar na raiz do projeto)
ENVIRONMENT=production
LOG_LEVEL=INFO
DATABASE_URL=mongodb://mongodb:27017/flight_predictions
MODEL_PATH=/app/model/modelo_arvore_decisao.pkl
API_HOST=0.0.0.0
API_PORT=8000
WORKERS=4
```

### 🏗️ Build Multi-stage

```dockerfile
# Dockerfile.optimized
FROM python:3.12-slim as builder

# Instalar dependências de build
RUN apt-get update && apt-get install -y gcc

# Instalar Poetry e dependências
COPY pyproject.toml poetry.lock ./
RUN pip install poetry && \
    poetry export -f requirements.txt --output requirements.txt --without dev

# Stage de produção
FROM python:3.12-slim as production

# Copiar apenas requirements
COPY --from=builder requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código
COPY src/ ./src/
COPY model/ ./model/

# Usuário não-root para segurança
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

CMD ["uvicorn", "src.routers.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 🛡️ Health Checks

```dockerfile
# Dockerfile com health check
FROM python:3.12-slim

# ... configurações anteriores ...

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.routers.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 📊 Monitoramento

### 🔍 Logs Estruturados

```bash
# Ver logs com timestamp
docker-compose logs -t api

# Filtrar logs por nível
docker-compose logs api | grep ERROR

# Salvar logs em arquivo
docker-compose logs api > api.log
```

### 📈 Métricas de Container

```bash
# Estatísticas em tempo real
docker stats

# Uso de recursos do container específico
docker stats ml-api

# Informações detalhadas
docker inspect ml-api
```

## 🚨 Troubleshooting

### ❌ Problemas Comuns

**Container não inicia:**
```bash
# Verificar logs
docker logs ml-api

# Verificar porta ocupada
netstat -an | grep :8000  # Linux/macOS
netstat -an | findstr :8000  # Windows

# Usar porta diferente
docker run -p 8001:8000 ml-engineer-api
```

**Build falha:**
```bash
# Limpar cache do Docker
docker system prune

# Build sem cache
docker build --no-cache -t ml-engineer-api .

# Verificar espaço em disco
docker system df
```

**MongoDB não conecta:**
```bash
# Verificar rede
docker network ls
docker network inspect ml_ml-network

# Testar conexão
docker-compose exec api ping mongodb

# Verificar logs do MongoDB
docker-compose logs mongodb
```

### 🔧 Comandos de Diagnóstico

```bash
# Informações do sistema Docker
docker info

# Versão do Docker
docker version

# Processos em execução
docker ps -a

# Imagens disponíveis
docker images

# Volumes
docker volume ls

# Redes
docker network ls

# Limpar recursos não utilizados
docker system prune -a
```

## 🎯 Otimizações de Performance

### 🚀 Melhorias de Build

```dockerfile
# .dockerignore
.git
.pytest_cache
__pycache__
*.pyc
*.pyo
*.pyd
.env
.venv
node_modules
.DS_Store
*.log
htmlcov/
.coverage
tests/
docs/
README.md
```

### ⚡ Otimizações de Runtime

```bash
# Executar com recursos limitados
docker run \
  --memory=1g \
  --cpus=1.0 \
  -p 8000:8000 \
  ml-engineer-api

# Com restart automático
docker run \
  --restart=unless-stopped \
  -d \
  -p 8000:8000 \
  ml-engineer-api
```

## 📚 Próximos Passos

- 🔧 [Docker Compose Avançado](compose.md)
- 🚀 [Deploy em Produção](deployment.md)
- 🧪 [Testes com Docker](../tests/integration.md)
- ⚙️ [Configuração de Ambiente](../dev/environment.md)

## 📞 Suporte

- 🐳 [Docker Documentation](https://docs.docker.com/)
- 🐛 [Issues](https://github.com/ulissesbomjardim/machine_learning_engineer/issues)
- 📧 [Email](mailto:ulisses.bomjardim@gmail.com)