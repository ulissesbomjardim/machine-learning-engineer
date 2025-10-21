# 💻 Ambiente de Desenvolvimento

Guia completo para configurar um ambiente de desenvolvimento produtivo para o projeto Machine Learning Engineer Challenge.

## 🎯 Visão Geral

Este guia apresenta as melhores práticas para configurar um ambiente de desenvolvimento completo, incluindo ferramentas, configurações e workflows recomendados.

## 🛠️ Stack de Desenvolvimento

### 📋 Ferramentas Essenciais

| **Categoria** | **Ferramenta** | **Versão** | **Propósito** |
|---------------|----------------|------------|---------------|
| 💻 **Editor** | VS Code | Latest | IDE principal |
| 🐍 **Python** | Python | 3.12.7 | Linguagem base |
| 📦 **Deps** | Poetry | 1.7+ | Gerenciamento de dependências |
| 🔧 **Git** | Git | 2.40+ | Controle de versão |
| 🐳 **Container** | Docker | 24.0+ | Containerização |
| 📓 **Notebook** | Jupyter | Latest | Análise de dados |

### 🎨 Extensões VS Code Recomendadas

```json
// .vscode/extensions.json
{
  "recommendations": [
    // Python essentials
    "ms-python.python",
    "ms-python.black-formatter",
    "ms-python.isort",
    "charliermarsh.ruff",
    "ms-python.pylint",
    
    // Jupyter
    "ms-toolsai.jupyter",
    "ms-toolsai.jupyter-keymap",
    
    // Git
    "eamodio.gitlens",
    "github.vscode-pull-request-github",
    
    // Docker
    "ms-azuretools.vscode-docker",
    
    // Markdown
    "yzhang.markdown-all-in-one",
    "shd101wyy.markdown-preview-enhanced",
    
    // Úteis
    "ms-vscode.errorlens",
    "wayou.vscode-todo-highlight",
    "streetsidesoftware.code-spell-checker"
  ]
}
```

### ⚙️ Configurações VS Code

```json
// .vscode/settings.json
{
  // Python
  "python.defaultInterpreterPath": "./venv/bin/python",
  "python.formatting.provider": "none",
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": true,
  "python.testing.pytestEnabled": true,
  "python.testing.unittestEnabled": false,
  
  // Formatação
  "[python]": {
    "editor.defaultFormatter": "ms-python.black-formatter",
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
      "source.organizeImports": true
    }
  },
  
  // Black formatter
  "black-formatter.args": ["--line-length=88"],
  
  // isort
  "isort.args": ["--profile", "black"],
  
  // Ruff
  "ruff.args": ["--line-length=88"],
  
  // Files
  "files.exclude": {
    "**/__pycache__": true,
    "**/*.pyc": true,
    ".pytest_cache": true,
    ".coverage": true,
    "htmlcov": true,
    ".ruff_cache": true
  },
  
  // Editor
  "editor.rulers": [88],
  "editor.wordWrap": "bounded",
  "editor.wordWrapColumn": 88,
  
  // Terminal
  "terminal.integrated.defaultProfile.windows": "PowerShell",
  "terminal.integrated.defaultProfile.linux": "bash",
  "terminal.integrated.defaultProfile.osx": "zsh"
}
```

## 🐍 Configuração Python Avançada

### 🎯 pyproject.toml Completo

```toml
[tool.poetry]
name = "machine-learning-engineer"
version = "1.0.0"
description = "Flight delay prediction API with ML pipeline"
authors = ["Ulisses Bomjardim <ulisses.bomjardim@gmail.com>"]
readme = "README.md"
packages = [{include = "src"}]

[tool.poetry.dependencies]
python = ">=3.12.0,<4.0"
fastapi = "^0.104.1"
uvicorn = "^0.24.0"
pandas = "^2.1.4"
scikit-learn = "^1.3.2"
pydantic = "^2.5.1"
python-multipart = "^0.0.6"
pymongo = {version = "^4.6.0", optional = true}
joblib = "^1.3.2"

[tool.poetry.group.dev.dependencies]
pytest = "^7.4.3"
pytest-cov = "^4.1.0"
pytest-xdist = "^3.5.0"
pytest-mock = "^3.12.0"
black = "^23.11.0"
isort = "^5.12.0"
ruff = "^0.1.7"
pylint = "^3.0.3"
mypy = "^1.7.1"
httpx = "^0.25.2"
jupyter = "^1.0.0"
notebook = "^7.0.6"
ipykernel = "^6.27.1"
mkdocs = "^1.5.3"
mkdocs-material = "^9.4.10"
pre-commit = "^3.6.0"

[tool.poetry.group.extras.dependencies]
matplotlib = "^3.8.2"
seaborn = "^0.13.0"
plotly = "^5.17.0"
ydata-profiling = "^4.6.4"

[tool.poetry.extras]
mongodb = ["pymongo"]
viz = ["matplotlib", "seaborn", "plotly"]
profiling = ["ydata-profiling"]

[tool.taskipy.tasks]
test = "pytest tests/ -v"
test-cov = "pytest tests/ --cov=src --cov-report=term-missing --cov-report=html"
format = "black src/ tests/ && isort src/ tests/"
lint = "ruff check src/ tests/ && pylint src/ tests/"
type-check = "mypy src/ tests/"
docs = "mkdocs serve"
api = "uvicorn src.routers.main:app --reload"
clean = "find . -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"

# Configurações de ferramentas
[tool.black]
line-length = 88
target-version = ['py312']
include = '\.pyi?$'
extend-exclude = '''
/(
  \.git
  | \.mypy_cache
  | \.pytest_cache
  | \.ruff_cache
  | __pycache__
  | build
  | dist
)/
'''

[tool.isort]
profile = "black"
line_length = 88
multi_line_output = 3
include_trailing_comma = true
force_grid_wrap = 0
use_parentheses = true
ensure_newline_before_comments = true

[tool.ruff]
line-length = 88
target-version = "py312"
select = [
  "E",   # pycodestyle errors
  "W",   # pycodestyle warnings  
  "F",   # pyflakes
  "I",   # isort
  "B",   # flake8-bugbear
  "C4",  # flake8-comprehensions
  "UP",  # pyupgrade
]
ignore = [
  "E501",  # line too long (handled by black)
  "B008",  # do not perform function calls in argument defaults
]

[tool.mypy]
python_version = "3.12"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
check_untyped_defs = true
disallow_untyped_decorators = true
no_implicit_optional = true
warn_redundant_casts = true
warn_unused_ignores = true
warn_no_return = true
warn_unreachable = true
strict_equality = true

[[tool.mypy.overrides]]
module = [
  "sklearn.*",
  "pandas.*",
  "numpy.*",
  "matplotlib.*",
  "seaborn.*",
]
ignore_missing_imports = true

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = [
  "-v",
  "--strict-markers",
  "--tb=short",
  "--cov=src",
  "--cov-report=term-missing",
  "--cov-fail-under=85",
]
markers = [
  "slow: marks tests as slow (deselect with '-m \"not slow\"')",
  "integration: marks tests as integration tests",
  "unit: marks tests as unit tests",
  "api: marks tests as API tests",
]
filterwarnings = [
  "ignore::UserWarning",
  "ignore::DeprecationWarning",
]

[tool.coverage.run]
source = ["src"]
omit = [
  "*/tests/*",
  "*/test_*",
  "*/__pycache__/*",
  "*/venv/*",
  "*/.venv/*",
]

[tool.coverage.report]
exclude_lines = [
  "pragma: no cover",
  "def __repr__",
  "raise AssertionError", 
  "raise NotImplementedError",
  "if __name__ == .__main__.:",
]
```

### 🔧 Pre-commit Configuration

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
      - id: check-merge-conflict
      - id: debug-statements

  - repo: https://github.com/psf/black
    rev: 23.11.0
    hooks:
      - id: black
        language_version: python3.12

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
        args: ["--profile", "black"]

  - repo: https://github.com/charliermarsh/ruff-pre-commit
    rev: v0.1.7
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.7.1
    hooks:
      - id: mypy
        additional_dependencies: [types-requests]
```

**Instalar pre-commit:**
```bash
# Instalar hooks
poetry run pre-commit install

# Executar em todos os arquivos
poetry run pre-commit run --all-files

# Atualizar hooks
poetry run pre-commit autoupdate
```

## 🧪 Configuração de Testes

### 📋 pytest.ini Avançado

```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    -v
    --strict-markers
    --tb=short
    --cov=src
    --cov-report=term-missing
    --cov-report=html
    --cov-fail-under=85
    --maxfail=5
    --disable-warnings

markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests  
    unit: marks tests as unit tests
    api: marks tests as API tests
    ml: marks tests as ML tests
    database: marks tests requiring database
    
filterwarnings =
    ignore::UserWarning
    ignore::DeprecationWarning
    ignore::PendingDeprecationWarning

log_cli = true
log_cli_level = INFO
log_cli_format = %(asctime)s [%(levelname)8s] %(name)s: %(message)s
log_cli_date_format = %Y-%m-%d %H:%M:%S
```

### 🔧 Configuração de Coverage

```ini
# .coveragerc
[run]
source = src
omit = 
    */tests/*
    */test_*
    */__pycache__/*
    */venv/*
    */.venv/*
    */migrations/*
    */settings/*

[report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise AssertionError
    raise NotImplementedError
    if __name__ == .__main__.:
    @abstract
    @abstractmethod

precision = 2
show_missing = true
skip_covered = false

[html]
directory = htmlcov
title = Machine Learning Engineer Coverage Report
```

## 🐳 Docker para Desenvolvimento

### 📋 Dockerfile.dev

```dockerfile
# Dockerfile.dev - Otimizado para desenvolvimento
FROM python:3.12-slim

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    gcc \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Configurar diretório de trabalho
WORKDIR /app

# Instalar Poetry
RUN pip install poetry

# Configurar Poetry para não criar venv (usará container)
RUN poetry config virtualenvs.create false

# Copiar arquivos de dependências
COPY pyproject.toml poetry.lock ./

# Instalar dependências (incluindo dev)
RUN poetry install

# Copiar código fonte
COPY . .

# Expor porta
EXPOSE 8000

# Comando padrão para desenvolvimento
CMD ["uvicorn", "src.routers.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
```

### 🔄 docker-compose.dev.yml

```yaml
# docker-compose.dev.yml - Setup de desenvolvimento
version: '3.8'

services:
  api:
    build:
      context: .
      dockerfile: Dockerfile.dev
    ports:
      - "8000:8000"
    volumes:
      # Hot reload - código em tempo real
      - .:/app
      # Cache Poetry
      - poetry-cache:/root/.cache/pypoetry
    environment:
      - ENVIRONMENT=development
      - LOG_LEVEL=DEBUG
      - DATABASE_URL=mongodb://mongodb:27017/flight_predictions
    depends_on:
      - mongodb
    networks:
      - ml-dev-network

  mongodb:
    image: mongo:7.0
    ports:
      - "27017:27017"
    volumes:
      - mongodb-dev-data:/data/db
      - ./docker/mongo-init:/docker-entrypoint-initdb.d:ro
    environment:
      - MONGO_INITDB_ROOT_USERNAME=devuser
      - MONGO_INITDB_ROOT_PASSWORD=devpass
      - MONGO_INITDB_DATABASE=flight_predictions
    networks:
      - ml-dev-network

  jupyter:
    build:
      context: .
      dockerfile: Dockerfile.dev
    ports:
      - "8888:8888"
    volumes:
      - .:/app
      - jupyter-data:/root/.jupyter
    command: >
      bash -c "
        pip install jupyterlab &&
        jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
          --NotebookApp.token='' --NotebookApp.password=''
      "
    networks:
      - ml-dev-network

volumes:
  mongodb-dev-data:
  poetry-cache:
  jupyter-data:

networks:
  ml-dev-network:
    driver: bridge
```

**Comandos de desenvolvimento:**
```bash
# Iniciar ambiente completo de dev
docker-compose -f docker-compose.dev.yml up --build

# Apenas API
docker-compose -f docker-compose.dev.yml up api

# Apenas Jupyter
docker-compose -f docker-compose.dev.yml up jupyter

# Logs em tempo real
docker-compose -f docker-compose.dev.yml logs -f api
```

## 📊 Debugging e Profiling

### 🔍 Configuração de Debug

```python
# debug_config.py
import logging
import sys
from typing import Any, Dict

def setup_debug_logging():
    """Configuração de logging para debug"""
    
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('debug.log')
        ]
    )
    
    # Configurar loggers específicos
    logging.getLogger('uvicorn').setLevel(logging.INFO)
    logging.getLogger('fastapi').setLevel(logging.DEBUG)
    logging.getLogger('sklearn').setLevel(logging.WARNING)

def debug_request(request_data: Dict[str, Any]) -> None:
    """Helper para debug de requests"""
    logger = logging.getLogger(__name__)
    logger.debug(f"Request received: {request_data}")
    
    # Validar estrutura
    if isinstance(request_data, dict):
        logger.debug(f"Request keys: {list(request_data.keys())}")
        
        if 'features' in request_data:
            features = request_data['features']
            logger.debug(f"Features type: {type(features)}")
            logger.debug(f"Features content: {features}")

class PerformanceProfiler:
    """Profiler para análise de performance"""
    
    def __init__(self):
        self.timings = {}
    
    def __enter__(self):
        import time
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        
    def time_function(self, func_name: str, func, *args, **kwargs):
        """Cronometrar execução de função"""
        import time
        
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        
        self.timings[func_name] = end - start
        
        logger = logging.getLogger(__name__)
        logger.debug(f"{func_name} executado em {end - start:.4f}s")
        
        return result

# Uso:
# with PerformanceProfiler() as profiler:
#     # código a ser medido
#     pass
# print(f"Execução levou {profiler.duration:.4f}s")
```

### 📊 Memory Profiling

```python
# memory_profiler.py
import psutil
import tracemalloc
from typing import Dict, Any
import logging

class MemoryProfiler:
    """Profiler de memória para debug"""
    
    def __init__(self):
        self.process = psutil.Process()
        self.initial_memory = None
        
    def start_profiling(self):
        """Inicia profiling de memória"""
        tracemalloc.start()
        self.initial_memory = self.process.memory_info().rss
        
    def get_memory_usage(self) -> Dict[str, Any]:
        """Retorna uso atual de memória"""
        current, peak = tracemalloc.get_traced_memory()
        process_memory = self.process.memory_info().rss
        
        return {
            "current_tracemalloc_mb": current / 1024 / 1024,
            "peak_tracemalloc_mb": peak / 1024 / 1024,
            "process_memory_mb": process_memory / 1024 / 1024,
            "memory_increase_mb": (process_memory - self.initial_memory) / 1024 / 1024 if self.initial_memory else 0
        }
    
    def log_top_memory_usage(self, limit: int = 10):
        """Log dos maiores consumidores de memória"""
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')
        
        logger = logging.getLogger(__name__)
        logger.info(f"Top {limit} memory consumers:")
        
        for index, stat in enumerate(top_stats[:limit], 1):
            logger.info(f"{index}. {stat}")

# Decorator para profiling automático
def profile_memory(func):
    """Decorator para profiling de memória"""
    def wrapper(*args, **kwargs):
        profiler = MemoryProfiler()
        profiler.start_profiling()
        
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            memory_info = profiler.get_memory_usage()
            logger = logging.getLogger(__name__)
            logger.info(f"{func.__name__} memory usage: {memory_info}")
            
    return wrapper
```

## 🔄 Workflow de Desenvolvimento

### 📋 Gitflow Simplificado

```bash
# 1. Configurar repositório
git config user.name "Seu Nome"
git config user.email "seu.email@exemplo.com"

# 2. Criar branch para feature
git checkout -b feature/nova-funcionalidade

# 3. Fazer alterações e commits frequentes
git add .
git commit -m "feat: adiciona nova funcionalidade"

# 4. Push da branch
git push -u origin feature/nova-funcionalidade

# 5. Criar Pull Request no GitHub
# 6. Após aprovação, merge via GitHub
# 7. Limpar branch local
git checkout main
git pull origin main
git branch -d feature/nova-funcionalidade
```

### 🎯 Conventional Commits

Padrão de mensagens de commit:

```bash
# Tipos de commit
feat: nova funcionalidade
fix: correção de bug  
docs: documentação
style: formatação
refactor: refatoração
test: testes
chore: tarefas de manutenção

# Exemplos
git commit -m "feat: adiciona endpoint de predição em lote"
git commit -m "fix: corrige validação de datas no modelo"
git commit -m "docs: atualiza documentação da API"
git commit -m "test: adiciona testes para serviço de ML"
```

### 🔄 Desenvolvimento Iterativo

```bash
# Ciclo típico de desenvolvimento
poetry shell                    # Ativar ambiente
task test                      # Executar testes
# Fazer alterações no código
task format                    # Formatar código
task lint                      # Verificar qualidade
task test                      # Testar novamente
uvicorn src.routers.main:app --reload  # Testar API
git add . && git commit -m "..."       # Commit
```

## 📊 Monitoramento Local

### 📈 Métricas de Desenvolvimento

```python
# dev_metrics.py
import time
import psutil
from typing import Dict, Any
from datetime import datetime

class DevMetrics:
    """Métricas para desenvolvimento local"""
    
    def __init__(self):
        self.start_time = time.time()
        self.request_count = 0
        self.error_count = 0
        
    def record_request(self, processing_time: float):
        """Registra métrica de request"""
        self.request_count += 1
        
        if hasattr(self, 'processing_times'):
            self.processing_times.append(processing_time)
        else:
            self.processing_times = [processing_time]
    
    def record_error(self):
        """Registra erro"""
        self.error_count += 1
    
    def get_summary(self) -> Dict[str, Any]:
        """Retorna sumário das métricas"""
        uptime = time.time() - self.start_time
        
        summary = {
            "uptime_seconds": uptime,
            "total_requests": self.request_count,
            "total_errors": self.error_count,
            "error_rate": self.error_count / max(self.request_count, 1),
            "requests_per_minute": self.request_count / (uptime / 60) if uptime > 0 else 0
        }
        
        if hasattr(self, 'processing_times') and self.processing_times:
            summary.update({
                "avg_processing_time": sum(self.processing_times) / len(self.processing_times),
                "min_processing_time": min(self.processing_times),
                "max_processing_time": max(self.processing_times)
            })
        
        # Métricas do sistema
        summary.update({
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent
        })
        
        return summary

# Instância global
dev_metrics = DevMetrics()
```

## 🚀 Scripts Úteis

### 📋 setup_dev.sh

```bash
#!/bin/bash
# setup_dev.sh - Script de setup completo

set -e

echo "🚀 Configurando ambiente de desenvolvimento..."

# Verificar Python
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Instalar Poetry se não existir
if ! command -v poetry &> /dev/null; then
    echo "📦 Instalando Poetry..."
    curl -sSL https://install.python-poetry.org | python3 -
fi

# Configurar Poetry
echo "⚙️ Configurando Poetry..."
poetry config virtualenvs.in-project true

# Instalar dependências
echo "📚 Instalando dependências..."
poetry install

# Configurar pre-commit
echo "🔧 Configurando pre-commit..."
poetry run pre-commit install

# Executar testes iniciais
echo "🧪 Executando testes..."
poetry run task test

# Verificar formatação
echo "🎨 Verificando formatação..."
poetry run task format

echo "✅ Ambiente configurado com sucesso!"
echo ""
echo "📋 Próximos passos:"
echo "1. poetry shell (ativar ambiente)"
echo "2. task api (iniciar API)"
echo "3. task docs (visualizar docs)"
```

### 📊 check_health.py

```python
#!/usr/bin/env python3
# check_health.py - Script de verificação de saúde

import requests
import json
import sys
from datetime import datetime

def check_api_health():
    """Verifica saúde da API"""
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ API está funcionando")
            print(f"Status: {data.get('status', 'unknown')}")
            return True
        else:
            print(f"❌ API retornou status {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Não foi possível conectar à API")
        return False
    except Exception as e:
        print(f"❌ Erro ao verificar API: {e}")
        return False

def check_dependencies():
    """Verifica dependências Python"""
    try:
        import fastapi
        import pandas
        import sklearn
        print("✅ Dependências principais OK")
        return True
    except ImportError as e:
        print(f"❌ Dependência faltando: {e}")
        return False

def main():
    """Verificação completa de saúde"""
    print(f"🔍 Verificação de saúde - {datetime.now()}")
    print("=" * 50)
    
    checks = [
        ("Dependências Python", check_dependencies),
        ("API Health", check_api_health)
    ]
    
    all_passed = True
    
    for check_name, check_func in checks:
        print(f"\n{check_name}:")
        if not check_func():
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("✅ Todas as verificações passaram!")
        sys.exit(0)
    else:
        print("❌ Algumas verificações falharam!")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

## 📚 Recursos Adicionais

### 📖 Documentação Útil

- **FastAPI**: [https://fastapi.tiangolo.com/](https://fastapi.tiangolo.com/)
- **Poetry**: [https://python-poetry.org/docs/](https://python-poetry.org/docs/)
- **Pytest**: [https://docs.pytest.org/](https://docs.pytest.org/)
- **Black**: [https://black.readthedocs.io/](https://black.readthedocs.io/)
- **Ruff**: [https://docs.astral.sh/ruff/](https://docs.astral.sh/ruff/)

### 🎯 Shortcuts Úteis

```bash
# Aliases úteis para .bashrc/.zshrc
alias pshell="poetry shell"
alias ptest="poetry run task test"
alias pformat="poetry run task format"
alias papi="poetry run task api"
alias pdocs="poetry run task docs"

# Git aliases
alias gst="git status"
alias gco="git checkout"
alias gcb="git checkout -b"
alias gp="git push"
alias gl="git pull"
```

## 📞 Suporte

- 🔧 [Troubleshooting](troubleshooting.md) - Solução de problemas
- 🚀 [Quick Start](../quick-start/setup.md) - Configuração inicial
- 🏗️ [Arquitetura](../architecture/overview.md) - Visão geral do sistema
- 🐛 [Issues](https://github.com/ulissesbomjardim/machine_learning_engineer/issues) - Reportar problemas