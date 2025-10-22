# 🔧 Troubleshooting

Guia completo para resolução de problemas comuns do projeto Machine Learning Engineer Challenge.

## 🚨 Problemas Mais Comuns

### 🐍 Problemas com Python

#### ❌ Versão incorreta do Python

**Sintoma:**
```bash
python --version
# Python 3.11.x ou outra versão
```

**Soluções:**

=== "Pyenv (Recomendado)"
    ```bash
    # Instalar Python 3.12.7
    pyenv install 3.12.7
    pyenv local 3.12.7
    
    # Verificar
    python --version
    # Esperado: Python 3.12.7
    ```

=== "Poetry"
    ```bash
    # Forçar Poetry a usar versão correta
    poetry env use 3.12.7
    poetry env info
    ```

=== "Manual"
    ```bash
    # Windows: baixar de python.org
    # Linux: usar apt/yum
    # macOS: usar homebrew
    ```

#### ❌ Poetry não reconhece Python

**Sintoma:**
```bash
poetry env use 3.12.7
# The specified Python version is not available
```

**Soluções:**
```bash
# 1. Verificar caminhos disponíveis
which python3.12
where python  # Windows

# 2. Usar caminho completo
poetry env use /usr/bin/python3.12  # Linux/macOS
poetry env use C:\Python312\python.exe  # Windows

# 3. Reinstalar Poetry
pip uninstall poetry
curl -sSL https://install.python-poetry.org | python3 -
```

### 📦 Problemas com Poetry

#### ❌ Conflitos de dependências

**Sintoma:**
```bash
poetry install
# Solving dependencies... (this may take a minute)
# Because project depends on package A (^1.0.0) and package B (^2.0.0),
# version solving failed.
```

**Soluções:**

=== "Limpar Cache"
    ```bash
    # Limpar cache do Poetry
    poetry cache clear pypi --all
    
    # Remover ambiente virtual
    poetry env remove python
    
    # Reinstalar
    poetry install
    ```

=== "Atualizar Dependências"
    ```bash
    # Atualizar pyproject.toml
    poetry update
    
    # Ou atualizar package específico
    poetry update fastapi
    ```

=== "Lock File"
    ```bash
    # Deletar lock file e recriar
    rm poetry.lock
    poetry install
    ```

#### ❌ Ambiente virtual corrompido

**Sintoma:**
```bash
poetry shell
# Virtual environment is corrupted
```

**Soluções:**
```bash
# 1. Remover ambiente completamente
poetry env remove --all

# 2. Recriar ambiente
poetry env use 3.12.7
poetry install

# 3. Verificar
poetry env info
```

### ⚡ Problemas com a API

#### ❌ API não inicia

**Sintomas:**
```bash
uvicorn src.routers.main:app --reload
# ModuleNotFoundError: No module named 'src'
# OU
# Error loading ASGI app
```

**Soluções:**

=== "PYTHONPATH"
    ```bash
    # Executar do diretório raiz
    cd machine_learning_engineer
    
    # Verificar estrutura
    ls src/
    
    # Executar
    poetry run uvicorn src.routers.main:app --reload
    ```

=== "Imports"
    ```bash
    # Verificar se __init__.py existe
    touch src/__init__.py
    touch src/routers/__init__.py
    
    # Testar imports
    poetry run python -c "from src.routers.main import app; print('OK')"
    ```

=== "Porta Ocupada"
    ```bash
    # Verificar porta 8000
    netstat -an | grep :8000  # Linux/macOS
    netstat -an | findstr :8000  # Windows
    
    # Usar porta diferente
    uvicorn src.routers.main:app --port 8001 --reload
    ```

#### ❌ Erro 500 na API

**Sintomas:**
```bash
curl http://localhost:8000/health
# {"detail": "Internal Server Error"}
```

**Diagnóstico:**
```bash
# 1. Verificar logs do uvicorn
# Os logs aparecem no terminal onde uvicorn está executando

# 2. Testar imports manualmente
poetry run python -c "
from src.services.database import get_database
db = get_database()
print('Database OK')
"

# 3. Verificar modelo
ls -la model/
poetry run python -c "
import pickle
with open('model/modelo_arvore_decisao.pkl', 'rb') as f:
    model = pickle.load(f)
print('Model OK')
"
```

### 🧪 Problemas com Testes

#### ❌ Testes falham por imports

**Sintoma:**
```bash
pytest
# ModuleNotFoundError: No module named 'src'
```

**Soluções:**
```bash
# 1. Executar do diretório raiz
cd machine_learning_engineer

# 2. Usar Poetry
poetry run pytest

# 3. Verificar PYTHONPATH no pytest.ini
cat pytest.ini
# testpaths = tests
# python_paths = .
```

#### ❌ Testes falham por dependências

**Sintoma:**
```bash
pytest tests/test_ml_pipeline.py
# ImportError: No module named 'sklearn'
```

**Soluções:**
```bash
# 1. Instalar dependências de teste
poetry install

# 2. Verificar se dependências estão instaladas
poetry run pip list | grep scikit-learn

# 3. Testes com skip condicional
pytest tests/test_ml_pipeline.py -v
# Deve mostrar SKIPPED para módulos não disponíveis
```

### 🐳 Problemas com Docker

#### ❌ Docker build falha

**Sintomas:**
```bash
docker build -t ml-engineer-api .
# Error: failed to solve: process "/bin/sh -c pip install poetry" did not complete successfully
```

**Soluções:**

=== "Cache e Network"
    ```bash
    # Build sem cache
    docker build --no-cache -t ml-engineer-api .
    
    # Verificar rede
    docker run --rm alpine ping google.com
    ```

=== "Multi-stage Debug"
    ```bash
    # Build até estágio específico
    docker build --target base -t ml-engineer-base .
    
    # Entrar no container para debug
    docker run -it ml-engineer-base bash
    ```

=== "Dockerfile Simplificado"
    ```dockerfile
    # Dockerfile.simple
    FROM python:3.12-slim
    
    WORKDIR /app
    
    # Instalar dependências diretamente
    COPY requirements.txt .
    RUN pip install -r requirements.txt
    
    COPY . .
    
    CMD ["uvicorn", "src.routers.main:app", "--host", "0.0.0.0", "--port", "8000"]
    ```

#### ❌ Docker compose falha

**Sintomas:**
```bash
docker-compose up
# Error: service 'api' failed to build
```

**Diagnóstico:**
```bash
# 1. Verificar compose file
docker-compose config

# 2. Build individual
docker-compose build api

# 3. Logs detalhados
docker-compose up --build --verbose

# 4. Verificar volumes
docker volume ls
docker volume inspect ml_mongodb_data
```

### 📊 Problemas com Dados

#### ❌ Arquivo de dados não encontrado

**Sintomas:**
```python
pd.read_json('data/input/voos.json')
# FileNotFoundError: [Errno 2] No such file or directory
```

**Soluções:**
```bash
# 1. Verificar estrutura de dados
ls -la data/input/

# 2. Verificar diretório atual
pwd

# 3. Usar caminhos absolutos
import os
data_path = os.path.join(os.getcwd(), 'data', 'input', 'voos.json')
```

#### ❌ Modelo não carrega

**Sintomas:**
```python
with open('model/modelo_arvore_decisao.pkl', 'rb') as f:
    model = pickle.load(f)
# FileNotFoundError
```

**Soluções:**
```bash
# 1. Verificar se modelo existe
ls -la model/

# 2. Se não existe, treinar modelo
jupyter lab notebook/Model.ipynb

# 3. Verificar compatibilidade de versão
poetry run python -c "
import sklearn
print(f'Scikit-learn version: {sklearn.__version__}')
"
```

### 🌐 Problemas de Rede

#### ❌ MongoDB não conecta

**Sintomas:**
```bash
docker-compose up
# api_1      | pymongo.errors.ServerSelectionTimeoutError: mongodb:27017: [Errno -2] Name or service not known
```

**Soluções:**

=== "Network Debug"
    ```bash
    # Verificar rede Docker
    docker network ls
    docker network inspect ml_default
    
    # Testar conectividade
    docker-compose exec api ping mongodb
    ```

=== "Compose Order"
    ```yaml
    # docker-compose.yml
    services:
      api:
        depends_on:
          - mongodb
        # ...
      mongodb:
        # ...
    ```

=== "Health Check"
    ```yaml
    # Adicionar health check ao MongoDB
    mongodb:
      image: mongo:7.0
      healthcheck:
        test: ["CMD", "mongo", "--eval", "db.adminCommand('ping')"]
        interval: 10s
        timeout: 5s
        retries: 5
    ```

### 📝 Problemas com Notebooks

#### ❌ Kernel não encontrado

**Sintomas:**
```bash
jupyter lab
# No kernel available for Python 3.12
```

**Soluções:**
```bash
# 1. Instalar ipykernel no ambiente Poetry
poetry add ipykernel

# 2. Registrar kernel
poetry run python -m ipykernel install --user --name ml-engineer

# 3. Selecionar kernel correto no Jupyter
# Kernel > Change Kernel > ml-engineer
```

#### ❌ Imports falham no notebook

**Sintomas:**
```python
from src.services.database import get_database
# ModuleNotFoundError: No module named 'src'
```

**Soluções:**
```python
# 1. Adicionar path no notebook
import sys
import os
sys.path.append(os.path.abspath('..'))

# 2. Ou usar PYTHONPATH
import os
os.chdir('..')  # Se notebook está em subpasta
```

## 🔍 Comandos de Diagnóstico

### 🐍 Verificação do Ambiente Python

```bash
# Informações completas do ambiente
poetry env info

# Versão do Python
python --version

# Localização do executável
which python  # Linux/macOS
where python  # Windows

# Pacotes instalados
poetry show

# Verificar imports críticos
poetry run python -c "
import sys
print(f'Python: {sys.version}')

try:
    import fastapi
    print(f'FastAPI: {fastapi.__version__}')
except ImportError as e:
    print(f'FastAPI Error: {e}')

try:
    import pandas
    print(f'Pandas: {pandas.__version__}')
except ImportError as e:
    print(f'Pandas Error: {e}')

try:
    import sklearn
    print(f'Sklearn: {sklearn.__version__}')
except ImportError as e:
    print(f'Sklearn Error: {e}')
"
```

### 🔧 Verificação de Sistema

```bash
# Informações do sistema
uname -a  # Linux/macOS
systeminfo  # Windows

# Espaço em disco
df -h  # Linux/macOS
dir  # Windows

# Memória disponível
free -h  # Linux
top  # macOS
tasklist  # Windows

# Processos Python
ps aux | grep python  # Linux/macOS
tasklist | findstr python  # Windows
```

### 🌐 Verificação de Rede

```bash
# Testar conectividade
ping google.com

# Portas em uso
netstat -tuln  # Linux
netstat -an  # Windows/macOS

# Testar porta específica
telnet localhost 8000

# Processos usando porta
lsof -i :8000  # Linux/macOS
netstat -ano | findstr :8000  # Windows
```

## 🚨 Soluções Rápidas

### ⚡ Reset Completo do Ambiente

```bash
# 1. Limpar tudo
poetry env remove --all
rm -rf .venv/
rm poetry.lock

# 2. Recriar ambiente
poetry env use 3.12.7
poetry install

# 3. Testar
poetry run python -c "print('Environment OK')"
poetry run task test
```

### 🐳 Reset Completo do Docker

```bash
# 1. Parar tudo
docker-compose down --volumes --remove-orphans

# 2. Limpar sistema
docker system prune -a
docker volume prune

# 3. Rebuild
docker-compose build --no-cache
docker-compose up
```

### 📊 Verificação Completa do Projeto

```bash
#!/bin/bash
# check_project.sh

echo "🔍 Verificando projeto..."

# Python e Poetry
echo "1. Python e Poetry:"
poetry --version
poetry env info

# Dependências
echo "2. Dependências críticas:"
poetry run python -c "
import fastapi, pandas, sklearn
print('✅ Dependências OK')
"

# Estrutura de arquivos
echo "3. Estrutura de arquivos:"
ls -la src/ data/ tests/ model/

# Testes
echo "4. Testes rápidos:"
poetry run pytest tests/ -x --tb=short

# API
echo "5. Testando API:"
poetry run python -c "
from fastapi.testclient import TestClient
from src.routers.main import app
client = TestClient(app)
response = client.get('/health')
print(f'Health check: {response.status_code}')
"

echo "✅ Verificação completa!"
```

## 📚 Logs e Debugging

### 📋 Configuração de Logs

```python
# logging_config.py
import logging
import sys

def setup_logging(level=logging.INFO):
    """Configurar logs para debugging"""
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('app.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Logs específicos
    logging.getLogger('uvicorn').setLevel(logging.INFO)
    logging.getLogger('fastapi').setLevel(logging.DEBUG)
    
# Uso nos módulos
import logging
logger = logging.getLogger(__name__)
logger.info("Debugging info here")
```

### 🔍 Debug da API

```python
# src/routers/main.py
import logging
from fastapi import FastAPI, Request

app = FastAPI()

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log de todas as requests"""
    start_time = time.time()
    
    logger.info(f"Request: {request.method} {request.url}")
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    logger.info(f"Response: {response.status_code} ({process_time:.3f}s)")
    
    return response
```

## 📞 Quando Pedir Ajuda

### 🐛 Criando Issues Efetivas

**Template de Issue:**
```markdown
## 🐛 Descrição do Problema
[Descreva o problema claramente]

## 🔄 Passos para Reproduzir
1. Primeiro passo
2. Segundo passo
3. Erro ocorre aqui

## 📋 Informações do Ambiente
- OS: [Windows/Linux/macOS]
- Python: [versão]
- Poetry: [versão]
- Docker: [versão se aplicável]

## 📊 Logs/Screenshots
```
[Cole logs ou screenshots relevantes]
```

## 💡 Tentativas de Solução
[O que já foi tentado]
```

### 📧 Contato Direto

- 📧 **Email**: [ulisses.bomjardim@gmail.com](mailto:ulisses.bomjardim@gmail.com)
- 🐛 **Issues**: [GitHub Issues](https://github.com/ulissesbomjardim/machine_learning_engineer/issues)
- 💬 **Discussões**: [GitHub Discussions](https://github.com/ulissesbomjardim/machine_learning_engineer/discussions)