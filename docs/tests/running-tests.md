# 🧪 Executando Testes

Guia completo para executar e entender os testes automatizados do projeto Machine Learning Engineer Challenge.

## 📋 Visão Geral dos Testes

O projeto possui uma suíte abrangente de testes automatizados com **100+ testes** cobrindo todas as camadas da aplicação:

- ⚡ **Testes de API** - Endpoints FastAPI
- 🤖 **Testes de ML** - Pipeline de Machine Learning  
- 🗄️ **Testes de Serviços** - Camada de dados
- 🔄 **Testes de Integração** - Fluxos completos
- 🛠️ **Testes de Utilitários** - Funções auxiliares

## 🚀 Executando Testes

### ⚡ Comando Rápido

```bash
# Com ambiente Poetry ativo
task test

# Ou sem ativar ambiente
poetry run task test

# Execução direta com pytest
poetry run pytest
```

### 📊 Com Relatório de Cobertura

```bash
# Testes com coverage
task test-cov

# Ou comando completo
poetry run pytest --cov=src --cov-report=term-missing --cov-report=html

# Visualizar relatório HTML
open htmlcov/index.html  # macOS
start htmlcov/index.html  # Windows
xdg-open htmlcov/index.html  # Linux
```

### 🎯 Execução Seletiva

```bash
# Executar arquivo específico
pytest tests/test_routers.py -v

# Executar classe específica
pytest tests/test_ml_pipeline.py::TestModelTraining -v

# Executar teste específico
pytest tests/test_routers.py::TestAPIMain::test_health_endpoint -v

# Executar por marcação
pytest -m "not slow" -v

# Executar testes que contém palavra
pytest -k "test_predict" -v
```

## 📊 Estrutura dos Testes

### 🗂️ Organização dos Arquivos

```
tests/
├── 🧪 conftest.py              # Configurações e fixtures globais
├── ⚡ test_routers.py          # Testes dos endpoints da API
├── 🤖 test_ml_pipeline.py      # Testes do pipeline de ML
├── 🔄 test_integration.py      # Testes de integração end-to-end
├── 🗄️ test_services.py         # Testes da camada de serviços
├── 🛠️ test_utils.py            # Testes de utilitários
└── 📋 run_tests.py            # Script de execução personalizado
```

### 📈 Cobertura Atual

```
======================== Coverage Report ========================
Name                           Stmts   Miss  Cover   Missing
------------------------------------------------------------
src/routers/main.py               45      3    93%   
src/routers/model/predict.py      68      5    93%   
src/routers/model/load.py         42      2    95%   
src/routers/model/history.py      35      1    97%   
src/services/database.py          58      8    86%   
------------------------------------------------------------
TOTAL                            248     19    92%
```

## ⚡ Testes da API (test_routers.py)

### 🎯 Cobertura dos Endpoints

| **Classe de Teste** | **Endpoints Testados** | **Cenários** |
|---------------------|------------------------|--------------|
| `TestAPIMain` | `/`, `/health`, `/docs` | Status codes, response format |
| `TestPredictEndpoint` | `/model/predict` | Predição única, batch, validação |
| `TestModelLoader` | `/model/load/*` | Carregamento, upload, erros |
| `TestHistoryEndpoint` | `/model/history/` | Paginação, filtros, estatísticas |
| `TestAPIErrorHandling` | Todos | Códigos de erro, mensagens |

**Exemplo de execução:**
```bash
# Executar apenas testes da API
pytest tests/test_routers.py -v

# Saída esperada:
tests/test_routers.py::TestAPIMain::test_root_endpoint PASSED
tests/test_routers.py::TestAPIMain::test_health_endpoint PASSED
tests/test_routers.py::TestPredictEndpoint::test_predict_success PASSED
tests/test_routers.py::TestPredictEndpoint::test_predict_batch PASSED
...
======================== 25 passed in 8.45s ========================
```

### 💡 Casos de Teste Principais

**Health Check:**
```python
def test_health_endpoint(api_client):
    response = api_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["healthy", "unhealthy"]
    assert "timestamp" in data
    assert "components" in data
```

**Predição com Validação:**
```python
def test_predict_with_validation(api_client, sample_flight_data):
    response = api_client.post("/model/predict", json=sample_flight_data)
    assert response.status_code == 200
    
    data = response.json()
    assert "prediction" in data
    assert "probability" in data["prediction"]
    assert 0 <= data["prediction"]["probability"] <= 1
```

## 🤖 Testes de ML (test_ml_pipeline.py)

### 🧠 Cobertura do Pipeline

| **Classe de Teste** | **Componente Testado** | **Validações** |
|---------------------|------------------------|----------------|
| `TestDataProcessing` | Preprocessamento | Limpeza, transformações |
| `TestFeatureEngineering` | Feature engineering | Criação de features |
| `TestModelTraining` | Treinamento | Algoritmos, hiperparâmetros |
| `TestModelEvaluation` | Avaliação | Métricas, validação cruzada |
| `TestModelPersistence` | Persistência | Salvamento, carregamento |

**Exemplo de execução:**
```bash
# Executar testes de ML
pytest tests/test_ml_pipeline.py -v --tb=short

# Com logs de ML
pytest tests/test_ml_pipeline.py -v -s
```

### 📊 Testes de Qualidade do Modelo

```python
def test_model_accuracy_threshold():
    """Garante que o modelo atende critério mínimo de qualidade"""
    model = load_trained_model()
    X_test, y_test = load_test_data()
    
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    # Critério de qualidade: mínimo 85% de accuracy
    assert accuracy >= 0.85, f"Accuracy {accuracy:.2f} abaixo do mínimo 0.85"
```

## 🔄 Testes de Integração (test_integration.py)

### 🌐 Fluxos End-to-End

```python
def test_complete_prediction_workflow(api_client):
    """Testa fluxo completo: carregar modelo → predição → histórico"""
    
    # 1. Carregar modelo
    load_response = api_client.get("/model/load/default")
    assert load_response.status_code == 200
    
    # 2. Fazer predição
    predict_response = api_client.post("/model/predict", json=sample_data)
    assert predict_response.status_code == 200
    
    prediction_id = predict_response.json()["prediction_id"]
    
    # 3. Verificar no histórico
    history_response = api_client.get("/model/history/")
    assert history_response.status_code == 200
    
    history_data = history_response.json()
    prediction_ids = [p["prediction_id"] for p in history_data["predictions"]]
    assert prediction_id in prediction_ids
```

### 🎯 Testes de Performance

```python
@pytest.mark.performance
def test_prediction_performance():
    """Garante que predições são executadas em tempo aceitável"""
    import time
    
    start_time = time.time()
    
    # Fazer múltiplas predições
    for _ in range(100):
        response = api_client.post("/model/predict", json=sample_data)
        assert response.status_code == 200
    
    elapsed_time = time.time() - start_time
    
    # Critério: máximo 10ms por predição em média
    assert elapsed_time / 100 < 0.01, f"Performance inadequada: {elapsed_time/100:.3f}s por predição"
```

## 🗄️ Testes de Serviços (test_services.py)

### 💾 Testes de Database

```python
def test_database_connection():
    """Testa conexão com banco de dados"""
    from src.services.database import get_database
    
    db = get_database()
    assert db is not None
    
    # Teste de inserção
    test_doc = {"test": "data", "timestamp": datetime.now()}
    result = db.predictions.insert_one(test_doc)
    assert result.inserted_id is not None
    
    # Limpeza
    db.predictions.delete_one({"_id": result.inserted_id})
```

## 🛠️ Configuração dos Testes

### 📋 Fixtures Principais (conftest.py)

```python
@pytest.fixture
def api_client():
    """Cliente de teste para a API"""
    from fastapi.testclient import TestClient
    from src.routers.main import app
    
    return TestClient(app)

@pytest.fixture
def sample_flight_data():
    """Dados de exemplo para testes"""
    return {
        "features": {
            "airline": "American Airlines",
            "flight_number": "AA123",
            "departure_airport": "JFK",
            "arrival_airport": "LAX",
            "scheduled_departure": "2024-01-15T10:00:00",
            "scheduled_arrival": "2024-01-15T14:00:00"
        }
    }

@pytest.fixture
def mock_model():
    """Modelo mock para testes sem dependências"""
    class MockModel:
        def predict(self, X):
            return [0] * len(X)  # Sempre não cancelado
        def predict_proba(self, X):
            return [[0.8, 0.2]] * len(X)
    
    return MockModel()
```

### ⚙️ Configuração pytest.ini

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
    --cov-fail-under=85

markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests
    unit: marks tests as unit tests
    performance: marks tests as performance tests
```

## 📊 Relatórios de Cobertura

### 🎯 Coverage HTML

```bash
# Gerar relatório HTML completo
poetry run pytest --cov=src --cov-report=html --cov-report=term

# Estrutura do relatório gerado:
htmlcov/
├── index.html              # Página principal
├── src_routers_main_py.html   # Coverage por arquivo
└── ...                     # Outros arquivos
```

### 📋 Coverage XML (CI/CD)

```bash
# Para integração com CI/CD
poetry run pytest --cov=src --cov-report=xml

# Gera: coverage.xml
```

## 🚨 Debugging de Testes

### 🔍 Execução com Debug

```bash
# Modo verbose com traceback completo
pytest -vvv --tb=long

# Parar no primeiro erro
pytest -x

# Executar com pdb (debugger)
pytest --pdb

# Mostrar prints durante execução
pytest -s

# Executar apenas testes que falharam na última execução
pytest --lf
```

### 📋 Logs Detalhados

```python
import logging

def test_with_logging(caplog):
    """Teste que captura logs"""
    with caplog.at_level(logging.INFO):
        # Código que gera logs
        pass
    
    assert "Expected log message" in caplog.text
```

## 🎯 Marcações de Teste

### 🏷️ Usando Markers

```python
import pytest

@pytest.mark.slow
def test_expensive_operation():
    """Teste que demora para executar"""
    pass

@pytest.mark.integration
def test_full_workflow():
    """Teste de integração"""
    pass

@pytest.mark.parametrize("input,expected", [
    ("JFK", "New York"),
    ("LAX", "Los Angeles"),
])
def test_airport_codes(input, expected):
    """Teste parametrizado"""
    pass
```

**Executar por marcação:**
```bash
# Apenas testes rápidos
pytest -m "not slow"

# Apenas testes de integração
pytest -m integration

# Combinações
pytest -m "integration and not slow"
```

## 🔄 CI/CD Integration

### 🚀 GitHub Actions

```yaml
# .github/workflows/tests.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.12
        
    - name: Install Poetry
      run: pip install poetry
      
    - name: Install dependencies
      run: poetry install
      
    - name: Run tests
      run: poetry run pytest --cov=src --cov-report=xml
      
    - name: Upload coverage
      uses: codecov/codecov-action@v1
```

## 📚 Melhores Práticas

### ✅ Princípios de Bons Testes

1. **🎯 AAA Pattern**: Arrange, Act, Assert
2. **🔬 Isolamento**: Testes independentes
3. **📋 Nomenclatura clara**: Nomes descritivos
4. **⚡ Velocidade**: Testes rápidos
5. **🔁 Determinismo**: Resultados consistentes

### 🚨 Evitar

- ❌ **Testes interdependentes**
- ❌ **Hard-coded values** sem contexto
- ❌ **Testes muito longos**
- ❌ **Múltiplas assertivas não relacionadas**
- ❌ **Dados de teste não realistas**

## 📞 Suporte

### 🐛 Problemas com Testes

- 🔧 [Troubleshooting](../dev/troubleshooting.md)
- 📖 [Coverage Detalhado](coverage.md)
- 🔄 [Testes de Integração](integration.md)
- 🐛 [Issues](https://github.com/ulissesbomjardim/machine_learning_engineer/issues)