# 🔗 Testes de Integração

Documentação completa dos testes de integração, incluindo configuração de ambiente, mocks de serviços externos, testes de API end-to-end e validação de pipeline de Machine Learning.

## 🎯 Visão Geral

Os testes de integração verificam se os componentes do sistema trabalham corretamente em conjunto, simulando cenários reais de uso e validando integrações entre diferentes módulos.

## 🏗️ Estrutura dos Testes de Integração

### 📁 Organização dos Arquivos

```
tests/
├── integration/
│   ├── __init__.py
│   ├── conftest.py                 # Configuração fixtures integração
│   ├── test_api_integration.py     # Testes API completos
│   ├── test_ml_pipeline.py         # Testes pipeline ML
│   ├── test_database_integration.py # Testes integração DB
│   ├── test_external_apis.py       # Testes APIs externas
│   └── fixtures/
│       ├── sample_data.json        # Dados de teste
│       ├── mock_responses/         # Respostas mockadas
│       └── test_datasets/          # Datasets de teste
└── e2e/
    ├── __init__.py
    ├── test_complete_workflow.py   # Fluxo completo E2E
    └── test_user_scenarios.py      # Cenários de usuário
```

## ⚙️ Configuração dos Testes

### 📋 conftest.py - Fixtures de Integração

```python
# tests/integration/conftest.py
import pytest
import asyncio
import pandas as pd
from pathlib import Path
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from unittest.mock import AsyncMock, patch
import json

from src.main import app
from src.database import get_db, Base
from src.config import settings
from src.services.database import DatabaseService
from src.services.external_apis import WeatherService, AirportService

# Database de teste
SQLALCHEMY_TEST_DATABASE_URL = "sqlite:///./test.db"

@pytest.fixture(scope="session")
def event_loop():
    """Cria loop de eventos para testes async"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
def test_engine():
    """Engine de teste do SQLAlchemy"""
    engine = create_engine(
        SQLALCHEMY_TEST_DATABASE_URL,
        connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)

@pytest.fixture(scope="session")
def test_session_factory(test_engine):
    """Factory de sessões de teste"""
    TestingSessionLocal = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=test_engine
    )
    return TestingSessionLocal

@pytest.fixture(scope="function")
def test_db(test_session_factory):
    """Sessão de banco para testes individuais"""
    session = test_session_factory()
    try:
        yield session
    finally:
        session.rollback()
        session.close()

@pytest.fixture(scope="function")
def test_app(test_db):
    """Cliente de teste FastAPI com DB mockado"""
    def override_get_db():
        try:
            yield test_db
        finally:
            pass
    
    app.dependency_overrides[get_db] = override_get_db
    
    with TestClient(app) as client:
        yield client
    
    app.dependency_overrides.clear()

@pytest.fixture
def sample_flight_data():
    """Dados de voo para testes"""
    return {
        "flight_number": "AA123",
        "airline": "American Airlines", 
        "origin_airport": "JFK",
        "destination_airport": "LAX",
        "scheduled_departure": "2024-01-15T10:00:00",
        "scheduled_arrival": "2024-01-15T13:30:00",
        "aircraft_type": "Boeing 737",
        "passenger_count": 150,
        "cargo_weight": 5000.5,
        "weather_conditions": {
            "temperature": 22.5,
            "humidity": 65.2,
            "wind_speed": 15.3,
            "visibility": 10.0,
            "precipitation": 0.0
        },
        "airport_info": {
            "origin": {
                "timezone": "America/New_York",
                "elevation": 13,
                "runway_count": 4
            },
            "destination": {
                "timezone": "America/Los_Angeles", 
                "elevation": 38,
                "runway_count": 4
            }
        }
    }

@pytest.fixture
def sample_historical_data():
    """Dataset histórico para testes"""
    data = {
        'flight_number': ['AA123', 'UA456', 'DL789'] * 100,
        'airline': ['American Airlines', 'United Airlines', 'Delta Airlines'] * 100,
        'origin_airport': ['JFK', 'LAX', 'ORD'] * 100,
        'destination_airport': ['LAX', 'ORD', 'JFK'] * 100,
        'scheduled_departure_hour': [10, 14, 18] * 100,
        'day_of_week': [1, 2, 3, 4, 5, 6, 7] * 43,  # ~300 registros
        'month': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] * 25,
        'weather_score': [0.8, 0.6, 0.9] * 100,
        'airport_congestion': [0.3, 0.7, 0.5] * 100,
        'delay_minutes': [0, 15, 30, 45, 0, 0, 10] * 43  # Target variable
    }
    
    return pd.DataFrame(data)

@pytest.fixture
def mock_weather_service():
    """Mock do serviço de clima"""
    with patch('src.services.external_apis.WeatherService') as mock:
        service = AsyncMock()
        
        # Configurar respostas padrão
        service.get_current_weather.return_value = {
            "temperature": 25.0,
            "humidity": 60.0,
            "wind_speed": 10.0,
            "visibility": 10.0,
            "precipitation": 0.0,
            "weather_score": 0.85
        }
        
        service.get_forecast.return_value = [
            {
                "datetime": "2024-01-15T10:00:00",
                "temperature": 25.0,
                "conditions": "Clear"
            }
        ]
        
        mock.return_value = service
        yield service

@pytest.fixture  
def mock_airport_service():
    """Mock do serviço de aeroportos"""
    with patch('src.services.external_apis.AirportService') as mock:
        service = AsyncMock()
        
        # Dados de aeroportos mockados
        airport_data = {
            "JFK": {
                "name": "John F. Kennedy International",
                "city": "New York",
                "country": "United States",
                "timezone": "America/New_York",
                "elevation": 13,
                "runway_count": 4,
                "capacity_score": 0.9
            },
            "LAX": {
                "name": "Los Angeles International",
                "city": "Los Angeles", 
                "country": "United States",
                "timezone": "America/Los_Angeles",
                "elevation": 38,
                "runway_count": 4,
                "capacity_score": 0.85
            }
        }
        
        service.get_airport_info.side_effect = lambda code: airport_data.get(code)
        service.get_congestion_level.return_value = 0.3
        
        mock.return_value = service
        yield service

@pytest.fixture
def test_model_artifacts(tmp_path):
    """Cria artefatos de modelo para teste"""
    
    # Criar diretório de modelos temporário
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    
    # Mock de modelo treinado (pickle)
    import pickle
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    
    # Modelo simples para teste
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    scaler = StandardScaler()
    
    # Treinar com dados dummy
    X_dummy = [[1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [3, 4, 5, 6, 7]]
    y_dummy = [10, 20, 30]
    
    scaler.fit(X_dummy)
    model.fit(scaler.transform(X_dummy), y_dummy)
    
    # Salvar artefatos
    model_path = model_dir / "flight_delay_model.pkl"
    scaler_path = model_dir / "feature_scaler.pkl"
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    # Metadados do modelo
    metadata = {
        "model_version": "1.0.0",
        "training_date": "2024-01-15",
        "features": ["hour", "day_of_week", "month", "weather_score", "congestion"],
        "metrics": {
            "mae": 12.5,
            "rmse": 18.7,
            "r2_score": 0.76
        }
    }
    
    metadata_path = model_dir / "model_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f)
    
    return {
        "model_path": model_path,
        "scaler_path": scaler_path,
        "metadata_path": metadata_path,
        "model_dir": model_dir
    }

@pytest.fixture
def integration_config():
    """Configuração específica para testes de integração"""
    return {
        "api_timeout": 30,
        "max_retries": 3,
        "batch_size": 100,
        "weather_api_key": "test_key_123",
        "airport_api_key": "test_key_456",
        "model_threshold": 0.8,
        "cache_ttl": 300
    }
```

## 🧪 Testes de API Completos

### 🌐 test_api_integration.py

```python
# tests/integration/test_api_integration.py
import pytest
import asyncio
from datetime import datetime, timedelta
import json

class TestAPIIntegration:
    """Testes completos de integração da API"""
    
    @pytest.mark.asyncio
    async def test_predict_endpoint_complete_flow(
        self, 
        test_app, 
        sample_flight_data,
        mock_weather_service,
        mock_airport_service,
        test_model_artifacts
    ):
        """Testa fluxo completo do endpoint de predição"""
        
        # Configurar mocks
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("src.ml.model_loader.MODEL_PATH", test_model_artifacts["model_dir"])
            
            # Fazer predição
            response = test_app.post(
                "/api/v1/predict",
                json=sample_flight_data
            )
            
            # Verificar resposta
            assert response.status_code == 200
            
            result = response.json()
            assert "prediction" in result
            assert "confidence" in result
            assert "metadata" in result
            
            # Verificar estrutura da predição
            prediction = result["prediction"]
            assert isinstance(prediction["delay_minutes"], (int, float))
            assert isinstance(prediction["probability_delayed"], float)
            assert 0 <= prediction["probability_delayed"] <= 1
            
            # Verificar metadados
            metadata = result["metadata"]
            assert "model_version" in metadata
            assert "prediction_time" in metadata
            assert "processing_time_ms" in metadata
    
    @pytest.mark.asyncio
    async def test_batch_prediction_endpoint(
        self,
        test_app,
        sample_flight_data,
        test_model_artifacts
    ):
        """Testa endpoint de predição em lote"""
        
        # Criar múltiplos voos
        flights = []
        for i in range(5):
            flight = sample_flight_data.copy()
            flight["flight_number"] = f"AA{123 + i}"
            flights.append(flight)
        
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("src.ml.model_loader.MODEL_PATH", test_model_artifacts["model_dir"])
            
            response = test_app.post(
                "/api/v1/predict/batch",
                json={"flights": flights}
            )
            
            assert response.status_code == 200
            
            result = response.json()
            assert "predictions" in result
            assert len(result["predictions"]) == 5
            
            # Verificar cada predição
            for pred in result["predictions"]:
                assert "flight_number" in pred
                assert "prediction" in pred
                assert "status" in pred
                assert pred["status"] == "success"
    
    @pytest.mark.asyncio 
    async def test_model_info_endpoint(self, test_app, test_model_artifacts):
        """Testa endpoint de informações do modelo"""
        
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("src.ml.model_loader.MODEL_PATH", test_model_artifacts["model_dir"])
            
            response = test_app.get("/api/v1/model/info")
            
            assert response.status_code == 200
            
            info = response.json()
            assert "version" in info
            assert "features" in info
            assert "metrics" in info
            assert "training_date" in info
    
    @pytest.mark.asyncio
    async def test_health_check_complete(self, test_app):
        """Testa health check completo do sistema"""
        
        response = test_app.get("/health")
        
        assert response.status_code == 200
        
        health = response.json()
        assert "status" in health
        assert "timestamp" in health
        assert "checks" in health
        
        # Verificar checks individuais
        checks = health["checks"]
        expected_checks = ["database", "model", "external_apis"]
        
        for check_name in expected_checks:
            assert check_name in checks
            assert "status" in checks[check_name]
            assert "response_time" in checks[check_name]
    
    def test_error_handling_invalid_data(self, test_app):
        """Testa tratamento de erros com dados inválidos"""
        
        invalid_data = {
            "flight_number": "",  # Inválido
            "origin_airport": "INVALID",  # Código inválido
            "scheduled_departure": "invalid-date"  # Data inválida
        }
        
        response = test_app.post(
            "/api/v1/predict",
            json=invalid_data
        )
        
        assert response.status_code == 422
        
        error = response.json()
        assert "detail" in error
        assert isinstance(error["detail"], list)
        
        # Verificar que todos os campos inválidos foram identificados
        field_errors = [err["loc"][-1] for err in error["detail"]]
        assert "flight_number" in field_errors
        assert "origin_airport" in field_errors
    
    def test_rate_limiting(self, test_app, sample_flight_data):
        """Testa rate limiting da API"""
        
        # Fazer muitas requisições rapidamente
        responses = []
        for i in range(50):  # Assumindo limite de 100/min
            response = test_app.post(
                "/api/v1/predict",
                json=sample_flight_data
            )
            responses.append(response)
        
        # Todas as primeiras requisições devem funcionar
        success_responses = [r for r in responses if r.status_code == 200]
        rate_limited = [r for r in responses if r.status_code == 429]
        
        # Deve haver pelo menos algumas respostas de sucesso
        assert len(success_responses) > 0
        
        # Se houver rate limiting, verificar headers
        if rate_limited:
            limited_response = rate_limited[0]
            headers = limited_response.headers
            assert "X-RateLimit-Limit" in headers
            assert "X-RateLimit-Remaining" in headers

class TestDatabaseIntegration:
    """Testes de integração com banco de dados"""
    
    def test_flight_prediction_storage(
        self, 
        test_db, 
        sample_flight_data,
        test_model_artifacts
    ):
        """Testa armazenamento de predições no banco"""
        
        from src.services.database import DatabaseService
        
        db_service = DatabaseService(test_db)
        
        # Simular predição
        prediction_data = {
            "flight_id": "AA123_20240115",
            "flight_number": "AA123",
            "prediction": {
                "delay_minutes": 25.5,
                "probability_delayed": 0.75,
                "confidence": 0.82
            },
            "input_features": sample_flight_data,
            "model_version": "1.0.0",
            "prediction_time": datetime.now()
        }
        
        # Salvar predição
        prediction_id = db_service.save_prediction(prediction_data)
        assert prediction_id is not None
        
        # Recuperar predição
        stored_prediction = db_service.get_prediction(prediction_id)
        assert stored_prediction is not None
        assert stored_prediction.flight_number == "AA123"
        assert stored_prediction.delay_minutes == 25.5
    
    def test_historical_data_query(self, test_db, sample_historical_data):
        """Testa consultas de dados históricos"""
        
        from src.services.database import DatabaseService
        
        db_service = DatabaseService(test_db)
        
        # Inserir dados históricos (mock)
        # Em um teste real, você popularia a tabela de histórico
        
        # Testar consulta por aeroporto
        jfk_data = db_service.get_historical_data(
            origin_airport="JFK",
            start_date=datetime.now() - timedelta(days=30),
            end_date=datetime.now()
        )
        
        # Verificar estrutura dos dados
        assert isinstance(jfk_data, list)
        
        if jfk_data:  # Se houver dados
            record = jfk_data[0]
            expected_fields = [
                "flight_number", "origin_airport", 
                "destination_airport", "delay_minutes"
            ]
            
            for field in expected_fields:
                assert hasattr(record, field)
    
    def test_model_metrics_tracking(self, test_db):
        """Testa rastreamento de métricas do modelo"""
        
        from src.services.database import DatabaseService
        
        db_service = DatabaseService(test_db)
        
        # Simular métricas de performance
        metrics_data = {
            "model_version": "1.0.0",
            "evaluation_date": datetime.now(),
            "mae": 12.5,
            "rmse": 18.7,
            "r2_score": 0.76,
            "accuracy_threshold_15min": 0.82,
            "sample_count": 1000,
            "dataset_period": "2024-01-01_to_2024-01-31"
        }
        
        # Salvar métricas
        metrics_id = db_service.save_model_metrics(metrics_data)
        assert metrics_id is not None
        
        # Recuperar métricas mais recentes
        latest_metrics = db_service.get_latest_model_metrics()
        assert latest_metrics is not None
        assert latest_metrics.model_version == "1.0.0"
        assert latest_metrics.mae == 12.5

class TestExternalAPIIntegration:
    """Testes de integração com APIs externas"""
    
    @pytest.mark.asyncio
    async def test_weather_api_integration(self, mock_weather_service):
        """Testa integração com API de clima"""
        
        # Testar obtenção de clima atual
        weather = await mock_weather_service.get_current_weather("JFK")
        
        assert weather is not None
        assert "temperature" in weather
        assert "weather_score" in weather
        assert 0 <= weather["weather_score"] <= 1
        
        # Verificar que o serviço foi chamado
        mock_weather_service.get_current_weather.assert_called_once_with("JFK")
    
    @pytest.mark.asyncio
    async def test_airport_api_integration(self, mock_airport_service):
        """Testa integração com API de aeroportos"""
        
        # Testar obtenção de informações do aeroporto
        airport_info = await mock_airport_service.get_airport_info("JFK")
        
        assert airport_info is not None
        assert "name" in airport_info
        assert "timezone" in airport_info
        assert "capacity_score" in airport_info
        
        # Testar nível de congestionamento
        congestion = await mock_airport_service.get_congestion_level("JFK")
        assert isinstance(congestion, float)
        assert 0 <= congestion <= 1
    
    @pytest.mark.asyncio
    async def test_api_failure_handling(self):
        """Testa tratamento de falhas em APIs externas"""
        
        from src.services.external_apis import WeatherService
        from unittest.mock import patch
        import aiohttp
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            # Simular timeout
            mock_get.side_effect = asyncio.TimeoutError("API timeout")
            
            weather_service = WeatherService()
            
            # Deve retornar dados padrão em caso de falha
            weather = await weather_service.get_current_weather("JFK")
            
            assert weather is not None
            assert "error" in weather or "default" in weather
    
    @pytest.mark.asyncio
    async def test_api_retry_mechanism(self):
        """Testa mecanismo de retry para APIs"""
        
        from src.services.external_apis import WeatherService
        from unittest.mock import patch, AsyncMock
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            # Primeira chamada falha, segunda funciona
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = {"temperature": 25.0}
            
            mock_get.side_effect = [
                aiohttp.ClientConnectorError("Connection failed"),
                mock_response
            ]
            
            weather_service = WeatherService(max_retries=2)
            weather = await weather_service.get_current_weather("JFK")
            
            # Deve ter funcionado na segunda tentativa
            assert weather["temperature"] == 25.0
            assert mock_get.call_count == 2
```

## 🤖 Testes de Pipeline ML

### 🔄 test_ml_pipeline.py

```python
# tests/integration/test_ml_pipeline.py
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import tempfile

class TestMLPipelineIntegration:
    """Testes completos do pipeline de ML"""
    
    def test_complete_training_pipeline(
        self, 
        sample_historical_data,
        tmp_path
    ):
        """Testa pipeline completo de treinamento"""
        
        from src.ml.trainer import ModelTrainer
        from src.ml.preprocessor import DataPreprocessor
        
        # Preparar dados
        preprocessor = DataPreprocessor()
        
        # Preprocessing
        processed_data = preprocessor.fit_transform(sample_historical_data)
        assert processed_data is not None
        assert len(processed_data) > 0
        
        # Dividir dados
        X = processed_data.drop(['delay_minutes'], axis=1)
        y = processed_data['delay_minutes']
        
        # Treinar modelo
        trainer = ModelTrainer(model_save_path=tmp_path)
        
        model, metrics = trainer.train(X, y)
        
        # Verificar modelo treinado
        assert model is not None
        assert metrics is not None
        
        # Verificar métricas
        assert "mae" in metrics
        assert "rmse" in metrics
        assert "r2_score" in metrics
        assert metrics["mae"] > 0
        assert metrics["rmse"] > 0
    
    def test_prediction_pipeline_end_to_end(
        self,
        sample_flight_data,
        test_model_artifacts
    ):
        """Testa pipeline completo de predição"""
        
        from src.ml.predictor import FlightDelayPredictor
        
        # Inicializar preditor com modelo de teste
        predictor = FlightDelayPredictor(
            model_path=test_model_artifacts["model_path"],
            scaler_path=test_model_artifacts["scaler_path"]
        )
        
        # Fazer predição
        prediction = predictor.predict(sample_flight_data)
        
        # Verificar estrutura da predição
        assert prediction is not None
        assert "delay_minutes" in prediction
        assert "probability_delayed" in prediction
        assert "confidence" in prediction
        
        # Verificar valores
        assert isinstance(prediction["delay_minutes"], (int, float))
        assert 0 <= prediction["probability_delayed"] <= 1
        assert 0 <= prediction["confidence"] <= 1
    
    def test_feature_engineering_pipeline(self, sample_historical_data):
        """Testa pipeline de engenharia de features"""
        
        from src.ml.feature_engineer import FeatureEngineer
        
        engineer = FeatureEngineer()
        
        # Aplicar engenharia de features
        enhanced_data = engineer.transform(sample_historical_data)
        
        # Verificar novas features criadas
        expected_features = [
            'hour_sin', 'hour_cos',           # Features cíclicas
            'day_of_week_sin', 'day_of_week_cos',
            'month_sin', 'month_cos',
            'weather_category',               # Feature categórica
            'congestion_level',               # Feature derivada
            'route_popularity',               # Feature de rota
            'airline_delay_history'           # Feature histórica
        ]
        
        for feature in expected_features:
            assert feature in enhanced_data.columns, f"Feature {feature} não encontrada"
        
        # Verificar que não há valores nulos em features críticas
        critical_features = ['hour_sin', 'hour_cos', 'weather_category']
        for feature in critical_features:
            if feature in enhanced_data.columns:
                assert not enhanced_data[feature].isnull().any(), f"Valores nulos em {feature}"
    
    def test_model_validation_pipeline(
        self,
        sample_historical_data,
        tmp_path
    ):
        """Testa pipeline de validação do modelo"""
        
        from src.ml.validator import ModelValidator
        from src.ml.trainer import ModelTrainer
        
        # Treinar modelo para validação
        trainer = ModelTrainer(model_save_path=tmp_path)
        
        # Preparar dados
        X = sample_historical_data.drop(['delay_minutes'], axis=1)
        y = sample_historical_data['delay_minutes']
        
        # Treinar
        model, _ = trainer.train(X, y)
        
        # Validar modelo
        validator = ModelValidator()
        
        validation_results = validator.validate_model(
            model=model,
            X_test=X,
            y_test=y
        )
        
        # Verificar resultados da validação
        assert validation_results is not None
        assert "metrics" in validation_results
        assert "validation_passed" in validation_results
        
        metrics = validation_results["metrics"]
        assert "accuracy_by_threshold" in metrics
        assert "confusion_matrix" in metrics
        assert "classification_report" in metrics
    
    def test_data_drift_detection(self, sample_historical_data):
        """Testa detecção de drift nos dados"""
        
        from src.ml.drift_detector import DataDriftDetector
        
        # Dividir dados em referência e atual
        split_idx = len(sample_historical_data) // 2
        reference_data = sample_historical_data[:split_idx]
        current_data = sample_historical_data[split_idx:]
        
        # Simular drift modificando distribuição
        current_data_with_drift = current_data.copy()
        current_data_with_drift['weather_score'] = current_data_with_drift['weather_score'] * 0.5  # Artificial drift
        
        detector = DataDriftDetector()
        
        # Detectar drift
        drift_report = detector.detect_drift(
            reference_data=reference_data,
            current_data=current_data_with_drift
        )
        
        # Verificar relatório
        assert drift_report is not None
        assert "overall_drift" in drift_report
        assert "feature_drift" in drift_report
        assert "drift_score" in drift_report
        
        # Deve detectar drift na feature weather_score
        feature_drift = drift_report["feature_drift"]
        assert "weather_score" in feature_drift
        
        weather_drift = feature_drift["weather_score"]
        assert weather_drift["has_drift"] == True
        assert weather_drift["p_value"] < 0.05  # Significativo
    
    def test_model_monitoring_pipeline(
        self,
        test_model_artifacts,
        sample_flight_data
    ):
        """Testa pipeline de monitoramento do modelo"""
        
        from src.ml.monitor import ModelMonitor
        
        monitor = ModelMonitor(
            model_path=test_model_artifacts["model_path"],
            metadata_path=test_model_artifacts["metadata_path"]
        )
        
        # Simular predições para monitoramento
        predictions = []
        actuals = []
        
        for i in range(10):
            flight_data = sample_flight_data.copy()
            flight_data["flight_number"] = f"TEST{i:03d}"
            
            # Fazer predição
            pred = monitor.predict_and_log(flight_data)
            predictions.append(pred["delay_minutes"])
            
            # Simular valor real (para teste)
            actual_delay = np.random.normal(pred["delay_minutes"], 10)
            actuals.append(max(0, actual_delay))  # Delays não podem ser negativos
            
            # Registrar resultado real
            monitor.log_actual_result(
                flight_id=flight_data["flight_number"],
                actual_delay=actual_delay
            )
        
        # Gerar relatório de performance
        performance_report = monitor.generate_performance_report()
        
        assert performance_report is not None
        assert "current_mae" in performance_report
        assert "current_rmse" in performance_report
        assert "prediction_count" in performance_report
        assert "drift_alerts" in performance_report
        
        # Verificar alertas se houver degradação significativa
        if performance_report["current_mae"] > 20:  # Threshold de alerta
            assert len(performance_report["drift_alerts"]) > 0
    
    def test_automated_retraining_trigger(
        self,
        sample_historical_data,
        tmp_path
    ):
        """Testa trigger automático de retreinamento"""
        
        from src.ml.retraining import AutoRetrainingService
        
        service = AutoRetrainingService(
            model_dir=tmp_path,
            performance_threshold=0.15,  # MAE threshold
            data_drift_threshold=0.1
        )
        
        # Simular degradação de performance
        mock_current_performance = {
            "mae": 25.0,  # Acima do threshold
            "rmse": 35.0,
            "prediction_count": 1000
        }
        
        # Verificar se retreinamento é necessário
        should_retrain = service.should_trigger_retraining(
            current_performance=mock_current_performance,
            baseline_mae=15.0  # Performance baseline
        )
        
        assert should_retrain == True
        
        # Testar retreinamento automático
        if should_retrain:
            retrain_results = service.execute_retraining(
                training_data=sample_historical_data
            )
            
            assert retrain_results is not None
            assert "new_model_path" in retrain_results
            assert "performance_improvement" in retrain_results
            assert "retrain_timestamp" in retrain_results

class TestEndToEndWorkflow:
    """Testes de fluxo completo end-to-end"""
    
    @pytest.mark.asyncio
    async def test_complete_prediction_workflow(
        self,
        test_app,
        sample_flight_data,
        mock_weather_service,
        mock_airport_service,
        test_model_artifacts
    ):
        """Testa fluxo completo desde requisição até resposta"""
        
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("src.ml.model_loader.MODEL_PATH", test_model_artifacts["model_dir"])
            
            # 1. Fazer requisição de predição
            start_time = datetime.now()
            
            response = test_app.post(
                "/api/v1/predict",
                json=sample_flight_data
            )
            
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            # 2. Verificar resposta
            assert response.status_code == 200
            result = response.json()
            
            # 3. Verificar que todos os serviços foram chamados
            mock_weather_service.get_current_weather.assert_called()
            mock_airport_service.get_airport_info.assert_called()
            
            # 4. Verificar performance (deve ser < 5 segundos)
            assert processing_time < 5.0
            
            # 5. Verificar estrutura completa da resposta
            assert "prediction" in result
            assert "confidence" in result
            assert "metadata" in result
            assert "external_data" in result
            
            external_data = result["external_data"]
            assert "weather" in external_data
            assert "airport_info" in external_data
    
    def test_batch_processing_workflow(
        self,
        test_app,
        sample_flight_data,
        test_model_artifacts
    ):
        """Testa processamento em lote completo"""
        
        # Criar lote de voos
        batch_size = 20
        flights = []
        
        for i in range(batch_size):
            flight = sample_flight_data.copy()
            flight["flight_number"] = f"BATCH{i:03d}"
            flight["scheduled_departure"] = (
                datetime.fromisoformat(flight["scheduled_departure"]) + 
                timedelta(hours=i)
            ).isoformat()
            flights.append(flight)
        
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("src.ml.model_loader.MODEL_PATH", test_model_artifacts["model_dir"])
            
            # Processar lote
            response = test_app.post(
                "/api/v1/predict/batch",
                json={"flights": flights}
            )
            
            assert response.status_code == 200
            
            result = response.json()
            predictions = result["predictions"]
            
            # Verificar que todas as predições foram processadas
            assert len(predictions) == batch_size
            
            # Verificar que não há falhas
            failed_predictions = [p for p in predictions if p["status"] == "error"]
            assert len(failed_predictions) == 0
            
            # Verificar tempo de processamento por item
            total_time = result["metadata"]["total_processing_time_ms"]
            avg_time_per_item = total_time / batch_size
            
            # Deve processar cada item em menos de 500ms
            assert avg_time_per_item < 500
```

## 🧪 Execução dos Testes

### 🚀 Comandos de Execução

```bash
# Executar apenas testes de integração
pytest tests/integration/ -v

# Executar com cobertura específica
pytest tests/integration/ --cov=src --cov-report=html

# Executar testes específicos
pytest tests/integration/test_api_integration.py::TestAPIIntegration::test_predict_endpoint_complete_flow -v

# Executar testes de integração com logs detalhados
pytest tests/integration/ -v -s --log-cli-level=INFO

# Executar testes paralelos (requer pytest-xdist)
pytest tests/integration/ -n auto

# Executar apenas testes marcados como 'slow'
pytest tests/integration/ -m "integration and not slow"
```

### ⚙️ Configuração de Ambiente para Testes

```yaml
# docker-compose.test.yml
version: '3.8'

services:
  test-db:
    image: postgres:15
    environment:
      POSTGRES_DB: test_flight_delays
      POSTGRES_USER: test_user
      POSTGRES_PASSWORD: test_pass
    ports:
      - "5433:5432"
    volumes:
      - test_db_data:/var/lib/postgresql/data
  
  test-redis:
    image: redis:7-alpine
    ports:
      - "6380:6379"
  
  test-api:
    build:
      context: .
      dockerfile: docker/api.Dockerfile
    environment:
      - DATABASE_URL=postgresql://test_user:test_pass@test-db:5432/test_flight_delays
      - REDIS_URL=redis://test-redis:6379/0
      - ENVIRONMENT=test
    depends_on:
      - test-db
      - test-redis
    ports:
      - "8001:8000"

volumes:
  test_db_data:
```

### 🔧 Makefile para Testes

```makefile
# Makefile (adições para integração)
.PHONY: test-integration test-e2e test-all-integration

test-integration:
	@echo "🔗 Executando testes de integração..."
	pytest tests/integration/ -v --cov=src --cov-report=term-missing

test-e2e:
	@echo "🌐 Executando testes end-to-end..."
	docker-compose -f docker-compose.test.yml up -d
	sleep 10
	pytest tests/e2e/ -v
	docker-compose -f docker-compose.test.yml down

test-all-integration: test-integration test-e2e
	@echo "✅ Todos os testes de integração concluídos!"

setup-test-env:
	@echo "⚙️ Configurando ambiente de teste..."
	docker-compose -f docker-compose.test.yml up -d test-db test-redis
	sleep 5

cleanup-test-env:
	@echo "🧹 Limpando ambiente de teste..."
	docker-compose -f docker-compose.test.yml down -v
```

## 📊 Relatórios de Integração

### 📋 Exemplo de Relatório de Teste

```
================== RELATÓRIO DE TESTES DE INTEGRAÇÃO ==================

🔗 Testes de API Completos:
  ✅ test_predict_endpoint_complete_flow         PASSED  (2.45s)
  ✅ test_batch_prediction_endpoint             PASSED  (1.87s)
  ✅ test_model_info_endpoint                   PASSED  (0.32s)
  ✅ test_health_check_complete                 PASSED  (0.89s)
  ✅ test_error_handling_invalid_data           PASSED  (0.21s)
  ✅ test_rate_limiting                         PASSED  (3.12s)

🗄️ Testes de Integração Database:
  ✅ test_flight_prediction_storage             PASSED  (0.67s)
  ✅ test_historical_data_query                 PASSED  (0.45s)
  ✅ test_model_metrics_tracking                PASSED  (0.38s)

🌐 Testes de APIs Externas:
  ✅ test_weather_api_integration               PASSED  (0.28s)
  ✅ test_airport_api_integration               PASSED  (0.19s)
  ✅ test_api_failure_handling                  PASSED  (1.15s)
  ✅ test_api_retry_mechanism                   PASSED  (2.34s)

🤖 Testes de Pipeline ML:
  ✅ test_complete_training_pipeline            PASSED  (8.67s)
  ✅ test_prediction_pipeline_end_to_end        PASSED  (1.23s)
  ✅ test_feature_engineering_pipeline          PASSED  (2.11s)
  ✅ test_model_validation_pipeline             PASSED  (3.45s)
  ✅ test_data_drift_detection                  PASSED  (1.89s)
  ✅ test_model_monitoring_pipeline             PASSED  (2.76s)
  ✅ test_automated_retraining_trigger          PASSED  (4.32s)

🌐 Testes End-to-End:
  ✅ test_complete_prediction_workflow          PASSED  (4.21s)
  ✅ test_batch_processing_workflow             PASSED  (6.78s)

📊 RESUMO:
  Total de Testes: 18
  Passou: 18 (100%)
  Falhou: 0 (0%)
  Tempo Total: 42.58s
  Cobertura: 89.4%

✅ TODOS OS TESTES DE INTEGRAÇÃO PASSARAM!
```

## 🔗 Próximos Passos

1. **[📊 Cobertura](coverage.md)** - Análise detalhada de cobertura
2. **[🧪 Testes](running-tests.md)** - Executar suite completa
3. **[📓 Notebooks](../notebooks/eda.md)** - Análise exploratória

---

## 📞 Referências

- 🧪 **[pytest-asyncio](https://pytest-asyncio.readthedocs.io/)** - Testes async
- 🌐 **[TestClient](https://fastapi.tiangolo.com/tutorial/testing/)** - Testes FastAPI
- 🗄️ **[SQLAlchemy Testing](https://docs.sqlalchemy.org/en/14/orm/session_transaction.html#joining-a-session-into-an-external-transaction-such-as-for-test-suites)** - Testes com banco