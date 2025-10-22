# 📋 Modelos de Dados (Pydantic)

Documentação completa dos schemas Pydantic utilizados na API do projeto Machine Learning Engineer Challenge.

## 📋 Visão Geral

A API utiliza **Pydantic** para validação automática, serialização e documentação dos dados. Todos os modelos seguem as melhores práticas de tipagem e validação.

### 🎯 Benefícios do Pydantic

- ✅ **Validação automática** de tipos e valores
- 📚 **Documentação automática** no Swagger
- 🔄 **Serialização** JSON automática  
- 🚨 **Mensagens de erro** claras e detalhadas
- 🛡️ **Segurança** na validação de entrada

## 🛩️ Modelos de Voo (Flight Models)

### FlightFeatures

Representa as características de entrada de um voo para predição.

```python
from pydantic import BaseModel, Field, validator
from datetime import datetime
from typing import Optional

class FlightFeatures(BaseModel):
    """Features de entrada para predição de cancelamento de voo"""
    
    airline: str = Field(
        ..., 
        description="Nome da companhia aérea",
        example="American Airlines",
        min_length=1,
        max_length=100
    )
    
    flight_number: str = Field(
        ...,
        description="Número do voo (formato: AA123)",
        example="AA123",
        regex=r'^[A-Z]{2,3}\d{1,4}$'
    )
    
    departure_airport: str = Field(
        ...,
        description="Código IATA do aeroporto de origem",
        example="JFK",
        min_length=3,
        max_length=3,
        regex=r'^[A-Z]{3}$'
    )
    
    arrival_airport: str = Field(
        ...,
        description="Código IATA do aeroporto de destino", 
        example="LAX",
        min_length=3,
        max_length=3,
        regex=r'^[A-Z]{3}$'
    )
    
    scheduled_departure: datetime = Field(
        ...,
        description="Data e hora prevista de partida (ISO 8601)",
        example="2024-01-15T10:00:00"
    )
    
    scheduled_arrival: datetime = Field(
        ...,
        description="Data e hora prevista de chegada (ISO 8601)",
        example="2024-01-15T14:00:00"
    )
    
    aircraft_type: Optional[str] = Field(
        None,
        description="Tipo de aeronave",
        example="Boeing 737-800",
        max_length=50
    )
    
    weather_condition: Optional[str] = Field(
        None,
        description="Condição climática no momento da partida",
        example="Clear",
        max_length=30
    )
    
    @validator('scheduled_arrival')
    def arrival_after_departure(cls, v, values):
        """Valida que chegada é após partida"""
        if 'scheduled_departure' in values and v <= values['scheduled_departure']:
            raise ValueError('Horário de chegada deve ser após partida')
        return v
    
    @validator('departure_airport', 'arrival_airport')
    def airports_different(cls, v, values):
        """Valida que aeroportos são diferentes"""
        if 'departure_airport' in values and v == values['departure_airport']:
            raise ValueError('Aeroportos de origem e destino devem ser diferentes')
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "airline": "American Airlines",
                "flight_number": "AA123", 
                "departure_airport": "JFK",
                "arrival_airport": "LAX",
                "scheduled_departure": "2024-01-15T10:00:00",
                "scheduled_arrival": "2024-01-15T14:00:00",
                "aircraft_type": "Boeing 737-800",
                "weather_condition": "Clear"
            }
        }
```

### BatchFlightFeatures

Para predições em lote de múltiplos voos.

```python
from typing import List
from pydantic import BaseModel, Field, validator

class BatchFlightFeatures(BaseModel):
    """Lista de voos para predição em lote"""
    
    flights: List[FlightFeatures] = Field(
        ...,
        description="Lista de voos para predição",
        min_items=1,
        max_items=100
    )
    
    @validator('flights')
    def validate_batch_size(cls, v):
        """Valida tamanho do batch"""
        if len(v) > 100:
            raise ValueError('Máximo de 100 voos por batch')
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "flights": [
                    {
                        "airline": "American Airlines",
                        "flight_number": "AA123",
                        "departure_airport": "JFK", 
                        "arrival_airport": "LAX",
                        "scheduled_departure": "2024-01-15T10:00:00",
                        "scheduled_arrival": "2024-01-15T14:00:00"
                    },
                    {
                        "airline": "Delta",
                        "flight_number": "DL456",
                        "departure_airport": "ATL",
                        "arrival_airport": "SEA", 
                        "scheduled_departure": "2024-01-15T11:00:00",
                        "scheduled_arrival": "2024-01-15T14:30:00"
                    }
                ]
            }
        }
```

## 🎯 Modelos de Predição (Prediction Models)

### PredictionRequest

Encapsula request para endpoint de predição.

```python
from typing import Union

class PredictionRequest(BaseModel):
    """Request para predição - suporta único ou batch"""
    
    features: Union[FlightFeatures, BatchFlightFeatures] = Field(
        ...,
        description="Features do voo ou lista de voos"
    )
    
    model_version: Optional[str] = Field(
        None,
        description="Versão específica do modelo (opcional)",
        example="v1.0.0"
    )
    
    include_explanation: bool = Field(
        False,
        description="Incluir explicação da predição (SHAP values)"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "features": {
                    "airline": "American Airlines", 
                    "flight_number": "AA123",
                    "departure_airport": "JFK",
                    "arrival_airport": "LAX",
                    "scheduled_departure": "2024-01-15T10:00:00",
                    "scheduled_arrival": "2024-01-15T14:00:00"
                },
                "include_explanation": False
            }
        }
```

### PredictionResult

Resultado individual de uma predição.

```python
from enum import Enum

class ConfidenceLevel(str, Enum):
    """Níveis de confiança da predição"""
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"

class PredictionResult(BaseModel):
    """Resultado de uma predição individual"""
    
    cancelled: bool = Field(
        ...,
        description="Predição: voo será cancelado?"
    )
    
    probability: float = Field(
        ...,
        ge=0.0,
        le=1.0, 
        description="Probabilidade de cancelamento (0-1)"
    )
    
    confidence: ConfidenceLevel = Field(
        ...,
        description="Nível de confiança na predição"
    )
    
    risk_factors: Optional[List[str]] = Field(
        None,
        description="Principais fatores de risco identificados"
    )
    
    @validator('confidence', pre=True)
    def calculate_confidence(cls, v, values):
        """Calcula confiança baseada na probabilidade"""
        if isinstance(v, str):
            return v
            
        prob = values.get('probability', 0.5)
        
        if prob <= 0.3 or prob >= 0.7:
            return ConfidenceLevel.HIGH
        elif prob <= 0.4 or prob >= 0.6:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW
    
    class Config:
        json_schema_extra = {
            "example": {
                "cancelled": False,
                "probability": 0.23,
                "confidence": "high",
                "risk_factors": ["weather_condition", "busy_route"]
            }
        }
```

### PredictionResponse

Response completa para predições únicas.

```python
from uuid import uuid4

class ModelInfo(BaseModel):
    """Informações do modelo utilizado"""
    
    name: str = Field(..., description="Nome do modelo")
    version: str = Field(..., description="Versão do modelo") 
    algorithm: str = Field(..., description="Algoritmo utilizado")
    accuracy: float = Field(..., description="Acurácia do modelo")
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "flight_cancellation_model",
                "version": "1.0.0", 
                "algorithm": "Decision Tree",
                "accuracy": 0.94
            }
        }

class PredictionResponse(BaseModel):
    """Response completa de predição única"""
    
    prediction: PredictionResult = Field(
        ...,
        description="Resultado da predição"
    )
    
    prediction_id: str = Field(
        default_factory=lambda: f"pred_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}",
        description="ID único da predição"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp da predição"
    )
    
    model_info: ModelInfo = Field(
        ...,
        description="Informações do modelo utilizado"
    )
    
    processing_time_ms: float = Field(
        ...,
        description="Tempo de processamento em milissegundos"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "prediction": {
                    "cancelled": False,
                    "probability": 0.23, 
                    "confidence": "high"
                },
                "prediction_id": "pred_20241221_101500_abc12345",
                "timestamp": "2024-12-21T10:15:00Z",
                "model_info": {
                    "name": "flight_cancellation_model",
                    "version": "1.0.0",
                    "algorithm": "Decision Tree", 
                    "accuracy": 0.94
                },
                "processing_time_ms": 15.6
            }
        }
```

### BatchPredictionResponse

Response para predições em lote.

```python
class BatchPredictionItem(BaseModel):
    """Item individual em predição batch"""
    
    prediction: PredictionResult = Field(..., description="Resultado da predição")
    prediction_id: str = Field(..., description="ID único desta predição")
    input_index: int = Field(..., description="Índice no input batch")

class BatchSummary(BaseModel):
    """Sumário das predições em batch"""
    
    total_predictions: int = Field(..., description="Total de predições realizadas")
    cancelled_count: int = Field(..., description="Número de voos preditos como cancelados")
    cancellation_rate: float = Field(..., description="Taxa de cancelamento do batch")
    average_probability: float = Field(..., description="Probabilidade média")
    processing_time_ms: float = Field(..., description="Tempo total de processamento")

class BatchPredictionResponse(BaseModel):
    """Response para predições em lote"""
    
    predictions: List[BatchPredictionItem] = Field(
        ...,
        description="Lista de predições individuais"
    )
    
    batch_id: str = Field(
        default_factory=lambda: f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}",
        description="ID único do batch"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp do processamento"
    )
    
    summary: BatchSummary = Field(
        ...,
        description="Sumário das predições"
    )
    
    model_info: ModelInfo = Field(
        ..., 
        description="Informações do modelo utilizado"
    )
```

## 📥 Modelos de Upload (Model Management)

### ModelUploadRequest

Para upload de novos modelos.

```python
from fastapi import UploadFile
from pydantic import BaseModel, Field

class ModelMetadata(BaseModel):
    """Metadados do modelo"""
    
    name: str = Field(..., description="Nome do modelo", max_length=50)
    version: str = Field(..., description="Versão (semver)", regex=r'^\d+\.\d+\.\d+$')
    algorithm: str = Field(..., description="Algoritmo utilizado", max_length=30)
    description: Optional[str] = Field(None, description="Descrição", max_length=200)
    
    # Métricas de performance
    accuracy: Optional[float] = Field(None, ge=0, le=1, description="Acurácia")
    precision: Optional[float] = Field(None, ge=0, le=1, description="Precisão")
    recall: Optional[float] = Field(None, ge=0, le=1, description="Recall") 
    f1_score: Optional[float] = Field(None, ge=0, le=1, description="F1 Score")
    
    # Features esperadas
    expected_features: List[str] = Field(
        ...,
        description="Lista de features que o modelo espera"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "flight_cancellation_v2",
                "version": "2.0.0",
                "algorithm": "Random Forest", 
                "description": "Modelo melhorado com mais features",
                "accuracy": 0.96,
                "precision": 0.91,
                "recall": 0.89,
                "f1_score": 0.90,
                "expected_features": [
                    "airline_encoded", "hour_departure", "day_of_week",
                    "flight_duration", "weather_encoded"
                ]
            }
        }

class ModelUploadResponse(BaseModel):
    """Response do upload de modelo"""
    
    status: str = Field(..., description="Status do upload")
    message: str = Field(..., description="Mensagem descritiva")
    
    model_id: str = Field(..., description="ID único do modelo")
    upload_timestamp: datetime = Field(..., description="Timestamp do upload")
    
    validation_results: Dict[str, Any] = Field(
        ...,
        description="Resultados da validação do modelo"
    )
    
    file_info: Dict[str, Any] = Field(
        ...,
        description="Informações do arquivo"
    )
```

## 📊 Modelos de Histórico (History Models)

### HistoryFilter

Filtros para consulta de histórico.

```python
class HistoryFilter(BaseModel):
    """Filtros para consulta de histórico"""
    
    start_date: Optional[datetime] = Field(
        None, 
        description="Data inicial (ISO 8601)"
    )
    
    end_date: Optional[datetime] = Field(
        None,
        description="Data final (ISO 8601)"
    )
    
    airline: Optional[str] = Field(
        None,
        description="Filtrar por companhia aérea"
    )
    
    cancelled_only: bool = Field(
        False,
        description="Mostrar apenas voos cancelados"
    )
    
    min_probability: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Probabilidade mínima de cancelamento"
    )
    
    confidence_level: Optional[ConfidenceLevel] = Field(
        None,
        description="Filtrar por nível de confiança"
    )
    
    @validator('end_date')
    def end_after_start(cls, v, values):
        """Valida que data final é após inicial"""
        if v and 'start_date' in values and values['start_date'] and v <= values['start_date']:
            raise ValueError('Data final deve ser após data inicial')
        return v

class PaginationParams(BaseModel):
    """Parâmetros de paginação"""
    
    limit: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Número máximo de registros por página"
    )
    
    offset: int = Field(
        default=0,
        ge=0,
        description="Número de registros para pular"
    )
    
    sort_by: str = Field(
        default="timestamp",
        description="Campo para ordenação"
    )
    
    sort_order: str = Field(
        default="desc",
        regex="^(asc|desc)$",
        description="Ordem da classificação"
    )

class HistoryQuery(BaseModel):
    """Query completa para histórico"""
    
    filters: Optional[HistoryFilter] = Field(None, description="Filtros aplicados")
    pagination: PaginationParams = Field(default_factory=PaginationParams, description="Paginação")
```

### HistoryResponse

Response da consulta de histórico.

```python
class HistoryPrediction(BaseModel):
    """Predição no histórico"""
    
    prediction_id: str = Field(..., description="ID da predição")
    timestamp: datetime = Field(..., description="Timestamp da predição")
    
    input_features: FlightFeatures = Field(..., description="Features de entrada")
    prediction_result: PredictionResult = Field(..., description="Resultado da predição")
    
    model_version: str = Field(..., description="Versão do modelo usado")
    processing_time_ms: float = Field(..., description="Tempo de processamento")

class HistoryStats(BaseModel):
    """Estatísticas do histórico"""
    
    total_predictions: int = Field(..., description="Total de predições")
    cancelled_predictions: int = Field(..., description="Predições de cancelamento")
    cancellation_rate: float = Field(..., description="Taxa de cancelamento")
    
    average_probability: float = Field(..., description="Probabilidade média")
    confidence_distribution: Dict[str, int] = Field(..., description="Distribuição de confiança")
    
    date_range: Dict[str, datetime] = Field(..., description="Range de datas")

class HistoryResponse(BaseModel):
    """Response da consulta de histórico"""
    
    predictions: List[HistoryPrediction] = Field(
        ...,
        description="Lista de predições"
    )
    
    pagination: Dict[str, Any] = Field(
        ..., 
        description="Informações de paginação"
    )
    
    filters_applied: Optional[Dict[str, Any]] = Field(
        None,
        description="Filtros aplicados na consulta"
    )
    
    statistics: HistoryStats = Field(
        ...,
        description="Estatísticas agregadas"
    )
    
    query_time_ms: float = Field(
        ...,
        description="Tempo de execução da query"
    )
```

## 🚨 Modelos de Erro (Error Models)

### ErrorDetail

Detalhes de erro estruturado.

```python
class ErrorDetail(BaseModel):
    """Detalhe específico de erro"""
    
    field: Optional[str] = Field(None, description="Campo que causou o erro")
    message: str = Field(..., description="Mensagem do erro")
    code: Optional[str] = Field(None, description="Código do erro")
    input_value: Optional[Any] = Field(None, description="Valor de entrada inválido")

class APIError(BaseModel):
    """Modelo padrão de erro da API"""
    
    error_type: str = Field(..., description="Tipo do erro")
    message: str = Field(..., description="Mensagem principal")
    
    details: Optional[List[ErrorDetail]] = Field(
        None,
        description="Detalhes específicos do erro"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp do erro"
    )
    
    request_id: str = Field(
        default_factory=lambda: uuid4().hex[:12],
        description="ID da requisição para tracking"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "error_type": "VALIDATION_ERROR",
                "message": "Erro de validação nos dados de entrada",
                "details": [
                    {
                        "field": "scheduled_departure",
                        "message": "Formato de data inválido", 
                        "code": "invalid_datetime",
                        "input_value": "invalid-date"
                    }
                ],
                "timestamp": "2024-12-21T10:15:00Z",
                "request_id": "abc123def456"
            }
        }
```

## 🔧 Utilitários de Validação

### Validators Customizados

```python
from pydantic import validator
import re
from datetime import datetime, timedelta

def validate_flight_number(flight_number: str) -> str:
    """Valida formato do número do voo"""
    pattern = r'^[A-Z]{2,3}\d{1,4}[A-Z]?$'
    if not re.match(pattern, flight_number):
        raise ValueError('Formato inválido. Use formato como: AA123, TAM1234')
    return flight_number.upper()

def validate_airport_code(code: str) -> str:
    """Valida código IATA do aeroporto"""
    if len(code) != 3 or not code.isalpha():
        raise ValueError('Código IATA deve ter 3 letras')
    return code.upper()

def validate_future_datetime(dt: datetime) -> datetime:
    """Valida que data é no futuro (para voos futuros)"""
    if dt <= datetime.now():
        raise ValueError('Data deve ser no futuro')
    return dt

def validate_reasonable_flight_duration(arrival: datetime, departure: datetime) -> datetime:
    """Valida duração razoável do voo (max 20 horas)"""
    duration = arrival - departure
    if duration > timedelta(hours=20):
        raise ValueError('Duração do voo muito longa (>20 horas)')
    if duration < timedelta(minutes=30):
        raise ValueError('Duração do voo muito curta (<30 min)')
    return arrival
```

### Base Models Reutilizáveis

```python
class TimestampedModel(BaseModel):
    """Base model com timestamp automático"""
    
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None
    
    class Config:
        # Permitir campos extras em subclasses
        extra = "allow"

class VersionedModel(BaseModel):
    """Base model com versionamento"""
    
    version: str = Field(default="1.0.0", regex=r'^\d+\.\d+\.\d+$')
    created_at: datetime = Field(default_factory=datetime.now)

class AuditableModel(BaseModel):
    """Base model com auditoria"""
    
    created_by: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_by: Optional[str] = None 
    updated_at: Optional[datetime] = None
```

## 📚 Uso Prático

### Exemplo de Validação

```python
from fastapi import HTTPException
from pydantic import ValidationError

try:
    # Criar modelo a partir de dados
    flight = FlightFeatures(**request_data)
    
except ValidationError as e:
    # Converter erro Pydantic para resposta HTTP
    errors = []
    for error in e.errors():
        errors.append({
            "field": ".".join(str(x) for x in error["loc"]),
            "message": error["msg"],
            "input_value": error.get("input")
        })
    
    raise HTTPException(
        status_code=422,
        detail={
            "error_type": "VALIDATION_ERROR",
            "message": "Dados de entrada inválidos",
            "details": errors
        }
    )
```

### Exemplo de Response

```python
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    # Processamento...
    
    return PredictionResponse(
        prediction=PredictionResult(
            cancelled=False,
            probability=0.23,
            confidence="high"
        ),
        model_info=ModelInfo(
            name="flight_model_v1",
            version="1.0.0", 
            algorithm="Decision Tree",
            accuracy=0.94
        ),
        processing_time_ms=15.6
    )
```

## 📚 Próximos Passos

- 🔗 [Endpoints da API](endpoints.md) - Como usar os modelos nos endpoints
- 💡 [Exemplos Práticos](examples.md) - Casos de uso completos
- 🧪 [Testes](../tests/running-tests.md) - Como testar validações
- 🏗️ [Arquitetura](../architecture/components.md) - Integração com outros componentes

## 📞 Suporte

- 📚 **Pydantic Docs**: [https://docs.pydantic.dev](https://docs.pydantic.dev)
- 🐛 **Issues**: [GitHub Issues](https://github.com/ulissesbomjardim/machine_learning_engineer/issues)
- 📧 **Email**: [ulisses.bomjardim@gmail.com](mailto:ulisses.bomjardim@gmail.com)