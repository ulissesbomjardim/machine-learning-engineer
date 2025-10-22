# 🔗 API Endpoints

Documentação completa de todos os endpoints disponíveis na API do Machine Learning Engineer Challenge.

## 📋 Visão Geral da API

A API foi desenvolvida com **FastAPI** seguindo os princípios REST e fornece endpoints para predição de cancelamento de voos, gerenciamento de modelos e consulta de histórico.

### 🌐 Base URL

```
http://localhost:8000
```

### 📊 Características

- ✅ **REST API** padrão
- 📚 **Documentação automática** (Swagger/OpenAPI)
- 🔒 **Validação automática** com Pydantic
- ⚡ **Responses assíncronos** 
- 🚨 **Error handling** padronizado
- 📈 **Health monitoring**

## 🏠 Endpoints Gerais

### GET / - Informações da API

Retorna informações básicas sobre a API.

**Request:**
```bash
curl -X GET "http://localhost:8000/"
```

**Response:**
```json
{
  "message": "Flight Delay Prediction API",
  "version": "1.0.0",
  "status": "running",
  "endpoints": {
    "health": "/health",
    "docs": "/docs",
    "predict": "/model/predict",
    "load_model": "/model/load/default",
    "history": "/model/history/"
  }
}
```

### GET /health - Health Check

Verifica o status de saúde da API e componentes.

**Request:**
```bash
curl -X GET "http://localhost:8000/health"
```

**Response Saudável:**
```json
{
  "status": "healthy",
  "timestamp": "2024-12-21T10:00:00Z",
  "version": "1.0.0",
  "components": {
    "api": "healthy",
    "database": "healthy",
    "model": "loaded"
  },
  "uptime": "2h 30m 15s"
}
```

**Response com Problemas:**
```json
{
  "status": "unhealthy",
  "timestamp": "2024-12-21T10:00:00Z",
  "version": "1.0.0",
  "components": {
    "api": "healthy",
    "database": "disconnected",
    "model": "not_loaded"
  },
  "errors": [
    "Database connection failed",
    "Model not loaded"
  ]
}
```

## 🤖 Endpoints de Machine Learning

### POST /model/predict - Predição de Cancelamento

Realiza predição de cancelamento de voo baseada nos dados fornecidos.

**Request:**
```bash
curl -X POST "http://localhost:8000/model/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "features": {
         "airline": "American Airlines",
         "flight_number": "AA123",
         "departure_airport": "JFK",
         "arrival_airport": "LAX",
         "scheduled_departure": "2024-01-15T10:00:00",
         "scheduled_arrival": "2024-01-15T14:00:00",
         "aircraft_type": "Boeing 737",
         "weather_condition": "Clear"
       }
     }'
```

**Response:**
```json
{
  "prediction": {
    "cancelled": false,
    "probability": 0.15,
    "confidence": "high"
  },
  "features_processed": {
    "airline_encoded": 1,
    "departure_hour": 10,
    "flight_duration_minutes": 360,
    "route_popularity": 0.85
  },
  "model_info": {
    "name": "decision_tree_v1",
    "version": "1.0.0",
    "accuracy": 0.94
  },
  "prediction_id": "pred_20241221_100000_abc123",
  "timestamp": "2024-12-21T10:00:00Z"
}
```

### POST /model/predict (Batch)

Predição em lote para múltiplos voos.

**Request:**
```bash
curl -X POST "http://localhost:8000/model/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "batch": [
         {
           "features": {
             "airline": "American Airlines",
             "flight_number": "AA123",
             "departure_airport": "JFK",
             "arrival_airport": "LAX"
           }
         },
         {
           "features": {
             "airline": "Delta",
             "flight_number": "DL456",
             "departure_airport": "ATL",
             "arrival_airport": "SEA"
           }
         }
       ]
     }'
```

**Response:**
```json
{
  "batch_predictions": [
    {
      "prediction": {
        "cancelled": false,
        "probability": 0.15
      },
      "prediction_id": "pred_20241221_100001_abc123"
    },
    {
      "prediction": {
        "cancelled": true,
        "probability": 0.78
      },
      "prediction_id": "pred_20241221_100002_def456"
    }
  ],
  "batch_summary": {
    "total_predictions": 2,
    "cancelled_count": 1,
    "average_probability": 0.465
  },
  "batch_id": "batch_20241221_100000_xyz789",
  "timestamp": "2024-12-21T10:00:00Z"
}
```

## 📥 Endpoints de Gerenciamento de Modelos

### GET /model/load/default - Carregar Modelo Padrão

Carrega o modelo padrão pré-treinado.

**Request:**
```bash
curl -X GET "http://localhost:8000/model/load/default"
```

**Response:**
```json
{
  "status": "success",
  "message": "Default model loaded successfully",
  "model_info": {
    "name": "decision_tree_v1",
    "version": "1.0.0",
    "file_path": "./model/modelo_arvore_decisao.pkl",
    "file_size_mb": 24.5,
    "accuracy": 0.94,
    "features": [
      "airline",
      "departure_airport",
      "arrival_airport",
      "scheduled_departure",
      "aircraft_type"
    ]
  },
  "loaded_at": "2024-12-21T10:00:00Z"
}
```

### POST /model/load/ - Upload de Modelo

Upload de um novo modelo via arquivo.

**Request:**
```bash
curl -X POST "http://localhost:8000/model/load/" \
     -F "model_file=@model.pkl" \
     -F "model_name=custom_model_v2" \
     -F "version=2.0.0"
```

**Response:**
```json
{
  "status": "success",
  "message": "Model uploaded and loaded successfully",
  "model_info": {
    "name": "custom_model_v2",
    "version": "2.0.0",
    "file_size_mb": 18.2,
    "upload_id": "upload_20241221_100000_xyz123"
  },
  "validation": {
    "format_valid": true,
    "features_compatible": true,
    "test_prediction_successful": true
  },
  "loaded_at": "2024-12-21T10:00:00Z"
}
```

## 📊 Endpoints de Histórico

### GET /model/history/ - Consultar Histórico

Consulta histórico de predições com filtros e paginação.

**Parâmetros de Query:**

| **Parâmetro** | **Tipo** | **Descrição** | **Padrão** |
|---------------|----------|---------------|------------|
| `limit` | int | Número máximo de registros | 10 |
| `offset` | int | Número de registros para pular | 0 |
| `start_date` | datetime | Data inicial (ISO format) | - |
| `end_date` | datetime | Data final (ISO format) | - |
| `cancelled_only` | bool | Filtrar apenas cancelados | false |
| `airline` | str | Filtrar por companhia aérea | - |

**Request:**
```bash
curl -X GET "http://localhost:8000/model/history/?limit=5&cancelled_only=true"
```

**Response:**
```json
{
  "predictions": [
    {
      "prediction_id": "pred_20241221_095900_abc123",
      "timestamp": "2024-12-21T09:59:00Z",
      "input_features": {
        "airline": "American Airlines",
        "flight_number": "AA123",
        "departure_airport": "JFK",
        "arrival_airport": "LAX"
      },
      "prediction_result": {
        "cancelled": true,
        "probability": 0.85,
        "confidence": "high"
      },
      "model_used": "decision_tree_v1"
    }
  ],
  "pagination": {
    "limit": 5,
    "offset": 0,
    "total_records": 150,
    "has_next": true,
    "has_previous": false
  },
  "filters_applied": {
    "cancelled_only": true
  },
  "summary_stats": {
    "total_predictions": 150,
    "cancelled_predictions": 45,
    "cancellation_rate": 0.30
  }
}
```

### GET /model/history/stats - Estatísticas do Histórico

Estatísticas agregadas do histórico de predições.

**Request:**
```bash
curl -X GET "http://localhost:8000/model/history/stats"
```

**Response:**
```json
{
  "overall_stats": {
    "total_predictions": 1500,
    "total_cancelled": 450,
    "cancellation_rate": 0.30,
    "average_confidence": 0.78
  },
  "time_series": {
    "predictions_by_day": {
      "2024-12-20": 150,
      "2024-12-21": 125
    },
    "cancellation_rate_by_day": {
      "2024-12-20": 0.28,
      "2024-12-21": 0.32
    }
  },
  "breakdown": {
    "by_airline": {
      "American Airlines": {"total": 400, "cancelled": 120},
      "Delta": {"total": 350, "cancelled": 95}
    },
    "by_hour": {
      "06:00-09:00": {"total": 300, "cancelled": 60},
      "09:00-12:00": {"total": 400, "cancelled": 100}
    }
  }
}
```

## 🚨 Error Responses

### Códigos de Status HTTP

| **Código** | **Descrição** | **Quando Ocorre** |
|------------|---------------|-------------------|
| 200 | Success | Request executado com sucesso |
| 400 | Bad Request | Dados de entrada inválidos |
| 404 | Not Found | Recurso não encontrado |
| 422 | Validation Error | Falha na validação Pydantic |
| 500 | Internal Server Error | Erro interno do servidor |

### Formato de Erro Padronizado

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Input validation failed",
    "details": [
      {
        "field": "scheduled_departure",
        "message": "Invalid datetime format",
        "input_value": "invalid-date"
      }
    ],
    "timestamp": "2024-12-21T10:00:00Z",
    "request_id": "req_20241221_100000_xyz123"
  }
}
```

### Exemplos de Erros Comuns

**Dados inválidos (400):**
```json
{
  "error": {
    "code": "INVALID_INPUT",
    "message": "Required field missing",
    "details": "Field 'airline' is required"
  }
}
```

**Modelo não carregado (500):**
```json
{
  "error": {
    "code": "MODEL_NOT_LOADED",
    "message": "No model is currently loaded",
    "details": "Load a model using /model/load/default or /model/load/"
  }
}
```

## 🔒 Schemas de Dados

### FlightPredictionRequest

```json
{
  "features": {
    "airline": "string",
    "flight_number": "string",
    "departure_airport": "string (IATA code)",
    "arrival_airport": "string (IATA code)",
    "scheduled_departure": "datetime (ISO format)",
    "scheduled_arrival": "datetime (ISO format)",
    "aircraft_type": "string (optional)",
    "weather_condition": "string (optional)"
  }
}
```

### PredictionResponse

```json
{
  "prediction": {
    "cancelled": "boolean",
    "probability": "float (0-1)",
    "confidence": "string (low/medium/high)"
  },
  "prediction_id": "string",
  "timestamp": "datetime (ISO format)",
  "model_info": {
    "name": "string",
    "version": "string",
    "accuracy": "float (0-1)"
  }
}
```

## 🧪 Testando a API

### 🔧 Ferramentas Recomendadas

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`
- **Postman**: Importar OpenAPI spec
- **curl**: Comandos via terminal
- **httpx**: Cliente Python para testes

### 🚀 Scripts de Teste

**Teste básico de funcionamento:**
```bash
#!/bin/bash
# test_api_basic.sh

echo "Testing API endpoints..."

# Health check
echo "1. Health check:"
curl -s http://localhost:8000/health | jq '.'

# Load default model
echo "2. Loading default model:"
curl -s http://localhost:8000/model/load/default | jq '.'

# Make prediction
echo "3. Making prediction:"
curl -s -X POST http://localhost:8000/model/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "airline": "American Airlines",
      "flight_number": "AA123",
      "departure_airport": "JFK",
      "arrival_airport": "LAX",
      "scheduled_departure": "2024-01-15T10:00:00",
      "scheduled_arrival": "2024-01-15T14:00:00"
    }
  }' | jq '.'

echo "API test completed!"
```

## 📚 Próximos Passos

- 📋 [Modelos de Dados](models.md) - Schemas Pydantic detalhados
- 💡 [Exemplos Práticos](examples.md) - Casos de uso completos
- 🧪 [Testes da API](../tests/integration.md) - Testes automatizados
- 🏗️ [Arquitetura](../architecture/overview.md) - Design da API

## 📞 Suporte

- 📚 **Swagger UI**: `http://localhost:8000/docs`
- 🐛 **Issues**: [GitHub Issues](https://github.com/ulissesbomjardim/machine_learning_engineer/issues)
- 📧 **Email**: [ulisses.bomjardim@gmail.com](mailto:ulisses.bomjardim@gmail.com)