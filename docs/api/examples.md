# 📋 Exemplos Práticos da API

Exemplos práticos de uso da API de predição de atrasos de voos, incluindo casos de uso reais, código de exemplo e integração com diferentes linguagens de programação.

## 🎯 Visão Geral

Esta seção fornece exemplos concretos de como utilizar a API de predição de atrasos, desde chamadas simples até integrações complexas em sistemas de produção.

## 🚀 Quick Start

### 1. 📡 Primeiro Request

```bash
# Teste básico da API
curl -X GET "http://localhost:8000/health" \
  -H "Content-Type: application/json"

# Resposta esperada
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "version": "1.0.0"
}
```

### 2. 🎯 Predição Simples

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "airline": "TAM",
    "departure_airport": "SBGR",
    "arrival_airport": "SBRJ",
    "departure_time": "2024-01-15T14:30:00",
    "aircraft_type": "A320",
    "weather_conditions": "partly_cloudy",
    "temperature": 25.5,
    "wind_speed": 15,
    "visibility": 8.0
  }'
```

## 💻 Exemplos por Linguagem

### 🐍 Python

#### **Exemplo Básico**

```python
import requests
import json
from datetime import datetime

# Configuração da API
API_BASE_URL = "http://localhost:8000"
headers = {"Content-Type": "application/json"}

def predict_flight_delay(flight_data):
    """Prediz atraso de voo usando a API"""
    
    url = f"{API_BASE_URL}/predict"
    
    try:
        response = requests.post(url, json=flight_data, headers=headers)
        response.raise_for_status()
        
        result = response.json()
        return result
        
    except requests.exceptions.RequestException as e:
        print(f"Erro na requisição: {e}")
        return None

# Exemplo de uso
flight_data = {
    "airline": "GOL",
    "departure_airport": "SBSP",
    "arrival_airport": "SBGL",
    "departure_time": "2024-01-15T18:45:00",
    "aircraft_type": "B737",
    "weather_conditions": "rain", 
    "temperature": 22.0,
    "wind_speed": 25,
    "visibility": 3.0
}

result = predict_flight_delay(flight_data)

if result:
    print(f"Probabilidade de atraso: {result['delay_probability']:.2%}")
    print(f"Predição: {'ATRASO' if result['is_delayed'] else 'PONTUAL'}")
    print(f"Confiança: {result['confidence']:.2f}")
```

#### **Cliente Python Avançado**

```python
import asyncio
import aiohttp
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class FlightPredictionResult:
    is_delayed: bool
    delay_probability: float
    confidence: float
    predicted_delay_minutes: Optional[int]
    factors: Dict[str, float]
    timestamp: str

class FlightDelayAPIClient:
    """Cliente avançado para API de predição de atrasos"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def health_check(self) -> Dict:
        """Verifica saúde da API"""
        async with self.session.get(f"{self.base_url}/health") as response:
            return await response.json()
    
    async def predict_single(self, flight_data: Dict) -> FlightPredictionResult:
        """Predição para um voo"""
        async with self.session.post(
            f"{self.base_url}/predict", 
            json=flight_data
        ) as response:
            data = await response.json()
            
            return FlightPredictionResult(
                is_delayed=data['is_delayed'],
                delay_probability=data['delay_probability'],
                confidence=data['confidence'],
                predicted_delay_minutes=data.get('predicted_delay_minutes'),
                factors=data.get('factors', {}),
                timestamp=data['timestamp']
            )
    
    async def predict_batch(self, flights: List[Dict]) -> List[FlightPredictionResult]:
        """Predição em lote"""
        async with self.session.post(
            f"{self.base_url}/predict/batch",
            json={"flights": flights}
        ) as response:
            data = await response.json()
            
            return [
                FlightPredictionResult(**result) 
                for result in data['predictions']
            ]

# Exemplo de uso assíncrono
async def main():
    flights = [
        {
            "airline": "TAM",
            "departure_airport": "SBGR",
            "arrival_airport": "SBGL",
            "departure_time": "2024-01-15T08:30:00",
            "aircraft_type": "A320",
            "weather_conditions": "clear",
            "temperature": 28.0,
            "wind_speed": 10,
            "visibility": 10.0
        },
        {
            "airline": "GOL", 
            "departure_airport": "SBSP",
            "arrival_airport": "SBRJ",
            "departure_time": "2024-01-15T19:15:00",
            "aircraft_type": "B737",
            "weather_conditions": "thunderstorm",
            "temperature": 24.0,
            "wind_speed": 40,
            "visibility": 1.5
        }
    ]
    
    async with FlightDelayAPIClient() as client:
        # Verificar saúde
        health = await client.health_check()
        print(f"API Status: {health['status']}")
        
        # Predição em lote
        results = await client.predict_batch(flights)
        
        for i, result in enumerate(results):
            print(f"\nVoo {i+1}:")
            print(f"  Atraso previsto: {'SIM' if result.is_delayed else 'NÃO'}")
            print(f"  Probabilidade: {result.delay_probability:.2%}")
            print(f"  Confiança: {result.confidence:.2f}")

# Executar
# asyncio.run(main())
```

### 🟨 JavaScript/Node.js

#### **Exemplo Básico**

```javascript
const axios = require('axios');

const API_BASE_URL = 'http://localhost:8000';

class FlightDelayPredictor {
  constructor(baseUrl = API_BASE_URL) {
    this.baseUrl = baseUrl;
    this.client = axios.create({
      baseURL: baseUrl,
      headers: {
        'Content-Type': 'application/json'
      }
    });
  }

  async healthCheck() {
    try {
      const response = await this.client.get('/health');
      return response.data;
    } catch (error) {
      console.error('Health check failed:', error.message);
      throw error;
    }
  }

  async predictDelay(flightData) {
    try {
      const response = await this.client.post('/predict', flightData);
      return response.data;
    } catch (error) {
      console.error('Prediction failed:', error.message);
      throw error;
    }
  }

  async predictBatch(flights) {
    try {
      const response = await this.client.post('/predict/batch', { flights });
      return response.data;
    } catch (error) {
      console.error('Batch prediction failed:', error.message);
      throw error;
    }
  }
}

// Exemplo de uso
async function main() {
  const predictor = new FlightDelayPredictor();

  try {
    // Verificar saúde da API
    const health = await predictor.healthCheck();
    console.log('API Status:', health.status);

    // Dados do voo
    const flightData = {
      airline: 'AZUL',
      departure_airport: 'SBCF',
      arrival_airport: 'SBGR',
      departure_time: '2024-01-15T16:20:00',
      aircraft_type: 'A320',
      weather_conditions: 'cloudy',
      temperature: 20.5,
      wind_speed: 18,
      visibility: 6.0
    };

    // Fazer predição
    const result = await predictor.predictDelay(flightData);

    console.log('\n=== RESULTADO DA PREDIÇÃO ===');
    console.log(`Voo: ${flightData.departure_airport} → ${flightData.arrival_airport}`);
    console.log(`Companhia: ${flightData.airline}`);
    console.log(`Predição: ${result.is_delayed ? 'ATRASO' : 'PONTUAL'}`);
    console.log(`Probabilidade: ${(result.delay_probability * 100).toFixed(1)}%`);
    console.log(`Confiança: ${result.confidence.toFixed(2)}`);

  } catch (error) {
    console.error('Erro:', error.message);
  }
}

// Executar
// main();
```

#### **React Component**

```jsx
import React, { useState, useEffect } from 'react';
import axios from 'axios';

const FlightDelayPredictor = () => {
  const [flightData, setFlightData] = useState({
    airline: '',
    departure_airport: '',
    arrival_airport: '',
    departure_time: '',
    aircraft_type: '',
    weather_conditions: 'clear',
    temperature: 25,
    wind_speed: 10,
    visibility: 10
  });
  
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFlightData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      const response = await axios.post(
        'http://localhost:8000/predict',
        flightData
      );
      setPrediction(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'Erro na predição');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flight-predictor">
      <h2>Predição de Atrasos de Voo</h2>
      
      <form onSubmit={handleSubmit}>
        <div className="form-group">
          <label>Companhia Aérea:</label>
          <select 
            name="airline" 
            value={flightData.airline}
            onChange={handleInputChange}
            required
          >
            <option value="">Selecione...</option>
            <option value="TAM">TAM</option>
            <option value="GOL">GOL</option>
            <option value="AZUL">Azul</option>
            <option value="LATAM">LATAM</option>
          </select>
        </div>

        <div className="form-group">
          <label>Aeroporto de Origem:</label>
          <select 
            name="departure_airport"
            value={flightData.departure_airport}
            onChange={handleInputChange}
            required
          >
            <option value="">Selecione...</option>
            <option value="SBGR">São Paulo (Guarulhos)</option>
            <option value="SBSP">São Paulo (Congonhas)</option>
            <option value="SBRJ">Rio de Janeiro (Santos Dumont)</option>
            <option value="SBGL">Rio de Janeiro (Galeão)</option>
          </select>
        </div>

        <div className="form-group">
          <label>Data/Hora da Partida:</label>
          <input
            type="datetime-local"
            name="departure_time"
            value={flightData.departure_time}
            onChange={handleInputChange}
            required
          />
        </div>

        <button type="submit" disabled={loading}>
          {loading ? 'Predizendo...' : 'Prever Atraso'}
        </button>
      </form>

      {error && (
        <div className="error">
          Erro: {error}
        </div>
      )}

      {prediction && (
        <div className="prediction-result">
          <h3>Resultado da Predição</h3>
          <div className={`prediction ${prediction.is_delayed ? 'delayed' : 'ontime'}`}>
            <span className="status">
              {prediction.is_delayed ? '⚠️ ATRASO' : '✅ PONTUAL'}
            </span>
            <span className="probability">
              Probabilidade: {(prediction.delay_probability * 100).toFixed(1)}%
            </span>
            <span className="confidence">
              Confiança: {prediction.confidence.toFixed(2)}
            </span>
          </div>
        </div>
      )}
    </div>
  );
};

export default FlightDelayPredictor;
```

### ☕ Java

```java
import com.fasterxml.jackson.databind.ObjectMapper;
import org.apache.http.HttpEntity;
import org.apache.http.client.methods.CloseableHttpResponse;
import org.apache.http.client.methods.HttpGet;
import org.apache.http.client.methods.HttpPost;
import org.apache.http.entity.StringEntity;
import org.apache.http.impl.client.CloseableHttpClient;
import org.apache.http.impl.client.HttpClients;
import org.apache.http.util.EntityUtils;

import java.io.IOException;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.HashMap;
import java.util.Map;

public class FlightDelayAPIClient {
    
    private final String baseUrl;
    private final ObjectMapper objectMapper;
    private final CloseableHttpClient httpClient;

    public FlightDelayAPIClient(String baseUrl) {
        this.baseUrl = baseUrl;
        this.objectMapper = new ObjectMapper();
        this.httpClient = HttpClients.createDefault();
    }

    public class FlightData {
        public String airline;
        public String departureAirport;
        public String arrivalAirport;
        public String departureTime;
        public String aircraftType;
        public String weatherConditions;
        public double temperature;
        public int windSpeed;
        public double visibility;

        // Construtor e getters/setters
        public FlightData(String airline, String departureAirport, String arrivalAirport,
                         LocalDateTime departureTime, String aircraftType, 
                         String weatherConditions, double temperature, 
                         int windSpeed, double visibility) {
            this.airline = airline;
            this.departureAirport = departureAirport;
            this.arrivalAirport = arrivalAirport;
            this.departureTime = departureTime.format(DateTimeFormatter.ISO_LOCAL_DATE_TIME);
            this.aircraftType = aircraftType;
            this.weatherConditions = weatherConditions;
            this.temperature = temperature;
            this.windSpeed = windSpeed;
            this.visibility = visibility;
        }
    }

    public class PredictionResult {
        public boolean isDelayed;
        public double delayProbability;
        public double confidence;
        public String timestamp;
    }

    public Map<String, Object> healthCheck() throws IOException {
        HttpGet request = new HttpGet(baseUrl + "/health");
        
        try (CloseableHttpResponse response = httpClient.execute(request)) {
            HttpEntity entity = response.getEntity();
            String jsonResponse = EntityUtils.toString(entity);
            
            return objectMapper.readValue(jsonResponse, Map.class);
        }
    }

    public PredictionResult predictDelay(FlightData flightData) throws IOException {
        HttpPost request = new HttpPost(baseUrl + "/predict");
        request.setHeader("Content-Type", "application/json");
        
        // Converter para JSON
        String jsonData = objectMapper.writeValueAsString(flightData);
        request.setEntity(new StringEntity(jsonData));
        
        try (CloseableHttpResponse response = httpClient.execute(request)) {
            HttpEntity entity = response.getEntity();
            String jsonResponse = EntityUtils.toString(entity);
            
            return objectMapper.readValue(jsonResponse, PredictionResult.class);
        }
    }

    public static void main(String[] args) {
        FlightDelayAPIClient client = new FlightDelayAPIClient("http://localhost:8000");
        
        try {
            // Verificar saúde
            Map<String, Object> health = client.healthCheck();
            System.out.println("API Status: " + health.get("status"));
            
            // Criar dados do voo
            FlightData flight = new FlightData(
                "TAM",
                "SBGR",
                "SBRJ", 
                LocalDateTime.of(2024, 1, 15, 14, 30),
                "A320",
                "partly_cloudy",
                25.5,
                15,
                8.0
            );
            
            // Fazer predição
            PredictionResult result = client.predictDelay(flight);
            
            System.out.println("\n=== RESULTADO ===");
            System.out.println("Atraso previsto: " + (result.isDelayed ? "SIM" : "NÃO"));
            System.out.println("Probabilidade: " + String.format("%.1f%%", result.delayProbability * 100));
            System.out.println("Confiança: " + String.format("%.2f", result.confidence));
            
        } catch (IOException e) {
            System.err.println("Erro: " + e.getMessage());
        }
    }
}
```

## 🔄 Casos de Uso Avançados

### 1. 🎯 Monitoramento em Tempo Real

```python
import asyncio
import websockets
import json
from datetime import datetime, timedelta

class RealTimeFlightMonitor:
    """Monitor de voos em tempo real"""
    
    def __init__(self, api_client):
        self.api_client = api_client
        self.monitored_flights = {}
        
    async def monitor_flight(self, flight_id, flight_data, check_interval=300):
        """Monitora um voo específico"""
        
        while datetime.now() < datetime.fromisoformat(flight_data['departure_time']):
            try:
                # Atualizar dados meteorológicos (simulado)
                updated_data = await self.update_weather_data(flight_data)
                
                # Fazer nova predição
                prediction = await self.api_client.predict_single(updated_data)
                
                # Verificar mudanças significativas
                await self.check_prediction_changes(flight_id, prediction)
                
                # Aguardar próxima verificação
                await asyncio.sleep(check_interval)
                
            except Exception as e:
                print(f"Erro ao monitorar voo {flight_id}: {e}")
                await asyncio.sleep(60)  # Retry em 1 minuto
    
    async def update_weather_data(self, flight_data):
        """Atualiza dados meteorológicos (integração com API de clima)"""
        # Implementar integração com serviço de clima
        return flight_data
    
    async def check_prediction_changes(self, flight_id, new_prediction):
        """Verifica mudanças na predição e envia alertas"""
        
        previous = self.monitored_flights.get(flight_id)
        
        if previous:
            prob_change = abs(new_prediction.delay_probability - previous.delay_probability)
            
            if prob_change > 0.2:  # Mudança > 20%
                await self.send_alert(flight_id, new_prediction, prob_change)
        
        self.monitored_flights[flight_id] = new_prediction
    
    async def send_alert(self, flight_id, prediction, change):
        """Envia alerta de mudança na predição"""
        alert = {
            'flight_id': flight_id,
            'new_probability': prediction.delay_probability,
            'change': change,
            'timestamp': datetime.now().isoformat(),
            'message': f'Mudança significativa na predição de atraso: {change:.1%}'
        }
        
        print(f"🚨 ALERTA: {alert['message']}")
        # Implementar envio por webhook, email, etc.
```

### 2. 📊 Análise de Tendências

```python
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

class FlightDelayAnalyzer:
    """Analisador de tendências de atrasos"""
    
    def __init__(self, api_client):
        self.api_client = api_client
        self.predictions_history = []
    
    async def analyze_route_trends(self, route, days_back=30):
        """Analisa tendências de uma rota específica"""
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        # Simular dados históricos da rota
        historical_data = self.generate_historical_data(route, start_date, end_date)
        
        # Fazer predições para todos os dados
        predictions = []
        for data in historical_data:
            pred = await self.api_client.predict_single(data)
            predictions.append({
                'date': data['departure_time'],
                'probability': pred.delay_probability,
                'weather': data['weather_conditions'],
                'temperature': data['temperature']
            })
        
        # Análise estatística
        df = pd.DataFrame(predictions)
        df['date'] = pd.to_datetime(df['date'])
        
        return {
            'average_delay_prob': df['probability'].mean(),
            'trend': self.calculate_trend(df['probability']),
            'weather_impact': df.groupby('weather')['probability'].mean().to_dict(),
            'daily_averages': df.groupby(df['date'].dt.date)['probability'].mean().to_dict()
        }
    
    def generate_historical_data(self, route, start_date, end_date):
        """Gera dados históricos simulados"""
        # Implementar carregamento de dados reais
        return []
    
    def calculate_trend(self, probabilities):
        """Calcula tendência da série temporal"""
        # Regressão linear simples
        x = range(len(probabilities))
        slope = np.polyfit(x, probabilities, 1)[0]
        
        if slope > 0.01:
            return "increasing"
        elif slope < -0.01:
            return "decreasing"
        else:
            return "stable"
```

## 🔧 Integração com Sistemas

### 1. 🎫 Sistema de Reservas

```python
class BookingSystemIntegration:
    """Integração com sistema de reservas"""
    
    def __init__(self, api_client, booking_db):
        self.api_client = api_client
        self.booking_db = booking_db
    
    async def predict_for_booking(self, booking_id):
        """Prediz atraso para uma reserva específica"""
        
        # Buscar dados da reserva
        booking = await self.booking_db.get_booking(booking_id)
        
        if not booking:
            raise ValueError(f"Reserva {booking_id} não encontrada")
        
        # Converter para formato da API
        flight_data = {
            'airline': booking['airline'],
            'departure_airport': booking['departure_airport'],
            'arrival_airport': booking['arrival_airport'],
            'departure_time': booking['departure_time'],
            'aircraft_type': booking['aircraft_type'],
            # Buscar dados meteorológicos atuais
            **await self.get_current_weather(booking['departure_airport'])
        }
        
        # Fazer predição
        prediction = await self.api_client.predict_single(flight_data)
        
        # Salvar resultado no banco
        await self.booking_db.save_prediction(booking_id, prediction)
        
        return prediction
    
    async def get_current_weather(self, airport_code):
        """Busca condições meteorológicas atuais"""
        # Integrar com API meteorológica
        return {
            'weather_conditions': 'partly_cloudy',
            'temperature': 25.0,
            'wind_speed': 15,
            'visibility': 8.0
        }
```

### 2. 📱 App Mobile

```kotlin
// Kotlin/Android Example
import kotlinx.coroutines.*
import retrofit2.*
import retrofit2.converter.gson.GsonConverterFactory
import retrofit2.http.*

data class FlightData(
    val airline: String,
    val departure_airport: String,
    val arrival_airport: String,
    val departure_time: String,
    val aircraft_type: String,
    val weather_conditions: String,
    val temperature: Double,
    val wind_speed: Int,
    val visibility: Double
)

data class PredictionResponse(
    val is_delayed: Boolean,
    val delay_probability: Double,
    val confidence: Double,
    val timestamp: String
)

interface FlightDelayAPI {
    @POST("predict")
    suspend fun predictDelay(@Body flightData: FlightData): PredictionResponse
    
    @GET("health")
    suspend fun healthCheck(): Map<String, Any>
}

class FlightDelayRepository {
    private val api: FlightDelayAPI
    
    init {
        val retrofit = Retrofit.Builder()
            .baseUrl("http://10.0.2.2:8000/") // Android emulator
            .addConverterFactory(GsonConverterFactory.create())
            .build()
        
        api = retrofit.create(FlightDelayAPI::class.java)
    }
    
    suspend fun predictFlightDelay(flightData: FlightData): Result<PredictionResponse> {
        return try {
            val prediction = api.predictDelay(flightData)
            Result.success(prediction)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }
}

// Usage in ViewModel
class FlightPredictionViewModel : ViewModel() {
    private val repository = FlightDelayRepository()
    
    fun predictDelay(flightData: FlightData) {
        viewModelScope.launch {
            val result = repository.predictFlightDelay(flightData)
            
            result.onSuccess { prediction ->
                // Update UI with prediction
                _predictionResult.value = prediction
            }.onFailure { error ->
                // Handle error
                _errorMessage.value = error.message
            }
        }
    }
}
```

## 🧪 Testes de Integração

```python
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock

class TestAPIIntegration:
    """Testes de integração da API"""
    
    @pytest.fixture
    async def api_client(self):
        """Cliente de teste"""
        return FlightDelayAPIClient("http://localhost:8000")
    
    @pytest.mark.asyncio
    async def test_health_endpoint(self, api_client):
        """Testa endpoint de saúde"""
        async with api_client as client:
            health = await client.health_check()
            
            assert health['status'] == 'healthy'
            assert 'timestamp' in health
            assert 'version' in health
    
    @pytest.mark.asyncio
    async def test_single_prediction(self, api_client):
        """Testa predição individual"""
        flight_data = {
            "airline": "TAM",
            "departure_airport": "SBGR",
            "arrival_airport": "SBRJ",
            "departure_time": "2024-01-15T14:30:00",
            "aircraft_type": "A320",
            "weather_conditions": "clear",
            "temperature": 25.0,
            "wind_speed": 10,
            "visibility": 10.0
        }
        
        async with api_client as client:
            result = await client.predict_single(flight_data)
            
            assert isinstance(result.is_delayed, bool)
            assert 0 <= result.delay_probability <= 1
            assert 0 <= result.confidence <= 1
            assert result.timestamp is not None
    
    @pytest.mark.asyncio
    async def test_batch_prediction(self, api_client):
        """Testa predição em lote"""
        flights = [
            {
                "airline": "GOL",
                "departure_airport": "SBSP",
                "arrival_airport": "SBGL",
                "departure_time": "2024-01-15T08:00:00",
                "aircraft_type": "B737",
                "weather_conditions": "cloudy",
                "temperature": 20.0,
                "wind_speed": 15,
                "visibility": 5.0
            },
            {
                "airline": "AZUL", 
                "departure_airport": "SBCF",
                "arrival_airport": "SBRJ",
                "departure_time": "2024-01-15T16:30:00", 
                "aircraft_type": "E190",
                "weather_conditions": "rain",
                "temperature": 18.0,
                "wind_speed": 25,
                "visibility": 2.0
            }
        ]
        
        async with api_client as client:
            results = await client.predict_batch(flights)
            
            assert len(results) == len(flights)
            
            for result in results:
                assert isinstance(result.is_delayed, bool)
                assert 0 <= result.delay_probability <= 1
```

## 📋 Códigos de Erro

| **Código** | **Descrição** | **Solução** |
|------------|---------------|-------------|
| 400 | Dados inválidos | Verificar formato dos dados de entrada |
| 401 | Não autorizado | Verificar token de autenticação |
| 404 | Endpoint não encontrado | Verificar URL da API |
| 422 | Erro de validação | Verificar tipos e valores dos campos |
| 500 | Erro interno | Verificar logs do servidor |
| 503 | Serviço indisponível | Tentar novamente mais tarde |

## 🔗 Próximos Passos

1. **[📋 Endpoints](endpoints.md)** - Documentação completa da API
2. **[🏗️ Modelos](models.md)** - Schemas de dados
3. **[🧪 Testes](../tests/integration.md)** - Testes de integração

---

## 📞 Suporte

- 🐛 **[Issues](https://github.com/ulissesbomjardim/machine-learning-engineer/issues)** - Reportar problemas
- 📖 **[Documentação](../index.md)** - Guia completo
- 🔧 **[Troubleshooting](../dev/troubleshooting.md)** - Solução de problemas