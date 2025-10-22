import logging
import pickle
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.services.database import InMemoryDatabase

router = APIRouter()
logger = logging.getLogger(__name__)

# Modelo para o payload de entrada
class FlightData(BaseModel):
    dep_delay: int = Field(
        ...,
        description='Atraso de partida em minutos (pode ser negativo se adiantado)',
    )
    sched_dep_time: int = Field(
        ..., description='Horário programado de partida (formato HHMM)'
    )
    dep_time: int = Field(
        ..., description='Horário real de partida (formato HHMM)'
    )
    sched_arr_time: int = Field(
        ..., description='Horário programado de chegada (formato HHMM)'
    )
    arr_time: int = Field(
        ..., description='Horário real de chegada (formato HHMM)'
    )
    hour: int = Field(..., description='Hora da partida (0-23)', ge=0, le=23)


class FlightPredictionResponse(BaseModel):
    predicted_arr_delay: float = Field(
        ..., description='Atraso de chegada previsto em minutos'
    )
    input_data: FlightData
    timestamp: str
    model_version: str = 'decision_tree_v1'


class BatchFlightRequest(BaseModel):
    flights: List[FlightData]


class BatchFlightResponse(BaseModel):
    predictions: List[FlightPredictionResponse]
    total_flights: int


# Variável global para armazenar o modelo carregado
loaded_model = None


def prepare_features(flight_data: FlightData) -> List:
    """
    Prepara as features no formato esperado pelo modelo
    Features: [dep_delay, sched_dep_time, dep_time, sched_arr_time, arr_time, hour]
    """
    return [
        flight_data.dep_delay,
        flight_data.sched_dep_time,
        flight_data.dep_time,
        flight_data.sched_arr_time,
        flight_data.arr_time,
        flight_data.hour,
    ]


@router.post('/predict/', response_model=FlightPredictionResponse)
async def predict_flight_delay(flight: FlightData):
    """
    Endpoint onde deverá receber um payload com as informações do voo e retornar a previsão do atraso no destino
    """
    global loaded_model

    if loaded_model is None:
        raise HTTPException(
            status_code=400,
            detail='Modelo não carregado. Use o endpoint /model/load/ primeiro.',
        )

    try:
        # Preparar features para o modelo
        features = prepare_features(flight)

        # Realizar predição
        prediction = loaded_model.predict([features])[0]

        # Preparar dados para salvar no histórico
        timestamp = datetime.now().isoformat()
        prediction_data = {
            'input_data': flight.dict(),
            'predicted_arr_delay': float(prediction),
            'timestamp': timestamp,
            'model_version': 'decision_tree_v1',
        }

        # Salvar no histórico usando InMemoryDatabase
        db = InMemoryDatabase()
        predictions_collection = db.get_collection('predictions')
        predictions_collection.insert_one(prediction_data)

        logger.info(f'Predição realizada: {prediction} minutos de atraso')

        return FlightPredictionResponse(
            predicted_arr_delay=float(prediction),
            input_data=flight,
            timestamp=timestamp,
        )

    except Exception as e:
        logger.error(f'Erro na predição: {str(e)}')
        raise HTTPException(
            status_code=500, detail=f'Erro na predição: {str(e)}'
        )


@router.post('/predict/batch', response_model=BatchFlightResponse)
async def predict_batch_flights(batch_request: BatchFlightRequest):
    """
    Endpoint para predição em lote de múltiplos voos
    """
    global loaded_model

    if loaded_model is None:
        raise HTTPException(
            status_code=400,
            detail='Modelo não carregado. Use o endpoint /model/load/ primeiro.',
        )

    if len(batch_request.flights) > 100:
        raise HTTPException(
            status_code=400, detail='Máximo de 100 voos por requisição'
        )

    try:
        predictions = []
        db = InMemoryDatabase()
        predictions_collection = db.get_collection('predictions')

        for flight in batch_request.flights:
            # Preparar features para o modelo
            features = prepare_features(flight)

            # Realizar predição
            prediction = loaded_model.predict([features])[0]

            timestamp = datetime.now().isoformat()

            # Preparar dados para salvar no histórico
            prediction_data = {
                'input_data': flight.dict(),
                'predicted_arr_delay': float(prediction),
                'timestamp': timestamp,
                'model_version': 'decision_tree_v1',
            }

            # Salvar no histórico
            predictions_collection.insert_one(prediction_data)

            # Adicionar à resposta
            predictions.append(
                FlightPredictionResponse(
                    predicted_arr_delay=float(prediction),
                    input_data=flight,
                    timestamp=timestamp,
                )
            )

        logger.info(
            f'Predições em lote realizadas: {len(predictions)} voos processados'
        )

        return BatchFlightResponse(
            predictions=predictions, total_flights=len(predictions)
        )

    except Exception as e:
        logger.error(f'Erro na predição em lote: {str(e)}')
        raise HTTPException(
            status_code=500, detail=f'Erro na predição em lote: {str(e)}'
        )
