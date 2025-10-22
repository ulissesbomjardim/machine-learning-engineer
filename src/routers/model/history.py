import logging
from datetime import datetime
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from src.services.database import InMemoryDatabase

router = APIRouter()
logger = logging.getLogger(__name__)


class PredictionHistory(BaseModel):
    input_data: Dict[str, Any]
    predicted_arr_delay: float
    timestamp: str
    model_version: str


class HistoryResponse(BaseModel):
    total_predictions: int
    predictions: List[PredictionHistory]
    page: int
    page_size: int


class HistoryStatsResponse(BaseModel):
    total_predictions: int
    avg_predicted_delay: float
    min_predicted_delay: float
    max_predicted_delay: float
    predictions_today: int


@router.get('/history/', response_model=HistoryResponse)
async def get_prediction_history(
    page: int = Query(default=1, description='Número da página', ge=1),
    page_size: int = Query(
        default=50, description='Itens por página', ge=1, le=100
    ),
):
    """
    Endpoint onde deverá exibir o histórico de predições realizadas (o payload de entrada + as saídas preditas)
    """
    try:
        db = InMemoryDatabase()
        predictions_collection = db.get_collection('predictions')

        # Calcular skip baseado na paginação
        skip = (page - 1) * page_size

        # Contar total de predições
        total_count = predictions_collection.count_documents({})

        # Buscar predições com paginação (ordenadas por timestamp decrescente)
        cursor = (
            predictions_collection.find({}, {'_id': 0})
            .sort('timestamp', -1)
            .skip(skip)
            .limit(page_size)
        )

        predictions = []
        for doc in cursor:
            predictions.append(
                PredictionHistory(
                    input_data=doc['input_data'],
                    predicted_arr_delay=doc['predicted_arr_delay'],
                    timestamp=doc['timestamp'],
                    model_version=doc.get('model_version', 'unknown'),
                )
            )

        logger.info(
            f'Retornando {len(predictions)} predições do histórico (página {page})'
        )

        return HistoryResponse(
            total_predictions=total_count,
            predictions=predictions,
            page=page,
            page_size=page_size,
        )

    except Exception as e:
        logger.error(f'Erro ao buscar histórico: {str(e)}')
        raise HTTPException(
            status_code=500, detail=f'Erro ao buscar histórico: {str(e)}'
        )


@router.get('/history/stats', response_model=HistoryStatsResponse)
async def get_prediction_stats():
    """
    Endpoint para obter estatísticas das predições
    """
    try:
        db = InMemoryDatabase()
        predictions_collection = db.get_collection('predictions')

        # Pipeline de agregação para estatísticas gerais
        pipeline = [
            {
                '$group': {
                    '_id': None,
                    'total_predictions': {'$sum': 1},
                    'avg_predicted_delay': {'$avg': '$predicted_arr_delay'},
                    'min_predicted_delay': {'$min': '$predicted_arr_delay'},
                    'max_predicted_delay': {'$max': '$predicted_arr_delay'},
                }
            }
        ]

        result = list(predictions_collection.aggregate(pipeline))

        # Contar predições de hoje
        today = datetime.now().date().isoformat()
        predictions_today = predictions_collection.count_documents(
            {'timestamp': {'$regex': f'^{today}'}}
        )

        if not result:
            return HistoryStatsResponse(
                total_predictions=0,
                avg_predicted_delay=0.0,
                min_predicted_delay=0.0,
                max_predicted_delay=0.0,
                predictions_today=0,
            )

        stats = result[0]

        return HistoryStatsResponse(
            total_predictions=stats['total_predictions'],
            avg_predicted_delay=float(stats['avg_predicted_delay'])
            if stats['avg_predicted_delay']
            else 0.0,
            min_predicted_delay=float(stats['min_predicted_delay'])
            if stats['min_predicted_delay']
            else 0.0,
            max_predicted_delay=float(stats['max_predicted_delay'])
            if stats['max_predicted_delay']
            else 0.0,
            predictions_today=predictions_today,
        )

    except Exception as e:
        logger.error(f'Erro ao calcular estatísticas: {str(e)}')
        raise HTTPException(
            status_code=500, detail=f'Erro ao calcular estatísticas: {str(e)}'
        )


@router.delete('/history/')
async def clear_history():
    """
    Endpoint para limpar histórico de predições
    """
    try:
        db = InMemoryDatabase()
        predictions_collection = db.get_collection('predictions')

        result = predictions_collection.delete_many({})

        logger.info(
            f'Histórico limpo: {result.deleted_count} registros removidos'
        )

        return {
            'message': 'Histórico limpo com sucesso',
            'deleted_count': result.deleted_count,
        }

    except Exception as e:
        logger.error(f'Erro ao limpar histórico: {str(e)}')
        raise HTTPException(
            status_code=500, detail=f'Erro ao limpar histórico: {str(e)}'
        )


@router.get('/history/search')
async def search_predictions(
    dep_delay_min: int = Query(None, description='Atraso mínimo de partida'),
    dep_delay_max: int = Query(None, description='Atraso máximo de partida'),
    hour_min: int = Query(None, description='Hora mínima', ge=0, le=23),
    hour_max: int = Query(None, description='Hora máxima', ge=0, le=23),
    limit: int = Query(
        default=50, description='Máximo de resultados', ge=1, le=100
    ),
):
    """
    Endpoint para buscar predições com filtros
    """
    try:
        db = InMemoryDatabase()
        predictions_collection = db.get_collection('predictions')

        # Construir query de filtro
        query = {}

        if dep_delay_min is not None:
            query.setdefault('input_data.dep_delay', {})[
                '$gte'
            ] = dep_delay_min

        if dep_delay_max is not None:
            query.setdefault('input_data.dep_delay', {})[
                '$lte'
            ] = dep_delay_max

        if hour_min is not None:
            query.setdefault('input_data.hour', {})['$gte'] = hour_min

        if hour_max is not None:
            query.setdefault('input_data.hour', {})['$lte'] = hour_max

        # Buscar com filtros
        cursor = (
            predictions_collection.find(query, {'_id': 0})
            .sort('timestamp', -1)
            .limit(limit)
        )

        predictions = []
        for doc in cursor:
            predictions.append(
                PredictionHistory(
                    input_data=doc['input_data'],
                    predicted_arr_delay=doc['predicted_arr_delay'],
                    timestamp=doc['timestamp'],
                    model_version=doc.get('model_version', 'unknown'),
                )
            )

        total_found = predictions_collection.count_documents(query)

        logger.info(
            f'Busca realizada: {len(predictions)} resultados encontrados'
        )

        return {
            'total_found': total_found,
            'returned': len(predictions),
            'predictions': predictions,
            'filters_applied': {
                'dep_delay_min': dep_delay_min,
                'dep_delay_max': dep_delay_max,
                'hour_min': hour_min,
                'hour_max': hour_max,
            },
        }

    except Exception as e:
        logger.error(f'Erro na busca de predições: {str(e)}')
        raise HTTPException(
            status_code=500, detail=f'Erro na busca de predições: {str(e)}'
        )
