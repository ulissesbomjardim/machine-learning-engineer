"""
Configurações e fixtures compartilhadas para os testes.
"""
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_flight_data():
    """Fixture com dados de exemplo para testes de voos."""
    return pd.DataFrame(
        {
            'flight_id': ['FL001', 'FL002', 'FL003', 'FL004', 'FL005'],
            'airline': ['TAM', 'GOL', 'AZUL', 'TAM', 'GOL'],
            'origin': ['GRU', 'CGH', 'BSB', 'GIG', 'FOR'],
            'destination': ['RIO', 'BSB', 'GRU', 'FOR', 'GRU'],
            'departure_time': ['08:00', '14:30', '16:45', '20:00', '06:15'],
            'scheduled_departure': pd.to_datetime(
                [
                    '2024-01-15 08:00',
                    '2024-01-15 14:30',
                    '2024-01-15 16:45',
                    '2024-01-15 20:00',
                    '2024-01-15 06:15',
                ]
            ),
            'weather_condition': ['clear', 'cloudy', 'rain', 'clear', 'fog'],
            'delay_minutes': [0, 15, 30, 0, 45],
            'is_cancelled': [0, 0, 1, 0, 1],
        }
    )


@pytest.fixture
def sample_processed_features():
    """Fixture com features processadas para testes de ML."""
    return pd.DataFrame(
        {
            'airline_encoded': [1, 2, 3, 1, 2],
            'origin_encoded': [1, 2, 3, 4, 5],
            'destination_encoded': [1, 3, 1, 5, 1],
            'departure_hour': [8, 14, 16, 20, 6],
            'weather_encoded': [1, 2, 3, 1, 4],
            'historical_delay': [0.1, 0.3, 0.8, 0.2, 0.9],
            'is_cancelled': [0, 0, 1, 0, 1],
        }
    )


@pytest.fixture
def mock_model():
    """Mock de modelo de ML para testes."""
    model = Mock()
    model.predict.return_value = np.array([0, 1, 0])
    model.predict_proba.return_value = np.array(
        [[0.8, 0.2], [0.3, 0.7], [0.9, 0.1]]
    )
    return model


@pytest.fixture
def temp_directory():
    """Fixture que cria um diretório temporário para testes."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_config():
    """Configuração de exemplo para testes."""
    return {
        'model': {
            'algorithm': 'random_forest',
            'n_estimators': 100,
            'max_depth': 10,
        },
        'data': {'input_path': 'data/input', 'output_path': 'data/output'},
        'api': {'host': '0.0.0.0', 'port': 8000},
    }


@pytest.fixture
def api_client():
    """Cliente de teste para a API FastAPI."""
    from fastapi.testclient import TestClient

    # Assumindo que a API principal está em src.routers.main
    try:
        from src.routers.main import app

        return TestClient(app)
    except ImportError:
        # Se não existir, retorna um mock
        return Mock()


@pytest.fixture
def sample_prediction_request():
    """Payload de exemplo para requisições de predição."""
    return {
        'flight_id': 'FL001',
        'airline': 'TAM',
        'origin': 'GRU',
        'destination': 'RIO',
        'departure_time': '08:00',
        'scheduled_departure': '2024-01-15T08:00:00',
        'weather_condition': 'clear',
    }


@pytest.fixture
def sample_prediction_response():
    """Resposta de exemplo para predições."""
    return {
        'flight_id': 'FL001',
        'prediction': 0,
        'probability': 0.15,
        'risk_level': 'low',
        'timestamp': '2024-01-15T10:00:00',
    }
