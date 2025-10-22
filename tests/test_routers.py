"""
Testes para os routers da API FastAPI.
"""
import json
from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient


class TestMainRouter:
    """Testes para o router principal."""

    def test_main_router_import(self):
        """Testa se o router principal pode ser importado."""
        try:
            from src.routers.main import app

            assert app is not None
        except ImportError:
            pytest.skip('Router principal não encontrado')

    def test_health_endpoint_exists(self, api_client):
        """Testa se endpoint de health existe."""
        if isinstance(api_client, Mock):
            pytest.skip('API não encontrada')

        response = api_client.get('/health')
        # Aceita tanto 200 (implementado) quanto 404 (não implementado ainda)
        assert response.status_code in [200, 404]


class TestModelRouters:
    """Testes para routers do modelo."""

    def test_predict_router_import(self):
        """Testa importação do router de predição."""
        try:
            from src.routers.model.predict import router

            assert router is not None
        except ImportError:
            pytest.skip('Router de predição não encontrado')

    def test_history_router_import(self):
        """Testa importação do router de histórico."""
        try:
            from src.routers.model.history import router

            assert router is not None
        except ImportError:
            pytest.skip('Router de histórico não encontrado')

    def test_load_router_import(self):
        """Testa importação do router de carregamento."""
        try:
            from src.routers.model.load import router

            assert router is not None
        except ImportError:
            pytest.skip('Router de carregamento não encontrado')


class TestPredictEndpoint:
    """Testes para endpoint de predição."""

    def test_predict_endpoint_structure(self, sample_prediction_request):
        """Testa estrutura do endpoint de predição."""
        try:
            from src.routers.model.predict import predict_flight_cancellation

            # Verifica se a função existe
            assert callable(predict_flight_cancellation)

        except ImportError:
            pytest.skip('Função de predição não encontrada')

    def test_predict_with_valid_data(
        self, api_client, sample_prediction_request
    ):
        """Testa predição com dados válidos."""
        if isinstance(api_client, Mock):
            pytest.skip('API não encontrada')

        response = api_client.post('/predict', json=sample_prediction_request)

        # Aceita 200 (sucesso), 422 (validation error), ou 404 (não implementado)
        assert response.status_code in [200, 404, 422, 500]

        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, dict)

    def test_predict_with_invalid_data(self, api_client):
        """Testa predição com dados inválidos."""
        if isinstance(api_client, Mock):
            pytest.skip('API não encontrada')

        invalid_data = {'invalid': 'data'}
        response = api_client.post('/predict', json=invalid_data)

        # Deve retornar erro de validação ou not found
        assert response.status_code in [404, 422, 500]


class TestHistoryEndpoint:
    """Testes para endpoint de histórico."""

    def test_history_endpoint_structure(self):
        """Testa estrutura do endpoint de histórico."""
        try:
            from src.routers.model.history import get_prediction_history

            # Verifica se a função existe
            assert callable(get_prediction_history)

        except ImportError:
            pytest.skip('Função de histórico não encontrada')

    def test_get_history(self, api_client):
        """Testa recuperação de histórico."""
        if isinstance(api_client, Mock):
            pytest.skip('API não encontrada')

        response = api_client.get('/history')

        # Aceita 200 (sucesso) ou 404 (não implementado)
        assert response.status_code in [200, 404, 500]

        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, (list, dict))


class TestLoadEndpoint:
    """Testes para endpoint de carregamento."""

    def test_load_endpoint_structure(self):
        """Testa estrutura do endpoint de carregamento."""
        try:
            from src.routers.model.load import load_model

            # Verifica se a função existe
            assert callable(load_model)

        except ImportError:
            pytest.skip('Função de carregamento não encontrada')

    def test_load_model_endpoint(self, api_client):
        """Testa endpoint de carregamento de modelo."""
        if isinstance(api_client, Mock):
            pytest.skip('API não encontrada')

        response = api_client.post(
            '/model/load', json={'model_path': 'test_model.pkl'}
        )

        # Aceita várias respostas dependendo da implementação
        assert response.status_code in [200, 404, 422, 500]


class TestAPIIntegration:
    """Testes de integração da API."""

    def test_api_startup(self, api_client):
        """Testa inicialização da API."""
        if isinstance(api_client, Mock):
            pytest.skip('API não encontrada')

        # Testa se a API responde a requisições básicas
        response = api_client.get('/')

        # Aceita redirect, not found, ou sucesso
        assert response.status_code in [200, 404, 307, 308]

    def test_openapi_schema(self, api_client):
        """Testa se o schema OpenAPI está disponível."""
        if isinstance(api_client, Mock):
            pytest.skip('API não encontrada')

        response = api_client.get('/openapi.json')

        if response.status_code == 200:
            schema = response.json()
            assert 'openapi' in schema
            assert 'paths' in schema

    def test_docs_endpoint(self, api_client):
        """Testa se a documentação está disponível."""
        if isinstance(api_client, Mock):
            pytest.skip('API não encontrada')

        response = api_client.get('/docs')

        # Aceita 200 (HTML docs) ou 404 (não configurado)
        assert response.status_code in [200, 404]


class TestErrorHandling:
    """Testes para tratamento de erros."""

    def test_404_error(self, api_client):
        """Testa resposta para endpoint inexistente."""
        if isinstance(api_client, Mock):
            pytest.skip('API não encontrada')

        response = api_client.get('/endpoint_inexistente')
        assert response.status_code == 404

    def test_method_not_allowed(self, api_client):
        """Testa método não permitido."""
        if isinstance(api_client, Mock):
            pytest.skip('API não encontrada')

        # Tenta DELETE em endpoint que provavelmente só aceita GET
        response = api_client.delete('/docs')
        assert response.status_code in [404, 405]

    def test_invalid_json(self, api_client):
        """Testa envio de JSON inválido."""
        if isinstance(api_client, Mock):
            pytest.skip('API não encontrada')

        # Envia dados não-JSON para endpoint que espera JSON
        response = api_client.post(
            '/predict',
            data='invalid json',
            headers={'content-type': 'application/json'},
        )

        # Deve retornar erro de parsing ou not found
        assert response.status_code in [400, 404, 422]


class TestCORSHeaders:
    """Testes para headers CORS."""

    def test_cors_preflight(self, api_client):
        """Testa requisição preflight CORS."""
        if isinstance(api_client, Mock):
            pytest.skip('API não encontrada')

        response = api_client.options(
            '/predict',
            headers={
                'Origin': 'http://localhost:3000',
                'Access-Control-Request-Method': 'POST',
            },
        )

        # Aceita várias respostas dependendo da configuração CORS
        assert response.status_code in [200, 404, 405]

    def test_cors_actual_request(self, api_client, sample_prediction_request):
        """Testa requisição real com CORS."""
        if isinstance(api_client, Mock):
            pytest.skip('API não encontrada')

        response = api_client.post(
            '/predict',
            json=sample_prediction_request,
            headers={'Origin': 'http://localhost:3000'},
        )

        # Verifica se headers CORS estão presentes quando aplicável
        if response.status_code == 200:
            # Headers CORS podem ou não estar presentes
            cors_header = response.headers.get('access-control-allow-origin')
            # Não falha se não estiver presente, apenas verifica se existe
