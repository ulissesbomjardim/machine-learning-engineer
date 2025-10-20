import os
import tempfile
import unittest
from unittest.mock import Mock, mock_open, patch

import pandas as pd
import requests

from src.pipeline.extract.extract_airportdb_api import (
    fetch_airport_coordinates,
    get_airport_codes,
)


class TestExtractAirportdbApi(unittest.TestCase):
    def setUp(self):
        """Configuração inicial para os testes"""
        self.temp_dir = tempfile.mkdtemp()
        self.sample_airport_codes = ['JFK', 'LAX', 'ORD']

        # Dados de exemplo para CSV
        self.sample_csv_data = pd.DataFrame(
            {
                'origin': ['JFK', 'LAX', 'JFK', 'ORD', None],
                'dest': ['LAX', 'ORD', 'ORD', 'JFK', 'SFO'],
                'flight_num': ['AA100', 'DL200', 'UA300', 'SW400', 'AA500'],
            }
        )

    def tearDown(self):
        """Limpeza após os testes"""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('src.pipeline.extract.extract_airportdb_api.pd.read_csv')
    def test_get_airport_codes_success(self, mock_read_csv):
        """Testa a extração bem-sucedida de códigos de aeroportos"""
        mock_read_csv.return_value = self.sample_csv_data

        input_path = 'test/path'
        result = get_airport_codes(input_path)

        # Verificar se o arquivo correto foi lido
        expected_path = os.path.join(input_path, 'airports-database.csv')
        mock_read_csv.assert_called_once_with(expected_path)

        # Verificar se os códigos únicos foram extraídos corretamente
        expected_codes = ['JFK', 'LAX', 'ORD', 'SFO']
        self.assertEqual(len(result), 4)
        for code in expected_codes:
            self.assertIn(code, result)

    @patch('src.pipeline.extract.extract_airportdb_api.pd.read_csv')
    def test_get_airport_codes_with_duplicates(self, mock_read_csv):
        """Testa se duplicatas são removidas corretamente"""
        # Dados com mais duplicatas
        duplicate_data = pd.DataFrame(
            {
                'origin': ['JFK', 'JFK', 'LAX', 'LAX'],
                'dest': ['LAX', 'LAX', 'JFK', 'JFK'],
            }
        )
        mock_read_csv.return_value = duplicate_data

        result = get_airport_codes('test/path')

        # Deve retornar apenas códigos únicos
        expected_unique_codes = ['JFK', 'LAX']
        self.assertEqual(len(result), 2)
        for code in expected_unique_codes:
            self.assertIn(code, result)

    @patch('src.pipeline.extract.extract_airportdb_api.pd.read_csv')
    def test_get_airport_codes_file_not_found(self, mock_read_csv):
        """Testa o comportamento quando o arquivo não é encontrado"""
        mock_read_csv.side_effect = FileNotFoundError('Arquivo não encontrado')

        with self.assertRaises(FileNotFoundError):
            get_airport_codes('invalid/path')

    @patch('src.pipeline.extract.extract_airportdb_api.pd.read_csv')
    def test_get_airport_codes_missing_columns(self, mock_read_csv):
        """Testa o comportamento quando colunas necessárias estão ausentes"""
        # DataFrame sem as colunas 'origin' e 'dest'
        invalid_data = pd.DataFrame(
            {'departure': ['JFK', 'LAX'], 'arrival': ['LAX', 'JFK']}
        )
        mock_read_csv.return_value = invalid_data

        with self.assertRaises(KeyError):
            get_airport_codes('test/path')

    @patch('src.pipeline.extract.extract_airportdb_api.requests.get')
    @patch('src.pipeline.extract.extract_airportdb_api.os.makedirs')
    @patch('src.pipeline.extract.extract_airportdb_api.pd.DataFrame.to_csv')
    @patch(
        'src.pipeline.extract.extract_airportdb_api.airportdb_key',
        'test_api_key',
    )
    def test_fetch_airport_coordinates_success(
        self, mock_to_csv, mock_makedirs, mock_requests_get
    ):
        """Testa a busca bem-sucedida de coordenadas de aeroportos"""
        # Mock da resposta da API
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'latitude_deg': 40.6413,
            'longitude_deg': -73.7781,
        }
        mock_requests_get.return_value = mock_response

        airport_codes = ['JFK']
        output_path = self.temp_dir

        result = fetch_airport_coordinates(airport_codes, output_path)

        # Verificar se a API foi chamada corretamente
        expected_url = (
            'https://airportdb.io/api/v1/airport/KJFK?apiToken=test_api_key'
        )
        mock_requests_get.assert_called_once_with(expected_url)

        # Verificar se o diretório foi criado
        mock_makedirs.assert_called_once_with(output_path, exist_ok=True)

        # Verificar se o CSV foi salvo
        expected_file_path = os.path.join(output_path, 'airportdb.csv')
        mock_to_csv.assert_called_once_with(expected_file_path, index=False)

        # Verificar mensagem de retorno
        self.assertIn('sucesso', result)

    @patch('src.pipeline.extract.extract_airportdb_api.requests.get')
    @patch(
        'src.pipeline.extract.extract_airportdb_api.airportdb_key',
        'test_api_key',
    )
    def test_fetch_airport_coordinates_api_error(self, mock_requests_get):
        """Testa o comportamento quando a API retorna erro"""
        # Mock da resposta com erro
        mock_response = Mock()
        mock_response.status_code = 404
        mock_requests_get.return_value = mock_response

        airport_codes = ['INVALID']

        result = fetch_airport_coordinates(airport_codes, self.temp_dir)

        # Verificar se o erro foi tratado
        self.assertEqual(result, 'Nenhum dado foi coletado')

    @patch('src.pipeline.extract.extract_airportdb_api.requests.get')
    @patch(
        'src.pipeline.extract.extract_airportdb_api.airportdb_key',
        'test_api_key',
    )
    def test_fetch_airport_coordinates_missing_coordinates(
        self, mock_requests_get
    ):
        """Testa o comportamento quando a API não retorna coordenadas"""
        # Mock da resposta sem coordenadas
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'name': 'Test Airport'
            # Sem latitude_deg e longitude_deg
        }
        mock_requests_get.return_value = mock_response

        airport_codes = ['TEST']

        result = fetch_airport_coordinates(airport_codes, self.temp_dir)

        # Verificar se nenhum dado foi coletado
        self.assertEqual(result, 'Nenhum dado foi coletado')

    @patch('src.pipeline.extract.extract_airportdb_api.requests.get')
    @patch(
        'src.pipeline.extract.extract_airportdb_api.airportdb_key',
        'test_api_key',
    )
    def test_fetch_airport_coordinates_network_error(self, mock_requests_get):
        """Testa o comportamento quando há erro de rede"""
        # Mock de exceção de rede
        mock_requests_get.side_effect = requests.exceptions.ConnectionError(
            'Erro de conexão'
        )

        airport_codes = ['JFK']

        result = fetch_airport_coordinates(airport_codes, self.temp_dir)

        # Verificar se o erro foi tratado
        self.assertEqual(result, 'Nenhum dado foi coletado')

    @patch('src.pipeline.extract.extract_airportdb_api.requests.get')
    @patch('src.pipeline.extract.extract_airportdb_api.os.makedirs')
    @patch('src.pipeline.extract.extract_airportdb_api.pd.DataFrame.to_csv')
    @patch(
        'src.pipeline.extract.extract_airportdb_api.airportdb_key',
        'test_api_key',
    )
    def test_fetch_airport_coordinates_multiple_airports(
        self, mock_to_csv, mock_makedirs, mock_requests_get
    ):
        """Testa a busca de coordenadas para múltiplos aeroportos"""
        # Mock de respostas para diferentes aeroportos
        def mock_response_side_effect(url):
            mock_response = Mock()
            mock_response.status_code = 200

            if 'KJFK' in url:
                mock_response.json.return_value = {
                    'latitude_deg': 40.6413,
                    'longitude_deg': -73.7781,
                }
            elif 'KLAX' in url:
                mock_response.json.return_value = {
                    'latitude_deg': 33.9425,
                    'longitude_deg': -118.4081,
                }

            return mock_response

        mock_requests_get.side_effect = mock_response_side_effect

        airport_codes = ['JFK', 'LAX']

        result = fetch_airport_coordinates(airport_codes, self.temp_dir)

        # Verificar se ambas as APIs foram chamadas
        self.assertEqual(mock_requests_get.call_count, 2)

        # Verificar mensagem de sucesso
        self.assertIn('sucesso', result)

    def test_fetch_airport_coordinates_empty_list(self):
        """Testa o comportamento com lista vazia de aeroportos"""
        airport_codes = []

        result = fetch_airport_coordinates(airport_codes, self.temp_dir)

        # Deve retornar mensagem de nenhum dado coletado
        self.assertEqual(result, 'Nenhum dado foi coletado')


if __name__ == '__main__':
    unittest.main()
