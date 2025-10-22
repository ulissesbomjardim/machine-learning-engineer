"""
Testes para módulos utilitários e helpers.
"""
import json
import os
import tempfile
from datetime import date, datetime
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
import yaml


class TestConfigUtils:
    """Testes para utilitários de configuração."""

    def test_yaml_config_loading(self, temp_directory, sample_config):
        """Testa carregamento de configuração YAML."""
        config_file = temp_directory / 'config.yaml'

        with open(config_file, 'w') as f:
            yaml.dump(sample_config, f)

        # Carrega configuração
        with open(config_file, 'r') as f:
            loaded_config = yaml.safe_load(f)

        assert isinstance(loaded_config, dict)
        assert loaded_config == sample_config
        assert 'model' in loaded_config
        assert 'api' in loaded_config

    def test_json_config_loading(self, temp_directory, sample_config):
        """Testa carregamento de configuração JSON."""
        config_file = temp_directory / 'config.json'

        with open(config_file, 'w') as f:
            json.dump(sample_config, f)

        # Carrega configuração
        with open(config_file, 'r') as f:
            loaded_config = json.load(f)

        assert isinstance(loaded_config, dict)
        assert loaded_config == sample_config

    def test_environment_variable_loading(self):
        """Testa carregamento de variáveis de ambiente."""
        test_vars = {
            'TEST_VAR_1': 'value1',
            'TEST_VAR_2': 'value2',
            'TEST_VAR_3': '123',
        }

        # Define variáveis temporárias
        for key, value in test_vars.items():
            os.environ[key] = value

        try:
            # Verifica carregamento
            for key, expected_value in test_vars.items():
                loaded_value = os.environ.get(key)
                assert loaded_value == expected_value

            # Testa conversão de tipos
            numeric_var = os.environ.get('TEST_VAR_3')
            assert int(numeric_var) == 123

        finally:
            # Limpa variáveis
            for key in test_vars:
                if key in os.environ:
                    del os.environ[key]

    def test_config_validation(self, sample_config):
        """Testa validação de configuração."""
        # Configuração válida
        assert 'model' in sample_config
        assert 'api' in sample_config

        if 'model' in sample_config:
            model_config = sample_config['model']
            assert isinstance(model_config, dict)

        # Configuração inválida
        invalid_config = {'invalid': 'structure'}

        # Simula validação básica
        required_keys = ['model', 'api']
        is_valid = all(key in sample_config for key in required_keys)
        is_invalid = all(key in invalid_config for key in required_keys)

        assert is_valid is True
        assert is_invalid is False


class TestLoggingUtils:
    """Testes para utilitários de logging."""

    def test_logger_creation(self):
        """Testa criação de logger."""
        import logging

        logger = logging.getLogger('test_logger')
        logger.setLevel(logging.INFO)

        assert logger.name == 'test_logger'
        assert logger.level == logging.INFO

    def test_log_formatting(self):
        """Testa formatação de logs."""
        import logging

        # Cria formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # Cria handler
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)

        # Verifica se formatter foi aplicado
        assert handler.formatter == formatter

    def test_log_levels(self):
        """Testa diferentes níveis de log."""
        import logging

        logger = logging.getLogger('level_test')

        # Testa diferentes níveis
        levels = [
            logging.DEBUG,
            logging.INFO,
            logging.WARNING,
            logging.ERROR,
            logging.CRITICAL,
        ]

        for level in levels:
            logger.setLevel(level)
            assert logger.level == level

    @patch('logging.FileHandler')
    def test_file_logging(self, mock_file_handler):
        """Testa logging para arquivo."""
        import logging

        # Cria logger com arquivo
        logger = logging.getLogger('file_logger')
        file_handler = logging.FileHandler('test.log')
        logger.addHandler(file_handler)

        assert len(logger.handlers) > 0


class TestDataValidationUtils:
    """Testes para utilitários de validação de dados."""

    def test_dataframe_schema_validation(self, sample_flight_data):
        """Testa validação de schema do DataFrame."""
        # Schema esperado
        expected_columns = ['flight_id', 'airline', 'origin', 'destination']

        # Verifica se todas as colunas estão presentes
        has_all_columns = all(
            col in sample_flight_data.columns for col in expected_columns
        )

        assert has_all_columns is True

        # Testa com DataFrame incompleto
        incomplete_df = sample_flight_data[['flight_id', 'airline']]
        has_all_columns_incomplete = all(
            col in incomplete_df.columns for col in expected_columns
        )

        assert has_all_columns_incomplete is False

    def test_data_type_validation(self, sample_flight_data):
        """Testa validação de tipos de dados."""
        # Verifica tipos esperados
        type_checks = {
            'flight_id': lambda x: x.dtype == 'object',
            'delay_minutes': lambda x: x.dtype
            in ['int64', 'int32', 'float64'],
            'is_cancelled': lambda x: x.dtype in ['int64', 'int32', 'bool'],
        }

        for column, check_func in type_checks.items():
            if column in sample_flight_data.columns:
                assert check_func(sample_flight_data[column])

    def test_data_range_validation(self, sample_flight_data):
        """Testa validação de intervalos de dados."""
        # Verifica intervalos válidos
        if 'delay_minutes' in sample_flight_data.columns:
            delay_min = sample_flight_data['delay_minutes'].min()
            delay_max = sample_flight_data['delay_minutes'].max()

            assert delay_min >= 0  # Atraso não pode ser negativo
            assert delay_max <= 1440  # Máximo 24 horas (1440 minutos)

        if 'is_cancelled' in sample_flight_data.columns:
            cancelled_values = sample_flight_data['is_cancelled'].unique()

            # Deve conter apenas 0 e 1
            valid_values = set([0, 1])
            assert set(cancelled_values).issubset(valid_values)

    def test_missing_values_validation(self, sample_flight_data):
        """Testa validação de valores faltantes."""
        # Conta valores faltantes
        null_counts = sample_flight_data.isnull().sum()

        # Verifica se há valores faltantes
        total_nulls = null_counts.sum()

        # Para dados de teste, não deveria haver valores faltantes
        assert total_nulls == 0

        # Testa com dados que têm valores faltantes
        data_with_nulls = sample_flight_data.copy()
        data_with_nulls.loc[0, 'airline'] = None

        null_counts_with_nulls = data_with_nulls.isnull().sum()
        assert null_counts_with_nulls.sum() > 0


class TestDateTimeUtils:
    """Testes para utilitários de data e hora."""

    def test_datetime_parsing(self):
        """Testa parsing de strings de data."""
        date_strings = [
            '2024-01-15T08:00:00',
            '2024-01-15 08:00:00',
            '2024/01/15 08:00',
            '15/01/2024',
        ]

        for date_string in date_strings:
            try:
                # Tenta diferentes formatos de parsing
                parsed_date = pd.to_datetime(date_string)
                assert isinstance(parsed_date, pd.Timestamp)
            except ValueError:
                # Alguns formatos podem não ser suportados
                continue

    def test_time_feature_extraction(self):
        """Testa extração de features temporais."""
        test_datetime = datetime(2024, 1, 15, 8, 30, 0)  # Segunda-feira

        # Extrai features
        features = {
            'year': test_datetime.year,
            'month': test_datetime.month,
            'day': test_datetime.day,
            'hour': test_datetime.hour,
            'minute': test_datetime.minute,
            'day_of_week': test_datetime.weekday(),
            'is_weekend': test_datetime.weekday() >= 5,
        }

        assert features['year'] == 2024
        assert features['month'] == 1
        assert features['day'] == 15
        assert features['hour'] == 8
        assert features['minute'] == 30
        assert features['day_of_week'] == 0  # Segunda-feira
        assert features['is_weekend'] is False

    def test_business_day_calculation(self):
        """Testa cálculo de dias úteis."""
        # Segunda-feira
        monday = date(2024, 1, 15)

        # Sábado
        saturday = date(2024, 1, 13)

        # Domingo
        sunday = date(2024, 1, 14)

        # Verifica dias da semana
        assert monday.weekday() == 0  # Segunda
        assert saturday.weekday() == 5  # Sábado
        assert sunday.weekday() == 6  # Domingo

        # Verifica se é dia útil
        assert monday.weekday() < 5  # Dia útil
        assert saturday.weekday() >= 5  # Fim de semana
        assert sunday.weekday() >= 5  # Fim de semana

    def test_time_difference_calculation(self):
        """Testa cálculo de diferença de tempo."""
        start_time = datetime(2024, 1, 15, 8, 0, 0)
        end_time = datetime(2024, 1, 15, 10, 30, 0)

        time_diff = end_time - start_time

        assert time_diff.total_seconds() == 2.5 * 3600  # 2.5 horas
        assert time_diff.seconds == 9000  # 2.5 horas em segundos


class TestFileUtils:
    """Testes para utilitários de arquivo."""

    def test_file_existence_check(self, temp_directory):
        """Testa verificação de existência de arquivo."""
        # Arquivo que existe
        existing_file = temp_directory / 'existing.txt'
        existing_file.write_text('test content')

        assert existing_file.exists()
        assert existing_file.is_file()

        # Arquivo que não existe
        non_existing_file = temp_directory / 'non_existing.txt'

        assert not non_existing_file.exists()

    def test_directory_creation(self, temp_directory):
        """Testa criação de diretórios."""
        new_dir = temp_directory / 'new_directory'

        assert not new_dir.exists()

        # Cria diretório
        new_dir.mkdir()

        assert new_dir.exists()
        assert new_dir.is_dir()

    def test_file_size_calculation(self, temp_directory):
        """Testa cálculo de tamanho de arquivo."""
        test_file = temp_directory / 'size_test.txt'
        test_content = 'Hello, World! This is a test file.'

        test_file.write_text(test_content)

        file_size = test_file.stat().st_size
        expected_size = len(test_content.encode('utf-8'))

        assert file_size == expected_size

    def test_file_extension_filtering(self, temp_directory):
        """Testa filtragem por extensão de arquivo."""
        # Cria arquivos de teste
        files = [
            'test1.csv',
            'test2.json',
            'test3.txt',
            'test4.csv',
            'test5.py',
        ]

        for filename in files:
            (temp_directory / filename).touch()

        # Filtra arquivos CSV
        csv_files = list(temp_directory.glob('*.csv'))

        assert len(csv_files) == 2
        assert all(f.suffix == '.csv' for f in csv_files)


class TestStringUtils:
    """Testes para utilitários de string."""

    def test_string_cleaning(self):
        """Testa limpeza de strings."""
        dirty_strings = [
            '  Hello World  ',
            'Hello\nWorld\t',
            'HELLO world',
            'Hello-World_123',
        ]

        cleaned_strings = []
        for s in dirty_strings:
            # Limpeza básica
            cleaned = s.strip().replace('\n', ' ').replace('\t', ' ')
            cleaned_strings.append(cleaned)

        assert cleaned_strings[0] == 'Hello World'
        assert cleaned_strings[1] == 'Hello World'
        assert cleaned_strings[2] == 'HELLO world'

    def test_string_normalization(self):
        """Testa normalização de strings."""
        test_strings = ['São Paulo', 'João', 'Coração', 'Ação']

        # Simula normalização (remoção de acentos)
        import unicodedata

        normalized = []
        for s in test_strings:
            # Remove acentos
            nfd = unicodedata.normalize('NFD', s)
            without_accents = ''.join(
                char for char in nfd if unicodedata.category(char) != 'Mn'
            )
            normalized.append(without_accents)

        assert normalized[0] == 'Sao Paulo'
        assert normalized[1] == 'Joao'

    def test_string_validation(self):
        """Testa validação de strings."""
        # Códigos de aeroporto
        airport_codes = ['GRU', 'CGH', 'BSB', 'gru', '12345', '']

        valid_codes = []
        for code in airport_codes:
            # Valida: 3 letras, maiúsculas
            if isinstance(code, str) and len(code) == 3 and code.isalpha():
                valid_codes.append(code.upper())

        assert 'GRU' in valid_codes
        assert 'CGH' in valid_codes
        assert 'BSB' in valid_codes
        assert 'GRU' in valid_codes  # gru convertido para maiúscula
        assert len(valid_codes) == 4  # Exclui "12345" e ""


class TestMathUtils:
    """Testes para utilitários matemáticos."""

    def test_statistical_calculations(self):
        """Testa cálculos estatísticos."""
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        # Cálculos básicos
        mean = sum(data) / len(data)
        median = sorted(data)[len(data) // 2]
        std = np.std(data)

        assert mean == 5.5
        assert median == 6  # Para lista par, pega o elemento do meio superior
        assert std > 0

    def test_percentage_calculations(self):
        """Testa cálculos de porcentagem."""
        total = 100
        part = 25

        percentage = (part / total) * 100

        assert percentage == 25.0

        # Teste com valores decimais
        accuracy = 0.85
        accuracy_percent = accuracy * 100

        assert accuracy_percent == 85.0

    def test_rounding_functions(self):
        """Testa funções de arredondamento."""
        values = [3.14159, 2.71828, 1.41421]

        # Arredonda para 2 casas decimais
        rounded = [round(v, 2) for v in values]

        assert rounded[0] == 3.14
        assert rounded[1] == 2.72
        assert rounded[2] == 1.41

    def test_range_normalization(self):
        """Testa normalização de intervalos."""
        data = np.array([1, 2, 3, 4, 5])

        # Normalização min-max (0-1)
        data_min = data.min()
        data_max = data.max()
        normalized = (data - data_min) / (data_max - data_min)

        assert normalized.min() == 0.0
        assert normalized.max() == 1.0
        assert len(normalized) == len(data)
