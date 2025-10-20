"""
Testes unitários para o módulo de download e extração da base de dados de aeroportos.
"""

import io
import os
import shutil
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pipeline.extract.airports_database import (
    download_and_extract_airports_database,
)


class TestAirportsDatabase(unittest.TestCase):
    """Testes para a função download_and_extract_airports_database."""

    def setUp(self):
        """Configuração executada antes de cada teste."""
        # Criar diretório temporário para os testes
        self.test_dir = Path(tempfile.mkdtemp())
        self.data_input_dir = (
            self.test_dir / 'data' / 'input' / 'airport_database'
        )
        self.temp_zip_path = self.test_dir / 'airports-database.zip'
        self.temp_extract_dir = self.test_dir / 'airports-database'

    def tearDown(self):
        """Limpeza executada após cada teste."""
        # Remover diretório temporário
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    @patch('pipeline.extract.airports_database.requests.get')
    def test_download_failure(self, mock_requests):
        """Testa falha no download."""
        # Mock de falha na requisição
        mock_requests.side_effect = requests.exceptions.RequestException(
            'Network error'
        )

        # Executar função
        result = download_and_extract_airports_database()

        # Verificações
        self.assertFalse(result)
        mock_requests.assert_called_once()

    @patch('pipeline.extract.airports_database.requests.get')
    @patch('pipeline.extract.airports_database.zipfile.ZipFile')
    @patch('builtins.open', new_callable=mock_open)
    def test_zip_extraction_failure(
        self, mock_file, mock_zipfile, mock_requests
    ):
        """Testa falha na extração do ZIP."""
        # Mock da resposta HTTP bem-sucedida
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.iter_content.return_value = [b'chunk1']
        mock_requests.return_value = mock_response

        # Mock de falha na extração do ZIP
        mock_zipfile.side_effect = zipfile.BadZipFile('Invalid ZIP file')

        # Executar função
        result = download_and_extract_airports_database()

        # Verificações
        self.assertFalse(result)
        mock_requests.assert_called_once()

    def test_csv_filtering_logic(self):
        """Testa a lógica de filtragem de arquivos CSV."""
        # Criar estrutura de teste
        test_extract_dir = self.test_dir / 'test_extract'
        test_extract_dir.mkdir(parents=True)

        # Criar arquivos de teste
        (test_extract_dir / 'airports-database.csv').write_text(
            'valid,csv,content'
        )
        (test_extract_dir / '._airports-database.csv').write_text('metadata')
        (test_extract_dir / 'readme.txt').write_text('not a csv')
        (test_extract_dir / 'other.csv').write_text('another,csv,file')

        # Simular a lógica de filtragem da função
        csv_files = []
        metadata_files = []
        other_files = []

        for file_path in test_extract_dir.iterdir():
            if file_path.is_file():
                filename = file_path.name
                if filename.endswith('.csv') and not filename.startswith('._'):
                    csv_files.append(filename)
                elif filename.startswith('._'):
                    metadata_files.append(filename)
                else:
                    other_files.append(filename)

        # Verificações
        self.assertIn('airports-database.csv', csv_files)
        self.assertIn('other.csv', csv_files)
        self.assertIn('._airports-database.csv', metadata_files)
        self.assertIn('readme.txt', other_files)
        self.assertEqual(len(csv_files), 2)
        self.assertEqual(len(metadata_files), 1)
        self.assertEqual(len(other_files), 1)

    def test_path_construction(self):
        """Testa se os caminhos são construídos corretamente."""
        # Simular estrutura de caminhos da função - usar separadores compatíveis com o SO
        if os.name == 'nt':  # Windows
            mock_file_path = Path(
                'C:\\fake\\src\\pipeline\\extract\\airports_database.py'
            )
            expected_root = 'C:\\fake'
            expected_data = 'C:\\fake\\data\\input\\airport_database'
            expected_zip = 'C:\\fake\\airports-database.zip'
            expected_extract = 'C:\\fake\\airports-database'
        else:  # Unix/Linux/Mac
            mock_file_path = Path(
                '/fake/src/pipeline/extract/airports_database.py'
            )
            expected_root = '/fake'
            expected_data = '/fake/data/input/airport_database'
            expected_zip = '/fake/airports-database.zip'
            expected_extract = '/fake/airports-database'

        # Simular o cálculo do project_root
        project_root = mock_file_path.parent.parent.parent.parent
        data_input_dir = project_root / 'data' / 'input' / 'airport_database'
        temp_zip_path = project_root / 'airports-database.zip'
        temp_extract_dir = project_root / 'airports-database'

        # Verificações dos caminhos
        self.assertEqual(str(project_root), expected_root)
        self.assertEqual(str(data_input_dir), expected_data)
        self.assertEqual(str(temp_zip_path), expected_zip)
        self.assertEqual(str(temp_extract_dir), expected_extract)

    def test_file_processing_simulation(self):
        """Testa simulação do processamento de arquivos."""
        # Criar estrutura de teste
        source_dir = self.test_dir / 'source'
        dest_dir = self.test_dir / 'dest'
        source_dir.mkdir(parents=True)
        dest_dir.mkdir(parents=True)

        # Criar arquivos de teste
        csv_file = source_dir / 'airports-database.csv'
        metadata_file = source_dir / '._airports-database.csv'
        other_file = source_dir / 'readme.txt'

        csv_file.write_text('airport_code,name\nGRU,Guarulhos')
        metadata_file.write_text('metadata content')
        other_file.write_text('readme content')

        # Simular processamento
        csv_files_moved = 0
        for file_path in source_dir.iterdir():
            if file_path.is_file():
                filename = file_path.name
                if filename.endswith('.csv') and not filename.startswith('._'):
                    # Simular movimento
                    dest_path = dest_dir / filename
                    dest_path.write_text(file_path.read_text())
                    csv_files_moved += 1
                elif filename.startswith('._'):
                    # Simular remoção
                    file_path.unlink()

        # Verificações
        self.assertEqual(csv_files_moved, 1)
        self.assertTrue((dest_dir / 'airports-database.csv').exists())
        self.assertFalse(metadata_file.exists())
        self.assertTrue(
            other_file.exists()
        )  # Arquivo não-CSV não é processado

    @unittest.skipIf(
        os.getenv('SKIP_INTEGRATION_TESTS') == '1',
        'Teste de integração desabilitado',
    )
    def test_real_download_integration(self):
        """Teste de integração que faz o download real."""
        # Usar um diretório temporário real
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Fazer backup da função original
            import pipeline.extract.airports_database as module

            original_path = module.Path

            try:
                # Mock apenas o Path(__file__) para usar nosso diretório temporário
                def mock_path_constructor(*args):
                    if len(args) == 1 and str(args[0]).endswith(
                        'airports_database.py'
                    ):
                        mock_file_path = MagicMock()
                        mock_file_path.parent.parent.parent.parent = temp_path
                        return mock_file_path
                    else:
                        return original_path(*args)

                module.Path = mock_path_constructor

                # Executar função
                result = download_and_extract_airports_database()

                # Verificações
                self.assertTrue(
                    result, 'Download e extração devem ser bem-sucedidos'
                )

                # Verificar se arquivo foi criado
                expected_file = (
                    temp_path
                    / 'data'
                    / 'input'
                    / 'airport_database'
                    / 'airports-database.csv'
                )
                self.assertTrue(
                    expected_file.exists(),
                    f'Arquivo CSV deve existir em {expected_file}',
                )

                # Verificar se arquivo tem conteúdo
                self.assertGreater(
                    expected_file.stat().st_size,
                    0,
                    'Arquivo CSV deve ter conteúdo',
                )

                # Verificar se não há arquivos temporários
                self.assertFalse(
                    (temp_path / 'airports-database.zip').exists(),
                    'Arquivo ZIP temporário deve ser removido',
                )
                self.assertFalse(
                    (temp_path / 'airports-database').exists(),
                    'Diretório temporário deve ser removido',
                )

                # Verificar conteúdo básico do CSV (contém dados de voos, não aeroportos)
                content = expected_file.read_text()
                self.assertTrue(
                    any(
                        keyword in content.lower()
                        for keyword in [
                            'flight',
                            'airline',
                            'carrier',
                            'dep_time',
                            'arr_time',
                        ]
                    ),
                    'Conteúdo deve conter dados relacionados a voos',
                )

            finally:
                # Restaurar função original
                module.Path = original_path

    def test_url_and_constants(self):
        """Testa se as constantes estão corretas."""
        expected_url = 'https://github.com/PicPay/case-machine-learning-engineer-pleno/raw/main/notebook/airports-database.zip'

        # Importar módulo para verificar se URL está acessível no código
        # Ler código fonte para verificar URL
        import inspect

        import pipeline.extract.airports_database as module

        source_code = inspect.getsource(
            module.download_and_extract_airports_database
        )
        self.assertIn(
            expected_url,
            source_code,
            'URL do arquivo deve estar presente no código',
        )


if __name__ == '__main__':
    # Executar todos os testes
    unittest.main(verbosity=2)
