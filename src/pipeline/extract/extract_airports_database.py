"""
Script para download e extração da base de dados de aeroportos.
"""

import os
import shutil
import zipfile
from pathlib import Path

import requests


def download_and_extract_airports_database():
    """
    Faz o download do arquivo ZIP de aeroportos e extrai o CSV na pasta data/input/airport_database.
    """
    # URLs e caminhos
    url = 'https://github.com/PicPay/case-machine-learning-engineer-pleno/raw/main/notebook/airports-database.zip'

    # Definir caminhos relativos à raiz do projeto
    project_root = Path(__file__).parent.parent.parent.parent
    data_input_dir = project_root / 'data' / 'input' / 'airport_database'
    temp_zip_path = project_root / 'airports-database.zip'
    temp_extract_dir = project_root / 'airports-database'

    try:
        # Criar diretório data/input/airport_database se não existir
        data_input_dir.mkdir(parents=True, exist_ok=True)

        print('Fazendo download do arquivo ZIP...')
        # Download do arquivo ZIP
        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(temp_zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print('Download concluído. Extraindo arquivos...')

        # Extrair o ZIP
        with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_extract_dir)

        # Mover arquivos CSV para data/input/airport_database
        csv_files_moved = 0
        for root, dirs, files in os.walk(temp_extract_dir):
            for file in files:
                if file.endswith('.csv') and not file.startswith('._'):
                    source_path = Path(root) / file
                    dest_path = data_input_dir / file
                    shutil.move(str(source_path), str(dest_path))
                    print(f'Arquivo movido: {file} -> {dest_path}')
                    csv_files_moved += 1
                elif file.startswith('._'):
                    # Remover arquivos de metadados do macOS
                    unwanted_file = Path(root) / file
                    unwanted_file.unlink()
                    print(f'Arquivo de metadados removido: {file}')

        print(
            f'Extração concluída. {csv_files_moved} arquivo(s) CSV movido(s) para data/input/airport_database/'
        )

    except requests.exceptions.RequestException as e:
        print(f'Erro no download: {e}')
        return False
    except zipfile.BadZipFile as e:
        print(f'Erro ao extrair ZIP: {e}')
        return False
    except Exception as e:
        print(f'Erro inesperado: {e}')
        return False

    finally:
        # Limpeza: remover arquivos temporários
        if temp_zip_path.exists():
            temp_zip_path.unlink()
            print('Arquivo ZIP temporário removido.')

        if temp_extract_dir.exists():
            shutil.rmtree(temp_extract_dir)
            print('Diretório temporário removido.')

    return True
