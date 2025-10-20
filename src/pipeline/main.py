from src.pipeline.extract.extract_airportdb_api import (
    fetch_airport_coordinates,
)
from src.pipeline.extract.extract_airports_database import (
    download_and_extract_airports_database,
)


def run_airports_database():
    """
    Função principal do script.
    """
    print('Iniciando download e extração da base de dados de aeroportos...')
    success = download_and_extract_airports_database()

    if success:
        print('✅ Processo concluído com sucesso!')
    else:
        print('❌ Processo falhou. Verifique os logs acima.')


def run_fetch_airport_coordinates():
    """
    Função principal do script.
    """
    input_path = 'data/input/airport_database'
    airport_codes = get_airport_codes(input_path)
    print(airport_codes)
    fetch_airport_coordinates(airport_codes)


if __name__ == '__main__':
    run_airports_database()
    run_fetch_airport_coordinates()
