import os

import pandas as pd
import requests
from dotenv import load_dotenv

# Carregar as variáveis de ambiente do arquivo .env
load_dotenv(dotenv_path='config/.env')
airportdb_key = os.getenv('AIRPORTDB_CLIENT_SECRET')


def get_airport_codes(input_path):
    """
    Extrai códigos únicos de aeroportos de um arquivo CSV contendo dados de voos.
    Esta função lê um arquivo CSV com informações de voos e extrai todos os códigos
    únicos de aeroportos das colunas de origem e destino.
    Args:
        input_path (str): O caminho do diretório onde o arquivo de entrada está localizado.
        input_file (str): O nome do arquivo CSV contendo dados de voos.
                         Espera-se que tenha colunas 'origin' e 'dest'.
    Returns:
        list: Uma lista de códigos únicos de aeroportos (str) extraídos das colunas
              de origem e destino. Duplicatas são removidas e valores NaN são excluídos.
    Raises:
        FileNotFoundError: Se o caminho do arquivo especificado não existir.
        KeyError: Se o arquivo CSV não contiver as colunas 'origin' ou 'dest'.
        pd.errors.EmptyDataError: Se o arquivo CSV estiver vazio.
    Example:
        codes = get_airport_codes('/data/flights', 'flights.csv')
        print(codes[:3])
        ['JFK', 'LAX', 'ORD']
    """
    airport_db_path = os.path.join(input_path, 'airports-database.csv')
    df = pd.read_csv(airport_db_path)

    # Coletar códigos de aeroportos distintos
    origin_codes = df['origin'].dropna().unique()
    dest_codes = df['dest'].dropna().unique()
    airport_codes = (
        pd.concat([pd.Series(origin_codes), pd.Series(dest_codes)])
        .unique()
        .tolist()
    )

    return airport_codes


def fetch_airport_coordinates(
    airport_codes, output_path='data/input/airportdb'
):
    """
    Função que recebe uma lista de códigos de aeroportos, chama a API airportdb.io
    para cada código e coleta latitude e longitude, salvando em um CSV.

    Args:
        airport_codes (list): Lista de códigos de aeroportos
        output_path (str): Caminho para salvar o arquivo CSV

    Returns:
        str: Mensagem de sucesso
    """
    airport_data = []

    print(f'Iniciando busca para {len(airport_codes)} aeroportos...')

    for airport_code in airport_codes:
        print(f'Buscando dados para o aeroporto: {airport_code}')

        try:
            url = f'https://airportdb.io/api/v1/airport/K{airport_code}?apiToken={airportdb_key}'
            response = requests.get(url)

            if response.status_code == 200:
                data = response.json()

                latitude = data.get('latitude_deg')
                longitude = data.get('longitude_deg')

                if latitude and longitude:
                    airport_info = {
                        'airport_code': airport_code,
                        'latitude': latitude,
                        'longitude': longitude,
                    }
                    airport_data.append(airport_info)
                    print(
                        f'Dados coletados com sucesso: {airport_code} - Latitude: {latitude}, Longitude: {longitude}'
                    )
                else:
                    print(
                        f'Aeroporto {airport_code}: coordenadas não disponíveis'
                    )
            else:
                print(
                    f'Erro na API para {airport_code}: Status {response.status_code}'
                )

        except Exception as e:
            print(f'Erro ao processar aeroporto {airport_code}: {e}')
            continue

    # Criar DataFrame e salvar CSV
    if airport_data:
        df_airports = pd.DataFrame(airport_data)

        # Criar diretório se não existir
        os.makedirs(output_path, exist_ok=True)

        output_file = os.path.join(output_path, 'airportdb.csv')
        df_airports.to_csv(output_file, index=False)

        print(
            f'Processo concluído com sucesso! CSV salvo com dados de {len(airport_data)} aeroportos em {output_file}'
        )
        return f'Arquivo airportdb.csv salvo com sucesso em {output_file}'
    else:
        print(
            'Nenhum dado coletado. Verifique os códigos dos aeroportos e a conectividade com a API.'
        )
        return 'Nenhum dado foi coletado'


if __name__ == '__main__':
    input_path = 'data/input/airport_database'
    airport_codes = get_airport_codes(input_path)
    print(airport_codes)
    fetch_airport_coordinates(airport_codes)
