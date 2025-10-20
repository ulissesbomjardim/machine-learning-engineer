import os

import requests
from dotenv import load_dotenv

# Carregar as variáveis de ambiente do arquivo .env
load_dotenv(dotenv_path='config/.env')
weatherbit_key = os.getenv('WEATHERBIT_CLIENT_SECRET')


latitude = 40.7128
longitude = -74.0060
start_date = '2023-01-01'
end_date = '2023-01-02'   # Sempre adicione +1 dia em relação ao start_date
url = 'https://api.weatherbit.io/v2.0/history/daily'


params = {
    'lat': latitude,
    'lon': longitude,
    'start_date': start_date,
    'end_date': end_date,
    'key': weatherbit_key,
}


headers = {
    'Accept': 'application/json',
}


response = requests.get(url, params=params, headers=headers)
data = response.json()
data
