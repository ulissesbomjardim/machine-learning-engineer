import logging
from datetime import datetime

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import src.routers.model.predict as predict_module

# Importar routers
from src.routers.model import history, load, predict
from src.services.database import InMemoryDatabase

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title='Flight Delay Prediction API',
    description='API para predição de atrasos em voos usando Machine Learning',
    version='1.0.0',
)

# Adicionar middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

# Incluir routers dos modelos - Seguindo especificação exata
app.include_router(predict.router, prefix='/model', tags=['prediction'])
app.include_router(load.router, prefix='/model', tags=['model'])
app.include_router(history.router, prefix='/model', tags=['history'])


@app.get('/health', status_code=200, tags=['health'], summary='Health check')
async def health():
    """
    Endpoint de verificação de saúde da API
    """
    try:
        # Verificar conexão com database com mais detalhes
        db = InMemoryDatabase()
        db_health = db.health_check()

        # Verificar se modelo está carregado
        model_loaded = predict_module.loaded_model is not None

        # Determinar status geral
        if db_health['connected'] and model_loaded:
            status = 'healthy'
        elif db_health['connected'] or model_loaded:
            status = 'degraded'
        else:
            status = 'unhealthy'

        return {
            'status': status,
            'timestamp': datetime.now().isoformat(),
            'database': db_health,
            'model_loaded': model_loaded,
            'model_type': type(predict_module.loaded_model).__name__
            if predict_module.loaded_model
            else None,
            'version': '1.0.0',
            'service': 'Flight Delay Prediction API',
        }

    except Exception as e:
        logger.error(f'Erro no health check: {str(e)}')
        return {
            'status': 'error',
            'timestamp': datetime.now().isoformat(),
            'database': {
                'status': 'unhealthy',
                'connected': False,
                'error': str(e),
            },
            'model_loaded': False,
            'version': '1.0.0',
            'error': str(e),
        }


@app.get('/', tags=['root'])
async def root():
    """
    Endpoint raiz da API
    """
    return {
        'message': 'Flight Delay Prediction API is running',
        'version': '1.0.0',
        'endpoints': {
            'health': '/health',
            'model_predict': '/model/predict/',
            'model_load': '/model/load/',
            'model_history': '/model/history/',
            'docs': '/docs',
        },
    }


# Manter endpoints de exemplo para compatibilidade
@app.post('/user/', tags=['example'], summary='Insert user')
async def insert(data: dict):
    db = InMemoryDatabase()
    users = db.get_collection('users')
    users.insert_one(data)
    return {'status': 'ok'}


@app.get(
    '/user/{name}',
    status_code=200,
    tags=['example'],
    summary='Get user by name',
)
async def get(name: str):
    db = InMemoryDatabase()
    users = db.get_collection('users')
    user = users.find_one({'name': name})
    return {'status': 'ok', 'user': user}


@app.get('/user/', tags=['example'], summary='List all users')
async def list():
    db = InMemoryDatabase()
    users = db.get_collection('users')
    return {'status': 'ok', 'users': [x for x in users.find({}, {'_id': 0})]}


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8080, log_level='debug')
