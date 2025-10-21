import logging
import os
import pickle

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

router = APIRouter()
logger = logging.getLogger(__name__)


class ModelLoadResponse(BaseModel):
    message: str
    model_loaded: bool
    model_info: dict = {}


# Importar módulo predict para compartilhar modelo
import src.routers.model.predict as predict_module


@router.post('/load/', response_model=ModelLoadResponse)
async def load_model(file: UploadFile = File(...)):
    """
    Endpoint onde deverá receber o arquivo .pkl do modelo e deixar a API pronta para realizar predições
    """
    try:
        # Verificar se é arquivo .pkl
        if not file.filename.endswith('.pkl'):
            raise HTTPException(
                status_code=400, detail='Arquivo deve ter extensão .pkl'
            )

        # Ler conteúdo do arquivo
        contents = await file.read()

        # Carregar modelo usando pickle
        model = pickle.loads(contents)

        # Atribuir modelo à variável global do módulo predict
        predict_module.loaded_model = model

        # Tentar extrair informações do modelo
        model_info = {'type': type(model).__name__, 'filename': file.filename}

        # Se for um modelo scikit-learn, tentar pegar mais informações
        if hasattr(model, 'feature_importances_'):
            model_info['has_feature_importance'] = True
        if hasattr(model, 'n_features_'):
            model_info['n_features'] = model.n_features_

        logger.info(f'Modelo carregado com sucesso: {file.filename}')

        return ModelLoadResponse(
            message=f'Modelo {file.filename} carregado com sucesso',
            model_loaded=True,
            model_info=model_info,
        )

    except pickle.UnpicklingError:
        logger.error('Erro ao deserializar arquivo pickle')
        raise HTTPException(
            status_code=400, detail='Arquivo pickle inválido ou corrompido'
        )
    except Exception as e:
        logger.error(f'Erro ao carregar modelo: {str(e)}')
        raise HTTPException(
            status_code=500,
            detail=f'Erro interno ao carregar modelo: {str(e)}',
        )


@router.post('/load/default', response_model=ModelLoadResponse)
async def load_default_model():
    """
    Carrega o modelo padrão do projeto (modelo_arvore_decisao.pkl)
    """
    try:
        model_path = 'model/modelo_arvore_decisao.pkl'

        if not os.path.exists(model_path):
            raise HTTPException(
                status_code=404,
                detail=f'Modelo padrão não encontrado em {model_path}',
            )

        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        # Atribuir modelo à variável global do módulo predict
        predict_module.loaded_model = model

        # Tentar extrair informações do modelo
        model_info = {
            'type': type(model).__name__,
            'filename': 'modelo_arvore_decisao.pkl',
            'path': model_path,
        }

        # Se for um modelo scikit-learn, tentar pegar mais informações
        if hasattr(model, 'feature_importances_'):
            model_info['has_feature_importance'] = True
            model_info[
                'feature_importances'
            ] = model.feature_importances_.tolist()
        if hasattr(model, 'n_features_'):
            model_info['n_features'] = model.n_features_

        logger.info('Modelo padrão carregado com sucesso')

        return ModelLoadResponse(
            message='Modelo padrão carregado com sucesso',
            model_loaded=True,
            model_info=model_info,
        )

    except Exception as e:
        logger.error(f'Erro ao carregar modelo padrão: {str(e)}')
        raise HTTPException(
            status_code=500, detail=f'Erro ao carregar modelo padrão: {str(e)}'
        )


@router.get('/load/status')
async def get_model_status():
    """
    Verifica o status do modelo carregado
    """
    try:
        is_loaded = predict_module.loaded_model is not None

        if is_loaded:
            model = predict_module.loaded_model
            model_info = {'type': type(model).__name__, 'loaded': True}

            # Tentar extrair mais informações se disponível
            if hasattr(model, 'feature_importances_'):
                model_info['has_feature_importance'] = True
            if hasattr(model, 'n_features_'):
                model_info['n_features'] = model.n_features_

            return {'status': 'loaded', 'model_info': model_info}
        else:
            return {
                'status': 'not_loaded',
                'message': 'Nenhum modelo carregado',
            }

    except Exception as e:
        logger.error(f'Erro ao verificar status do modelo: {str(e)}')
        raise HTTPException(
            status_code=500,
            detail=f'Erro ao verificar status do modelo: {str(e)}',
        )
