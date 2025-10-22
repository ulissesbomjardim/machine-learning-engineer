import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

try:
    # Tentar usar MongoDB real primeiro
    from pymongo import MongoClient as PyMongoClient

    MONGO_AVAILABLE = True
except ImportError:
    # Fallback para mongomock se pymongo não estiver disponível
    from mongomock.mongo_client import MongoClient as PyMongoClient

    MONGO_AVAILABLE = False
    logger.warning(
        'PyMongo não encontrado, usando mongomock para desenvolvimento'
    )


class InMemoryDatabase:
    _instance = None
    _client = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Inicializa a conexão com MongoDB"""
        try:
            # Configurações do MongoDB - primeiro tenta usar MONGODB_URI completa
            mongodb_uri = os.getenv('MONGODB_URI')
            if mongodb_uri:
                # Se temos uma URI completa, usar ela
                mongo_host = None  # Será usado na URI
                mongo_port = None
                mongo_db = os.getenv('DATABASE_NAME', 'picpay_ml')
            else:
                # Configurações tradicionais
                mongo_host = os.getenv(
                    'MONGO_HOST', 'mongo'
                )  # 'mongo' é o hostname no Docker
                mongo_port = int(os.getenv('MONGO_PORT', '27017'))
                mongo_db = os.getenv('DATABASE_NAME', 'picpay_ml')

            if MONGO_AVAILABLE:
                # Tentar conectar com MongoDB real
                try:
                    if mongodb_uri:
                        # Usar URI completa
                        self._client = PyMongoClient(
                            mongodb_uri,
                            serverSelectionTimeoutMS=5000,  # 5 segundos timeout
                        )
                        logger.info(
                            f'Conectando ao MongoDB via URI: {mongodb_uri}'
                        )
                    else:
                        # Usar configurações tradicionais
                        self._client = PyMongoClient(
                            host=mongo_host,
                            port=mongo_port,
                            serverSelectionTimeoutMS=5000,  # 5 segundos timeout
                        )
                        logger.info(
                            f'Conectando ao MongoDB: {mongo_host}:{mongo_port}'
                        )

                    # Testar conexão
                    self._client.admin.command('ping')
                    self._database = self._client.get_database(mongo_db)
                    logger.info(
                        f'Conectado ao MongoDB com sucesso - Database: {mongo_db}'
                    )
                except Exception as e:
                    logger.warning(f'Falha ao conectar com MongoDB real: {e}')
                    # Fallback para mongomock
                    self._client = PyMongoClient()
                    self._database = self._client.get_database(mongo_db)
                    logger.info('Usando mongomock como fallback')
            else:
                # Usar mongomock diretamente
                self._client = PyMongoClient()
                self._database = self._client.get_database(mongo_db)
                logger.info('Usando mongomock para desenvolvimento')

        except Exception as e:
            logger.error(f'Erro ao inicializar database: {e}')
            # Último fallback - usar mongomock
            from mongomock.mongo_client import MongoClient as MockClient

            self._client = MockClient()
            self._database = self._client.get_database('flight_delay_db')

    def get_database(self):
        """Retorna a instância do database"""
        return self._database

    def get_collection(self, collection_name: str):
        """Retorna uma collection específica"""
        return self._database.get_collection(collection_name)

    def close(self):
        """Fecha a conexão com o database"""
        if self._client:
            self._client.close()
            logger.info('Conexão com database fechada')

    def health_check(self) -> dict:
        """Verifica a saúde da conexão com o database"""
        try:
            # Ping no database
            if MONGO_AVAILABLE:
                self._client.admin.command('ping')

            # Testar operação básica
            test_collection = self.get_collection('health_test')
            test_collection.find_one()

            return {
                'status': 'healthy',
                'type': 'MongoDB' if MONGO_AVAILABLE else 'MongoMock',
                'connected': True,
            }
        except Exception as e:
            logger.error(f'Database health check falhou: {e}')
            return {
                'status': 'unhealthy',
                'type': 'MongoDB' if MONGO_AVAILABLE else 'MongoMock',
                'connected': False,
                'error': str(e),
            }
