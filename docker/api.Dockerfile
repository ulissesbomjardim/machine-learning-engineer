FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Instalar dependências do sistema necessárias para o MongoDB e outras bibliotecas
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY docker/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copiar código fonte
COPY src/ src/

# Copiar modelo treinado
COPY model/modelo_arvore_decisao.pkl model/modelo_arvore_decisao.pkl

# Criar diretório para dados se necessário
RUN mkdir -p data/input data/output

EXPOSE 8080

CMD ["uvicorn", "src.routers.main:app", "--host", "0.0.0.0", "--port", "8080", "--reload"]
