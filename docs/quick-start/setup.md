# ⚙️ Setup do Projeto

Este guia detalha como configurar o ambiente de desenvolvimento para o projeto Machine Learning Engineer Challenge.

## 📥 Clone do Repositório

### 🌐 Opções de Clone

=== "SSH (Recomendado)"
    ```bash
    git clone git@github.com:ulissesbomjardim/machine_learning_engineer.git
    cd machine_learning_engineer
    ```

=== "HTTPS"
    ```bash
    git clone https://github.com/ulissesbomjardim/machine_learning_engineer.git
    cd machine_learning_engineer
    ```

=== "GitHub CLI"
    ```bash
    gh repo clone ulissesbomjardim/machine_learning_engineer
    cd machine_learning_engineer
    ```

## 🐍 Configuração do Python

### 📌 Versão Específica

O projeto utiliza Python 3.12.7 e já possui arquivo `.python-version` configurado.

```bash
# Verificar se a versão está correta
python --version
# Esperado: Python 3.12.7

# Se usando pyenv, ele detectará automaticamente
# Caso contrário, definir manualmente:
pyenv local 3.12.7  # se usando pyenv
```

### 🔧 Configuração do Poetry

```bash
# 1. Configurar Poetry para usar Python 3.12.7
poetry env use 3.12.7

# 2. Verificar configuração
poetry env info
```

**Saída esperada:**
```
Virtualenv
Python:         3.12.7
Implementation: CPython
Path:           .../.venv
Valid:          True
```

## 📦 Instalação de Dependências

### 🚀 Instalação Completa

```bash
# Instalar todas as dependências (produção + desenvolvimento)
poetry install

# Ou apenas produção (para deploy)
poetry install --without dev
```

### 📋 O que é instalado:

**Dependências de Produção:**
```toml
python = ">=3.12.0,<4.0"
fastapi = "^0.104.1"
uvicorn = "^0.24.0"
pandas = "^2.1.4"
scikit-learn = "^1.3.2"
pydantic = "^2.5.1"
python-multipart = "^0.0.6"
```

**Dependências de Desenvolvimento:**
```toml
pytest = "^7.4.3"
pytest-cov = "^4.1.0"
black = "^23.11.0"
isort = "^5.12.0"
ruff = "^0.1.7"
httpx = "^0.25.2"
mkdocs = "^1.5.3"
mkdocs-material = "^9.4.10"
```

## 🚀 Ativação do Ambiente

### 💡 Formas de Ativar

=== "Poetry Shell (Recomendado)"
    ```bash
    # Ativar ambiente virtual Poetry
    poetry shell
    
    # Agora todos os comandos usarão o ambiente virtual
    python --version
    pip list
    ```

=== "Poetry Run"
    ```bash
    # Executar comandos no ambiente virtual sem ativar
    poetry run python --version
    poetry run pytest
    poetry run uvicorn src.routers.main:app
    ```

=== "Ativação Manual"
    
    **Windows PowerShell:**
    ```powershell
    $path = poetry env info --path
    & "$path\Scripts\Activate.ps1"
    ```
    
    **Windows CMD:**
    ```cmd
    # Obter caminho
    poetry env info --path
    # Usar o caminho retornado
    C:\Users\...\venv\Scripts\activate.bat
    ```
    
    **Linux/macOS:**
    ```bash
    source $(poetry env info --path)/bin/activate
    ```

## 📊 Configuração dos Dados

### 📁 Estrutura de Dados

```bash
# Verificar se os dados estão presentes
ls data/input/
# Esperado: voos.json, airport_database/

ls data/output/
# Esperado: diretórios para outputs processados
```

### 📥 Download dos Dados (se necessário)

Se os dados não estiverem presentes no repositório:

```bash
# Criar estrutura de diretórios
mkdir -p data/input data/output model

# Os dados geralmente estão incluídos no repositório
# Se não estiverem, verificar instruções específicas
```

## 🎯 Tasks Disponíveis

O projeto usa `taskipy` para automatizar tarefas comuns. Veja as tasks disponíveis:

```bash
# Visualizar todas as tasks
poetry run task --list
```

### 🧪 Tasks Principais

```bash
# Executar testes
poetry run task test

# Executar testes com coverage
poetry run task test-cov

# Formatação de código
poetry run task format

# Linting
poetry run task lint

# Executar API
poetry run task api

# Gerar documentação
poetry run task docs
```

## ✅ Verificação da Configuração

### 🔍 Checklist de Verificação

Execute estes comandos para validar a configuração:

```bash
# 1. Verificar Poetry
poetry env info

# 2. Verificar Python no ambiente virtual
poetry run python --version

# 3. Verificar dependências
poetry run pip list

# 4. Executar teste rápido
poetry run python -c "import fastapi, pandas, sklearn; print('✅ Dependências OK!')"

# 5. Verificar estrutura do projeto
ls src/ data/ tests/ docs/
```

### 📊 Verificação de Importações

```bash
# Testar importações principais
poetry run python -c "
import sys
print(f'Python: {sys.version}')

import fastapi
print(f'FastAPI: {fastapi.__version__}')

import pandas as pd
print(f'Pandas: {pd.__version__}')

import sklearn
print(f'Scikit-learn: {sklearn.__version__}')

print('✅ Todas as dependências funcionando!')
"
```

## 🔧 Configuração de Desenvolvimento

### 🎨 Configuração do VS Code

Se estiver usando VS Code, instale as extensões recomendadas:

```json
// .vscode/extensions.json (já incluído no projeto)
{
  "recommendations": [
    "ms-python.python",
    "ms-python.black-formatter",
    "ms-python.isort",
    "charliermarsh.ruff",
    "ms-toolsai.jupyter"
  ]
}
```

### ⚙️ Configuração do Python Interpreter

1. Abrir VS Code no diretório do projeto
2. `Ctrl+Shift+P` → "Python: Select Interpreter"
3. Escolher o ambiente Poetry: `./venv/bin/python` ou similar

### 🐳 Configuração Docker (Opcional)

Se preferir usar Docker:

```bash
# Build da imagem
docker build -t ml-engineer-api .

# Ou usar docker-compose
docker-compose build

# Verificar se funcionou
docker-compose up --build
```

## 🚨 Problemas Comuns

### ❌ Environment não reconhecido

```bash
# Remover e recriar environment
poetry env remove python
poetry env use 3.12.7
poetry install
```

### ❌ Conflitos de dependências

```bash
# Limpar cache do Poetry
poetry cache clear pypi --all

# Reinstalar dependências
poetry install
```

### ❌ ImportError ao executar código

```bash
# Verificar se está no ambiente correto
poetry env info

# Ativar ambiente
poetry shell

# Ou usar poetry run
poetry run python seu_script.py
```

### ❌ Taskipy não funciona

```bash
# Verificar se taskipy está instalado
poetry run pip show taskipy

# Reinstalar se necessário
poetry install
```

## 📚 Próximos Passos

Após configurar o projeto:

1. ✅ [Executar o Projeto](running.md)
2. ✅ [Executar Testes](../tests/running-tests.md)
3. ✅ [Explorar Notebooks](../notebooks/eda.md)
4. ✅ [API Documentation](../api/endpoints.md)

## 📞 Suporte

Se encontrar problemas na configuração:

- 🔧 [Troubleshooting](../dev/troubleshooting.md)
- 🐛 [Issues](https://github.com/ulissesbomjardim/machine_learning_engineer/issues)
- 📧 [Email](mailto:ulisses.bomjardim@gmail.com)