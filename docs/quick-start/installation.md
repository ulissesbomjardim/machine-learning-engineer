# 📥 Instalação

Este guia apresenta os pré-requisitos e instalação das ferramentas necessárias para executar o projeto Machine Learning Engineer Challenge.

## 📋 Pré-requisitos

### 🐍 Python 3.12.7

O projeto foi desenvolvido e testado com Python 3.12.7. É altamente recomendado usar essa versão específica para evitar problemas de compatibilidade.

#### Windows

**Opção 1: Download Oficial**
```bash
# Baixar do site oficial: https://www.python.org/downloads/
# Instalar Python 3.12.7 e marcar "Add Python to PATH"
```

**Opção 2: Pyenv-win (Recomendado)**
```powershell
# Instalar Pyenv-win
git clone https://github.com/pyenv-win/pyenv-win.git %USERPROFILE%\.pyenv

# Adicionar ao PATH do sistema
# Adicionar estas variáveis ao PATH:
# %USERPROFILE%\.pyenv\pyenv-win\bin
# %USERPROFILE%\.pyenv\pyenv-win\shims

# Instalar Python 3.12.7
pyenv install 3.12.7
pyenv global 3.12.7
```

#### Linux/macOS

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3.12 python3.12-pip python3.12-venv

# macOS (Homebrew)
brew install python@3.12

# Ou usar pyenv (recomendado)
curl https://pyenv.run | bash
pyenv install 3.12.7
pyenv global 3.12.7
```

### 📦 Poetry

Poetry é o gerenciador de dependências utilizado no projeto.

#### Instalação

**Windows (PowerShell)**
```powershell
# Opção 1: pip
pip install poetry

# Opção 2: Instalador oficial
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -

# Opção 3: pipx (recomendado)
pip install pipx
pipx install poetry
```

**Linux/macOS**
```bash
# Instalador oficial (recomendado)
curl -sSL https://install.python-poetry.org | python3 -

# Ou via pip
pip install poetry

# Ou via pipx
pip install pipx
pipx install poetry
```

#### Configuração do Poetry

```bash
# Verificar instalação
poetry --version

# Configurar para criar venv na pasta do projeto (opcional)
poetry config virtualenvs.in-project true

# Verificar configuração
poetry config --list
```

### 💻 Git

#### Windows

```bash
# Baixar do site oficial: https://git-scm.com/download/win
# Ou via chocolatey
choco install git

# Ou via winget
winget install Git.Git
```

#### Linux/macOS

```bash
# Ubuntu/Debian
sudo apt install git

# macOS
brew install git
# Ou usar Xcode command line tools
xcode-select --install
```

### 🐳 Docker (Opcional)

Para executar o projeto com containers Docker.

#### Windows

```bash
# Docker Desktop: https://desktop.docker.com/win/main/amd64/Docker%20Desktop%20Installer.exe
# Ou via chocolatey
choco install docker-desktop

# Ou via winget
winget install Docker.DockerDesktop
```

#### Linux

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install docker.io docker-compose

# Iniciar serviço
sudo systemctl start docker
sudo systemctl enable docker

# Adicionar usuário ao grupo docker
sudo usermod -aG docker $USER
```

#### macOS

```bash
# Docker Desktop: https://desktop.docker.com/mac/main/amd64/Docker.dmg
# Ou via homebrew
brew install --cask docker
```

## ✅ Verificação da Instalação

Execute estes comandos para verificar se tudo foi instalado corretamente:

```bash
# Verificar Python
python --version
# Esperado: Python 3.12.7

# Verificar Poetry
poetry --version
# Esperado: Poetry (version 1.7.1 ou superior)

# Verificar Git
git --version
# Esperado: git version 2.x.x

# Verificar Docker (opcional)
docker --version
docker-compose --version
```

## 🔧 Configuração Adicional

### 🎯 Configuração do Poetry

```bash
# Configurar para usar Python 3.12.7
poetry env use python3.12

# Ou especificar caminho completo (se necessário)
poetry env use /usr/bin/python3.12  # Linux/macOS
poetry env use C:\Python312\python.exe  # Windows
```

### 📝 Configuração do Git (Primeira vez)

```bash
# Configurar nome e email
git config --global user.name "Seu Nome"
git config --global user.email "seuemail@exemplo.com"

# Configurar editor padrão (opcional)
git config --global core.editor "code --wait"  # VS Code
```

### 🛡️ Configuração SSH (GitHub)

```bash
# Gerar chave SSH
ssh-keygen -t ed25519 -C "seuemail@exemplo.com"

# Adicionar à SSH agent
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519

# Copiar chave pública para GitHub
cat ~/.ssh/id_ed25519.pub
# Adicionar em: GitHub > Settings > SSH and GPG keys
```

## 🚨 Problemas Comuns

### ❌ Poetry não encontrado

```bash
# Verificar PATH
echo $PATH  # Linux/macOS
echo $env:PATH  # Windows PowerShell

# Reiniciar terminal ou recarregar perfil
source ~/.bashrc  # Linux
source ~/.zshrc   # macOS
# Reiniciar PowerShell no Windows
```

### ❌ Python versão incorreta

```bash
# Verificar versões disponíveis
python --version
python3 --version
python3.12 --version

# Usar pyenv para trocar versão
pyenv versions
pyenv local 3.12.7
```

### ❌ Problemas de permissão (Linux/macOS)

```bash
# Usar pyenv em vez de sudo
# NUNCA usar: sudo pip install

# Se necessário, instalar via usuário
pip install --user poetry
```

## 📚 Próximos Passos

Após instalar todas as dependências:

1. ✅ [Configuração do Projeto](setup.md)
2. ✅ [Executando o Projeto](running.md)
3. ✅ [Testes](../tests/running-tests.md)

## 📞 Suporte

Se encontrar problemas durante a instalação:

- 🐛 [Abrir Issue](https://github.com/ulissesbomjardim/machine_learning_engineer/issues)
- 📧 [Email](mailto:ulisses.bomjardim@gmail.com)
- 📖 [Troubleshooting](../dev/troubleshooting.md)