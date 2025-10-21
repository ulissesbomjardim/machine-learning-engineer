# 🏗️ Visão Geral da Arquitetura

Este documento apresenta a arquitetura geral do projeto Machine Learning Engineer Challenge, detalhando a organização dos componentes e fluxos de dados.

## 🎯 Arquitetura de Alto Nível

```mermaid
graph TB
    subgraph "🌐 Client Layer"
        A[Web Browser]
        B[HTTP Client]
        C[Swagger UI]
    end
    
    subgraph "⚡ API Layer"
        D[FastAPI Application]
        E[Routers]
        F[Middlewares]
    end
    
    subgraph "🧠 Business Layer"
        G[ML Services]
        H[Prediction Logic]
        I[Model Management]
    end
    
    subgraph "🗄️ Data Layer"
        J[Database Service]
        K[Model Storage]
        L[Data Processing]
    end
    
    subgraph "🤖 ML Pipeline"
        M[Feature Engineering]
        N[Model Training]
        O[Model Evaluation]
    end
    
    A --> D
    B --> D
    C --> D
    D --> E
    E --> G
    G --> H
    H --> I
    I --> K
    G --> J
    J --> L
    M --> N
    N --> O
    O --> K

    style A fill:#e3f2fd
    style D fill:#e8f5e8
    style G fill:#fff3e0
    style J fill:#f3e5f5
    style M fill:#fce4ec
```

## 📂 Organização do Código

### 🏛️ Estrutura Hierárquica

```
machine-learning-engineer/
├── 🚀 src/                          # Código fonte principal
│   ├── routers/                     # 🔗 Camada de API/Rotas
│   │   ├── main.py                  # 📡 Aplicação FastAPI principal
│   │   └── model/                   # 🤖 Endpoints de Machine Learning
│   │       ├── predict.py           # 🎯 Predições
│   │       ├── load.py              # 📥 Carregamento de modelos
│   │       └── history.py           # 📊 Histórico de predições
│   └── services/                    # ⚙️ Camada de Serviços/Negócio
│       └── database.py              # 🗄️ Gerenciamento de dados
├── 🧪 tests/                       # Testes automatizados
├── 📊 data/                        # Datasets e dados processados
├── 🤖 model/                       # Modelos treinados e artefatos
├── 📓 notebook/                    # Jupyter Notebooks (EDA/Experimentos)
├── 📖 docs/                        # Documentação MkDocs
└── 🐳 docker/                      # Configurações de containers
```

### 🎨 Padrões Arquiteturais

| **Padrão** | **Implementação** | **Benefícios** |
|------------|------------------|----------------|
| **Layered Architecture** | API → Business → Data | Separação de responsabilidades |
| **Repository Pattern** | `database.py` | Abstração de dados |
| **Dependency Injection** | FastAPI dependencies | Testabilidade |
| **Factory Pattern** | Model loading | Flexibilidade de modelos |

## ⚡ Camada de API (FastAPI)

### 🔗 Estrutura de Rotas

```mermaid
graph TD
    A[FastAPI App] --> B[Main Router]
    B --> C[Health Routes]
    B --> D[Model Routes]
    B --> E[User Routes]
    
    D --> F[Predict Endpoint]
    D --> G[Load Endpoint]
    D --> H[History Endpoint]
    
    style A fill:#e8f5e8
    style B fill:#e3f2fd
    style F fill:#fff3e0
    style G fill:#fff3e0
    style H fill:#fff3e0
```

**Responsabilidades:**
- 🌐 **Exposição de endpoints** REST
- 🔒 **Validação** de requests (Pydantic)
- 📋 **Serialização** de responses
- ⚠️ **Tratamento** de erros
- 📚 **Documentação** automática (Swagger)

**Tecnologias:**
- `FastAPI` - Framework web moderno
- `Pydantic` - Validação de dados
- `Uvicorn` - ASGI server

### 📡 Endpoints Principais

```python
# src/routers/main.py
@app.get("/")                    # ℹ️ Informações da API
@app.get("/health")              # 💚 Health check
@app.get("/docs")                # 📚 Documentação Swagger

# src/routers/model/predict.py
@router.post("/model/predict")   # 🎯 Predições ML

# src/routers/model/load.py
@router.get("/model/load/default")   # 📥 Carregar modelo padrão
@router.post("/model/load/")         # 📤 Upload modelo

# src/routers/model/history.py
@router.get("/model/history/")   # 📊 Histórico predições
```

## 🧠 Camada de Negócio

### 🤖 Machine Learning Services

```mermaid
graph LR
    A[Request] --> B[Data Validation]
    B --> C[Feature Engineering]
    C --> D[Model Prediction]
    D --> E[Result Processing]
    E --> F[Response]
    
    G[Model Storage] --> D
    D --> H[History Storage]
    
    style A fill:#e3f2fd
    style D fill:#fff3e0
    style F fill:#e8f5e8
    style G fill:#f3e5f5
    style H fill:#fce4ec
```

**Componentes:**
- 🎯 **Prediction Engine** - Core de predição
- 🔧 **Feature Engineering** - Transformação de dados
- 📥 **Model Loader** - Carregamento dinâmico
- 📊 **History Manager** - Gestão de histórico
- ✅ **Validation Layer** - Validação de dados

### 🔄 Fluxo de Predição

```python
# Fluxo típico de predição
1. 📥 Receber dados do cliente
2. ✅ Validar entrada (Pydantic)
3. 🔧 Aplicar feature engineering
4. 🤖 Executar predição no modelo
5. 📊 Salvar no histórico
6. 📤 Retornar resultado
```

## 🗄️ Camada de Dados

### 💾 Gerenciamento de Dados

```mermaid
graph TB
    subgraph "🗄️ Data Storage"
        A[MongoDB/MockDB]
        B[Model Files]
        C[CSV Datasets]
    end
    
    subgraph "⚙️ Data Services"
        D[Database Service]
        E[Model Repository]
        F[Data Processors]
    end
    
    subgraph "📊 Data Operations"
        G[CRUD Operations]
        H[Model Persistence]
        I[Data Validation]
    end
    
    A --> D
    B --> E
    C --> F
    D --> G
    E --> H
    F --> I

    style A fill:#f3e5f5
    style D fill:#e8f5e8
    style G fill:#fff3e0
```

**Responsabilidades:**
- 💾 **Persistência** de predições
- 🔄 **CRUD operations** para histórico
- 📁 **Gerenciamento** de modelos
- 🔍 **Queries** otimizadas
- 🛡️ **Backup** e recuperação

**Tecnologias:**
- `MongoDB` / `mongomock` - Banco NoSQL
- `pandas` - Manipulação de dados
- `pickle` / `joblib` - Serialização de modelos

## 🤖 Pipeline de Machine Learning

### 🔄 Fluxo do ML Pipeline

```mermaid
graph LR
    subgraph "📊 Data Processing"
        A[Raw Data] --> B[Cleaning]
        B --> C[Feature Engineering]
    end
    
    subgraph "🎯 Model Development"
        C --> D[Model Training]
        D --> E[Validation]
        E --> F[Hyperparameter Tuning]
    end
    
    subgraph "🚀 Model Deployment"
        F --> G[Model Persistence]
        G --> H[API Integration]
        H --> I[Production Serving]
    end
    
    subgraph "📈 Monitoring"
        I --> J[Performance Tracking]
        J --> K[Model Retraining]
        K --> D
    end

    style A fill:#e3f2fd
    style D fill:#fff3e0
    style G fill:#e8f5e8
    style J fill:#fce4ec
```

### 🧩 Componentes do Pipeline

| **Componente** | **Responsabilidade** | **Localização** |
|----------------|---------------------|-----------------|
| 🔧 **Data Preprocessing** | Limpeza e transformação | `notebook/Transform.ipynb` |
| 📊 **Feature Engineering** | Criação de features | `notebook/Model.ipynb` |
| 🤖 **Model Training** | Treinamento de algoritmos | `notebook/Model.ipynb` |
| 📈 **Model Evaluation** | Métricas e validação | `notebook/Model.ipynb` |
| 💾 **Model Persistence** | Salvamento de modelos | `model/` directory |
| ⚡ **Model Serving** | API de predição | `src/routers/model/` |

## 🔄 Fluxos de Dados

### 📊 Fluxo de Treinamento

```mermaid
sequenceDiagram
    participant D as 📊 Data Source
    participant N as 📓 Notebook
    participant M as 🤖 Model
    participant S as 💾 Storage
    
    D->>N: Load raw data
    N->>N: EDA & preprocessing
    N->>N: Feature engineering
    N->>M: Train model
    M->>N: Return trained model
    N->>S: Save model artifact
    Note over S: modelo_arvore_decisao.pkl
```

### 🎯 Fluxo de Predição

```mermaid
sequenceDiagram
    participant C as 👤 Client
    participant A as ⚡ API
    participant S as ⚙️ Service
    participant M as 🤖 Model
    participant D as 🗄️ Database
    
    C->>A: POST /model/predict
    A->>S: Validate & process
    S->>M: Execute prediction
    M->>S: Return prediction
    S->>D: Save to history
    S->>A: Format response
    A->>C: Return JSON result
```

## 🛡️ Princípios Arquiteturais

### 🎯 Design Principles

| **Princípio** | **Implementação** | **Benefício** |
|---------------|------------------|---------------|
| **Separation of Concerns** | Camadas distintas | Manutenibilidade |
| **Single Responsibility** | Classes focadas | Testabilidade |
| **Dependency Inversion** | Interfaces abstratas | Flexibilidade |
| **Don't Repeat Yourself** | Utilitários compartilhados | Consistência |
| **Keep It Simple** | Soluções diretas | Compreensibilidade |

### 🚀 Escalabilidade

**Estratégias de Scaling:**
- 🔄 **Horizontal scaling** via containers
- ⚡ **Load balancing** com múltiplos workers
- 💾 **Database sharding** para grande volume
- 🎯 **Model versioning** para A/B testing
- 📊 **Caching** de predições frequentes

### 🛡️ Confiabilidade

**Garantias de Qualidade:**
- 🧪 **Testes automatizados** (>95% coverage)
- 🔍 **Validação** rigorosa de entrada
- ⚠️ **Error handling** gracioso
- 📊 **Logging** estruturado
- 🔄 **Health checks** periódicos

## 🔧 Configuração e Deploy

### 🐳 Containerização

```mermaid
graph TB
    subgraph "🏗️ Build Stage"
        A[Base Image] --> B[Dependencies]
        B --> C[Source Code]
    end
    
    subgraph "🚀 Runtime Stage"
        C --> D[Production Image]
        D --> E[Container Instance]
    end
    
    subgraph "🔄 Orchestration"
        E --> F[Docker Compose]
        F --> G[Load Balancer]
        G --> H[Multiple Containers]
    end

    style A fill:#e3f2fd
    style D fill:#e8f5e8
    style F fill:#fff3e0
```

**Benefícios:**
- 🔄 **Reprodutibilidade** entre ambientes
- ⚡ **Deploy rápido** e consistente
- 🎯 **Isolamento** de dependências
- 📊 **Escalabilidade** horizontal

## 📚 Próximos Passos

Para entender melhor a arquitetura:

1. 🧩 [Componentes Detalhados](components.md)
2. 🤖 [Pipeline de ML](ml-pipeline.md)
3. ⚡ [API Reference](../api/endpoints.md)
4. 🧪 [Testes de Arquitetura](../tests/integration.md)

## 📞 Suporte Técnico

- 🏗️ [Discussões de Arquitetura](https://github.com/ulissesbomjardim/machine_learning_engineer/discussions)
- 🐛 [Issues Técnicas](https://github.com/ulissesbomjardim/machine_learning_engineer/issues)
- 📧 [Contato Direto](mailto:ulisses.bomjardim@gmail.com)