# Documentação Automática

Este repositório está configurado para atualizar automaticamente a documentação sempre que houver mudanças na branch `main`.

## Como Funciona

### Estrutura
- **Branch `main`**: Contém o código fonte e documentação em markdown (pasta `docs/`)
- **Branch `docs`**: Contém apenas o site estático gerado pelo MkDocs

### Automação
O GitHub Actions foi configurado para:
1. Monitorar mudanças na pasta `docs/`, arquivo `mkdocs.yml` ou no próprio workflow
2. Executar o build do MkDocs automaticamente
3. Atualizar a branch `docs` com o novo conteúdo

### Workflow Manual
Você também pode executar o build manualmente:

1. **Localmente:**
   ```bash
   poetry install --with doc
   poetry run mkdocs build
   ```

2. **No GitHub:**
   - Acesse a aba "Actions"
   - Selecione "Deploy Documentation"
   - Clique em "Run workflow"

### Visualização
- **Desenvolvimento**: `poetry run mkdocs serve` (http://localhost:8000)
- **Produção**: A branch `docs` pode ser usada com GitHub Pages ou qualquer servidor estático

### Estrutura da Documentação
```
docs/
├── index.md                    # Página inicial
├── api/                        # Documentação da API
├── architecture/               # Arquitetura do sistema
├── ml/                         # Machine Learning
├── docker/                     # Docker e deployment
├── tests/                      # Testes
├── notebooks/                  # Jupyter notebooks
├── dev/                        # Desenvolvimento
└── quick-start/               # Início rápido
```

### Configuração
A configuração do MkDocs está em `mkdocs.yml` na raiz do projeto.