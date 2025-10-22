# 📊 Análise de Cobertura

Guia completo para análise e monitoramento da cobertura de testes, incluindo métricas detalhadas, relatórios automáticos e estratégias para melhorar a qualidade do código.

## 🎯 Visão Geral

A análise de cobertura é fundamental para garantir que o código esteja adequadamente testado. Esta seção documenta como monitorar, analisar e melhorar a cobertura de testes do projeto.

## 📊 Métricas de Cobertura Atual

### 📋 Resumo Geral

```
Total Coverage: 87.3%
Lines Covered: 1,247 / 1,428
Branches Covered: 234 / 267 (87.6%)
Functions Covered: 156 / 172 (90.7%)
```

### 📁 Cobertura por Módulo

| **Módulo** | **Linhas** | **Cobertura** | **Status** | **Prioridade** |
|------------|------------|---------------|------------|----------------|
| `src/routers/` | 245/267 | 91.8% | ✅ Excelente | Baixa |
| `src/services/` | 378/412 | 91.7% | ✅ Excelente | Baixa |  
| `src/ml/` | 456/523 | 87.2% | ⚠️ Boa | Média |
| `src/utils/` | 134/156 | 85.9% | ⚠️ Boa | Média |
| `src/database/` | 89/98 | 90.8% | ✅ Excelente | Baixa |
| `src/config/` | 23/28 | 82.1% | ⚠️ Aceitável | Alta |

## 🔧 Configuração do Coverage

### 📋 .coveragerc

```ini
[run]
source = src
omit = 
    */tests/*
    */test_*
    */__pycache__/*
    */venv/*
    */.venv/*
    */migrations/*
    */settings.py
    */manage.py
    */wsgi.py
    */asgi.py

branch = true
parallel = true

[report]
# Regras para relatório
exclude_lines =
    pragma: no cover
    def __repr__
    raise AssertionError
    raise NotImplementedError
    if __name__ == .__main__.:
    @abstract
    @abstractmethod
    # Type checking
    if TYPE_CHECKING:
    if typing.TYPE_CHECKING:

precision = 2
show_missing = true
skip_covered = false
sort = Cover

[html]
directory = htmlcov
title = Flight Delay Prediction - Coverage Report
show_contexts = true
skip_covered = false
skip_empty = false

[xml]
output = coverage.xml

[json]
output = coverage.json
show_contexts = true
```

### ⚙️ pytest.ini Coverage Config

```ini
[tool:pytest]
addopts = 
    --strict-markers
    --strict-config
    --cov=src
    --cov-report=term-missing
    --cov-report=html:htmlcov
    --cov-report=xml:coverage.xml
    --cov-report=json:coverage.json
    --cov-fail-under=85
    --cov-branch
    --cov-context=test

markers =
    unit: Unit tests
    integration: Integration tests
    slow: Slow running tests
    api: API endpoint tests
    ml: Machine learning tests
    database: Database tests

testpaths = tests
python_files = test_*.py *_test.py
python_classes = Test* *Tests
python_functions = test_*

filterwarnings =
    ignore::UserWarning
    ignore::DeprecationWarning
```

## 📊 Relatórios de Cobertura

### 🌐 Relatório HTML Interativo

```bash
# Gerar relatório HTML completo
pytest --cov=src --cov-report=html --cov-report=term-missing

# Abrir relatório no navegador
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows
```

### 📋 Relatório de Terminal

```bash
# Relatório detalhado no terminal
pytest --cov=src --cov-report=term-missing --cov-report=term:skip-covered

# Exemplo de saída:
Name                           Stmts   Miss Branch BrPart  Cover   Missing
--------------------------------------------------------------------------
src/__init__.py                    0      0      0      0   100%
src/routers/__init__.py            1      0      0      0   100%
src/routers/main.py               45      3      8      1    91%   23-25, 67->exit
src/services/database.py          67      5     12      2    89%   45-47, 89-91
src/ml/preprocessor.py           123     18     24      3    82%   156-162, 245-251
--------------------------------------------------------------------------
TOTAL                           1428    156    267     33    87%
```

### 📊 Relatório JSON para Automação

```python
# scripts/analyze_coverage.py
import json
import sys
from pathlib import Path

def analyze_coverage_json():
    """Analisa relatório JSON de cobertura"""
    
    coverage_file = Path("coverage.json")
    
    if not coverage_file.exists():
        print("❌ Arquivo coverage.json não encontrado")
        return False
    
    with open(coverage_file) as f:
        data = json.load(f)
    
    # Estatísticas gerais
    totals = data['totals']
    
    print("📊 ANÁLISE DE COBERTURA")
    print("=" * 50)
    print(f"Cobertura Total: {totals['percent_covered']:.1f}%")
    print(f"Linhas Cobertas: {totals['covered_lines']}/{totals['num_statements']}")
    print(f"Branches Cobertos: {totals['covered_branches']}/{totals['num_branches']}")
    
    # Análise por arquivo
    files = data['files']
    
    # Arquivos com baixa cobertura
    low_coverage_files = []
    
    for filepath, file_data in files.items():
        coverage_percent = file_data['summary']['percent_covered']
        
        if coverage_percent < 80:
            low_coverage_files.append({
                'file': filepath,
                'coverage': coverage_percent,
                'missing_lines': len(file_data['missing_lines'])
            })
    
    if low_coverage_files:
        print(f"\n⚠️ ARQUIVOS COM BAIXA COBERTURA ({len(low_coverage_files)})")
        print("-" * 50)
        
        for file_info in sorted(low_coverage_files, key=lambda x: x['coverage']):
            print(f"{file_info['file']}: {file_info['coverage']:.1f}% "
                  f"({file_info['missing_lines']} linhas não cobertas)")
    
    # Verificar meta de cobertura
    target_coverage = 85.0
    
    if totals['percent_covered'] >= target_coverage:
        print(f"\n✅ Meta de cobertura atingida ({target_coverage}%)")
        return True
    else:
        print(f"\n❌ Meta de cobertura não atingida ({target_coverage}%)")
        print(f"Faltam {target_coverage - totals['percent_covered']:.1f}% para atingir a meta")
        return False

if __name__ == "__main__":
    success = analyze_coverage_json()
    sys.exit(0 if success else 1)
```

## 🎯 Estratégias para Melhorar Cobertura

### 1. 📊 Identificar Gaps de Cobertura

```python
# scripts/find_coverage_gaps.py
import ast
import os
from pathlib import Path

class CoverageGapAnalyzer(ast.NodeVisitor):
    """Analisa código para identificar gaps de cobertura"""
    
    def __init__(self):
        self.functions = []
        self.classes = []
        self.branches = []
        self.current_class = None
    
    def visit_FunctionDef(self, node):
        """Visita definições de função"""
        func_info = {
            'name': node.name,
            'lineno': node.lineno,
            'class': self.current_class,
            'has_docstring': ast.get_docstring(node) is not None,
            'is_private': node.name.startswith('_'),
            'has_decorators': len(node.decorator_list) > 0,
            'complexity': self._calculate_complexity(node)
        }
        
        self.functions.append(func_info)
        self.generic_visit(node)
    
    def visit_ClassDef(self, node):
        """Visita definições de classe"""
        old_class = self.current_class
        self.current_class = node.name
        
        class_info = {
            'name': node.name,
            'lineno': node.lineno,
            'has_docstring': ast.get_docstring(node) is not None,
            'methods': [],
            'is_exception': any(base.id == 'Exception' for base in node.bases if isinstance(base, ast.Name))
        }
        
        self.classes.append(class_info)
        self.generic_visit(node)
        
        self.current_class = old_class
    
    def visit_If(self, node):
        """Visita estruturas condicionais"""
        self.branches.append({
            'type': 'if',
            'lineno': node.lineno,
            'has_else': node.orelse is not None
        })
        self.generic_visit(node)
    
    def visit_Try(self, node):
        """Visita blocos try/except"""
        self.branches.append({
            'type': 'try',
            'lineno': node.lineno,
            'handlers': len(node.handlers),
            'has_finally': node.finalbody is not None
        })
        self.generic_visit(node)
    
    def _calculate_complexity(self, node):
        """Calcula complexidade ciclomática básica"""
        complexity = 1  # Base
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        
        return complexity

def analyze_file_coverage_gaps(filepath):
    """Analisa gaps de cobertura em um arquivo"""
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return None
    
    analyzer = CoverageGapAnalyzer()
    analyzer.visit(tree)
    
    return {
        'file': str(filepath),
        'functions': analyzer.functions,
        'classes': analyzer.classes,
        'branches': analyzer.branches,
        'total_functions': len(analyzer.functions),
        'total_branches': len(analyzer.branches)
    }

def find_untested_functions():
    """Encontra funções potencialmente não testadas"""
    
    src_path = Path("src")
    gaps = []
    
    for py_file in src_path.rglob("*.py"):
        if py_file.name.startswith("test_") or "/tests/" in str(py_file):
            continue
            
        analysis = analyze_file_coverage_gaps(py_file)
        if analysis:
            gaps.append(analysis)
    
    print("🔍 ANÁLISE DE GAPS DE COBERTURA")
    print("=" * 50)
    
    for file_analysis in gaps:
        print(f"\n📁 {file_analysis['file']}")
        
        # Funções complexas
        complex_functions = [f for f in file_analysis['functions'] if f['complexity'] > 5]
        if complex_functions:
            print(f"  ⚠️ Funções complexas ({len(complex_functions)}):")
            for func in complex_functions:
                print(f"    - {func['name']} (linhas {func['lineno']}, complexidade {func['complexity']})")
        
        # Funções sem docstring
        undocumented = [f for f in file_analysis['functions'] if not f['has_docstring'] and not f['is_private']]
        if undocumented:
            print(f"  📝 Funções sem docstring ({len(undocumented)}):")
            for func in undocumented:
                print(f"    - {func['name']} (linha {func['lineno']})")

if __name__ == "__main__":
    find_untested_functions()
```

### 2. 🧪 Gerar Testes Automaticamente

```python
# scripts/generate_tests.py
import ast
import os
from pathlib import Path
from textwrap import dedent

class TestGenerator:
    """Gerador automático de esqueletos de teste"""
    
    def __init__(self, source_file):
        self.source_file = Path(source_file)
        self.module_name = self._get_module_name()
        
        with open(source_file, 'r') as f:
            self.source_code = f.read()
        
        self.tree = ast.parse(self.source_code)
        self.functions = []
        self.classes = []
        
        self._analyze_code()
    
    def _get_module_name(self):
        """Extrai nome do módulo do caminho"""
        parts = self.source_file.parts
        if 'src' in parts:
            src_index = parts.index('src')
            module_parts = parts[src_index + 1:]
            return '.'.join(module_parts).replace('.py', '')
        return self.source_file.stem
    
    def _analyze_code(self):
        """Analisa código fonte"""
        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef):
                if not node.name.startswith('_'):  # Apenas funções públicas
                    self.functions.append(self._analyze_function(node))
            elif isinstance(node, ast.ClassDef):
                self.classes.append(self._analyze_class(node))
    
    def _analyze_function(self, node):
        """Analisa uma função"""
        return {
            'name': node.name,
            'args': [arg.arg for arg in node.args.args if arg.arg != 'self'],
            'lineno': node.lineno,
            'docstring': ast.get_docstring(node),
            'returns': self._has_return(node),
            'raises': self._extract_exceptions(node),
            'is_async': isinstance(node, ast.AsyncFunctionDef)
        }
    
    def _analyze_class(self, node):
        """Analisa uma classe"""
        methods = []
        for child in node.body:
            if isinstance(child, ast.FunctionDef) and not child.name.startswith('_'):
                methods.append(self._analyze_function(child))
        
        return {
            'name': node.name,
            'methods': methods,
            'lineno': node.lineno,
            'docstring': ast.get_docstring(node),
            'bases': [base.id for base in node.bases if isinstance(base, ast.Name)]
        }
    
    def _has_return(self, node):
        """Verifica se função tem return"""
        for child in ast.walk(node):
            if isinstance(child, ast.Return) and child.value is not None:
                return True
        return False
    
    def _extract_exceptions(self, node):
        """Extrai exceções que podem ser levantadas"""
        exceptions = []
        for child in ast.walk(node):
            if isinstance(child, ast.Raise) and child.exc:
                if isinstance(child.exc, ast.Name):
                    exceptions.append(child.exc.id)
                elif isinstance(child.exc, ast.Call) and isinstance(child.exc.func, ast.Name):
                    exceptions.append(child.exc.func.id)
        return exceptions
    
    def generate_test_file(self):
        """Gera arquivo de teste"""
        test_content = self._generate_test_content()
        
        # Criar diretório de testes se não existir
        test_dir = Path("tests")
        test_dir.mkdir(exist_ok=True)
        
        # Nome do arquivo de teste
        test_file = test_dir / f"test_{self.source_file.stem}.py"
        
        # Escrever apenas se não existir
        if not test_file.exists():
            with open(test_file, 'w') as f:
                f.write(test_content)
            print(f"✅ Gerado: {test_file}")
        else:
            print(f"⚠️ Já existe: {test_file}")
        
        return test_file
    
    def _generate_test_content(self):
        """Gera conteúdo do arquivo de teste"""
        
        imports = f"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from {self.module_name} import *
"""
        
        # Gerar testes para funções
        function_tests = []
        for func in self.functions:
            function_tests.append(self._generate_function_test(func))
        
        # Gerar testes para classes
        class_tests = []
        for cls in self.classes:
            class_tests.append(self._generate_class_test(cls))
        
        content = dedent(f"""
        '''
        Testes para {self.module_name}
        
        Este arquivo foi gerado automaticamente.
        Revise e complete os testes conforme necessário.
        '''
        {imports.strip()}
        
        
        {''.join(function_tests)}
        
        {''.join(class_tests)}
        """).strip()
        
        return content
    
    def _generate_function_test(self, func):
        """Gera teste para uma função"""
        
        test_name = f"test_{func['name']}"
        
        # Preparar argumentos
        args_str = ", ".join([f"{arg}=Mock()" for arg in func['args']])
        call_str = f"{func['name']}({args_str})" if args_str else f"{func['name']}()"
        
        # Casos de teste básicos
        basic_test = f"""
    def {test_name}_basic(self):
        '''Teste básico para {func['name']}'''
        # Arrange
        # TODO: Configurar dados de teste
        
        # Act
        {'result = ' if func['returns'] else ''}{call_str}
        
        # Assert
        {'assert result is not None' if func['returns'] else '# TODO: Adicionar assertions'}
"""
        
        # Testes para exceções
        exception_tests = ""
        for exc in func['raises']:
            exception_tests += f"""
    def {test_name}_raises_{exc.lower()}(self):
        '''Teste para exceção {exc}'''
        # TODO: Configurar condições que causam {exc}
        
        with pytest.raises({exc}):
            {call_str}
"""
        
        return f"""
class Test{func['name'].title()}:
    '''Testes para a função {func['name']}'''
{basic_test}
{exception_tests}
"""
    
    def _generate_class_test(self, cls):
        """Gera testes para uma classe"""
        
        # Teste de inicialização
        init_test = f"""
class Test{cls['name']}:
    '''Testes para a classe {cls['name']}'''
    
    def test_init(self):
        '''Teste de inicialização'''
        # TODO: Configurar parâmetros de inicialização
        instance = {cls['name']}()
        assert instance is not None
"""
        
        # Testes para métodos
        method_tests = ""
        for method in cls['methods']:
            method_tests += f"""
    def test_{method['name']}(self):
        '''Teste para o método {method['name']}'''
        # Arrange
        instance = {cls['name']}()
        # TODO: Configurar dados de teste
        
        # Act
        {'result = ' if method['returns'] else ''}instance.{method['name']}()
        
        # Assert
        {'assert result is not None' if method['returns'] else '# TODO: Adicionar assertions'}
"""
        
        return init_test + method_tests

def generate_tests_for_file(filepath):
    """Gera testes para um arquivo específico"""
    generator = TestGenerator(filepath)
    return generator.generate_test_file()

def generate_tests_for_project():
    """Gera testes para todo o projeto"""
    src_path = Path("src")
    generated_files = []
    
    for py_file in src_path.rglob("*.py"):
        if py_file.name == "__init__.py":
            continue
            
        try:
            test_file = generate_tests_for_file(py_file)
            generated_files.append(test_file)
        except Exception as e:
            print(f"❌ Erro ao gerar teste para {py_file}: {e}")
    
    print(f"\n📊 Gerados {len(generated_files)} arquivos de teste")
    return generated_files

if __name__ == "__main__":
    generate_tests_for_project()
```

## 📈 Monitoramento Contínuo

### 1. 🔄 GitHub Actions para Coverage

```yaml
# .github/workflows/coverage.yml
name: Coverage Report

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  coverage:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.12'
    
    - name: Install dependencies
      run: |
        pip install poetry
        poetry install
    
    - name: Run tests with coverage
      run: |
        poetry run pytest --cov=src --cov-report=xml --cov-report=html --cov-fail-under=85
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests
        name: codecov-umbrella
        fail_ci_if_error: true
    
    - name: Coverage comment
      uses: py-cov-action/python-coverage-comment-action@v3
      with:
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        MINIMUM_GREEN: 90
        MINIMUM_ORANGE: 80
    
    - name: Upload HTML coverage report
      uses: actions/upload-artifact@v3
      with:
        name: coverage-report
        path: htmlcov/
```

### 2. 📊 Badge de Cobertura

```markdown
<!-- README.md -->
[![Coverage Status](https://codecov.io/gh/username/machine-learning-engineer/branch/main/graph/badge.svg)](https://codecov.io/gh/username/machine-learning-engineer)

[![Coverage](https://img.shields.io/badge/Coverage-87.3%25-yellow.svg)](htmlcov/index.html)
```

## 🎯 Metas de Cobertura

### 📋 Configuração por Módulo

```yaml
# coverage-goals.yml
coverage_goals:
  global:
    minimum: 85%
    target: 90%
  
  modules:
    src/routers/:
      minimum: 90%
      target: 95%
      critical: true
    
    src/services/:
      minimum: 90%
      target: 95%
      critical: true
    
    src/ml/:
      minimum: 80%
      target: 90%
      critical: false
    
    src/utils/:
      minimum: 85%
      target: 90%
      critical: false
    
    src/config/:
      minimum: 70%
      target: 85%
      critical: false

  exceptions:
    # Arquivos excluídos da análise
    exclude:
      - "src/migrations/"
      - "src/settings.py"
      - "src/__main__.py"
    
    # Linhas excluídas
    exclude_lines:
      - "pragma: no cover"
      - "raise NotImplementedError"
      - "if __name__ == .__main__.:"
```

### 🚨 Alertas de Cobertura

```python
# scripts/coverage_alerts.py
import json
import yaml
from pathlib import Path

def check_coverage_goals():
    """Verifica se as metas de cobertura foram atingidas"""
    
    # Carregar configuração
    with open("coverage-goals.yml") as f:
        goals = yaml.safe_load(f)
    
    # Carregar dados de cobertura
    with open("coverage.json") as f:
        coverage_data = json.load(f)
    
    alerts = []
    
    # Verificar cobertura global
    global_coverage = coverage_data['totals']['percent_covered']
    global_minimum = float(goals['coverage_goals']['global']['minimum'].rstrip('%'))
    
    if global_coverage < global_minimum:
        alerts.append({
            'type': 'global',
            'severity': 'error',
            'message': f"Cobertura global ({global_coverage:.1f}%) abaixo do mínimo ({global_minimum}%)"
        })
    
    # Verificar cobertura por módulo
    files = coverage_data['files']
    module_goals = goals['coverage_goals']['modules']
    
    for module_path, module_config in module_goals.items():
        module_files = [f for f in files.keys() if f.startswith(module_path)]
        
        if not module_files:
            continue
        
        # Calcular cobertura do módulo
        total_statements = sum(files[f]['summary']['num_statements'] for f in module_files)
        covered_statements = sum(files[f]['summary']['covered_lines'] for f in module_files)
        
        module_coverage = (covered_statements / total_statements) * 100 if total_statements > 0 else 100
        
        minimum = float(module_config['minimum'].rstrip('%'))
        
        if module_coverage < minimum:
            severity = 'error' if module_config.get('critical', False) else 'warning'
            
            alerts.append({
                'type': 'module',
                'module': module_path,
                'severity': severity,
                'current': module_coverage,
                'minimum': minimum,
                'message': f"Módulo {module_path}: {module_coverage:.1f}% < {minimum}%"
            })
    
    return alerts

def send_coverage_alerts(alerts):
    """Envia alertas de cobertura"""
    
    if not alerts:
        print("✅ Todas as metas de cobertura foram atingidas!")
        return
    
    print("🚨 ALERTAS DE COBERTURA")
    print("=" * 50)
    
    errors = [a for a in alerts if a['severity'] == 'error']
    warnings = [a for a in alerts if a['severity'] == 'warning']
    
    if errors:
        print(f"\n❌ ERROS ({len(errors)}):")
        for alert in errors:
            print(f"  • {alert['message']}")
    
    if warnings:
        print(f"\n⚠️ AVISOS ({len(warnings)}):")
        for alert in warnings:
            print(f"  • {alert['message']}")
    
    # TODO: Integrar com Slack, Teams, etc.
    return len(errors) > 0

if __name__ == "__main__":
    alerts = check_coverage_goals()
    has_errors = send_coverage_alerts(alerts)
    exit(1 if has_errors else 0)
```

## 📊 Relatórios Avançados

### 1. 🕒 Tendências de Cobertura

```python
# scripts/coverage_trends.py
import json
import sqlite3
from datetime import datetime
import matplotlib.pyplot as plt

class CoverageTrendTracker:
    """Rastreia tendências de cobertura ao longo do tempo"""
    
    def __init__(self, db_path="coverage_history.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Inicializa banco de dados"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS coverage_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                commit_hash TEXT,
                total_coverage REAL NOT NULL,
                lines_covered INTEGER NOT NULL,
                lines_total INTEGER NOT NULL,
                branches_covered INTEGER NOT NULL,
                branches_total INTEGER NOT NULL,
                module_data TEXT  -- JSON com dados por módulo
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def record_coverage(self, coverage_file="coverage.json", commit_hash=None):
        """Registra dados de cobertura atual"""
        
        with open(coverage_file) as f:
            data = json.load(f)
        
        totals = data['totals']
        
        # Dados por módulo
        module_data = {}
        for filepath, file_data in data['files'].items():
            module = filepath.split('/')[0] if '/' in filepath else 'root'
            
            if module not in module_data:
                module_data[module] = {
                    'covered_lines': 0,
                    'total_lines': 0,
                    'files': 0
                }
            
            module_data[module]['covered_lines'] += file_data['summary']['covered_lines']
            module_data[module]['total_lines'] += file_data['summary']['num_statements']
            module_data[module]['files'] += 1
        
        # Calcular cobertura por módulo
        for module, stats in module_data.items():
            if stats['total_lines'] > 0:
                stats['coverage'] = (stats['covered_lines'] / stats['total_lines']) * 100
            else:
                stats['coverage'] = 100.0
        
        # Salvar no banco
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO coverage_history 
            (timestamp, commit_hash, total_coverage, lines_covered, lines_total, 
             branches_covered, branches_total, module_data)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            commit_hash,
            totals['percent_covered'],
            totals['covered_lines'],
            totals['num_statements'],
            totals['covered_branches'],
            totals['num_branches'],
            json.dumps(module_data)
        ))
        
        conn.commit()
        conn.close()
        
        print(f"📊 Cobertura registrada: {totals['percent_covered']:.1f}%")
    
    def generate_trend_report(self, days=30):
        """Gera relatório de tendências"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT timestamp, total_coverage, lines_covered, lines_total
            FROM coverage_history
            WHERE datetime(timestamp) >= datetime('now', '-{} days')
            ORDER BY timestamp
        '''.format(days))
        
        results = cursor.fetchall()
        conn.close()
        
        if not results:
            print("Não há dados suficientes para gerar relatório")
            return
        
        # Preparar dados para gráfico
        timestamps = [datetime.fromisoformat(row[0]) for row in results]
        coverages = [row[1] for row in results]
        
        # Gerar gráfico
        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, coverages, marker='o', linewidth=2, markersize=4)
        plt.title(f'Tendência de Cobertura - Últimos {days} dias')
        plt.xlabel('Data')
        plt.ylabel('Cobertura (%)')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Salvar gráfico
        plt.savefig('coverage_trend.png', dpi=150, bbox_inches='tight')
        print("📈 Gráfico salvo: coverage_trend.png")
        
        # Estatísticas
        if len(coverages) >= 2:
            trend = coverages[-1] - coverages[0]
            avg_coverage = sum(coverages) / len(coverages)
            
            print(f"\n📊 ESTATÍSTICAS ({days} dias)")
            print(f"Cobertura atual: {coverages[-1]:.1f}%")
            print(f"Cobertura inicial: {coverages[0]:.1f}%")
            print(f"Tendência: {trend:+.1f}%")
            print(f"Média: {avg_coverage:.1f}%")

# Uso
if __name__ == "__main__":
    tracker = CoverageTrendTracker()
    
    # Registrar cobertura atual
    tracker.record_coverage()
    
    # Gerar relatório de tendências
    tracker.generate_trend_report()
```

## 🔗 Próximos Passos

1. **[🧪 Testes](running-tests.md)** - Executar testes completos
2. **[🔄 Integração](integration.md)** - Testes de integração
3. **[🏗️ Arquitetura](../architecture/overview.md)** - Visão geral do sistema

---

## 📞 Referências

- 📊 **[Coverage.py](https://coverage.readthedocs.io/)** - Documentação oficial
- 🧪 **[Pytest Coverage](https://pytest-cov.readthedocs.io/)** - Plugin de cobertura
- 📈 **[Codecov](https://docs.codecov.com/)** - Serviço de cobertura online