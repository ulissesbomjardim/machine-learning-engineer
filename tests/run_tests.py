"""
Script para executar testes específicos baseado na implementação real.
"""
import subprocess
import sys
from pathlib import Path


def run_specific_tests():
    """Executa testes específicos baseados nos módulos implementados."""

    # Testes que sempre devem passar (utilitários e integração)
    always_run_tests = [
        'tests/test_utils.py',
        'tests/test_integration.py',
        'tests/test_ml_pipeline.py',
    ]

    # Testa se routers existem
    src_path = Path('src')
    if (src_path / 'routers').exists():
        always_run_tests.append('tests/test_routers.py')

    # Testa se services existem e estão implementados
    if (src_path / 'services' / 'database.py').exists():
        with open(src_path / 'services' / 'database.py', 'r') as f:
            content = f.read()
            # Se tem mais que apenas imports/comentários
            if len(content.strip()) > 100:
                always_run_tests.append('tests/test_services.py')

    print('🧪 Executando testes específicos...')
    print(f'📁 Testes a executar: {len(always_run_tests)}')

    for test_file in always_run_tests:
        print(f'\n▶️ Executando: {test_file}')

    # Executa todos os testes selecionados
    cmd = ['poetry', 'run', 'pytest', *always_run_tests, '-v', '--tb=short']

    result = subprocess.run(cmd, capture_output=False)
    return result.returncode


def run_coverage_tests():
    """Executa testes com cobertura."""
    print('📊 Executando testes com cobertura...')

    cmd = [
        'poetry',
        'run',
        'pytest',
        'tests/',
        '--cov=src',
        '--cov-report=html',
        '--cov-report=xml',
        '--cov-report=term',
        '-v',
    ]

    result = subprocess.run(cmd, capture_output=False)
    return result.returncode


def main():
    """Função principal."""
    if len(sys.argv) > 1 and sys.argv[1] == '--coverage':
        return run_coverage_tests()
    else:
        return run_specific_tests()


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
