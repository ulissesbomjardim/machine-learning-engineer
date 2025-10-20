from extract.airports_database import download_and_extract_airports_database


def run_airports_database():
    """
    Função principal do script.
    """
    print('Iniciando download e extração da base de dados de aeroportos...')
    success = download_and_extract_airports_database()

    if success:
        print('✅ Processo concluído com sucesso!')
    else:
        print('❌ Processo falhou. Verifique os logs acima.')


if __name__ == '__main__':
    run_airports_database()
