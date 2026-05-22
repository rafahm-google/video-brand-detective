import pandas as pd
import os

os.makedirs('universal_reports', exist_ok=True)
data = {
    'Marca': ['Marca Teste 1', 'Marca Teste 2'],
    'Data Analise': ['2026-03-30', '2026-03-30'],
    'Tempo no Video': ['0:00:01', '0:00:03'],
    'Timestamp Absoluto': ['2026-03-30 00:00:01', '2026-03-30 00:00:03'],
    'Tipo Original': ['Visual', 'Visual'],
    'Formato': ['Logo', 'Produto'],
    'Localizacao Exata': ['[200, 300, 400, 600]', '[500, 500, 800, 800]'],
    'Texto na Tela Exibido': ['N/A', 'N/A'],
    'Contexto Geral': ['Teste', 'Teste'],
    'O que foi Falado': ['N/A', 'N/A'],
    'Link para o Momento': ['http', 'http'],
    'Titulo Video': ['Dummy', 'Dummy'],
    'Source': ['http']
}
df = pd.DataFrame(data)
df.to_csv('universal_reports/dummy_detailed.csv', index=False)
