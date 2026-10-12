# Análise de Dados de Saúde – AVC

Análise em Python de pacientes com e sem registro de AVC, a partir do arquivo `Atividade2.csv`. O programa faz a limpeza dos dados, compara os dois grupos, gera três gráficos com Matplotlib e salva os dados tratados em um banco SQLite.

## Como executar

Requer Python 3.9 ou superior.

1. Instale as bibliotecas:

   ```
   pip install -r requirements.txt
   ```

2. Coloque o arquivo `Atividade2.csv` na mesma pasta do script.

3. Execute:

   ```
   python analise_avc.py
   ```

O programa mostra a análise no terminal e cria o banco `saude_avc.db` e os gráficos em PNG na mesma pasta.

## Autores

Willian Shoei Koide e Lorenzo de Lima Volaco
