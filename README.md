# Trabalho Adult Income

## Executar localmente
Python 3.12 recomendado. No terminal, dentro desta pasta:

```bash
python -m pip install -r requirements.txt
python analise.py
```

O script usa os arquivos locais em dados/. Se faltarem, baixa o ZIP oficial da UCI. Gera ou sobrescreve resultados em resultados/. O notebook Adult_Income_Colab.ipynb permite reproduzir o trabalho no Colab.

## Protocolo e resultados
Classe positiva >50K = 1. Semente 42. Treino oficial limpo: 32.535 registros. Teste oficial limpo: 16.248 registros. StratifiedKFold com 10 folds idênticos entre configurações. 3 árvores, 3 florestas e 4 regressões logísticas, além do baseline. Imputação, StandardScaler e OneHotEncoder dentro do Pipeline. Escolha por F1 médio. Apenas LR4 foi avaliada no teste final.

Arquivos em resultados/:
- auditoria.json: valores únicos brutos, ausências, duplicatas, versões e hashes SHA-256.
- perfil_atributos.csv e estatisticas_numericas.csv: investigação inicial.
- configuracoes.json: todos os hiperparâmetros efetivos.
- metricas_por_fold.csv: 110 avaliações de validação.
- comparacao_cv.csv: médias e desvios-padrão.
- predicoes_oof.csv e matrizes_cv.json: previsões fora do treino e matrizes.
- teste_final.json e predicoes_teste.csv: avaliação do vencedor.
- f1_cv.png, acuracia_cv.png e matriz_teste.png: gráficos.

execucao.log registra a execução que fundamenta o relatório. Tempos dependem do equipamento. O desvio-padrão entre folds não é intervalo de confiança. A vantagem de LR4 sobre RF2 não foi submetida a teste de significância. Não houve SMOTE, reamostragem nem ajuste de limiar. Os números nos documentos foram gerados a partir destes resultados.

## Fonte e licença
BECKER, B.; KOHAVI, R. Adult. UCI Machine Learning Repository, 1996.
https://archive.ics.uci.edu/dataset/2/adult
https://doi.org/10.24432/C5XW20
Dados sob CC BY 4.0. Arquivos oficiais incluídos sem modificação; limpeza somente em memória.

