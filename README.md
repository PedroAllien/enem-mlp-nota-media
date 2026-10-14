# Previsão da nota média no ENEM com MLP

Trabalho prático da disciplina de Deep Learning (frente **3.1 – Multilayer Perceptrons**).

**Problema.** Prever a nota média de um participante do ENEM (média de Ciências da Natureza, Ciências Humanas,
Linguagens, Matemática e Redação) usando **só informações disponíveis antes da prova**: questionário
socioeconômico da inscrição, situação escolar, faixa etária e local da prova. Regressão; unidade de observação = participante.

**Dados.** [Microdados do ENEM – INEP](https://dados.gov.br/dados/conjuntos-dados/inep-microdados-do-enem), edições 2022 e 2023.

| Partição | Origem | Participantes |
|---|---|---|
| treino | ENEM 2022 (80%) | 1.873.529 |
| validação | ENEM 2022 (10%) | 234.192 |
| teste-2022 | ENEM 2022 (10%) | 234.192 |
| teste-2023 (temporal) | ENEM 2023 (todos) | 2.658.763 |

As edições 2024 e 2025 não servem para este problema: o INEP passou a publicar questionário e notas em
arquivos separados, sem chave de ligação (Leia-Me ENEM 2024).

## Estrutura

```
src/enem/
  config.py      colunas, caminhos, sementes
  data.py        leitura do zip do INEP, filtros, alvo, divisão estratificada
  features.py    pré-processamento ajustado só no treino (one-hot, z-score, índice de município)
  model.py       MLP (one-hot + embedding do município + camadas ocultas configuráveis)
  train.py       laço de treino: early stopping, L1/L2, LR scheduler, métricas e custo
  pipeline.py    partições e cache dos dados codificados
  runs.py        registro de cada treino em JSON
  plots.py       estilo das figuras
scripts/
  01_preparar_dados.py     zip -> parquet filtrado
  02_capacidade.py         profundidade x largura (20 mil e 500 mil exemplos)
  03_regularizacao.py      dropout, L1, L2, BatchNorm, LayerNorm, early stopping
  04_optuna.py             busca de hiperparâmetros (TPE + MedianPruner, 60 trials)
  05_modelos_finais.py     3 arquiteturas + referências, treino completo, 3 sementes, testes 2022/2023
  build_notebooks.py       gera os notebooks
notebooks/
  01_analise_exploratoria.ipynb
  02_experimentos_e_resultados.ipynb
outputs/
  figures/ tables/         figuras e tabelas do relatório
  runs/                    histórico completo de cada treino (JSON)
  optuna.db                estudo Optuna (SQLite) - registro da busca
  modelos/                 pesos dos modelos finais (semente 0)
```

## Como reproduzir

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
mkdir -p data/raw
curl -L -o data/raw/microdados_enem_2022.zip https://download.inep.gov.br/microdados/microdados_enem_2022.zip
curl -L -o data/raw/microdados_enem_2023.zip https://download.inep.gov.br/microdados/microdados_enem_2023.zip
.venv/bin/python scripts/01_preparar_dados.py
.venv/bin/python -c "import sys; sys.path.insert(0,'src'); from enem.pipeline import build_cache; build_cache()"
.venv/bin/python scripts/02_capacidade.py
.venv/bin/python scripts/03_regularizacao.py
.venv/bin/python scripts/04_optuna.py 60
.venv/bin/python scripts/05_modelos_finais.py
.venv/bin/python scripts/build_notebooks.py
cd notebooks && ../.venv/bin/jupyter nbconvert --to notebook --execute --inplace *.ipynb
```

Tudo foi executado num MacBook M2 (16 GB) usando a GPU via MPS; o código usa CUDA ou CPU se não houver MPS.
