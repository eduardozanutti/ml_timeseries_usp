# Hierarchical Retail Forecasting Pipeline
**Modelos Estatísticos, Machine Learning e Reconciliação Hierárquica para Previsão de Vendas no Varejo de Moda**

[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)]()
[![License](https://img.shields.io/badge/license-MIT-green.svg)]()
[![Status](https://img.shields.io/badge/status-Production%20Ready-yellowgreen)]()

---

## Sumário

- [Visão Geral](#visão-geral)
- [Principais funcionalidades](#principais-funcionalidades)
- [Arquitetura do Projeto](#arquitetura-do-projeto)
- [Requisitos do Sistema](#requisitos-do-sistema)
- [Instalação Rápida](#instalação-rápida)
- [Configuração (`config.yaml`)](#configuração-configyaml)
- [Uso / Execução](#uso--execução)
- [Estrutura do `main.py`](#estrutura-do-mainpy)
- [Exemplo do `main.py` completo](#exemplo-do-mainpy-completo)
- [Métricas e Avaliação](#métricas-e-avaliação)
- [Reconciliação Hierárquica - Métodos Disponíveis](#reconciliação-hierárquica---métodos-disponíveis)
- [Boas práticas e dicas de performance](#boas-práticas-e-dicas-de-performance)
- [Debugging e checklist antes de rodar](#debugging-e-checklist-antes-de-rodar)
- [Contribuindo](#contribuindo)
- [Licença](#licença)
- [Contato / Referências](#contato--referências)

---

## Visão Geral

Este repositório contém um pipeline modular e reprodutível para **previsão hierárquica de vendas** em varejo de moda. Ele integra:

- Modelos estatísticos: `AutoARIMA`, `AutoETS`, `SeasonalNaive` (via *statsforecast*);
- Modelos de Machine Learning: `LightGBM`, `XGBoost` (via *mlforecast*);
- Otimização de hiperparâmetros com `Optuna` (TPE);
- Cross-validation temporal com janelas deslizantes;
- Reconciliação hierárquica (MinTrace — várias variantes, Top-Down, Middle-Out, Bottom-Up);
- Avaliação via SMAPE, MASE, RMSE, ND;
- Geração de artefatos: modelos candidatos, métricas, gráficos e tabelas de comparação.

O pipeline foi projetado para cenários com múltiplos níveis hierárquicos (ex.: total → produto → CD → loja) e grandes quantidades de séries (centenas).

---

## Principais funcionalidades

- Modularidade: cada etapa (tuning, criação de candidato, avaliação, reconciliação) tem módulo próprio em `src/`.
- Scalabilidade: suporte a modelos *globais* (ML) e *locais* (estatísticos).
- Automatização: execução completa desde leitura do `config.yaml` até exportação de métricas e resultados.
- Reconciliação avançada: MinTrace com estimadores WLS Struct/Var, Shrink, OLS; Top-Down, Middle-Out, Bottom-Up.
- Instrumentação: logging, salvamento de modelos candidatos, versionamento de métricas.

---

## Arquitetura do Projeto

```
project/
│
├── config.yaml              # configuração do pipeline
├── main.py                  # orquestrador principal (exemplo abaixo)
├── requirements.txt
│
├── src/
│   ├── model_tuning.py      # tuning (Optuna + mlforecast objective)
│   ├── create_candidate.py  # cria e salva candidato
│   ├── model_evaluate.py    # função de avaliação e métricas
│   ├── reconciliator.py     # lógica de reconciliação hierárquica
│   └── utils/               # utilitários (I/O, métricas, logging)
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── interim/
│   └── features/
│
├── models/
│   ├── candidates/
│   └── champion/
│
└── plots/
```

---

## Requisitos do Sistema

- Python 3.10+
- Memória: dependendo do tamanho do dataset (recomendado ≥ 16 GB para datasets maiores)
- Recomendado executar em máquina com GPU para acelerar treinamento de LightGBM/XGBoost (não obrigatório)

**Dependências principais** (veja `requirements.txt`):

```
hierarchicalforecast>=1.3.0
matplotlib>=3.10.1
mlforecast>=1.0.2
numpy>=1.26.4
openmeteo_requests>=1.7.4
openmeteo_sdk>=1.23.0
optuna>=4.6.0
pandas>=2.2.3
plotly>=6.0.0
requests>=2.32.3
requests-cache>=1.2.1
retry-requests>=2.0.0
scikit-learn>=1.6.1
tqdm>=4.67.1
utilsforecast>=0.2.14
xgboost>=3.1.2
lightgbm>=4.6.0
shap>=0.48.0
statsforecast>=2.0.2
```

---

## Instalação Rápida

1. Clone o repositório:

```bash
git clone https://github.com/seu-usuario/seu-repo.git
cd seu-repo
```

2. Crie e ative um ambiente virtual:

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows
```

3. Instale dependências:

```bash
pip install -r requirements.txt
```

---

## Configuração (`config.yaml`)

O pipeline é controlado por `config.yaml`. Abaixo um exemplo mínimo e comentado:

```yaml
# Exemplo de config.yaml (resumido)
time_col: date
id_col: id
target_col: sales

paths:
  data:
    raw: data/raw/
    processed: data/processed/
  metrics:
    train_path: outputs/metrics/train
    evaluation_path: outputs/metrics/eval

modeling:
  cv_config:
    horizon: 6         # horizonte de validação (meses)
    windows: 6         # número de folds/janelas
  training_metric: mase
  validation_metric: smape
  mlforecast:
    lag_columns: [1,2,3,12]
    # outros parâmetros próprios do mlforecast objective

models:
  LightGBM:
    enabled: true
    type: mlforecast
    name: LGBMRegressor
    fixed_params:
      n_estimators: 500
  XGBoost:
    enabled: true
    type: mlforecast
    name: XGBoostRegressor
  AutoARIMA:
    enabled: true
    type: statsforecast
    name: AutoARIMA

parameter_space:
  LightGBM:
    num_leaves: [31, 127]
    learning_rate: [0.01, 0.1]
  XGBoost:
    max_depth: [3, 10]

reconciliation:
  methods:
    - min_trace_wls_struct
    - min_trace_mint_shrink
    - top_down
    - middle_out
  min_trace_methods:
    - wls_struct
    - mint_shrink
  middle_level: 2
```

A configuração real no projeto pode ser mais rica; o importante é garantir caminhos, nomes de colunas e parâmetros de modelagem.

---

## Uso / Execução

1. Ajuste `config.yaml` conforme seus dados.
2. Garanta que os dados processados (com colunas `id_col`, `time_col`, `target_col`) estejam em `data/processed`.
3. Execute:

```bash
python main.py
```

O pipeline produzirá:
- Modelos candidatos salvos em `models/candidates/`
- Métricas agregadas (CSV) em `outputs/metrics/`
- Gráficos em `plots/`
- Relatórios de comparação por modelo e por método de reconciliação

---

## Estrutura do `main.py`

O `main.py` orquestra:

1. Carregamento de `config.yaml`;
2. Limpeza de métricas antigas;
3. Loop pelos modelos configurados:
   - Tuning (se ML e habilitado)
   - Criação do candidato
   - Avaliação no treino e no teste
   - Reconciliação das previsões
   - Avaliação dos métodos reconciliados
4. Salvamento dos resultados.

---

## Exemplo do `main.py` completo

> Abaixo está o `main.py` completo — cole diretamente no seu `main.py` se desejar.

```python
import os
import yaml  # Assumindo que config é carregado de YAML
from src.model_tuning import model_tuning
from src.create_candidate import create_candidate_model
from src.model_evaluate import model_evaluate
from src.reconciliator import dataset_reconciliator, compare_rec_methods

# Carregue config (ex.: de YAML)
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

models = config['models']
cv_config = config['modeling']['cv_config']
methods = config['reconciliation']['methods']
min_trace_methods = config['reconciliation']['min_trace_methods']
mid_level = config['reconciliation']['middle_level']
train_metrics_path = config['paths']['metrics']['train_path']
evaluation_metrics_path = config['paths']['metrics']['evaluation_path']
time_col = config['time_col']
id_col = config['id_col']
target_col = config['target_col']

# Limpe métricas existentes
if os.path.exists(os.path.join(train_metrics_path, 'metrics_summary.csv')):
    os.remove(os.path.join(train_metrics_path, 'metrics_summary.csv'))

if os.path.exists(os.path.join(evaluation_metrics_path, 'metrics_summary.csv')):
    os.remove(os.path.join(evaluation_metrics_path, 'metrics_summary.csv'))

candidate_info = {}
candidate_performances = {}

for model in [m for m in models if m not in ['LightGBM', 'XGBoost', 'AutoARIMA']]:
    train_to_model = train.copy()  # Assumindo 'train' e 'test' definidos
    test_to_model = test.copy()

    is_enable = models[model]['enabled']

    if is_enable:
        type_model = models[model]['type']
        training_metric = config['modeling']['training_metric']
        validation_metric = config['modeling']['validation_metric']
        train_compare_metrics = config['modeling']['train_compare_metrics']
        test_compare_metrics = config['modeling']['test_compare_metrics']
        model_name = models[model]['name']

        if type_model == 'mlforecast':
            fixed_params = models[model]['fixed_params']
            param_space = config['parameter_space'][model]
            mlforecast_params = config['modeling']['mlforecast']

            # Tuning
            best_value, best_model_params, best_mlforecast_params, mlf_fit_params = model_tuning(
                df=train_to_model,
                config=config,
                model_name=model_name,
                fixed_params=fixed_params,
                param_space=param_space,
                cv_config=cv_config,
                mlforecast_params=mlforecast_params,
                tuning_metric=training_metric
            )

            # Create candidate
            candidate, fitted_values, metric, results_metrics = create_candidate_model(
                df=train_to_model,
                config=config,
                cv_config=cv_config,
                type_model=type_model,
                model_name=model_name,
                metric=best_value,
                cv_metric=training_metric,
                compare_metrics=train_compare_metrics,
                model_params=best_model_params,
                mlf_params=best_mlforecast_params,
                mlf_fit_params=mlf_fit_params
            )

        else:  # statsforecast
            train_to_model = train_to_model[[id_col, time_col, target_col]]
            test_to_model = test_to_model[[id_col, time_col, target_col]]

            # Create candidate sem tuning
            candidate, fitted_values, metric, results_metrics = create_candidate_model(
                df=train_to_model,
                config=config,
                cv_config=cv_config,
                cv_metric=training_metric,
                type_model=type_model,
                compare_metrics=train_compare_metrics,
                model_name=model_name
            )

        candidate_info[metric] = candidate
        candidate_performances[model] = results_metrics

        # Evaluate
        prediction, validation_metric, results_metrics = model_evaluate(
            candidate_model=candidate,
            model_name=model_name,
            fitted_values=fitted_values,
            type_model=type_model,
            train=train_to_model,
            test=test_to_model,
            validation_metric=validation_metric,
            compare_metrics=test_compare_metrics,
            config=config
        )

        # Reconcile
        df_post_processed, recmethods = dataset_reconciliator(
            Y_df=Y_df,  # Assumindo definidos
            S_df=S_df,
            tags=tags,
            model_name=model_name,
            methods=methods,
            mid_level=mid_level,
            min_trace_methods=min_trace_methods,
            fitted_values=fitted_values,
            prediction=prediction
        )

        compare_rec_methods(
            model_name=model_name,
            df_post_processed=df_post_processed,
            fitted_values=fitted_values,
            test=test_to_model[[id_col, time_col, target_col]],
            config=config
        )

        for method in recmethods:
            df_rec_method = df_post_processed[[id_col, time_col, method]]

            # Evaluate com cada método de reconciliação
            prediction, validation_metric, results_metrics = model_evaluate(
                candidate_model=None,
                model_name=method,
                fitted_values=fitted_values,
                type_model=type_model,
                train=train_to_model,
                test=test_to_model,
                validation_metric=validation_metric,
                compare_metrics=test_compare_metrics,
                config=config,
                prediction=df_rec_method
            )
```

> **Observação**: O exemplo acima assume que variáveis como `train`, `test`, `Y_df`, `S_df`, e `tags` já foram definidas pelo pipeline de preprocessamento. Garanta que o módulo de preparação dos dados produza estes artefatos antes da execução.

---

## Métricas e Avaliação

As métricas implementadas e utilizadas para comparação são:

- **SMAPE** — Symmetric Mean Absolute Percentage Error (métrica principal para avaliação final)
- **MASE** — Mean Absolute Scaled Error (usada para tuning)
- **RMSE** — Root Mean Squared Error
- **ND** — Normalized Deviation

As métricas são gravadas em CSV por etapa (treino / avaliação) e consolidadas em um sumário `metrics_summary.csv`.

---

## Reconciliação Hierárquica - Métodos Disponíveis

- **Bottom-Up**: soma das séries desagregadas.
- **Top-Down**: desagrega a partir do nível mais alto (com proporções).
- **Middle-Out**: combina bottom-up e top-down a partir de um nível intermediário.
- **MinTrace (Optimal Combination)**:
  - OLS
  - WLS Var
  - WLS Struct
  - Mint Shrink

No código, `dataset_reconciliator` orquestra a aplicação desses métodos e retorna as previsões reconciliadas para avaliação.

---

## Boas práticas e dicas de performance

- Execute tuning (Optuna) com amostragem inicial e aumente trials gradualmente; use pruning.
- Para LightGBM/XGBoost, habilite `early_stopping_rounds`.
- Se datasets forem muito grandes, use amostragem estratificada por série para tuning.
- Grave checkpoints de estudos do Optuna (`study.optimize(..., storage='sqlite:///optuna.db')`).
- Use processamento em paralelo para avaliar folds quando possível (atenção a consumo de memória).
- Mantenha versão dos pacotes (requirements.lock) para reprodutibilidade.

---

## Debugging e checklist antes de rodar

- [ ] `config.yaml` apontando para pastas corretas.
- [ ] Dados em `data/processed/` com colunas corretas (`id_col`, `time_col`, `target_col`).
- [ ] Pastas de saída (`outputs/metrics/`, `models/candidates/`) existentes e com permissões de escrita.
- [ ] Dependências instaladas na versão correta.
- [ ] Se usar GPU, drivers e bibliotecas (LightGBM/XGBoost) configuradas adequadamente.

---

## Contribuindo

1. Fork o repositório.
2. Crie uma branch: `git checkout -b feature/minha-feature`.
3. Faça commits pequenos e descritivos.
4. Abra um Pull Request detalhando as mudanças.
5. Inclua testes unitários para novos utilitários.

---

## Licença

Distribuído sob a licença **MIT**. Veja o arquivo `LICENSE` para mais detalhes.

---

## Contato / Referências

Autores: Eduardo Soares Zanutti¹, Gustavo Ferreira de Lima², José Alejandro Encinas Riveros², Fernando Ferreira de Lima².

¹Instituto de Ciências Matemáticas e Computacional (ICMC)
Universidade de São Paulo

²Departamento de Engenharia Elétrica e de Computação (SEL)
UNiversidade de Sâo Paulo

E-mail: eduardozanutti@usp.br, fernando.lima0@usp.br, encinasriveros@usp.br, gustavosprondon@usp.br

Prof.: Dr. Diego Furtado Silva

Principais referências utilizadas:
- Hyndman, R. J., Athanasopoulos, G., et al. *Forecasting: Principles and Practice, the Pythonic Way*.
- Wickramasuriya, Athanasopoulos, Hyndman (2019) — MinTrace.
- Makridakis et al. (M4/M5 competitions).

---

**Fim do README.**
