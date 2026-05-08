# 🚀 COMO EXECUTAR O ABLATION STUDY

## 📋 Resumo

O ablation study avalia sistematicamente o impacto de cada componente da rede neural DNN:
- Número de camadas
- Número de neurônios
- BatchNormalization
- Dropout rates
- Features de entrada
- Funções de ativação

**Resultado**: Recomendação de melhor variante para cada caso de uso (produção, pesquisa, etc.)

---

## 📦 Arquivos Criados

| Arquivo | Descrição |
|---------|-----------|
| **ABLATION_STUDY_PLAN.md** | Plano estratégico com 6 dimensões de ablation |
| **ablation_study_dnn.py** | Script Python para executar os testes |
| **COMO_EXECUTAR.md** | Este arquivo (instruções) |

---

## 🔧 PREREQUISITOS

1. **Dataset gerado**:
   ```bash
   Redes Neurais/results/data/dataset_nn_stratified_0_30dB.h5
   ```
   ou executar primeiro:
   ```bash
   jupyter notebook Redes\ Neurais/notebooks/NN_01_DataGeneration.ipynb
   ```

2. **Dependências Python**:
   ```bash
   pip install tensorflow numpy pandas matplotlib scikit-learn
   ```

3. **Ambiente virtual ativado**:
   ```bash
   # No PowerShell:
   & ".\venv_nn\Scripts\Activate.ps1"
   
   # Ou verificar se venv_nn/ existe
   ```

---

## ▶️ EXECUÇÃO

### Opção 1: Via Terminal (Recomendado)

```bash
# Navegar para diretório do projeto
cd c:\Users\thami\OneDrive\Documents\TAG-Authentication

# Ativar ambiente virtual (se necessário)
& ".\.venv-1\Scripts\Activate.ps1"

# Executar ablation study
python Redes\ Neurais/ablation_study_dnn.py \
    --dataset "Redes Neurais/results/data/dataset_nn_stratified_0_30dB.h5" \
    --output "Redes Neurais/results/ablation_results" \
    --runs 3
```

### Opção 2: Simplificada (Path Automático)

```bash
cd "Redes Neurais"
python ablation_study_dnn.py \
    --dataset results/data/dataset_nn_stratified_0_30dB.h5 \
    --output results/ablation_results \
    --runs 3
```

### Opção 3: Python Script (Dentro do Notebook ou IDE)

```python
from pathlib import Path
import sys

# Adicionar caminho se necessário
sys.path.insert(0, str(Path.cwd() / "Redes Neurais"))

from ablation_study_dnn import run_ablation_study

run_ablation_study(
    dataset_path="Redes Neurais/results/data/dataset_nn_stratified_0_30dB.h5",
    output_dir="Redes Neurais/results/ablation_results",
    num_runs=3
)
```

---

## 📊 SAÍDAS ESPERADAS

Após a execução, a pasta `Redes Neurais/results/ablation_results/` conterá:

```
ablation_results/
├── ablation_results.csv           # Tabela principal com todas as métricas
├── ablation_results.json          # Dados em formato JSON
├── ablation_pareto.png            # Gráfico: FNR vs. Parâmetros
├── ablation_training_time.png     # Gráfico: FNR vs. Tempo Treino
└── ablation_efficiency.png        # Gráfico: Scores de Eficiência
```

### Interpretação dos Resultados

#### 1. **ablation_results.csv**

Exemplo de linha:
```
variant_name      | config_description        | num_parameters | fnr_mean  | auc_mean | efficiency_score
Baseline          | 256→128→64               | 43265          | 1.05e-07  | 0.9999   | 1.00
B1: 50% redução   | 128→64→32                | 11000          | 1.50e-06  | 0.9990   | 2.85
```

**Colunas principais**:
- `variant_name`: Nome da variante
- `num_parameters`: Total de parâmetros treináveis
- `fnr_mean`: Taxa média de falso negativo (menor é melhor)
- `auc_mean`: AUC-ROC (mais próximo de 1.0 é melhor)
- `training_time_s`: Tempo de treino em segundos
- `inference_time_ms`: Tempo de inferência por amostra
- `efficiency_score`: Score de eficiência (>2 = excelente, >1 = bom)

#### 2. **ablation_pareto.png**

```
Gráfico em escala log-log mostrando:
- Eixo X: Número de parâmetros
- Eixo Y: FNR (lower is better)
- Cada ponto é uma variante

Interpretação:
- Pontos abaixo/esquerda = melhores (menos parâmetros, melhor FNR)
- Curva Pareto = frontier de soluções eficientes
```

#### 3. **ablation_efficiency.png**

Gráfico de barras com cores:
```
🟢 Verde:  Excelente (score ≥ 2.0)  - Altamente recomendado
🟠 Laranja: Bom (score 1.0-2.0)     - Recomendado
🔴 Vermelho: Ruim (score < 1.0)     - Não recomendado
```

---

## 🎯 INTERPRETAÇÃO: O QUE PROCURAR

### Cenário 1: Você quer máxima eficiência

Procure na tabela:
1. Ordene por `efficiency_score` descending
2. Verifique se `fnr_mean` ainda está aceitável (ex: < 1e-6)
3. Confirme redução de `num_parameters`

**Exemplo**: Se B1 tem score 2.85 e FNR de 1.5e-6, é ótimo!

### Cenário 2: Você quer manter performance (FNR < 1e-7)

1. Filtre variantes com FNR < 1e-7
2. Escolha a com menor `num_parameters`
3. Se nenhuma, escolha a com maior `auc_mean`

### Cenário 3: Você quer deploy em edge device (latência baixa)

1. Procure por menor `inference_time_ms`
2. Verifique `num_parameters` (deve caber em memória)
3. Confirme `fnr_mean` aceitável para seu caso de uso

---

## 🔍 VARIANTES EXPLICADAS

### Por que rodar ablation study?

Cada variante testa UMA dimensão:

```
BASELINE (256→128→64, BatchNorm, Dropout(0.3,0.3,0.2))
    │
    ├─ A1: Remove uma camada → Impacto de profundidade?
    ├─ B1: Reduz neurônios 50% → Impacto de largura?
    ├─ C1: Remove BatchNorm → Impacto de normalização?
    ├─ D1: Remove Dropout → Impacto de regularização?
    └─ E2: Reduz features → Impacto de entrada?
```

**Resposta esperada**:
- Se A1 ≈ Baseline em FNR: última camada é redundante ✂️
- Se B1 >> Baseline em FNR: neurônios são críticos 🔴
- Se C1 ≈ Baseline em FNR: BatchNorm pode ser opcional ✂️
- Se D1 >> Baseline em FNR: Dropout é importante 🟢
- Se E2 ≈ Baseline em FNR: feature extra não ajuda ✂️

---

## 📈 EXEMPLO DE ANÁLISE

### Supposições: Resultados esperados

| Variante | Parâmetros | FNR | Score | Recomendação |
|----------|-----------|-----|-------|--------------|
| Baseline | 43,265 | 1.0e-7 | 1.00 | Referência |
| A1 | 33,000 | 1.1e-7 | 3.21 | ⭐ Recomendado |
| B1 | 11,000 | 1.5e-6 | 2.85 | ⭐⭐ Muito bom |
| B2 | 3,000 | 3.2e-5 | 0.15 | ❌ Muito ruim |
| C1 | 43,265 | 5.0e-7 | 0.50 | ⚠️ Sem BatchNorm piora |
| D1 | 43,265 | 5.0e-6 | 0.50 | ⚠️ Sem Dropout piora |

### Interpretação

1. **A1 (Remover camada 3)**: Melhor candidato!
   - Reduz parâmetros 24%
   - Mantém FNR praticamente igual
   - ✂️ **Ação**: Considere usar 2 camadas no lugar de 3

2. **B1 (Reduzir neurônios 50%)**: Muito bom para edge devices
   - Reduz parâmetros 74%
   - FNR aumenta, mas ainda < 2e-6 (aceitável)
   - ⚡ **Ação**: Para produção tempo-real, use B1

3. **B2 (Reduzir neurônios 75%)**: Muito compacto mas frágil
   - FNR piora MUITO
   - ❌ **Ação**: Não recomendado

4. **C1 e D1**: BatchNorm e Dropout são importantes
   - ✂️ **Ação**: Mantenha ambos

---

## 🎬 PRÓXIMOS PASSOS

### 1. Executar Ablation

```bash
python Redes\ Neurais/ablation_study_dnn.py \
    --dataset "Redes Neurais/results/data/dataset_nn_stratified_0_30dB.h5" \
    --output "Redes Neurais/results/ablation_results" \
    --runs 3
```

**Tempo esperado**: 
- Baseline: ~15 min (CPU), ~2 min (GPU)
- Todas as 14 variantes: ~3-4 horas (CPU) ou ~30 min (GPU)

### 2. Analisar Resultados

```bash
# Abrir CSV em Excel/Pandas
import pandas as pd
df = pd.read_csv("Redes Neurais/results/ablation_results/ablation_results.csv")
print(df.sort_values('efficiency_score', ascending=False))
```

### 3. Escolher Melhor Variante

Com base em seus critérios:
- **Pesquisa/Academia**: Use Baseline ou A3 (máxima performance)
- **Produção Geral**: Use A1 (balanceado)
- **Edge Device**: Use B1 (compacto, eficiente)
- **Tempo Real**: Otimize B1 ou D1 (mais rápido)

### 4. Implementar Recomendação

Atualize [NN_02_DNN_Correlator.ipynb](../notebooks/NN_02_DNN_Correlator.ipynb):

```python
# Linha onde constrói modelo:
# ANTES (Baseline):
model = build_dnn_correlator(
    input_dim=X_train.shape[1],
    units=(256, 128, 64),
    dropout_rates=(0.3, 0.3, 0.2)
)

# DEPOIS (Exemplo: Usar A1):
model = build_dnn_correlator(
    input_dim=X_train.shape[1],
    units=(256, 128),  # Remova o 64
    dropout_rates=(0.3, 0.3)
)
```

### 5. Validar e Documentar

```bash
# Retreinar com nova configuração
jupyter notebook Redes\ Neurais/notebooks/NN_02_DNN_Correlator.ipynb

# Comparar resultados
jupyter notebook Redes\ Neurais/notebooks/NN_06_Comparison_vs_Baseline.ipynb

# Documentar conclusões
# Adicionar seção em ARQUITETURA_DNN_EXPLICACAO.md:
#   "## Ablation Study Results"
#   "Based on ablation study, recommended variant is A1..."
```

---

## ⚠️ TROUBLESHOOTING

### Erro: Dataset não encontrado

```
ERROR: Could not load dataset: File not found
```

**Solução**:
```bash
# Verificar path correto
ls "Redes Neurais/results/data/"

# Se não existe, rodar geração:
jupyter notebook Redes\ Neurais/notebooks/NN_01_DataGeneration.ipynb
```

### Erro: Out of Memory (OOM)

```
ResourceExhaustedError: OOM when allocating tensor
```

**Solução**:
```bash
# Opção 1: Reduzir batch_size no script
# Opção 2: Usar GPU
# Opção 3: Usar dataset menor (--dataset com 10k amostras)
```

### Script muito lento

**Tempo esperado**:
- 1 variante, 1 run, CPU: ~10 min
- 14 variantes, 3 runs, CPU: ~4 horas
- 14 variantes, 3 runs, GPU: ~30 min

**Para acelerar**:
```bash
# Use --runs 1 para testes rápidos
python Redes\ Neurais/ablation_study_dnn.py \
    --dataset "..." \
    --output "..." \
    --runs 1
```

---

## 📞 CONTATO / DÚVIDAS

Se tiver problemas ao rodar o ablation study:

1. Verificar logs completos
2. Consultar [ABLATION_STUDY_PLAN.md](ABLATION_STUDY_PLAN.md) para entender cada variante
3. Validar que dataset está correto com `NN_01_DataGeneration.ipynb`

---

**Última atualização**: Abril 2026
**Status**: ✅ Pronto para execução
