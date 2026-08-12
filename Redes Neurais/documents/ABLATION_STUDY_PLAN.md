# 🔬 ABLATION STUDY - DNN Correlator para TAG Authentication

## Objetivo
Avaliar o impacto de cada componente da arquitetura DNN na performance final, identificando oportunidades de otimização com manutenção de eficiência computacional.

---

## 📊 ARQUITETURA BASELINE ATUAL

```
Input (4D): [τ_correlator, h_estimate, SNR_local, energy]
    ↓
Dense(256) + ReLU + BatchNorm + Dropout(0.3)
    ↓
Dense(128) + ReLU + BatchNorm + Dropout(0.3)
    ↓
Dense(64) + ReLU + Dropout(0.2)
    ↓
Dense(1) + Sigmoid
    ↓
Output: P(autêntico) ∈ [0,1]

Parâmetros Baseline: ~43,265
Tempo Treino: ~5-10 min (CPU)
Tempo Inferência: ~1 ms/sample
FNR Target: ≤ 10⁻⁷
```

---

## 🔍 DIMENSÕES DE ABLATION

### 1. PROFUNDIDADE (Número de Camadas Ocultas)

| Variante | Arquitetura | Parâmetros | Esperado | Nota |
|----------|------------|-----------|----------|------|
| **Baseline** | 256→128→64 | 43,265 | FNR ~10⁻⁷ | Referência |
| **A1: 2 camadas** | 256→128→1 | ~33,000 | FNR ~10⁻⁶ | Remove camada 3 |
| **A2: 1 camada** | 256→1 | ~1,280 | FNR ~10⁻⁵ | Remove 2 camadas |
| **A3: 4 camadas** | 256→128→64→32→1 | ~55,000 | FNR ~10⁻⁷·² | Adiciona profundidade |

---

### 2. LARGURA (Número de Neurônios)

| Variante | Arquitetura | Parâmetros | Esperado | Nota |
|----------|------------|-----------|----------|------|
| **Baseline** | 256→128→64 | 43,265 | FNR ~10⁻⁷ | Referência |
| **B1: 50% menor** | 128→64→32 | 11,000 | FNR ~10⁻⁶ | Mais compacto |
| **B2: 75% menor** | 64→32→16 | 3,000 | FNR ~10⁻⁵·⁵ | Muito compacto |
| **B3: Assimétrico** | 192→96→48 | 25,000 | FNR ~10⁻⁶·⁸ | Intermediário |

---

### 3. BATCH NORMALIZATION

| Variante | Com/Sem | Parâmetros | Esperado | Nota |
|----------|---------|-----------|----------|------|
| **Baseline** | Com BatchNorm | 43,265 + 768 (BN params) | FNR ~10⁻⁷ | Estabiliza treino |
| **C1: Sem BatchNorm** | Sem | 43,265 | FNR ~10⁻⁶·⁵ | Convergência mais lenta |
| **C2: BatchNorm só na 1ª** | Parcial | 43,265 + 256 | FNR ~10⁻⁶·⁸ | Menos overhead |

---

### 4. REGULARIZAÇÃO (Dropout)

| Variante | Dropout Rates | Esperado | Nota |
|----------|--------------|----------|------|
| **Baseline** | (0.3, 0.3, 0.2) | FNR ~10⁻⁷ | Referência |
| **D1: Sem Dropout** | (0, 0, 0) | FNR ~10⁻⁶ | Overfitting maior |
| **D2: Leve** | (0.1, 0.1, 0.1) | FNR ~10⁻⁶·⁸ | Menos regularização |
| **D3: Forte** | (0.5, 0.5, 0.3) | FNR ~10⁻⁶·⁹ | Underfitting possível |

---

### 5. FEATURES (Número de Entradas)

| Variante | Features | Parâmetros | Esperado | Nota |
|----------|----------|-----------|----------|------|
| **Baseline** | 4 (τ, ĥ, SNR, E) | 43,265 | FNR ~10⁻⁷ | Referência |
| **E1: 3 features** | (τ, ĥ, SNR) | 37,000 | FNR ~10⁻⁶·⁵ | Remove energia |
| **E2: 2 features** | (τ, ĥ) | 33,000 | FNR ~10⁻⁶ | Apenas correlator + canal |
| **E3: 1 feature** | (τ) | 32,000 | FNR ~10⁻⁶·⁰ | Baseline clássico |

---

### 6. FUNÇÃO DE ATIVAÇÃO

| Variante | Ativação | Esperado | Nota |
|----------|----------|----------|------|
| **Baseline** | ReLU | FNR ~10⁻⁷ | Padrão |
| **F1: Tanh** | Tanh | FNR ~10⁻⁶·⁸ | Convergência diferente |
| **F2: Leaky ReLU** | LeakyReLU(0.2) | FNR ~10⁻⁶·⁹ | Menos "dead neurons" |
| **F3: Sigmoid** | Sigmoid | FNR ~10⁻⁶·⁵ | Pior em camadas intermediárias |

---

## 🎯 MÉTODO DE AVALIAÇÃO

### Protocolo de Teste

Para cada variante, executar:

```python
1. Treinar 3 vezes (diferentes seeds)
   └─ Calcular média e desvio padrão

2. Medir:
   ├─ Número total de parâmetros
   ├─ Tempo de treino (segundos)
   ├─ Tempo de inferência por batch (ms)
   ├─ FNR @ P(FA) = 0.01
   ├─ AUC-ROC
   ├─ Accuracy
   └─ Throughput (samples/sec)

3. Calcular Eficiência:
   ├─ FNR vs. Parâmetros
   ├─ FNR vs. Tempo Treino
   ├─ FNR vs. Tempo Inferência
   └─ Score Pareto
```

### Matriz de Comparação

```python
DataFrame com colunas:
├─ Variante
├─ Config (descrição)
├─ Parâmetros
├─ Tempo_Treino_s
├─ Tempo_Inf_ms
├─ FNR_médio
├─ FNR_std
├─ AUC
├─ Accuracy
├─ Degradação_FNR%  # vs baseline
├─ Ganho_Compactação%  # vs baseline
└─ Eficiência_Score
```

---

## 📈 MÉTRICAS DE EFICIÊNCIA

### 1. **Trade-off Performance vs. Tamanho**

```
Pareto Efficiency Score = FNR_melhoria / Parâmetros_redução

Exemplo:
├─ Baseline: FNR=10⁻⁷, Params=43k
├─ Variante: FNR=10⁻⁶·⁵, Params=11k
└─ Score = (10⁻⁶·⁵ / 10⁻⁷) / (11k / 43k) = 3.16 / 0.26 = 12.15

Interpretação:
  Score > 1: Bom (menos parâmetros, performance aceitável)
  Score > 3: Excelente (redução significativa, degradação pequena)
```

### 2. **Trade-off Performance vs. Latência**

```
Latência Efficiency = (1 - FNR_degradação%) / (1 + Tempo_Inf_overhead%)

Exemplo:
├─ Baseline: FNR=10⁻⁷, Inf=1.0ms
├─ Variante: FNR=10⁻⁶·⁵, Inf=0.5ms
└─ Score = (1 - 0.0005) / (1 + (-0.5)) = 0.9995 / 0.5 = 1.999

Interpretação:
  Score > 1: Bom (latência menor, performance aceitável)
```

### 3. **Custo Computacional Total**

```
Custo_Total = (Treino_s × freq_retreino) + (Inf_ms × freq_inferência)

Exemplo (100k inferências/dia, retreino 1x/mês):
├─ Baseline: (8min × 1) + (1ms × 100k × 30) = 8 min + 50 min = 58 min/mês
├─ Variante B1: (4min × 1) + (0.5ms × 100k × 30) = 4 min + 25 min = 29 min/mês
└─ Ganho: -50% custo computacional
```

---

## 🧪 EXPERIMENTOS PRIORITÁRIOS

### Experimento 1: Redução de Camadas (CRÍTICO)

```python
# A1: Remover camada 3 (64 neurônios)
# Justificativa: A camada 3 tem menos impacto, pode ser redundante

baseline = (256 → 128 → 64 → 1)
A1       = (256 → 128 → 1)       # Remove últimaocamada oculta

Resultado esperado:
├─ Parâmetros reduzem de 43k para 33k (-23%)
├─ Tempo treino reduz ~15-20%
├─ FNR aumenta de 10⁻⁷ para 10⁻⁶ (degradação ~1%)
└─ Eficiência: BOA se FNR ainda < 10⁻⁶·⁵
```

### Experimento 2: Redução de Neurônios (CRITICO)

```python
# B1: 50% menos neurônios em cada camada
# Justificativa: 256 pode ser overkill para 4 features

baseline = (256 → 128 → 64 → 1)
B1       = (128 → 64 → 32 → 1)    # 50% redução

Resultado esperado:
├─ Parâmetros reduzem de 43k para 11k (-74%)
├─ Tempo treino reduz ~50%
├─ Tempo inferência reduz ~30-40%
├─ FNR aumenta para ~10⁻⁶ (degradação ~10%)
└─ Eficiência: EXCELENTE se FNR ≥ 10⁻⁶·⁵
```

### Experimento 3: Remover BatchNormalization Parcialmente (MÉDIO)

```python
# C2: BatchNorm apenas na camada 1
# Justificativa: BatchNorm adiciona 768 parâmetros, pode não ser necessário em todas

baseline = (BatchNorm em todas as 3 camadas)
C2       = (BatchNorm apenas na 1ª camada)

Resultado esperado:
├─ Parâmetros reduzem de 43k para ~43k - 512 = 42.5k (-1%)
├─ Tempo treino reduz ~5%
├─ FNR aumenta levemente para ~10⁻⁶·⁸ (degradação <5%)
└─ Eficiência: MARGINAL
```

### Experimento 4: Redução de Features (ESPECULATIVO)

```python
# E2: Apenas 2 features (τ e ĥ)
# Justificativa: SNR e Energia podem ser redundantes

baseline = (τ, ĥ, SNR, E) → 4D
E2       = (τ, ĥ) → 2D

Resultado esperado:
├─ Parâmetros reduzem de 43k para ~20k (-53%)
├─ Tempo treino reduz ~40%
├─ FNR aumenta para ~10⁻⁶ (degradação ~10%)
└─ Eficiência: ÓTIMA se FNR ≥ 10⁻⁶·⁵ (teoria supõe que correlador + canal são suficientes)

Risco:
└─ Pode perder robustez a model misspecification (SNR/energia ajudam em anomalias)
```

---

## 📋 TABELA ESPERADA DE RESULTADOS

| Variante | Config | Params | Train(s) | Inf(ms) | FNR | AUC | Degradação% | Compactação% | Eficiência |
|----------|--------|--------|----------|---------|-----|-----|------------|------------|-----------|
| Baseline | 256→128→64 | 43,265 | 480 | 1.0 | 10⁻⁷ | 0.9999 | 0% | 0% | 1.00 |
| A1 | 256→128 | 33,000 | 410 | 0.9 | 10⁻⁶·⁰ | 0.9995 | +1% | 24% | 3.21 ⭐ |
| A2 | 256→1 | 1,280 | 180 | 0.5 | 10⁻⁵ | 0.9970 | +20% | 97% | 0.04 |
| B1 | 128→64→32 | 11,000 | 240 | 0.6 | 10⁻⁶·⁵ | 0.9990 | +0.5% | 74% | 2.85 ⭐⭐ |
| B2 | 64→32→16 | 3,000 | 120 | 0.4 | 10⁻⁵·⁵ | 0.9980 | +5% | 93% | 0.15 |
| C2 | BN só em L1 | 42,500 | 460 | 1.0 | 10⁻⁶·⁸ | 0.9998 | +0.2% | 2% | 0.90 |
| D1 | Sem Dropout | 43,265 | 450 | 1.0 | 10⁻⁶ | 0.9992 | +1% | 0% | 0.50 |
| E2 | τ + ĥ só | 20,000 | 350 | 0.7 | 10⁻⁶ | 0.9985 | +1% | 54% | 1.80 |

**Legenda Eficiência**: 
- ⭐⭐: Altamente Recomendado (ganho ≥ 2.0)
- ⭐: Recomendado (ganho ≥ 1.5)
- OK: Aceitável (ganho 0.5-1.5)

---

## 💡 CENÁRIOS RECOMENDADOS

### Cenário 1: "Maximum Efficiency" (Produção Tempo-Real)

```
Objetivo: Mínimo tamanho/latência, FNR aceitável (≥10⁻⁶)

Recomendação: Variante B1 (128→64→32)
├─ Parâmetros: 11k (-74%)
├─ Treino: 240s (-50%)
├─ Inferência: 0.6ms (-40%)
├─ FNR: ~10⁻⁶·⁵
├─ Deploy: Em microcontroladores, edge devices
└─ Trade-off: Perda marginal de robustez

Python Config:
  model = build_dnn_correlator(
    units=[128, 64, 32],
    dropout_rates=[0.3, 0.3, 0.2]
  )
```

### Cenário 2: "Balanced" (Padrão)

```
Objetivo: Bom balanço performance/eficiência

Recomendação: Variante A1 (256→128 com 2 camadas)
├─ Parâmetros: 33k (-24%)
├─ Treino: 410s (-15%)
├─ Inferência: 0.9ms (-10%)
├─ FNR: ~10⁻⁶·⁰
├─ Deploy: Edge devices, smartphones
└─ Trade-off: Pequena degradação, ótimo balanço

Python Config:
  model = build_dnn_correlator(
    units=[256, 128],  # Remove camada 3
    dropout_rates=[0.3, 0.3]
  )
```

### Cenário 3: "Robustness First" (Pesquisa)

```
Objetivo: Máxima performance, size/latência secundário

Recomendação: Baseline atual ou A3 (256→128→64→32)
├─ Parâmetros: 55k
├─ Treino: 550s
├─ Inferência: 1.1ms
├─ FNR: ~10⁻⁷·² (melhor)
├─ Deploy: Servidores, cloud
└─ Trade-off: Tamanho/latência maiores, máxima robustez

Python Config:
  model = build_dnn_correlator(
    units=[256, 128, 64, 32],  # 4 camadas
    dropout_rates=[0.3, 0.3, 0.2, 0.1]
  )
```

---

## 🚀 PRÓXIMOS PASSOS

### Fase 1: Implementação Ablation Script

```bash
notebook: ABLATION_STUDY_DNN.ipynb
├─ Carregar dataset
├─ Loop sobre variantes A1, A2, B1, B2, C2, D1, E2
├─ Para cada: treinar 3x, medir métricas, salvar resultados
├─ Gerar tabela comparativa
└─ Plotar Pareto front (FNR vs. Parâmetros)
```

### Fase 2: Validação da Melhor Variante

```
├─ Treinar variante recomendada (B1)
├─ Testar em datasets diferentes (OOD)
├─ Comparar vs. baseline em múltiplos SNRs
└─ Validar FNR ≥ 10⁻⁶·⁵
```

### Fase 3: Documentação

```
├─ Atualizar ARQUITETURA_DNN_EXPLICACAO.md
├─ Adicionar seção "Ablation Study Results"
├─ Recomendar melhor variante por use-case
└─ Publicar resultados
```

---

## 📊 VISUALIZAÇÕES ESPERADAS

### Plot 1: Pareto Front

```
FNR (log10)
    ↑
-5  │ A2 (outlier)
    │   •
-6  │   B2•─────• E2─────•B1
    │        •C2        •
-7  │        BASELINE ••A1
    │                  •
    └────────────────────→ Parâmetros (log10)
         10   100   1000   10k  100k

Legenda:
  • Pontos acima/direita = menos eficientes
  • Pontos abaixo/esquerda = mais eficientes
  • Curva convexa = Pareto front
```

### Plot 2: FNR vs. Parâmetros

```
FNR (linear)
    ↑
10⁻⁵ │ A2━━B2
     │  │
10⁻⁶ │  ├─E2, B1, D1
     │  │
10⁻⁷ │  └─BASELINE, A1, C2
     │
     └────────────────────→ Parâmetros
         1k    10k    100k
```

### Plot 3: Trade-off Matrix

```
        Pequeno            Médio            Grande
Rápido   B2         ✓B1             A1
         (3%)       (Best)    (24% params)

Médio    E2         Baseline    A3
         (54%)      (Baseline)

Lento    C2         LSTM        Ens
         (BN)       (No)        (Overkill)
```

---

## 🎬 IMPLEMENTAÇÃO

Para começar, crie `ABLATION_STUDY_DNN.ipynb` baseado em `NN_02_DNN_Correlator.ipynb` com:

```python
# Pseudocódigo
variants = {
    'Baseline': (256, 128, 64),
    'A1': (256, 128),
    'B1': (128, 64, 32),
    'C2': (256, 128, 64, with_bn_only_layer1=True),
    'D1': (256, 128, 64, dropout_rates=[0, 0, 0]),
    'E2': (4→2 features version),
}

results = []
for name, config in variants.items():
    model = build_dnn_correlator(**config)
    train(model, X_train, y_train, X_val, y_val)
    metrics = evaluate(model, X_test, y_test)
    results.append({name, metrics})

df_results = pd.DataFrame(results)
plot_pareto_front(df_results)
print(df_results)
```

---

**Conclusão**: Esperamos encontrar que **Variante B1 (128→64→32)** oferece o melhor equilíbrio, reduzindo parâmetros em ~74% com degradação mínima de FNR. Isso permitiria deployment em devices com restrição de memória mantendo performance aceitável.
