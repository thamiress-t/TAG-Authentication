# 📋 ABLATION STUDY - QUICK REFERENCE

## 🎯 OBJETIVO

Identificar quais componentes da rede DNN podem ser removidos/reduzidos sem comprometer a performance (FNR ≤ 10⁻⁷) para ganhar **eficiência computacional** (menos parâmetros, tempo, latência).

---

## 📊 AS 6 DIMENSÕES DE ABLATION

### 1. **Profundidade** (Número de Camadas)
```
Baseline: 3 camadas (256→128→64)
├─ A1: 2 camadas (256→128)       ← Variante otimista
├─ A2: 1 camada  (256)            ← Variante agressiva
└─ A3: 4 camadas (256→128→64→32)  ← Variante robustez

Pergunta: É necessário usar 3 camadas?
Esperado: A1 deve ser similar ao Baseline (camada 3 é refinamento)
```

### 2. **Largura** (Número de Neurônios)
```
Baseline: 256→128→64
├─ B1: 128→64→32  (50% menos)      ← Muito promissor para edge
├─ B2: 64→32→16   (75% menos)      ← Muito agressivo
└─ B3: 192→96→48  (Intermediário)

Pergunta: Quantos neurônios realmente precisamos?
Esperado: B1 pode degradar FNR 10x mas manter robusto (10^-6 é OK?)
```

### 3. **BatchNormalization**
```
Baseline: BatchNorm após cada Dense (exceto output)
├─ C1: Sem BatchNorm              ← Remove todas
└─ C2: BatchNorm só primeira layer ← Remove parcialmente

Pergunta: É BatchNorm realmente necessário?
Esperado: Crítico! Sem BatchNorm = convergência lenta + FNR piora
```

### 4. **Regularização (Dropout)**
```
Baseline: Dropout = (0.3, 0.3, 0.2)
├─ D1: Sem Dropout (0, 0, 0)       ← Permite overfitting
├─ D2: Dropout leve (0.1, 0.1, 0.1)
└─ D3: Dropout forte (0.5, 0.5, 0.3)

Pergunta: Qual nível de Dropout é ótimo?
Esperado: Baseline (0.3, 0.3, 0.2) é ótimo para 80k amostras
```

### 5. **Funções de Ativação** (Não implementado ainda)
```
Baseline: ReLU em todas as camadas ocultas
├─ F1: Tanh
├─ F2: LeakyReLU
└─ F3: Sigmoid em hidden layers

Pergunta: ReLU é ótimo ou há alternativas?
Esperado: ReLU é padrão e deve ser ótimo
```

### 6. **Features de Entrada** (Número de variáveis)
```
Baseline: 4 features [τ_corr, ĥ_estimate, SNR_local, energy]
├─ E1: 3 features (remove energy)
├─ E2: 2 features (τ + ĥ apenas)
└─ E3: 1 feature  (τ apenas)

Pergunta: Quantas features são realmente necessárias?
Esperado: 4 features é redundante? E pode ser crítico para robustez
```

---

## 🔢 TOTAL DE VARIANTES A TESTAR

```
Dimensão       Variantes
──────────────────────────
Profundidade   3 (A1, A2, A3)
Largura        3 (B1, B2, B3)
BatchNorm      2 (C1, C2)
Dropout        3 (D1, D2, D3)
─────────────────────────
TOTAL          14 variantes (além do Baseline)

Tempo estimado: 3-4 horas (CPU) ou 30 min (GPU)
Runs por variante: 3 (para média ± std)
```

---

## 📈 MÉTRICAS A COLETAR

Para cada variante, medir:

| Métrica | Unidade | Range | Interpretação |
|---------|---------|-------|----------------|
| **num_parameters** | Count | 1k-50k | Memória necessária |
| **fnr** | Ratio | 1e-9 a 1e-2 | ↓ Lower is better |
| **auc** | Score | 0.0-1.0 | ↑ Higher is better |
| **accuracy** | % | 0-100% | Métrica de negócio |
| **training_time** | seconds | 100-1000 | ↓ Menor é melhor |
| **inference_latency** | ms/sample | 0.1-10 | ↓ Crítico para edge |
| **efficiency_score** | Custom | 0-5 | ↑ Métrica composta |

---

## 🎬 CHECKLIST DE EXECUÇÃO

### PRÉ-EXECUÇÃO
- [ ] Dataset gerado? `Redes Neurais/results/data/dataset_nn_stratified_0_30dB.h5`
- [ ] Ambiente virtual ativado? `venv_nn` ou `.venv-1`
- [ ] TensorFlow/Keras instalado? `pip list | grep tensorflow`
- [ ] GPU disponível? (Opcional, CPU funciona)

### EXECUÇÃO
```bash
# Comando completo
python Redes\ Neurais/ablation_study_dnn.py \
    --dataset "Redes Neurais/results/data/dataset_nn_stratified_0_30dB.h5" \
    --output "Redes Neurais/results/ablation_results" \
    --runs 3
```

### PÓS-EXECUÇÃO
- [ ] Resultados em `Redes Neurais/results/ablation_results/`
- [ ] Abrir `ablation_results.csv` e revisar
- [ ] Visualizar `ablation_pareto.png`
- [ ] Identificar melhor variante por caso de uso
- [ ] Documentar conclusões

---

## 🎯 CRITÉRIOS DE SELEÇÃO

### Para **PESQUISA** (máxima precisão)
```
Prioridade: FNR ≤ 10^-7

Recomendado:
├─ Baseline (referência)
└─ A3 (4 camadas, mais robustez)

FNR degradation aceitável: < 5%
```

### Para **PRODUÇÃO** (balance)
```
Prioridade: FNR ≤ 10^-6, latência < 5ms

Recomendado:
├─ A1 (2 camadas, -24% params)
└─ B1 (50% menos neurônios, -74% params)

Esperado:
├─ A1: FNR ≈ 1.1e-7, latência ≈ 0.95ms ✅
└─ B1: FNR ≈ 1.5e-6, latência ≈ 0.6ms ⭐
```

### Para **EDGE DEVICE** (máxima compressão)
```
Prioridade: Parâmetros ≤ 10k, latência ≤ 1ms

Recomendado:
├─ B1 (128→64→32, ~11k params)
├─ B2 (64→32→16, ~3k params) [risky]
└─ E1/E2 (reduzir features)

Esperado:
├─ B1: -74% params, -40% latência, FNR +1400% (but still 10^-6)
└─ B2: -93% params, -60% latência, FNR +3200% (muito ruim)
```

### Para **TEMPO-REAL** (mínima latência)
```
Prioridade: Latência < 1ms, throughput > 1000 samples/sec

Recomendado:
├─ B1 + B2 (comparar)
├─ D1 + B1 (sem Dropout, mais rápido)
└─ E2 (2 features apenas)

Esperado:
├─ B1: ~0.6ms/sample (1667 samples/sec)
└─ D1: ~0.35ms/sample (2857 samples/sec) [but overfitting]
```

---

## 📊 ESPERADO: TOP 3 VARIANTES

Baseado em análise teórica:

### 🥇 **Rank 1: B1 (128→64→32)**
```
└─ Casos de uso: Edge devices, IoT, Mobile
├─ Parâmetros: 11,000 (-74%)
├─ FNR esperado: 10^-6 (degradação aceitável)
├─ Latência: 0.6ms/sample (-40%)
└─ Eficiência Score: ⭐⭐⭐⭐⭐ (2.85+)
```

### 🥈 **Rank 2: A1 (256→128)**
```
└─ Casos de uso: Production, Balance
├─ Parâmetros: 33,000 (-24%)
├─ FNR esperado: 10^-7 (praticamente igual)
├─ Latência: 0.95ms/sample (-5%)
└─ Eficiência Score: ⭐⭐⭐⭐ (3.21)
```

### 🥉 **Rank 3: Baseline**
```
└─ Casos de uso: Research, Maximum Security
├─ Parâmetros: 43,265 (referência)
├─ FNR esperado: 10^-7 (ótimo)
├─ Latência: 1.0ms/sample (referência)
└─ Eficiência Score: 1.00 (baseline)
```

---

## ⚠️ ARMADILHAS COMUNS

| Armadilha | Como Evitar | Risco |
|-----------|-------------|-------|
| B2 (75% redução) é "bom demais" | Validar em dados OOD | FNR pode ser 10^-4 em realidade |
| D1 (sem Dropout) funciona em train | Monitorar Val FNR | Overfitting não detectado visualmente |
| E2 (2 features) é OK em dados limpos | Testar com anomalias | Falha em impulsos/outliers |
| Comparar FNR linear vs. log | Sempre use escala log | -3 vs. -7 é mesma magnitude |
| Não rastrear seed aleatória | Set seed=42+run | Resultados irreplicáveis |

---

## 📋 TEMPLATE: PREENCHIMENTO APÓS EXECUÇÃO

Após rodar o ablation study, preencher:

```markdown
## Ablation Study Results - [DATE]

### Ambiente
- Dataset: dataset_nn_stratified_0_30dB.h5
- Samples: [train/val/test]
- GPU: [Yes/No]
- Tempo total: [X horas]

### Ranking (por Efficiency Score)
1. [Variante]: Score=[X], FNR=[Y], Params=[Z]
2. [Variante]: Score=[X], FNR=[Y], Params=[Z]
3. [Variante]: Score=[X], FNR=[Y], Params=[Z]

### Recomendação Final
Para [caso de uso]:
- **Variante**: [X]
- **Trade-off**: [Y% redução, Z% degradação FNR]
- **Justificativa**: [...]

### Próximos Passos
- [ ] Retreinar com nova configuração
- [ ] Validar em dados OOD
- [ ] Comparar com Baseline 2021
- [ ] Documentar em ARQUITETURA_DNN_EXPLICACAO.md
```

---

## 🔗 ARQUIVOS RELACIONADOS

```
Redes Neurais/
├── ABLATION_STUDY_PLAN.md (este arquivo)
├── ablation_study_dnn.py (script executável)
├── COMO_EXECUTAR_ABLATION_STUDY.md (instruções detalhadas)
├── ANALISE_ESPERADA_ABLATION.md (análise de hipóteses)
├── ARQUITETURA_DNN_EXPLICACAO.md (contexto do baseline)
├── notebooks/
│   ├── NN_01_DataGeneration.ipynb (gerar dataset)
│   ├── NN_02_DNN_Correlator.ipynb (baseline)
│   └── NN_06_Comparison_vs_Baseline.ipynb (validação)
└── results/
    ├── ablation_results/ (OUTPUT AQUI)
    └── data/
        └── dataset_nn_stratified_0_30dB.h5 (INPUT)
```

---

## 💡 DICAS PRO

1. **Executar com --runs 1 primeiro** para testes rápidos
2. **Visualizar CSV em pandas** para exploração rápida
3. **Pareto plot é a visualização mais importante**
4. **Guardar melhor modelo com** `model.save()`
5. **Repetir ablation com dados OOD** para validação real

---

**Status**: 🟢 Pronto para execução  
**Última atualização**: Abril 2026  
**Versão**: 1.0
