# 🔬 ANÁLISE TÉCNICA: ESPERADO DO ABLATION STUDY

## Hipóteses e Análises Esperadas

---

## 📊 HIPÓTESE 1: Redução de Profundidade (Camadas)

### A1: Remover última camada (256→128 instead of 256→128→64)

**Justificativa Teórica**:

```
Camada 3 (64→1):
├─ Input: 128 features já refinadas (após ReLU+BN)
├─ Output: Score final
├─ Parâmetros: 128×64 + 64 = 8,256 (19% do total)
└─ Questão: É essencial ou redundante?

Hipótese: 
├─ Se 256→128 já produz features discriminativas
├─ E 128 neurônios já comprimem bem a informação
├─ Então 64 é "overkill"
└─ Impacto esperado: Pequeno (-1 a +5% em FNR)
```

**Análise Esperada**:

```
RESULTADO ESPERADO:
├─ Parâmetros: 43k → 33k (-24%)
├─ Tempo treino: 480s → 410s (-15%)
├─ Tempo inf: 1.0ms → 0.95ms (-5%)
├─ FNR: 1.0e-7 → 1.1e-7 (+10%)
└─ Conclusão: ✂️ Camada 3 pode ser removida (GANHO DE EFICIÊNCIA)
```

**Quando isso falharia?**:
- Se Large Deviations Theory predia que 3 camadas são CRÍTICAS
- Mas empiricamente: Baseline já usa ReLU+BatchNorm×2
  - Profundidade efetiva é alta mesmo com 2 camadas
  - Terceira camada é principalmente "refinamento"

---

## 📊 HIPÓTESE 2: Redução de Largura (Neurônios)

### B1: 50% menos neurônios (128→64→32 instead of 256→128→64)

**Justificativa Teórica**:

```
Camada 1: 256 neurônios para 4 inputs
├─ Razão: 256/4 = 64× expansão
├─ Pergunta: É necessário expandir 64x?
├─ Alternativa: 128× expansão (128/4=32) ainda seria muito

Análise:
├─ Informação de 4 features é limitada
├─ Cada feature já é uma estatística condensada (τ, ĥ, SNR, E)
├─ Não há padrões locais a explorar (diferentemente de imagens)
└─ Logo: 128→64→32 pode ser suficiente
```

**Análise Esperada**:

```
RESULTADO ESPERADO:
├─ Parâmetros: 43k → 11k (-74%)
├─ Tempo treino: 480s → 240s (-50%)
├─ Tempo inf: 1.0ms → 0.6ms (-40%)
├─ FNR: 1.0e-7 → 1.5e-6 (+1400%)
│   ├─ Sem decimais: 10^-7 → 10^-6 (piora ~10x)
│   └─ Mas ainda < 10^-5 (aceitável!)
└─ GANHO: Enorme eficiência com degradação ACEITÁVEL
```

**Crítico**: Qual é o limite de FNR aceitável?

```
Contexto do projeto:
├─ Target: FNR ≤ 10⁻⁷
├─ Baseline 2021: 10⁻⁷
├─ Ensemble Braca: 10⁻⁷·⁵

Se B1 atinge 10⁻⁶:
├─ Piora em log10: -7 → -6 = +1 ordem (10x)
├─ Em termos práticos: 1 falso negativo em 10M vs. 1 em 100M
├─ Contexto: Autenticação TAG em sistemas embarcados
│   ├─ Se usar 1M TAGs/dia: espera 1 falso a cada 10-100 dias
│   └─ Trade-off: -40% latência vs. +10x FNR

DECISÃO:
├─ Se latência é crítica (tempo-real): USE B1
├─ Se security é crítica (zero falsos): USE Baseline
├─ Se pesquisa acadêmica: USE Baseline + Ensemble
```

---

## 📊 HIPÓTESE 3: Importância de BatchNormalization

### C1: Remover todas as BatchNorm / C2: BatchNorm apenas 1ª camada

**Justificativa Teórica**:

```
BatchNormalization custa:
├─ Parâmetros: 2×(units per layer) = 2×(256+128+64) = 896
├─ Tempo computacional: +5-10%
├─ Benefício: Estabiliza gradientes

Questão:
├─ Com Dropout forte (0.3, 0.3, 0.2), BatchNorm ainda é necessário?
├─ Dropout já regulariza - é redundante?
└─ Em batch_size=256, ruidoso suficiente?

Hipótese:
└─ BatchNorm é ESSENCIAL para:
   ├─ Convergência rápida (nossa observação em NN_02)
   ├─ Estabilidade em múltiplas épocas
   ├─ Generalização
   └─ Remover = underfitting / convergência lenta
```

**Análise Esperada**:

```
RESULTADO ESPERADO:

C1: Sem BatchNorm
├─ Parâmetros: 43k → 43k (mesmo)
├─ Tempo treino: 480s → 600s (+25%, sem BN é mais lento)
├─ FNR: 1.0e-7 → 5.0e-7 (+400%)
│   └─ Razão: Convergência ruim, instabilidade
└─ Conclusão: ❌ NÃO RECOMENDADO

C2: BatchNorm apenas primeira camada
├─ Parâmetros: 43k → 43k - 512 ≈ 43k (negligível -1%)
├─ Tempo treino: 480s → 470s (-2%, pouco ganho)
├─ FNR: 1.0e-7 → 2.5e-7 (+150%)
│   └─ Razão: 2ª camada instável sem normalização
└─ Conclusão: ⚠️ MARGINAL, não vale a pena
```

**Insight**:
- BatchNorm é crítico após ReLU (mitiga dead neurons)
- Remover todas = erro
- Remover parcialmente = pouco ganho
- ✅ **Conclusão**: Mantenha BatchNorm em todas as camadas

---

## 📊 HIPÓTESE 4: Importância de Dropout

### D1: Sem Dropout / D2: Dropout Leve / D3: Dropout Forte

**Justificativa Teórica**:

```
Regularização em dataset grande:
├─ Train: 80k amostras
├─ Ratio: 43k parâmetros / 80k amostras = 0.54
├─ Risco: Overfitting MÉDIO (não crítico, mas presente)

Dropout force:
├─ 0.0: Sem proteção (baseline tem 0.3)
├─ 0.1: Leve (baseline)
├─ 0.3: Moderado (baseline)
├─ 0.5: Agressivo (causaria underfitting)

Expectativa:
├─ Sem Dropout: Overfitting em treino, pior generalização
├─ Leve: Pode funcionar com 80k amostras
├─ Moderado: Ótimo balanço
├─ Forte: Underfitting (perde capacidade)
```

**Análise Esperada**:

```
RESULTADO ESPERADO:

D1: Sem Dropout (0, 0, 0)
├─ Parâmetros: 43k (mesmo)
├─ Tempo treino: 480s → 420s (-12%, sem cálculo dropout)
├─ Train FNR: 1.0e-8 (muito bom!)
├─ Val FNR: 1.5e-6 (ruim!)
├─ Test FNR: 1.0e-6 (+900%)
│   └─ OVERFITTING DETECTADO
└─ Conclusão: ❌ Sem regularização = overfit

D2: Dropout Leve (0.1, 0.1, 0.1)
├─ Parâmetros: 43k (mesmo)
├─ Tempo treino: 480s (mesmo)
├─ Test FNR: 3.0e-7 (+200%)
│   └─ Regularização leve, ainda funciona
└─ Conclusão: ⚠️ Poderia usar, mas não é melhor

D3: Dropout Forte (0.5, 0.5, 0.3)
├─ Parâmetros: 43k (mesmo)
├─ Tempo treino: 480s → 550s (+15%, mais computação)
├─ Test FNR: 5.0e-7 (+400%)
│   └─ Underfitting leve
└─ Conclusão: ⚠️ Agressivo demais
```

**Insight**:
- Dropout moderado (0.3, 0.3, 0.2) é ótimo para 80k samples
- Sem Dropout = overfitting detectável
- ✅ **Conclusão**: Mantenha Dropout como está (0.3, 0.3, 0.2)

---

## 📊 HIPÓTESE 5: Número de Features

### E2: Apenas 2 features (τ e ĥ) instead of 4

**Justificativa Teórica**:

```
Features no Baseline:
├─ τ: Correlator (estatística clássica ótima)
├─ ĥ: Estimativa de canal (detecta ataque)
├─ SNR: Contexto de qualidade
└─ E: Energia robusta

Qual é redundante?
├─ τ + ĥ: Suficientes? (teoria de Braca pode concordar)
├─ SNR: Ajuda em adaptação a diferentes regimes
├─ E: Robustez a anomalias (impulsos, outliers)

Risco de remover:
└─ Perda de robustez a model misspecification
   ├─ Se canal não é Rayleigh exato: ĥ adapta
   ├─ Se ruído tem impulsos: E detecta
   └─ Se SNR varia: SNR contextualiza
```

**Análise Esperada**:

```
RESULTADO ESPERADO (E2: τ + ĥ apenas):

Em dados NORMAIS (como gerados):
├─ Parâmetros: 43k → 20k (-53%)
├─ Tempo treino: 480s → 320s (-33%)
├─ Test FNR: 1.0e-7 → 1.2e-7 (+20%)
│   └─ Praticamente nenhuma degradação!
└─ Conclusão: ✂️ SNR + E podem ser redundantes em dados limpos

MAS em dados COM ANOMALIAS (simulado adicional):
├─ Teste: Adicione impulsos ao test set
├─ Resultado:
│   ├─ E2 (sem E feature): FNR→ 2.0e-5 (RUIM)
│   └─ Baseline (com E): FNR→ 1.5e-6 (OK)
└─ Conclusão: ⚠️ E feature É importante para robustez
```

**Crítico - Decisão**:

```
Se você valida APENAS em dados sintéticos limpos:
  └─ E2 pode parecer bom (reduz parâmetros 53%)

Se você testa em dados REAIS (com anomalias):
  └─ E feature é crítico para robustez

RECOMENDAÇÃO:
├─ Para pesquisa (dados idealizados): E2 está OK
├─ Para produção (dados reais): Mantenha 4 features
└─ Compromise: Manter E, remover SNR? (precisaria testar)
```

---

## 📊 RESUMO ESPERADO

### Tabela de Impactos Esperados

| Componente | Remover Impacto | Status | Ação |
|-----------|-----------------|--------|------|
| **Camada 3 (64 neurônios)** | FNR +10% | Redundante | ✂️ Pode remover (A1) |
| **50% dos neurônios** | FNR +1400% | Trade-off | ⚠️ Depende do caso (B1) |
| **BatchNormalization** | FNR +300% | Crítico | ✅ Manter |
| **Dropout** | FNR +900% (overfit) | Crítico | ✅ Manter |
| **SNR + E features** | FNR +20% (limpo) / +1000% (anomalias) | Importante | ✅ Manter (E sim, SNR débil) |

### Matriz de Decisão

```
USE CASE                    RECOMENDAÇÃO           TRADE-OFF
────────────────────────────────────────────────────────────
Pesquisa (Max Perf)        Baseline               FNR=10^-7
Produção (Balance)         A1 (2 camadas)        -24% params, FNR+10%
Edge/Mobile (Min Size)     B1 (128-64-32)        -74% params, FNR+1400% (10^-6)
Tempo-Real (<10ms inf)     B1 + B2 testes        Otimizar latência
Robustez/OOD (Pesquisa)    A3 (4 camadas)        +30% params, FNR-10%
```

---

## 🔬 COMO VALIDAR HIPÓTESES

### Experimento 1: Validação de Profundidade

```python
# Run A1, A2, A3 e plotar:
depth = [1, 2, 3, 4]
fnr_list = [...]  # FNR de cada profundidade
params_list = [...]  # Parâmetros

# Expectativa:
├─ FNR melhora até profundidade=3
├─ Depois platô (função de utilidade decrescente)
└─ Converge no baseline (3 camadas ótimo)
```

### Experimento 2: Validação de Regularização

```python
# Comparar train FNR vs test FNR:
# C1 (sem BN): gap grande (overfit)
# D1 (sem Dropout): gap grande (overfit)
# Baseline: gap pequeno (bom generalization)

# Plot:
plt.plot(epochs, train_loss, label='Train')
plt.plot(epochs, val_loss, label='Val')
# Expectativa:
# └─ D1: train↓↓ mas val↗↗ (divergência = overfit)
```

### Experimento 3: Validação de Robustez

```python
# Testar em datasets OOD:
# E.g., treinar em SNR=10dB, testar em SNR=5dB

test_snrs = [5, 10, 15, 20]
fnr_baseline = [...]
fnr_e2 = [...]

# Expectativa:
# └─ E2 (sem E feature) = pior em SNR baixo (confirmando importância)
```

---

## 🎬 PRÓXIMOS PASSOS APÓS ABLATION

1. **Rodar ablation study** (3-4 horas)
2. **Validar hipóteses** acima com dados reais
3. **Fazer ablation adicional** se surpreso:
   - Ex: Se B1 for bom demais, testar B1.5 (110→55→27)
4. **Escolher melhor variante** por caso de uso
5. **Retreinar e documentar** novo modelo recomendado
6. **Publicar resultados** (ablation study é excelente para papers!)

---

**Conclusão Esperada**: A redução para 2 camadas (A1) ou 50% neurônios (B1) oferecerá ganho significativo em eficiência com degradação marginal ou aceitável em FNR, dependendo do seu caso de uso específico.
