# Results Archive — TAG-Authentication

**Data de Geração:** 2026-05-10T20:05:43.076196

## 📊 Resumo dos Resultados

Os 3 notebooks foram executados com sucesso sequencialmente:

1. **NN_02_DNN_Correlator_GPU.ipynb** ✅
   - Treino do CNN 1D para autenticação TAG
   - GPU-acelerado com mixed precision
   - Modelo: `model_dnn_correlator.keras`
   - Threshold: `cnn1d_threshold.json`

2. **NN_02b_4feat_DNN_Aligned.ipynb** ✅
   - Modelo alternativo com 4 features alinhadas
   - Modelo: `model_dnn_4feat_aligned.keras`
   - Scaler: `scaler_4feat_aligned.json`

3. **NN_08_Architecture_Comparison.ipynb** ✅
   - Comparação de arquiteturas
   - Visualizações: `NN08_Architecture_Comparison.png`

## 📁 Estrutura de Pastas

```
results/
├── data/
│   ├── dataset_cnn_yeq_0_30dB.h5       # Dataset principal
│   ├── dataset_nn_stratified_0_30dB.h5 # Dataset estratificado
│   ├── figure2_pd_vs_snr_gpu.json       # Métricas PD vs SNR
│   └── nn08_architecture_comparison.json # Comparação arquiteturas
├── models/
│   ├── cnn1d_tag_auth_best.keras       # Melhor modelo CNN
│   ├── cnn1d_threshold.json             # Threshold α-constrained
│   ├── model_dnn_4feat_aligned.keras   # Modelo 4-features
│   ├── model_dnn_correlator.h5         # Modelo legado
│   └── scaler_4feat_aligned.json        # Scaler para normalização
└── visualizations/
    ├── NN02_GPU_CNN1D_results.png           # Resultados NN_02
    ├── NN02b_4feat_aligned_results.png      # Resultados NN_02b
    ├── NN08_Architecture_Comparison.png     # Comparação NN_08
    ├── Figure2_PD_vs_SNR_GPU.png            # PD vs SNR
    ├── Figure2_DNN_vs_Classical_Auth-SUP.png # DNN vs Clássico
    └── [outros gráficos de análise]

_results_archive/
├── results_index.html       # Índice interativo
├── results_manifest.json    # Manifesto com metadados
└── README.md               # Este arquivo
```

## 🚀 Como Usar os Resultados

### Carregar um modelo treinado:
```python
import tensorflow as tf
model = tf.keras.models.load_model('results/models/cnn1d_tag_auth_best.keras')
```

### Usar o threshold otimizado:
```python
import json
with open('results/models/cnn1d_threshold.json') as f:
    threshold_data = json.load(f)
best_threshold = threshold_data['cnn_threshold']
```

### Explorar os dados:
```python
import h5py
with h5py.File('results/data/dataset_cnn_yeq_0_30dB.h5', 'r') as f:
    X_test = f['test/y_eq'][:]
    y_test = f['test/y'][:]
```

## 📊 Métricas Principais

### CNN 1D (NN_02):
- **AUC:** None
- **PD (α≤1e-3):** N/A
- **GPU-acelerado:** Sim (mixed_float16)

### 4-Feature Aligned (NN_02b):
- Status: ✅ Executado com sucesso
- Modelos e métricas salvos

### Architecture Comparison (NN_08):
- Status: ✅ Executado com sucesso
- Gráficos de comparação salvos

## 📋 Logs de Execução

- `execution_log_v2.txt` — Log completo do orquestrador
- `notebook_orchestrator_v2.py` — Script que executou tudo
- `monitor_progress.py` — Monitor de progresso em tempo real

## 🔍 Visualizar Resultados

1. **HTML Interativo:** Abra `_results_archive/results_index.html` em um navegador
2. **Manifesto JSON:** Veja `_results_archive/results_manifest.json` para dados estruturados
3. **Imagens:** Acesse `results/visualizations/` para todos os gráficos

## ⚙️ Próximas Etapas

Para análise futura:
1. ✅ Todos os modelos estão salvos e versionados
2. ✅ Datasets completos estão disponíveis
3. ✅ Visualizações e gráficos documentados
4. ✅ Métricas e thresholds registrados

Use os modelos para:
- Inferência em novos dados
- Fine-tuning para casos específicos
- Comparação com novos métodos
- Publicação de resultados

---

**Gerado:** 2026-05-10T20:05:43.076196
**Repositório:** /mnt/c/Users/thami/OneDrive/Documents/TAG-Authentication
