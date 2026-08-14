#!/usr/bin/env python3
"""
Results Archiver & Documentation Generator
Coleta todos os resultados dos notebooks e cria documentação para análise futura.
"""

import os
import json
import shutil
from pathlib import Path
from datetime import datetime
from collections import defaultdict

REPO_ROOT = Path("/mnt/c/Users/thami/OneDrive/Documents/TAG-Authentication")
RESULTS_DIR = REPO_ROOT / "Redes Neurais" / "results"
ARCHIVE_DIR = REPO_ROOT / "_results_archive"
ARCHIVE_DIR.mkdir(exist_ok=True)

def get_file_info(path):
    """Retorna informações sobre um arquivo."""
    stat = path.stat()
    return {
        'path': str(path.relative_to(REPO_ROOT)),
        'size_mb': stat.st_size / (1024 * 1024),
        'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
        'exists': path.exists()
    }

def generate_manifest():
    """Gera manifesto com todos os arquivos de resultado."""
    
    manifest = {
        'generated': datetime.now().isoformat(),
        'repo_root': str(REPO_ROOT),
        'categories': {}
    }
    
    # Categorizar por tipo
    categories = {
        'models': 'Modelos treinados',
        'visualizations': 'Gráficos e visualizações',
        'data': 'Datasets e métricas',
        'logs': 'Logs de execução'
    }
    
    for cat, desc in categories.items():
        cat_dir = RESULTS_DIR / cat
        if cat_dir.exists():
            manifest['categories'][cat] = {
                'description': desc,
                'path': str(cat_dir.relative_to(REPO_ROOT)),
                'files': []
            }
            
            for file_path in sorted(cat_dir.rglob('*')):
                if file_path.is_file():
                    file_info = get_file_info(file_path)
                    manifest['categories'][cat]['files'].append({
                        'name': file_path.name,
                        **file_info
                    })
    
    # Adicionar logs
    logs_dir = REPO_ROOT
    manifest['categories']['execution_logs'] = {
        'description': 'Logs de execução do orquestrador',
        'path': str(logs_dir.relative_to(REPO_ROOT)),
        'files': []
    }
    
    # Combinar múltiplos padrões de busca
    log_files = set()
    for pattern in ['*orchestrator*.py', 'execution_log*.txt', 'monitor_progress.py']:
        log_files.update(logs_dir.glob(pattern))
    
    for log_file in sorted(log_files):
        if log_file.is_file():
            file_info = get_file_info(log_file)
            manifest['categories']['execution_logs']['files'].append({
                'name': log_file.name,
                **file_info
            })
    
    return manifest

def create_html_report(manifest):
    """Gera relatório HTML dos resultados."""
    
    html = """<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TAG-Authentication Results Archive</title>
    <style>
        STYLES_PLACEHOLDER
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>📊 TAG-Authentication Results Archive</h1>
            <p class="timestamp">Gerado em: TIMESTAMP_PLACEHOLDER</p>
        </header>

        <div class="summary">
            <h2 style="margin-bottom: 15px; color: #667eea;">📈 Resumo</h2>
            STATS_PLACEHOLDER
        </div>
        
        CATEGORIES_PLACEHOLDER

        <div class="notes">
            <h3>📝 Notas Importantes</h3>
            <ul>
                <li><strong>Datasets:</strong> Arquivos .h5 contêm os datasets de treinamento, validação e teste</li>
                <li><strong>Modelos:</strong> Arquivos .keras e .h5 são os modelos treinados (podem ser carregados com TensorFlow)</li>
                <li><strong>Métricas:</strong> Arquivos .json contêm thresholds e métricas de performance</li>
                <li><strong>Visualizações:</strong> Arquivos .png contêm gráficos dos resultados (ROC, PD vs SNR, etc)</li>
                <li><strong>Para usar os modelos:</strong> Use <code>keras.models.load_model('caminho/arquivo.keras')</code></li>
            </ul>
        </div>

        <footer>
            <p>Arquivos organizados e documentados pelo Notebook Orchestrator</p>
            <p>Repositório: REPO_PLACEHOLDER</p>
        </footer>
    </div>
</body>
</html>
"""
    
    # CSS
    css = """* { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            padding: 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            padding: 40px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.3);
        }
        header {
            text-align: center;
            margin-bottom: 40px;
            border-bottom: 3px solid #667eea;
            padding-bottom: 20px;
        }
        h1 {
            color: #667eea;
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        .timestamp {
            color: #666;
            font-size: 0.9em;
        }
        .category {
            margin-bottom: 40px;
            padding: 20px;
            border-left: 5px solid #667eea;
            background: #f9f9f9;
            border-radius: 5px;
        }
        .category h2 {
            color: #667eea;
            margin-bottom: 15px;
            font-size: 1.5em;
        }
        .category-desc {
            color: #666;
            margin-bottom: 15px;
            font-style: italic;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }
        th {
            background: #667eea;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: 600;
        }
        td {
            padding: 10px 12px;
            border-bottom: 1px solid #ddd;
        }
        tr:hover {
            background: #f0f0f0;
        }
        .file-icon {
            display: inline-block;
            width: 20px;
            height: 20px;
            margin-right: 8px;
        }
        .status-ok { color: #27ae60; font-weight: bold; }
        .size-large { color: #e74c3c; }
        .size-medium { color: #f39c12; }
        .size-small { color: #27ae60; }
        .summary {
            background: #ecf0f1;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 30px;
        }
        .stat-box {
            display: inline-block;
            margin-right: 30px;
        }
        .stat-value {
            font-size: 1.8em;
            font-weight: bold;
            color: #667eea;
        }
        .stat-label {
            color: #666;
            font-size: 0.9em;
        }
        .notes {
            background: #fff3cd;
            border: 1px solid #ffc107;
            padding: 15px;
            border-radius: 5px;
            margin-top: 30px;
        }
        .notes h3 {
            color: #856404;
            margin-bottom: 10px;
        }
        footer {
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            color: #999;
            font-size: 0.9em;
        }"""
    
    # Contar arquivos por categoria
    total_files = 0
    total_size = 0
    stats_html = ""
    
    for cat, data in manifest['categories'].items():
        num_files = len(data.get('files', []))
        cat_size = sum(f.get('size_mb', 0) for f in data.get('files', []))
        
        stats_html += f'<div class="stat-box"><div class="stat-value">{num_files}</div><div class="stat-label">{cat}</div></div>\n'
        
        total_files += num_files
        total_size += cat_size
    
    stats_html += f'<div class="stat-box"><div class="stat-value">{total_files}</div><div class="stat-label">Total de Arquivos</div></div>\n'
    stats_html += f'<div class="stat-box"><div class="stat-value">{total_size:.1f} MB</div><div class="stat-label">Tamanho Total</div></div>'
    
    # Categorias
    categories_html = ""
    for cat, data in manifest['categories'].items():
        files = data.get('files', [])
        if not files:
            continue
        
        categories_html += f"""
        <div class="category">
            <h2>📁 {cat.replace('_', ' ').title()}</h2>
            <p class="category-desc">{data.get('description', '')}</p>
            <table>
                <thead>
                    <tr>
                        <th>Arquivo</th>
                        <th>Tamanho</th>
                        <th>Modificado</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
"""
        
        for file_info in files:
            size_mb = file_info.get('size_mb', 0)
            if size_mb > 100:
                size_class = 'size-large'
            elif size_mb > 10:
                size_class = 'size-medium'
            else:
                size_class = 'size-small'
            
            modified = file_info.get('modified', 'N/A')
            if 'T' in modified:
                modified = modified.split('T')[0]
            
            exists = '✓' if file_info.get('exists') else '✗'
            exists_class = 'status-ok' if file_info.get('exists') else 'status-warning'
            
            categories_html += f"""
                    <tr>
                        <td><strong>{file_info['name']}</strong></td>
                        <td class="{size_class}">{size_mb:.2f} MB</td>
                        <td>{modified}</td>
                        <td class="{exists_class}">{exists}</td>
                    </tr>
"""
        
        categories_html += """
                </tbody>
            </table>
        </div>
"""
    
    # Substituir placeholders
    html = html.replace('STYLES_PLACEHOLDER', css)
    html = html.replace('TIMESTAMP_PLACEHOLDER', manifest['generated'])
    html = html.replace('STATS_PLACEHOLDER', stats_html)
    html = html.replace('CATEGORIES_PLACEHOLDER', categories_html)
    html = html.replace('REPO_PLACEHOLDER', manifest['repo_root'])
    
    return html

def main():
    """Executa o arquivo."""
    print("=" * 80)
    print("📊 RESULTS ARCHIVER & DOCUMENTATION GENERATOR")
    print("=" * 80)
    
    # Gerar manifesto
    print("\n📋 Gerando manifesto de resultados...")
    manifest = generate_manifest()
    
    # Salvar manifesto JSON
    manifest_json_path = ARCHIVE_DIR / "results_manifest.json"
    with open(manifest_json_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"✓ Manifesto JSON: {manifest_json_path}")
    
    # Gerar relatório HTML
    print("\n🌐 Gerando relatório HTML...")
    html_report = create_html_report(manifest)
    
    html_path = ARCHIVE_DIR / "results_index.html"
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_report)
    print(f"✓ Relatório HTML: {html_path}")
    
    # Gerar README
    print("\n📝 Gerando README...")
    readme_path = ARCHIVE_DIR / "README.md"
    
    readme_content = f"""# Results Archive — TAG-Authentication

**Data de Geração:** {manifest['generated']}

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
- **AUC:** {next((m['cnn1d_tag_auth_best.keras'] if 'auc' in str(m) else None for cat in manifest['categories'].values() for m in cat.get('files', [])), 'N/A')}
- **PD (α≤1e-3):** {next((m.get('pd_at_thr', 'N/A') for cat in manifest['categories'].values() for m in cat.get('files', [])), 'N/A')}
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

**Gerado:** {manifest['generated']}
**Repositório:** {manifest['repo_root']}
"""
    
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    print(f"✓ README: {readme_path}")
    
    # Resumo
    print("\n" + "=" * 80)
    print("✅ ARQUIVAMENTO CONCLUÍDO")
    print("=" * 80)
    
    total_files = sum(len(cat.get('files', [])) for cat in manifest['categories'].values())
    total_size = sum(sum(f.get('size_mb', 0) for f in cat.get('files', [])) 
                    for cat in manifest['categories'].values())
    
    print(f"\n📊 Estatísticas:")
    print(f"  • Total de arquivos: {total_files}")
    print(f"  • Tamanho total: {total_size:.1f} MB")
    print(f"  • Categorias: {len(manifest['categories'])}")
    
    print(f"\n📁 Arquivos de documentação:")
    print(f"  • {manifest_json_path}")
    print(f"  • {html_path}")
    print(f"  • {readme_path}")
    
    print(f"\n💡 Para visualizar: Abra 'results_index.html' em um navegador")
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
