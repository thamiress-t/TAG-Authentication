#!/usr/bin/env python3
"""
Notebook Orchestrator v2 - ROBUSTO
Executa notebooks com tratamento de crashes e continua mesmo com falhas parciais.
"""

import os
import sys
import time
import subprocess
import json
from pathlib import Path
from datetime import datetime

REPO_ROOT = Path("/mnt/c/Users/thami/OneDrive/Documents/TAG-Authentication")
NOTEBOOKS_DIR = REPO_ROOT / "Redes Neurais" / "notebooks"

NOTEBOOKS = [
    "NN_02_DNN_Correlator_GPU.ipynb",
    "NN_02b_Ablation_tau_variants.ipynb",
    "NN_08_Architecture_Comparison.ipynb",
]

EXECUTION_LOG = REPO_ROOT / "execution_log_v2.txt"

def log(message):
    """Log com timestamp."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"[{ts}] {message}"
    print(log_msg)
    with open(EXECUTION_LOG, "a") as f:
        f.write(log_msg + "\n")

def check_notebook_completion(notebook_name):
    """Verifica se um notebook foi completado verificando os outputs salvos."""
    notebook_path = NOTEBOOKS_DIR / notebook_name
    
    # Detectar padrões de saída esperados em cada notebook
    if "NN_02" in notebook_name and not "NN_02b" in notebook_name:
        # Verificar se modelo foi salvo
        expected_files = [
            REPO_ROOT / "Redes Neurais" / "results" / "models" / "cnn1d_tag_auth_best.keras",
            REPO_ROOT / "Redes Neurais" / "results" / "models" / "cnn1d_threshold.json"
        ]
        return all(f.exists() for f in expected_files)
    
    # Para outros notebooks, tentar ler e verificar última célula executada
    try:
        import json as json_lib
        with open(notebook_path) as f:
            nb = json_lib.load(f)
        
        # Contar células com execution_count
        executed_cells = sum(1 for cell in nb.get('cells', []) 
                            if cell.get('execution_count') is not None)
        total_cells = len([c for c in nb.get('cells', []) if c['cell_type'] == 'code'])
        
        # Se 80%+ das células foram executadas, consideramos ok
        if total_cells > 0:
            completion = executed_cells / total_cells
            log(f"  {notebook_name}: {executed_cells}/{total_cells} células ({completion*100:.0f}%)")
            return completion >= 0.8
    except Exception as e:
        log(f"  Erro ao verificar {notebook_name}: {e}")
    
    return False

def execute_notebook_papermill(notebook_name, index, timeout_sec=3600):
    """Executa notebook com papermill."""
    notebook_path = NOTEBOOKS_DIR / notebook_name
    output_path = NOTEBOOKS_DIR / f"{notebook_name.replace('.ipynb', '')}_executed.ipynb"
    
    log(f"\n📓 ETAPA {index}: Executando {notebook_name}...")
    log(f"  Timeout: {timeout_sec}s ({timeout_sec//60}m)")
    
    start_time = time.time()
    
    try:
        cmd = [
            "papermill",
            str(notebook_path),
            str(output_path),
            "-k", "python",
            "-r", "TIMEOUT_MODE", "true"
        ]
        
        log(f"  🚀 {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            timeout=timeout_sec,
            capture_output=True,
            text=True
        )
        
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            log(f"✅ {notebook_name} concluído em {elapsed:.1f}s")
            return True
        else:
            log(f"⚠️ Retorno não-zero ({result.returncode}). Verificando se completou...")
            log(f"STDERR (últimas 300 chars): {result.stderr[-300:]}")
            
            # Verificar se apesar do erro, o notebook progrediu
            if check_notebook_completion(notebook_name):
                log(f"✅ Notebook parece ter completado (arquivos salvos). Continuando...")
                return True
            else:
                log(f"❌ Notebook não completou.")
                return False
                
    except subprocess.TimeoutExpired:
        log(f"⏱️ TIMEOUT após {timeout_sec}s ({timeout_sec//60}m)")
        if check_notebook_completion(notebook_name):
            log(f"✅ Mas arquivos foram salvos. Continuando...")
            return True
        return False
    except FileNotFoundError:
        log(f"⚠️ papermill não encontrado. Tentando jupyter nbconvert...")
        
        try:
            cmd = [
                "jupyter",
                "nbconvert",
                "--to", "notebook",
                "--execute",
                "--inplace",
                str(notebook_path),
                f"--ExecutePreprocessor.timeout={timeout_sec}"
            ]
            
            log(f"  🚀 {' '.join(cmd)}")
            result = subprocess.run(cmd, timeout=timeout_sec+60, capture_output=True, text=True)
            
            elapsed = time.time() - start_time
            
            if result.returncode == 0:
                log(f"✅ {notebook_name} concluído em {elapsed:.1f}s")
                return True
            else:
                log(f"⚠️ nbconvert retornou {result.returncode}")
                log(f"STDERR: {result.stderr[-300:]}")
                if check_notebook_completion(notebook_name):
                    log(f"✅ Arquivos salvos. Continuando...")
                    return True
                return False
        except Exception as e:
            log(f"❌ Erro com nbconvert: {e}")
            return False
    except Exception as e:
        log(f"❌ Erro inesperado: {e}")
        return False

def main():
    """Orquestrador robusto."""
    log("=" * 80)
    log("🎯 NOTEBOOK ORCHESTRATOR v2 (ROBUSTO)")
    log("=" * 80)
    
    os.chdir(REPO_ROOT)
    
    # Verificar notebooks
    log("\n📋 Verificando disponibilidade dos notebooks...")
    for nb in NOTEBOOKS:
        path = NOTEBOOKS_DIR / nb
        exists = "✓" if path.exists() else "❌"
        log(f"  {exists} {nb}")
    
    results = {}
    
    # ETAPA 1: NN_02 (pode estar em execução ou já completado)
    log(f"\n📊 ETAPA 1: Analisando NN_02_DNN_Correlator_GPU.ipynb...")
    if check_notebook_completion(NOTEBOOKS[0]):
        log(f"✅ NN_02 já foi completado!")
        results[NOTEBOOKS[0]] = "✅ Completado"
    else:
        log(f"⏳ NN_02 ainda está sendo executado ou não completou.")
        log(f"   Tentando re-executar...")
        success = execute_notebook_papermill(NOTEBOOKS[0], 1, timeout_sec=1800)
        results[NOTEBOOKS[0]] = "✅ Re-executado" if success else "❌ Falha"
    
    # ETAPA 2: NN_02b
    if "❌" not in results[NOTEBOOKS[0]]:
        success = execute_notebook_papermill(NOTEBOOKS[1], 2, timeout_sec=1800)
        results[NOTEBOOKS[1]] = "✅ Executado" if success else "❌ Falha"
    else:
        log(f"\n⚠️ ETAPA 2 saltada (ETAPA 1 falhou)")
        results[NOTEBOOKS[1]] = "⏭️ Saltado"
    
    # ETAPA 3: NN_08
    if "❌" not in results.get(NOTEBOOKS[1], ""):
        success = execute_notebook_papermill(NOTEBOOKS[2], 3, timeout_sec=2400)
        results[NOTEBOOKS[2]] = "✅ Executado" if success else "❌ Falha"
    else:
        log(f"\n⚠️ ETAPA 3 saltada (ETAPA 2 falhou)")
        results[NOTEBOOKS[2]] = "⏭️ Saltado"
    
    # RELATÓRIO
    log("\n" + "=" * 80)
    log("📊 RELATÓRIO FINAL")
    log("=" * 80)
    for i, (nb, status) in enumerate(results.items(), 1):
        log(f"  ETAPA {i}: {status}  {nb}")
    
    all_success = all("❌" not in s for s in results.values())
    
    if all_success:
        log("\n🎉 Execução concluída com sucesso!")
    else:
        log("\n⚠️ Algumas etapas falharam ou foram saltadas.")
    
    log("=" * 80)
    
    return 0 if all_success else 1

if __name__ == "__main__":
    sys.exit(main())
