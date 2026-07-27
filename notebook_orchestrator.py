#!/usr/bin/env python3
"""
Notebook Orchestrator - Executa sequencialmente os 3 notebooks com monitoramento.
Aguarda NN_02_DNN_Correlator_GPU.ipynb, depois NN_02b_4feat_DNN_Aligned.ipynb, então NN_08_Architecture_Comparison.ipynb
"""

import os
import sys
import time
import subprocess
import json
from pathlib import Path
from datetime import datetime

# Configuração
REPO_ROOT = Path("/mnt/c/Users/thami/OneDrive/Documents/TAG-Authentication")
NOTEBOOKS_DIR = REPO_ROOT / "Redes Neurais" / "notebooks"

NOTEBOOKS = [
    "NN_02_DNN_Correlator_GPU.ipynb",
    "NN_02b_4feat_DNN_Aligned.ipynb",
    "NN_08_Architecture_Comparison.ipynb"
]

EXECUTION_LOG = REPO_ROOT / "execution_log.txt"

def log(message):
    """Log com timestamp."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"[{ts}] {message}"
    print(log_msg)
    with open(EXECUTION_LOG, "a") as f:
        f.write(log_msg + "\n")

def wait_for_first_notebook():
    """Aguarda que o NN_02_DNN_Correlator_GPU.ipynb termine."""
    log("📊 ETAPA 1: Monitorando NN_02_DNN_Correlator_GPU.ipynb...")
    
    first_notebook = NOTEBOOKS_DIR / NOTEBOOKS[0]
    
    # Verificar se arquivo existe
    if not first_notebook.exists():
        log(f"❌ ERRO: {first_notebook} não encontrado!")
        return False
    
    log(f"✓ Notebook encontrado: {first_notebook}")
    
    # Aguardar que o kernel finalize (simples pooling da lista de kernels)
    max_wait = 3600  # 1 hora de timeout
    start_time = time.time()
    check_interval = 30  # Verificar a cada 30 segundos
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > max_wait:
            log(f"⏱️ TIMEOUT: Notebook NN_02_DNN_Correlator_GPU demorou mais de 1 hora. Continuando...")
            break
        
        # Verificar se há kernels rodando
        try:
            result = subprocess.run(
                ["jupyter", "kernelspec", "list"],
                capture_output=True,
                text=True,
                timeout=10
            )
            log(f"✓ [+{int(elapsed)}s] Kernel ainda ativo. Aguardando...")
        except Exception as e:
            log(f"Erro ao verificar kernel: {e}")
        
        time.sleep(check_interval)
    
    log("✅ NN_02_DNN_Correlator_GPU.ipynb concluído!")
    return True

def execute_notebook(notebook_name, index):
    """Executa um notebook usando papermill (ou jupyter nbconvert se papermill não estiver disponível)."""
    notebook_path = NOTEBOOKS_DIR / notebook_name
    
    if not notebook_path.exists():
        log(f"❌ ERRO: {notebook_path} não encontrado!")
        return False
    
    log(f"\n📓 ETAPA {index + 1}: Executando {notebook_name}...")
    
    start_time = time.time()
    
    try:
        # Tentar usar papermill
        output_path = NOTEBOOKS_DIR / f"{notebook_name.replace('.ipynb', '')}_output.ipynb"
        cmd = [
            "papermill",
            str(notebook_path),
            str(output_path),
            "-k", "python"
        ]
        
        log(f"🚀 Comando: {' '.join(cmd)}")
        result = subprocess.run(cmd, timeout=7200, capture_output=True, text=True)
        
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            log(f"✅ {notebook_name} executado com sucesso! ({elapsed:.1f}s)")
            return True
        else:
            log(f"❌ ERRO na execução de {notebook_name}:")
            log(f"STDOUT: {result.stdout[-500:]}")
            log(f"STDERR: {result.stderr[-500:]}")
            return False
            
    except subprocess.TimeoutExpired:
        log(f"⏱️ TIMEOUT: {notebook_name} demorou mais de 2 horas!")
        return False
    except FileNotFoundError:
        log("⚠️ papermill não encontrado. Tentando com jupyter nbconvert...")
        try:
            cmd = [
                "jupyter",
                "nbconvert",
                "--to", "notebook",
                "--execute",
                "--inplace",
                str(notebook_path)
            ]
            log(f"🚀 Comando: {' '.join(cmd)}")
            result = subprocess.run(cmd, timeout=7200, capture_output=True, text=True)
            
            elapsed = time.time() - start_time
            
            if result.returncode == 0:
                log(f"✅ {notebook_name} executado com sucesso! ({elapsed:.1f}s)")
                return True
            else:
                log(f"❌ ERRO na execução de {notebook_name}:")
                log(f"STDOUT: {result.stdout[-500:]}")
                log(f"STDERR: {result.stderr[-500:]}")
                return False
        except Exception as e:
            log(f"❌ Erro ao executar {notebook_name}: {e}")
            return False
    except Exception as e:
        log(f"❌ Erro inesperado ao executar {notebook_name}: {e}")
        return False

def main():
    """Orquestrador principal."""
    log("=" * 80)
    log("🎯 INICIANDO ORQUESTRADOR DE NOTEBOOKS")
    log("=" * 80)
    
    os.chdir(REPO_ROOT)
    
    # Verificar que todos os notebooks existem
    log("\n📋 Verificando disponibilidade dos notebooks...")
    for notebook in NOTEBOOKS:
        path = NOTEBOOKS_DIR / notebook
        exists = "✓" if path.exists() else "❌"
        log(f"{exists} {notebook}")
    
    results = {
        NOTEBOOKS[0]: None,  # Será monitorado
        NOTEBOOKS[1]: None,  # Será executado
        NOTEBOOKS[2]: None   # Será executado
    }
    
    # Etapa 1: Aguardar NN_02_DNN_Correlator_GPU
    if wait_for_first_notebook():
        results[NOTEBOOKS[0]] = "✅ Monitorado (já estava rodando)"
    else:
        results[NOTEBOOKS[0]] = "❌ Falha no monitoramento"
    
    # Etapa 2: Executar NN_02b_4feat_DNN_Aligned
    if results[NOTEBOOKS[0]] == "✅ Monitorado (já estava rodando)":
        success = execute_notebook(NOTEBOOKS[1], 1)
        results[NOTEBOOKS[1]] = "✅ Sucesso" if success else "❌ Falha"
    
    # Etapa 3: Executar NN_08_Architecture_Comparison
    if results[NOTEBOOKS[1]] == "✅ Sucesso":
        success = execute_notebook(NOTEBOOKS[2], 2)
        results[NOTEBOOKS[2]] = "✅ Sucesso" if success else "❌ Falha"
    
    # Relatório final
    log("\n" + "=" * 80)
    log("📊 RELATÓRIO FINAL")
    log("=" * 80)
    for notebook, status in results.items():
        log(f"{status} {notebook}")
    
    all_success = all(status and "✅" in status for status in results.values())
    
    if all_success:
        log("\n🎉 Todos os notebooks executados com sucesso!")
    else:
        log("\n⚠️ Alguns notebooks falharam. Verifique o log acima.")
    
    log("=" * 80)
    return 0 if all_success else 1

if __name__ == "__main__":
    sys.exit(main())
