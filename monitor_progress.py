#!/usr/bin/env python3
"""
Monitor de Progresso - Acompanha a execução dos notebooks em tempo real
"""
import subprocess
import json
from pathlib import Path
from datetime import datetime
import time
import sys

REPO_ROOT = Path("/mnt/c/Users/thami/OneDrive/Documents/TAG-Authentication")
LOG_FILE = REPO_ROOT / "execution_log_v2.txt"

def get_process_info():
    """Retorna informações dos processos ativos."""
    try:
        result = subprocess.run(
            ["ps", "aux"],
            capture_output=True,
            text=True
        )
        
        processes = []
        for line in result.stdout.split('\n'):
            if 'notebook_orchestrator' in line or 'nbconvert' in line or 'ipykernel' in line:
                if 'grep' not in line:
                    processes.append(line)
        
        return processes
    except:
        return []

def get_file_sizes():
    """Retorna tamanhos dos notebooks para detectar mudanças."""
    sizes = {}
    notebooks_dir = REPO_ROOT / "Redes Neurais" / "notebooks"
    
    for nb in ["NN_02_DNN_Correlator_GPU.ipynb", "NN_02b_4feat_DNN_Aligned.ipynb", "NN_08_Architecture_Comparison.ipynb"]:
        path = notebooks_dir / nb
        if path.exists():
            sizes[nb] = path.stat().st_size
    
    return sizes

def monitor():
    """Monitor em tempo real."""
    print("\n" + "="*80)
    print("📊 MONITOR DE PROGRESSO - Notebooks TAG-Authentication")
    print("="*80)
    
    prev_sizes = get_file_sizes()
    start_time = time.time()
    
    while True:
        elapsed = int(time.time() - start_time)
        
        print(f"\n⏱️  {datetime.now().strftime('%H:%M:%S')} (elapsed: {elapsed}s)")
        print("-" * 80)
        
        # Processos ativos
        processes = get_process_info()
        if processes:
            print("🔄 Processos ativos:")
            for i, proc in enumerate(processes[:3], 1):
                parts = proc.split()
                if len(parts) > 10:
                    print(f"   {i}. PID={parts[1]} | {' '.join(parts[10:12])}")
        else:
            print("⏳ Nenhum processo ativo (aguardando conclusão ou erro)")
        
        # Tamanho dos arquivos
        curr_sizes = get_file_sizes()
        print("\n📁 Tamanho dos notebooks:")
        for nb in ["NN_02_DNN_Correlator_GPU.ipynb", "NN_02b_4feat_DNN_Aligned.ipynb", "NN_08_Architecture_Comparison.ipynb"]:
            if nb in curr_sizes:
                size_mb = curr_sizes[nb] / (1024 * 1024)
                status = "↑" if nb in prev_sizes and curr_sizes[nb] > prev_sizes[nb] else " "
                print(f"   {status} {nb}: {size_mb:.2f} MB")
        
        # Última linha do log
        try:
            with open(LOG_FILE) as f:
                lines = f.readlines()
                if lines:
                    print(f"\n📝 Última atualização do log:")
                    print(f"   {lines[-1].strip()}")
        except:
            pass
        
        prev_sizes = curr_sizes
        
        # Verificar se completou
        if not processes:
            print("\n✅ Nenhum processo ativo - execução pode ter completado")
            break
        
        time.sleep(15)

if __name__ == "__main__":
    try:
        monitor()
    except KeyboardInterrupt:
        print("\n\n❌ Monitor interrompido pelo usuário")
        sys.exit(0)
