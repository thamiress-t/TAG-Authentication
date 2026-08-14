#!/usr/bin/env python3
"""
Diagnóstico e setup CUDA/GPU em WSL para TensorFlow
Author: GitHub Copilot
Date: 2026-07-12

PROBLEMA: TensorFlow não detecta GPU em WSL
CAUSA: Bibliotecas CUDA/cuDNN não encontradas

Este script:
  1. Diagnostica o ambiente WSL/CUDA
  2. Testa se bibliotecas CUDA estão instaladas
  3. Fornece instruções de setup
"""

import os
import subprocess
import sys

def check_wsl():
    """Verifica se está rodando em WSL"""
    try:
        with open('/proc/version', 'r') as f:
            content = f.read().lower()
            if 'microsoft' in content or 'wsl' in content:
                return True
    except:
        pass
    return False

def check_nvidia_gpu():
    """Verifica se GPU NVIDIA é detectada"""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✅ nvidia-smi found - GPU driver installed")
            print(result.stdout)
            return True
    except Exception as e:
        print(f"❌ nvidia-smi not found: {e}")
    return False

def check_cuda_env():
    """Verifica variáveis de ambiente CUDA"""
    print("\n[CUDA Environment Variables]")
    vars_to_check = {
        'CUDA_HOME': 'CUDA installation directory',
        'CUDA_PATH': 'CUDA installation path (Windows-style)',
        'LD_LIBRARY_PATH': 'Library search path',
        'NVIDIA_DOCKER_VERSION': 'Docker NVIDIA runtime',
    }
    
    found = False
    for var, desc in vars_to_check.items():
        value = os.environ.get(var, '[NOT SET]')
        if '[NOT SET]' in value:
            print(f"⚠️  {var:25} : {desc:40} {value}")
        else:
            print(f"✅ {var:25} : {desc:40} {value}")
            found = True
    
    return found

def check_cuda_libs():
    """Verifica se bibliotecas CUDA essenciais estão instaladas"""
    print("\n[CUDA Libraries Check]")
    
    cuda_libs = [
        'libcudart.so',
        'libcublas.so',
        'libcufft.so',
        'libcudnn.so',
    ]
    
    # Locais comuns onde CUDA é instalado
    possible_cuda_dirs = [
        '/usr/local/cuda',
        '/opt/cuda',
        os.path.expanduser('~/cuda'),
        '/usr/local/cuda-12.2',
        '/usr/local/cuda-12.1',
        '/usr/local/cuda-12.0',
    ]
    
    found_dir = None
    for cuda_dir in possible_cuda_dirs:
        lib_dir = os.path.join(cuda_dir, 'lib64')
        if os.path.exists(lib_dir):
            print(f"✅ Found CUDA directory: {cuda_dir}")
            found_dir = cuda_dir
            break
    
    if not found_dir:
        print(f"❌ CUDA not found in standard locations")
        print(f"   Checked: {', '.join(possible_cuda_dirs)}")
        return False
    
    # Verifica bibliotecas
    lib_dir = os.path.join(found_dir, 'lib64')
    print(f"\n   Checking libraries in {lib_dir}:")
    for lib in cuda_libs:
        lib_path = os.path.join(lib_dir, lib)
        if os.path.exists(lib_path) or any(os.path.exists(os.path.join(lib_dir, f'{lib}.12')) for lib_dir in [lib_dir]):
            print(f"   ✅ {lib}")
        else:
            print(f"   ⚠️  {lib} (not found)")
    
    return True

def main():
    print("="*80)
    print("WSL CUDA/GPU DIAGNOSTIC FOR TENSORFLOW")
    print("="*80)
    
    # Check if WSL
    if check_wsl():
        print("\n✅ Running on WSL")
    else:
        print("\n❌ Not running on WSL (native Linux or Windows)")
    
    # Check NVIDIA GPU
    print("\n[GPU Detection]")
    has_gpu = check_nvidia_gpu()
    
    if not has_gpu:
        print("\n❌ GPU NOT DETECTED - nvidia-smi failed")
    
    # Check CUDA environment
    has_env = check_cuda_env()
    
    # Check CUDA libraries
    has_libs = check_cuda_libs()
    
    # Summary and recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    if not has_gpu:
        print("""
⚠️  GPU not detected. Fix:
   1. Ensure NVIDIA GPU drivers are installed in Windows
   2. Install CUDA Toolkit in WSL:
      $ wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.0-1_all.deb
      $ sudo dpkg -i cuda-keyring_1.0-1_all.deb
      $ sudo apt-get update
      $ sudo apt-get install -y cuda-toolkit-12-2
   3. Add to ~/.bashrc:
      export PATH=/usr/local/cuda-12.2/bin:$PATH
      export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH
      export CUDA_HOME=/usr/local/cuda-12.2
   4. Reload: $ source ~/.bashrc
        """)
    
    if not has_env:
        print("""
⚠️  CUDA environment variables not set. Fix:
   1. Add to ~/.bashrc:
      export CUDA_HOME=/usr/local/cuda-12.2
      export PATH=$CUDA_HOME/bin:$PATH
      export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
   2. Reload: $ source ~/.bashrc
   3. Verify: $ echo $CUDA_HOME && echo $LD_LIBRARY_PATH
        """)
    
    if has_gpu and has_env and has_libs:
        print("""
✅ CUDA setup looks good!
   Try restarting Python/Jupyter and re-run the training notebook.
   If TensorFlow still doesn't detect GPU:
   - Restart WSL: $ wsl --shutdown (in Windows PowerShell)
   - Then open WSL again
        """)
    
    print("\n" + "="*80)

if __name__ == '__main__':
    main()
