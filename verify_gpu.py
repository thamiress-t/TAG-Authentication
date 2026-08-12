#!/usr/bin/env python3
"""
Verificar se GPU foi detectada corretamente em TensorFlow e PyTorch.
"""

print("=" * 80)
print("GPU VERIFICATION CHECK")
print("=" * 80)

# TensorFlow
print("\n[1] TensorFlow GPU Check:")
try:
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"  ✅ TensorFlow detectou {len(gpus)} GPU(s):")
        for gpu in gpus:
            print(f"     - {gpu}")
    else:
        print("  ⚠️  TensorFlow não detectou GPUs (pode usar CUDA CPU)")
except Exception as e:
    print(f"  ❌ Erro ao verificar TensorFlow: {e}")

# PyTorch
print("\n[2] PyTorch GPU Check:")
try:
    import torch
    if torch.cuda.is_available():
        print(f"  ✅ PyTorch detectou CUDA disponível")
        print(f"     GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"     CUDA Version: {torch.version.cuda}")
        print(f"     cuDNN Version: {torch.backends.cudnn.version()}")
    else:
        print("  ⚠️  PyTorch não detectou CUDA (usando CPU)")
except Exception as e:
    print(f"  ❌ Erro ao verificar PyTorch: {e}")

# Versões
print("\n[3] Library Versions:")
try:
    import numpy
    print(f"  ✅ NumPy: {numpy.__version__}")
except:
    print(f"  ❌ NumPy não instalado")

try:
    import scipy
    print(f"  ✅ SciPy: {scipy.__version__}")
except:
    print(f"  ❌ SciPy não instalado")

try:
    import h5py
    print(f"  ✅ h5py: {h5py.__version__}")
except:
    print(f"  ❌ h5py não instalado")

try:
    import pandas
    print(f"  ✅ Pandas: {pandas.__version__}")
except:
    print(f"  ❌ Pandas não instalado")

try:
    import jupyter
    print(f"  ✅ Jupyter instalado")
except:
    print(f"  ❌ Jupyter não instalado")

print("\n" + "=" * 80)
print("Verificação concluída!")
print("=" * 80)
