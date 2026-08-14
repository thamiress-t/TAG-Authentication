# 🚀 SETUP GPU - JUPYTER NOTEBOOK COM TENSORFLOW CUDA

## ✅ Status Atual

- ✅ **GPU detectada**: NVIDIA RTX 4050 (6GB VRAM)
- ✅ **TensorFlow instalado**: 2.21.0
- ✅ **CUDA libraries**: Instaladas em `~/venv_tf_gpu/`
- ⚠️ **Problema**: Notebook não está usando o venv_tf_gpu

## 🔧 SOLUÇÃO: Executar Jupyter com o Kernel Correto

### Opção 1: Ativar o venv ANTES de abrir Jupyter (Recomendado)

```bash
# Abra o terminal WSL e execute:
cd /mnt/c/Users/thami/OneDrive/Documents/TAG-Authentication

# Ative o venv com CUDA/GPU
source /home/thami/venv_tf_gpu/bin/activate

# Inicie o Jupyter (escolha uma opção)

# Opção A: Jupyter Lab (moderno)
jupyter lab

# Opção B: Jupyter Notebook (clássico)
jupyter notebook

# Você verá algo como:
# (venv_tf_gpu) thami@...
# [I 2026-07-12 ...] Jupyter Server ... is running at:
#     http://localhost:8888/tree
```

### Opção 2: Registrar o kernel na lista de kernels do Jupyter

Se você já tem Jupyter instalado em outro lugar:

```bash
# Com o venv ativado, registre o kernel
source /home/thami/venv_tf_gpu/bin/activate
python -m ipykernel install --user --name tf-gpu --display-name "Python 3.10 (TensorFlow GPU)"

# Depois, qualquer Jupyter que você abrir terá esse kernel disponível
# Selecione "Python 3.10 (TensorFlow GPU)" no notebook
```

## ✅ Verificar que está Funcionando

No notebook, execute a Célula 1 e você deverá ver:

```
TensorFlow : 2.21.0
GPUs found : 1
  GPU 0: PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')
✅ Memory growth enabled on 1 GPU(s).
Mixed precision policy : mixed_float16
XLA JIT : disabled (avoids 50+ min autotuner hang)
```

Se vir `GPUs found: 0` ou `WARNING: No GPU detected`, a GPU não está sendo usada. Verifique:
1. Está usando o venv_tf_gpu? (veja acima)
2. Executou `source /home/thami/venv_tf_gpu/bin/activate`?

## 🔍 Diagnosticar Problemas

Se a GPU ainda não for detectada:

```bash
# Ative o venv e rode o diagnóstico
source /home/thami/venv_tf_gpu/bin/activate
python /mnt/c/Users/thami/OneDrive/Documents/TAG-Authentication/fix_gpu_wsl.py

# Ou teste diretamente
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

## 📊 Esperado durante o Treinamento

Com GPU funcionando:
- **Célula 1**: GPU detectada e memória configurada
- **Célula 6 (Treinamento)**:
  - Primeira época deve começar rapidamente (< 30 segundos)
  - GPU Utilization: 60-90%
  - Tempo por época: 2-5 minutos (dependendo do batch size)
  - **NÃO deve travar** (XLA está desabilitado)

Sem GPU (CPU only):
- Treinamento será 5-10x mais lento
- Tempo por época: 15-30 minutos

## 🛠️ Ambiente Virtual (`venv_tf_gpu`)

Pacotes instalados:
- TensorFlow 2.21.0
- CUDA libraries (nvidia-*)
- CUDNN
- NumPy, Pandas, Scikit-learn
- Matplotlib
- Jupyter

Para instalar mais pacotes neste venv:

```bash
source /home/thami/venv_tf_gpu/bin/activate
pip install <package_name>
```

## 📝 Checklist

- [ ] Abrir WSL Terminal
- [ ] `cd /mnt/c/Users/thami/OneDrive/Documents/TAG-Authentication`
- [ ] `source /home/thami/venv_tf_gpu/bin/activate`
- [ ] `jupyter lab` ou `jupyter notebook`
- [ ] Abrir `Redes Neurais/notebooks/NN_02_DNN_Correlator_GPU.ipynb`
- [ ] Executar Célula 1 e verificar se GPU foi detectada
- [ ] Executar Célula 6 (treinamento)

## 📚 Referências

- [WSL + CUDA + TensorFlow Setup Guide](https://www.tensorflow.org/install/gpu_wsl)
- [NVIDIA Docker Image for TensorFlow](https://hub.docker.com/r/tensorflow/tensorflow)
- [Jupyter Kernels Documentation](https://jupyter.readthedocs.io/en/latest/projects/jupyter-kernels.html)

---

**Última atualização**: 2026-07-12
**Status**: ✅ GPU configurada e testada com sucesso
