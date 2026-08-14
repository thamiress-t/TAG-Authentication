#!/usr/bin/env python3
"""
Test script to validate Cell 2 + 3 lazy loading fix without OOM
Runs sequentially: import → GPU setup → load metadata → create generators → test pipeline
"""

import os
import sys
import time
from pathlib import Path

# Ensure venv_tf_gpu is active
venv_path = Path.home() / "venv_tf_gpu" / "bin" / "activate_this.py"
if not venv_path.exists():
    print("❌ venv_tf_gpu not found. Please activate it manually.")
    sys.exit(1)

# ==============================================================================
# PART 1: IMPORTS & GPU SETUP
# ==============================================================================
print("="*80)
print("PART 1: IMPORTS & GPU SETUP")
print("="*80)

import numpy as np
import h5py
import psutil
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

print(f"TensorFlow: {tf.__version__}")
print(f"NumPy: {np.__version__}")
print(f"psutil: {psutil.__version__}")

# GPU setup
gpus = tf.config.list_physical_devices('GPU')
print(f"GPUs found: {len(gpus)}")
for i, gpu in enumerate(gpus):
    print(f"  GPU {i}: {gpu}")

USE_GPU = False
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        USE_GPU = True
        print(f"✅ Memory growth enabled on {len(gpus)} GPU(s)")
    except Exception as e:
        print(f"⚠️ Error enabling GPU memory growth: {e}")

# Mixed precision
if USE_GPU:
    policy = keras.mixed_precision.Policy('mixed_float16')
    keras.mixed_precision.set_global_policy(policy)
    print(f"Mixed precision: {policy.name}")

# Disable XLA
tf.config.optimizer.set_jit(False)
print("XLA JIT: disabled")

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ==============================================================================
# PART 2: PATHS & MEMORY MONITORING
# ==============================================================================
print("\n" + "="*80)
print("PART 2: PATHS & MEMORY MONITORING")
print("="*80)

def print_memory_usage(label):
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / 1024 / 1024
    available_mb = psutil.virtual_memory().available / 1024 / 1024
    print(f"  [{label}] RAM usage: {mem_mb:.1f} MB | Available: {available_mb:.1f} MB")

print_memory_usage("Start")

notebook_dir = Path.cwd()
project_root = notebook_dir.parent if notebook_dir.name == "notebooks" else notebook_dir
results_dir = project_root / "Redes Neurais" / "results"
data_dir = results_dir / "data"
models_dir = results_dir / "models"

print(f"Project root: {project_root}")
print(f"Data dir: {data_dir}")

# ==============================================================================
# PART 3: LOAD DATASET METADATA (LAZY)
# ==============================================================================
print("\n" + "="*80)
print("PART 3: LOAD DATASET METADATA (LAZY)")
print("="*80)

dataset_path = data_dir / "dataset_cnn_yeq_0_30dB.h5"

if not dataset_path.exists():
    print(f"❌ Dataset not found: {dataset_path}")
    sys.exit(1)

print(f"Dataset: {dataset_path.name}")
print(f"Size: {dataset_path.stat().st_size / 1e9:.2f} GB")

with h5py.File(str(dataset_path), 'r') as f:
    if 'train/z_eq' not in f:
        print("❌ z_eq not found in dataset")
        sys.exit(1)
    
    L_FIXED = int(f.attrs['L_FIXED'])
    RHO_S = float(f.attrs.get('RHO_S', 0.992472))
    
    shape_train = f['train/z_eq'].shape
    shape_val = f['val/z_eq'].shape
    shape_test = f['test/z_eq'].shape
    
    print(f"✅ Metadata loaded (no data in RAM)")
    print(f"   L_FIXED: {L_FIXED}")
    print(f"   Train shape: {shape_train}")
    print(f"   Val shape: {shape_val}")
    print(f"   Test shape: {shape_test}")

print_memory_usage("After metadata load")

# ==============================================================================
# PART 4: CREATE LAZY-LOAD GENERATORS
# ==============================================================================
print("\n" + "="*80)
print("PART 4: CREATE LAZY-LOAD GENERATORS")
print("="*80)

def load_h5_generator(h5_path, split, chunk_size=2000):
    """Generator that reads HDF5 dataset in chunks."""
    with h5py.File(str(h5_path), 'r') as f:
        z_data = f[f'{split}/z_eq']
        y_data = f[f'{split}/y']
        
        n_samples = len(z_data)
        for start_idx in range(0, n_samples, chunk_size):
            end_idx = min(start_idx + chunk_size, n_samples)
            z_chunk = z_data[start_idx:end_idx].astype(np.float32)
            y_chunk = y_data[start_idx:end_idx].astype(np.float32)
            
            for i in range(len(z_chunk)):
                yield z_chunk[i], y_chunk[i]

def create_tf_dataset_from_h5(h5_path, split):
    gen = load_h5_generator(h5_path, split, chunk_size=2000)
    
    with h5py.File(str(h5_path), 'r') as f:
        n_samples = len(f[f'{split}/z_eq'])
    
    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(L_FIXED,), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.float32)
        )
    )
    
    # Manually set cardinality (from_generator can't infer it)
    ds = ds.with_cardinality(n_samples)
    
    return ds, n_samples

print("Creating generators...")
start = time.time()
train_gen_ds, n_train = create_tf_dataset_from_h5(dataset_path, 'train')
val_gen_ds, n_val = create_tf_dataset_from_h5(dataset_path, 'val')
test_gen_ds, n_test = create_tf_dataset_from_h5(dataset_path, 'test')
elapsed = time.time() - start

print(f"✅ Generators created in {elapsed:.2f}s")
print(f"   Train samples: {n_train}")
print(f"   Val samples: {n_val}")
print(f"   Test samples: {n_test}")

print_memory_usage("After generator creation")

# ==============================================================================
# PART 5: PREPARE TF.DATA PIPELINE
# ==============================================================================
print("\n" + "="*80)
print("PART 5: PREPARE TF.DATA PIPELINE")
print("="*80)

BATCH_SIZE = 128 if USE_GPU else 64
AUTOTUNE = tf.data.AUTOTUNE

def prepare_dataset(ds, n_samples, shuffle=False, batch_size=BATCH_SIZE):
    if shuffle:
        buffer_size = min(10000, n_samples)
        ds = ds.shuffle(buffer_size=buffer_size, reshuffle_each_iteration=True, seed=42)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds

print(f"Batch size: {BATCH_SIZE}")
train_ds = prepare_dataset(train_gen_ds, n_train, shuffle=True)
val_ds = prepare_dataset(val_gen_ds, n_val, shuffle=False)
test_ds = prepare_dataset(test_gen_ds, n_test, shuffle=False)

print(f"✅ Pipelines ready")
print_memory_usage("After pipeline setup")

# ==============================================================================
# PART 6: TEST PIPELINE (FIRST BATCH)
# ==============================================================================
print("\n" + "="*80)
print("PART 6: TEST PIPELINE (FIRST BATCH)")
print("="*80)

print("Loading first training batch...")
start = time.time()
for batch_X, batch_y in train_ds.take(1):
    elapsed = time.time() - start
    print(f"✅ Batch loaded in {elapsed:.2f}s")
    print(f"   Batch shape: X={batch_X.shape}, y={batch_y.shape}")
    print(f"   X dtype: {batch_X.dtype}")
    print(f"   X stats: min={batch_X.numpy().min():.4f}, max={batch_X.numpy().max():.4f}, mean={batch_X.numpy().mean():.4f}")
    print(f"   y unique values: {np.unique(batch_y.numpy())}")

print_memory_usage("After first batch load")

# ==============================================================================
# FINAL SUMMARY
# ==============================================================================
print("\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)
print(f"✅ ALL TESTS PASSED - No OOM!")
print(f"   GPU: {USE_GPU}")
print(f"   Batch size: {BATCH_SIZE}")
print(f"   Memory usage: {psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024:.1f} MB")
print(f"   Available: {psutil.virtual_memory().available / 1024 / 1024:.1f} MB")
print(f"\n📝 The notebook Cell 2 + Cell 3 should now work without OOM!")
print("="*80)
