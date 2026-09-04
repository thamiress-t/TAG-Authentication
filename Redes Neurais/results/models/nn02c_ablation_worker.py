#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""NN_02c isolated ablation worker.

One invocation = one OS process = one TensorFlow/CUDA context.
The parent notebook launches this script once per variant and waits for
the process to terminate before launching the next variant.
"""
import os
import sys
import json
import time
import argparse
from pathlib import Path

# Must be set before importing TensorFlow.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import numpy as np
import h5py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import AUC

parser = argparse.ArgumentParser()
parser.add_argument("--variant", required=True, choices=["A", "A2", "B", "B2", "D", "E", "E2", "F", "F2", "G"])
parser.add_argument("--dataset", required=True)
parser.add_argument("--models-dir", required=True)
parser.add_argument("--output-dir", required=True)
parser.add_argument("--epochs", type=int, default=150)
parser.add_argument("--patience", type=int, default=20)
parser.add_argument("--batch-size", type=int, default=512)
parser.add_argument("--alpha", type=float, default=0.01)
parser.add_argument("--rho-t", type=float, required=True)
parser.add_argument("--l-fixed", type=int, required=True)
args = parser.parse_args()

variant = args.variant
dataset_path = Path(args.dataset)
models_dir = Path(args.models_dir)
output_dir = Path(args.output_dir)
models_dir.mkdir(parents=True, exist_ok=True)
output_dir.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# TensorFlow/CUDA is initialized ONLY in this child process.
# --------------------------------------------------------------------------
gpus = tf.config.list_physical_devices("GPU")
USE_GPU = bool(gpus)
if USE_GPU:
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"WARNING memory_growth: {e}", flush=True)
    try:
        policy = keras.mixed_precision.Policy("mixed_float16")
        keras.mixed_precision.set_global_policy(policy)
        print(f"GPUs={len(gpus)} | mixed precision={policy.name}", flush=True)
    except Exception as e:
        print(f"WARNING mixed precision failed: {e}", flush=True)
else:
    print("GPUs=0 | CPU execution", flush=True)

ENABLE_XLA = False
L_FIXED = args.l_fixed
RHO_T = args.rho_t
BATCH_SIZE = args.batch_size
AUTOTUNE = tf.data.AUTOTUNE

# Original dataset helpers from NN_02c.
def load_h5_generator(h5_path, split, chunk_size=2000):
    """Lazy HDF5 loader — yields (z_eq, h, snr, tau_eq, label, tag_ref) per sample."""
    with h5py.File(str(h5_path), 'r') as f:
        z_data   = f[f'{split}/z_eq']
        h_data   = f[f'{split}/h']
        snr_data = f[f'{split}/snr']
        tau_data = f[f'{split}/tau_eq']
        y_data   = f[f'{split}/y']

        # Accept multiple aliases for the legitimate reference TAG tensor.
        tag_keys = [
            f'{split}/tag_ref',
            f'{split}/tag_legit',
            f'{split}/tag',
            f'{split}/t_ref',
            f'{split}/tag_reference',
        ]
        tag_key = next((k for k in tag_keys if k in f), None)
        tag_data = f[tag_key] if tag_key is not None else None

        for start_idx in range(0, len(z_data), chunk_size):
            end_idx  = min(start_idx + chunk_size, len(z_data))
            z_chunk   = z_data[start_idx:end_idx].astype(np.float32)
            h_chunk   = h_data[start_idx:end_idx].astype(np.float32)
            snr_chunk = snr_data[start_idx:end_idx].astype(np.float32)
            tau_chunk = tau_data[start_idx:end_idx].astype(np.float32)
            y_chunk   = y_data[start_idx:end_idx].astype(np.float32)
            if tag_data is not None:
                tag_chunk = tag_data[start_idx:end_idx].astype(np.float32)
            else:
                tag_chunk = np.zeros_like(z_chunk, dtype=np.float32)

            for i in range(len(z_chunk)):
                yield z_chunk[i], h_chunk[i], snr_chunk[i], tau_chunk[i], y_chunk[i], tag_chunk[i]

def concat_scalar_features(z, *scalars):
    scalar_vec = tf.stack([tf.cast(s, tf.float32) for s in scalars], axis=0)
    return tf.concat([z, scalar_vec], axis=0)

def create_dataset(split, variant='A'):
    """Create tf.data.Dataset with a single DNN-ready feature vector per variant.
   
    Todas as variantes são convertidas para um único vetor de entrada para que
    a comparação seja puramente entre conjuntos de features, não entre
    arquiteturas ou esquemas de fusão diferentes.
   
    Variant A: z_eq[L]
    Variant B: concat(z_eq[L], h[1])
    Variant D: concat(z_eq[L], |tau_eq|[1])
    Variant E: z_eq[L] - tag_ref[L]
    Variant F: h*rho_t*(z_eq[L] - tag_ref[L])  =  y_raw - h*(rho_s*msg + rho_t*tag_ref)
               (residuo pre-equalizacao: ruido w puro, variancia NAO distorcida por 1/h)
    """
    gen_ds = tf.data.Dataset.from_generator(
        lambda: load_h5_generator(dataset_path, split, chunk_size=2000),
        output_signature=(
            tf.TensorSpec(shape=(L_FIXED,), dtype=tf.float32),  # z_eq
            tf.TensorSpec(shape=(),         dtype=tf.float32),  # h
            tf.TensorSpec(shape=(),         dtype=tf.float32),  # snr
            tf.TensorSpec(shape=(),         dtype=tf.float32),  # tau_eq
            tf.TensorSpec(shape=(),         dtype=tf.float32),  # y (label)
            tf.TensorSpec(shape=(L_FIXED,), dtype=tf.float32),  # tag_ref
        )
    )

    if variant == 'A':
        ds = gen_ds.map(lambda z, h, snr, tau, y, tag: (z, y))
    elif variant == 'A2':
        ds = gen_ds.map(lambda z, h, snr, tau, y, tag: (concat_scalar_features(z, tf.reduce_sum(tf.square(z))), y))
    elif variant == 'B':
        ds = gen_ds.map(lambda z, h, snr, tau, y, tag: (concat_scalar_features(z, h), y))
    elif variant == 'B2':
        ds = gen_ds.map(lambda z, h, snr, tau, y, tag: (concat_scalar_features(z, h, tf.reduce_sum(tf.square(z))), y))
    elif variant == 'D':
        ds = gen_ds.map(lambda z, h, snr, tau, y, tag: (concat_scalar_features(z, tf.abs(tau)), y))
    elif variant == 'E':
        if not HAS_TAG_REF:
            raise ValueError(
                "Variante E requer TAG legítima por amostra no HDF5 "
                "(ex.: tag_ref/tag_legit/tag_reference)."
            )
        ds = gen_ds.map(lambda z, h, snr, tau, y, tag: (z - tag, y))
    elif variant == 'E2':
        if not HAS_TAG_REF:
            raise ValueError(
                "Variante E2 requer TAG legítima por amostra no HDF5."
            )
        ds = gen_ds.map(lambda z, h, snr, tau, y, tag: (
            concat_scalar_features(z - tag, tf.reduce_sum(tf.square(z - tag))), y
        ))
    elif variant == 'F':
        if not HAS_TAG_REF:
            raise ValueError(
                "Variante F requer TAG legítima por amostra no HDF5 "
                "(ex.: tag_ref/tag_legit/tag_reference)."
            )
        # x = h*rho_t*(z_eq - tag_ref) = y_raw - h*(rho_s*msg + rho_t*tag_ref)
        # Cancela exatamente o fator 1/(rho_t*h) presente em z_eq, deixando o
        # ruido aditivo w com variancia NAO distorcida pelo desvanecimento h.
        ds = gen_ds.map(lambda z, h, snr, tau, y, tag: ((h * RHO_T) * (z - tag), y))
    elif variant == 'F2':
        if not HAS_TAG_REF:
            raise ValueError(
                "Variante F2 requer TAG legítima por amostra no HDF5."
            )
        ds = gen_ds.map(lambda z, h, snr, tau, y, tag: (
            concat_scalar_features(
                (h * RHO_T) * (z - tag),
                tf.reduce_sum(tf.square((h * RHO_T) * (z - tag)))
            ), y
        ))
    elif variant == 'G':
        if not HAS_TAG_REF:
            raise ValueError(
                "Variante G requer TAG legítima por amostra no HDF5 "
                "(ex.: tag_ref/tag_legit/tag_reference)."
            )
        # G = z_eq ⊙ tag_ref: mantém os L produtos elemento a elemento.
        # NÃO somar antes da DNN.
        ds = gen_ds.map(lambda z, h, snr, tau, y, tag: (z * tag, y))
    else:
        raise ValueError(f'Variante desconhecida: {variant}')

    if split == 'train':
        ds = ds.shuffle(buffer_size=min(10000, n_train),
                        reshuffle_each_iteration=True, seed=42)

    ds = ds.batch(BATCH_SIZE, drop_remainder=(split == 'train'))
    ds = ds.prefetch(AUTOTUNE)
    return ds


# The original create_dataset uses these globals.
with h5py.File(str(dataset_path), "r") as f:
    n_train = len(f["train/z_eq"])
    n_val = len(f["val/z_eq"])
    n_test = len(f["test/z_eq"])
    TAG_REF_KEY = next(
        (k for k in [
            "train/tag_ref", "train/tag_legit", "train/tag",
            "train/t_ref", "train/tag_reference"
        ] if k in f),
        None
    )
HAS_TAG_REF = TAG_REF_KEY is not None

VARIANT_INPUT_DIMS = {
    "A": L_FIXED, "A2": L_FIXED + 1,
    "B": L_FIXED + 1, "B2": L_FIXED + 2,
    "D": L_FIXED + 1,
    "E": L_FIXED, "E2": L_FIXED + 1,
    "F": L_FIXED, "F2": L_FIXED + 1,
    "G": L_FIXED,
}

VARIANT_INFO = {
    "A": "z_eq[L_FIXED] (DNN pura)",
    "A2": "z_eq[L_FIXED] + Σz²",
    "B": "z_eq + h (DNN + canal)",
    "B2": "z_eq + h + Σz²",
    "D": "z_eq + |tau| (DNN + correlador)",
    "E": "z_eq - tag_ref[L_FIXED]",
    "E2": "(z_eq - tag_ref) + Σ(z_eq - tag_ref)²",
    "F": "h*rho_t*(z_eq - tag_ref) (resíduo pré-equalização)",
    "F2": "F + ΣF² (energia do resíduo escalado)",
    "G": "z_eq ⊙ tag_ref[L_FIXED] (produto elemento a elemento)",
}

def build_dnn_classifier(input_dim, variant):
    inputs = layers.Input(shape=(input_dim,), name=f"features_variant_{variant}")
    x = layers.Dense(256, kernel_initializer="he_uniform")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, kernel_initializer="he_uniform")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, kernel_initializer="he_uniform")(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation="sigmoid", dtype="float32", name="p_auth")(x)
    model = keras.Model(
        inputs=inputs, outputs=outputs, name=f"DNN_variant_{variant}"
    )
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss=BinaryCrossentropy(from_logits=False),
        metrics=["accuracy", AUC(name="auc")],
        jit_compile=ENABLE_XLA,
    )
    return model

ALPHA_D3F = args.alpha
Q_INV_D3F = 2.3263478740408408
SNR_BINS = [0, 5, 10, 15, 20, 25, 30]

with h5py.File(str(dataset_path), "r") as f:
    snr_val_arr = f["val/snr"][:].astype(np.float32)
    snr_test_arr = f["test/snr"][:].astype(np.float32)
    lbl_val_arr = f["val/y"][:].astype(np.float32)
    lbl_test_arr = f["test/y"][:].astype(np.float32)

def predict_batches(model, ds):
    scores = []
    for batch_x, _ in ds:
        p = model(batch_x, training=False)
        scores.append(p.numpy().flatten())
    return np.concatenate(scores)

print("\n" + "=" * 70, flush=True)
print(f"Variante {variant}: {VARIANT_INFO[variant]}", flush=True)
print(f"PID={os.getpid()} | batch={BATCH_SIZE} | GPU={USE_GPU}", flush=True)
print("=" * 70, flush=True)

keras.backend.clear_session()

train_ds = create_dataset("train", variant=variant)
val_ds = create_dataset("val", variant=variant)
test_ds = create_dataset("test", variant=variant)
model = build_dnn_classifier(VARIANT_INPUT_DIMS[variant], variant)

checkpoint_path = models_dir / f"dnn_variant_{variant}_best.keras"
cbs = [
    callbacks.EarlyStopping(
        monitor="val_auc", patience=args.patience,
        mode="max", restore_best_weights=True
    ),
    callbacks.ModelCheckpoint(
        str(checkpoint_path), monitor="val_auc",
        mode="max", save_best_only=True
    ),
    callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=8, min_lr=1e-6
    ),
]

train_start = time.perf_counter()
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=args.epochs,
    callbacks=cbs,
    verbose=1,
)
train_time = time.perf_counter() - train_start
best_auc_val = float(max(history.history["val_auc"]))
n_epochs = int(len(history.history["loss"]))

print(
    f"Treino concluido: {train_time:.1f}s | epocas={n_epochs} | "
    f"melhor val_AUC={best_auc_val:.6f}",
    flush=True,
)

print("Computando scores (validacao + teste)...", flush=True)
sc_val = predict_batches(model, val_ds)[:len(snr_val_arr)]
sc_test = predict_batches(model, test_ds)[:len(snr_test_arr)]

pd_d3f = {}
for snr in SNR_BINS:
    m0 = (
        (snr_val_arr >= snr - 2.5) &
        (snr_val_arr < snr + 2.5) &
        (lbl_val_arr == 0)
    )
    m1 = (
        (snr_test_arr >= snr - 2.5) &
        (snr_test_arr < snr + 2.5) &
        (lbl_test_arr == 1)
    )
    if m0.sum() < 50 or m1.sum() < 10:
        pd_d3f[str(snr)] = 0.0
        continue
    mu = sc_val[m0].mean()
    sig = sc_val[m0].std()
    if sig < 1e-10:
        pd_d3f[str(snr)] = 0.0
        continue
    thr = float(mu + Q_INV_D3F * sig)
    pd_d3f[str(snr)] = float((sc_test[m1] >= thr).mean())

print(
    "PD (%) D3F: " +
    str([round(pd_d3f[str(s)] * 100, 1) for s in SNR_BINS]),
    flush=True,
)

npz_path = output_dir / f"nn02c_worker_{variant}_scores.npz"
json_path = output_dir / f"nn02c_worker_{variant}_meta.json"

np.savez_compressed(
    str(npz_path),
    sc_val=sc_val.astype(np.float32),
    sc_test=sc_test.astype(np.float32),
)

meta = {
    "variant": variant,
    "label": VARIANT_INFO[variant],
    "best_val_auc": best_auc_val,
    "n_epochs": n_epochs,
    "train_time": train_time,
    "pd_d3f": pd_d3f,
    "batch_size": BATCH_SIZE,
    "pid": os.getpid(),
}
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)

print(f"Resultados salvos em: {npz_path}", flush=True)
print(f"Metadata salva em: {json_path}", flush=True)

# Complementary cleanup. The decisive reset is PROCESS EXIT immediately after
# this script returns to the OS.
del train_ds, val_ds, test_ds, model, history
keras.backend.clear_session()
print(
    "Worker finalizado. Encerrando processo para destruir completamente "
    "o contexto CUDA/TF.",
    flush=True,
)
