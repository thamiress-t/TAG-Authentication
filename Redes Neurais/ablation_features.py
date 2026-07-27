"""
Feature Ablation Study — 4-feat DNN
Trains 4 new variants (V2-V5) and evaluates with D3F (alpha=1e-7).
Baseline V1 already exists (model_dnn_4feat_aligned.keras).

Variants:
  V1 baseline  : [|τ|, h_est,   SNR_true, E]   — already trained
  V2 perfect-h : [|τ|, h_true,  SNR_true, E]   — replace proxy with true h
  V3 no-SNR    : [|τ|, h_est,   E]             — drop SNR column
  V4 no-E      : [|τ|, h_est,   SNR_true]      — drop energy column
  V5 tau-only  : [|τ|]                          — single-feature baseline

Run:
  cd "Redes Neurais"
  python3 ablation_features.py
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # force CPU — GPU ptxas/cuBLAS broken in this env
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import h5py
import json
from pathlib import Path
from scipy.stats import norm

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy

tf.random.set_seed(42)
np.random.seed(42)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE       = Path(__file__).parent
DATA_DIR   = BASE / "results" / "data"
MODELS_DIR = BASE / "results" / "models"

DS_PATH    = DATA_DIR / "dataset_cnn_yeq_0_30dB.h5"
OUT_JSON   = DATA_DIR / "nn15_feature_ablation.json"

ALPHA   = 1e-7
Q_INV   = norm.ppf(1 - ALPHA)          # ≈ 5.199
SNR_BINS = [0, 5, 10, 15, 20, 25, 30]

# ── Load dataset once ─────────────────────────────────────────────────────────
print("Loading dataset …")
with h5py.File(str(DS_PATH), "r") as f:
    # Raw signals (for proxy features)
    Y_train = f["train/y_eq"][:].astype(np.float32)
    Y_val   = f["val/y_eq"][:].astype(np.float32)
    Y_test  = f["test/y_eq"][:].astype(np.float32)

    # Pre-computed scalars
    TAU_train = np.abs(f["train/tau_eq"][:]).astype(np.float32)
    TAU_val   = np.abs(f["val/tau_eq"][:]).astype(np.float32)
    TAU_test  = np.abs(f["test/tau_eq"][:]).astype(np.float32)

    SNR_train = f["train/snr"][:].astype(np.float32)
    SNR_val   = f["val/snr"][:].astype(np.float32)
    SNR_test  = f["test/snr"][:].astype(np.float32)

    H_train = f["train/h"][:].astype(np.float32)   # true Rayleigh gain
    H_val   = f["val/h"][:].astype(np.float32)
    H_test  = f["test/h"][:].astype(np.float32)

    LBL_train = f["train/y"][:].astype(np.float32)
    LBL_val   = f["val/y"][:].astype(np.float32)
    LBL_test  = f["test/y"][:].astype(np.float32)

# Derived proxy features (same as NN_02b)
H_EST_train = np.abs(Y_train).mean(axis=1)
H_EST_val   = np.abs(Y_val).mean(axis=1)
H_EST_test  = np.abs(Y_test).mean(axis=1)

E_train = (Y_train ** 2).mean(axis=1)
E_val   = (Y_val   ** 2).mean(axis=1)
E_test  = (Y_test  ** 2).mean(axis=1)

print(f"  Train: {len(LBL_train):,}   Val: {len(LBL_val):,}   Test: {len(LBL_test):,}")


# ── Feature matrices per variant ──────────────────────────────────────────────
def build_features(split, variant):
    tau   = {"train": TAU_train,   "val": TAU_val,   "test": TAU_test}[split]
    h_est = {"train": H_EST_train, "val": H_EST_val, "test": H_EST_test}[split]
    h_tr  = {"train": H_train,     "val": H_val,     "test": H_test}[split]
    snr   = {"train": SNR_train,   "val": SNR_val,   "test": SNR_test}[split]
    energy= {"train": E_train,     "val": E_val,     "test": E_test}[split]

    if variant == "V1_baseline":
        return np.stack([tau, h_est, snr, energy], axis=1)
    elif variant == "V2_perfect_h":
        return np.stack([tau, h_tr,  snr, energy], axis=1)
    elif variant == "V3_no_snr":
        return np.stack([tau, h_est, energy],       axis=1)
    elif variant == "V4_no_energy":
        return np.stack([tau, h_est, snr],           axis=1)
    elif variant == "V5_tau_only":
        return tau[:, None]
    # ── 2-feature combinations ──────────────────────────────────────────────
    elif variant == "V6_tau_h":
        return np.stack([tau, h_est], axis=1)
    elif variant == "V7_tau_snr":
        return np.stack([tau, snr], axis=1)
    elif variant == "V8_tau_energy":
        return np.stack([tau, energy], axis=1)
    else:
        raise ValueError(f"Unknown variant: {variant}")


# ── DNN architecture ──────────────────────────────────────────────────────────
def build_dnn(input_dim):
    inp = layers.Input(shape=(input_dim,))
    x = layers.Dense(256, kernel_initializer="he_uniform")(inp)
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
    out = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inp, out)
    model.compile(
        optimizer=Adam(1e-3),
        loss=BinaryCrossentropy(),
        metrics=[keras.metrics.AUC(name="auc")],
    )
    return model


# ── D3F evaluation ────────────────────────────────────────────────────────────
def d3f_eval(sc_val, sc_test, snr_v, snr_t, lbl_v, lbl_t):
    pd = {}
    for snr in SNR_BINS:
        m0 = (snr_v >= snr - 2.5) & (snr_v < snr + 2.5) & (lbl_v == 0)
        m1 = (snr_t >= snr - 2.5) & (snr_t < snr + 2.5) & (lbl_t == 1)
        if m0.sum() < 50 or m1.sum() < 10:
            pd[snr] = 0.0
            continue
        mu, sigma = sc_val[m0].mean(), sc_val[m0].std()
        thr = float(mu + Q_INV * sigma)
        pd[snr] = float((sc_test[m1] >= thr).mean())
    return pd


# ── Training loop ─────────────────────────────────────────────────────────────
VARIANTS = ["V1_baseline", "V2_perfect_h", "V3_no_snr", "V4_no_energy", "V5_tau_only"]

results = {}

for vname in VARIANTS:
    print(f"\n{'='*60}")
    print(f"  Variant: {vname}")
    print(f"{'='*60}")

    X_tr = build_features("train", vname)
    X_v  = build_features("val",   vname)
    X_te = build_features("test",  vname)

    # z-score normalisation (fit on train)
    mu  = X_tr.mean(axis=0)
    std = X_tr.std(axis=0)
    std[std < 1e-8] = 1.0          # guard against zero-variance columns
    X_tr_n = ((X_tr - mu) / std).astype(np.float32)
    X_v_n  = ((X_v  - mu) / std).astype(np.float32)
    X_te_n = ((X_te - mu) / std).astype(np.float32)

    print(f"  Input dim : {X_tr.shape[1]}   Train rows: {len(LBL_train):,}")

    model = build_dnn(X_tr.shape[1])

    ckpt_path = str(MODELS_DIR / f"ablation_{vname}.keras")
    cbs = [
        callbacks.EarlyStopping(
            monitor="val_auc", mode="max", patience=20,
            restore_best_weights=True, verbose=0,
        ),
        callbacks.ModelCheckpoint(
            ckpt_path, monitor="val_auc", mode="max",
            save_best_only=True, verbose=0,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=8,
            min_lr=1e-6, verbose=0,
        ),
    ]

    hist = model.fit(
        X_tr_n, LBL_train,
        validation_data=(X_v_n, LBL_val),
        epochs=150, batch_size=512,
        callbacks=cbs, verbose=1,
    )

    best_auc = max(hist.history["val_auc"])
    n_epochs = len(hist.history["loss"])
    print(f"  Best val AUC: {best_auc:.5f}   Epochs: {n_epochs}")

    # D3F evaluation
    sc_val  = model.predict(X_v_n,  batch_size=1024, verbose=0).ravel()
    sc_test = model.predict(X_te_n, batch_size=1024, verbose=0).ravel()

    pd = d3f_eval(sc_val, sc_test, SNR_val, SNR_test, LBL_val, LBL_test)

    pd_list = [round(pd[s] * 100, 1) for s in SNR_BINS]
    print(f"  SNR bins : {SNR_BINS}")
    print(f"  PD (%)   : {pd_list}")

    results[vname] = {
        "input_dim":  int(X_tr.shape[1]),
        "best_val_auc": round(best_auc, 5),
        "n_epochs":   n_epochs,
        "pd_vs_snr":  {str(s): pd[s] for s in SNR_BINS},
    }

    keras.backend.clear_session()

# ── Save ──────────────────────────────────────────────────────────────────────
with open(str(OUT_JSON), "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved → {OUT_JSON}")

# ── Summary table ─────────────────────────────────────────────────────────────
print("\n── Summary: PD (%) at α=1e-7 ──────────────────────────────────────────")
header = f"{'Variant':<22}" + "".join(f"{s:>7}" for s in SNR_BINS)
print(header)
print("-" * len(header))
for vname, r in results.items():
    pd_row = [round(r["pd_vs_snr"][str(s)] * 100, 1) for s in SNR_BINS]
    print(f"{vname:<22}" + "".join(f"{v:>7.1f}" for v in pd_row))
