"""
2-feature ablation: find minimum feature count to beat classical correlator.

Classical correlator baseline:
  0 dB: 8.1%   5 dB: 58.7%   10 dB: 99.9%   >=15 dB: 100%

New variants (V6–V8):
  V6: [|τ|, ĥ]      — correlator + channel estimate
  V7: [|τ|, SNR]    — correlator + SNR
  V8: [|τ|, E]      — correlator + energy

Merges results with nn15_feature_ablation.json.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
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

BASE       = Path(__file__).parent
DATA_DIR   = BASE / "results" / "data"
MODELS_DIR = BASE / "results" / "models"
DS_PATH    = DATA_DIR / "dataset_cnn_yeq_0_30dB.h5"
OUT_JSON   = DATA_DIR / "nn15_feature_ablation.json"

ALPHA    = 1e-7
Q_INV    = norm.ppf(1 - ALPHA)
SNR_BINS = [0, 5, 10, 15, 20, 25, 30]

CLASSICAL = {0: 0.081, 5: 0.587, 10: 0.999, 15: 1.0, 20: 1.0, 25: 1.0, 30: 1.0}

# ── Load dataset ──────────────────────────────────────────────────────────────
print("Loading dataset …")
with h5py.File(str(DS_PATH), "r") as f:
    Y_train = f["train/y_eq"][:].astype(np.float32)
    Y_val   = f["val/y_eq"][:].astype(np.float32)
    Y_test  = f["test/y_eq"][:].astype(np.float32)

    TAU_train = np.abs(f["train/tau_eq"][:]).astype(np.float32)
    TAU_val   = np.abs(f["val/tau_eq"][:]).astype(np.float32)
    TAU_test  = np.abs(f["test/tau_eq"][:]).astype(np.float32)

    SNR_train = f["train/snr"][:].astype(np.float32)
    SNR_val   = f["val/snr"][:].astype(np.float32)
    SNR_test  = f["test/snr"][:].astype(np.float32)

    LBL_train = f["train/y"][:].astype(np.float32)
    LBL_val   = f["val/y"][:].astype(np.float32)
    LBL_test  = f["test/y"][:].astype(np.float32)

H_EST_train = np.abs(Y_train).mean(axis=1)
H_EST_val   = np.abs(Y_val).mean(axis=1)
H_EST_test  = np.abs(Y_test).mean(axis=1)

E_train = (Y_train ** 2).mean(axis=1)
E_val   = (Y_val   ** 2).mean(axis=1)
E_test  = (Y_test  ** 2).mean(axis=1)

print(f"  Train: {len(LBL_train):,}   Val: {len(LBL_val):,}   Test: {len(LBL_test):,}")


# ── Feature builder ───────────────────────────────────────────────────────────
def feats(split, variant):
    tau = {"train": TAU_train, "val": TAU_val, "test": TAU_test}[split]
    h   = {"train": H_EST_train, "val": H_EST_val, "test": H_EST_test}[split]
    snr = {"train": SNR_train, "val": SNR_val, "test": SNR_test}[split]
    e   = {"train": E_train, "val": E_val, "test": E_test}[split]
    return {
        "V6_tau_h":      np.stack([tau, h],   axis=1),
        "V7_tau_snr":    np.stack([tau, snr], axis=1),
        "V8_tau_energy": np.stack([tau, e],   axis=1),
    }[variant]


# ── DNN ───────────────────────────────────────────────────────────────────────
def build_dnn(input_dim):
    inp = layers.Input(shape=(input_dim,))
    x = layers.Dense(256, kernel_initializer="he_uniform")(inp)
    x = layers.BatchNormalization()(x); x = layers.Activation("relu")(x); x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, kernel_initializer="he_uniform")(x)
    x = layers.BatchNormalization()(x); x = layers.Activation("relu")(x); x = layers.Dropout(0.3)(x)
    x = layers.Dense(64,  kernel_initializer="he_uniform")(x)
    x = layers.Activation("relu")(x); x = layers.Dropout(0.2)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    m = keras.Model(inp, out)
    m.compile(optimizer=Adam(1e-3), loss=BinaryCrossentropy(),
              metrics=[keras.metrics.AUC(name="auc")])
    return m


# ── D3F evaluation ────────────────────────────────────────────────────────────
def d3f_eval(sc_val, sc_test, snr_v, snr_t, lbl_v, lbl_t):
    pd = {}
    for snr in SNR_BINS:
        m0 = (snr_v >= snr - 2.5) & (snr_v < snr + 2.5) & (lbl_v == 0)
        m1 = (snr_t >= snr - 2.5) & (snr_t < snr + 2.5) & (lbl_t == 1)
        if m0.sum() < 50 or m1.sum() < 10:
            pd[snr] = 0.0; continue
        mu, sigma = sc_val[m0].mean(), sc_val[m0].std()
        pd[snr] = float((sc_test[m1] >= float(mu + Q_INV * sigma)).mean())
    return pd


# ── Train V6, V7, V8 ─────────────────────────────────────────────────────────
new_results = {}

for vname in ["V6_tau_h", "V7_tau_snr", "V8_tau_energy"]:
    print(f"\n{'='*60}\n  Variant: {vname}\n{'='*60}")

    X_tr = feats("train", vname)
    X_v  = feats("val",   vname)
    X_te = feats("test",  vname)

    mu  = X_tr.mean(axis=0); std = X_tr.std(axis=0); std[std < 1e-8] = 1.0
    X_tr_n = ((X_tr - mu) / std).astype(np.float32)
    X_v_n  = ((X_v  - mu) / std).astype(np.float32)
    X_te_n = ((X_te - mu) / std).astype(np.float32)

    print(f"  Input dim: {X_tr.shape[1]}")
    model = build_dnn(X_tr.shape[1])

    cbs = [
        callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=20,
                                restore_best_weights=True, verbose=0),
        callbacks.ModelCheckpoint(str(MODELS_DIR / f"ablation_{vname}.keras"),
                                  monitor="val_auc", mode="max", save_best_only=True, verbose=0),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8,
                                    min_lr=1e-6, verbose=0),
    ]

    hist = model.fit(X_tr_n, LBL_train, validation_data=(X_v_n, LBL_val),
                     epochs=150, batch_size=512, callbacks=cbs, verbose=1)

    best_auc = max(hist.history["val_auc"])
    n_ep = len(hist.history["loss"])
    print(f"  Best val AUC: {best_auc:.5f}   Epochs: {n_ep}")

    sc_val  = model.predict(X_v_n,  batch_size=1024, verbose=0).ravel()
    sc_test = model.predict(X_te_n, batch_size=1024, verbose=0).ravel()
    pd = d3f_eval(sc_val, sc_test, SNR_val, SNR_test, LBL_val, LBL_test)

    pd_list = [round(pd[s] * 100, 1) for s in SNR_BINS]
    beats   = ["✓" if pd[s] > CLASSICAL[s] else "✗" for s in SNR_BINS]
    print(f"  PD (%): {pd_list}")
    print(f"  > classical: {beats}")

    new_results[vname] = {
        "input_dim": int(X_tr.shape[1]),
        "best_val_auc": round(best_auc, 5),
        "n_epochs": n_ep,
        "pd_vs_snr": {str(s): pd[s] for s in SNR_BINS},
    }
    keras.backend.clear_session()

# ── Merge with previous results and save ─────────────────────────────────────
existing = {}
if OUT_JSON.exists():
    with open(str(OUT_JSON)) as f:
        existing = json.load(f)

existing.update(new_results)
with open(str(OUT_JSON), "w") as f:
    json.dump(existing, f, indent=2)
print(f"\nSaved → {OUT_JSON}")

# ── Full summary ──────────────────────────────────────────────────────────────
print("\n── Full ablation summary: PD (%) at α=1e-7 ────────────────────────────")
header = f"{'Variant':<22}" + "".join(f"{s:>7}" for s in SNR_BINS)
sep    = "-" * len(header)
print(header); print(sep)

CLASSICAL_ROW = [round(CLASSICAL[s] * 100, 1) for s in SNR_BINS]
print(f"{'Classical (baseline)':<22}" + "".join(f"{v:>7.1f}" for v in CLASSICAL_ROW))
print(sep)

ORDER = ["V1_baseline","V2_perfect_h","V3_no_snr","V4_no_energy",
         "V5_tau_only","V6_tau_h","V7_tau_snr","V8_tau_energy"]
for vname in ORDER:
    if vname not in existing:
        continue
    r = existing[vname]
    row = [round(r["pd_vs_snr"][str(s)] * 100, 1) for s in SNR_BINS]
    flag = " ← MIN?" if vname in ("V6_tau_h", "V7_tau_snr", "V8_tau_energy") else ""
    print(f"{vname:<22}" + "".join(f"{v:>7.1f}" for v in row) + flag)
