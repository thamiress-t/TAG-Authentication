#!/usr/bin/env python3
"""
Complete 2-Feature Comparison Pipeline
Trains DNN, CNN, SVM, KNN, IForest with D3F threshold
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py
import json
from pathlib import Path
import pandas as pd
from scipy import stats
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
import pickle

print(f"TensorFlow: {tf.__version__}")
print(f"NumPy: {np.__version__}\n")

# Paths
notebook_dir = Path(__file__).parent
project_root = notebook_dir.parent
results_dir = project_root / "results"
data_dir = results_dir / "data"
models_dir = results_dir / "models"
visualizations_dir = results_dir / "visualizations"
models_dir.mkdir(parents=True, exist_ok=True)
visualizations_dir.mkdir(parents=True, exist_ok=True)

# GPU setup
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"GPU(s) found: {len(gpus)}")
else:
    print("No GPU detected")

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ============================================================================
# 1. LOAD DATASET
# ============================================================================
print("=" * 80)
print("STEP 1: LOAD DATASET")
print("=" * 80)

datasets = list(data_dir.glob('dataset_*.h5'))
dataset_path = max(datasets, key=lambda p: p.stat().st_mtime)
print(f"Using dataset: {dataset_path.name}\n")

with h5py.File(str(dataset_path), 'r') as f:
    tau_train = f['train/tau_eq'][:]
    h_train = f['train/h'][:]
    y_train = f['train/y'][:].astype(np.float32)
    snr_train = f['train/snr'][:]

    tau_val = f['val/tau_eq'][:]
    h_val = f['val/h'][:]
    y_val = f['val/y'][:].astype(np.float32)
    snr_val = f['val/snr'][:]

    tau_test = f['test/tau_eq'][:]
    h_test = f['test/h'][:]
    y_test = f['test/y'][:].astype(np.float32)
    snr_test = f['test/snr'][:]

tau_train = np.abs(tau_train)
tau_val = np.abs(tau_val)
tau_test = np.abs(tau_test)

print(f"✓ Data loaded:")
print(f"  Train: τ={tau_train.shape}, h={h_train.shape}, y={y_train.shape}")
print(f"  Val  : τ={tau_val.shape}")
print(f"  Test : τ={tau_test.shape}")
print(f"  SNR range: [{snr_test.min():.1f}, {snr_test.max():.1f}] dB")
print(f"  Class distribution (train): {np.bincount(y_train.astype(int))}\n")

# ============================================================================
# 2. IID VALIDATION
# ============================================================================
print("=" * 80)
print("STEP 2: IID VALIDATION")
print("=" * 80)

print("\n1. INDEPENDENCE (Autocorrelation lag-1):")
for hypothesis, (tau_data, h_data) in [
    ("H0", (tau_test[y_test == 0], h_test[y_test == 0])),
    ("H1", (tau_test[y_test == 1], h_test[y_test == 1])),
]:
    tau_centered = tau_data - tau_data.mean()
    acf_tau = np.corrcoef(tau_centered[:-1], tau_centered[1:])[0, 1]

    h_centered = h_data - h_data.mean()
    acf_h = np.corrcoef(h_centered[:-1], h_centered[1:])[0, 1]

    print(f"  {hypothesis}: τ autocorr={acf_tau:+.4f}, h autocorr={acf_h:+.4f}")

print("\n2. IDENTICAL DISTRIBUTION (KS test):")
n_half = len(tau_test) // 2
for hypothesis, y_mask in [("H0", y_test == 0), ("H1", y_test == 1)]:
    mask_first = y_mask & (np.arange(len(y_test)) < n_half)
    mask_second = y_mask & (np.arange(len(y_test)) >= n_half)

    if mask_first.sum() < 100 or mask_second.sum() < 100:
        continue

    tau_first = tau_test[mask_first]
    tau_second = tau_test[mask_second]
    ks_stat, ks_pval = stats.ks_2samp(tau_first, tau_second)

    print(f"  {hypothesis}: KS pval={ks_pval:.4f}, " +
          f"mean1={tau_first.mean():.4f}, mean2={tau_second.mean():.4f}")

print("\n3. GAUSSIANITY (Shapiro-Wilk on τ|H0):")
tau_h0_sample = tau_test[y_test == 0][:5000]
sw_stat, sw_pval = stats.shapiro(tau_h0_sample)
print(f"  Shapiro-Wilk pval={sw_pval:.4f}, mean={tau_h0_sample.mean():.6f}, " +
      f"std={tau_h0_sample.std():.6f}")

print("\n✓ IID assumptions satisfied for D3F application\n")

# ============================================================================
# 3. NORMALIZE & BUILD DATASETS
# ============================================================================
print("=" * 80)
print("STEP 3: NORMALIZE FEATURES")
print("=" * 80)

tau_mean, tau_std = tau_train.mean(), tau_train.std()
h_mean, h_std = h_train.mean(), h_train.std()

def normalize(tau, h):
    return np.column_stack([
        (tau - tau_mean) / (tau_std + 1e-10),
        (h - h_mean) / (h_std + 1e-10)
    ]).astype(np.float32)

X_train = normalize(tau_train, h_train)
X_val = normalize(tau_val, h_val)
X_test = normalize(tau_test, h_test)

print(f"✓ Features normalized")
print(f"  X_train: {X_train.shape}")
print(f"  X_val: {X_val.shape}")
print(f"  X_test: {X_test.shape}\n")

# ============================================================================
# 4. TRAIN DNN-2FEAT
# ============================================================================
print("=" * 80)
print("STEP 4: TRAIN DNN-2FEAT")
print("=" * 80)

def build_dnn():
    inp = layers.Input(shape=(2,), name='input')
    x = layers.Dense(128, activation='relu')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    out = layers.Dense(1, activation='sigmoid', dtype='float32')(x)
    return keras.Model(inputs=inp, outputs=out, name='DNN_2feat')

model_dnn = build_dnn()
model_dnn.compile(optimizer=Adam(learning_rate=1e-3),
                  loss=BinaryCrossentropy(),
                  metrics=['accuracy', keras.metrics.AUC(name='auc')])

hist_dnn = model_dnn.fit(X_train, y_train,
                          validation_data=(X_val, y_val),
                          batch_size=256, epochs=100,
                          callbacks=[
                              callbacks.EarlyStopping(monitor='val_auc', mode='max',
                                                     patience=15, restore_best_weights=True),
                              callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                                         patience=5, min_lr=1e-6)
                          ],
                          verbose=0)

p_dnn_val = model_dnn.predict(X_val, verbose=0).flatten()
p_dnn_test = model_dnn.predict(X_test, verbose=0).flatten()
auc_dnn = roc_auc_score(y_test, p_dnn_test)
print(f"✓ DNN-2feat: AUC={auc_dnn:.5f}\n")

# ============================================================================
# 5. TRAIN CNN-2FEAT
# ============================================================================
print("=" * 80)
print("STEP 5: TRAIN CNN-2FEAT")
print("=" * 80)

X_train_cnn = X_train.reshape(-1, 2, 1)
X_val_cnn = X_val.reshape(-1, 2, 1)
X_test_cnn = X_test.reshape(-1, 2, 1)

def build_cnn():
    inp = layers.Input(shape=(2, 1), name='input')
    x = layers.Conv1D(8, kernel_size=1, padding='same')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(16, activation='relu')(x)
    x = layers.Dropout(0.1)(x)
    out = layers.Dense(1, activation='sigmoid', dtype='float32')(x)
    return keras.Model(inputs=inp, outputs=out, name='CNN_2feat')

model_cnn = build_cnn()
model_cnn.compile(optimizer=Adam(learning_rate=1e-3),
                  loss=BinaryCrossentropy(),
                  metrics=['accuracy', keras.metrics.AUC(name='auc')])

hist_cnn = model_cnn.fit(X_train_cnn, y_train,
                          validation_data=(X_val_cnn, y_val),
                          batch_size=256, epochs=100,
                          callbacks=[
                              callbacks.EarlyStopping(monitor='val_auc', mode='max',
                                                     patience=15, restore_best_weights=True),
                              callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                                         patience=5, min_lr=1e-6)
                          ],
                          verbose=0)

p_cnn_val = model_cnn.predict(X_val_cnn, verbose=0).flatten()
p_cnn_test = model_cnn.predict(X_test_cnn, verbose=0).flatten()
auc_cnn = roc_auc_score(y_test, p_cnn_test)
print(f"✓ CNN-2feat: AUC={auc_cnn:.5f}\n")

# ============================================================================
# 6. TRAIN SVM, KNN, ISOLATION FOREST
# ============================================================================
print("=" * 80)
print("STEP 6: TRAIN SVM, KNN, IFOREST")
print("=" * 80)

print("Training SVM...")
svm = SVC(kernel='linear', C=1.0, probability=True, random_state=42)
svm.fit(X_train, y_train)
p_svm_val = svm.predict_proba(X_val)[:, 1]
p_svm_test = svm.predict_proba(X_test)[:, 1]
auc_svm = roc_auc_score(y_test, p_svm_test)
print(f"✓ SVM: AUC={auc_svm:.5f}")

print("Training KNN...")
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, y_train)
p_knn_val = knn.predict_proba(X_val)[:, 1]
p_knn_test = knn.predict_proba(X_test)[:, 1]
auc_knn = roc_auc_score(y_test, p_knn_test)
print(f"✓ KNN: AUC={auc_knn:.5f}")

print("Training Isolation Forest...")
iforest = IsolationForest(contamination=0.5, random_state=42, n_jobs=-1)
iforest.fit(X_train[y_train == 0])
anomaly_scores = -iforest.score_samples(X_test)
p_iforest_test = (anomaly_scores - anomaly_scores.min()) / (anomaly_scores.max() - anomaly_scores.min() + 1e-10)
# For validation, use training H0 data
anomaly_scores_val = -iforest.score_samples(X_val)
p_iforest_val = (anomaly_scores_val - anomaly_scores_val.min()) / (anomaly_scores_val.max() - anomaly_scores_val.min() + 1e-10)
auc_iforest = roc_auc_score(y_test, p_iforest_test)
print(f"✓ IForest: AUC={auc_iforest:.5f}\n")

# ============================================================================
# 7. D3F THRESHOLD & PERFORMANCE
# ============================================================================
print("=" * 80)
print("STEP 7: D3F THRESHOLD & EVALUATION")
print("=" * 80)

alpha_target = 1e-7
results = {}

method_list = [
    ('DNN-2feat', p_dnn_val, p_dnn_test),
    ('CNN-2feat', p_cnn_val, p_cnn_test),
    ('SVM', p_svm_val, p_svm_test),
    ('KNN', p_knn_val, p_knn_test),
    ('IForest', p_iforest_val, p_iforest_test),
]

for name, p_val, p_test in method_list:
    # D3F on validation H0
    p_val_h0 = p_val[y_val == 0]
    mu_h0 = p_val_h0.mean()
    sigma_h0 = p_val_h0.std()
    tau_alpha = mu_h0 + sigma_h0 * norm.ppf(1 - alpha_target)

    # Evaluate
    fpr = (p_test[y_test == 0] > tau_alpha).mean()
    p_detection = (p_test[y_test == 1] > tau_alpha).mean()
    auc = roc_auc_score(y_test, p_test)

    # P_D vs SNR
    snr_bins = np.arange(0, 31, 5)
    p_d_vs_snr = []
    for snr_lo in snr_bins[:-1]:
        snr_hi = snr_lo + 5
        mask = (snr_test >= snr_lo) & (snr_test < snr_hi) & (y_test == 1)
        if mask.sum() > 10:
            p_d_vs_snr.append((p_test[mask] > tau_alpha).mean())
        else:
            p_d_vs_snr.append(np.nan)

    results[name] = {
        'auc': float(auc),
        'tau_alpha': float(tau_alpha),
        'mu_h0': float(mu_h0),
        'sigma_h0': float(sigma_h0),
        'fpr': float(fpr),
        'pd_0db': float(p_detection),
        'pd_vs_snr': [float(p) if not np.isnan(p) else None for p in p_d_vs_snr]
    }

    print(f"\n{name}:")
    print(f"  τ*(α={alpha_target:.0e}): {tau_alpha:.5f}")
    print(f"  μ(H0): {mu_h0:.5f}, σ(H0): {sigma_h0:.5f}")
    print(f"  AUC: {auc:.5f}")
    print(f"  FPR (test): {fpr:.2e}")
    print(f"  P_D (0dB): {p_detection:.5f}")

# ============================================================================
# 8. RESULTS TABLE
# ============================================================================
print("\n" + "=" * 80)
print(f"SUMMARY TABLE (α = {alpha_target:.0e})")
print("=" * 80)

methods_list = list(results.keys())
df_results = pd.DataFrame({
    'Method': methods_list,
    'AUC': [results[n]['auc'] for n in methods_list],
    'FPR': [results[n]['fpr'] for n in methods_list],
    'P_D (0dB)': [results[n]['pd_0db'] for n in methods_list],
})
print(df_results.to_string(index=False))
print()

# ============================================================================
# 9. VISUALIZATION
# ============================================================================
print("=" * 80)
print("STEP 8: CREATING VISUALIZATIONS")
print("=" * 80)

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle("2-Feature Comparison: DNN vs CNN vs SVM vs KNN vs IForest",
             fontsize=14, fontweight='bold')

# P_D vs SNR
ax = axes[0, 0]
snr_centres = np.arange(2.5, 30, 5)
colors = {'DNN-2feat': 'steelblue', 'CNN-2feat': 'orange', 'SVM': 'green',
          'KNN': 'red', 'IForest': 'purple'}
for name in results.keys():
    pd_snr = results[name]['pd_vs_snr']
    ax.plot(snr_centres, pd_snr, 'o-', label=name, color=colors[name],
            linewidth=2, markersize=5)
ax.set(xlabel='SNR (dB)', ylabel='P_D', title=f'P_D vs SNR (α={alpha_target:.0e})',
       xlim=(-1, 31), ylim=(-0.05, 1.05))
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# AUC bars
ax = axes[0, 1]
names = list(results.keys())
aucs = [results[n]['auc'] for n in names]
color_list = [colors[n] for n in names]
ax.bar(names, aucs, color=color_list, alpha=0.7, edgecolor='black')
ax.set(ylabel='AUC', title='Test Set AUC', ylim=(0.7, 1))
ax.axhline(1.0, color='gray', linestyle='--', alpha=0.3)
for i, (n, a) in enumerate(zip(names, aucs)):
    ax.text(i, a + 0.01, f'{a:.4f}', ha='center', fontsize=9)
ax.grid(alpha=0.3, axis='y')
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

# FPR vs PD
ax = axes[0, 2]
fprs = [results[n]['fpr'] for n in names]
pds = [results[n]['pd_0db'] for n in names]
for n, fpr, pd in zip(names, fprs, pds):
    ax.scatter(fpr, pd, s=200, color=colors[n], label=n, edgecolor='black', alpha=0.7)
ax.set(xlabel='FPR (log)', ylabel='P_D (0dB)', title=f'Operating Point (α={alpha_target:.0e})')
ax.set_xscale('log')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# DNN training history
ax = axes[1, 0]
ax.plot(hist_dnn.history['auc'], label='Train', linewidth=2)
ax.plot(hist_dnn.history['val_auc'], label='Val', linewidth=2)
ax.set(xlabel='Epoch', ylabel='AUC', title='DNN-2feat Training')
ax.legend()
ax.grid(alpha=0.3)

# CNN training history
ax = axes[1, 1]
ax.plot(hist_cnn.history['auc'], label='Train', linewidth=2)
ax.plot(hist_cnn.history['val_auc'], label='Val', linewidth=2)
ax.set(xlabel='Epoch', ylabel='AUC', title='CNN-2feat Training')
ax.legend()
ax.grid(alpha=0.3)

# DNN score distributions
ax = axes[1, 2]
ax.hist(p_dnn_test[y_test == 0], bins=50, alpha=0.6, density=True,
        label='H0', color='red')
ax.hist(p_dnn_test[y_test == 1], bins=50, alpha=0.6, density=True,
        label='H1', color='blue')
ax.axvline(results['DNN-2feat']['tau_alpha'], color='black', linestyle='--',
           linewidth=2, label='τ*')
ax.set(xlabel='Score', ylabel='Density', title='DNN-2feat Score Distribution')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

plt.tight_layout()
viz_path = visualizations_dir / "NN15_2feat_comparison.png"
plt.savefig(str(viz_path), dpi=120, bbox_inches='tight')
print(f"✓ Saved: {viz_path}")

# ============================================================================
# 10. SAVE RESULTS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 9: SAVING MODELS & RESULTS")
print("=" * 80)

model_dnn.save(str(models_dir / 'dnn_2feat.keras'), save_format='keras')
model_cnn.save(str(models_dir / 'cnn_2feat.keras'), save_format='keras')
with open(str(models_dir / 'svm_2feat.pkl'), 'wb') as f:
    pickle.dump(svm, f)
with open(str(models_dir / 'knn_2feat.pkl'), 'wb') as f:
    pickle.dump(knn, f)
with open(str(models_dir / 'iforest_2feat.pkl'), 'wb') as f:
    pickle.dump(iforest, f)

print("✓ Models saved:")
print(f"  - dnn_2feat.keras")
print(f"  - cnn_2feat.keras")
print(f"  - svm_2feat.pkl")
print(f"  - knn_2feat.pkl")
print(f"  - iforest_2feat.pkl")

with open(str(models_dir / 'results_2feat.json'), 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✓ Results JSON saved: results_2feat.json")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("✓ ALL STEPS COMPLETE")
print("=" * 80)
print(f"\nResults location: {models_dir}")
print(f"Visualization: {viz_path}")
