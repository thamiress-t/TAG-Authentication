#!/usr/bin/env python3
"""
Generate publication-quality figures for article2feat.tex
All figures in high DPI with proper styling for academic journals
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
import h5py
import json
from pathlib import Path
from scipy import stats
from scipy.stats import norm
import pickle

# Publication-quality settings
rcParams['figure.figsize'] = (8, 6)
rcParams['font.size'] = 10
rcParams['font.family'] = 'serif'
rcParams['axes.labelsize'] = 11
rcParams['axes.titlesize'] = 12
rcParams['xtick.labelsize'] = 9
rcParams['ytick.labelsize'] = 9
rcParams['legend.fontsize'] = 9
rcParams['lines.linewidth'] = 1.5
rcParams['lines.markersize'] = 6
rcParams['savefig.dpi'] = 300
rcParams['savefig.format'] = 'pdf'

# Paths
project_root = Path("/mnt/c/Users/thami/OneDrive/Documents/TAG-Authentication/Redes Neurais")
results_dir = project_root / "results"
data_dir = results_dir / "data"
models_dir = results_dir / "models"
fig_dir = results_dir / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("GENERATING PUBLICATION-QUALITY FIGURES")
print("=" * 80)

# ============================================================================
# Load Data and Results
# ============================================================================
print("\n[1/6] Loading data and results...")

# Dataset
datasets = list(data_dir.glob('dataset_*.h5'))
dataset_path = max(datasets, key=lambda p: p.stat().st_mtime)

with h5py.File(str(dataset_path), 'r') as f:
    tau_test = np.abs(f['test/tau_eq'][:])
    h_test = f['test/h'][:]
    y_test = f['test/y'][:].astype(np.float32)
    snr_test = f['test/snr'][:]

# Results JSON
with open(str(models_dir / 'results_2feat.json'), 'r') as f:
    results = json.load(f)

# Load models for predictions
import tensorflow as tf
from tensorflow import keras
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import IsolationForest

# Normalize features (using same stats as training)
tau_mean, tau_std = 131.2486, 140.0  # From training
h_mean, h_std = 0.8856, 0.65  # From training

X_test_norm = np.column_stack([
    (tau_test - tau_mean) / (tau_std + 1e-10),
    (h_test - h_mean) / (h_std + 1e-10)
]).astype(np.float32)

# Load trained models
dnn_model = keras.models.load_model(str(models_dir / 'dnn_2feat.keras'))
cnn_model = keras.models.load_model(str(models_dir / 'cnn_2feat.keras'))
with open(str(models_dir / 'svm_2feat.pkl'), 'rb') as f:
    svm = pickle.load(f)
with open(str(models_dir / 'knn_2feat.pkl'), 'rb') as f:
    knn = pickle.load(f)
with open(str(models_dir / 'iforest_2feat.pkl'), 'rb') as f:
    iforest = pickle.load(f)

# Get predictions
p_dnn = dnn_model.predict(X_test_norm, verbose=0).flatten()
p_cnn = cnn_model.predict(X_test_norm.reshape(-1, 2, 1), verbose=0).flatten()
p_svm = svm.predict_proba(X_test_norm)[:, 1]
p_knn = knn.predict_proba(X_test_norm)[:, 1]
anomaly_scores = -iforest.score_samples(X_test_norm)
p_iforest = (anomaly_scores - anomaly_scores.min()) / (anomaly_scores.max() - anomaly_scores.min() + 1e-10)

predictions = {
    'DNN-2feat': p_dnn,
    'CNN-2feat': p_cnn,
    'SVM': p_svm,
    'KNN': p_knn,
    'IForest': p_iforest
}

print("✓ Data and models loaded")

# ============================================================================
# Figure 1: IID Validation - Autocorrelation
# ============================================================================
print("[2/6] Figure 1: IID Validation (Autocorrelation)...")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# H0
tau_h0 = tau_test[y_test == 0]
h_h0 = h_test[y_test == 0]
lags = np.arange(0, 51)
acf_tau_h0 = [np.corrcoef(tau_h0[:-lag] if lag > 0 else tau_h0,
                           tau_h0[lag:] if lag > 0 else tau_h0)[0, 1]
              for lag in lags]
acf_h_h0 = [np.corrcoef(h_h0[:-lag] if lag > 0 else h_h0,
                         h_h0[lag:] if lag > 0 else h_h0)[0, 1]
           for lag in lags]

axes[0].stem(lags, acf_tau_h0, linefmt='C0-', markerfmt='C0o', basefmt=' ', label='$|\\tau|$')
axes[0].stem(lags, acf_h_h0, linefmt='C1--', markerfmt='C1s', basefmt=' ', label='$\\hat{h}$')
axes[0].axhline(0, color='k', linestyle='-', linewidth=0.5)
axes[0].axhline(0.05, color='r', linestyle='--', linewidth=0.8, alpha=0.5, label='95% CI')
axes[0].axhline(-0.05, color='r', linestyle='--', linewidth=0.8, alpha=0.5)
axes[0].set(xlabel='Lag', ylabel='Autocorrelation', title='Under $H_0$ (Non-authenticated)')
axes[0].legend()
axes[0].grid(alpha=0.3)
axes[0].set_ylim(-0.15, 0.4)

# H1
tau_h1 = tau_test[y_test == 1]
h_h1 = h_test[y_test == 1]
acf_tau_h1 = [np.corrcoef(tau_h1[:-lag] if lag > 0 else tau_h1,
                           tau_h1[lag:] if lag > 0 else tau_h1)[0, 1]
              for lag in lags]
acf_h_h1 = [np.corrcoef(h_h1[:-lag] if lag > 0 else h_h1,
                         h_h1[lag:] if lag > 0 else h_h1)[0, 1]
           for lag in lags]

axes[1].stem(lags, acf_tau_h1, linefmt='C0-', markerfmt='C0o', basefmt=' ', label='$|\\tau|$')
axes[1].stem(lags, acf_h_h1, linefmt='C1--', markerfmt='C1s', basefmt=' ', label='$\\hat{h}$')
axes[1].axhline(0, color='k', linestyle='-', linewidth=0.5)
axes[1].axhline(0.05, color='r', linestyle='--', linewidth=0.8, alpha=0.5, label='95% CI')
axes[1].axhline(-0.05, color='r', linestyle='--', linewidth=0.8, alpha=0.5)
axes[1].set(xlabel='Lag', ylabel='Autocorrelation', title='Under $H_1$ (Authenticated)')
axes[1].legend()
axes[1].grid(alpha=0.3)
axes[1].set_ylim(-0.15, 0.4)

plt.tight_layout()
fig.savefig(str(fig_dir / 'fig01_iid_autocorrelation.pdf'), dpi=300, bbox_inches='tight')
print(f"✓ Saved: fig01_iid_autocorrelation.pdf")
plt.close()

# ============================================================================
# Figure 2: IID Validation - Distribution Check
# ============================================================================
print("[3/6] Figure 2: IID Validation (Distribution Check)...")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# H0: tau histogram with KS test
n_half = len(tau_test) // 2
mask_h0_first = (y_test == 0) & (np.arange(len(y_test)) < n_half)
mask_h0_second = (y_test == 0) & (np.arange(len(y_test)) >= n_half)
tau_h0_first = tau_test[mask_h0_first]
tau_h0_second = tau_test[mask_h0_second]
ks_stat_h0, ks_pval_h0 = stats.ks_2samp(tau_h0_first, tau_h0_second)

axes[0].hist(tau_h0_first, bins=50, alpha=0.6, label=f'First half (n={len(tau_h0_first)})', density=True)
axes[0].hist(tau_h0_second, bins=50, alpha=0.6, label=f'Second half (n={len(tau_h0_second)})', density=True)
axes[0].set(xlabel='$|\\tau|$', ylabel='Density', title=f'$H_0$: KS p-value = {ks_pval_h0:.3f}')
axes[0].legend()
axes[0].grid(alpha=0.3)

# H1: tau histogram
mask_h1_first = (y_test == 1) & (np.arange(len(y_test)) < n_half)
mask_h1_second = (y_test == 1) & (np.arange(len(y_test)) >= n_half)
tau_h1_first = tau_test[mask_h1_first]
tau_h1_second = tau_test[mask_h1_second]
ks_stat_h1, ks_pval_h1 = stats.ks_2samp(tau_h1_first, tau_h1_second)

axes[1].hist(tau_h1_first, bins=50, alpha=0.6, label=f'First half (n={len(tau_h1_first)})', density=True)
axes[1].hist(tau_h1_second, bins=50, alpha=0.6, label=f'Second half (n={len(tau_h1_second)})', density=True)
axes[1].set(xlabel='$|\\tau|$', ylabel='Density', title=f'$H_1$: KS p-value = {ks_pval_h1:.3f}')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
fig.savefig(str(fig_dir / 'fig02_iid_distribution.pdf'), dpi=300, bbox_inches='tight')
print(f"✓ Saved: fig02_iid_distribution.pdf")
plt.close()

# ============================================================================
# Figure 3: Gaussian Score Distribution (D3F Validation)
# ============================================================================
print("[4/6] Figure 3: Score Distributions & D3F Gaussian Assumption...")

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

methods_to_plot = ['DNN-2feat', 'SVM', 'CNN-2feat']
colors = ['C0', 'C2', 'C1']

for idx, (method, color) in enumerate(zip(methods_to_plot, colors)):
    ax = axes[idx]
    p_method = predictions[method]
    p_h0 = p_method[y_test == 0]
    p_h1 = p_method[y_test == 1]

    # Plot distributions
    ax.hist(p_h0, bins=50, alpha=0.6, density=True, label='$H_0$ (Observed)', color='red')
    ax.hist(p_h1, bins=50, alpha=0.6, density=True, label='$H_1$ (Observed)', color='blue')

    # Overlay Gaussian fit for H0
    mu_h0 = p_h0.mean()
    sigma_h0 = p_h0.std()
    x_gauss = np.linspace(p_h0.min(), p_h0.max(), 100)
    y_gauss = norm.pdf(x_gauss, mu_h0, sigma_h0)
    ax.plot(x_gauss, y_gauss, 'r--', linewidth=2, label=f'Gaussian fit ($\\mu={mu_h0:.3f}$, $\\sigma={sigma_h0:.3f}$)')

    # Threshold line
    tau_alpha = results[method]['tau_alpha']
    ax.axvline(tau_alpha, color='k', linestyle='--', linewidth=1.5, label=f'$\\tau^*$ = {tau_alpha:.3f}')

    ax.set(xlabel='Score', ylabel='Density', title=f'{method}')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

plt.tight_layout()
fig.savefig(str(fig_dir / 'fig03_score_distributions.pdf'), dpi=300, bbox_inches='tight')
print(f"✓ Saved: fig03_score_distributions.pdf")
plt.close()

# ============================================================================
# Figure 4: P_D vs SNR Comparison
# ============================================================================
print("[5/6] Figure 4: Probability of Detection vs SNR...")

fig, ax = plt.subplots(figsize=(10, 6))

snr_bins = np.arange(0, 31, 5)
snr_centres = np.arange(2.5, 30, 5)

methods_to_plot = ['DNN-2feat', 'SVM', 'KNN', 'IForest', 'CNN-2feat']
colors_plot = {'DNN-2feat': 'steelblue', 'SVM': 'forestgreen', 'KNN': 'coral',
               'IForest': 'purple', 'CNN-2feat': 'orange'}
markers = {'DNN-2feat': 'o', 'SVM': 's', 'KNN': '^', 'IForest': 'D', 'CNN-2feat': 'v'}

for method in methods_to_plot:
    p_method = predictions[method]
    tau_alpha = results[method]['tau_alpha']

    pd_vs_snr = []
    for snr_lo in snr_bins[:-1]:
        snr_hi = snr_lo + 5
        mask = (snr_test >= snr_lo) & (snr_test < snr_hi) & (y_test == 1)
        if mask.sum() > 10:
            pd_vs_snr.append((p_method[mask] > tau_alpha).mean())
        else:
            pd_vs_snr.append(np.nan)

    ax.plot(snr_centres, pd_vs_snr, marker=markers[method], markersize=7,
            label=method, color=colors_plot[method], linewidth=2, linestyle='-')

ax.set(xlabel='SNR (dB)', ylabel='Probability of Detection', xlim=(-1, 31), ylim=(-0.05, 1.05))
ax.set_title(f'P_D vs SNR ($\\alpha = 10^{{-7}}$)', fontsize=12, fontweight='bold')
ax.legend(fontsize=10, loc='lower right')
ax.grid(alpha=0.3, linestyle='--')
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

plt.tight_layout()
fig.savefig(str(fig_dir / 'fig04_pd_vs_snr.pdf'), dpi=300, bbox_inches='tight')
print(f"✓ Saved: fig04_pd_vs_snr.pdf")
plt.close()

# ============================================================================
# Figure 5a: AUC Comparison (Bar Chart)
# ============================================================================
print("[6/6] Figure 5a: AUC Comparison...")

methods = list(results.keys())
aucs = [results[m]['auc'] for m in methods]
colors_bar = [colors_plot[m] for m in methods]

fig, ax = plt.subplots(figsize=(8, 5))

bars = ax.bar(range(len(methods)), aucs, color=colors_bar, alpha=0.7, edgecolor='black', linewidth=1.5)
ax.set(ylabel='AUC', title='Test Set AUC Comparison', ylim=(0.95, 1.001), xticks=range(len(methods)))
ax.set_xticklabels(methods, rotation=45, ha='right')
ax.axhline(1.0, color='gray', linestyle='--', alpha=0.3, linewidth=1)

for i, (m, auc) in enumerate(zip(methods, aucs)):
    ax.text(i, auc + 0.002, f'{auc:.5f}', ha='center', fontsize=9, fontweight='bold')

ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
fig.savefig(str(fig_dir / 'fig05a_auc_comparison.pdf'), dpi=300, bbox_inches='tight')
print(f"✓ Saved: fig05a_auc_comparison.pdf")
plt.close()

# ============================================================================
# Figure 5b: Operating Point (FPR vs P_D)
# ============================================================================
print("[6/6] Figure 5b: Operating Point Comparison...")

fig, ax = plt.subplots(figsize=(8, 5))

for method in methods:
    fpr = results[method]['fpr']
    pd = results[method]['pd_0db']
    ax.scatter(fpr, pd, s=150, color=colors_plot[method], marker=markers[method],
              label=method, edgecolor='black', linewidth=1, alpha=0.8, zorder=10)

ax.set(xlabel='FPR (linear scale)', ylabel='$P_D$ at 0 dB', title='Operating Point Comparison ($\\alpha = 10^{-7}$)')
ax.legend(fontsize=9, loc='lower right')
ax.grid(alpha=0.3, linestyle='--')
ax.set_ylim(-0.05, 1.05)

plt.tight_layout()
fig.savefig(str(fig_dir / 'fig05b_operating_point.pdf'), dpi=300, bbox_inches='tight')
print(f"✓ Saved: fig05b_operating_point.pdf")
plt.close()

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 80)
print("✓ ALL PUBLICATION-QUALITY FIGURES GENERATED")
print("=" * 80)
print(f"\nAll figures saved to: {fig_dir}")
print("\nFigure list:")
print("  1. fig01_iid_autocorrelation.pdf - IID independence test")
print("  2. fig02_iid_distribution.pdf - Identical distribution test (KS)")
print("  3. fig03_score_distributions.pdf - D3F Gaussian assumption validation")
print("  4. fig04_pd_vs_snr.pdf - Main result: P_D vs SNR for all methods")
print("  5. fig05_performance_comparison.pdf - AUC and operating point comparison")
print("\nRecommended article inclusion order:")
print("  - Seção III (Métodos): fig05_performance_comparison.pdf (AUC bars)")
print("  - Seção IV (Validação IID): fig01_iid_autocorrelation.pdf + fig02_iid_distribution.pdf")
print("  - Seção V (Resultados): fig04_pd_vs_snr.pdf (main result)")
print("  - Appendix (D3F): fig03_score_distributions.pdf (score calibration)")
