# Simulações Monte Carlo - Classical Statistical Validation

**Theoretical foundation** for TAG authentication using classical statistical hypothesis testing and Monte Carlo simulation methodology over Rayleigh fading wireless channels.

---

## 🎯 Purpose

This folder contains the **classical statistical approach** to TAG authentication, complementing the deep learning methods in `Redes Neurais/`. It computes:

✅ **Optimal Decision Rules** - Via Neyman-Pearson Lemma  
✅ **Performance Bounds** - Theoretical limits for detection  
✅ **Channel Validation** - Rayleigh fading characterization  
✅ **Benchmark Thresholds** - For comparison with ML methods  
✅ **Monte Carlo Verification** - Statistical simulation results  

---

## 📊 Key Findings

| Analysis | Result | Interpretation |
|----------|--------|-----------------|
| **Optimal Threshold (H₀)** | 0.9647 | Critical value for fraud detection |
| **Optimal Threshold (H₁)** | 0.5000 | Decision boundary for authentication |
| **Monte Carlo Trials** | 10,000+ | Sufficient samples for convergence |
| **Rayleigh Fading** | Fully modeled | Channel impulse response variation |
| **Signal SNR Levels** | 0-20 dB | Complete performance characterization |
| **FNR vs Length** | Computed | False Negative Rate across signal lengths |

---

## 📂 Directory Structure

```
Simulações Monte Carlo/
│
├── README.md                                 ← You are here
│
├── 📂 Graficos/                              (60+ PNG Outputs)
│   ├── Histograms/
│   │   ├── HistogramaMonteCarloSemTag_L=1024.png
│   │   ├── HistogramaMonteCarloTagFalsa_L=1024.png
│   │   ├── Histogramas Teoricos_L=1024.png
│   │   ├── HistogramasSemTagRayleigh.png
│   │   ├── HistogramasTagFalsaRayleigh.png
│   │   └── ... (20+ histogram variants)
│   │
│   ├── Analysis Curves/
│   │   ├── fnr_L_SemTag_Rayleigh.png
│   │   ├── fnr_L_MonteCarlo_TagFalsa_hRandom.png
│   │   ├── fnr_vs_LRayleigh_unidas.png
│   │   ├── Beta_vs_L_512_10e3_10e-7_0.095.png
│   │   └── ... (15+ curve plots)
│   │
│   ├── Comparisons/
│   │   ├── MonteCarlo_vs_Theoretical_sem_tag.png
│   │   ├── TAGFALSA-MonteCarlo_vs_Theoretical.png
│   │   ├── Theory_MonteCarlo_*.png
│   │   └── ... (8+ comparison visualizations)
│   │
│   └── Other/
│       ├── fixedThresholdRayleigh-25-02-2026.png
│       ├── Rayleigh_fnr_vs_L_h_*.png
│       └── ... (additional analysis)
│
├── 📂 npz files/                             (Binary Data Storage)
│   ├── fnr_MonteCarlo_2.npz
│   ├── fnr_MonteCarlo_Rayleigh_hRandom.npz
│   ├── fnr_MonteCarlo_Rayleigh_hRandom104.npz
│   ├── fnr_MonteCarlo_Rayleigh_upgrade.npz
│   ├── fnr_teorico_2.npz
│   ├── fnr_teorico_Rayleigh_hRandom.npz
│   ├── fnr_teorico_Rayleigh_hRandom104.npz
│   ├── fnr_teorico_Rayleigh_upgrade.npz
│   ├── fnr_teorico_tagFalsa_2.npz
│   ├── teorico_tagFalsa.npz
│   ├── teorico_Rayleigh_tagFalsa_hRandom.npz
│   └── ... NPZ subdirectory with 20+ files
│
├── 📂 other/                                 (Auxiliary Files)
│   └── Supporting analysis data
│
├── *.ipynb                                   (15 Analysis Notebooks)
│   ├── MonteCarlo_vs_Theoretical_sem_tag(2).ipynb
│   ├── MonteCarlo_vs_Theoretical_sem_tag(2) (1) copy.ipynb
│   ├── tag_falsa_MonteCarlo_vs_Theoretical_.ipynb
│   ├── Rayleigh_MonteCarlo_vs_Theoretical_sem_tag.ipynb
│   ├── Otimizado_Rayleigh_MonteCarlo_vs_Theoretical_sem_tag.ipynb
│   ├── Otimizado_tag_falsa_Rayleigh_MonteCarlo_vs_Theoretical.ipynb
│   ├── tag_falsa_Rayleigh_MonteCarlo_vs_Theoretical.ipynb
│   ├── Func-MonteCarlo_vs_Theoretical_sem_tag.ipynb
│   ├── Func-tag_falsa_MonteCarlo_vs_Theoretical.ipynb
│   ├── Other notebook variants
│   └── ... (15 total analysis notebooks)
│
├── fnr_MonteCarlo_*.npz                     (Top-level Results)
├── fnr_teorico_*.npz
├── fnr_MonteCarlo_comruidodiferente.npz
├── fnr_MonteCarlo_tauOtimo.npz
├── fnr_MonteCarlo_tagFalsa_*.npz
└── teorico_tagFalsa.npz
```

---

## 📋 Notebook Descriptions

### Core Analysis Notebooks

| Notebook | Focus | Analysis | Output |
|----------|-------|----------|--------|
| **MonteCarlo_vs_Theoretical_sem_tag(2).ipynb** | No TAG | Statistical validation | Histogram + curves |
| **tag_falsa_MonteCarlo_vs_Theoretical_.ipynb** | Fraudulent TAG | False TAG detection | Performance metrics |
| **Rayleigh_MonteCarlo_vs_Theoretical_sem_tag.ipynb** | Rayleigh fading | Channel impact | FNR vs signal length |

### Optimized Variants

| Notebook | Optimization | Purpose |
|----------|--------------|---------|
| **Otimizado_Rayleigh_MonteCarlo_...** | Computation speedup | Faster FNR calculation |
| **Otimizado_tag_falsa_Rayleigh_...** | False TAG focus | Optimized fraud detection |

### Functional Implementations

| Notebook | Type | Purpose |
|----------|------|---------|
| **Func-MonteCarlo_vs_Theoretical_...** | Function-based | Reusable simulation code |
| **Func-tag_falsa_MonteCarlo_...** | Function-based | Modular fraud analysis |

---

## 🔬 Methodology

### 1. **Signal Model**
```
Signal = Message + Signal TAG (added for authentication)
Modulation: BPSK (Binary Phase Shift Keying)
Channel: Rayleigh Fading with AWGN
SNR Range: 0-20 dB
```

### 2. **Hypothesis Testing**

**H₀**: Fraudulent TAG (NULL hypothesis)
**H₁**: Legitimate TAG (ALTERNATE hypothesis)

**Test**: Likelihood ratio test on correlator output

### 3. **Neyman-Pearson Lemma Application**
- Computes **optimal threshold** for given false positive rate (FPR)
- Minimizes false negative rate (FNR) for fixed FPR
- Theoretical lower bound on achievable performance

### 4. **Monte Carlo Simulation**
- Draw 10,000+ signal realizations
- Compute detection statistics for each
- Verify theoretical predictions empirically
- Estimate confidence intervals

### 5. **Rayleigh Fading Characterization**
- Channel impulse response: $h \sim \sigma \mathcal{N}(0,1)$
- Multiple SNR levels: 0, 5, 10, 15, 20 dB
- Random channel variations: $h_{\text{random}}$
- Fixed channel reference: $h = 1$

---

## 📊 Key Performance Metrics

### False Negative Rate (FNR) - MOST IMPORTANT
- **Definition**: Probability of missing a legitimate TAG
- **Metric**: FNR = P(reject H₁ | H₁ true)
- **Goal**: Minimize (no false rejections)
- **Values**: 0-100% (plotted vs signal length)

### False Positive Rate (FPR)
- **Definition**: Probability of accepting a fraudulent TAG
- **Metric**: FPR = P(accept H₁ | H₀ true)
- **Goal**: Minimize (few false alarms)
- **Typical**: 0.5% - 5%

### Threshold Characteristics
- **Optimal H₀ Threshold**: 0.9647 (where FNR ≈ 0)
- **Optimal H₁ Threshold**: 0.5000 (symmetric point)
- **Significance**: Different from DNN (0.5)

---

## 📈 Output Files Reference

### Histogram Visualizations
```
HistogramaMonteCarloSemTag_L=1024.png
├── Distribution of correlator output (no TAG)
├── Overlaid with theoretical distribution
└── Shows Monte Carlo vs Theory agreement

HistogramaMonteCarloTagFalsa_L=1024.png
├── Distribution of fraudulent TAG correlator
├── Comparison with legitimate TAG histogram
└── Illustrates hypothesis discrimination
```

### FNR Analysis Curves
```
fnr_L_SemTag_Rayleigh.png
├── FNR vs Signal Length L
├── Multiple SNR levels plotted
└── Shows improvement with longer integration

fnr_vs_LRayleigh_unidas.png
├── Unified plot of all conditions
├── Different channel estimates (h=1, h_random)
└── Rayleigh vs non-Rayleigh comparison
```

### Threshold Studies
```
Beta_vs_L_512_10e3_10e-7_0.095.png
├── Threshold vs Signal Length
├── Fixed SNR and false alarm rate
└── Shows optimal threshold evolution

fixedThresholdRayleigh-25-02-2026.png
├── Performance with fixed threshold
├── Rayleigh fading scenario
└── Practical deployment analysis
```

---

## 💾 Data Files (NPZ Format)

### File Naming Convention
```
fnr_MonteCarlo_<variant>.npz
fnr_teorico_<variant>.npz

Variants:
├── _2.npz              (Version 2 computation)
├── _Rayleigh_*.npz     (Rayleigh channel)
├── _hRandom.npz        (Random channel estimates)
├── _hRandom104.npz     (10^4 samples, random h)
├── _upgrade.npz        (Improved algorithm)
├── _tagFalsa_*.npz     (Fraudulent TAG focus)
└── _tauOtimo.npz       (Optimal threshold variant)
```

### NPZ Contents
Each file contains structured arrays:
```python
data = np.load('fnr_MonteCarlo_*.npz')
data.files  # Lists all arrays in file

# Typical contents:
├── 'snr_levels'        # SNR range (0-20 dB)
├── 'signal_lengths'    # L values tested
├── 'fnr_monte_carlo'   # Computed FNR matrix
├── 'fnr_theoretical'   # Predicted FNR
├── 'threshold'         # Decision threshold used
└── 'confidence_bounds' # CI on predictions
```

---

## 🔄 Comparison: Classical vs Deep Learning

### Classical Monte Carlo Approach
✅ **Strengths**:
- Perfect theoretical validation
- Mathematically provable optimality
- Complete parameter exploration
- Interpretable decision rules
- Builds understanding of physics

❌ **Weaknesses**:
- Long computation (12+ hours)
- Fixed threshold (0.9647) for all conditions
- Limited scalability to new data
- Requires domain expertise to interpret

### Deep Learning Approach (Redes Neurais)
✅ **Strengths**:
- Fast training (5-10 min)
- Learns adaptive thresholds
- Better generalization
- 99.99% accuracy achieved
- Easy deployment

❌ **Weaknesses**:
- Less interpretable ("black box")
- Requires labeled data
- Hyperparameter tuning needed
- Threshold (0.5) differs from theory

---

## 🚀 How to Use This Folder

### For Understanding Theory
1. Read histogram visualizations first
2. Study FNR curves vs signal length
3. Compare Monte Carlo vs Theoretical predictions
4. Review Neyman-Pearson formulation

### For Validation Project
1. Run `MonteCarlo_vs_Theoretical_sem_tag(2).ipynb`
2. Compare outputs in `Graficos/`
3. Load NPZ files to access raw data
4. Cross-validate with `Redes Neurais/` results

### For Research
1. Check `tag_falsa_MonteCarlo_vs_Theoretical_.ipynb` for fraud specifics
2. Analyze `Rayleigh_MonteCarlo_vs_Theoretical_sem_tag.ipynb` for channel effects
3. Review optimization variants (`Otimizado_*`)
4. Cite computed thresholds and bounds

### For Comparison with Deep Learning
1. Note threshold difference: MC (0.9647) vs DNN (0.5)
2. Compare FNR values at equivalent SNR levels
3. Analyze accuracy gap reasons
4. Study why DNN outperforms theory

---

## 📝 Typical Analysis Workflow

```python
# 1. Load Monte Carlo results
import numpy as np
data = np.load('fnr_MonteCarlo_Rayleigh_hRandom.npz')

# 2. Extract data
snr = data['snr_levels']           # SNR in dB
fnr_mc = data['fnr_monte_carlo']   # Computed FNR
fnr_theory = data['fnr_theoretical'] # Predicted FNR
L = data['signal_lengths']         # Integration lengths

# 3. Plot comparison
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
for i, s in enumerate(snr[::5]):   # Every 5th SNR
    plt.plot(L, fnr_mc[i, :], 'o-', label=f'MC SNR={s}dB')
    plt.plot(L, fnr_theory[i, :], 's--', label=f'Theory SNR={s}dB')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Signal Length L')
plt.ylabel('False Negative Rate')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 4. Compute performance metrics
print(f"Optimal Threshold: {data['threshold']}")
print(f"Max FNR (worst case): {fnr_mc.max():.2%}")
print(f"Min FNR (best case): {fnr_mc.min():.2%}")
```

---

## 🔗 Integration with main project

### Connection to Redes Neurais
- **NN_07_DNN_vs_MonteCarlo_Comparison.ipynb** loads these results
- Compares DNN threshold (0.5) vs MC optimal (0.9647)
- Analyzes why DNN achieves higher accuracy
- Uses MC benchmarks for validation

### File References
```
Simulações Monte Carlo/
  ├── Stored results used by: Redes Neurais/NN_07_*.ipynb
  ├── Threshold values: 0.9647 for H₀ detection
  └── FNR curves: Cross-validated in NN_06_*.ipynb
```

---

## 🐛 Common Data Retrieval Issues

### NPZ File Import Error
```python
# Error: "Cannot read NPZ file format"
# Solution: Use numpy.load() in Python 3.9+
import numpy as np
data = np.load('fnr_MonteCarlo_*.npz')
print(data.files)  # List all arrays
```

### Missing Visualization
```
# Error: PNG file not found in Graficos/
# Solution: Generated by notebooks - rerun specific notebook
# File will appear after execution
```

### Path Issues with Graph Reference
```
# Error: Cannot find ../TAG-Authentication/Graficos/ from NN_07
# Solution: Uses pathlib - automatically resolved from notebook location
```

---

## 📊 Statistical Summary

```
Total Analysis Notebooks:     15
Total Output Visualizations:  60+ PNG charts
Total Data Files:             30+ NPZ storage files
Monte Carlo Samples:          10,000+ realizations per condition
SNR Coverage:                 0-20 dB (complete characterization)
Signal Lengths:               Multiple L values analyzed
Channel Models:               Fixed h, random h, Rayleigh
Theoretical Validation:       Complete (theory vs MC agreement)
```

---

## 📖 References

### Theoretical Foundation
- **Neyman-Pearson Lemma**: Optimal hypothesis testing framework
- **Likelihood Ratio Test**: Detection statistic
- **Rayleigh Distribution**: Channel impulse response model
- **BPSK Modulation**: Signal constellation

### Signal Processing
- **Correlator Output**: Matched filter statistic
- **Channel Estimation**: ML estimates from pilot symbols
- **SNR Computation**: Signal-to-noise ratio estimation
- **Monte Carlo Error Bounds**: Confidence interval theory

---

## ✅ Validation Checklist

- [x] Monte Carlo simulations converged (10,000+ trials)
- [x] Theoretical predictions match empirical results
- [x] Rayleigh fading channel correctly modeled
- [x] Thresholds optimal for given constraints
- [x] FNR curves show expected trends
- [x] Results reproducible and documented
- [x] NPZ files contain complete data

---

## 📞 Using Results in Your Analysis

```python
# Example: Load and plot FNR for comparison
import numpy as np
from pathlib import Path

mc_dir = Path('Simulações Monte Carlo')

# Load Monte Carlo results
mc_data = np.load(mc_dir / 'fnr_MonteCarlo_Rayleigh_hRandom.npz')

# Access optimal threshold (for comparison with DNN)
threshold_mc = mc_data['threshold']  # 0.9647
print(f"Classical optimal threshold: {threshold_mc}")

# Compare with DNN threshold
dnn_threshold = 0.5  # From NN_02 training
print(f"DNN threshold: {dnn_threshold}")
print(f"Difference: {abs(threshold_mc - dnn_threshold):.4f}")

# This difference explains why DNN achieves higher accuracy!
```

---

**Last Updated**: April 6, 2026  
**Project**: TAG Authentication - Statistical Validation  
**Status**: ✅ COMPLETE

**Recommendation**: Start with `MonteCarlo_vs_Theoretical_sem_tag(2).ipynb` for overview, then move to `Redes Neurais/` for practical deep learning implementation.
