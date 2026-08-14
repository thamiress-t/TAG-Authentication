"""
ABLATION STUDY - DNN Correlator for TAG Authentication
=========================================================

Script para avaliar sistematicamente o impacto de cada componente
da arquitetura DNN na performance e eficiência computacional.

Uso:
    python ablation_study_dnn.py --dataset <path> --output <dir>
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import json
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, List

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class AblationConfig:
    """Configuration for a single ablation variant"""
    name: str
    units: Tuple[int, ...]  # e.g., (256, 128, 64) or (128, 64, 32)
    dropout_rates: Tuple[float, ...]
    batch_norm: bool = True
    batch_norm_only_first: bool = False
    num_features: int = 4  # 1 to 4
    activation: str = 'relu'  # 'relu', 'tanh', 'leaky_relu'
    description: str = ""

# ============================================================================
# ABLATION VARIANTS
# ============================================================================

ABLATION_VARIANTS = [
    # Baseline
    AblationConfig(
        name='Baseline',
        units=(256, 128, 64),
        dropout_rates=(0.3, 0.3, 0.2),
        batch_norm=True,
        description='256→128→64 + BatchNorm + Dropout(0.3,0.3,0.2)'
    ),
    
    # A: Profundidade
    AblationConfig(
        name='A1: 2 camadas',
        units=(256, 128),
        dropout_rates=(0.3, 0.3),
        batch_norm=True,
        description='Remove última camada oculta (64)'
    ),
    AblationConfig(
        name='A2: 1 camada',
        units=(256,),
        dropout_rates=(0.3,),
        batch_norm=True,
        description='Apenas 256 neurônios'
    ),
    AblationConfig(
        name='A3: 4 camadas',
        units=(256, 128, 64, 32),
        dropout_rates=(0.3, 0.3, 0.2, 0.1),
        batch_norm=True,
        description='Adiciona camada 32 neurônios'
    ),
    
    # B: Largura
    AblationConfig(
        name='B1: 50% redução',
        units=(128, 64, 32),
        dropout_rates=(0.3, 0.3, 0.2),
        batch_norm=True,
        description='Metade dos neurônios: 128→64→32'
    ),
    AblationConfig(
        name='B2: 75% redução',
        units=(64, 32, 16),
        dropout_rates=(0.3, 0.3, 0.2),
        batch_norm=True,
        description='3/4 redução: 64→32→16'
    ),
    AblationConfig(
        name='B3: Assimétrico',
        units=(192, 96, 48),
        dropout_rates=(0.3, 0.3, 0.2),
        batch_norm=True,
        description='Intermediário: 192→96→48'
    ),
    
    # C: BatchNormalization
    AblationConfig(
        name='C1: Sem BatchNorm',
        units=(256, 128, 64),
        dropout_rates=(0.3, 0.3, 0.2),
        batch_norm=False,
        description='Remove todas as BatchNorm layers'
    ),
    AblationConfig(
        name='C2: BatchNorm L1 only',
        units=(256, 128, 64),
        dropout_rates=(0.3, 0.3, 0.2),
        batch_norm=True,
        batch_norm_only_first=True,
        description='BatchNorm apenas na primeira camada'
    ),
    
    # D: Dropout
    AblationConfig(
        name='D1: Sem Dropout',
        units=(256, 128, 64),
        dropout_rates=(0.0, 0.0, 0.0),
        batch_norm=True,
        description='Remove todos os Dropout'
    ),
    AblationConfig(
        name='D2: Dropout leve',
        units=(256, 128, 64),
        dropout_rates=(0.1, 0.1, 0.1),
        batch_norm=True,
        description='Dropout reduzido: (0.1, 0.1, 0.1)'
    ),
    AblationConfig(
        name='D3: Dropout forte',
        units=(256, 128, 64),
        dropout_rates=(0.5, 0.5, 0.3),
        batch_norm=True,
        description='Dropout aumentado: (0.5, 0.5, 0.3)'
    ),
    
    # E: Features
    # Nota: Número de features afeta o input_dim
    # Será tratado separadamente no script
]

# ============================================================================
# MODEL BUILDERS
# ============================================================================

def build_dnn_model(
    input_dim: int,
    units: Tuple[int, ...],
    dropout_rates: Tuple[float, ...],
    batch_norm: bool = True,
    batch_norm_only_first: bool = False,
    activation: str = 'relu'
) -> keras.Model:
    """Build DNN architecture with flexible configuration"""
    
    assert len(units) == len(dropout_rates), "Units and dropout_rates length mismatch"
    
    # Use activation based on string
    if activation == 'tanh':
        act_fn = 'tanh'
    elif activation == 'leaky_relu':
        act_fn = keras.layers.LeakyReLU(alpha=0.2)
    else:  # 'relu' (default)
        act_fn = 'relu'
    
    model = models.Sequential()
    
    # Input layer
    model.add(layers.Input(shape=(input_dim,)))
    
    # Hidden layers
    for i, (u, dr) in enumerate(zip(units, dropout_rates)):
        model.add(layers.Dense(u, activation=act_fn))
        
        # BatchNorm (conditional)
        if batch_norm:
            if batch_norm_only_first and i > 0:
                pass  # Skip BatchNorm
            else:
                model.add(layers.BatchNormalization())
        
        # Dropout (always)
        if dr > 0:
            model.add(layers.Dropout(dr))
    
    # Output layer
    model.add(layers.Dense(1, activation='sigmoid'))
    
    return model

# ============================================================================
# TRAINING & EVALUATION
# ============================================================================

@dataclass
class EvaluationMetrics:
    """Metrics for a single evaluation"""
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc: float
    fnr: float
    fpr: float
    tp: int
    fp: int
    tn: int
    fn: int

def evaluate_model(
    model: keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> EvaluationMetrics:
    """Evaluate model on test set"""
    
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    return EvaluationMetrics(
        accuracy=accuracy_score(y_test, y_pred),
        precision=precision_score(y_test, y_pred, zero_division=0),
        recall=recall_score(y_test, y_pred, zero_division=0),
        f1=f1_score(y_test, y_pred, zero_division=0),
        auc=roc_auc_score(y_test, y_pred_proba),
        fnr=fnr,
        fpr=fpr,
        tp=int(tp),
        fp=int(fp),
        tn=int(tn),
        fn=int(fn)
    )

@dataclass
class AblationResult:
    """Result of a single ablation variant"""
    variant_name: str
    config_description: str
    num_parameters: int
    training_time_s: float
    inference_time_ms: float
    fnr_mean: float
    fnr_std: float
    auc_mean: float
    accuracy_mean: float
    
    # Computed fields
    @property
    def parameter_reduction_pct(self) -> float:
        """% reduction vs baseline (43265 params)"""
        baseline = 43265
        return 100 * (1 - self.num_parameters / baseline)
    
    @property
    def fnr_degradation_pct(self) -> float:
        """% degradation vs baseline (FNR=1e-7)"""
        baseline_fnr = 1e-7
        return 100 * (self.fnr_mean - baseline_fnr) / baseline_fnr
    
    @property
    def efficiency_score(self) -> float:
        """Efficiency metric (higher is better)"""
        # Score based on parameter reduction and FNR maintenance
        param_reduction = self.parameter_reduction_pct / 100  # Normalize to [0,1]
        fnr_maintenance = 1 - (self.fnr_degradation_pct / 100)  # Higher is better
        
        if fnr_maintenance < 0:
            return 0  # No reward for worse performance
        
        return fnr_maintenance / (1 + param_reduction)  # Reward efficiency

def train_and_evaluate(
    X_train, y_train, X_val, y_val, X_test, y_test,
    config: AblationConfig,
    input_dim: int = 4,
    num_runs: int = 1,
    verbose: bool = False
) -> AblationResult:
    """Train a model variant and collect metrics"""
    
    print(f"\n{'='*70}")
    print(f"  Variant: {config.name}")
    print(f"  Config: {config.description}")
    print(f"{'='*70}")
    
    # Build model
    model = build_dnn_model(
        input_dim=input_dim,
        units=config.units,
        dropout_rates=config.dropout_rates,
        batch_norm=config.batch_norm,
        batch_norm_only_first=config.batch_norm_only_first,
        activation=config.activation
    )
    
    num_params = model.count_params()
    print(f"  Parameters: {num_params:,}")
    
    # Compile
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss=BinaryCrossentropy(),
        metrics=['accuracy', keras.metrics.AUC()]
    )
    
    # Callbacks
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True, verbose=0
    )
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=0
    )
    
    # Calculate class weights
    class_weight = {
        0: np.sum(y_train == 1) / np.sum(y_train == 0),
        1: 1.0
    }
    
    # Training
    metrics_list = []
    times_training = []
    times_inference = []
    
    for run in range(num_runs):
        if num_runs > 1:
            print(f"  Run {run+1}/{num_runs}...")
        
        # Reset random seeds for reproducibility per run
        np.random.seed(42 + run)
        tf.random.set_seed(42 + run)
        
        # Measure training time
        t_start = time.time()
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=256,
            class_weight=class_weight,
            callbacks=[early_stopping, reduce_lr],
            verbose=0
        )
        
        t_train = time.time() - t_start
        times_training.append(t_train)
        
        # Measure inference time
        t_start = time.time()
        _ = model.predict(X_test[:100], verbose=0)  # Warm-up
        t_start = time.time()
        _ = model.predict(X_test, verbose=0)  # Actual inference
        t_inf = (time.time() - t_start) / len(X_test) * 1000  # ms per sample
        times_inference.append(t_inf)
        
        # Evaluate
        metrics = evaluate_model(model, X_test, y_test)
        metrics_list.append(metrics)
    
    # Aggregate metrics
    fnr_values = [m.fnr for m in metrics_list]
    auc_values = [m.auc for m in metrics_list]
    acc_values = [m.accuracy for m in metrics_list]
    
    result = AblationResult(
        variant_name=config.name,
        config_description=config.description,
        num_parameters=num_params,
        training_time_s=np.mean(times_training),
        inference_time_ms=np.mean(times_inference),
        fnr_mean=np.mean(fnr_values),
        fnr_std=np.std(fnr_values),
        auc_mean=np.mean(auc_values),
        accuracy_mean=np.mean(acc_values)
    )
    
    print(f"  FNR: {result.fnr_mean:.2e} ± {result.fnr_std:.2e}")
    print(f"  AUC: {result.auc_mean:.6f}")
    print(f"  Training: {result.training_time_s:.1f}s")
    print(f"  Inference: {result.inference_time_ms:.2f}ms/sample")
    print(f"  Efficiency Score: {result.efficiency_score:.2f}")
    
    return result

# ============================================================================
# MAIN ABLATION STUDY
# ============================================================================

def run_ablation_study(
    dataset_path: str,
    output_dir: str,
    num_runs: int = 1
):
    """Run complete ablation study"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("  ABLATION STUDY: DNN Correlator for TAG Authentication")
    print("="*70)
    
    # Load dataset
    print("\nLoading dataset...")
    try:
        import h5py
        with h5py.File(dataset_path, 'r') as f:
            X_train = f['X_train'][:]
            y_train = f['y_train'][:]
            X_val = f['X_val'][:]
            y_val = f['y_val'][:]
            X_test = f['X_test'][:]
            y_test = f['y_test'][:]
    except Exception as e:
        print(f"ERROR: Could not load dataset: {e}")
        print("Please provide a valid HDF5 dataset with X_train, y_train, X_val, y_val, X_test, y_test")
        return
    
    print(f"✓ Dataset loaded:")
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_val:   {X_val.shape}, y_val: {y_val.shape}")
    print(f"  X_test:  {X_test.shape}, y_test: {y_test.shape}")
    
    # Run ablations
    results = []
    for config in ABLATION_VARIANTS:
        try:
            result = train_and_evaluate(
                X_train, y_train, X_val, y_val, X_test, y_test,
                config,
                input_dim=X_train.shape[1],
                num_runs=num_runs,
                verbose=True
            )
            results.append(result)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
    
    # Create results DataFrame
    results_data = [asdict(r) for r in results]
    df = pd.DataFrame(results_data)
    
    # Save results
    csv_path = output_dir / "ablation_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Results saved to: {csv_path}")
    
    json_path = output_dir / "ablation_results.json"
    with open(json_path, 'w') as f:
        json.dump(results_data, f, indent=2)
    print(f"✓ JSON saved to: {json_path}")
    
    # Print summary table
    print("\n" + "="*70)
    print("  SUMMARY TABLE")
    print("="*70)
    print(df[['variant_name', 'num_parameters', 'fnr_mean', 'auc_mean', 
              'training_time_s', 'inference_time_ms', 'efficiency_score']].to_string())
    
    # Plots
    print("\nGenerating plots...")
    
    # Plot 1: Pareto Front (FNR vs. Parameters)
    plt.figure(figsize=(10, 6))
    plt.scatter(df['num_parameters'], df['fnr_mean'], s=100, alpha=0.6)
    for idx, row in df.iterrows():
        plt.annotate(row['variant_name'], 
                    (row['num_parameters'], row['fnr_mean']),
                    fontsize=8, alpha=0.7)
    plt.xlabel('Number of Parameters')
    plt.ylabel('False Negative Rate (FNR)')
    plt.title('Ablation Study: Pareto Front (FNR vs. Model Size)')
    plt.yscale('log')
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "ablation_pareto.png", dpi=150)
    print(f"  ✓ Pareto plot saved")
    
    # Plot 2: FNR vs Training Time
    plt.figure(figsize=(10, 6))
    plt.scatter(df['training_time_s'], df['fnr_mean'], s=100, alpha=0.6)
    for idx, row in df.iterrows():
        plt.annotate(row['variant_name'],
                    (row['training_time_s'], row['fnr_mean']),
                    fontsize=8, alpha=0.7)
    plt.xlabel('Training Time (seconds)')
    plt.ylabel('False Negative Rate (FNR)')
    plt.title('Ablation Study: FNR vs. Training Time')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "ablation_training_time.png", dpi=150)
    print(f"  ✓ Training time plot saved")
    
    # Plot 3: Efficiency Score Histogram
    plt.figure(figsize=(10, 6))
    bars = plt.barh(range(len(df)), df['efficiency_score'])
    plt.yticks(range(len(df)), df['variant_name'])
    plt.xlabel('Efficiency Score')
    plt.title('Ablation Study: Efficiency Scores (higher is better)')
    plt.grid(True, alpha=0.3, axis='x')
    
    # Color bars
    for i, bar in enumerate(bars):
        if df.iloc[i]['efficiency_score'] >= 2.0:
            bar.set_color('green')
            bar.set_alpha(0.8)
        elif df.iloc[i]['efficiency_score'] >= 1.0:
            bar.set_color('orange')
            bar.set_alpha(0.8)
        else:
            bar.set_color('red')
            bar.set_alpha(0.6)
    
    plt.tight_layout()
    plt.savefig(output_dir / "ablation_efficiency.png", dpi=150)
    print(f"  ✓ Efficiency plot saved")
    
    print("\n" + "="*70)
    print("  ABLATION STUDY COMPLETE")
    print("="*70)
    
    # Recommendations
    print("\nRECOMMENDATIONS:")
    best_efficiency = df.loc[df['efficiency_score'].idxmax()]
    print(f"  Best Efficiency: {best_efficiency['variant_name']}")
    print(f"    - Parameters: {best_efficiency['num_parameters']:,}")
    print(f"    - FNR: {best_efficiency['fnr_mean']:.2e}")
    print(f"    - Efficiency Score: {best_efficiency['efficiency_score']:.2f}")

# ============================================================================
# CLI
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Ablation study for DNN Correlator'
    )
    parser.add_argument(
        '--dataset',
        required=True,
        help='Path to HDF5 dataset'
    )
    parser.add_argument(
        '--output',
        default='./ablation_results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--runs',
        type=int,
        default=3,
        help='Number of runs per variant (default: 3)'
    )
    
    args = parser.parse_args()
    
    run_ablation_study(
        dataset_path=args.dataset,
        output_dir=args.output,
        num_runs=args.runs
    )
