"""
Protocolo de validação das 4 hipóteses levantadas na discussão teórica de
Neyman-Pearson / CFAR (ver dissertacao_merged.tex, Seções sec:np_limit e
sec:info_audit).

Ordem de execução: da hipótese MAIS BARATA (sem GPU, sem treino) para a
MAIS CARA (treino incremental). Rode passo a passo — cada função é
independente e grava seu próprio JSON em results/data/.

  Passo 0  (custo ~zero, sem GPU): correlador CFAR-adaptativo "justo".
           Testa se um clássico com limiar condicionado a ĥ por amostra
           recupera paridade com V6/D (Hipótese 4).
  Passo 1  (custo zero): hipótese do logit D3F já foi respondida em
           NN_09 (results/data/nn09_logit_d3f.json) — apenas relemos e
           interpretamos aqui, sem novo treino (Hipótese 1: REJEITADA).
  Passo 2  (custo baixo, ~poucos min/época, warm-start): Variante E do
           NN_02c + ĥ explícito, reaproveitando os pesos do ramo z_eq já
           treinado em dnn_variant_E_best.keras (Hipótese 2).
  Passo 3  (custo baixo): camada de correlação com prior linear
           (matched filter) inicializada a partir de tag_ref, sobre a
           mesma entrada da Variante A/E, fine-tuning curto (Hipótese 3).

Uso:
    python validate_np_hypotheses.py --step 0
    python validate_np_hypotheses.py --step 1
    python validate_np_hypotheses.py --step 2 --epochs 15
    python validate_np_hypotheses.py --step 3 --epochs 15
    python validate_np_hypotheses.py --step all
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.stats import norm as scipy_norm

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "results" / "data"
MODELS_DIR = PROJECT_ROOT / "results" / "models"

DATASET_PATH = DATA_DIR / "dataset_cnn_yeq_0_30dB.h5"
ALPHA = 1e-7
Q_INV = float(scipy_norm.ppf(1 - ALPHA))  # ~5.199
SNR_BINS = [0, 5, 10, 15, 20, 25, 30]


def _d3f_pd(sc_val, sc_test, snr_val, snr_test, lbl_val, lbl_test):
    """D3F: ajusta N(mu, sigma^2) em H0 (val) por bin de SNR, extrapola
    limiar com Q_INV, mede PD em H1 (test)."""
    pd = {}
    for snr in SNR_BINS:
        m0 = (snr_val >= snr - 2.5) & (snr_val < snr + 2.5) & (lbl_val == 0)
        m1 = (snr_test >= snr - 2.5) & (snr_test < snr + 2.5) & (lbl_test == 1)
        if m0.sum() < 50 or m1.sum() < 10:
            pd[snr] = 0.0
            continue
        mu, sig = sc_val[m0].mean(), sc_val[m0].std()
        if sig < 1e-10:
            pd[snr] = 0.0
            continue
        thr = float(mu + Q_INV * sig)
        pd[snr] = float((sc_test[m1] >= thr).mean())
    return pd


# ---------------------------------------------------------------------------
# Passo 0 — Correlador CFAR-adaptativo "justo" (Hipótese 4)
# ---------------------------------------------------------------------------
def step0_cfar_adaptive_classical():
    """Sem treino, sem GPU. Reusa apenas o dataset já gerado.

    Constrói um correlador clássico cujo limiar é condicionado ao ĥ
    instantâneo de CADA amostra (não à faixa nominal de SNR). Se a teoria
    de sec:np_limit estiver correta, este correlador deve recuperar
    paridade com V6 (tau+h, ~84,7% em 0 dB) e não mais ficar em 8,1%.

    Estratégia: para cada amostra de teste, calibra o limiar usando as
    amostras de validação H0 cujo ĥ está numa vizinhança estreita do ĥ da
    amostra de teste (janela deslizante em h, não em SNR nominal). Isso
    aproxima um GLRT/CFAR de referência sem precisar re-treinar nada.
    """
    print("=" * 70)
    print("PASSO 0 — Correlador CFAR-adaptativo (condicionado a h, não a SNR)")
    print("=" * 70)

    with h5py.File(str(DATASET_PATH), "r") as f:
        tau_val = np.abs(f["val/tau_eq"][:]).astype(np.float64)
        h_val = f["val/h"][:].astype(np.float64)
        snr_val = f["val/snr"][:].astype(np.float64)
        lbl_val = f["val/y"][:].astype(np.float64)

        tau_test = np.abs(f["test/tau_eq"][:]).astype(np.float64)
        h_test = f["test/h"][:].astype(np.float64)
        snr_test = f["test/snr"][:].astype(np.float64)
        lbl_test = f["test/y"][:].astype(np.float64)

    # Baseline "clássico como reportado" (limiar fixo por faixa de SNR nominal)
    pd_classical_fixed = _d3f_pd(tau_val, tau_test, snr_val, snr_test, lbl_val, lbl_test)

    # CFAR-adaptativo: para cada amostra H1 de teste, usa uma janela em h
    # (não em SNR) para selecionar o characterization set H0 de validação.
    h0_val_mask = lbl_val == 0
    h_val_h0 = h_val[h0_val_mask]
    tau_val_h0 = tau_val[h0_val_mask]

    pd_cfar = {}
    rng = np.random.default_rng(42)
    for snr in SNR_BINS:
        m1 = (snr_test >= snr - 2.5) & (snr_test < snr + 2.5) & (lbl_test == 1)
        idx_h1 = np.where(m1)[0]
        if len(idx_h1) < 10:
            pd_cfar[snr] = 0.0
            continue

        # Amostragem para custo computacional (até 3000 pacotes por bin)
        if len(idx_h1) > 3000:
            idx_h1 = rng.choice(idx_h1, size=3000, replace=False)

        hits = 0
        for i in idx_h1:
            h_i = h_test[i]
            # janela adaptativa em h (expande até achar >=200 amostras H0)
            window = 0.02
            while True:
                sel = np.abs(h_val_h0 - h_i) < window
                if sel.sum() >= 200 or window > 2.0:
                    break
                window *= 1.5
            mu = tau_val_h0[sel].mean()
            sig = tau_val_h0[sel].std()
            if sig < 1e-10:
                continue
            thr = mu + Q_INV * sig
            if tau_test[i] >= thr:
                hits += 1
        pd_cfar[snr] = hits / len(idx_h1)

    result = {
        "description": (
            "Correlador classico com limiar condicionado a h instantaneo "
            "(janela deslizante em h) vs. limiar fixo por faixa nominal de SNR."
        ),
        "alpha": ALPHA,
        "classical_fixed_threshold": pd_classical_fixed,
        "classical_cfar_adaptive_h": pd_cfar,
        "reference_V6_tau_h_nn02b": {
            "0": 0.8466, "5": 0.9956, "10": 1.0, "15": 1.0, "20": 1.0, "25": 1.0, "30": 1.0
        },
    }
    out_path = DATA_DIR / "hypothesis4_cfar_adaptive_classical.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\n0dB:  fixo={pd_classical_fixed[0]:.3f}  CFAR-h={pd_cfar[0]:.3f}  "
          f"V6(NN)={result['reference_V6_tau_h_nn02b']['0']:.3f}")
    print(f"Salvo em: {out_path}")
    print(
        "\nInterpretação: se PD(CFAR-h) se aproximar de PD(V6), a Hipótese 4 é "
        "CONFIRMADA (a rede não supera o ótimo, apenas uma implementação de "
        "referência sem adaptação por pacote)."
    )
    return result


# ---------------------------------------------------------------------------
# Passo 1 — Releitura do resultado já existente do logit D3F (Hipótese 1)
# ---------------------------------------------------------------------------
def step1_logit_d3f_readout():
    """Não treina nada. NN_09 já testou D3F com logit pré-sigmoid em vez de
    probabilidade sigmoide. Resultado (results/data/nn09_logit_d3f.json):
    logit piora o PD (0,297 vs 0,842 em 0dB para 4feat) em vez de melhorar.
    Hipótese 1 (calibração D3F via sigmoide seria a causa da perda de
    desempenho) fica REJEITADA — não é necessário re-treinar nada."""
    print("=" * 70)
    print("PASSO 1 — Releitura de NN_09 (logit vs sigmoid D3F)")
    print("=" * 70)
    path = DATA_DIR / "nn09_logit_d3f.json"
    if not path.exists():
        print(f"Arquivo não encontrado: {path}. Rode NN_09 antes.")
        return None
    data = json.loads(path.read_text())
    sig0 = data["4feat_sigmoid"]["0"]
    log0 = data["4feat_logit"]["0"]
    print(f"4feat sigmoid @0dB: {sig0:.3f}")
    print(f"4feat logit   @0dB: {log0:.3f}")
    verdict = "REJEITADA" if log0 < sig0 else "CONFIRMADA (inesperado, investigar)"
    print(f"\nHipótese 1 (logit melhora calibração D3F): {verdict}")
    print("(logit piora o PD; a calibração sigmoide já é adequada.)")
    return data


# ---------------------------------------------------------------------------
# Passo 2 — Variante E + h (Hipótese 2), warm-start do checkpoint existente
# ---------------------------------------------------------------------------
def step2_variant_E_plus_h(epochs: int = 15):
    """Reaproveita os pesos já treinados de dnn_variant_E_best.keras
    (ramo que processa z_eq - tag_ref) e adiciona h_est como entrada
    auxiliar concatenada antes da primeira camada densa, com fine-tuning
    curto (poucas épocas) em vez de treinar do zero."""
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.losses import BinaryCrossentropy
    from tensorflow.keras.optimizers import Adam

    print("=" * 70)
    print("PASSO 2 — Variante E (z_eq - tag_ref) + h, warm-start de checkpoint")
    print("=" * 70)

    ckpt_path = MODELS_DIR / "dnn_variant_E_best.keras"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint não encontrado: {ckpt_path}")

    base_model = keras.models.load_model(str(ckpt_path))
    print(f"Checkpoint base carregado: {ckpt_path.name}")

    with h5py.File(str(DATASET_PATH), "r") as f:
        L_FIXED = int(f.attrs["L_FIXED"])

        def load_split(split):
            z = f[f"{split}/z_eq"][:].astype(np.float32)
            tag_keys = [f"{split}/tag_ref", f"{split}/tag_legit", f"{split}/tag"]
            tag_key = next((k for k in tag_keys if k in f), None)
            tag = f[tag_key][:].astype(np.float32)
            h = f[f"{split}/h"][:].astype(np.float32)
            y = f[f"{split}/y"][:].astype(np.float32)
            snr = f[f"{split}/snr"][:].astype(np.float32)
            x_e = z - tag  # Variante E
            h_est = np.abs(z).mean(axis=1).astype(np.float32)  # mesmo proxy do NN_02b
            return x_e, h_est, h, y, snr

        x_train, hest_train, _, y_train, _ = load_split("train")
        x_val, hest_val, _, y_val, snr_val = load_split("val")
        x_test, hest_test, _, y_test, snr_test = load_split("test")

    # Reconstrói uma rede idêntica à base, mas concatenando h_est logo na
    # entrada e reaproveitando os pesos das camadas Dense internas.
    inp_z = layers.Input(shape=(L_FIXED,), name="z_minus_tag")
    inp_h = layers.Input(shape=(1,), name="h_est")
    x = layers.Concatenate()([inp_z, inp_h])
    x = layers.Dense(256, kernel_initializer="he_uniform", name="dense_256")(x)
    x = layers.BatchNormalization(name="bn_256")(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, kernel_initializer="he_uniform", name="dense_128")(x)
    x = layers.BatchNormalization(name="bn_128")(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, kernel_initializer="he_uniform", name="dense_64")(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(1, activation="sigmoid", dtype="float32", name="p_auth")(x)
    model = keras.Model([inp_z, inp_h], out, name="DNN_variant_E_plus_h")

    # Warm-start: copia os pesos das camadas Dense/BN da variante E treinada
    # (a primeira Dense muda de shape por causa da coluna extra de h_est,
    # então copiamos apenas as camadas 128/64/saída, que são idênticas).
    # Nomes no checkpoint são auto-gerados pelo Keras (dense, dense_1, dense_2,
    # batch_normalization, batch_normalization_1, p_auth) — não os nomes
    # explícitos usados no modelo novo (dense_128, bn_128, dense_64).
    name_map = {
        "dense_1": "dense_128", "batch_normalization_1": "bn_128",
        "dense_2": "dense_64", "p_auth": "p_auth",
    }
    copied = 0
    for layer in base_model.layers:
        if layer.name in name_map:
            try:
                model.get_layer(name_map[layer.name]).set_weights(layer.get_weights())
                copied += 1
            except Exception as exc:  # shape mismatch etc.
                print(f"  (aviso) não copiou {layer.name}: {exc}")
    print(f"Camadas reaproveitadas do checkpoint: {copied}")

    model.compile(
        optimizer=Adam(learning_rate=5e-4),  # LR menor: já partimos de pesos bons
        loss=BinaryCrossentropy(),
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )

    model.fit(
        [x_train, hest_train], y_train,
        validation_data=([x_val, hest_val], y_val),
        epochs=epochs, batch_size=512, verbose=2,
        callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
    )

    sc_val = model.predict([x_val, hest_val], batch_size=1024, verbose=0).flatten()
    sc_test = model.predict([x_test, hest_test], batch_size=1024, verbose=0).flatten()
    pd_e_plus_h = _d3f_pd(sc_val, sc_test, snr_val, snr_test, y_val, y_test)

    result = {
        "description": "Variante E (z_eq - tag_ref) + h_est, warm-start de dnn_variant_E_best.keras",
        "epochs_run": epochs,
        "layers_reused_from_checkpoint": copied,
        "pd_d3f": pd_e_plus_h,
        "reference_variant_E_no_h": {
            "0": 0.002, "5": 0.002, "10": 0.0, "15": 0.0, "20": 1.0, "25": 1.0, "30": 1.0
        },
    }
    out_path = DATA_DIR / "hypothesis2_variantE_plus_h.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\n0dB: E+h={pd_e_plus_h[0]:.3f}  (E sem h={result['reference_variant_E_no_h']['0']:.3f})")
    print(f"Salvo em: {out_path}")
    return result


# ---------------------------------------------------------------------------
# Passo 3 — Camada de correlação com prior linear (Hipótese 3)
# ---------------------------------------------------------------------------
def step3_correlation_prior_layer(epochs: int = 15):
    """Adiciona uma camada Dense(1, use_bias=False) que computa o produto
    interno com tag_ref, inicializada com os pesos = tag_ref/||tag_ref||^2
    (prior de correlação), em vez de inicialização aleatória. Testa se dar
    à rede o "atalho" estrutural do correlador via inicialização (mas
    ainda treinável) permite que a Variante A se aproxime do teto teórico
    de paridade, sem precisar aprendê-lo do zero em alta dimensão."""
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.losses import BinaryCrossentropy
    from tensorflow.keras.optimizers import Adam

    print("=" * 70)
    print("PASSO 3 — Camada de correlação com prior linear (init = tag_ref)")
    print("=" * 70)

    with h5py.File(str(DATASET_PATH), "r") as f:
        L_FIXED = int(f.attrs["L_FIXED"])
        tag_keys = ["train/tag_ref", "train/tag_legit", "train/tag"]
        tag_key = next((k for k in tag_keys if k in f), None)
        tag_ref_sample = f[tag_key][0].astype(np.float32)  # usa 1a amostra como referência

        def load_split(split):
            z = f[f"{split}/z_eq"][:].astype(np.float32)
            y = f[f"{split}/y"][:].astype(np.float32)
            snr = f[f"{split}/snr"][:].astype(np.float32)
            return z, y, snr

        x_train, y_train, _ = load_split("train")
        x_val, y_val, snr_val = load_split("val")
        x_test, y_test, snr_test = load_split("test")

    # Inicializador da camada de correlação: w = tag_ref (produz algo
    # proporcional a tau quando aplicado a z_eq), permitindo que a rede
    # comece perto do correlador clássico e refine a partir daí.
    w_init = tag_ref_sample.reshape(L_FIXED, 1) / (np.linalg.norm(tag_ref_sample) + 1e-8)

    inp = layers.Input(shape=(L_FIXED,), name="z_eq")
    corr = layers.Dense(
        1, use_bias=False, name="correlation_prior",
        kernel_initializer=keras.initializers.Constant(w_init),
    )(inp)
    x = layers.Concatenate()([inp, corr])
    x = layers.Dense(256, kernel_initializer="he_uniform")(x)
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
    out = layers.Dense(1, activation="sigmoid", dtype="float32", name="p_auth")(x)
    model = keras.Model(inp, out, name="DNN_variant_A_corr_prior")

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss=BinaryCrossentropy(),
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )

    model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=epochs, batch_size=512, verbose=2,
        callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
    )

    sc_val = model.predict(x_val, batch_size=1024, verbose=0).flatten()
    sc_test = model.predict(x_test, batch_size=1024, verbose=0).flatten()
    pd_corr_prior = _d3f_pd(sc_val, sc_test, snr_val, snr_test, y_val, y_test)

    result = {
        "description": "Variante A (z_eq) com camada de correlação inicializada por tag_ref",
        "epochs_run": epochs,
        "pd_d3f": pd_corr_prior,
        "reference_variant_A_random_init": {
            "0": 0.004, "5": 0.001, "10": 0.0, "15": 0.862, "20": 1.0, "25": 1.0, "30": 1.0
        },
        "reference_variant_D_explicit_tau": {
            "0": 0.883, "5": 0.997, "10": 1.0, "15": 1.0, "20": 1.0, "25": 1.0, "30": 1.0
        },
    }
    out_path = DATA_DIR / "hypothesis3_correlation_prior_layer.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\n0dB: A+prior={pd_corr_prior[0]:.3f}  "
          f"(A rand-init={result['reference_variant_A_random_init']['0']:.3f}, "
          f"D expl.={result['reference_variant_D_explicit_tau']['0']:.3f})")
    print(f"Salvo em: {out_path}")
    return result


STEPS = {
    "0": step0_cfar_adaptive_classical,
    "1": step1_logit_d3f_readout,
    "2": step2_variant_E_plus_h,
    "3": step3_correlation_prior_layer,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", choices=["0", "1", "2", "3", "all"], required=True)
    parser.add_argument("--epochs", type=int, default=15, help="Épocas para os passos 2 e 3 (fine-tuning curto)")
    args = parser.parse_args()

    if args.step == "all":
        for key in ["0", "1", "2", "3"]:
            fn = STEPS[key]
            if key in ("2", "3"):
                fn(epochs=args.epochs)
            else:
                fn()
            print()
    else:
        fn = STEPS[args.step]
        if args.step in ("2", "3"):
            fn(epochs=args.epochs)
        else:
            fn()


if __name__ == "__main__":
    main()
