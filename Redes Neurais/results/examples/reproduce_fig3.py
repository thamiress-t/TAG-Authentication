"""
Reproduce Fig. 3 (Xie et al. 2021 -- Auth-SUP scheme): PD and PFA at Bob
vs SNR at Bob (0-30 dB), for a target PFA bound eps=0.01 and rho_t^2=0.1.

Construction (matches the constant-PFA design used in that figure):
  For every SNR value, solve tau0(SNR) so that alpha(tau0) = eps EXACTLY
  (using the Rayleigh-fading-averaged closed form from the pics), then
  evaluate PD = 1 - beta at that same tau0. This is why PFA is a flat
  line at eps across the whole SNR sweep, while PD rises with SNR.

Two independent curves per quantity:
  - "Theo": semi-analytic, using the closed-form alpha_i/beta_i averaged
            over a chaotic tag ensemble (your notebook's real tag generator)
  - "Sim" : full Monte Carlo, explicit random Rayleigh fading + AWGN +
            chaotic tag generation, empirical alpha/beta at the SAME tau0
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm as sp_norm

rng = np.random.default_rng(7)

# ---------------------------------------------------------------------
# Tag generation -- identical to the notebook (tent map, msg XOR key seed)
# ---------------------------------------------------------------------
def _compute_seed(msg, key, K):
    msg_bits = ((msg + 1) // 2).astype(np.uint8)
    n       = min(K, len(msg_bits), len(key))
    m_pad   = np.resize(msg_bits[:n], K)
    k_pad   = np.resize(key[:n].astype(np.uint8), K)
    xor_seq = np.bitwise_xor(m_pad, k_pad)
    packed  = np.packbits(xor_seq[:64])
    xor_int = int.from_bytes(packed.tobytes(), 'big')
    xi      = (6 * xor_int**2 + xor_int + 1) % (2**64)
    return (xi / (2**64 - 1)) * 2.0 - 1.0

def generate_tags_batch(seeds, L, K=512):
    beta    = 1e-6
    warm_up = max(0, K - L)
    x   = np.asarray(seeds, dtype=np.float64)
    buf = np.empty((len(seeds), L), dtype=np.float64)
    k = 0
    for i in range(warm_up + L):
        x = np.where(x < beta,
                     2.0/(beta+1)*x + (1.0-beta)/(beta+1),
                     2.0/(beta-1)*x - (beta+1)/(beta-1))
        if i >= warm_up:
            buf[:, k] = x
            k += 1
    std = buf.std(axis=1, keepdims=True)
    std[std < 1e-10] = 1.0
    return (buf / (std * np.sqrt(3))).astype(np.float32)   # E[tag^2] ~ 1/3


# ---------------------------------------------------------------------
# Parameters -- matching the figure's legend
# ---------------------------------------------------------------------
RHO_T2   = 0.1                     # rho_t^2, given in the figure legend
RHO_T    = np.sqrt(RHO_T2)
RHO_S    = np.sqrt(1 - RHO_T2 * (1/3))   # power split: rho_s^2 + rho_t^2*E[tag^2]=1
EPS      = 0.01                    # target PFA bound
OMEGA    = 1.0                     # E[|h|^2], Rayleigh channel normalization
K_VAL    = 512
L_VAL    = 48                      # tag length (not given in fig legend -- tuned so
                                    # the PD transition sits around 5-11 dB, matching Fig. 3)
N_TAGS   = 300                     # tag ensemble size for the "Theo" curve
N_MC     = 20_000                  # channel/noise trials per SNR point, for "Sim"

snr_db_grid = np.linspace(0, 30, 31)


# ---------------------------------------------------------------------
# Closed-form alpha/beta (Rayleigh-fading-averaged, from the pics)
# ---------------------------------------------------------------------
def alpha_beta_rayleigh(t2, tau0, rho_t, sigma_w2, Omega):
    A = (tau0 ** 2 * rho_t ** 2) / (t2 * sigma_w2)
    alpha_i = 0.5 * (1.0 - np.sqrt(A * Omega / (2.0 + A * Omega)))
    B = ((tau0 - t2) ** 2 * rho_t ** 2) / (t2 * sigma_w2)
    term = 0.5 * (1.0 - np.sqrt(B * Omega / (2.0 + B * Omega)))
    beta_i = np.where(tau0 >= t2, 1.0 - term, term)
    return alpha_i.mean(), beta_i.mean()

def solve_tau0_for_target_alpha(t2_mean, eps, rho_t, sigma_w2, Omega):
    """Invert alpha(tau0) = eps for the fixed threshold tau0."""
    k = 2.0 * (1.0 - 2.0 * eps) ** 2 / (1.0 - (1.0 - 2.0 * eps) ** 2)
    A = k / Omega
    return np.sqrt(A * t2_mean * sigma_w2 / rho_t ** 2)


# ---------------------------------------------------------------------
# Full Monte Carlo (independent check): real fading, real tag gen,
# empirical alpha/beta at a GIVEN fixed threshold tau0
# ---------------------------------------------------------------------
def full_mc_alpha_beta(L, tau0, sigma_w2, N, rng):
    keys  = rng.integers(0, 2, (N, K_VAL), dtype=np.uint8)
    msgs  = rng.integers(0, 2, (N, L)) * 2 - 1
    seeds = np.array([_compute_seed(msgs[i], keys[i], K_VAL) for i in range(N)])
    tags  = generate_tags_batch(seeds, L, K_VAL)

    sigma_h = np.sqrt(OMEGA / 2.0)
    real = rng.normal(0, sigma_h, N)
    imag = rng.normal(0, sigma_h, N)
    h = np.maximum(np.sqrt(real**2 + imag**2), 1e-6)

    w = rng.normal(0, np.sqrt(sigma_w2), (N, L))

    # H0: attacker sends raw msg (no tag, no rho_s scaling)
    y_eq0 = msgs + w / h[:, None]
    tau_h0 = np.sum((y_eq0 - RHO_S * msgs) * tags, axis=1) / RHO_T
    alpha_mc = float(np.mean(tau_h0 > tau0))

    # H1: authentic, superimposed tag
    x1 = RHO_S * msgs + RHO_T * tags
    y_eq1 = x1 + w / h[:, None]
    tau_h1 = np.sum((y_eq1 - RHO_S * msgs) * tags, axis=1) / RHO_T
    beta_mc = float(np.mean(tau_h1 < tau0))

    return alpha_mc, beta_mc


# ---------------------------------------------------------------------
# Sweep SNR
# ---------------------------------------------------------------------
# fixed tag ensemble for the "Theo" curve (independent of SNR)
keys  = rng.integers(0, 2, (N_TAGS, K_VAL), dtype=np.uint8)
msgs  = rng.integers(0, 2, (N_TAGS, L_VAL)) * 2 - 1
seeds = np.array([_compute_seed(msgs[i], keys[i], K_VAL) for i in range(N_TAGS)])
tags  = generate_tags_batch(seeds, L_VAL, K_VAL)
t2 = np.sum(tags.astype(np.float64) ** 2, axis=1)
t2_mean = t2.mean()
print(f"L={L_VAL}, E[|t|^2]={t2_mean:.3f} (theory: L/3={L_VAL/3:.3f})")

alpha_theo, beta_theo, pd_theo = [], [], []
alpha_sim,  beta_sim,  pd_sim  = [], [], []

for snr_db in snr_db_grid:
    snr_lin = 10 ** (snr_db / 10.0)
    sigma_w2 = RHO_T2 * OMEGA / snr_lin        # SNR = rho_t^2*Omega/sigma_w^2

    tau0 = solve_tau0_for_target_alpha(t2_mean, EPS, RHO_T, sigma_w2, OMEGA)

    a_th, b_th = alpha_beta_rayleigh(t2, tau0, RHO_T, sigma_w2, OMEGA)
    alpha_theo.append(a_th); beta_theo.append(b_th); pd_theo.append(1 - b_th)

    a_mc, b_mc = full_mc_alpha_beta(L_VAL, tau0, sigma_w2, N_MC, rng)
    alpha_sim.append(a_mc); beta_sim.append(b_mc); pd_sim.append(1 - b_mc)

alpha_theo = np.array(alpha_theo); pd_theo = np.array(pd_theo)
alpha_sim  = np.array(alpha_sim);  pd_sim  = np.array(pd_sim)

# ---------------------------------------------------------------------
# Plot, styled like Fig. 3
# ---------------------------------------------------------------------
plt.figure(figsize=(6.5, 5.2))
plt.plot(snr_db_grid, pd_sim,   'bo-',  markerfacecolor='none', linewidth=1,
         label='Auth-SUP(PD)-Bob-Sim')
plt.plot(snr_db_grid, alpha_sim,'co--', markerfacecolor='none', linewidth=1,
         label='Auth-SUP(PFA)-Bob-Sim')
plt.plot(snr_db_grid, pd_theo,  'rx-',  linewidth=1.3,
         label='Auth-SUP(PD)-Bob-Theo')
plt.plot(snr_db_grid, alpha_theo,'m*--', linewidth=1.3,
         label='Auth-SUP(PFA)-Bob-Theo')

plt.xlabel('SNR at Bob (dB)')
plt.ylabel('Probability of Authentication')
plt.title(rf'Reproduction of Fig. 3  ($\varepsilon_{{PFA}}$={EPS}, $\rho_t^2$={RHO_T2}, L={L_VAL})')
plt.ylim([-0.02, 1.05])
plt.grid(alpha=0.3, linestyle=':')
plt.legend(loc='center right', fontsize=9)
plt.tight_layout()
plt.savefig("fig3_reproduction.png", dpi=150)
print("Saved fig3_reproduction.png")

print("\nSNR(dB)  PFA_theo   PFA_sim    PD_theo    PD_sim")
for i, s in enumerate(snr_db_grid):
    print(f"{s:6.1f}  {alpha_theo[i]:8.4f}  {alpha_sim[i]:8.4f}  {pd_theo[i]:8.4f}  {pd_sim[i]:8.4f}")
