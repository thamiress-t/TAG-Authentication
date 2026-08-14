"""
Monte Carlo simulation of the false-alarm probability (alpha) and
miss-detection probability (beta) for a tag-based (chaos-spread) detector
operating over a Rayleigh fading channel.

Model (from the derivation):

Conditioned on tag vector t^i, fading gain h_i and hypothesis:

  H0:  tau^i | H0, h_i, t^i ~ Normal(0, sigma_v_i^2)
  H1:  tau^i | H1, h_i, t^i ~ Normal(|t^i|^2, sigma_v_i^2)

  sigma_v_i^2 = |t^i|^2 * sigma_w^2 / (rho_t^2 * |h_i|^2)

with |h_i| Rayleigh distributed such that |h_i|^2 is Exponential with
mean Omega = E[|h_i|^2]  (Omega = 1 is the normalization used in the notes).

Detector rule: declare H1 if tau^i > tau0.

  alpha = Pr(tau^i > tau0 | H0)   -> false alarm / false positive
  beta  = Pr(tau^i < tau0 | H1)   -> miss detection / false negative

Because |t^i|^2 depends on a chaotic map whose pdf is not known in closed
form, the expectation over |t^i|^2 must be estimated by Monte Carlo:
several initial conditions of the chaotic map are drawn, a tag vector t^i
is generated from each one, and the closed-form expressions (already
averaged analytically over the fading and over the Gaussian decision
statistic) are averaged numerically over the resulting ensemble of tags.

Two independent estimators are implemented and cross-checked:

 1. Semi-analytical estimator: closed-form alpha_i(t^i) / beta_i(t^i)
    (a Gaussian-Q-function expression, already averaged over Rayleigh
    fading) evaluated per tag vector and averaged over the tag ensemble.

 2. Full Monte Carlo estimator: for every tag vector, many independent
    fading/noise realizations are drawn explicitly and tau^i is sampled
    from its conditional Gaussian law; alpha/beta are then obtained as
    empirical frequencies. This is a direct check of the semi-analytical
    formula and does not use the closed form at all.
"""

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------
# 1. Chaotic map / tag-vector generation
# --------------------------------------------------------------------------
def logistic_map(x0, n_samples, r=4.0):
    """Generate n_samples from the logistic map x_{k+1} = r*x_k*(1-x_k)."""
    x = np.empty(n_samples)
    x[0] = x0
    for k in range(1, n_samples):
        x[k] = r * x[k - 1] * (1.0 - x[k - 1])
    return x


def tent_map(x0, n_samples, mu=2.0):
    """Generate n_samples from the (piecewise-linear) tent map."""
    x = np.empty(n_samples)
    x[0] = x0
    for k in range(1, n_samples):
        xp = x[k - 1]
        x[k] = mu * xp if xp < 0.5 else mu * (1.0 - xp)
    return x


def generate_tag_vector(x0, L, map_type="logistic", bipolar=True):
    """
    Generate a length-L tag vector t^i from a chaotic map started at
    initial condition x0 in (0, 1).
    """
    if map_type == "logistic":
        seq = logistic_map(x0, L)
    elif map_type == "tent":
        seq = tent_map(x0, L)
    else:
        raise ValueError("unknown map_type: choose 'logistic' or 'tent'")

    if bipolar:
        # map (0,1) -> (-1,1), standard for chaos-based spreading sequences
        seq = 2.0 * seq - 1.0

    return seq


def generate_tag_ensemble(num_tags, L, map_type="logistic", bipolar=True,
                           seed=None):
    """
    Draw `num_tags` independent chaotic initial conditions and generate
    one tag vector t^i per initial condition (this is the Monte Carlo
    ensemble over which the expectation E_{|t^i|^2}[...] is estimated).

    Returns
    -------
    tags : ndarray, shape (num_tags, L)
    t2   : ndarray, shape (num_tags,)   -- |t^i|^2 for each tag
    """
    rng = np.random.default_rng(seed)
    x0s = rng.uniform(1e-4, 1 - 1e-4, size=num_tags)  # avoid fixed points 0, 1

    tags = np.array([generate_tag_vector(x0, L, map_type, bipolar)
                      for x0 in x0s])
    t2 = np.sum(tags ** 2, axis=1)
    return tags, t2


# --------------------------------------------------------------------------
# 2. Semi-analytical alpha / beta
#    (closed form in tau0 and |h|^2 via Rayleigh averaging; the remaining
#     expectation over |t^i|^2 is done numerically, over the tag ensemble)
# --------------------------------------------------------------------------
def Q(x):
    """Gaussian Q-function, Q(x) = P(Z > x) for Z ~ N(0,1)."""
    return norm.sf(x)


def alpha_beta_semianalytic(t2, tau0, rho_t, sigma_w2, Omega):
    """
    Evaluate, for every tag in the ensemble, the closed-form
    (Rayleigh-fading-averaged) expressions

        A_i     = tau0^2 * rho_t^2 / (|t^i|^2 * sigma_w^2)
        alpha_i = 1/2 * (1 - sqrt(A_i*Omega / (2 + A_i*Omega)))

        B_i     = (tau0 - |t^i|^2)^2 * rho_t^2 / (|t^i|^2 * sigma_w^2)
        beta_i  = 1 - 1/2*(1 - sqrt(B_i*Omega/(2+B_i*Omega)))   if tau0 >= |t^i|^2
                = 1/2*(1 - sqrt(B_i*Omega/(2+B_i*Omega)))       if tau0 <  |t^i|^2

    and returns the ensemble (Monte Carlo, over |t^i|^2) averages
    (alpha, beta) together with the per-tag values.
    """
    A = (tau0 ** 2 * rho_t ** 2) / (t2 * sigma_w2)
    alpha_i = 0.5 * (1.0 - np.sqrt(A * Omega / (2.0 + A * Omega)))

    B = ((tau0 - t2) ** 2 * rho_t ** 2) / (t2 * sigma_w2)
    term = 0.5 * (1.0 - np.sqrt(B * Omega / (2.0 + B * Omega)))
    beta_i = np.where(tau0 >= t2, 1.0 - term, term)

    return alpha_i.mean(), beta_i.mean(), alpha_i, beta_i


# --------------------------------------------------------------------------
# 3. Full Monte Carlo alpha / beta
#    (explicit sampling of fading + the Gaussian decision statistic,
#     no closed-form Q-function expression used -- a direct check of §2)
# --------------------------------------------------------------------------
def alpha_beta_full_mc(t2, tau0, rho_t, sigma_w2, Omega,
                        n_trials=20000, seed=None):
    """
    For every tag in the ensemble, draw n_trials independent fading/noise
    realizations and form the decision statistic tau^i explicitly under
    H0 and H1; alpha/beta are the resulting empirical probabilities,
    averaged over the tag ensemble.
    """
    rng = np.random.default_rng(seed)

    false_alarms = 0
    misses = 0
    total = t2.shape[0] * n_trials

    for t2_i in t2:
        # |h_i|^2 ~ Exponential(mean = Omega)
        h2 = rng.exponential(scale=Omega, size=n_trials)
        sigma_v2 = t2_i * sigma_w2 / (rho_t ** 2 * h2)
        std = np.sqrt(sigma_v2)

        # H0: tau ~ N(0, sigma_v2)
        tau_h0 = rng.normal(0.0, std)
        false_alarms += np.sum(tau_h0 > tau0)

        # H1: tau ~ N(|t^i|^2, sigma_v2)
        tau_h1 = rng.normal(t2_i, std)
        misses += np.sum(tau_h1 < tau0)

    alpha_mc = false_alarms / total
    beta_mc = misses / total
    return alpha_mc, beta_mc


# --------------------------------------------------------------------------
# 4. SNR helper
# --------------------------------------------------------------------------
def snr_db_to_sigma_w2(snr_db, rho_t, Omega):
    """
    Receiver SNR is defined as

        SNR = rho_t^2 * Omega / sigma_w^2         (Omega = E[|h_i|^2])

    Given a target SNR in dB, return the noise variance sigma_w^2 that
    produces it (rho_t and Omega held fixed).
    """
    snr_lin = 10.0 ** (snr_db / 10.0)
    return (rho_t ** 2 * Omega) / snr_lin


# --------------------------------------------------------------------------
# 5. Parameter sweeps
# --------------------------------------------------------------------------
def sweep_alpha_beta_vs_snr(snr_db_grid, L, num_tags, map_type, bipolar,
                             rho_t, Omega, tau0_fraction, seed=1):
    """
    Fixed tag length L. For each SNR (dB) in snr_db_grid, generate a tag
    ensemble, set tau0 = tau0_fraction * E[|t^i|^2], derive sigma_w^2 from
    the SNR definition, and evaluate the semi-analytical alpha and beta.
    """
    tags, t2 = generate_tag_ensemble(num_tags, L, map_type, bipolar, seed=seed)
    tau0 = tau0_fraction * t2.mean()

    alphas, betas = [], []
    for snr_db in snr_db_grid:
        sigma_w2 = snr_db_to_sigma_w2(snr_db, rho_t, Omega)
        a, b, _, _ = alpha_beta_semianalytic(t2, tau0, rho_t, sigma_w2, Omega)
        alphas.append(a)
        betas.append(b)

    return np.array(alphas), np.array(betas), tau0, t2


def sweep_beta_vs_L(L_grid, num_tags, map_type, bipolar,
                     rho_t, Omega, snr_db, tau0_fraction, seed=1):
    """
    Fixed SNR (dB). For each tag length L in L_grid, generate a (new) tag
    ensemble of that length, set tau0 = tau0_fraction * E[|t^i|^2], and
    evaluate the semi-analytical beta.
    """
    sigma_w2 = snr_db_to_sigma_w2(snr_db, rho_t, Omega)

    betas = []
    for L in L_grid:
        tags, t2 = generate_tag_ensemble(num_tags, L, map_type, bipolar, seed=seed)
        tau0 = tau0_fraction * t2.mean()
        _, b, _, _ = alpha_beta_semianalytic(t2, tau0, rho_t, sigma_w2, Omega)
        betas.append(b)

    return np.array(betas)


# --------------------------------------------------------------------------
# 6. Example run
# --------------------------------------------------------------------------
if __name__ == "__main__":

    # ---- common parameters ------------------------------------------------
    num_tags = 500          # number of chaotic initial conditions (ensemble)
    map_type = "logistic"   # 'logistic' or 'tent'
    bipolar = True

    rho_t = 1.0              # tag correlation / gain parameter
    Omega = 1.0                # E[|h_i|^2]  (notes use the normalization Omega = 1)
    tau0_fraction = 0.5          # tau0 = tau0_fraction * E[|t^i|^2]  (energy-detector threshold)

    # ---- sanity check: semi-analytic vs full Monte Carlo -------------------
    L0, snr0_db = 32, 15.0
    tags0, t2_0 = generate_tag_ensemble(num_tags, L0, map_type, bipolar, seed=1)
    tau0_0 = tau0_fraction * t2_0.mean()
    sigma_w2_0 = snr_db_to_sigma_w2(snr0_db, rho_t, Omega)

    alpha_sa0, beta_sa0, _, _ = alpha_beta_semianalytic(
        t2_0, tau0_0, rho_t, sigma_w2_0, Omega)
    alpha_mc0, beta_mc0 = alpha_beta_full_mc(
        t2_0, tau0_0, rho_t, sigma_w2_0, Omega, n_trials=20000, seed=2)

    print(f"Sanity check @ L={L0}, SNR={snr0_db} dB, tau0={tau0_0:.3f}:")
    print(f"  alpha: semi-analytic = {alpha_sa0:.6f}   full-MC = {alpha_mc0:.6f}")
    print(f"  beta : semi-analytic = {beta_sa0:.6f}   full-MC = {beta_mc0:.6f}")

    # =======================================================================
    # Plot 1: alpha vs SNR (0-30 dB), fixed L
    # =======================================================================
    snr_db_grid = np.linspace(0, 30, 31)
    alphas_vs_snr, betas_vs_snr, tau0_snr, t2_snr = sweep_alpha_beta_vs_snr(
        snr_db_grid, L=32, num_tags=num_tags, map_type=map_type, bipolar=bipolar,
        rho_t=rho_t, Omega=Omega, tau0_fraction=tau0_fraction, seed=1)

    plt.figure(figsize=(6.5, 5))
    plt.semilogy(snr_db_grid, alphas_vs_snr, marker="o", markersize=3)
    plt.xlabel("Receiver SNR (dB)")
    plt.ylabel(r"$\alpha$ (probability of false alarm)")
    plt.title(rf"$\alpha$ vs SNR  (L = 32, $\tau_0$ = {tau0_snr:.2f})")
    plt.grid(alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig("alpha_vs_snr.png", dpi=150)
    plt.close()
    print("Saved alpha_vs_snr.png")

    # =======================================================================
    # Plot 2: beta vs L (tag length), fixed SNR
    # =======================================================================
    L_grid = np.array([4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256])
    snr_fixed_db = 15.0
    betas_vs_L = sweep_beta_vs_L(
        L_grid, num_tags=num_tags, map_type=map_type, bipolar=bipolar,
        rho_t=rho_t, Omega=Omega, snr_db=snr_fixed_db,
        tau0_fraction=tau0_fraction, seed=1)

    plt.figure(figsize=(6.5, 5))
    plt.semilogy(L_grid, betas_vs_L, marker="o", markersize=4)
    plt.xlabel("Tag length L")
    plt.ylabel(r"$\beta$ (probability of miss detection)")
    plt.title(rf"$\beta$ vs tag length L  (SNR = {snr_fixed_db:.0f} dB, "
              rf"$\tau_0$ = {tau0_fraction:g}$\cdot E[|t^i|^2]$)")
    plt.grid(alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig("beta_vs_L.png", dpi=150)
    plt.close()
    print("Saved beta_vs_L.png")

    # =======================================================================
    # Plot 3: beta vs SNR (0-30 dB), fixed L
    # =======================================================================
    plt.figure(figsize=(6.5, 5))
    plt.semilogy(snr_db_grid, betas_vs_snr, marker="o", markersize=3, color="tab:red")
    plt.xlabel("Receiver SNR (dB)")
    plt.ylabel(r"$\beta$ (probability of miss detection)")
    plt.title(rf"$\beta$ vs SNR  (L = 32, $\tau_0$ = {tau0_snr:.2f})")
    plt.grid(alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig("beta_vs_snr.png", dpi=150)
    plt.close()
    print("Saved beta_vs_snr.png")
