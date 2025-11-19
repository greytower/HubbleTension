import numpy as np
import matplotlib

# --- CRITICAL CHANGE 1: Set a non-interactive backend to force file saving ---
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm

# --- 1. Observational Target (SH0ES vs. Planck) ---
TARGET_BOOST = 73.0 / 67.4  # approx 1.083


# --- 2. The Holographic Theory Function (Physics) ---
def theoretical_H0_boost(c, width):
    z_rec = 1100
    z_trans = 3400

    decay_exponent = -(z_rec - z_trans) / width
    c2_eff = c ** 2 / (1 + np.exp(decay_exponent))

    if c2_eff >= 0.999: return 999.0
    return np.sqrt(1 / (1 - c2_eff))


# --- 3. The Likelihood Function (Statistical Proof) ---
def ln_likelihood(theta):
    c, width = theta

    if not (0.0 < c < 0.9 and 1000 < width < 20000):
        return -np.inf

    model_boost = theoretical_H0_boost(c, width)
    sigma = 0.01

    chi2 = (model_boost - TARGET_BOOST) ** 2 / sigma ** 2
    return -0.5 * chi2


# --- 4. The Metropolis-Hastings MCMC Sampler ---
def run_mcmc(n_steps, initial_theta):
    chain = np.zeros((n_steps, 2))
    chain[0] = initial_theta
    current_lnprob = ln_likelihood(initial_theta)

    accepted = 0

    print(f"Starting MCMC Chain ({n_steps} steps)...")

    for i in range(1, n_steps):
        c_step = np.random.normal(0, 0.01)
        w_step = np.random.normal(0, 100)
        proposal = chain[i - 1] + np.array([c_step, w_step])

        proposal_lnprob = ln_likelihood(proposal)

        if np.log(np.random.rand()) < (proposal_lnprob - current_lnprob):
            chain[i] = proposal
            current_lnprob = proposal_lnprob
            accepted += 1
        else:
            chain[i] = chain[i - 1]

        # Omitted print statements for silent run...

    return chain


# --- 5. Execution ---
initial_guess = [0.42, 8500]
chain_data = run_mcmc(n_steps=10000, initial_theta=initial_guess)

# --- 6. Corner Plot (Visualization) ---
flat_chain = chain_data[2000:]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Histogram for c
axes[0].hist(flat_chain[:, 0], bins=50, color='royalblue', alpha=0.7)
axes[0].set_title('Posterior Distribution: Parameter $c$')
axes[0].set_xlabel('Holographic Parameter $c$')
axes[0].axvline(x=1.0 / np.sqrt(3), color='red', linestyle='--', label='Theoretical Unity (1/√3)')
axes[0].legend()

# Histogram for Width
axes[1].hist(flat_chain[:, 1], bins=50, color='seagreen', alpha=0.7)
axes[1].set_title('Posterior Distribution: Width $w$')
axes[1].set_xlabel('Transition Width (Δz)')

plt.suptitle('MCMC Results: The Final Parameter Estimation')
plt.tight_layout()

# --- CRITICAL CHANGE 2: Save figure instead of displaying ---
plt.savefig('final_mcmc_results.png')

print("\nSuccessfully saved MCMC histograms to 'final_mcmc_results.png'.")