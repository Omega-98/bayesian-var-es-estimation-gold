import pandas as pd
import numpy as np
import math

# Load the data
file_path = 'Gold_Futures_Historical_Data.csv'
df = pd.read_csv(file_path)
df['Price'] = df['Price'].astype(str).str.replace(',', '').astype(float)
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)
df['Log_Return'] = np.log(df['Price'] / df['Price'].shift(1))
returns = df['Log_Return'].dropna().values
n = len(returns)

# MCMC Settings (adaptive)
min_iterations = 5000
max_iterations = 20000
target_ess = 500
target_acceptance = 0.234
acceptance_window_size = 500
proposal_scale_nu = 0.5

# Initial Values
mu = np.mean(returns)
sigma2 = np.var(returns)
nu = 5.0
lambdas = np.ones(n)

# Priors
mu_0 = 0.0
tau2_0 = 10.0
a_0 = 0.01
b_0 = 0.01

# Storage (dynamic growth)
mu_samples = []
sigma2_samples = []
nu_samples = []
acceptance_window = []

def log_likelihood_nu(nu, returns, mu, sigma2):
    """Log-likelihood for t-distribution using auxiliary variables."""
    if nu < 2 or nu > 100:
        return -np.inf
    n = len(returns)
    return np.sum(
        math.lgamma((nu + 1) / 2) - math.lgamma(nu / 2)
        - 0.5 * np.log(nu * np.pi * sigma2)
        - ((nu + 1) / 2) * np.log(1 + (returns - mu) ** 2 / (nu * sigma2))
    )

def log_prior_nu(nu):
    """Log prior for nu (Gamma(2, scale=2))."""
    if nu < 2 or nu > 100:
        return -np.inf
    return (2 - 1) * np.log(nu) - nu / 2 - (math.log(math.factorial(1)) + 2 * np.log(2))

def log_posterior_nu(nu, returns, mu, sigma2):
    """Log posterior for nu = log likelihood + log prior."""
    if nu < 2 or nu > 100:
        return -np.inf
    return log_likelihood_nu(nu, returns, mu, sigma2) + log_prior_nu(nu)

# MCMC Loop with adaptive Metropolis and ESS-based convergence
def compute_ess(samples):
    """Compute effective sample size using autocorrelation at lag 1."""
    n = len(samples)
    if n < 100:
        return 0
    mean = np.mean(samples)
    var = np.var(samples)
    if var == 0:
        return n
    autocorr = np.correlate(samples - mean, samples - mean, mode='full')[n - 1:] / (var * n)
    return n / (1 + 2 * np.sum(autocorr[1:min(50, n // 2)]))

def proposal_adapted(acceptance_rate, scale):
    """Adapt proposal scale based on acceptance rate."""
    if acceptance_rate < 0.20:
        return scale * 0.9
    elif acceptance_rate > 0.30:
        return scale * 1.1
    return scale

def check_converged(iteration, samples_mu, samples_sigma2, samples_nu):
    """Check if MCMC has converged based on ESS only."""
    if iteration < min_iterations:
        return False
    ess_mu = compute_ess(np.array(samples_mu))
    ess_sigma2 = compute_ess(np.array(samples_sigma2))
    ess_nu = compute_ess(np.array(samples_nu))
    min_ess = min(ess_mu, ess_sigma2, ess_nu)
    return min_ess > target_ess

converged = False
while not converged:
    # 1. Sample lambda_i (auxiliary variable for t-distribution)
    lambdas = np.random.gamma((nu + 1) / 2, 1 / ((nu + (returns - mu) ** 2 / sigma2) / 2))

    # 2. Sample mu (Gibbs step - Normal)
    prec_0 = 1 / tau2_0
    prec_n = prec_0 + np.sum(lambdas) / sigma2
    mu_n = (prec_0 * mu_0 + np.sum(lambdas * returns) / sigma2) / prec_n
    mu = np.random.normal(mu_n, np.sqrt(1 / prec_n))

    # 3. Sample sigma^2 (Gibbs step - Inverse Gamma)
    a_n = a_0 + n / 2
    b_n = b_0 + 0.5 * np.sum(lambdas * (returns - mu) ** 2)
    sigma2 = 1.0 / np.random.gamma(a_n, 1.0 / b_n)

    # 4. Sample nu (Metropolis step with adaptive proposal)
    nu_prop = np.random.normal(nu, proposal_scale_nu)
    if nu_prop < 2:
        nu_prop = 2.0
    if nu_prop > 100:
        nu_prop = 100.0
    log_alpha = log_posterior_nu(nu_prop, returns, mu, sigma2) - log_posterior_nu(nu, returns, mu, sigma2)
    accepted = False
    if np.log(np.random.rand()) < log_alpha:
        nu = nu_prop
        accepted = True
    acceptance_window.append(1 if accepted else 0)

    # Adapt proposal scale based on acceptance rate
    if len(acceptance_window) >= acceptance_window_size:
        acc_rate = np.mean(acceptance_window)
        proposal_scale_nu = proposal_adapted(acc_rate, proposal_scale_nu)
        proposal_scale_nu = np.clip(proposal_scale_nu, 0.01, 10.0)
        acceptance_window = []

    # Store samples
    mu_samples.append(mu)
    sigma2_samples.append(sigma2)
    nu_samples.append(nu)

    # Check convergence
    current_iteration = len(mu_samples)
    if current_iteration >= max_iterations:
        print(f"Max iterations {max_iterations} reached")
        break
    converged = check_converged(current_iteration, mu_samples, sigma2_samples, nu_samples)

    if current_iteration % 2000 == 0:
        print(f"Iteration {current_iteration}")

iterations = len(mu_samples)
burn_in = min(1000, iterations // 5)
print(f"Converged at iteration {iterations}")

# Posterior estimates (convert to arrays for indexing)
mu_samples = np.array(mu_samples)
sigma2_samples = np.array(sigma2_samples)
nu_samples = np.array(nu_samples)

post_mu = np.mean(mu_samples[burn_in:])
post_sigma2 = np.mean(sigma2_samples[burn_in:])
post_nu = np.mean(nu_samples[burn_in:])

print(f"Posterior mean (mu): {post_mu:.6f}")
print(f"Posterior variance (sigma^2): {post_sigma2:.6f}")
print(f"Posterior degrees of freedom (nu): {post_nu:.2f}")

# VaR and ES calculation (95%)
alpha = 0.05
# Using posterior parameters to simulate t-distribution samples
# Simplified: use normal with inflated variance to simulate t
posterior_samples = np.random.normal(post_mu, np.sqrt(post_sigma2) * np.sqrt(post_nu/(post_nu-2)), 10000)
var = -np.percentile(posterior_samples, alpha * 100)
es = -np.mean(posterior_samples[posterior_samples < -var])

print(f"VaR (95%): {var:.6f}")
print(f"Expected Shortfall (ES) (95%): {es:.6f}")
