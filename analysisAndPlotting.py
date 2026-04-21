import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from scipy import stats

# Load and prepare data
file_path = 'Gold_Futures_Historical_Data.csv'
df = pd.read_csv(file_path)
df['Price'] = df['Price'].astype(str).str.replace(',', '').astype(float)
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)
df['Log_Return'] = np.log(df['Price'] / df['Price'].shift(1))
returns = df['Log_Return'].dropna().values
n = len(returns)

# MCMC Settings
iterations = 10000000
burn_in = 10000
thin_factor = 180

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

# Storage for raw samples
mu_raw = np.zeros(iterations)
sigma2_raw = np.zeros(iterations)
nu_raw = np.zeros(iterations)

# Storage for VaR and ES samples
VaR_raw = np.zeros(iterations)
ES_raw = np.zeros(iterations)

# Precompute constant for log posterior nu
log_gamma_2 = 0.6931471805599453  # log(2) from math module

def log_posterior_nu(nu, lambdas, returns, mu, sigma2):
    if nu <= 0:
        return -np.inf
    # log prior for nu: Gamma(2, scale=2)
    log_prior = np.log(nu) - nu / 2 - (log_gamma_2 + 2 * np.log(2))
    # log likelihood (simplified, vectorized)
    residuals = (returns - mu) ** 2 / sigma2
    log_lik = np.sum(np.log(1 + residuals / nu)) * (nu + 1) / 2
    return log_prior - log_lik

# MCMC Loop
for i in range(iterations):
    # 1. Sample lambda_i from Gamma distribution (vectorized)
    shape_lambda = (nu + 1) / 2
    rate_lambda = (nu + (returns - mu) ** 2 / sigma2) / 2
    lambdas = np.random.gamma(shape_lambda, 1.0 / rate_lambda)
    
    # 2. Sample mu from Normal distribution (conditional posterior)
    prec_0 = 1.0 / tau2_0
    prec_n = prec_0 + np.sum(lambdas) / sigma2
    mean_n = (prec_0 * mu_0 + np.sum(lambdas * returns) / sigma2) / prec_n
    mu = np.random.normal(mean_n, np.sqrt(1.0 / prec_n))
    
    # 3. Sample sigma^2 from Inverse Gamma (via Gamma)
    a_n = a_0 + n / 2.0
    b_n = b_0 + 0.5 * np.sum(lambdas * (returns - mu) ** 2)
    sigma2 = 1.0 / np.random.gamma(a_n, 1.0 / b_n)
    
    # 4. Sample nu using Metropolis-Hastings
    nu_prop = np.random.normal(nu, 0.5)
    if nu_prop > 0:
        log_alpha = log_posterior_nu(nu_prop, lambdas, returns, mu, sigma2) - \
                    log_posterior_nu(nu, lambdas, returns, mu, sigma2)
        if np.log(np.random.rand()) < log_alpha:
            nu = nu_prop
    
    # Store raw samples
    mu_raw[i] = mu
    sigma2_raw[i] = sigma2
    nu_raw[i] = nu

    # Calculate VaR and ES at 95% confidence level using current parameters
    # Returns follow Student's t-distribution: t(nu, mu, sigma2)
    sigma = np.sqrt(sigma2)
    VaR = stats.t.ppf(0.05, df=nu, loc=mu, scale=sigma)
    # ES: expected shortfall beyond VaR (mean of tail below VaR)
    # Computed as conditional expectation E[X | X < VaR] for t-distribution
    ES = mu - sigma * (nu + VaR ** 2 / sigma2) / (nu - 1) * stats.t.pdf(VaR, df=nu, loc=mu, scale=sigma) / stats.t.cdf(VaR, df=nu, loc=mu, scale=sigma)
    VaR_raw[i] = VaR
    ES_raw[i] = ES

    if (i + 1) % 5000 == 0:
        print(f"Iteration {i + 1}/{iterations}")

# Apply burn-in
mu_burned = mu_raw[burn_in:]
sigma2_burned = sigma2_raw[burn_in:]
nu_burned = nu_raw[burn_in:]

# Apply thinning
mu_thinned = mu_burned[::thin_factor]
sigma2_thinned = sigma2_burned[::thin_factor]
nu_thinned = nu_burned[::thin_factor]
VaR_thinned = VaR_raw[burn_in:][::thin_factor]
ES_thinned = ES_raw[burn_in:][::thin_factor]

print(f"\nRaw samples: {iterations}")
print(f"After burn-in: {len(mu_burned)}")
print(f"After thinning: {len(mu_thinned)}")

# Function to compute autocorrelation for a given lag
def compute_autocorrelation(samples, max_lag=50):
    n = len(samples)
    mean = np.mean(samples)
    var = np.var(samples)
    acf = np.zeros(max_lag + 1)
    for lag in range(max_lag + 1):
        if lag == 0:
            acf[lag] = 1.0
        else:
            cov = np.mean((samples[lag:] - mean) * (samples[:-lag] - mean))
            acf[lag] = cov / var
    return acf

# Function to plot autocorrelation
def plot_autocorrelation(samples, name, ax):
    acf = compute_autocorrelation(samples)
    lags = np.arange(len(acf))
    ax.bar(lags, acf, color='steelblue', alpha=0.7)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.axhline(y=1.96 / np.sqrt(len(samples)), color='red', linestyle='--', label='95% CI')
    ax.axhline(y=-1.96 / np.sqrt(len(samples)), color='red', linestyle='--')
    ax.set_title(f'Autocorrelation: {name}')
    ax.set_xlabel('Lag')
    ax.set_ylabel('ACF')
    ax.legend()
    ax.grid(True, alpha=0.3)

# Function to plot trace plot
def plot_trace(samples, name, ax):
    # Downsample for plotting to avoid OverflowError with large datasets
    step = max(1, len(samples) // 500000)
    iterations_range = np.arange(len(samples))[::step]
    samples_downsampled = samples[::step]
    ax.plot(iterations_range, samples_downsampled, color='steelblue', linewidth=0.5, alpha=0.7)
    ax.axhline(y=np.mean(samples), color='red', linestyle='--', label=f'Mean: {np.mean(samples):.4f}')
    ax.set_title(f'Trace Plot: {name}')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Value')
    ax.legend()
    ax.grid(True, alpha=0.3)

# Function to plot posterior histogram with KDE fitted line
def plot_histogram(samples, name, ax):
    # Plot histogram with higher precision (100 bins)
    ax.hist(samples, bins=100, color='steelblue', alpha=0.7, edgecolor='black', density=True)
    # Add KDE fitted line with color different from mean (red) and median (green)
    kde = stats.gaussian_kde(samples)
    x_range = np.linspace(min(samples), max(samples), 200)
    ax.plot(x_range, kde(x_range), color='purple', linewidth=2, label='Fit Curve')
    ax.axvline(x=np.mean(samples), color='red', linestyle='--', label=f'Mean: {np.mean(samples):.4f}')
    ax.axvline(x=np.median(samples), color='green', linestyle='--', label=f'Median: {np.median(samples):.4f}')
    ax.set_title(f'Posterior Histogram: {name}')
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True, alpha=0.3)

# Create plots directory if not exists
import os
os.makedirs('plots', exist_ok=True)

# Create figure for mu autocorrelation plot
fig_mu_ac, ax_mu_ac = plt.subplots(figsize=(10, 5))
plot_autocorrelation(mu_burned, 'mu (burned, no thinning)', ax_mu_ac)
plt.tight_layout()
plt.savefig(f'plots/mu_autocorrelation_{iterations}.png', dpi=600)
print(f"mu autocorrelation plot saved to 'plots/mu_autocorrelation_{iterations}.png'")
plt.close()

# Create figure for mu autocorrelation (thinned)
fig_mu_ac_t, ax_mu_ac_t = plt.subplots(figsize=(10, 5))
plot_autocorrelation(mu_thinned, 'mu (thinned)', ax_mu_ac_t)
plt.tight_layout()
plt.savefig(f'plots/mu_autocorrelation_thinned_{iterations}.png', dpi=600)
print(f"mu autocorrelation (thinned) plot saved to 'plots/mu_autocorrelation_thinned_{iterations}.png'")
plt.close()

# Create figure for sigma2 autocorrelation plot
fig_sigma2_ac, ax_sigma2_ac = plt.subplots(figsize=(10, 5))
plot_autocorrelation(sigma2_burned, 'sigma^2 (burned, no thinning)', ax_sigma2_ac)
plt.tight_layout()
plt.savefig(f'plots/sigma2_autocorrelation_{iterations}.png', dpi=600)
print(f"sigma2 autocorrelation plot saved to 'plots/sigma2_autocorrelation_{iterations}.png'")
plt.close()

# Create figure for sigma2 autocorrelation (thinned)
fig_sigma2_ac_t, ax_sigma2_ac_t = plt.subplots(figsize=(10, 5))
plot_autocorrelation(sigma2_thinned, 'sigma^2 (thinned)', ax_sigma2_ac_t)
plt.tight_layout()
plt.savefig(f'plots/sigma2_autocorrelation_thinned_{iterations}.png', dpi=600)
print(f"sigma2 autocorrelation (thinned) plot saved to 'plots/sigma2_autocorrelation_thinned_{iterations}.png'")
plt.close()

# Create figure for nu autocorrelation plot
fig_nu_ac, ax_nu_ac = plt.subplots(figsize=(10, 5))
plot_autocorrelation(nu_burned, 'nu (burned, no thinning)', ax_nu_ac)
plt.tight_layout()
plt.savefig(f'plots/nu_autocorrelation_{iterations}.png', dpi=600)
print(f"nu autocorrelation plot saved to 'plots/nu_autocorrelation_{iterations}.png'")
plt.close()

# Create figure for nu autocorrelation (thinned)
fig_nu_ac_t, ax_nu_ac_t = plt.subplots(figsize=(10, 5))
plot_autocorrelation(nu_thinned, 'nu (thinned)', ax_nu_ac_t)
plt.tight_layout()
plt.savefig(f'plots/nu_autocorrelation_thinned_{iterations}.png', dpi=600)
print(f"nu autocorrelation (thinned) plot saved to 'plots/nu_autocorrelation_thinned_{iterations}.png'")
plt.close()

# Create figure for mu trace plot
fig_mu_tr, ax_mu_tr = plt.subplots(figsize=(10, 5))
plot_trace(mu_thinned, 'mu', ax_mu_tr)
plt.tight_layout()
plt.savefig(f'plots/mu_trace_{iterations}.png', dpi=600)
print(f"mu trace plot saved to 'plots/mu_trace_{iterations}.png'")
plt.close()

# Create figure for sigma2 trace plot
fig_sigma2_tr, ax_sigma2_tr = plt.subplots(figsize=(10, 5))
plot_trace(sigma2_thinned, 'sigma^2', ax_sigma2_tr)
plt.tight_layout()
plt.savefig(f'plots/sigma2_trace_{iterations}.png', dpi=600)
print(f"sigma2 trace plot saved to 'plots/sigma2_trace_{iterations}.png'")
plt.close()

# Create figure for nu trace plot
fig_nu_tr, ax_nu_tr = plt.subplots(figsize=(10, 5))
plot_trace(nu_thinned, 'nu', ax_nu_tr)
plt.tight_layout()
plt.savefig(f'plots/nu_trace_{iterations}.png', dpi=600)
print(f"nu trace plot saved to 'plots/nu_trace_{iterations}.png'")
plt.close()

# Create figure for mu posterior histogram
fig_mu_h, ax_mu_h = plt.subplots(figsize=(10, 6))
plot_histogram(mu_thinned, 'mu', ax_mu_h)
plt.tight_layout()
plt.savefig(f'plots/mu_histogram_{iterations}.png', dpi=600)
print(f"mu histogram saved to 'plots/mu_histogram_{iterations}.png'")
plt.close()

# Create figure for sigma2 posterior histogram
fig_sigma2_h, ax_sigma2_h = plt.subplots(figsize=(10, 6))
plot_histogram(sigma2_thinned, 'sigma^2', ax_sigma2_h)
plt.tight_layout()
plt.savefig(f'plots/sigma2_histogram_{iterations}.png', dpi=600)
print(f"sigma2 histogram saved to 'plots/sigma2_histogram_{iterations}.png'")
plt.close()

# Create figure for nu posterior histogram
fig_nu_h, ax_nu_h = plt.subplots(figsize=(10, 6))
plot_histogram(nu_thinned, 'nu', ax_nu_h)
plt.tight_layout()
plt.savefig(f'plots/nu_histogram_{iterations}.png', dpi=600)
print(f"nu histogram saved to 'plots/nu_histogram_{iterations}.png'")
plt.close()

# Create figure for VaR posterior histogram
fig_VaR_h, ax_VaR_h = plt.subplots(figsize=(10, 6))
plot_histogram(VaR_thinned, 'VaR (95%)', ax_VaR_h)
plt.tight_layout()
plt.savefig(f'plots/VaR_histogram_{iterations}.png', dpi=600)
print(f"VaR histogram saved to 'plots/VaR_histogram_{iterations}.png'")
plt.close()

# Create figure for ES posterior histogram
fig_ES_h, ax_ES_h = plt.subplots(figsize=(10, 6))
plot_histogram(ES_thinned, 'ES (95%)', ax_ES_h)
plt.tight_layout()
plt.savefig(f'plots/ES_histogram_{iterations}.png', dpi=600)
print(f"ES histogram saved to 'plots/ES_histogram_{iterations}.png'")
plt.close()

# Print summary statistics
print("\n" + "="*60)
print("SUMMARY STATISTICS (after burn-in and thinning)")
print("="*60)
for name, samples in [('mu', mu_thinned), ('sigma^2', sigma2_thinned), ('nu', nu_thinned), ('VaR', VaR_thinned), ('ES', ES_thinned)]:
    print(f"\n{name}:")
    print(f"  Mean: {np.mean(samples):.6f}")
    print(f"  Std:  {np.std(samples):.6f}")
    print(f"  2.5%: {np.percentile(samples, 2.5):.6f}")
    print(f"  97.5%:{np.percentile(samples, 97.5):.6f}")

# Effective sample size approximation
def effective_sample_size(samples, max_lag=100):
    n = len(samples)
    acf = compute_autocorrelation(samples, max_lag)
    for k in range(1, max_lag + 1):
        if acf[k] <= 0 or k == max_lag:
            return n / (1 + 2 * np.sum(acf[1:k])) if k > 1 else n
    return n

print("\n" + "="*60)
print("EFFECTIVE SAMPLE SIZE (ESS)")
print("="*60)
for name, samples in [('mu', mu_thinned), ('sigma^2', sigma2_thinned), ('nu', nu_thinned), ('VaR', VaR_thinned), ('ES', ES_thinned)]:
    ess = effective_sample_size(samples)
    print(f"{name}: ESS = {ess:.0f} (original: {len(samples)})")

print("\nThinning analysis completed!")
print("Check the saved plots to verify thinning effectiveness:")
print("  1. Autocorrelation should drop quickly (within ~10-20 lags)")
print("  2. Trace plots should show 'fuzzy caterpillar' pattern (iid)")
print("  3. Posterior histograms should show smooth distributions")
