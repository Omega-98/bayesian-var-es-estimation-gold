import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

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
thin_factor = 50

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
    iterations_range = np.arange(len(samples))
    ax.plot(iterations_range, samples, color='steelblue', linewidth=0.5, alpha=0.7)
    ax.axhline(y=np.mean(samples), color='red', linestyle='--', label=f'Mean: {np.mean(samples):.4f}')
    ax.set_title(f'Trace Plot: {name}')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Value')
    ax.legend()
    ax.grid(True, alpha=0.3)

# Function to plot posterior histogram
def plot_histogram(samples, name, ax):
    ax.hist(samples, bins=50, color='steelblue', alpha=0.7, edgecolor='black', density=True)
    ax.axvline(x=np.mean(samples), color='red', linestyle='--', label=f'Mean: {np.mean(samples):.4f}')
    ax.axvline(x=np.median(samples), color='green', linestyle='--', label=f'Median: {np.median(samples):.4f}')
    ax.set_title(f'Posterior Histogram: {name}')
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True, alpha=0.3)

# Create figure for autocorrelation plots
fig1, axes1 = plt.subplots(3, 2, figsize=(14, 10))

plot_autocorrelation(mu_burned, 'mu (burned, no thinning)', axes1[0, 0])
plot_autocorrelation(sigma2_burned, 'sigma2 (burned, no thinning)', axes1[1, 0])
plot_autocorrelation(nu_burned, 'nu (burned, no thinning)', axes1[2, 0])

plot_autocorrelation(mu_thinned, 'mu (thinned)', axes1[0, 1])
plot_autocorrelation(sigma2_thinned, 'sigma2 (thinned)', axes1[1, 1])
plot_autocorrelation(nu_thinned, 'nu (thinned)', axes1[2, 1])

plt.tight_layout()
plt.savefig('autocorrelation_plots.png', dpi=150)
print("Autocorrelation plots saved to 'autocorrelation_plots.png'")

# Create figure for trace plots
fig2, axes2 = plt.subplots(3, 1, figsize=(14, 10))

plot_trace(mu_thinned, 'mu', axes2[0])
plot_trace(sigma2_thinned, 'sigma2', axes2[1])
plot_trace(nu_thinned, 'nu', axes2[2])

plt.tight_layout()
plt.savefig('trace_plots.png', dpi=150)
print("Trace plots saved to 'trace_plots.png'")

# Create figure for posterior histograms
fig3, axes3 = plt.subplots(3, 1, figsize=(10, 12))

plot_histogram(mu_thinned, 'mu', axes3[0])
plot_histogram(sigma2_thinned, 'sigma2', axes3[1])
plot_histogram(nu_thinned, 'nu', axes3[2])

plt.tight_layout()
plt.savefig('posterior_histograms.png', dpi=150)
print("Posterior histograms saved to 'posterior_histograms.png'")

# Print summary statistics
print("\n" + "="*60)
print("SUMMARY STATISTICS (after burn-in and thinning)")
print("="*60)
for name, samples in [('mu', mu_thinned), ('sigma2', sigma2_thinned), ('nu', nu_thinned)]:
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
for name, samples in [('mu', mu_thinned), ('sigma2', sigma2_thinned), ('nu', nu_thinned)]:
    ess = effective_sample_size(samples)
    print(f"{name}: ESS = {ess:.0f} (original: {len(samples)})")

print("\nThinning analysis completed!")
print("Check the saved plots to verify thinning effectiveness:")
print("  1. Autocorrelation should drop quickly (within ~10-20 lags)")
print("  2. Trace plots should show 'fuzzy caterpillar' pattern (iid)")
print("  3. Posterior histograms should show smooth distributions")
