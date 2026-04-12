import pandas as pd
import numpy as np

# Load the data
file_path = 'Gold_Futures_Historical_Data.csv'
df = pd.read_csv(file_path)
df['Price'] = df['Price'].astype(str).str.replace(',', '').astype(float)
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)
df['Log_Return'] = np.log(df['Price'] / df['Price'].shift(1))
returns = df['Log_Return'].dropna().values

# Bayesian Gaussian model with conjugate Normal-Inverse-Gamma prior
# Model: x_i ~ N(mu, sigma^2)
# Prior: mu ~ N(mu_0, sigma^2 / kappa_0), sigma^2 ~ IG(alpha_0, beta_0)

# Hyperparameters (initial guesses)
mu_0 = 0.0
kappa_0 = 1.0
alpha_0 = 1.0
beta_0 = 0.01

n = len(returns)
x_bar = np.mean(returns)
sum_sq_diff = np.sum((returns - x_bar)**2)

# Posterior parameters
kappa_n = kappa_0 + n
mu_n = (kappa_0 * mu_0 + n * x_bar) / kappa_n
alpha_n = alpha_0 + n / 2
beta_n = beta_0 + 0.5 * sum_sq_diff + (kappa_0 * n * (x_bar - mu_0)**2) / (2 * kappa_n)

# Posterior mean and variance
# E[mu | data] = mu_n
# E[sigma^2 | data] = beta_n / (alpha_n - 1)

posterior_mu = mu_n
posterior_sigma_sq = beta_n / (alpha_n - 1)

print(f"Prior parameters: mu_0={mu_0}, kappa_0={kappa_0}, alpha_0={alpha_0}, beta_0={beta_0}")
print(f"Posterior parameters: mu_n={posterior_mu:.6f}, kappa_n={kappa_n}, alpha_n={alpha_n:.2f}, beta_n={beta_n:.6f}")
print(f"Posterior estimated mean (mu): {posterior_mu:.6f}")
print(f"Posterior estimated variance (sigma^2): {posterior_sigma_sq:.6f}")
