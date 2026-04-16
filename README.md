### STATS 211: Stochastic Process Course Project
- Topic: Gold VaR (Value at Risk) and ES (Expected Shortfall) Analysis Based on Bayesian Risk Analysis
- Data source: https://www.investing.com/commodities/gold-historical-data, from Apr. 2, 2022 to Apr. 2, 2026
- Research Method: t distribution, Metropolis within Gibbs
- Tools: Python (numpy, pandas, matplotlib, seaborn, yfinance), AI tools (Gemini, OpenCode, Minimax)



## Documentation for `bayesian_estimation.py` at Apr. 16, 2026.

---

## 1. Code Architecture

### 1.1 Overview

This code implements a **Metropolis-within-Gibbs MCMC sampler** to estimate the parameters of a **t-distribution** modeling gold futures log returns. The estimated parameters are then used to compute **Value-at-Risk (VaR)** and **Expected Shortfall (ES)** at the 95% confidence level.

### 1.2 Data Flow

```
Gold_Futures_Historical_Data.csv
           │
           ▼
    ┌─────────────────┐
    │  Data Loading   │  (Lines 6-13)
    │  & Preprocessing│
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │  MCMC Sampling  │  (Lines 96-148)
    │  Loop           │
    └────────┬────────┘
             │
      ┌───────┴───────┐
      ▼               ▼
┌─────────┐   ┌─────────────┐
│ Gibbs   │   │ Metropolis  │
│ Steps   │   │ Step        │
│ (mu,    │   │ (nu)        │
│ sigma2) │   │             │
└─────────┘   └──────┬──────┘
                     │
                     ▼
            ┌─────────────────┐
            │  Convergence    │
            │  Check (ESS)    │
            └────────┬────────┘
                     │
                     ▼
    ┌─────────────────────────┐
    │  Posterior Estimates    │
    │  & VaR/ES Calculation   │
    └─────────────────────────┘
```

### 1.3 Code Sections

| Lines | Section | Purpose |
|-------|---------|---------|
| 1-3 | Imports | pandas, numpy, math |
| 5-13 | Data Loading | Load CSV, compute log returns |
| 15-21 | MCMC Settings | Adaptive iteration parameters |
| 23-33 | Initial Values & Priors | Starting values and prior hyperparameters |
| 35-39 | Storage | Dynamic lists for MCMC samples |
| 41-62 | Log Functions | Log-likelihood, log-prior, log-posterior for nu |
| 65-93 | Helper Functions | ESS computation, proposal adaptation, convergence check |
| 95-148 | MCMC Loop | Main sampling algorithm |
| 150-172 | Output | Posterior estimates and VaR/ES |

---

## 2. Variables and Their Purposes

### 2.1 Data Variables

| Variable | Type | Purpose |
|----------|------|---------|
| `file_path` | str | Path to CSV data file |
| `df` | DataFrame | Raw DataFrame loaded from CSV |
| `returns` | ndarray | Log returns of gold futures prices |
| `n` | int | Number of observations in returns |

### 2.2 Model Parameters (Updated During MCMC)

| Variable | Type | Description |
|----------|------|-------------|
| `mu` | float | Mean parameter of the t-distribution |
| `sigma2` | float | Variance parameter of the t-distribution |
| `nu` | float | Degrees of freedom parameter (ν > 2) |
| `lambdas` | ndarray | Auxiliary variables (one per observation) for data augmentation |

### 2.3 Prior Hyperparameters

| Variable | Value | Distribution | Parameter Meaning |
|----------|-------|--------------|------------------|
| `mu_0` | 0.0 | Normal | Prior mean for μ |
| `tau2_0` | 10.0 | Normal | Prior variance for μ (τ²₀) |
| `a_0` | 0.01 | Inverse Gamma | Shape parameter α₀ |
| `b_0` | 0.01 | Inverse Gamma | Scale parameter β₀ |

### 2.4 MCMC Settings

| Variable | Value | Purpose |
|----------|-------|---------|
| `min_iterations` | 5000 | Minimum iterations before checking convergence |
| `max_iterations` | 20000 | Hard stop to prevent infinite loops |
| `target_ess` | 500 | Target Effective Sample Size for convergence |
| `target_acceptance` | 0.234 | Optimal acceptance rate (for reference) |
| `acceptance_window_size` | 500 | Window for computing rolling acceptance rate |
| `proposal_scale_nu` | 0.5 | Initial standard deviation for nu proposal |

### 2.5 Storage Variables

| Variable | Type | Purpose |
|----------|------|---------|
| `mu_samples` | list | MCMC samples for μ (dynamically grown) |
| `sigma2_samples` | list | MCMC samples for σ² (dynamically grown) |
| `nu_samples` | list | MCMC samples for ν (dynamically grown) |
| `acceptance_window` | list | Circular buffer tracking accept/reject (0/1) |

### 2.6 Post-MCMC Variables

| Variable | Type | Purpose |
|----------|------|---------|
| `iterations` | int | Total number of MCMC iterations run |
| `burn_in` | int | Number of initial iterations to discard |
| `post_mu` | float | Posterior mean estimate of μ |
| `post_sigma2` | float | Posterior mean estimate of σ² |
| `post_nu` | float | Posterior mean estimate of ν |

### 2.7 Variable Relationships

```
┌─────────────────────────────────────────────────────────────────┐
│                      MCMC Update Cycle                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   returns ─────────────────────────────────────────────────┐   │
│       │                                                      │   │
│       ▼                                                      │   │
│   ┌─────────┐    lambdas     ┌─────────────────────────┐    │   │
│   │ Sample  │◄───────────────│ Auxiliary Variables     │    │   │
│   │ lambdas │                │ λᵢ ~ Gamma((ν+1)/2, ...) │    │   │
│   └────┬────┘                └─────────────────────────┘    │   │
│        │                                                      │   │
│        │ σ², ν                                               │   │
│        ▼                                                      │   │
│   ┌─────────┐                                                │   │
│   │ Sample  │◄──── Gibbs Step (conjugate prior)              │   │
│   │   μ    │                                                 │   │
│   └────┬────┘                                                │   │
│        │                                                      │   │
│        │ λs, μ, σ²                                            │   │
│        ▼                                                      │   │
│   ┌─────────┐                                                │   │
│   │ Sample  │◄──── Gibbs Step (conjugate prior)              │   │
│   │   σ²   │                                                 │   │
│   └────┬────┘                                                │   │
│        │                                                      │   │
│        │ λs, μ, σ²                                            │   │
│        ▼                                                      │   │
│   ┌─────────┐     ┌──────────────────────────────────────┐   │   │
│   │ Sample  │◄────►│ Metropolis Step                      │───┘   │
│   │   ν    │      │ (with adaptive proposal scale)       │       │
│   └────┬────┘      └──────────────────────────────────────┘       │
│        │                                                            │
│        ▼                                                            │
│   ┌─────────────────────────────────────────────────────────┐       │
│   │ Store: mu_samples, sigma2_samples, nu_samples           │       │
│   └─────────────────────────────────────────────────────────┘       │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 3. Core Algorithm

### 3.1 Statistical Model

The log returns {r₁, r₂, ..., rₙ} are modeled as:

$$r_t \mid \mu, \sigma^2, \nu \sim t_{\nu}(\mu, \sigma^2)$$

where the t-distribution has:

- Mean: $\mu$ (location)
- Variance: $\sigma^2$ (scale)
- Degrees of freedom: $\nu$ (controls tail thickness)

### 3.2 Data Augmentation (Auxiliary Variables)

The t-distribution is represented as a scale mixture of normals:

$$r_t \mid \lambda_t, \mu, \sigma^2 \sim N\left(\mu, \frac{\sigma^2}{\lambda_t}\right)$$

$$\lambda_t \mid \nu \sim \text{Gamma}\left(\frac{\nu}{2}, \frac{\nu}{2}\right)$$

This representation enables **Gibbs sampling** for μ and σ².

### 3.3 MCMC Sampling Steps

#### Step 1: Sample λ (Auxiliary Variables)

For each observation t = 1, ..., n:

$$\lambda_t \sim \text{Gamma}\left(\frac{\nu + 1}{2}, \frac{2}{\nu + \frac{(r_t - \mu)^2}{\sigma^2}}\right)$$

**Code (Line 98):**
```python
lambdas = np.random.gamma((nu + 1) / 2, 1 / ((nu + (returns - mu) ** 2 / sigma2) / 2))
```

#### Step 2: Sample μ (Gibbs Sampler)

Conditional posterior:

$$\mu \mid \lambda, \sigma^2, r \sim N(\mu_n, \tau_n^2)$$

where:

$$\tau_n^2 = \left(\frac{1}{\tau_0^2} + \frac{\sum_{t=1}^n \lambda_t}{\sigma^2}\right)^{-1}$$

$$\mu_n = \tau_n^2 \left(\frac{\mu_0}{\tau_0^2} + \frac{\sum_{t=1}^n \lambda_t r_t}{\sigma^2}\right)$$

**Code (Lines 101-104):**
```python
prec_0 = 1 / tau2_0                          # 1/τ₀²
prec_n = prec_0 + np.sum(lambdas) / sigma2   # ∑λₜ/σ²
mu_n = (prec_0 * mu_0 + np.sum(lambdas * returns) / sigma2) / prec_n
mu = np.random.normal(mu_n, np.sqrt(1 / prec_n))
```

#### Step 3: Sample σ² (Gibbs Sampler)

Conditional posterior:

$$\sigma^2 \mid \lambda, \mu, r \sim \text{Inverse-Gamma}(\alpha_n, \beta_n)$$

where:

$$\alpha_n = \alpha_0 + \frac{n}{2}$$

$$\beta_n = \beta_0 + \frac{1}{2}\sum_{t=1}^n \lambda_t (r_t - \mu)^2$$

**Code (Lines 107-109):**
```python
a_n = a_0 + n / 2
b_n = b_0 + 0.5 * np.sum(lambdas * (returns - mu) ** 2)
sigma2 = 1.0 / np.random.gamma(a_n, 1.0 / b_n)
```

#### Step 4: Sample ν (Metropolis-Hastings)

Unlike μ and σ², ν does not have a conjugate prior. We use a **Metropolis step** with a **random walk Gaussian proposal**:

$$q(\nu' \mid \nu) = \text{Normal}(\nu, \sigma_{\text{prop}}^2)$$

**Acceptance probability:**

$$\alpha = \min\left(1, \frac{p(\nu' \mid r, \mu, \sigma^2)}{p(\nu \mid r, \mu, \sigma^2)}\right)$$

**Code (Lines 111-122):**
```python
nu_prop = np.random.normal(nu, proposal_scale_nu)  # Proposal
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
```

### 3.4 Adaptive Proposal Scaling

The proposal scale σ_prop is adaptively adjusted based on the acceptance rate:

| Acceptance Rate | Action |
|-----------------|--------|
| < 0.20 | Shrink: σ_prop ← σ_prop × 0.9 |
| > 0.30 | Expand: σ_prop ← σ_prop × 1.1 |
| [0.20, 0.30] | No change |

**Code (Lines 77-83, 125-129):**
```python
def proposal_adapted(acceptance_rate, scale):
    if acceptance_rate < 0.20:
        return scale * 0.9
    elif acceptance_rate > 0.30:
        return scale * 1.1
    return scale

# Every acceptance_window_size iterations:
acc_rate = np.mean(acceptance_window)
proposal_scale_nu = proposal_adapted(acc_rate, proposal_scale_nu)
proposal_scale_nu = np.clip(proposal_scale_nu, 0.01, 10.0)
acceptance_window = []
```

### 3.5 Convergence Detection (ESS)

Convergence is determined by **Effective Sample Size (ESS)**:

$$\text{ESS} = \frac{n}{1 + 2\sum_{k=1}^{\infty} \rho_k}$$

where ρ_k is the autocorrelation at lag k.

In practice, we approximate using lags 1 to min(50, n/2):

**Code (Lines 65-75):**
```python
def compute_ess(samples):
    n = len(samples)
    if n < 100:
        return 0
    mean = np.mean(samples)
    var = np.var(samples)
    if var == 0:
        return n
    autocorr = np.correlate(samples - mean, samples - mean, mode='full')[n - 1:] / (var * n)
    return n / (1 + 2 * np.sum(autocorr[1:min(50, n // 2)]))
```

**Convergence criteria:** ESS > target_ess (500) for all three parameters.

---

## 4. Parameters and Mathematical Derivations

### 4.1 Model Parameters

| Parameter | Symbol | Domain | Description |
|-----------|--------|--------|-------------|
| μ | mu | ℝ | Location parameter (mean) |
| σ² | sigma2 | (0, ∞) | Scale parameter (variance) |
| ν | nu | (2, ∞) | Degrees of freedom (controls tail weight) |

### 4.2 Prior Distributions

#### Prior for μ
$$p(\mu) = \text{Normal}(\mu_0, \tau_0^2)$$

With μ₀ = 0.0 and τ²₀ = 10.0, this is a weakly informative prior centered at 0.

#### Prior for σ²
$$p(\sigma^2) = \text{Inverse-Gamma}(a_0, b_0)$$

With a₀ = 0.01 and b₀ = 0.01, this is a weakly informative prior.

#### Prior for ν
$$p(\nu) = \text{Gamma}(k=2, \theta=2) = \text{Gamma}(\alpha=2, \beta=0.5)$$

where the Gamma pdf is:
$$p(\nu) = \frac{\beta^\alpha \nu^{\alpha-1} e^{-\beta\nu}}{\Gamma(\alpha)}$$

### 4.3 Full Conditional Posteriors

#### Full Conditional for μ

$$p(\mu \mid \lambda, \sigma^2, r) \propto p(\mu) \prod_{t=1}^n p(r_t \mid \mu, \sigma^2, \lambda_t)$$

Since the likelihood is:
$$\prod_{t=1}^n N\left(r_t \mid \mu, \frac{\sigma^2}{\lambda_t}\right) \propto \exp\left(-\frac{1}{2}\sum_{t=1}^n \frac{\lambda_t(r_t - \mu)^2}{\sigma^2}\right)$$

And the prior is:
$$p(\mu) \propto \exp\left(-\frac{(\mu - \mu_0)^2}{2\tau_0^2}\right)$$

Combining exponents:

$$\text{Exponent} = -\frac{(\mu - \mu_0)^2}{2\tau_0^2} - \frac{1}{2\sigma^2}\sum_{t=1}^n \lambda_t(r_t - \mu)^2$$

$$= -\frac{1}{2}\left[\frac{(\mu - \mu_0)^2}{\tau_0^2} + \frac{\sum\lambda_t(r_t - \mu)^2}{\sigma^2}\right]$$

$$= -\frac{1}{2}\left[\frac{\mu^2 - 2\mu\mu_0 + \mu_0^2}{\tau_0^2} + \frac{\sum\lambda_tr_t^2 - 2\mu\sum\lambda_tr_t + \mu^2\sum\lambda_t}{\sigma^2}\right]$$

Collecting μ terms:
$$= -\frac{1}{2}\left[\mu^2\left(\frac{1}{\tau_0^2} + \frac{\sum\lambda_t}{\sigma^2}\right) - 2\mu\left(\frac{\mu_0}{\tau_0^2} + \frac{\sum\lambda_tr_t}{\sigma^2}\right) + \text{const}\right]$$

This is proportional to a Normal distribution with:

**Posterior precision:**
$$\text{Prec}_n = \frac{1}{\tau_0^2} + \frac{\sum_{t=1}^n \lambda_t}{\sigma^2}$$

**Posterior mean:**
$$\mu_n = \text{Prec}_n^{-1} \left(\frac{\mu_0}{\tau_0^2} + \frac{\sum_{t=1}^n \lambda_t r_t}{\sigma^2}\right)$$

#### Full Conditional for σ²

$$p(\sigma^2 \mid \lambda, \mu, r) \propto p(\sigma^2) \prod_{t=1}^n N\left(r_t \mid \mu, \frac{\sigma^2}{\lambda_t}\right)$$

The likelihood part:
$$\prod_{t=1}^n \sqrt{\frac{\lambda_t}{2\pi\sigma^2}} \exp\left(-\frac{\lambda_t(r_t - \mu)^2}{2\sigma^2}\right) \propto (\sigma^2)^{-n/2} \exp\left(-\frac{1}{2\sigma^2}\sum_{t=1}^n \lambda_t(r_t - \mu)^2\right)$$

Combined with prior $p(\sigma^2) \propto (\sigma^2)^{-(\alpha_0+1)} \exp(-\beta_0/\sigma^2)$:

$$p(\sigma^2 \mid \cdot) \propto (\sigma^2)^{-(\alpha_0 + n/2 + 1)} \exp\left(-\frac{\beta_0 + \frac{1}{2}\sum\lambda_t(r_t-\mu)^2}{\sigma^2}\right)$$

This is **Inverse-Gamma**(αₙ, βₙ) with:

**Shape:**
$$\alpha_n = \alpha_0 + \frac{n}{2}$$

**Scale:**
$$\beta_n = \beta_0 + \frac{1}{2}\sum_{t=1}^n \lambda_t (r_t - \mu)^2$$

#### Full Conditional for ν

Unlike μ and σ², ν does not have a conjugate prior. The log-posterior is:

$$\log p(\nu \mid r, \mu, \sigma^2) = \log L(r \mid \mu, \sigma^2, \nu) + \log p(\nu) + C$$

**Log-likelihood:**

The t-distribution likelihood is:
$$L = \prod_{t=1}^n \frac{\Gamma\left(\frac{\nu+1}{2}\right)}{\sqrt{\nu\pi\sigma^2}\,\Gamma\left(\frac{\nu}{2}\right)} \left(1 + \frac{(r_t - \mu)^2}{\nu\sigma^2}\right)^{-\frac{\nu+1}{2}}$$

Log-likelihood:
$$\log L = \sum_{t=1}^n \left[\log\Gamma\left(\frac{\nu+1}{2}\right) - \log\Gamma\left(\frac{\nu}{2}\right) - \frac{1}{2}\log(\nu\pi\sigma^2) - \frac{\nu+1}{2}\log\left(1 + \frac{(r_t-\mu)^2}{\nu\sigma^2}\right)\right]$$

**Code (Lines 41-50):**
```python
def log_likelihood_nu(nu, returns, mu, sigma2):
    n = len(returns)
    return np.sum(
        math.lgamma((nu + 1) / 2) - math.lgamma(nu / 2)
        - 0.5 * np.log(nu * np.pi * sigma2)
        - ((nu + 1) / 2) * np.log(1 + (returns - mu) ** 2 / (nu * sigma2))
    )
```

**Log-prior (Gamma(2, scale=2)):**

$$p(\nu) = \frac{1}{2^2 \Gamma(2)} \nu^{2-1} e^{-\nu/2} = \frac{\nu e^{-\nu/2}}{4}$$

$$\log p(\nu) = \log\nu - \frac{\nu}{2} - \log 4$$

**Code (Lines 52-56):**
```python
def log_prior_nu(nu):
    return (2 - 1) * np.log(nu) - nu / 2 - (math.log(math.factorial(1)) + 2 * np.log(2))
```

### 4.4 Metropolis-Hastings Acceptance Ratio

For the random walk proposal $q(\nu' \mid \nu) = \text{Normal}(\nu, \sigma_{\text{prop}}^2)$:

Since the proposal is **symmetric**: $q(\nu \mid \nu') = q(\nu' \mid \nu)$

The acceptance ratio simplifies to the **ratio of posteriors**:

$$\alpha = \min\left(1, \frac{p(\nu' \mid r, \mu, \sigma^2)}{p(\nu \mid r, \mu, \sigma^2)}\right)$$

Taking logs for numerical stability:

$$\log \alpha = \log p(\nu' \mid \cdot) - \log p(\nu \mid \cdot)$$

**Code (Line 117):**
```python
log_alpha = log_posterior_nu(nu_prop, returns, mu, sigma2) - log_posterior_nu(nu, returns, mu, sigma2)
```

### 4.5 VaR and ES Calculation

Using the posterior parameter estimates, we simulate from the t-distribution:

The t-distribution with ν > 2 has variance:
$$\text{Var}(X) = \sigma^2 \cdot \frac{\nu}{\nu - 2}$$

So the equivalent scale for a Normal approximation is:
$$\sigma_{\text{approx}} = \sigma \cdot \sqrt{\frac{\nu}{\nu - 2}}$$

**Code (Lines 167-169):**
```python
posterior_samples = np.random.normal(post_mu, np.sqrt(post_sigma2) * np.sqrt(post_nu/(post_nu-2)), 10000)
var = -np.percentile(posterior_samples, alpha * 100)
es = -np.mean(posterior_samples[posterior_samples < -var])
```

**VaR (Value-at-Risk):**
$$\text{VaR}_\alpha = -\inf\{x: F_X(x) \geq \alpha\}$$

At α = 0.05, VaR is the 5th percentile (made positive by the negative sign).

**Expected Shortfall (ES):**
$$\text{ES}_\alpha = -\frac{1}{\alpha}\int_0^\alpha F_X^{-1}(u)\,du = \mathbb{E}[X \mid X < -\text{VaR}_\alpha]$$

---

## 5. Summary of Mathematical Relationships

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         PRIOR → POSTERIOR                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  μ:  Normal(μ₀, τ²₀)        →    Normal(μₙ, τ²ₙ)                        │
│                                  τ²ₙ = (1/τ²₀ + Σλₜ/σ²)⁻¹              │
│                                  μₙ   = τ²ₙ(μ₀/τ²₀ + Σλₜrₜ/σ²)          │
│                                                                         │
│  σ²: Inverse-Gamma(a₀,b₀)  →    Inverse-Gamma(aₙ, bₙ)                  │
│                                  aₙ = a₀ + n/2                          │
│                                  bₙ = b₀ + ½Σλₜ(rₜ-μ)²                  │
│                                                                         │
│  ν:  Gamma(2, scale=2)      →    Metropolis-Hastings                     │
│                                  α = min(1, p(ν'│·) / p(ν│·))          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Key References

1. **Metropolis-within-Gibbs**: A two-stage MCMC scheme where some parameters are sampled via Gibbs (conjugate) and others via Metropolis-Hastings.

2. **Optimal Acceptance Rate**: For random walk Metropolis with Gaussian proposals, the optimal acceptance rate is approximately **0.234** (Gelman et al., 1996).

3. **Effective Sample Size (ESS)**: Measures how many independent samples the autocorrelated chain is worth. Accounts for chain autocorrelation.

4. **Scale Mixture Representation**: The t-distribution as a mixture of normals with Gamma-distributed precision enables efficient Gibbs sampling.
