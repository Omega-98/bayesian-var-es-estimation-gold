# Bayesian VAR-ES Estimation: Code Architecture, Mathematical Principles and Workflow

## 1. Code Architecture

### 1.1 Overall Structure

The code follows a standard Bayesian MCMC pipeline:

```
Data Loading → MCMC Sampling → Burn-in → Thinning → Diagnostics → Visualization
```

### 1.2 Module Organization

| Component | Description |
|-----------|-------------|
| **Data Loading** | Load gold futures CSV, compute log returns |
| **MCMC Sampler** | Gibbs sampler with Metropolis-Hastings for nu |
| **Burn-in Handler** | Discards initial samples before convergence |
| **Thinning Handler** | Keeps every k-th sample to reduce autocorrelation |
| **Autocorrelation Analyzer** | Computes ACF to diagnose sample dependence |
| **Trace Plotter** | Visualizes sample trajectories |
| **Histogram Plotter** | Visualizes marginal posterior distributions |

### 1.3 Key Variables

| Variable | Meaning |
|----------|---------|
| `returns` | Log returns of gold futures prices |
| `n` | Number of observations |
| `mu` | Location parameter of t-distribution |
| `sigma2` | Scale parameter (variance) |
| `nu` | Degrees of freedom of t-distribution |
| `lambdas` | Auxiliary variables (inverse variances) for Gibbs sampling |
| `iterations` | Total MCMC iterations |
| `burn_in` | Number of initial iterations to discard |
| `thin_factor` | Keep every k-th sample |

---

## 2. Mathematical Principles

### 2.1 Model Specification

We model log returns as following a Student's t-distribution with unknown location and scale:

$$r_t \sim t_\nu(\mu, \sigma^2)$$

The t-distribution PDF is:

$$f(r | \mu, \sigma^2, \nu) = \frac{\Gamma\left(\frac{\nu+1}{2}\right)}{\Gamma\left(\frac{\nu}{2}\right)\sqrt{\nu\pi\sigma^2}} \left(1 + \frac{(r-\mu)^2}{\nu\sigma^2}\right)^{-\frac{\nu+1}{2}}$$

### 2.2 Mixture Representation (Gaussian Mixture)

The t-distribution can be expressed as a scale mixture of normals:

$$r_t | \lambda_t \sim N\left(\mu, \frac{\sigma^2}{\lambda_t}\right), \quad \lambda_t \sim \text{Gamma}\left(\frac{\nu}{2}, \frac{\nu}{2}\right)$$

This representation enables **Gibbs sampling** by introducing auxiliary variables $\lambda_t$.

**Derivation:**

The marginal distribution of $r_t$ is:

$$f(r) = \int_0^\infty f_{N}\left(r | \mu, \frac{\sigma^2}{\lambda}\right) f_{\text{Gamma}}(\lambda | \frac{\nu}{2}, \frac{\nu}{2}) d\lambda$$

Substituting the forms:

$$f(r) = \int_0^\infty \sqrt{\frac{\lambda}{2\pi\sigma^2}} \exp\left(-\frac{\lambda(r-\mu)^2}{2\sigma^2}\right) \cdot \frac{(\nu/2)^{\nu/2}}{\Gamma(\nu/2)} \lambda^{\nu/2-1} \exp\left(-\frac{\nu\lambda}{2}\right) d\lambda$$

$$= \frac{1}{\sqrt{2\pi\sigma^2}} \frac{(\nu/2)^{\nu/2}}{\Gamma(\nu/2)} \int_0^\infty \lambda^{\frac{\nu+1}{2}-1} \exp\left(-\frac{\lambda}{2}\left(\frac{(r-\mu)^2}{\sigma^2} + \nu\right)\right) d\lambda$$

Using the Gamma integral $\int_0^\infty x^{a-1}e^{-bx}dx = \Gamma(a)/b^a$:

$$f(r) = \frac{1}{\sqrt{2\pi\sigma^2}} \frac{(\nu/2)^{\nu/2}}{\Gamma(\nu/2)} \cdot \frac{\Gamma\left(\frac{\nu+1}{2}\right)}{\left(\frac{1}{2}\left(\frac{(r-\mu)^2}{\sigma^2}+\nu\right)\right)^{\frac{\nu+1}{2}}}$$

$$= \frac{\Gamma\left(\frac{\nu+1}{2}\right)}{\Gamma\left(\frac{\nu}{2}\right)\sqrt{\nu\pi\sigma^2}} \left(1 + \frac{(r-\mu)^2}{\nu\sigma^2}\right)^{-\frac{\nu+1}{2}}$$

which is exactly the t-distribution PDF.

### 2.3 Prior Distributions

We adopt conjugate priors for computational convenience:

| Parameter | Prior | Hyperparameters |
|-----------|-------|-----------------|
| $\mu$ | Normal | $\mu_0 = 0$, $\tau_0^2 = 10$ |
| $\sigma^2$ | Inverse Gamma$(a_0, b_0)$ | $a_0 = 0.01$, $b_0 = 0.01$ |
| $\nu$ | Gamma$(2, \text{scale}=2)$ | shape=2, scale=2 |

### 2.4 Full Conditional Distributions

Gibbs sampling requires the **full conditional** distributions for each parameter.

#### 2.4.1 Full Conditional for $\lambda_t$

$$\lambda_t | r_t, \mu, \sigma^2, \nu \sim \text{Gamma}\left(\frac{\nu+1}{2}, \frac{\nu + (r_t - \mu)^2/\sigma^2}{2}\right)$$

**Derivation:**

From the mixture representation, the conditional posterior of $\lambda_t$ is proportional to:

$$f(\lambda_t | r_t) \propto f_N(r_t | \mu, \sigma^2/\lambda_t) \cdot f_{\text{Gamma}}(\lambda_t | \nu/2, \nu/2)$$

$$\propto \sqrt{\lambda_t} \exp\left(-\frac{\lambda_t(r_t-\mu)^2}{2\sigma^2}\right) \cdot \lambda_t^{\nu/2-1} \exp\left(-\frac{\nu\lambda_t}{2}\right)$$

$$= \lambda_t^{\frac{\nu+1}{2}-1} \exp\left(-\frac{\lambda_t}{2}\left(\frac{(r_t-\mu)^2}{\sigma^2} + \nu\right)\right)$$

This is the kernel of $\text{Gamma}\left(\frac{\nu+1}{2}, \frac{\nu + (r_t-\mu)^2/\sigma^2}{2}\right)$.

#### 2.4.2 Full Conditional for $\mu$

$$\mu | \mathbf{r}, \boldsymbol{\lambda}, \sigma^2 \sim N(\mu_n, \tau_n^2)$$

where:

$$\tau_n^{-2} = \tau_0^{-2} + \frac{\sum_{t=1}^n \lambda_t}{\sigma^2}$$

$$\mu_n = \frac{\tau_0^{-2}\mu_0 + \frac{\sum_{t=1}^n \lambda_t r_t}{\sigma^2}}{\tau_n^{-2}}$$

**Derivation:**

The conditional posterior:

$$f(\mu | \text{data}) \propto f_N(\mu | \mu_0, \tau_0^2) \cdot \prod_{t=1}^n f_N(r_t | \mu, \sigma^2/\lambda_t)$$

$$\propto \exp\left(-\frac{(\mu-\mu_0)^2}{2\tau_0^2}\right) \cdot \exp\left(-\frac{1}{2\sigma^2}\sum_{t=1}^n \lambda_t(r_t-\mu)^2\right)$$

Expanding the second term:

$$\sum_{t=1}^n \lambda_t(r_t-\mu)^2 = \sum_{t=1}^n \lambda_t r_t^2 - 2\mu\sum_{t=1}^n \lambda_t r_t + \mu^2\sum_{t=1}^n \lambda_t$$

So the exponent becomes:

$$-\frac{(\mu-\mu_0)^2}{2\tau_0^2} - \frac{1}{2\sigma^2}\left(\sum \lambda_t r_t^2 - 2\mu\sum \lambda_t r_t + \mu^2\sum \lambda_t\right)$$

Collecting terms in $\mu^2$, $\mu$, and constants:

$$\text{Coefficient of } \mu^2: -\frac{1}{2\tau_0^2} - \frac{\sum \lambda_t}{2\sigma^2} = -\frac{\tau_n^{-2}}{2}$$

$$\text{Coefficient of } \mu: \frac{\mu_0}{\tau_0^2} + \frac{\sum \lambda_t r_t}{\sigma^2}$$

Completing the square gives the Normal distribution with mean $\mu_n$ and variance $\tau_n^2$ as stated.

#### 2.4.3 Full Conditional for $\sigma^2$

$$\sigma^2 | \mathbf{r}, \boldsymbol{\lambda}, \mu \sim \text{Inverse-Gamma}(a_n, b_n)$$

where:

$$a_n = a_0 + \frac{n}{2}$$

$$b_n = b_0 + \frac{1}{2}\sum_{t=1}^n \lambda_t(r_t - \mu)^2$$

**Derivation:**

$$f(\sigma^2 | \text{data}) \propto f_{\text{IG}}(\sigma^2 | a_0, b_0) \cdot \prod_{t=1}^n f_N(r_t | \mu, \sigma^2/\lambda_t)$$

$$\propto (\sigma^2)^{-a_0-1}\exp\left(-\frac{b_0}{\sigma^2}\right) \cdot \prod_{t=1}^n \sqrt{\frac{\lambda_t}{2\pi\sigma^2}} \exp\left(-\frac{\lambda_t(r_t-\mu)^2}{2\sigma^2}\right)$$

$$= (\sigma^2)^{-a_0-n/2-1} \exp\left(-\frac{1}{\sigma^2}\left(b_0 + \frac{1}{2}\sum_{t=1}^n \lambda_t(r_t-\mu)^2\right)\right)$$

This is the kernel of $\text{IG}(a_n, b_n)$.

#### 2.4.4 Posterior for $\nu$ (Metropolis-Hastings)

Since there is no conjugate prior for $\nu$, we use **Metropolis-Hastings** with a Normal proposal:

$$\nu^* | \nu^{(t)} \sim N(\nu^{(t)}, \ell^2)$$

where $\ell = 0.5$ is the proposal standard deviation.

The acceptance ratio is:

$$\alpha = \min\left\{1, \frac{f(\nu^* | \text{data})}{f(\nu^{(t)} | \text{data})}\right\}$$

The log-posterior for $\nu$:

$$\log f(\nu | \text{data}) = \log f(\nu) + \sum_{t=1}^n \log f(r_t | \lambda_t, \mu, \sigma^2, \nu)$$

$$= \left[(2-1)\log\nu - \frac{\nu}{2} - \log\Gamma(2) - 2\log 2\right] - \frac{\nu+1}{2}\sum_{t=1}^n \log\left(1 + \frac{(r_t-\mu)^2}{\nu\sigma^2}\right)$$

Note: Stirling's approximation is used for $\log\Gamma(2) = \log(1!) = 0$.

### 2.5 Value-at-Risk (VaR) and Expected Shortfall (ES)

Once posterior samples are obtained, VaR and ES at level $\alpha$ are computed via simulation:

1. For each posterior sample $(\mu^{(s)}, \sigma^{2(s)}, \nu^{(s)})$, draw $M$ samples from $t_{\nu^{(s)}}(\mu^{(s)}, \sigma^{2(s)})$
2. Compute the empirical $\alpha$-quantile across all simulated returns to get VaR
3. Compute the mean of returns exceeding VaR to get ES

---

## 3. Operation Workflow

### 3.1 Flowchart

```
┌─────────────────────────────────────────────────────────────────┐
│                    1. DATA PREPARATION                          │
│  • Load CSV → Clean prices → Compute log returns               │
│  • Result: returns array of length n                            │
└─────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│                    2. MCMC SAMPLING                             │
│  for iteration i = 1, ..., N:                                  │
│    (a) Sample λ_t ~ Gamma((ν+1)/2, ...)  [all t simultaneously]│
│    (b) Sample μ ~ N(μ_n, τ_n²)                                  │
│    (c) Sample σ² ~ IG(a_n, b_n)                                │
│    (d) Sample ν via Metropolis-Hastings                         │
│    (e) Store (μ, σ², ν)                                         │
└─────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│                    3. BURN-IN                                   │
│  • Discard first burn_in samples                                │
│  • Rationale: samples before chain converges                    │
└─────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│                    4. THINNING                                 │
│  • Keep every thin_factor-th sample                            │
│  • Rationale: reduce autocorrelation between samples           │
└─────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│                    5. DIAGNOSTICS                               │
│  (a) Autocorrelation Function (ACF) Plot                       │
│      - Compare before/after thinning                           │
│      - ACF should decay quickly (< 20 lags) for iid samples    │
│  (b) Trace Plot                                                 │
│      - Should show "fuzzy caterpillar" pattern                 │
│      - No trends, no jumps                                     │
│  (c) Effective Sample Size (ESS)                                │
│      - ESS = n / (1 + 2∑ρ_k)                                   │
│      - Higher ESS indicates better mixing                      │
└─────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│                    6. POSTERIOR SUMMARY                        │
│  • Histogram of marginal posteriors                            │
│  • Point estimates: mean, median                               │
│  • Credible intervals: 2.5%, 97.5% percentiles                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Burn-in Rationale

During initial MCMC iterations, the chain has not yet reached the target distribution. Discarding burn-in samples prevents biased posterior estimates.

### 3.3 Thinning Rationale

Thinning keeps every $k$-th sample to reduce **autocorrelation**. Highly autocorrelated samples inflate variance of Monte Carlo estimates. If $\rho_k$ (autocorrelation at lag $k$) remains high, the effective number of independent samples is much smaller than the nominal number.

**Decision Rules:**
- If ACF still shows significant correlation after thinning, increase `thin_factor`
- ESS should be at least 1000 for reliable estimates

### 3.4 Diagnostic Interpretation

| Diagnostic | Good Sign | Bad Sign |
|------------|-----------|----------|
| ACF | Drops to near zero within ~20 lags | Significant correlations persist beyond 50 lags |
| Trace Plot | "Fuzzy caterpillar", constant variance | Trends, cycles, or jumps |
| ESS | High (>1000) | Low (<500) |

---

## 4. Parameter Settings

| Parameter | Value | Rationale |
|-----------|-------|----------|
| `iterations` | 20,000 | Balance between precision and computation time |
| `burn_in` | 5,000 | 25% of total, typical for converging chains |
| `thin_factor` | 10 | Reduces autocorrelation while retaining enough samples |

---

## 5. Output Files

| File | Content |
|------|---------|
| `autocorrelation_plots.png` | ACF before and after thinning for all parameters |
| `trace_plots.png` | Sample trajectories (thinned) for all parameters |
| `posterior_histograms.png` | Marginal posterior distributions |
| Console output | Summary statistics and ESS |
