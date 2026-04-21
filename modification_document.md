# Code Modification Document

## File: analysisAndPlotting.py

## Modification Requirements

### 1. Add Distribution Fitting Lines to Histograms
- Add a fitted distribution line to all histogram plots
- The line color must be different from existing lines (red for mean, green for median, steelblue for histogram)
- Use kernel density estimation (KDE) for fitting

### 2. Add Iteration Suffix to All Export File Names
- All exported plot files should have "_{iterations}" suffix before the file extension
- Example: `autocorrelation_plots.png` → `autocorrelation_plots_50000000.png`

### 3. Minimal Changes Principle
- Make the smallest changes possible to achieve the requirements
- Maintain code style consistency and readability
- Preserve existing functionality

### 4. Complete English Comments
- All new comments must be written in English
- Comments should be clear and descriptive

### 5. Increase Histogram Precision
- Increase the number of bins in histogram plots for higher precision
- Change from current 50 bins to a higher value (e.g., 100 bins)

### 6. Calculate VaR and ES per Iteration
- During each MCMC iteration, calculate the VaR (Value at Risk) and ES (Expected Shortfall)
- Use the parameters (mu, sigma2, nu) from the current iteration
- VaR and ES calculation based on Student's t-distribution

### 7. Plot VaR and ES Distribution Histograms
- Plot histogram for VaR samples (similar to mu, sigma2, nu histograms)
- Plot histogram for ES samples
- These should be saved separately in the plots folder

### 8. Separate Plot Files with Naming Convention
- Create a `plots` folder in the directory
- Save each parameter's three plots (autocorrelation, trace, histogram) separately
- Save VaR histogram and ES histogram separately
- Naming convention in plots folder:
  - `mu_autocorrelation_{iterations}.png`
  - `mu_trace_{iterations}.png`
  - `mu_histogram_{iterations}.png`
  - `sigma2_autocorrelation_{iterations}.png`
  - `sigma2_trace_{iterations}.png`
  - `sigma2_histogram_{iterations}.png`
  - `nu_autocorrelation_{iterations}.png`
  - `nu_trace_{iterations}.png`
  - `nu_histogram_{iterations}.png`
  - `VaR_histogram_{iterations}.png`
  - `ES_histogram_{iterations}.png`

## Implementation Notes

### VaR and ES Calculation
For a Student's t-distribution with parameters (mu, sigma2, nu):
- Scale parameter: sigma = sqrt(sigma2)
- VaR at confidence level (e.g., 95%): quantile of t-distribution
- ES: mean of the distribution beyond VaR threshold

The returns follow a Student's t-distribution:
Return ~ t(nu, mu, sigma2)

### Distribution Fitting
- Use scipy.stats.gaussian_kde for kernel density estimation
- Plot the KDE curve on the histogram

## Summary of Changes Needed
1. Add scipy import for KDE and t-distribution
2. Add VaR and ES calculation in MCMC loop
3. Store VaR and ES samples
4. Modify histogram function to accept fitted line parameter
5. Add KDE fitting to histogram function
6. Create plots folder before saving
7. Update all file saving to include iteration suffix and plots folder path
8. Increase histogram bins from 50 to 100