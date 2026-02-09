# Comprehensive Uncertainty Analysis: Bootstrap Probit Dawid-Skene

## Executive Summary

This document provides a systematic analysis of all sources of uncertainty in estimating ML expert headcounts from crowdsourced annotations, and describes how the bootstrap probit Dawid-Skene algorithm addresses each source. The algorithm targets **population-level inference** under the assumption that observed employees are a representative sample from each company's broader population.

## Inferential Goal and Estimand

**Target Quantity**: Number of ML experts in the broader population at each company

**Observed Data**:
- Test set: Sample of \(n_c\) employees from each company \(c\) with crowdsourced annotations
- Validation set: Labeled data with ground truth expertise labels
- Assumption: Test employees are a representative random sample from their company's population

**Target Distribution**: Posterior predictive distribution \(P(\text{Count}_c = k | \mathcal{D})\), giving probabilities for discrete counts \(k = 0, 1, 2, \ldots\)

## Complete Taxonomy of Uncertainty

### 1. Parameter Estimation Uncertainty

**Nature**: Epistemic (knowledge-based) uncertainty about model parameters

**Source**: We estimate annotator confusion matrices \(\theta_j\) and correlation structures \(\Sigma_0, \Sigma_1\) from finite validation data

**Mathematical Form**:
- True parameters: \((\boldsymbol{\theta}_{\text{true}}, \boldsymbol{\Sigma}_{0,\text{true}}, \boldsymbol{\Sigma}_{1,\text{true}})\)
- Estimates: \((\hat{\boldsymbol{\theta}}, \hat{\boldsymbol{\Sigma}}_0, \hat{\boldsymbol{\Sigma}}_1)\)
- Uncertainty: Posterior distribution \(p(\boldsymbol{\theta}, \boldsymbol{\Sigma}_0, \boldsymbol{\Sigma}_1 | \mathcal{D}_{\text{val}})\)

**Properties**:
- Reducible: More validation data → tighter posterior
- Propagates: Affects all downstream predictions
- Magnitude: Depends on validation sample size and label quality

**Algorithm Approach**: Validation data bootstrap
```
For each iteration b:
    Resample validation data with replacement (stratified by class)
    Re-estimate (θ^(b), Σ₀^(b), Σ₁^(b)) from bootstrap sample
```

**Why This Works**: The bootstrap distribution of parameter estimates approximates the posterior distribution under the empirical Bayes framework. By resampling validation data, we simulate the sampling variability in our parameter estimates.

### 2. Base Rate (Prior) Uncertainty

**Nature**: Epistemic uncertainty about population prevalence

**Source**: We don't know the true proportion \(\pi = P(z=1)\) of ML experts in the company populations

**Mathematical Form**:
- Unknown parameter: \(\pi \in [0, 1]\)
- Hyperprior: \(\pi \sim \text{Beta}(2, 10)\)

**Properties**:
- Not reducible from test data alone (no labels)
- Could be reduced with external data (industry surveys, etc.)
- Impacts all predictions through Bayesian prior

**Algorithm Approach**: Prior sampling
```
For each iteration b:
    Sample π^(b) ~ Beta(2, 10)
    Use π^(b) as class prior in Bayesian inference
```

**Why This Works**: By treating the prior as uncertain and sampling from a hyperprior, we propagate uncertainty about the base rate through to final predictions. The Beta(2,10) distribution:
- Mean: 1/6 ≈ 0.167 (belief that ML experts are relatively rare)
- Allows substantial variation (roughly 0.05 to 0.35 at 90% credibility)

**Sensitivity**: Results should ideally be checked under alternative hyperpriors

### 3. Sampling Uncertainty

**Nature**: Epistemic uncertainty about which individuals were sampled

**Source**: The observed employees at each company are a sample from a larger population; different samples would yield different counts

**Mathematical Form**:
- Population: All employees at company \(c\) (or some defined subpopulation)
- Sample: Observed \(n_c\) employees \(S_c \subset \text{Population}_c\)
- Uncertainty: Alternative samples \(S_c'\) would have different characteristics

**Properties**:
- Reducible: Larger samples → less uncertainty
- Company-specific: Affects each company separately
- Magnitude: Depends on sample size and population heterogeneity

**Algorithm Approach**: Test data bootstrap (within companies)
```
For each iteration b:
    For each company c:
        Resample nc employees with replacement from observed sample
        This simulates alternative samples from the population
```

**Why This Works**: 

**Assumption**: If the original sample is representative, bootstrap resamples approximate alternative representative samples from the population.

**Mechanism**: 
- Employee #42 appearing 0 times ≈ "didn't sample someone like #42"
- Employee #42 appearing 2 times ≈ "sampled two employees with similar characteristics to #42"

**Limitation**: Only valid if original sample is representative. If LinkedIn visibility or search ranking introduces bias, bootstrap won't correct for this (would need additional modeling).

### 4. Latent Label (Realization) Uncertainty

**Nature**: Aleatoric (irreducible) uncertainty about true expertise status

**Source**: For each specific employee, even given perfect knowledge of parameters and annotations, their true expertise status \(z_i\) is uncertain

**Mathematical Form**:
- For employee \(i\) with annotations \(y_i\):
- Posterior: \(P(z_i = 1 | y_i, \theta, \Sigma, \pi) = p_i\)
- True state: \(z_i \sim \text{Bernoulli}(p_i)\)

**Properties**:
- Irreducible: Fundamental uncertainty given the annotation quality
- Individual-level: Different employees have different uncertainty
- Aggregates: For \(n\) employees with probabilities \(\{p_i\}\), variance is \(\sum_i p_i(1-p_i)\)

**Algorithm Approach**: Bernoulli sampling
```
For each iteration b and each employee i:
    Compute p_i^(b) = P(z_i=1 | y_i, θ^(b), Σ^(b), π^(b))
    Sample z_i^(b) ~ Bernoulli(p_i^(b))
    
Company count: Count_c^(b) = sum of z_i^(b) for employees in company c
```

**Why This Is Essential**:

Without Bernoulli sampling (summing probabilities instead):
- Output: Distribution of \(\mathbb{E}[\text{Count}_c | \theta, \pi, S]\)
- Meaning: Uncertainty about expected count
- Variance: Only epistemic components (1, 2, 3)

With Bernoulli sampling:
- Output: Distribution of \(\text{Count}_c\) (actual discrete counts)
- Meaning: Posterior predictive distribution
- Variance: Epistemic + aleatoric = total uncertainty

**Quantitative Impact**: For a company with \(n\) employees each having \(p \approx 0.5\):
- Missing variance without Bernoulli: \(\sum p(1-p) \approx n/4\)
- Missing standard deviation: \(\sqrt{n}/2\)
- Example: \(n=100 \Rightarrow\) missing SD ≈ 5 experts

This is **not negligible**, especially for larger companies.

## Variance Decomposition

The total variance decomposes as:

$$\text{Var}[\text{Count}_c | \mathcal{D}] = \mathbb{E}_{\theta, \pi, S}[\text{Var}[\text{Count} | \theta, \pi, S]] + \text{Var}_{\theta, \pi, S}[\mathbb{E}[\text{Count} | \theta, \pi, S]]$$

**Inner variance** (Realization uncertainty):
$$\text{Var}[\text{Count} | \theta, \pi, S] = \sum_{i \in S} p_i(1-p_i)$$
- Given parameters and employees, counts vary due to \(z_i\) randomness
- Captured by: Bernoulli sampling (Component 4)

**Outer variance** (Epistemic uncertainty):
$$\text{Var}_{\theta, \pi, S}[\mathbb{E}[\text{Count} | \theta, \pi, S]]$$
- Expected count varies across parameters and samples
- Captured by: Bootstrap resampling (Components 1, 2, 3)

## Bootstrap Algorithm: Complete Integration

The algorithm integrates all four sources through a single bootstrap loop:

```
For b = 1 to B:
    # Component 1: Parameter uncertainty
    Resample validation data (stratified by class) → D_val^(b)
    Estimate (θ^(b), Σ₀^(b), Σ₁^(b)) from D_val^(b)
    
    # Component 2: Prior uncertainty  
    Sample π^(b) ~ Beta(2, 10)
    
    # Component 3: Sampling uncertainty
    For each company c:
        Resample nc employees with replacement → S_c^(b)
    
    # Component 4: Latent label uncertainty
    For each employee i in S_c^(b):
        Compute p_i^(b) = P(z_i=1 | y_i, θ^(b), Σ^(b), π^(b))
        Sample z_i^(b) ~ Bernoulli(p_i^(b))
    
    # Aggregate
    Count_c^(b) = sum of z_i^(b) for employees in company c
```

The bootstrap distribution \(\{\text{Count}_c^{(1)}, \ldots, \text{Count}_c^{(B)}\}\) is an empirical approximation to the full posterior predictive distribution.

## Validation of Uncertainty Coverage

To verify that all sources are properly accounted for:

**Component 1 (Parameters)**: 
- ✅ Validation bootstrap creates variation in \(\theta\), \(\Sigma\)
- ✅ Different parameters → different \(p_i\) values

**Component 2 (Prior)**:
- ✅ Prior sampling creates variation in \(\pi\)
- ✅ Different priors → different \(p_i\) values via Bayes' rule

**Component 3 (Sampling)**:
- ✅ Test bootstrap creates variation in which employees appear
- ✅ Different samples → different employees → different company counts

**Component 4 (Realization)**:
- ✅ Bernoulli sampling creates variation in \(z_i\) given \(p_i\)
- ✅ Same employees, same parameters → still variable counts

**No double-counting**: 
- Components 1-2 affect \(p_i\) values
- Component 3 affects which \(i\)'s contribute
- Component 4 affects \(z_i\) realizations for given \(p_i\)
- These are mathematically distinct and properly combined

**No missing sources**:
- All randomness in the data-generating process is accounted for
- Conditional independence structure is respected

## Interpreting the Output

The bootstrap distribution gives:

**Point Estimate**: 
$$\hat{\text{Count}}_c = \frac{1}{B} \sum_{b=1}^B \text{Count}_c^{(b)}$$
- Posterior predictive mean

**Uncertainty Quantification**:
- Standard deviation: \(\sqrt{\frac{1}{B-1} \sum_b (\text{Count}_c^{(b)} - \hat{\text{Count}}_c)^2}\)
- Credible intervals: 10th and 90th percentiles (or any other quantiles)
- Full distribution: Histogram of bootstrap values

**Properties**:
- Accounts for all four uncertainty sources
- Asymptotically valid as \(B \to \infty\), \(n_{\text{val}} \to \infty\)
- Discrete support (integer counts) reflecting reality

## Limitations and Assumptions

### Critical Assumption: Representative Sample
The test data bootstrap (Component 3) assumes observed employees are a representative sample. Violations:
- **LinkedIn visibility bias**: Not all employees have profiles
- **Search ranking bias**: Top results may over-represent certain characteristics
- **Self-selection**: Profile completeness may correlate with expertise

**Impact**: If systematic, creates bias in point estimates. Bootstrap captures sampling variance but not bias from non-representative sampling.

**Potential Fix**: Would require modeling the selection mechanism and/or post-stratification adjustment.

### Other Assumptions
1. **Probit model specification**: Assumes latent variables are multivariate normal
2. **Validation data quality**: Assumes ground truth labels in validation set are correct
3. **Annotation independence across items**: Assumes annotations on different employees are independent
4. **Hyperprior choice**: Beta(2,10) is somewhat arbitrary; sensitivity analysis recommended

### What Is Not Captured
- **Temporal changes**: Employee expertise may change over time
- **Annotation context effects**: Same annotator may behave differently across batches
- **Extreme events**: Model assumes smooth distributions; may underestimate tail risks

## Conclusion

The bootstrap probit Dawid-Skene algorithm provides a comprehensive framework for uncertainty quantification in crowdsourced company headcount estimation. By systematically addressing four distinct sources of uncertainty through validation bootstrap, prior sampling, test data bootstrap, and Bernoulli sampling, it produces posterior predictive distributions that honestly reflect the limits of what can be inferred from noisy, sampled data. The framework is statistically principled, computationally tractable, and produces interpretable discrete count distributions suitable for decision-making under uncertainty.

