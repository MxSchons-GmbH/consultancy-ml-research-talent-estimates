# Mathematical Framework: Bootstrap Probit Dawid-Skene Inference

## Problem Setup

Consider a crowdsourced annotation task where we have:
- **Test items**: \(N\) employees sampled from companies whose ML expertise status is unknown
- **Annotators**: \(J\) independent labelers providing binary labels \(y_{ij} \in \{0, 1\}\)
- **Company structure**: Employees are grouped into companies; we want to estimate the count of ML experts per company in the broader population
- **Validation data**: A smaller set of items with known ground truth labels \(z_i \in \{0, 1\}\)

The challenge: annotators make errors and their errors may be correlated, so we cannot simply take majority votes or average their labels. Additionally, the observed employees are a sample from a larger company population.

## The Dawid-Skene Model

The classical Dawid-Skene model (1979) characterizes each annotator by a **confusion matrix**:

$$\theta_j = \begin{bmatrix} P(y_{ij}=0 | z_i=0) & P(y_{ij}=1 | z_i=0) \\ P(y_{ij}=0 | z_i=1) & P(y_{ij}=1 | z_i=1) \end{bmatrix}$$

Under the standard Dawid-Skene assumption, **annotators are conditionally independent** given the true label:

$$P(\mathbf{y}_i | z_i) = \prod_{j=1}^J P(y_{ij} | z_i; \theta_j)$$

This independence assumption is often violated in practice—annotators may share similar biases, training, or confusion patterns.

## Probit Extension with Correlated Errors

The algorithm implements a **latent variable probit model** that allows for annotator correlations. For each item \(i\) and annotator \(j\), we introduce a latent continuous variable:

$$y_{ij}^* \sim \mathcal{N}(\mu_j(z_i), \Sigma_{jk}(z_i))$$

The observed binary label is determined by thresholding:

$$y_{ij} = \mathbb{1}[y_{ij}^* > 0]$$

**Key innovation**: The latent scores follow a **multivariate normal distribution**:

$$\mathbf{y}_i^* | z_i \sim \mathcal{N}(\boldsymbol{\mu}(z_i), \boldsymbol{\Sigma}(z_i))$$

where:
- \(\boldsymbol{\mu}(z_i) = (\Phi^{-1}(P(y_{i1}=1|z_i)), \ldots, \Phi^{-1}(P(y_{iJ}=1|z_i)))^\top\)
- \(\boldsymbol{\Sigma}(z_i)\) is a positive-definite covariance matrix capturing annotator correlations
- \(\Phi^{-1}\) is the probit link function (inverse standard normal CDF)

Importantly, we maintain **separate distributions for each true label**:
- \(\mathcal{N}(\boldsymbol{\mu}_0, \boldsymbol{\Sigma}_0)\) when \(z_i = 0\)
- \(\mathcal{N}(\boldsymbol{\mu}_1, \boldsymbol{\Sigma}_1)\) when \(z_i = 1\)

This allows annotators to have different correlation structures when labeling positive vs. negative instances.

## Bayesian Inference for New Items

Given a new set of annotations \(\mathbf{y}_{\text{new}}\), we want to compute the posterior probability:

$$P(z_{\text{new}} = 1 | \mathbf{y}_{\text{new}})$$

By Bayes' theorem:

$$P(z = 1 | \mathbf{y}) = \frac{P(\mathbf{y} | z=1) \cdot P(z=1)}{P(\mathbf{y} | z=0) \cdot P(z=0) + P(\mathbf{y} | z=1) \cdot P(z=1)}$$

The likelihoods are **multivariate normal orthant probabilities**:

$$P(\mathbf{y} | z=k) = P\left(\bigcap_{j: y_j=1} \{y_j^* > 0\} \cap \bigcap_{j: y_j=0} \{y_j^* \leq 0\} \,\Big|\, z=k\right)$$

$$= \int_{\mathcal{R}(\mathbf{y})} \phi(\mathbf{y}^*; \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k) \, d\mathbf{y}^*$$

where \(\mathcal{R}(\mathbf{y})\) is the orthant defined by the observed binary pattern, and \(\phi\) is the multivariate normal density.

Computing these orthant probabilities is non-trivial for \(J > 2\). The algorithm uses **Quasi-Monte Carlo integration** with Sobol sequences, which achieves convergence rate \(O(\log(n)^d / n)\) compared to \(O(1/\sqrt{n})\) for standard Monte Carlo.

## Parameter Estimation from Validation Data

The algorithm uses a **simple empirical estimator** rather than full Bayesian inference:

### Confusion Matrices
For each annotator \(j\):

$$\hat{\theta}_{j,k,\ell} = \frac{\#\{i : z_i = k, y_{ij} = \ell\}}{\#\{i : z_i = k\}}$$

### Covariance Matrices
For each true label class \(k \in \{0,1\}\):

1. Extract annotations where \(z_i = k\)
2. Transform to probit space: \(\tilde{y}_{ij} = \text{sign}(y_{ij} - 0.5) \cdot \Phi^{-1}(\hat{\theta}_{j,k,1})\)
3. Compute empirical covariance: \(\hat{\boldsymbol{\Sigma}}_k = \text{Cov}(\tilde{\mathbf{y}}_i : z_i = k)\)
4. Regularize for numerical stability: \(\hat{\boldsymbol{\Sigma}}_k \leftarrow \hat{\boldsymbol{\Sigma}}_k + \lambda \mathbf{I}\)

This gives point estimates \((\hat{\boldsymbol{\theta}}, \hat{\boldsymbol{\Sigma}}_0, \hat{\boldsymbol{\Sigma}}_1)\) for the model parameters.

## Posterior Predictive Distribution for Company Counts

For company \(c\) with \(n_c\) employees sampled from the broader company population, the posterior predictive distribution of ML expert count is:

$$P(\text{Count}_c = k | \mathcal{D}) = \int \sum_{\mathbf{z}} P(\text{Count}_c = k | \mathbf{z}) \cdot P(\mathbf{z} | \mathbf{Y}, \boldsymbol{\theta}, \boldsymbol{\Sigma}, \pi) \cdot p(\boldsymbol{\theta}, \boldsymbol{\Sigma}, \pi | \mathcal{D}_{\text{val}}) \, d\boldsymbol{\theta} \, d\boldsymbol{\Sigma} \, d\pi$$

where \(\mathbf{z} = (z_1, \ldots, z_{n_c})\) are the latent expertise labels for sampled employees.

This distribution accounts for:
1. Uncertainty in each employee's true label \(z_i\)
2. Uncertainty in model parameters \((\boldsymbol{\theta}, \boldsymbol{\Sigma})\)
3. Uncertainty in the population base rate \(\pi\)
4. Sampling uncertainty (which employees were observed)

## Bootstrap Uncertainty Quantification

The algorithm addresses four sources of uncertainty through a comprehensive bootstrap procedure:

### 1. Parameter Estimation Uncertainty

Resample validation data with replacement (stratified by true label):

$$\mathcal{D}_{\text{val}}^{(b)} \sim \text{Resample}(\mathcal{D}_{\text{val}})$$

For each bootstrap sample, re-estimate:

$$(\hat{\boldsymbol{\theta}}^{(b)}, \hat{\boldsymbol{\Sigma}}_0^{(b)}, \hat{\boldsymbol{\Sigma}}_1^{(b)}) = \text{Estimate}(\mathcal{D}_{\text{val}}^{(b)})$$

This captures how uncertainty in parameter estimates propagates to predictions.

### 2. Prior Uncertainty

The base rate \(\pi = P(z=1)\) is unknown. Rather than using a fixed value, we treat it as random:

$$\pi^{(b)} \sim \text{Beta}(2, 10)$$

The Beta(2, 10) hyperprior has:
- Mean: \(\mathbb{E}[\pi] = 2/(2+10) \approx 0.167\)
- Mode: \(1/11 \approx 0.091\)
- Variance: Allows substantial uncertainty

This reflects prior belief that ML experts are relatively rare, with appropriate uncertainty.

### 3. Sampling Uncertainty in Test Data

Under the assumption that observed employees are a representative random sample from each company's population, we resample within companies:

$$\mathcal{D}_{\text{test}}^{(b)} = \{\text{Resample}_c(\mathcal{C}_c) : c \in \text{Companies}\}$$

where \(\text{Resample}_c\) performs bootstrap within each company independently. This simulates alternative samples we might have observed from the company population, preserving company structure while varying which employees appear.

### 4. Latent Label Uncertainty (Realization Uncertainty)

Even given the model parameters and which employees are sampled, the true expertise status of each individual is uncertain. For each employee:

$$z_i^{(b)} \sim \text{Bernoulli}(P(z_i = 1 | y_i, \hat{\boldsymbol{\theta}}^{(b)}, \hat{\boldsymbol{\Sigma}}^{(b)}, \pi^{(b)}))$$

This **Bernoulli sampling step** is crucial—it ensures that the bootstrap distribution reflects actual discrete counts (15, 16, 17 experts) rather than expected values (16.3 experts). Without this step, the posterior predictive variance would be underestimated.

### Complete Bootstrap Procedure

For \(b = 1, \ldots, B\):

1. Sample \(\mathcal{D}_{\text{val}}^{(b)}\) (stratified by class)
2. Estimate parameters from \(\mathcal{D}_{\text{val}}^{(b)}\): get \((\hat{\boldsymbol{\theta}}^{(b)}, \hat{\boldsymbol{\Sigma}}_0^{(b)}, \hat{\boldsymbol{\Sigma}}_1^{(b)})\)
3. Sample \(\pi^{(b)} \sim \text{Beta}(2, 10)\)
4. Sample \(\mathcal{D}_{\text{test}}^{(b)}\) (stratified by company)
5. For each test item in \(\mathcal{D}_{\text{test}}^{(b)}\):
   - Compute \(p_i^{(b)} = P(z_i=1 | \mathbf{y}_i; \hat{\boldsymbol{\theta}}^{(b)}, \hat{\boldsymbol{\Sigma}}^{(b)}, \pi^{(b)})\)
   - Sample \(z_i^{(b)} \sim \text{Bernoulli}(p_i^{(b)})\)
6. Aggregate to company level: \(\text{Count}_c^{(b)} = \sum_{i \in \mathcal{C}_c^{(b)}} z_i^{(b)}\)

The bootstrap distribution \(\{\text{Count}_c^{(1)}, \ldots, \text{Count}_c^{(B)}\}\) provides:
- Point estimate: \(\hat{\mu}_c = B^{-1} \sum_{b=1}^B \text{Count}_c^{(b)}\)
- Uncertainty: percentiles, standard errors, credible intervals

## Variance Decomposition

The total variance in the posterior predictive distribution can be decomposed as:

$$\text{Var}[\text{Count}_c | \mathcal{D}] = \underbrace{\mathbb{E}_{\theta, \pi, S}[\text{Var}[\text{Count} | \theta, \pi, S]]}_{\text{Realization variance}} + \underbrace{\text{Var}_{\theta, \pi, S}[\mathbb{E}[\text{Count} | \theta, \pi, S]]}_{\text{Epistemic variance}}$$

where \(S\) denotes the sampled employees.

**Realization variance** (captured by Bernoulli sampling):
- For a given set of parameters and employees, the actual count varies because each \(z_i\) is a random variable
- Magnitude: \(\sum_{i \in S} p_i(1-p_i)\), roughly \(n \times \bar{p}(1-\bar{p})\) for a company with \(n\) employees

**Epistemic variance** (captured by bootstrap resampling):
- Uncertainty about which parameters, prior, and employees are correct
- Reducible with more validation data (for parameters) or more test data (for sampling)

Both components are essential for proper uncertainty quantification. The realization variance alone can contribute substantial uncertainty (e.g., \(\sigma \approx \sqrt{n}/2\) for \(p \approx 0.5\)), making it critical to include Bernoulli sampling.

## Theoretical Properties

### Consistency
Under regularity conditions:
- As \(n_{\text{val}} \to \infty\): \(\hat{\boldsymbol{\theta}}^{(b)} \xrightarrow{p} \boldsymbol{\theta}_{\text{true}}\)
- As \(B \to \infty\): Bootstrap distribution approximates the true posterior predictive distribution
- The combination of test data resampling and Bernoulli sampling properly accounts for both sampling and realization uncertainty

### Advantages over Independence Assumption
When annotator errors are correlated (e.g., \(\Sigma_{jk} \neq 0\) for \(j \neq k\)):
- Independence model: biased variance estimates, overconfident predictions
- Probit model: correctly accounts for correlation, wider (more honest) intervals

### Limitations
- **Representative sample assumption**: Assumes observed employees are a random sample from the company population (may not hold if LinkedIn visibility or search results are biased)
- **Identifiability**: Requires validation data with ground truth; cannot estimate \(\boldsymbol{\Sigma}\) from unlabeled data alone
- **Model specification**: Assumes errors follow multivariate normal in probit space
- **Computational cost**: Orthant probabilities require numerical integration

## Comparison to Alternatives

**Majority voting**: Treats all annotators equally, ignores error patterns
- Optimal only when all annotators have equal accuracy

**Weighted averaging**: Uses annotator-specific weights
- Still assumes independence

**Full Bayesian Dawid-Skene**: Places priors on \(\theta_j\), performs MCMC
- Assumes conditional independence given true label
- Misses correlations

**Expected value summation** (without Bernoulli sampling): Sums probabilities instead of sampling
- Computes posterior of E[Count] rather than posterior predictive of Count
- Underestimates variance by missing realization uncertainty

**This approach**: Combines Dawid-Skene confusion matrices with multivariate probit correlation structure and full posterior predictive sampling
- Captures both annotator-specific biases (\(\theta_j\)) and correlations (\(\Sigma\))
- Uses bootstrap for uncertainty rather than full Bayesian MCMC (simpler, faster)
- Includes Bernoulli sampling for complete uncertainty quantification
- Propagates all sources of uncertainty through to company-level aggregates

This framework is particularly appropriate when:
1. Annotators may share common biases or training
2. We need uncertainty quantification at the aggregate level for discrete counts
3. We have limited validation data (empirical estimates more robust than fitting complex hierarchical models)
4. Observed samples are reasonably representative of the broader population
