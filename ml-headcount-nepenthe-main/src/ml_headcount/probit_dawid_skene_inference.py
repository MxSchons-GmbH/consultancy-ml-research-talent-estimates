import numpy as np
from scipy import stats
from scipy.stats import multivariate_normal
import pandas as pd

def predict_new_item_probit_point_estimate(y_new, theta, chol_0, chol_1, class_prior=0.5, pattern_probs=None):
    """
    Compute p(true_label = 1 | y_new) using probit model with point estimates.
    
    Supports missing annotators (NaN values) by marginalizing over missing dimensions.
    
    Args:
        y_new: array of shape (n_annotators,) with new annotations {0,1} or NaN for missing
        theta: array of shape (n_annotators, 2, 2) with confusion matrix parameters
        chol_0: array of shape (n_annotators, n_annotators) with Cholesky factor for true_label=0
        chol_1: array of shape (n_annotators, n_annotators) with Cholesky factor for true_label=1
        class_prior: prior probability P(z=1), default 0.5 (uniform prior)
        pattern_probs: dict with precomputed pattern probabilities (optional)
                      Patterns can include None for missing annotators
        
    Returns:
        float: probability that true_label = 1
    """
    ll_0, ll_1 = _compute_likelihoods(y_new, theta, chol_0, chol_1, pattern_probs)
    
    # Bayes rule with explicit prior
    # P(z=1 | y) = P(y | z=1) * P(z=1) / P(y)
    # P(y) = P(y | z=0) * P(z=0) + P(y | z=1) * P(z=1)
    prior_0 = 1 - class_prior
    prior_1 = class_prior
    
    denominator = ll_0 * prior_0 + ll_1 * prior_1
    
    # Handle numerical issues
    if denominator <= 0 or np.isnan(denominator):
        # Fallback to uniform prior if likelihoods are invalid
        p_1 = 0.5
    else:
        p_1 = (ll_1 * prior_1) / denominator
    
    # Ensure probability is in valid range [0, 1] (0.0 and 1.0 are valid!)
    p_1 = np.clip(p_1, 0.0, 1.0)
    
    return p_1

def predict_new_item_probit_posterior(y_new, theta_samples, chol_0_samples, chol_1_samples):
    """
    Compute p(true_label = 1 | y_new) using probit model with posterior samples.
    
    Args:
        y_new: array of shape (6,) with new annotations {0,1}
        theta_samples: array of shape (n_posterior_samples, 6, 2, 2) with posterior samples
        chol_0_samples: array of shape (n_posterior_samples, 6, 6) with posterior samples
        chol_1_samples: array of shape (n_posterior_samples, 6, 6) with posterior samples
        
    Returns:
        float: probability that true_label = 1
    """
    n_posterior = len(theta_samples)
    log_likelihoods = []
    
    for i in range(n_posterior):
        ll_0, ll_1 = _compute_likelihoods(y_new, theta_samples[i], chol_0_samples[i], chol_1_samples[i])
        log_likelihoods.append([np.log(ll_0 + 1e-20), np.log(ll_1 + 1e-20)])
    
    # Average over posterior samples
    log_likelihoods = np.array(log_likelihoods)
    log_p_0 = logsumexp(log_likelihoods[:, 0]) - np.log(n_posterior)
    log_p_1 = logsumexp(log_likelihoods[:, 1]) - np.log(n_posterior)
    
    # Bayes rule with uniform prior
    p_1 = 1 / (1 + np.exp(log_p_0 - log_p_1))
    return p_1

def precompute_pattern_probabilities(mu, Sigma, n_samples=8192):
    """
    Precompute QMC probabilities for all possible annotation patterns.
    
    Args:
        mu: Mean vector (length determines number of annotators)
        Sigma: Covariance matrix
        n_samples: Number of QMC samples
        
    Returns:
        dict: {pattern_tuple: probability} for all 2^n_annotators patterns
    """
    n_annotators = len(mu)
    n_patterns = 2 ** n_annotators
    pattern_probs = {}
    
    # Generate all possible patterns
    for i in range(n_patterns):
        pattern = [(i >> j) & 1 for j in range(n_annotators)]
        pattern_tuple = tuple(pattern)
        
        # Use the same QMC function as individual computation
        prob = mvn_orthant_probability(np.array(pattern), mu, Sigma)
        pattern_probs[pattern_tuple] = prob
    
    return pattern_probs

def precompute_all_patterns(theta, chol_0, chol_1, n_samples=8192):
    """
    Precompute QMC probabilities for all patterns for both true label classes.
    
    Args:
        theta: array of shape (n_annotators, 2, 2) with confusion matrix parameters
        chol_0: array of shape (n_annotators, n_annotators) with Cholesky factor for true_label=0
        chol_1: array of shape (n_annotators, n_annotators) with Cholesky factor for true_label=1
        n_samples: Number of QMC samples
        
    Returns:
        dict: {pattern_tuple: {'ll_0': prob_0, 'll_1': prob_1}} for all patterns
    """
    # Convert Cholesky factors to covariance matrices
    Sigma_0 = chol_0 @ chol_0.T
    Sigma_1 = chol_1 @ chol_1.T
    
    # Probit-transformed base probabilities
    mu_0 = stats.norm.ppf(np.clip(theta[:, 0, 1], 1e-10, 1-1e-10))  # probit(p_base) for true_label=0
    mu_1 = stats.norm.ppf(np.clip(theta[:, 1, 1], 1e-10, 1-1e-10))  # probit(p_base) for true_label=1
    
    # Check for numerical issues in mu values
    if np.any(np.isnan(mu_0)) or np.any(np.isnan(mu_1)):
        print(f"Warning: NaN in probit-transformed means")
        print(f"  theta[:, 0, 1]: {theta[:, 0, 1]}")
        print(f"  theta[:, 1, 1]: {theta[:, 1, 1]}")
        mu_0 = np.nan_to_num(mu_0, nan=0.0)
        mu_1 = np.nan_to_num(mu_1, nan=0.0)
    
    # Precompute patterns for both classes
    pattern_probs_0 = precompute_pattern_probabilities(mu_0, Sigma_0, n_samples)
    pattern_probs_1 = precompute_pattern_probabilities(mu_1, Sigma_1, n_samples)
    
    # Combine into single lookup
    pattern_probs = {}
    for pattern in pattern_probs_0:
        ll_0 = pattern_probs_0[pattern]
        ll_1 = pattern_probs_1[pattern]
        
        # Handle numerical issues
        if np.isnan(ll_0) or np.isnan(ll_1):
            print(f"Warning: NaN likelihoods for pattern {pattern}")
            ll_0 = 1e-20 if np.isnan(ll_0) else ll_0
            ll_1 = 1e-20 if np.isnan(ll_1) else ll_1
        
        # Ensure positive likelihoods
        ll_0 = max(ll_0, 1e-20)
        ll_1 = max(ll_1, 1e-20)
        
        pattern_probs[pattern] = {'ll_0': ll_0, 'll_1': ll_1}
    
    return pattern_probs

def collect_unique_patterns(annotation_data):
    """
    Collect all unique annotation patterns from data, treating NaN as missing (None).
    
    Args:
        annotation_data: array of shape (n_items, n_annotators) with annotations {0,1} or NaN
        
    Returns:
        set: Set of pattern tuples, where each tuple has length n_annotators
             and contains 0, 1, or None (for missing)
    """
    unique_patterns = set()
    for y_new in annotation_data:
        # Convert to tuple with None for missing values
        pattern = tuple(None if np.isnan(x) else int(x) for x in y_new)
        unique_patterns.add(pattern)
    return unique_patterns

def precompute_observed_patterns(theta, chol_0, chol_1, unique_patterns, n_samples=8192):
    """
    Precompute QMC probabilities for observed patterns (including those with missing values).
    
    For patterns with missing annotators, marginalizes over the missing dimensions by
    computing MVN probability over only the observed annotators.
    
    Args:
        theta: array of shape (n_annotators, 2, 2) with confusion matrix parameters
        chol_0: array of shape (n_annotators, n_annotators) with Cholesky factor for true_label=0
        chol_1: array of shape (n_annotators, n_annotators) with Cholesky factor for true_label=1
        unique_patterns: set of pattern tuples (each element is 0, 1, or None)
        n_samples: Number of QMC samples
        
    Returns:
        dict: {pattern_tuple: {'ll_0': prob_0, 'll_1': prob_1}} for all observed patterns
    """
    # Convert Cholesky factors to covariance matrices
    Sigma_0 = chol_0 @ chol_0.T
    Sigma_1 = chol_1 @ chol_1.T
    
    # Probit-transformed base probabilities
    mu_0 = stats.norm.ppf(np.clip(theta[:, 0, 1], 1e-10, 1-1e-10))  # probit(p_base) for true_label=0
    mu_1 = stats.norm.ppf(np.clip(theta[:, 1, 1], 1e-10, 1-1e-10))  # probit(p_base) for true_label=1
    
    # Check for numerical issues in mu values
    if np.any(np.isnan(mu_0)) or np.any(np.isnan(mu_1)):
        print(f"Warning: NaN in probit-transformed means")
        print(f"  theta[:, 0, 1]: {theta[:, 0, 1]}")
        print(f"  theta[:, 1, 1]: {theta[:, 1, 1]}")
        mu_0 = np.nan_to_num(mu_0, nan=0.0)
        mu_1 = np.nan_to_num(mu_1, nan=0.0)
    
    pattern_probs = {}
    
    for pattern in unique_patterns:
        # Extract present annotators (those that are not None)
        present_indices = [i for i, val in enumerate(pattern) if val is not None]
        
        if len(present_indices) == 0:
            # All annotators missing - no information, return uniform likelihoods
            pattern_probs[pattern] = {'ll_0': 1.0, 'll_1': 1.0}
            continue
        
        # Extract observed annotation values
        y_obs = np.array([pattern[i] for i in present_indices], dtype=int)
        
        # Extract sub-matrices for observed annotators only
        mu_0_obs = mu_0[present_indices]
        mu_1_obs = mu_1[present_indices]
        Sigma_0_obs = Sigma_0[np.ix_(present_indices, present_indices)]
        Sigma_1_obs = Sigma_1[np.ix_(present_indices, present_indices)]
        
        # Compute MVN probability over observed annotators (marginalization)
        ll_0 = mvn_orthant_probability(y_obs, mu_0_obs, Sigma_0_obs)
        ll_1 = mvn_orthant_probability(y_obs, mu_1_obs, Sigma_1_obs)
        
        # Handle numerical issues
        if np.isnan(ll_0) or np.isnan(ll_1):
            ll_0 = 1e-20 if np.isnan(ll_0) else ll_0
            ll_1 = 1e-20 if np.isnan(ll_1) else ll_1
        
        # Ensure positive likelihoods
        ll_0 = max(ll_0, 1e-20)
        ll_1 = max(ll_1, 1e-20)
        
        pattern_probs[pattern] = {'ll_0': ll_0, 'll_1': ll_1}
    
    return pattern_probs

def _compute_likelihoods(y_new, theta, chol_0, chol_1, pattern_probs=None):
    """
    Compute likelihoods for a single parameter set using pattern lookup if available.
    
    Supports missing annotators (NaN values) by using 3-state patterns (0, 1, None).
    
    Args:
        y_new: array of shape (n_annotators,) with new annotations {0,1} or NaN for missing
        theta: array of shape (n_annotators, 2, 2) with confusion matrix parameters
        chol_0: array of shape (n_annotators, n_annotators) with Cholesky factor for true_label=0
        chol_1: array of shape (n_annotators, n_annotators) with Cholesky factor for true_label=1
        pattern_probs: dict with precomputed pattern probabilities (optional)
        
    Returns:
        tuple of (ll_0, ll_1) likelihoods for true_label=0 and true_label=1
    """
    # If pattern lookup is available, use it
    if pattern_probs is not None:
        # Convert to pattern tuple with None for missing values
        pattern = tuple(None if np.isnan(x) else int(x) for x in y_new)
        if pattern in pattern_probs:
            return pattern_probs[pattern]['ll_0'], pattern_probs[pattern]['ll_1']
        else:
            # This shouldn't happen if we precomputed all observed patterns
            raise ValueError(f"Pattern {pattern} not found in precomputed patterns")
    
    # Fallback to original computation (for backward compatibility)
    # Handle missing values by marginalizing over missing annotators
    valid_mask = ~np.isnan(y_new)
    n_valid = valid_mask.sum()
    
    # Convert Cholesky factors to covariance matrices
    Sigma_0 = chol_0 @ chol_0.T
    Sigma_1 = chol_1 @ chol_1.T
    
    # Probit-transformed base probabilities
    mu_0 = stats.norm.ppf(np.clip(theta[:, 0, 1], 1e-10, 1-1e-10))  # probit(p_base) for true_label=0
    mu_1 = stats.norm.ppf(np.clip(theta[:, 1, 1], 1e-10, 1-1e-10))  # probit(p_base) for true_label=1
    
    # Check for numerical issues in mu values
    if np.any(np.isnan(mu_0)) or np.any(np.isnan(mu_1)):
        print(f"Warning: NaN in probit-transformed means")
        print(f"  theta[:, 0, 1]: {theta[:, 0, 1]}")
        print(f"  theta[:, 1, 1]: {theta[:, 1, 1]}")
        mu_0 = np.nan_to_num(mu_0, nan=0.0)
        mu_1 = np.nan_to_num(mu_1, nan=0.0)
    
    # Extract only observed annotators if there are missing values
    if n_valid < len(y_new):
        y_obs = y_new[valid_mask]
        mu_0_obs = mu_0[valid_mask]
        mu_1_obs = mu_1[valid_mask]
        Sigma_0_obs = Sigma_0[np.ix_(valid_mask, valid_mask)]
        Sigma_1_obs = Sigma_1[np.ix_(valid_mask, valid_mask)]
    else:
        y_obs = y_new
        mu_0_obs = mu_0
        mu_1_obs = mu_1
        Sigma_0_obs = Sigma_0
        Sigma_1_obs = Sigma_1
    
    # Compute marginal likelihoods using MVN CDF over observed annotators
    if n_valid == 0:
        # All missing - no information, return uniform likelihoods
        ll_0 = 1.0
        ll_1 = 1.0
    else:
        ll_0 = mvn_orthant_probability(y_obs, mu_0_obs, Sigma_0_obs)
        ll_1 = mvn_orthant_probability(y_obs, mu_1_obs, Sigma_1_obs)
    
    # Check for numerical issues in likelihoods (only print if problematic)
    if np.isnan(ll_0) or np.isnan(ll_1):
        print(f"Warning: NaN likelihoods - ll_0: {ll_0}, ll_1: {ll_1} for pattern {y_new}")
        ll_0 = 1e-20 if np.isnan(ll_0) else ll_0
        ll_1 = 1e-20 if np.isnan(ll_1) else ll_1
    
    # Ensure positive likelihoods (0 is mathematically valid, but problematic for division)
    ll_0 = max(ll_0, 1e-20)
    ll_1 = max(ll_1, 1e-20)
    
    return ll_0, ll_1

def mvn_orthant_probability(y_obs, mu, Sigma):
    """
    Compute P(z_j > 0 for j where y_obs[j]=1, z_j ≤ 0 for j where y_obs[j]=0)
    where z ~ MVN(mu, Sigma)
    
    Uses the fastest available method based on the annotation pattern.
    """
    # Check if we can use scipy's faster CDF (only upper bounds)
    if np.all(y_obs == 0):
        # All annotations are 0: P(all y_j* ≤ 0) = CDF(0, ..., 0)
        upper = np.zeros(len(mu))
        return multivariate_normal.cdf(upper, mean=mu, cov=Sigma)
    
    elif np.all(y_obs == 1):
        # All annotations are 1: P(all y_j* > 0) = 1 - CDF(0, ..., 0)  
        upper = np.zeros(len(mu))
        return 1 - multivariate_normal.cdf(upper, mean=mu, cov=Sigma)
    
    else:
        # Mixed annotations: need rectangular region - use Quasi-Monte Carlo
        return mvn_orthant_probability_qmc(y_obs, mu, Sigma, n_samples=8192)  # 2^13

# Module-level caches for optimized version
_cached_z = {}  # key: (d, n_samples, seed) -> np.ndarray of shape (n_samples, d)

def _get_cached_standard_normal_z(d: int, n_samples: int, seed: int) -> np.ndarray:
    """Get cached standard normal Z samples from Sobol sequence."""
    key = (d, n_samples, seed)
    if key in _cached_z:
        return _cached_z[key]
    
    # Generate Sobol samples and convert to standard normal
    from scipy.stats import qmc, norm
    sampler = qmc.Sobol(d=d, scramble=True, seed=seed)
    u = sampler.random(n=n_samples)
    z = norm.ppf(u)  # standard normal
    _cached_z[key] = z
    return z

def _factorize_covariance(Sigma: np.ndarray) -> np.ndarray:
    """Robust covariance factorization (Cholesky or eigendecomposition)."""
    try:
        return np.linalg.cholesky(Sigma)
    except np.linalg.LinAlgError:
        eigvals, eigvecs = np.linalg.eigh(Sigma)
        eigvals = np.maximum(eigvals, 1e-12)
        return eigvecs @ np.diag(np.sqrt(eigvals))

def mvn_orthant_probability_qmc(y_obs, mu, Sigma, n_samples=8192):
    """
    Optimized Quasi-Monte Carlo using vectorized operations and caching
    
    Args:
        y_obs: Binary observation vector
        mu: Mean vector
        Sigma: Covariance matrix
        n_samples: Number of QMC samples (default: 8192 = 2^13)
        
    Returns:
        Probability estimate
        
    Notes:
        - Uses cached Sobol sequences for efficiency
        - Vectorized constraint checking for speed
        - Power of 2 samples are optimal for Sobol sequences
        - QMC converges as O(log(n)^d/n) vs O(1/sqrt(n)) for MC
    """
    from scipy.stats import qmc
    
    # Use power of 2 for optimal QMC properties
    if n_samples & (n_samples - 1) != 0:  # Not a power of 2
        n_samples = 2 ** int(np.log2(n_samples) + 1)  # Round up to next power of 2
    
    def _compute_qmc_probability(n_samp, seed_offset=0):
        """Helper function to compute QMC probability with given sample size and seed"""
        try:
            # Direct multivariate normal QMC sampling (2x faster than Sobol+transform)
            mv_sampler = qmc.MultivariateNormalQMC(mean=mu, cov=Sigma, seed=42+seed_offset)
            samples = mv_sampler.random(n=n_samp)
            
        except Exception:
            # Fallback to manual Sobol+transform for older scipy versions
            # Use cached standard normal samples for efficiency
            z = _get_cached_standard_normal_z(d=len(mu), n_samples=n_samp, seed=42+seed_offset)
            
            # Robust Cholesky decomposition
            L = _factorize_covariance(Sigma)
            
            # Vectorized transformation: samples = mu + z @ L.T
            samples = mu + (z @ L.T)
        
        # Optimized vectorized constraint checking
        # Create masks for positive and negative constraints
        pos_mask = (y_obs == 1)
        neg_mask = (y_obs == 0)
        
        # Initialize valid array
        valid = np.ones(n_samp, dtype=bool)
        
        # Optimized constraint checking: combine positive and negative checks
        # Create constraint violations: True means constraint is violated
        pos_violations = (samples <= 0) & pos_mask  # samples <= 0 where we need > 0
        neg_violations = (samples > 0) & neg_mask   # samples > 0 where we need <= 0
        
        # A sample is valid if it has NO violations of any kind
        all_violations = pos_violations | neg_violations
        valid = ~np.any(all_violations, axis=1)
        
        # Handle edge case where no constraints exist (shouldn't happen in practice)
        if not np.any(pos_mask) and not np.any(neg_mask):
            return 1.0  # All samples are valid by default
        
        prob = np.mean(valid)
        return prob
    
    # First attempt with standard sample size
    prob = _compute_qmc_probability(n_samples)
    
    # Check for problematic results and retry with more samples if needed
    # Only retry on NaN, not on 0.0 (which is a valid probability)
    if np.isnan(prob):
        print(f"Warning: QMC returned NaN with {n_samples} samples for pattern {y_obs}")
        print(f"  Retrying with {n_samples*4} samples...")
        
        # Retry with 4x more samples
        prob_retry = _compute_qmc_probability(n_samples * 4, seed_offset=1)
        
        if np.isnan(prob_retry):
            print(f"  Retry also returned NaN, using fallback MC...")
            # Final fallback to regular Monte Carlo
            prob = mvn_orthant_probability_mc(y_obs, mu, Sigma, n_samples=n_samples*2)
        else:
            prob = prob_retry
            print(f"  Retry successful: {prob}")
    
    # For very small probabilities (including 0.0), just ensure they're not NaN
    if np.isnan(prob):
        print(f"Warning: Final probability is NaN, using small positive value")
        prob = 1e-20
    
    # Ensure probability is in valid range [0, 1] (0.0 is valid!)
    prob = np.clip(prob, 0.0, 1.0)
    
    return prob

def mvn_orthant_probability_mc(y_obs, mu, Sigma, n_samples=10000):
    """Monte Carlo fallback for numerical issues"""
    samples = np.random.multivariate_normal(mu, Sigma, n_samples)
    
    # Check which samples satisfy the constraints
    valid = np.ones(n_samples, dtype=bool)
    for j in range(len(y_obs)):
        if y_obs[j] == 1:
            valid &= (samples[:, j] > 0)
        else:
            valid &= (samples[:, j] <= 0)
    
    return np.mean(valid)

def logsumexp(x):
    """Numerically stable log-sum-exp"""
    x_max = np.max(x)
    return x_max + np.log(np.sum(np.exp(x - x_max)))

def compute_truncated_mvn_moments(y_obs, mu, Sigma, n_samples=8192):
    """
    Compute E[z|y] and E[zz^T|y] for truncated multivariate normal.
    
    Uses QMC sampling to estimate conditional moments where z ~ N(mu, Sigma)
    and y = 1[z > 0] (element-wise thresholding).
    
    Args:
        y_obs: Binary observation vector (0/1) of length d
        mu: Mean vector of length d
        Sigma: Covariance matrix of shape (d, d)
        n_samples: Number of QMC samples for moment estimation
        
    Returns:
        tuple: (E[z|y], E[zz^T|y])
    """
    from scipy.stats import qmc
    
    d = len(mu)
    
    # Generate samples from truncated MVN using QMC
    try:
        mv_sampler = qmc.MultivariateNormalQMC(mean=mu, cov=Sigma, seed=42)
        samples = mv_sampler.random(n=n_samples)
    except Exception:
        # Fallback to manual Sobol+transform
        z = _get_cached_standard_normal_z(d=d, n_samples=n_samples, seed=42)
        L = _factorize_covariance(Sigma)
        samples = mu + (z @ L.T)
    
    # Filter samples to those satisfying the orthant constraints
    pos_mask = (y_obs == 1)
    neg_mask = (y_obs == 0)
    
    valid = np.ones(n_samples, dtype=bool)
    pos_violations = (samples <= 0) & pos_mask
    neg_violations = (samples > 0) & neg_mask
    all_violations = pos_violations | neg_violations
    valid = ~np.any(all_violations, axis=1)
    
    if valid.sum() == 0:
        # No valid samples - return mean and outer product of mean
        return mu.copy(), np.outer(mu, mu) + Sigma
    
    # Compute conditional moments from valid samples
    valid_samples = samples[valid]
    E_z = np.mean(valid_samples, axis=0)
    
    # E[zz^T] = mean of outer products
    E_z_outer = np.mean([np.outer(s, s) for s in valid_samples], axis=0)
    
    return E_z, E_z_outer

def estimate_covariance_multivariate_probit_em(
    annotations, 
    mu, 
    init_Sigma=None, 
    max_iter=20, 
    tol=1e-4,
    n_samples=8192,
    logger=None
):
    """
    Estimate covariance matrix using EM algorithm for multivariate probit model.
    
    Args:
        annotations: Array of shape (n_items, n_annotators) with binary annotations {0,1} or NaN
        mu: Mean vector of length n_annotators (fixed)
        init_Sigma: Initial covariance matrix (if None, uses identity)
        max_iter: Maximum EM iterations
        tol: Convergence tolerance
        n_samples: QMC samples per moment computation
        
    Returns:
        Sigma: Estimated correlation matrix (diagonal = 1)
    """
    n_items, n_annotators = annotations.shape
    
    # Initialize Sigma (correlation matrix)
    if init_Sigma is None:
        Sigma = np.eye(n_annotators)
    else:
        # Ensure it's a correlation matrix (diagonal = 1)
        diag_sqrt = np.sqrt(np.diag(init_Sigma))
        Sigma = init_Sigma / np.outer(diag_sqrt, diag_sqrt)
    
    # Handle missing values by working with observed subsets
    import time
    for iteration in range(max_iter):
        iter_start_time = time.time()
        Sigma_old = Sigma.copy()
        
        # E-step: compute conditional moments for each item
        E_z_sum = np.zeros(n_annotators)
        E_z_outer_sum = np.zeros((n_annotators, n_annotators))
        n_valid = 0
        
        if logger:
            logger.info(f"    EM iteration {iteration + 1}/{max_iter}: processing {n_items} items...")
        
        # Log progress every 10% or every 100 items, whichever is more frequent
        log_interval = max(1, min(n_items // 10, 100))
        
        for i in range(n_items):
            if logger and (i + 1) % log_interval == 0:
                logger.info(f"      Processed {i + 1}/{n_items} items...")
            
            y_obs = annotations[i]
            valid_mask = ~np.isnan(y_obs)
            
            if valid_mask.sum() == 0:
                continue  # Skip items with all missing
            
            # Extract observed subset
            y_obs_subset = y_obs[valid_mask].astype(int)
            mu_subset = mu[valid_mask]
            Sigma_subset = Sigma[np.ix_(valid_mask, valid_mask)]
            
            # Compute truncated moments for observed subset
            E_z_subset, E_z_outer_subset = compute_truncated_mvn_moments(
                y_obs_subset, mu_subset, Sigma_subset, n_samples=n_samples
            )
            
            # Expand back to full dimension (missing dimensions have zero contribution)
            E_z_full = np.zeros(n_annotators)
            E_z_outer_full = np.zeros((n_annotators, n_annotators))
            
            E_z_full[valid_mask] = E_z_subset
            E_z_outer_full[np.ix_(valid_mask, valid_mask)] = E_z_outer_subset
            
            E_z_sum += E_z_full
            E_z_outer_sum += E_z_outer_full
            n_valid += 1
        
        if n_valid == 0:
            break
        
        # M-step: update Sigma
        # For correlation matrix, we need to normalize
        # E[zz^T] = E[z]E[z]^T + Cov[z|y]
        # We estimate Cov[z|y] from the conditional moments
        
        # Average conditional moments
        E_z_mean = E_z_sum / n_valid
        E_z_outer_mean = E_z_outer_sum / n_valid
        
        # Update covariance: Cov = E[zz^T] - E[z]E[z]^T
        Cov_update = E_z_outer_mean - np.outer(E_z_mean, E_z_mean)
        
        # Convert to correlation matrix
        diag_sqrt = np.sqrt(np.diag(Cov_update))
        diag_sqrt = np.maximum(diag_sqrt, 1e-6)  # Avoid division by zero
        Sigma = Cov_update / np.outer(diag_sqrt, diag_sqrt)
        
        # Ensure positive definiteness
        try:
            np.linalg.cholesky(Sigma)
        except np.linalg.LinAlgError:
            # Regularize if not PD
            eigvals, eigvecs = np.linalg.eigh(Sigma)
            eigvals = np.maximum(eigvals, 1e-6)
            Sigma = eigvecs @ np.diag(eigvals) @ eigvecs.T
            # Renormalize to correlation matrix
            diag_sqrt = np.sqrt(np.diag(Sigma))
            Sigma = Sigma / np.outer(diag_sqrt, diag_sqrt)
        
        # Check convergence
        diff = np.max(np.abs(Sigma - Sigma_old))
        iter_time = time.time() - iter_start_time
        if logger:
            logger.info(f"    Iteration {iteration + 1} completed in {iter_time:.1f}s (max diff: {diff:.6f}, n_valid: {n_valid})")
        if diff < tol:
            if logger:
                logger.info(f"    Converged after {iteration + 1} iterations")
            break
    
    return Sigma

def estimate_class_covariance(validation_data, true_label, method="tetrachoric", n_samples=8192):
    """
    Estimate covariance matrix for a specific true label class.
    
    Args:
        validation_data: DataFrame with annotations and 'category' column
        true_label: 0 or 1 (which class to estimate for)
        method: "em" for EM algorithm, "tetrachoric" for pairwise tetrachoric
        n_samples: QMC samples for EM (if method="em")
        
    Returns:
        Sigma: Estimated correlation matrix (6×6, diagonal = 1)
    """
    # Extract annotation columns
    annotator_cols = [col for col in validation_data.columns 
                     if col not in ['cv_text', 'category']]
    n_annotators = len(annotator_cols)
    
    # Filter by true label
    mask = (validation_data['category'] == true_label)
    class_data = validation_data[mask].copy()
    
    if len(class_data) < 2:
        # Not enough data - return identity
        return np.eye(n_annotators)
    
    annotations = class_data[annotator_cols].values
    
    # Compute mean vector from marginal probabilities
    mu = np.zeros(n_annotators)
    for j in range(n_annotators):
        # P(y_j = 1 | z = true_label)
        p_j = np.nanmean(annotations[:, j])
        p_j = np.clip(p_j, 1e-6, 1.0 - 1e-6)
        mu[j] = stats.norm.ppf(p_j)
    
    if method == "tetrachoric":
        # Use pairwise tetrachoric correlations
        from ml_headcount.synthetic_data_generation import estimate_tetrachoric_correlation_binary
        
        Sigma = np.eye(n_annotators)
        for i in range(n_annotators):
            for j in range(i + 1, n_annotators):
                x = annotations[:, i]
                y = annotations[:, j]
                rho_ij = estimate_tetrachoric_correlation_binary(x, y)
                Sigma[i, j] = rho_ij
                Sigma[j, i] = rho_ij
        
        # Ensure positive definiteness
        try:
            np.linalg.cholesky(Sigma)
        except np.linalg.LinAlgError:
            # Project to nearest PD matrix
            eigvals, eigvecs = np.linalg.eigh(Sigma)
            eigvals = np.maximum(eigvals, 1e-6)
            Sigma = eigvecs @ np.diag(eigvals) @ eigvecs.T
            # Renormalize to correlation matrix
            diag_sqrt = np.sqrt(np.diag(Sigma))
            Sigma = Sigma / np.outer(diag_sqrt, diag_sqrt)
    
    elif method == "em":
        # Initialize from tetrachoric for faster convergence
        from ml_headcount.synthetic_data_generation import estimate_tetrachoric_correlation_binary
        
        init_Sigma = np.eye(n_annotators)
        for i in range(n_annotators):
            for j in range(i + 1, n_annotators):
                x = annotations[:, i]
                y = annotations[:, j]
                rho_ij = estimate_tetrachoric_correlation_binary(x, y)
                init_Sigma[i, j] = rho_ij
                init_Sigma[j, i] = rho_ij
        
        # Run EM
        Sigma = estimate_covariance_multivariate_probit_em(
            annotations, mu, init_Sigma=init_Sigma, n_samples=n_samples
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return Sigma

def extract_point_estimates_from_trace(trace):
    """
    Extract point estimates (posterior means) from PyMC trace
    
    Args:
        trace: PyMC trace object with posterior samples
        
    Returns:
        dict with theta, chol_0, chol_1 point estimates
    """
    # Extract posterior means
    theta_mean = trace.posterior['theta'].mean(('chain', 'draw')).values
    chol_0_mean = trace.posterior['chol_0'].mean(('chain', 'draw')).values  
    chol_1_mean = trace.posterior['chol_1'].mean(('chain', 'draw')).values
    
    return {
        'theta': theta_mean,
        'chol_0': chol_0_mean, 
        'chol_1': chol_1_mean
    }

def create_simple_point_estimate(validation_data):
    """
    Create a simple point estimate from validation data using empirical confusion matrices.
    
    Args:
        validation_data: DataFrame with annotations and true labels
        
    Returns:
        dict with theta, chol_0, chol_1 point estimates
    """
    # Extract annotation columns (exclude cv_text and category)
    annotator_cols = [col for col in validation_data.columns if col not in ['cv_text', 'category']]
    n_annotators = len(annotator_cols)
    
    # Get true labels and annotations
    true_labels = validation_data['category'].values
    annotations = validation_data[annotator_cols].values
    
    # Reduced verbosity for bootstrap iterations
    # print(f"Creating point estimate from {len(validation_data)} validation items with {n_annotators} annotators")
    # print(f"True label distribution: {np.bincount(true_labels)}")
    
    # Compute empirical confusion matrices for each annotator
    theta = np.zeros((n_annotators, 2, 2))
    
    for j in range(n_annotators):
        # Count confusion matrix entries
        # theta[j, k, l] = P(annotation = l | true_label = k, annotator = j)
        for k in [0, 1]:  # true_label
            for l in [0, 1]:  # annotation
                mask = (true_labels == k) & (annotations[:, j] == l)
                total_k = (true_labels == k).sum()
                if total_k > 0:
                    theta[j, k, l] = mask.sum() / total_k
                else:
                    theta[j, k, l] = 0.5  # uniform if no data
    
    # Estimate covariance matrices using tetrachoric correlations for each true label class
    # For true_label = 0: estimate using tetrachoric correlations
    mask_0 = (true_labels == 0)
    if mask_0.sum() > 1:  # Need at least 2 samples for covariance
        try:
            # Use tetrachoric estimator (function filters by true_label internally)
            Sigma_0 = estimate_class_covariance(validation_data, true_label=0, method="tetrachoric")
        except Exception as e:
            # Fallback to old heuristic if tetrachoric fails
            print(f"Warning: Tetrachoric estimation failed for class 0: {e}, using fallback")
            annotations_0 = annotations[mask_0]
            probit_annotations_0 = np.zeros_like(annotations_0, dtype=float)
            for j in range(n_annotators):
                p_annotate_1_given_0 = theta[j, 0, 1]
                probit_annotations_0[:, j] = stats.norm.ppf(np.clip(p_annotate_1_given_0, 1e-10, 1-1e-10))
                probit_annotations_0[:, j] = np.where(annotations_0[:, j] == 1, 
                                                       probit_annotations_0[:, j], 
                                                       -probit_annotations_0[:, j])
            Sigma_0 = np.cov(probit_annotations_0.T)
            Sigma_0 = Sigma_0 + np.eye(n_annotators) * 1e-6
    else:
        Sigma_0 = np.eye(n_annotators) * 0.1  # Small diagonal if no data
    
    # For true_label = 1: estimate using tetrachoric correlations
    mask_1 = (true_labels == 1)
    if mask_1.sum() > 1:  # Need at least 2 samples for covariance
        try:
            # Use tetrachoric estimator (function filters by true_label internally)
            Sigma_1 = estimate_class_covariance(validation_data, true_label=1, method="tetrachoric")
        except Exception as e:
            # Fallback to old heuristic if tetrachoric fails
            print(f"Warning: Tetrachoric estimation failed for class 1: {e}, using fallback")
            annotations_1 = annotations[mask_1]
            probit_annotations_1 = np.zeros_like(annotations_1, dtype=float)
            for j in range(n_annotators):
                p_annotate_1_given_1 = theta[j, 1, 1]
                probit_annotations_1[:, j] = stats.norm.ppf(np.clip(p_annotate_1_given_1, 1e-10, 1-1e-10))
                probit_annotations_1[:, j] = np.where(annotations_1[:, j] == 1, 
                                                       probit_annotations_1[:, j], 
                                                       -probit_annotations_1[:, j])
            Sigma_1 = np.cov(probit_annotations_1.T)
            Sigma_1 = Sigma_1 + np.eye(n_annotators) * 1e-6
    else:
        Sigma_1 = np.eye(n_annotators) * 0.1  # Small diagonal if no data
    
    # Compute Cholesky decompositions
    try:
        chol_0 = np.linalg.cholesky(Sigma_0)
    except np.linalg.LinAlgError:
        # If Cholesky fails, use eigendecomposition
        eigvals, eigvecs = np.linalg.eigh(Sigma_0)
        eigvals = np.maximum(eigvals, 1e-6)  # Ensure positive
        chol_0 = eigvecs @ np.diag(np.sqrt(eigvals))
    
    try:
        chol_1 = np.linalg.cholesky(Sigma_1)
    except np.linalg.LinAlgError:
        # If Cholesky fails, use eigendecomposition
        eigvals, eigvecs = np.linalg.eigh(Sigma_1)
        eigvals = np.maximum(eigvals, 1e-6)  # Ensure positive
        chol_1 = eigvecs @ np.diag(np.sqrt(eigvals))
    
    # Reduced verbosity for bootstrap iterations
    # print(f"Created point estimate:")
    # print(f"  Theta shape: {theta.shape}")
    # print(f"  Chol_0 shape: {chol_0.shape}")
    # print(f"  Chol_1 shape: {chol_1.shape}")
    
    return {
        'theta': theta,
        'chol_0': chol_0,
        'chol_1': chol_1
    }

def run_probit_validation(validation_data, point_estimates, threshold=0.5):
    """
    Run probit model on validation data and compute confusion matrices
    
    Args:
        validation_data: DataFrame with annotations and true labels
        point_estimates: dict with theta, chol_0, chol_1 point estimates
        threshold: probability threshold for binary classification
        
    Returns:
        dict with predictions and confusion matrices
    """
    # Extract annotation columns (exclude cv_text and category)
    annotator_cols = [col for col in validation_data.columns if col not in ['cv_text', 'category']]
    n_annotators = len(annotator_cols)
    n_items = len(validation_data)
    
    # Get true labels
    true_labels = validation_data['category'].values
    
    # Get annotations
    annotations = validation_data[annotator_cols].values
    
    print(f"Running probit model on {n_items} validation items with {n_annotators} annotators")
    print(f"Annotation columns: {annotator_cols}")
    print(f"True label distribution: {np.bincount(true_labels)}")
    
    # Make predictions using probit model
    probit_predictions = np.zeros(n_items)
    
    for i in range(n_items):
        y_new = annotations[i]
        prob_positive = predict_new_item_probit_point_estimate(
            y_new, 
            point_estimates['theta'], 
            point_estimates['chol_0'], 
            point_estimates['chol_1']
        )
        probit_predictions[i] = prob_positive
    
    # Convert probabilities to binary predictions
    probit_binary = (probit_predictions >= threshold).astype(int)
    
    # Compute confusion matrix for probit model
    from sklearn.metrics import confusion_matrix, classification_report
    
    probit_cm = confusion_matrix(true_labels, probit_binary)
    
    print(f"\nProbit Model Confusion Matrix (threshold={threshold}):")
    print(probit_cm)
    print(f"Probit Model Classification Report:")
    print(classification_report(true_labels, probit_binary, target_names=['Negative', 'Positive']))
    
    # Compute sensitivity and specificity for probit model
    tn, fp, fn, tp = probit_cm.ravel()
    probit_sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    probit_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    print(f"Probit Model - Sensitivity: {probit_sensitivity:.3f}, Specificity: {probit_specificity:.3f}")
    
    # Compute confusion matrices for individual annotators
    annotator_cms = {}
    annotator_reports = {}
    annotator_metrics = {}
    
    print(f"\n{'='*80}")
    print("SENSITIVITY & SPECIFICITY COMPARISON")
    print(f"{'='*80}")
    print(f"{'Model/Annotator':<25} {'Sensitivity':<12} {'Specificity':<12} {'Accuracy':<10}")
    print(f"{'-'*80}")
    
    # Probit model metrics
    probit_accuracy = (tp + tn) / (tp + tn + fp + fn)
    print(f"{'Probit Model':<25} {probit_sensitivity:<12.3f} {probit_specificity:<12.3f} {probit_accuracy:<10.3f}")
    
    for j, col in enumerate(annotator_cols):
        annotator_pred = annotations[:, j]
        annotator_cm = confusion_matrix(true_labels, annotator_pred)
        annotator_cms[col] = annotator_cm
        annotator_reports[col] = classification_report(true_labels, annotator_pred, target_names=['Negative', 'Positive'], output_dict=True)
        
        # Compute sensitivity and specificity
        tn, fp, fn, tp = annotator_cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        
        annotator_metrics[col] = {
            'sensitivity': sensitivity,
            'specificity': specificity,
            'accuracy': accuracy,
            'confusion_matrix': annotator_cm
        }
        
        print(f"{col:<25} {sensitivity:<12.3f} {specificity:<12.3f} {accuracy:<10.3f}")
        
        print(f"\n{col} Confusion Matrix:")
        print(annotator_cm)
        print(f"{col} Classification Report:")
        print(classification_report(true_labels, annotator_pred, target_names=['Negative', 'Positive']))
    
    return {
        'probit_predictions': probit_predictions,
        'probit_binary': probit_binary,
        'probit_cm': probit_cm,
        'probit_sensitivity': probit_sensitivity,
        'probit_specificity': probit_specificity,
        'probit_accuracy': probit_accuracy,
        'annotator_cms': annotator_cms,
        'annotator_reports': annotator_reports,
        'annotator_metrics': annotator_metrics,
        'true_labels': true_labels,
        'annotations': annotations,
        'annotator_cols': annotator_cols
    }


def bootstrap_probit_validation_metrics(
    validation_data,
    prior_alpha: float,
    prior_beta: float,
    n_bootstrap: int = 32,
    n_pattern_samples: int = 4096,
    random_state: int = 1234,
    use_beta_posterior_prior: bool = False,
):
    """
    Run a bootstrap version of the probit model on the validation set.

    This mirrors the production correlated probit bootstrap behavior, with the
    key differences being:
      - Uses the held-out validation data (with ground-truth labels) instead of test data
      - Draws a class prior from Beta(alpha, beta) for each bootstrap
      - Uses Bernoulli draws from the inferred p(z=1 | y) instead of fixed thresholding

    Returns a list of per-bootstrap metric dicts (sensitivity, specificity,
    accuracy, bias, n_positive, n_negative, n_total).
    """
    annotator_cols = [col for col in validation_data.columns if col not in ['cv_text', 'category']]
    if len(annotator_cols) == 0:
        return []

    rng = np.random.default_rng(random_state)
    n_items = len(validation_data)
    metrics = []

    for _ in range(n_bootstrap):
        # Bootstrap resample rows with replacement
        sample_idx = rng.integers(0, n_items, size=n_items)
        sample_df = validation_data.iloc[sample_idx].reset_index(drop=True)

        # Sample class prior; default uses the provided hyperprior directly
        if use_beta_posterior_prior:
            n_pos = int(sample_df['category'].sum())
            n_neg = int(len(sample_df) - n_pos)
            class_prior = rng.beta(prior_alpha + n_pos, prior_beta + n_neg)
        else:
            class_prior = rng.beta(prior_alpha, prior_beta)

        # Fit point estimates on the bootstrap sample
        point_estimates = create_simple_point_estimate(sample_df)

        # Precompute pattern likelihoods for observed patterns (supports missing annotators)
        sample_annotations = sample_df[annotator_cols].values
        unique_patterns = collect_unique_patterns(sample_annotations)
        pattern_probs = precompute_observed_patterns(
            point_estimates['theta'],
            point_estimates['chol_0'],
            point_estimates['chol_1'],
            unique_patterns,
            n_samples=n_pattern_samples,
        )

        # Infer probabilities for each item in the bootstrap sample
        probs = np.zeros(len(sample_df))
        for i, y_new in enumerate(sample_annotations):
            probs[i] = predict_new_item_probit_point_estimate(
                y_new,
                point_estimates['theta'],
                point_estimates['chol_0'],
                point_estimates['chol_1'],
                class_prior=class_prior,
                pattern_probs=pattern_probs,
            )

        # Bernoulli draws instead of hard thresholding
        draws = rng.binomial(1, probs)
        true_labels = sample_df['category'].values

        tp = np.sum((true_labels == 1) & (draws == 1))
        fp = np.sum((true_labels == 0) & (draws == 1))
        tn = np.sum((true_labels == 0) & (draws == 0))
        fn = np.sum((true_labels == 1) & (draws == 0))

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        accuracy = (tp + tn) / len(true_labels) if len(true_labels) > 0 else 0.0

        true_prevalence = true_labels.mean() if len(true_labels) > 0 else 0.0
        predicted_prevalence = draws.mean() if len(draws) > 0 else 0.0
        bias = predicted_prevalence - true_prevalence

        metrics.append(
            {
                'sensitivity': sensitivity,
                'specificity': specificity,
                'accuracy': accuracy,
                'bias': bias,
                'n_positive': int((true_labels == 1).sum()),
                'n_negative': int((true_labels == 0).sum()),
                'n_total': int(len(true_labels)),
            }
        )

    return metrics


def summarize_bootstrap_metrics(metrics, ci_width: float = 0.80):
    """
    Aggregate bootstrap metric draws into mean and quantiles.

    Returns a dict keyed by summary name (mean, q10, q50, q90) with metric dicts.
    """
    if not metrics:
        return {}

    q_low = max(0.0, (1.0 - ci_width) / 2.0)
    q_high = min(1.0, 1.0 - q_low)

    def extract(field):
        return np.array([m[field] for m in metrics], dtype=float)

    summaries = {
        'mean': {
            'sensitivity': float(np.mean(extract('sensitivity'))),
            'specificity': float(np.mean(extract('specificity'))),
            'accuracy': float(np.mean(extract('accuracy'))),
            'bias': float(np.mean(extract('bias'))),
        },
        'q10': {
            'sensitivity': float(np.quantile(extract('sensitivity'), q_low)),
            'specificity': float(np.quantile(extract('specificity'), q_low)),
            'accuracy': float(np.quantile(extract('accuracy'), q_low)),
            'bias': float(np.quantile(extract('bias'), q_low)),
        },
        'q50': {
            'sensitivity': float(np.quantile(extract('sensitivity'), 0.50)),
            'specificity': float(np.quantile(extract('specificity'), 0.50)),
            'accuracy': float(np.quantile(extract('accuracy'), 0.50)),
            'bias': float(np.quantile(extract('bias'), 0.50)),
        },
        'q90': {
            'sensitivity': float(np.quantile(extract('sensitivity'), q_high)),
            'specificity': float(np.quantile(extract('specificity'), q_high)),
            'accuracy': float(np.quantile(extract('accuracy'), q_high)),
            'bias': float(np.quantile(extract('bias'), q_high)),
        },
    }

    return summaries


# Example usage
if __name__ == "__main__":
    # Load validation data and run probit model
    print("Loading validation data...")
    validation_data = pd.read_csv('outputs/dawid_skene_validation_data.csv')
    
    print(f"Validation data shape: {validation_data.shape}")
    print(f"Columns: {list(validation_data.columns)}")
    
    # Create point estimate from validation data
    print("\nCreating point estimate...")
    point_estimates = create_simple_point_estimate(validation_data)
    
    # Run probit validation
    print("\nRunning probit validation...")
    results = run_probit_validation(validation_data, point_estimates, threshold=0.5)
    
    # Test prediction on a single item
    print("\nTesting single item prediction...")
    y_new = np.array([1, 1, 0, 1, 0, 1])  # Mixed annotations
    prob_positive = predict_new_item_probit_point_estimate(
        y_new, 
        point_estimates['theta'], 
        point_estimates['chol_0'], 
        point_estimates['chol_1']
    )
    
    print(f"New item annotations: {y_new}")
    print(f"P(true_label = 1 | annotations) = {prob_positive:.3f}")
    print(f"Predicted label: {'Positive' if prob_positive > 0.5 else 'Negative'}")
