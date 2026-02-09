"""
Modal functions for ML Headcount Pipeline.

This module contains thin wrappers for Modal cloud execution.
All actual logic is implemented in the scripts/ directory.

Modal resources are dynamically configured via .with_options() based on config.
"""

import modal
import logging
from typing import Dict, Any, Optional
import pandas as pd

logger = logging.getLogger(__name__)

# Create Modal app
app = modal.App("ml-headcount")

# Function to download models (no GPU needed)
def download_models():
    """Download and cache models during image build"""
    from sentence_transformers import SentenceTransformer
    
    # Download both models
    SentenceTransformer("all-MiniLM-L6-v2")
    SentenceTransformer("Salesforce/SFR-Embedding-Mistral")
    print("✅ Models downloaded and cached successfully")

# Define the image with pre-downloaded models
image = (
    modal.Image
        .debian_slim(python_version="3.11")
        .uv_sync()
        .run_function(download_models)  # No GPU needed for download
        .add_local_python_source("ml_headcount")
)

# Import KeyBERT functions
from .scripts.text_analysis.keybert_extraction import (
    run_keyword_extraction_only as _run_keyword_extraction_only,
    run_keyword_clustering as _run_keyword_clustering
)

# KeyBERT Modal functions
# Note: Resources hardcoded from config/default.yaml (Modal functions don't support .with_options())
@app.function(
    image=image,
    cpu=4,
    memory=16384,  # 16GB
    timeout=3600,  # 1 hour
    gpu="A100-40GB",
    retries=5,  # Retry up to 5 times on machine failure
    # secrets=[modal.Secret.from_name("huggingface")]  # Only needed for private models
)
def run_keybert_extraction(
    cv_data,
    *,
    model_name: str,
    batch_size: int,
    max_seq_length: int,
    top_n: int,
    ngram_range: tuple
) -> pd.DataFrame:
    """
    Modal wrapper for KeyBERT keyword extraction.
    
    Args:
        cv_data: DataFrame with 'category' and 'processed_text' columns
        model_name: Sentence transformer model name
        batch_size: Batch size for embeddings
        max_seq_length: Maximum sequence length
        top_n: Number of top keywords per document
        ngram_range: Range of n-grams to extract
        
    Returns:
        DataFrame containing keyword extraction results with scores
    """
    import torch
    
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    if device == "cuda":
        logger.info(f"CUDA device: {torch.cuda.get_device_name()}")
        logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Call the core function from scripts
    return _run_keyword_extraction_only(
        df=cv_data,
        model_name=model_name,
        device=device,
        batch_size=batch_size,
        max_seq_length=max_seq_length,
        top_n=top_n,
        ngram_range=ngram_range
    )

# Keyword clustering Modal functions
# Note: Resources hardcoded from config/default.yaml (Modal functions don't support .with_options())
@app.function(
    image=image,
    cpu=4,
    memory=16384,  # 16GB
    timeout=1800,  # 30 minutes
    gpu="A100-40GB",
    retries=5,  # Retry up to 5 times on machine failure
)
def run_keyword_clustering(
    discriminative_keywords: Dict[str, Dict[str, float]],
    *,
    min_cluster_size: int,
    n_clusters: Optional[int],
    model_name: str
) -> Dict[str, Any]:
    """
    Modal wrapper for keyword clustering.
    
    Args:
        discriminative_keywords: Dictionary mapping categories to keyword scores
        min_cluster_size: Minimum cluster size for HDBSCAN
        n_clusters: Number of clusters for KMeans fallback
        model_name: Name of the sentence transformer model to use
        
    Returns:
        Dictionary containing clustering results
    """
    # Initialize the sentence model inside the Modal function
    from sentence_transformers import SentenceTransformer
    import torch
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sentence_model = SentenceTransformer(model_name, device=device)
    
    # Call the core function from scripts
    return _run_keyword_clustering(
        discriminative_keywords=discriminative_keywords,
        sentence_model=sentence_model,
        min_cluster_size=min_cluster_size,
        n_clusters=n_clusters
    )

# Bootstrap Modal functions
# Note: Resources hardcoded (Modal functions don't support .with_options())
NUM_CORES_PER_MACHINE = 64  # Global constant for Modal decorator and process pool
BOOTSTRAP_MEMORY_MB = 65536  # 64GB - hardcoded memory for bootstrap functions
# NUM_CORES_PER_MACHINE = 1  # Global constant for Modal decorator and process pool
# BOOTSTRAP_MEMORY_MB = 2048  # 2GB - reduced memory for bootstrap functions

@app.function(
    image=image,
    cpu=NUM_CORES_PER_MACHINE,  # Use global constant
    memory=BOOTSTRAP_MEMORY_MB,  # Use hardcoded memory constant
    timeout=7200,  # 2 hours
    retries=5,  # Retry up to 3 times on machine failure
)
def run_correlated_probit_bootstrap_modal(
    test_data_dict: dict,
    validation_data_dict: dict,
    start_iter: int,
    n_samples: int,
    machine_idx: int,
    n_machines: int,
    n_cores_per_machine: int,
    prior_alpha: float,
    prior_beta: float,
    company_database_dict: dict = None,
    companies_needing_synthetic_dict: dict = None,
    tetrachoric_corr_list: list = None,
    use_synthetic_only: bool = False,
    org_type: str = "consulting"
):
    """
    Modal wrapper for correlated probit bootstrap analysis.
    
    Args:
        test_data_dict: Test data as dictionary for transfer
        validation_data_dict: Validation data as dictionary for transfer
        start_iter: Starting iteration number for this machine
        n_samples: Number of bootstrap samples to run on this machine
        machine_idx: Index of this machine (0-based)
        n_machines: Total number of machines
        n_cores_per_machine: Number of cores per machine for parallel processing
        prior_alpha: Alpha parameter for Beta prior on prevalence
        prior_beta: Beta parameter for Beta prior on prevalence
        company_database_dict: Company database as dictionary (optional, for synthetic data)
        companies_needing_synthetic_dict: Companies needing synthetic data as dictionary (optional)
        tetrachoric_corr_list: Tetrachoric correlation matrix as list (optional)
        org_type: Type of organizations ("consulting", "comparator_ml", "comparator_non_ml")
        
    Returns:
        Dictionary with company-level counts from this machine
    """
    from ml_headcount.correlated_probit_bootstrap import _run_single_machine_bootstrap
    
    return _run_single_machine_bootstrap(
        test_data_dict=test_data_dict,
        validation_data_dict=validation_data_dict,
        start_iter=start_iter,
        n_samples=n_samples,
        machine_idx=machine_idx,
        n_machines=n_machines,
        n_cores_per_machine=n_cores_per_machine,
        prior_alpha=prior_alpha,
        prior_beta=prior_beta,
        company_database_dict=company_database_dict,
        companies_needing_synthetic_dict=companies_needing_synthetic_dict,
        tetrachoric_corr_list=tetrachoric_corr_list,
        use_synthetic_only=use_synthetic_only,
        org_type=org_type
    )


# ---------------------------------------------------------------------------
# Annotator metrics bootstrap (Modal, single machine, 64 cores)
# ---------------------------------------------------------------------------


@app.function(
    image=image,
    cpu=NUM_CORES_PER_MACHINE,
    memory=BOOTSTRAP_MEMORY_MB,
    timeout=3600,
    retries=5,  # Retry up to 3 times on machine failure
)
def run_annotator_bootstrap_modal(
    validation_data_dict: dict,
    prior_alpha: float,
    prior_beta: float,
    n_bootstrap: int,
    n_pattern_samples: int,
    ci_width: float,
    use_beta_posterior_prior: bool = False,
    n_cores_per_machine: int = NUM_CORES_PER_MACHINE,
):
    """
    Run validation-side probit bootstrap on a single Modal machine using multiple cores.

    Returns summaries (mean/q10/q50/q90) to keep transfer minimal.
    """
    import pandas as pd
    import numpy as np
    from concurrent.futures import ProcessPoolExecutor
    from math import ceil
    from ml_headcount.probit_dawid_skene_inference import (
        bootstrap_probit_validation_metrics,
        summarize_bootstrap_metrics,
    )

    validation_df = pd.DataFrame(validation_data_dict)

    # Split bootstrap iterations into roughly equal chunks across cores
    chunk_size = max(1, ceil(n_bootstrap / n_cores_per_machine))
    seeds = np.arange(0, n_bootstrap, chunk_size)
    tasks = []
    with ProcessPoolExecutor(max_workers=n_cores_per_machine) as pool:
        for seed in seeds:
            this_n = min(chunk_size, n_bootstrap - seed)
            if this_n <= 0:
                continue
            tasks.append(
                pool.submit(
                    bootstrap_probit_validation_metrics,
                    validation_df,
                    prior_alpha,
                    prior_beta,
                    this_n,
                    n_pattern_samples,
                    int(seed + 1234),
                    use_beta_posterior_prior,
                )
            )

        metrics = []
        for t in tasks:
            metrics.extend(t.result())

    summaries = summarize_bootstrap_metrics(metrics, ci_width=ci_width)
    return summaries