"""
Configuration management for ML Headcount Pipeline.

This module provides centralized configuration using Pydantic for validation
and YAML for human-readable config files.
"""

from pathlib import Path
from typing import Optional, List, Tuple
from pydantic import BaseModel, Field, field_validator
import yaml
import logging

logger = logging.getLogger(__name__)


class KeyBERTExtractionConfig(BaseModel):
    """Configuration for KeyBERT keyword extraction."""
    model_name: str = Field(description="Sentence transformer model name")
    batch_size: int = Field(gt=0, description="Batch size for embeddings")
    max_seq_length: int = Field(gt=0, description="Maximum sequence length")
    top_n: int = Field(gt=0, description="Number of keywords per document")
    ngram_max: int = Field(gt=0, le=10, description="Maximum n-gram size")


class KeyBERTClusteringConfig(BaseModel):
    """Configuration for KeyBERT clustering."""
    min_cluster_size: int = Field(gt=0, description="Minimum cluster size for HDBSCAN")
    n_clusters: Optional[int] = Field(default=None, gt=0, description="Number of clusters for KMeans fallback")


class TFIDFConfig(BaseModel):
    """Configuration for TF-IDF vectorization."""
    ngram_range: List[int] = Field(description="N-gram range as [min, max]")
    max_features: int = Field(gt=0, description="Maximum number of features")
    min_df: int = Field(ge=1, description="Minimum document frequency")
    max_df: float = Field(gt=0, le=1, description="Maximum document frequency")
    
    @field_validator('ngram_range')
    @classmethod
    def validate_ngram_range(cls, v):
        if len(v) != 2:
            raise ValueError("ngram_range must be a list of exactly 2 integers [min, max]")
        if v[0] < 1 or v[1] < v[0]:
            raise ValueError("Invalid ngram_range: min must be >= 1 and max must be >= min")
        return tuple(v)  # Convert to tuple for consistency


class DiscriminativeKeywordsConfig(BaseModel):
    """Configuration for discriminative keywords extraction."""
    min_score: float = Field(ge=0, le=1, description="Minimum discriminative score")
    max_keywords: int = Field(gt=0, description="Maximum keywords per category")


class KeywordFilteringConfig(BaseModel):
    """Configuration for keyword filtering thresholds."""
    strict_threshold: float = Field(ge=0, le=1, description="Threshold for strict keywords")
    broad_threshold: float = Field(ge=0, le=1, description="Threshold for broad keywords")
    
    @field_validator('broad_threshold')
    @classmethod
    def validate_thresholds(cls, v, info):
        if 'strict_threshold' in info.data and v >= info.data['strict_threshold']:
            raise ValueError("broad_threshold must be < strict_threshold")
        return v


class DawidSkeneFilteringConfig(BaseModel):
    """Configuration for Dawid-Skene dataset filtering."""
    max_items: int = Field(gt=0, description="Maximum number of profiles to include in analysis")
    max_companies: int = Field(gt=0, description="Maximum number of companies to include")
    min_profiles: int = Field(ge=0, description="Minimum profiles per company to be included")


class LatentCovarianceDiagnosticConfig(BaseModel):
    """Configuration for latent covariance diagnostic."""
    min_employees: int = Field(default=10, gt=0, description="Minimum employees per company to estimate covariance")


class DataPathsConfig(BaseModel):
    """Configuration for data paths."""
    data_dir: str = Field(description="Data directory path")
    output_dir: str = Field(description="Output directory path")


class ExecutionConfig(BaseModel):
    """Configuration for execution settings."""
    use_remote: bool = Field(default=True, description="Use Modal Labs for expensive operations")
    disable_cache: bool = Field(default=False, description="Disable Hamilton caching")
    enable_telemetry: bool = Field(default=False, description="Enable Hamilton UI telemetry")
    project_id: Optional[str] = Field(default=None, description="Hamilton project ID")
    username: Optional[str] = Field(default=None, description="Username for telemetry")
    dag_name: str = Field(default="ml_headcount_pipeline", description="DAG name for telemetry")
    telemetry_tags: dict = Field(
        default_factory=lambda: {"environment": "DEV", "team": "ML_HEADCOUNT"},
        description="Additional tags for telemetry"
    )
    
    @field_validator('project_id')
    @classmethod
    def validate_project_id_if_telemetry(cls, v, info):
        """Validate that project_id is provided if telemetry is enabled."""
        if info.data.get('enable_telemetry') and not v:
            raise ValueError("project_id is required when enable_telemetry=true")
        return v
    
    # Note: username validator removed - falls back to OS username if None


class LinkedInDatasetsConfig(BaseModel):
    """Configuration for enabling/disabling LinkedIn profile datasets."""
    enable_85k_profiles: bool = Field(default=True, description="Enable 85k profiles dataset (main set)")
    enable_big_consulting: bool = Field(default=False, description="Enable 49k big consulting profiles")
    enable_comparator: bool = Field(default=False, description="Enable 115k comparator profiles")


class PlottingConfig(BaseModel):
    """Configuration for plotting parameters."""
    confidence_interval_width: float = Field(
        default=0.80, 
        gt=0, 
        lt=1, 
        description="Width for confidence intervals (e.g., 0.80 = 10% - 90%)"
    )


class CorrelatedProbitBootstrapConfig(BaseModel):
    """Configuration for correlated probit bootstrap analysis."""
    n_samples: int = Field(default=400, gt=0, description="Number of bootstrap samples per machine")
    n_machines: int = Field(default=1, gt=0, description="Number of machines to use for parallel processing")
    n_cores_per_machine: int = Field(default=64, gt=0, description="Number of cores per machine")
    timeout: int = Field(default=7200, gt=0, description="Timeout in seconds for each machine")
    memory: int = Field(default=32768, gt=0, description="Memory in MB for each machine")
    prior_alpha: float = Field(default=2.0, gt=0, description="Alpha parameter for Beta prior on prevalence")
    prior_beta: float = Field(default=10.0, gt=0, description="Beta parameter for Beta prior on prevalence")


class AnnotatorConfig(BaseModel):
    """Configuration for a single annotator's confusion matrix parameters."""
    sensitivity: float = Field(ge=0, le=1, description="True positive rate (P(Y=1|Z=1))")
    specificity: float = Field(ge=0, le=1, description="True negative rate (P(Y=0|Z=0))")


class CompanyPrevalencesConfig(BaseModel):
    """Configuration for company prevalence distribution using double-truncated lognormal."""
    log_mean: float = Field(description="Mean prevalence in log10 space")
    log_std: float = Field(ge=0, description="Standard deviation in log10 space")
    log_min: float = Field(description="Lower truncation in log10 space (e.g., -3.0 for 1e-3)")
    log_max: float = Field(description="Upper truncation in log10 space (e.g., 0.0 for 1.0)")
    
    @property
    def min_prevalence(self) -> float:
        """Get minimum prevalence in linear space."""
        return 10 ** self.log_min
    
    @property
    def max_prevalence(self) -> float:
        """Get maximum prevalence in linear space."""
        return 10 ** self.log_max


class TestDataConfig(BaseModel):
    """Configuration for fake test data generation."""
    num_profiles: int = Field(gt=0, description="Number of fake test profiles to generate")
    num_companies: int = Field(gt=0, description="Number of fake companies to generate")
    company_prevalences: CompanyPrevalencesConfig = Field(description="Company prevalence distribution parameters")


class ValidationDataConfig(BaseModel):
    """Configuration for fake validation data generation."""
    num_profiles: int = Field(gt=0, description="Number of fake validation profiles to generate")
    observed_positive_rate: float = Field(ge=0, le=1, description="Target overall positive rate in validation data")


class FakeDataConfig(BaseModel):
    """Configuration for fake data generation."""
    enable: bool = Field(default=False, description="Enable fake data generation for model evaluation")
    test_data: TestDataConfig = Field(description="Test data configuration")
    validation_data: ValidationDataConfig = Field(description="Validation data configuration")
    annotators: dict[str, AnnotatorConfig] = Field(description="Annotator confusion matrix parameters")


class PipelineConfig(BaseModel):
    """Complete pipeline configuration."""
    keybert_extraction: KeyBERTExtractionConfig
    keybert_clustering: KeyBERTClusteringConfig
    tfidf: TFIDFConfig
    discriminative_keywords: DiscriminativeKeywordsConfig
    keyword_filtering: KeywordFilteringConfig
    dawid_skene_filtering: DawidSkeneFilteringConfig
    data_paths: DataPathsConfig
    linkedin_datasets: LinkedInDatasetsConfig
    plotting: PlottingConfig
    execution: ExecutionConfig
    correlated_probit_bootstrap: CorrelatedProbitBootstrapConfig
    latent_covariance_diagnostic: LatentCovarianceDiagnosticConfig
    fake_data: Optional[FakeDataConfig] = Field(default=None, description="Fake data configuration")
    
    def to_hamilton_inputs(self) -> dict:
        """
        Convert config to flat dictionary for Hamilton inputs.
        
        Returns:
            Dictionary with all config values as Hamilton input parameters
        """
        inputs = {}
        
        # KeyBERT extraction parameters
        inputs['ke_model_name'] = self.keybert_extraction.model_name
        inputs['ke_batch_size'] = self.keybert_extraction.batch_size
        inputs['ke_max_seq_length'] = self.keybert_extraction.max_seq_length
        inputs['ke_top_n'] = self.keybert_extraction.top_n
        inputs['ke_ngram_max'] = self.keybert_extraction.ngram_max
        
        # KeyBERT clustering parameters
        inputs['kc_min_cluster_size'] = self.keybert_clustering.min_cluster_size
        inputs['kc_n_clusters'] = self.keybert_clustering.n_clusters
        
        # TF-IDF parameters
        inputs['tfidf_ngram_range'] = self.tfidf.ngram_range
        inputs['tfidf_max_features'] = self.tfidf.max_features
        inputs['tfidf_min_df'] = self.tfidf.min_df
        inputs['tfidf_max_df'] = self.tfidf.max_df
        
        # Discriminative keywords parameters
        inputs['dk_min_score'] = self.discriminative_keywords.min_score
        inputs['dk_max_keywords'] = self.discriminative_keywords.max_keywords
        
        # Keyword filtering parameters
        inputs['kf_strict_threshold'] = self.keyword_filtering.strict_threshold
        inputs['kf_broad_threshold'] = self.keyword_filtering.broad_threshold
        
        # Dawid-Skene filtering parameters
        inputs['ds_max_items'] = self.dawid_skene_filtering.max_items
        inputs['ds_max_companies'] = self.dawid_skene_filtering.max_companies
        inputs['ds_min_profiles'] = self.dawid_skene_filtering.min_profiles
        
        # Plotting parameters
        inputs['plot_ci_width'] = self.plotting.confidence_interval_width
        
        # LinkedIn datasets configuration
        inputs['enable_85k_profiles'] = self.linkedin_datasets.enable_85k_profiles
        inputs['enable_big_consulting'] = self.linkedin_datasets.enable_big_consulting
        inputs['enable_comparator'] = self.linkedin_datasets.enable_comparator
        
        # Data paths
        inputs['data_dir'] = self.data_paths.data_dir
        inputs['output_dir'] = self.data_paths.output_dir
        
        # Correlated probit bootstrap parameters
        inputs['correlated_probit_bootstrap_n_samples'] = self.correlated_probit_bootstrap.n_samples
        inputs['correlated_probit_bootstrap_n_machines'] = self.correlated_probit_bootstrap.n_machines
        inputs['correlated_probit_bootstrap_timeout'] = self.correlated_probit_bootstrap.timeout
        inputs['correlated_probit_bootstrap_prior_alpha'] = self.correlated_probit_bootstrap.prior_alpha
        inputs['correlated_probit_bootstrap_prior_beta'] = self.correlated_probit_bootstrap.prior_beta
        
        # Latent covariance diagnostic parameters
        inputs['latent_covariance_diagnostic_min_employees'] = self.latent_covariance_diagnostic.min_employees
        
        # Fake data configuration
        if self.fake_data:
            inputs['fake_data_enable'] = self.fake_data.enable
            # Test data parameters
            inputs['fake_data_test_num_profiles'] = self.fake_data.test_data.num_profiles
            inputs['fake_data_test_num_companies'] = self.fake_data.test_data.num_companies
            inputs['fake_data_test_prevalence_log_mean'] = self.fake_data.test_data.company_prevalences.log_mean
            inputs['fake_data_test_prevalence_log_std'] = self.fake_data.test_data.company_prevalences.log_std
            inputs['fake_data_test_prevalence_log_min'] = self.fake_data.test_data.company_prevalences.log_min
            inputs['fake_data_test_prevalence_log_max'] = self.fake_data.test_data.company_prevalences.log_max
            # Validation data parameters
            inputs['fake_data_validation_num_profiles'] = self.fake_data.validation_data.num_profiles
            inputs['fake_data_validation_observed_positive_rate'] = self.fake_data.validation_data.observed_positive_rate
            # Convert annotators dict to individual parameters
            for annotator_name, annotator_config in self.fake_data.annotators.items():
                inputs[f'fake_data_annotator_{annotator_name}_sensitivity'] = annotator_config.sensitivity
                inputs[f'fake_data_annotator_{annotator_name}_specificity'] = annotator_config.specificity
        else:
            inputs['fake_data_enable'] = False
        
        return inputs


def load_config(config_path: Optional[Path] = None) -> PipelineConfig:
    """
    Load pipeline configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file. If None, uses config/default.yaml
        
    Returns:
        Validated PipelineConfig object
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config validation fails
    """
    if config_path is None:
        # Default to config/default.yaml in project root
        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / "config" / "default.yaml"
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    logger.info(f"Loading configuration from: {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    try:
        config = PipelineConfig(**config_dict)
        logger.info("Configuration loaded and validated successfully")
        return config
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        raise ValueError(f"Invalid configuration: {e}") from e


def get_modal_resources(config: PipelineConfig, function_name: str) -> dict:
    """
    Get Modal resource configuration for a specific function.
    
    Args:
        config: Pipeline configuration
        function_name: Name of the Modal function
        
    Returns:
        Dictionary with resource parameters for Modal decorator
    """
    resource_map = {
        'keybert_extraction': config.modal_resources.keybert_extraction,
        'keybert_clustering': config.modal_resources.keybert_clustering,
    }
    
    if function_name not in resource_map:
        raise ValueError(f"Unknown Modal function: {function_name}")
    
    resources = resource_map[function_name]
    
    result = {
        'cpu': resources.cpu,
        'memory': resources.memory,
        'timeout': resources.timeout,
    }
    
    if resources.gpu:
        result['gpu'] = resources.gpu
    
    return result

