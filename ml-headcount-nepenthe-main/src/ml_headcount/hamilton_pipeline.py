"""
Fixed Hamilton-based ML Headcount Pipeline

This module implements the main Hamilton pipeline that orchestrates
all data processing steps using Hamilton's function-based approach,
with fixes for recursion issues.
"""

from hamilton import driver
from hamilton.function_modifiers import config
from hamilton import graph_types
from hamilton_sdk import adapters
from typing import Dict, Any, Optional, List, Tuple
import logging
from pathlib import Path
import sys

logger = logging.getLogger(__name__)

def custom_style(
    *, node: graph_types.HamiltonNode, node_class: str
) -> Tuple[dict, Optional[str], Optional[str]]:
    """Custom style function for the visualization.

    :param node: node that Apache Hamilton is styling.
    :param node_class: class used to style the default visualization
    :return: a triple of (style, node_class, legend_name)
    """
    
    # Make nodes ending in _raw or _schema_validator less noticeable
    if node.name.endswith('_raw') or node.name.endswith('_schema_validator') or node.name.endswith('.loader'):
        style = {
            "fillcolor": "lightgray",
            "color": "gray",
            "fontcolor": "black",
            "style": "filled,rounded"
        }
        return (style, node_class, "Hamilton decorator nodes")
    
    # Keep original styling for other nodes
    if node.type in [float, int]:
        style = {"fillcolor": "aquamarine"}
        return (style, node_class, "numbers")
    else:
        style = {}
        return (style, node_class, None)

class HamiltonMLHeadcountPipeline:
    """
    Hamilton-based ML Headcount Pipeline.
    
    This class orchestrates the complete pipeline using Hamilton's
    function-based approach with automatic dependency resolution.
    """
    
    def __init__(
        self, 
        *,
        config_dict: Dict[str, Any],
        selected_subgraphs: Optional[List[str]] = None, 
        use_remote: bool = True, 
        disable_cache: bool = False,
        recompute: Optional[List[str]] = None,
        # Telemetry parameters
        enable_telemetry: bool = False,
        project_id: Optional[str] = None,
        username: Optional[str] = None,
        dag_name: Optional[str] = None,
        telemetry_tags: Optional[Dict[str, str]] = None
    ):
        """
        Initialize the Hamilton pipeline.
        
        Args:
            config_dict: Dictionary of all configuration parameters from config YAML
            selected_subgraphs: List of subgraphs to run (None means run all available)
            use_remote: Whether to use Modal Labs for expensive operations (default: True)
            disable_cache: Whether to disable caching entirely for this run (default: False)
            recompute: List of function names to force recomputation (default: None)
            enable_telemetry: Whether to enable Hamilton telemetry tracking (default: False)
            project_id: Project ID for Hamilton telemetry (required if enable_telemetry=True)
            username: Username/email for Hamilton telemetry (defaults to OS username if None)
            dag_name: DAG name for Hamilton telemetry (default: "ml_headcount_pipeline")
            telemetry_tags: Additional tags for Hamilton telemetry (default: {"environment": "DEV", "team": "ML_HEADCOUNT"})
        """
        # Store configuration
        self.config_dict = config_dict
        self.data_dir = config_dict.get('data_dir', 'data')
        self.output_dir = config_dict.get('output_dir', 'outputs')
        self.selected_subgraphs = selected_subgraphs
        self.use_remote = use_remote
        self.disable_cache = disable_cache
        self.recompute = recompute
        
        # Telemetry configuration
        self.enable_telemetry = enable_telemetry
        self.project_id = project_id
        # Fall back to OS username if username is None
        if enable_telemetry and not username:
            import getpass
            self.username = getpass.getuser()
            logger.info(f"No username provided for telemetry, using OS username: {self.username}")
        else:
            self.username = username
        self.dag_name = dag_name or "ml_headcount_pipeline"
        self.telemetry_tags = telemetry_tags or {"environment": "DEV", "team": "ML_HEADCOUNT"}
        
        # Validate telemetry configuration
        if self.enable_telemetry:
            if not self.project_id:
                raise ValueError("project_id is required when enable_telemetry=True")
        
        # Create output directory if it doesn't exist
        Path(self.output_dir).mkdir(exist_ok=True)
        
        # Import modules containing Hamilton functions
        from . import hamilton_dataloaders
        from . import hamilton_plots
        from . import hamilton_processors
        
        # Set execution mode for unified processors
        from .execution_config import set_execution_mode
        set_execution_mode(use_remote=use_remote)
        
        if use_remote:
            logger.info("Using unified processors with remote execution for expensive operations")
        else:
            logger.info("Using unified processors with local execution for all operations")
        
        # Initialize Hamilton driver with modules - using a safer approach
        try:
            # Increase recursion limit to handle complex dependencies
            sys.setrecursionlimit(5000)
            
            # Initialize driver with safe modules and optional caching
            # Import INPUT_DATA_CONFIG and OUTPUT_DATA_CONFIG to use as single source of truth
            from .hamilton_dataloaders import INPUT_DATA_CONFIG, OUTPUT_DATA_CONFIG
            
            # Static config - file paths and other pipeline-wide settings
            # Expand relative paths from configs by prepending data_dir/output_dir
            static_config = {
                "output_dir": self.output_dir  # Keep for backward compatibility
            }
            
            # Add all input file paths from INPUT_DATA_CONFIG (expanded with data_dir)
            for data_type, config in INPUT_DATA_CONFIG.items():
                # Expand file_path by prepending data_dir
                static_config[f"{data_type}_file_path"] = f"{self.data_dir}/{config['file_path']}"
                
                # Add sheet name if present
                if "sheet_name" in config:
                    static_config[f"{data_type}_sheet_name"] = config["sheet_name"]
            
            # Add all output file paths from OUTPUT_DATA_CONFIG (expanded with output_dir)
            for output_key, relative_path in OUTPUT_DATA_CONFIG.items():
                static_config[output_key] = f"{self.output_dir}/{relative_path}"
            
            # Add config parameters needed for @config.when decorators
            # These must be in with_config(), not just in inputs to execute()
            static_config['fake_data_enable'] = config_dict.get('fake_data_enable', False)
            static_config['correlated_probit_bootstrap_enable'] = config_dict.get('correlated_probit_bootstrap_enable', False)
            
            # Import correlated probit bootstrap module
            from . import correlated_probit_bootstrap
            from . import annotator_validation_plots
            from . import latent_covariance_diagnostic
            
            # Build driver with unified processors
            driver_builder = (
                driver.Builder()
                .with_modules(hamilton_dataloaders, hamilton_processors, hamilton_plots, correlated_probit_bootstrap, annotator_validation_plots, latent_covariance_diagnostic)
                .with_config(static_config)
            )
            
            # Add telemetry if enabled
            if self.enable_telemetry:
                tracker = adapters.HamiltonTracker(
                    project_id=self.project_id,
                    username=self.username,
                    dag_name=self.dag_name,
                    tags=self.telemetry_tags
                )
                driver_builder = driver_builder.with_adapters(tracker)
                logger.info(f"Hamilton telemetry enabled for project: {self.project_id}, user: {self.username}")
            
            if not disable_cache:
                # Enable caching for all operations
                cache_dir = Path(self.output_dir) / ".hamilton_cache"
                cache_dir.mkdir(exist_ok=True)
                
                # Configure cache with recompute parameter if specified
                if recompute:
                    driver_builder = driver_builder.with_cache(path=str(cache_dir), recompute=recompute)
                    logger.info(f"Initialized Hamilton driver with caching enabled at {cache_dir}, recomputing: {', '.join(recompute)}")
                else:
                    driver_builder = driver_builder.with_cache(path=str(cache_dir))
                    logger.info(f"Initialized Hamilton driver with caching enabled at {cache_dir}")
            else:
                logger.info("Initialized Hamilton driver with caching disabled")
            
            self.dr = driver_builder.build()
        except Exception as e:
            logger.error(f"Failed to initialize Hamilton driver: {str(e)}")
            raise
        
        # Define subgraph mappings
        self.subgraph_mappings = {
            "cv_analysis": {
                "outputs": [
                    "preprocessed_text",
                    "cv_keyword_frequencies",
                    "keyword_extraction_results",
                    "keyword_filter_data",
                    "filtered_keywords_list",
                    "keyword_extraction_results_filtered",
                    "discriminative_keywords", 
                    "keyword_lists",
                    "validation_cvs_scored",
                    "keyword_extraction_report",
                    "confusion_matrix_metrics",
                    # Auto-generated save.* outputs from @save_to decorators
                    "save.preprocessed_text",
                    "save.cv_keyword_frequencies",
                    "save.keyword_extraction_results",
                    "save.discriminative_keywords",
                    "save.keyword_lists",
                    "save.keyword_extraction_report",
                    "save.filtered_keywords_list",
                    "save.keyword_extraction_results_filtered",
                    "save.confusion_matrix_metrics",
                    "save_keywords_json",
                    "save_keyword_extraction_report_json"
                ],
                "required_inputs": ["validation_cvs_file_path", "validation_cvs_sheet_name"]
            },
            "clustering": {
                "outputs": [
                    "clustering_results",
                    "keyword_visualization",
                    # Auto-generated save.* outputs from @save_to decorators
                    "save.clustering_results",
                    "save_cv_keyword_clusters_json"
                ],
                "required_inputs": ["validation_cvs_file_path", "validation_cvs_sheet_name"]
            },
            "affiliation": {
                "outputs": [
                    "non_academic_affiliations",
                    "cleaned_affiliations",
                    # Auto-generated save.* outputs from @save_to decorators
                    "save.non_academic_affiliations",
                    "save.cleaned_affiliations"
                ],
                "required_inputs": ["affiliation_data_file_path"]
            },
            "probit_bootstrap": {
                "outputs": [
                    # Reuse existing data from dawid_skene subgraph
                    "dawid_skene_validation_data",
                    "dawid_skene_test_data",
                    "dawid_skene_test_data_main",
                    "dawid_skene_test_data_comparator_ml",
                    "dawid_skene_test_data_comparator_non_ml",
                    # Combined employee-level data outputs (NEW)
                    "real_employee_level_data_all",
                    "synthetic_employee_level_data_all",
                    # Test keyword filter correlation matrix (for synthetic data generation)
                    "test_keyword_filter_correlation_matrix",
                    # Bootstrap analysis - raw results (cached)
                    "correlated_probit_bootstrap_raw_results",
                    "correlated_probit_bootstrap_raw_results_main",
                    "correlated_probit_bootstrap_raw_results_comparator_ml",
                    "correlated_probit_bootstrap_raw_results_comparator_non_ml",
                    # Synthetic-only bootstrap (for all companies)
                    "correlated_probit_bootstrap_raw_results_synthetic",
                    "correlated_probit_bootstrap_results_synthetic",
                    # Bootstrap analysis - processed outputs (not cached)
                    "correlated_probit_bootstrap_results",
                    "correlated_probit_bootstrap_results_main",
                    "correlated_probit_bootstrap_results_comparator_ml",
                    "correlated_probit_bootstrap_results_comparator_non_ml",
                    "correlated_probit_bootstrap_distributions",
                    "correlated_probit_bootstrap_plots_with_annotators",
                    "correlated_probit_bootstrap_plots_with_annotators_main",
                    "correlated_probit_bootstrap_plots_with_annotators_comparator_ml",
                    "correlated_probit_bootstrap_plots_with_annotators_comparator_non_ml",
                    # New visualization outputs (now depend on summary DataFrame)
                    "probit_bootstrap_talent_landscape_plot",
                    "probit_bootstrap_talent_landscape_plot_main",
                    "probit_bootstrap_talent_landscape_plot_comparator_ml",
                    "probit_bootstrap_talent_landscape_plot_comparator_non_ml",
                    "probit_bootstrap_per_organization_estimates_plot",
                    "probit_bootstrap_per_organization_estimates_plot_main",
                    "probit_bootstrap_per_organization_estimates_plot_comparator_ml",
                    "probit_bootstrap_per_organization_estimates_plot_comparator_non_ml",
                    "probit_bootstrap_prior_distribution_plot",
                    "probit_bootstrap_prior_distribution_plot_main",
                    "probit_bootstrap_prior_distribution_plot_comparator_ml",
                    "probit_bootstrap_prior_distribution_plot_comparator_non_ml",
                    # Diagnostic plots
                    "keyword_annotator_prevalence_plot",
                    "empirical_keyword_correlations_pairplot",
                    "real_vs_synthetic_scatter_plot",
                    "synthetic_real_ratio_distributions_plot",
                    # Final output tables
                    "probit_results_main_orgs",
                    "probit_results_comparator_ml_orgs",
                    "probit_results_comparator_non_ml_orgs",
                    # Final joined results (company DB + probit)
                    "final_results_main_orgs",
                    "final_results_core_orgs",
                    "final_results_comparator_ml",
                    "final_results_comparator_non_ml",
                    "final_results_all",
                    # Annotator validation plots
                    "annotator_metrics",
                    "annotator_metrics_table",
                    "annotator_performance_sens_vs_spec",
                    "bias_vs_accuracy",
                    "annotator_bias_analysis",
                    "empirical_correlations_by_class",
                    # Real vs Synthetic comparison
                    "real_vs_synthetic_comparison_plot",
                    "real_vs_synthetic_per_company_plot",
                    # Auto-generated save.* outputs from @save_to decorators
                    "save.dawid_skene_validation_data",
                    "save.dawid_skene_test_data",
                    "save.dawid_skene_test_data_main",
                    "save.dawid_skene_test_data_comparator_ml",
                    "save.dawid_skene_test_data_comparator_non_ml",
                    "save.real_employee_level_data_all",
                    "save.synthetic_employee_level_data_all",
                    "save.annotator_metrics_table",
                    "save.correlated_probit_bootstrap_results",
                    "save.correlated_probit_bootstrap_results_main",
                    "save.correlated_probit_bootstrap_results_comparator_ml",
                    "save.correlated_probit_bootstrap_results_comparator_non_ml",
                    "save.correlated_probit_bootstrap_results_synthetic",
                    "save.probit_results_main_orgs",
                    "save.probit_results_comparator_ml_orgs",
                    "save.probit_results_comparator_non_ml_orgs",
                    "save.final_results_main_orgs",
                    "save.final_results_core_orgs",
                    "save.final_results_comparator_ml",
                    "save.final_results_comparator_non_ml",
                    "save.final_results_all"
                ],
                "required_inputs": [
                    "data_dir", 
                    "company_database_complete_file_path", 
                    "company_database_complete_sheet_name",
                    "company_database_complete",
                    "correlated_probit_bootstrap_distributions_output_path",
                    "correlated_probit_bootstrap_plots_output_path",
                    "correlated_probit_bootstrap_plots_with_annotators_output_path",
                    "correlated_probit_bootstrap_plots_with_annotators_main_output_path",
                    "correlated_probit_bootstrap_plots_with_annotators_comparator_ml_output_path",
                    "correlated_probit_bootstrap_plots_with_annotators_comparator_non_ml_output_path",
                    "probit_bootstrap_talent_landscape_plot_output_path",
                    "probit_bootstrap_talent_landscape_plot_main_output_path",
                    "probit_bootstrap_talent_landscape_plot_comparator_ml_output_path",
                    "probit_bootstrap_talent_landscape_plot_comparator_non_ml_output_path",
                    "probit_bootstrap_per_organization_estimates_plot_output_path",
                    "probit_bootstrap_per_organization_estimates_plot_main_output_path",
                    "probit_bootstrap_per_organization_estimates_plot_comparator_ml_output_path",
                    "probit_bootstrap_per_organization_estimates_plot_comparator_non_ml_output_path",
                    "probit_bootstrap_prior_distribution_plot_output_path",
                    "probit_bootstrap_prior_distribution_plot_main_output_path",
                    "probit_bootstrap_prior_distribution_plot_comparator_ml_output_path",
                    "probit_bootstrap_prior_distribution_plot_comparator_non_ml_output_path",
                    "annotator_metrics_output_path",
                    "annotator_performance_sens_vs_spec_output_path",
                    "bias_vs_accuracy_output_path",
                    "annotator_bias_analysis_output_path",
                    "empirical_correlations_by_class_output_path"
                ]
            },
            "annotator_metrics": {
                "outputs": [
                    # Core metrics and plots
                    "annotator_metrics",
                    "annotator_metrics_table",
                    "annotator_performance_sens_vs_spec",
                    "bias_vs_accuracy",
                    # Auto-generated save.* outputs from @save_to decorators
                    "save.annotator_metrics_table",
                ],
                "required_inputs": [
                    "validation_cvs_file_path",
                    "validation_cvs_sheet_name",
                    "annotator_metrics_output_path",
                    "annotator_performance_sens_vs_spec_output_path",
                    "bias_vs_accuracy_output_path",
                    "correlated_probit_bootstrap_prior_alpha",
                    "correlated_probit_bootstrap_prior_beta",
                    "correlated_probit_bootstrap_n_samples",
                    "correlated_probit_bootstrap_n_machines",
                    "plot_ci_width",
                ],
            },
            "log_debias": {
                "outputs": [
                    # Reuse existing data from dawid_skene subgraph
                    "dawid_skene_test_data",
                    "dawid_skene_test_data_main",
                    "dawid_skene_test_data_comparator_ml",
                    "dawid_skene_test_data_comparator_non_ml",
                    # Dependencies for probit bootstrap (needed for combined plot)
                    "dawid_skene_validation_data",
                    "test_keyword_filter_correlation_matrix",
                    "companies_needing_synthetic_data",
                    # Log-debiasing analysis
                    "log_debias_company_aggregates",
                    "log_debias_company_aggregates_main",
                    "log_debias_company_aggregates_comparator_ml",
                    "log_debias_company_aggregates_comparator_non_ml",
                    "log_debias_results",
                    "log_debias_results_main",
                    "log_debias_results_comparator_ml",
                    "log_debias_results_comparator_non_ml",
                    "log_debias_summary",
                    "log_debias_plots",
                    "log_debias_plots_main",
                    "log_debias_plots_comparator_ml",
                    "log_debias_plots_comparator_non_ml",
                    "log_debias_uncertainty_plots",
                    "log_debias_uncertainty_plots_main",
                    "log_debias_uncertainty_plots_comparator_ml",
                    "log_debias_uncertainty_plots_comparator_non_ml",
                    # Dependencies for probit bootstrap results (needed for combined plot)
                    "test_keyword_filter_correlation_matrix",
                    "companies_needing_synthetic_data",
                    "correlated_probit_bootstrap_raw_results",
                    "correlated_probit_bootstrap_raw_results_synthetic",
                    "correlated_probit_bootstrap_results_synthetic",
                    # Combined comparison plot (requires probit results)
                    "correlated_probit_bootstrap_results",
                    "correlated_probit_bootstrap_results_main",
                    "correlated_probit_bootstrap_results_comparator_ml",
                    "correlated_probit_bootstrap_results_comparator_non_ml",
                    "final_results_main_orgs",
                    "combined_estimates_comparison_plot",
                    "combined_estimates_comparison_plot_main",
                    "combined_estimates_comparison_plot_comparator_ml",
                    "combined_estimates_comparison_plot_comparator_non_ml",
                    # Final output tables
                    "log_debias_results_main_orgs",
                    "log_debias_results_comparator_ml_orgs",
                    "log_debias_results_comparator_non_ml_orgs",
                    # Auto-generated save.* outputs from @save_to decorators
                    "save.log_debias_company_aggregates",
                    "save.log_debias_company_aggregates_main",
                    "save.log_debias_company_aggregates_comparator_ml",
                    "save.log_debias_company_aggregates_comparator_non_ml",
                    "save.log_debias_summary",
                    "save.log_debias_all_orgs",
                    "save.log_debias_orgs_ml",
                    "save.log_debias_orgs_talent_dense",
                    "save.log_debias_orgs_stage5_work_trial",
                    "save.log_debias_orgs_enterprise_500ml_0p5pct",
                    "save.log_debias_orgs_midscale_50ml_1pct",
                    "save.log_debias_orgs_boutique_10ml_5pct",
                    "save.log_debias_orgs_stage5_work_trial_recommended",
                    "save.log_debias_results_main_orgs",
                    "save.log_debias_results_comparator_ml_orgs",
                    "save.log_debias_results_comparator_non_ml_orgs"
                ],
                "required_inputs": [
                    "data_dir",
                    "log_debias_plots_output_path",
                    "log_debias_plots_main_output_path",
                    "log_debias_plots_comparator_ml_output_path",
                    "log_debias_plots_comparator_non_ml_output_path",
                    "log_debias_uncertainty_plots_output_path",
                    "log_debias_uncertainty_plots_main_output_path",
                    "log_debias_uncertainty_plots_comparator_ml_output_path",
                    "log_debias_uncertainty_plots_comparator_non_ml_output_path",
                    "combined_estimates_comparison_plot_output_path",
                    "combined_estimates_comparison_plot_main_output_path",
                    "combined_estimates_comparison_plot_comparator_ml_output_path",
                    "combined_estimates_comparison_plot_comparator_non_ml_output_path"
                ]
            },
            "synthetic_diagnostic": {
                "outputs": [
                    "validation_covariance",
                    "company_covariances",
                    "company_sizes_for_diagnostic",
                    "latent_covariance_parallel_coordinates_data",
                    "latent_covariance_parallel_coordinates_plot",
                    "latent_covariance_diagnostic_summary",
                    "validation_covariance_matrix",
                    "company_covariance_matrices",
                    # Auto-generated save.* outputs from @save_to decorators
                    "save.latent_covariance_parallel_coordinates_data",
                    "save.latent_covariance_diagnostic_summary",
                    "save.validation_covariance_matrix",
                    "save.company_covariance_matrices"
                ],
                "required_inputs": [
                    "dawid_skene_validation_data",
                    "dawid_skene_test_data",
                    "latent_covariance_parallel_plot_output_path",
                    "latent_covariance_parallel_coordinates_data_output_path",
                    "latent_covariance_diagnostic_summary_output_path",
                    "validation_covariance_matrix_output_path",
                    "company_covariance_matrices_output_path",
                    "latent_covariance_diagnostic_min_employees"
                ]
            },
        }
        
        logger.info(f"Hamilton ML Headcount Pipeline initialized (remote: {use_remote})")
        if selected_subgraphs:
            logger.info(f"Selected subgraphs: {selected_subgraphs}")
        else:
            logger.info("Running all available subgraphs")
    
    
    def run(self, outputs: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run pipeline with specified outputs or all available subgraphs.
        
        Args:
            outputs: List of specific outputs to compute. If None, runs all available subgraphs.
        
        Returns:
            Dictionary containing pipeline results
        """
        logger.info("Running Hamilton pipeline")
        
        try:
            # Note: Data file validation now handled directly in Hamilton dataloaders using source('data_dir')
            logger.info(f"Data directory: {self.data_dir}")
            
            # Prepare inputs - all config parameters are inputs
            # BUT exclude data_dir and output_dir since paths are already expanded in static_config
            inputs = self.config_dict.copy()
            inputs.pop('data_dir', None)  # Removed - paths already expanded
            inputs.pop('output_dir', None)  # Removed - paths already expanded
            
            # Determine outputs to compute
            if outputs is None:
                outputs = self._get_default_outputs()
            
            if not outputs:
                logger.warning("No outputs specified for execution")
                return {}
            
            # Remove config parameters from inputs (they're already in with_config)
            # Hamilton requires config and inputs to be disjoint
            config_params = ['fake_data_enable', 'correlated_probit_bootstrap_enable']
            for param in config_params:
                inputs.pop(param, None)  # Remove if present
            
            # Log what we're about to execute
            logger.info(f"Executing Hamilton pipeline with outputs: {outputs}")
            logger.info(f"Using inputs: {inputs}")
            
            # Execute pipeline with Modal app context if using remote functions
            if self.use_remote:
                logger.info("Using remote execution - wrapping with Modal app context")
                import modal
                from .modal_functions import app as modal_app
                
                with modal.enable_output():
                    with modal_app.run():
                        result = self.dr.execute(outputs, inputs=inputs)
            else:
                # Execute pipeline locally
                result = self.dr.execute(outputs, inputs=inputs)
            
            logger.info("Pipeline completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise
    
    def _get_default_outputs(self) -> List[str]:
        """Get default outputs based on selected subgraphs or all available."""
        outputs = []
        
        if self.selected_subgraphs:
            # Run only selected subgraphs
            for subgraph in self.selected_subgraphs:
                if subgraph in self.subgraph_mappings:
                    subgraph_outputs = self.subgraph_mappings[subgraph]["outputs"]
                    outputs.extend(subgraph_outputs)
                    logger.info(f"Added subgraph '{subgraph}' outputs: {subgraph_outputs}")
                else:
                    logger.warning(f"Unknown subgraph: {subgraph}")
        else:
            # Run all available subgraphs - let Hamilton handle missing files gracefully
            all_subgraph_outputs = []
            for subgraph_info in self.subgraph_mappings.values():
                all_subgraph_outputs.extend(subgraph_info["outputs"])
            outputs = all_subgraph_outputs
            logger.info("Running all available subgraphs")
        
        return outputs
    
    def visualize_pipeline(self, output_path: Optional[str] = None, save_dot: bool = False):
        """
        Visualize the Hamilton DAG.
        
        Args:
            output_path: Optional path to save the visualization (supports .png, .svg, .pdf, .dot formats)
            save_dot: If True, also save the raw DOT file alongside the rendered image
        """
        logger.info("Visualizing Hamilton pipeline DAG")
        
        try:
            # Determine which functions to visualize based on selected subgraphs
            if self.selected_subgraphs:
                # Get the output functions for selected subgraphs
                target_functions = []
                for subgraph in self.selected_subgraphs:
                    if subgraph in self.subgraph_mappings:
                        subgraph_outputs = self.subgraph_mappings[subgraph]["outputs"]
                        target_functions.extend(subgraph_outputs)
                        logger.info(f"Added {len(subgraph_outputs)} functions from subgraph '{subgraph}': {subgraph_outputs}")
                    else:
                        logger.warning(f"Unknown subgraph: {subgraph}")
                
                if not target_functions:
                    logger.warning("No valid subgraph outputs found, showing all functions")
                    target_functions = None
                else:
                    logger.info(f"Visualizing upstream of {len(target_functions)} target functions: {target_functions}")
            else:
                target_functions = None
                logger.info("No subgraphs selected, showing all functions")
            
            if output_path:
                # For file output, use appropriate display method
                if target_functions:
                    # Show only functions upstream of the selected subgraph outputs
                    self.dr.display_upstream_of(
                        *target_functions,
                        output_file_path=output_path, 
                        keep_dot=save_dot,
                        custom_style_function=custom_style,

                    )
                else:
                    # Show all functions
                    self.dr.display_all_functions(
                        output_file_path=output_path, 
                        keep_dot=save_dot,
                        custom_style_function=custom_style,
                    )
                logger.info(f"Pipeline visualization saved to {output_path}")
                
                if save_dot and not output_path.endswith('.dot'):
                    # Save DOT file alongside the rendered image
                    from pathlib import Path
                    dot_path = Path(output_path).with_suffix('.dot')
                    if target_functions:
                        self.dr.display_upstream_of(
                            *target_functions,
                            output_file_path=str(dot_path), 
                            keep_dot=True,
                            custom_style_function=custom_style
                        )
                    else:
                        self.dr.display_all_functions(
                            output_file_path=str(dot_path), 
                            keep_dot=True,
                            custom_style_function=custom_style
                        )
                    logger.info(f"DOT file saved to {dot_path}")
            else:
                # For interactive display, use appropriate display method
                if target_functions:
                    return self.dr.display_upstream_of(*target_functions, custom_style_function=custom_style)
                else:
                    return self.dr.display_all_functions(custom_style_function=custom_style)
        except Exception as e:
            logger.error(f"Failed to visualize pipeline: {str(e)}")
            raise
    
    def get_available_functions(self) -> List[str]:
        """
        Get list of available Hamilton functions.
        
        Returns:
            List of available function names
        """
        return list(self.dr.graph.nodes.keys())
    
    def get_pipeline_dependencies(self) -> Dict[str, List[str]]:
        """
        Get pipeline dependency graph.
        
        Returns:
            Dictionary mapping functions to their dependencies
        """
        try:
            return self.dr.graph.get_dependencies()
        except AttributeError:
            # Fallback for different Hamilton versions
            return {}
    
    def clear_cache(self, function_names: Optional[List[str]] = None):
        """
        Clear Hamilton cache for specific functions or all functions.
        
        Args:
            function_names: List of function names to clear cache for. If None, clears all caches.
        """
        if self.disable_cache:
            logger.warning("Cache is disabled for this pipeline instance")
            return
            
        try:
            if function_names is None:
                # Clear all caches
                self.dr.cache.result_store.delete_all()
                self.dr.cache.metadata_store.delete_all()
                logger.info("Cleared all caches")
            else:
                # Clear cache for specific functions
                cleared_count = 0
                function_names_set = set(function_names)
                
                # Iterate through all run IDs and clear specific functions
                for run_id in self.dr.cache.run_ids:
                    if run_id in self.dr.cache.cache_keys:
                        cache_keys = self.dr.cache.cache_keys[run_id]
                        for function_name in list(cache_keys.keys()):
                            if function_name in function_names_set:
                                cache_key = cache_keys[function_name]
                                try:
                                    # Get the data version for this cache key
                                    data_version = self.dr.cache.metadata_store.get(cache_key)
                                    if data_version:
                                        # Delete the result associated with the data version
                                        self.dr.cache.result_store.delete(data_version)
                                    # Delete the metadata entry
                                    self.dr.cache.metadata_store.delete(cache_key)
                                    cleared_count += 1
                                    logger.debug(f"Cleared cache for function: {function_name}")
                                except Exception as e:
                                    logger.warning(f"Failed to clear cache for function {function_name}: {str(e)}")
                
                if cleared_count > 0:
                    logger.info(f"Cleared {cleared_count} cache entries for {len(function_names)} functions")
                else:
                    logger.warning(f"No cache entries found for specified functions: {', '.join(function_names)}")
                
        except Exception as e:
            logger.error(f"Failed to clear cache: {str(e)}")
            raise
    
    
    def list_cached_functions(self) -> List[str]:
        """
        List all functions that have cached results.
        
        Returns:
            List of function names with cached results
        """
        if self.disable_cache:
            logger.warning("Cache is disabled for this pipeline instance")
            return []
            
        try:
            # Use Hamilton's SQLiteMetadataStore to get all cached functions
            from hamilton.caching.stores.sqlite import SQLiteMetadataStore
            
            cache_dir = Path(self.output_dir) / ".hamilton_cache"
            if not cache_dir.exists():
                return []
            
            sqlite_ms = SQLiteMetadataStore(path=str(cache_dir))
            
            # Get all run IDs from the metadata store
            run_ids = sqlite_ms.get_run_ids()
            if not run_ids:
                return []
            
            # Collect all unique function names from all runs
            cached_functions = set()
            for run_id in run_ids:
                run_metadata = sqlite_ms.get_run(run_id)
                if run_metadata:
                    for node_metadata in run_metadata:
                        if 'node_name' in node_metadata:
                            cached_functions.add(node_metadata['node_name'])
            
            return sorted(list(cached_functions))
            
        except Exception as e:
            logger.error(f"Failed to list cached functions: {str(e)}")
            return []
    
    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get information about the cache.
        
        Returns:
            Dictionary with cache statistics
        """
        if self.disable_cache:
            return {
                "cache_enabled": False,
                "message": "Cache is disabled for this pipeline instance"
            }
            
        try:
            # Use Hamilton's SQLiteMetadataStore to get cache information
            from hamilton.caching.stores.sqlite import SQLiteMetadataStore
            
            cache_dir = Path(self.output_dir) / ".hamilton_cache"
            if not cache_dir.exists():
                return {
                    "cache_enabled": True,
                    "total_result_entries": 0,
                    "total_metadata_entries": 0,
                    "cached_functions": 0,
                    "function_counts": {},
                    "cache_directory": str(cache_dir)
                }
            
            sqlite_ms = SQLiteMetadataStore(path=str(cache_dir))
            
            # Get all run IDs from the metadata store
            run_ids = sqlite_ms.get_run_ids()
            if not run_ids:
                return {
                    "cache_enabled": True,
                    "total_result_entries": 0,
                    "total_metadata_entries": 0,
                    "cached_functions": 0,
                    "function_counts": {},
                    "cache_directory": str(cache_dir)
                }
            
            # Collect function counts from all runs
            function_counts = {}
            total_entries = 0
            for run_id in run_ids:
                run_metadata = sqlite_ms.get_run(run_id)
                if run_metadata:
                    for node_metadata in run_metadata:
                        if 'node_name' in node_metadata:
                            node_name = node_metadata['node_name']
                            function_counts[node_name] = function_counts.get(node_name, 0) + 1
                            total_entries += 1
            
            return {
                "cache_enabled": True,
                "total_result_entries": total_entries,
                "total_metadata_entries": total_entries,
                "cached_functions": len(function_counts),
                "function_counts": function_counts,
                "cache_directory": str(cache_dir)
            }
            
        except Exception as e:
            logger.error(f"Failed to get cache info: {str(e)}")
            return {}

# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def run_hamilton_pipeline(
    config_path: str = "config/default.yaml",
    use_remote: bool = True, 
    disable_cache: bool = False,
    selected_subgraphs: Optional[List[str]] = None,
    outputs: Optional[List[str]] = None,
    # Telemetry parameters
    enable_telemetry: bool = False,
    project_id: Optional[str] = None,
    username: Optional[str] = None,
    dag_name: Optional[str] = None,
    telemetry_tags: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """
    Run Hamilton pipeline with data files from the data directory.
    
    Args:
        config_path: Path to YAML config file (default: config/default.yaml)
        use_remote: Whether to use Modal Labs for expensive operations (default: True)
        disable_cache: Whether to disable caching entirely for this run (default: False)
        selected_subgraphs: List of subgraphs to run (None means run all available)
        outputs: List of specific outputs to compute (None means use selected_subgraphs or all)
        enable_telemetry: Whether to enable Hamilton telemetry tracking (default: False)
        project_id: Project ID for Hamilton telemetry (required if enable_telemetry=True)
        username: Username/email for Hamilton telemetry (defaults to OS username if None)
        dag_name: DAG name for Hamilton telemetry (default: "ml_headcount_pipeline")
        telemetry_tags: Additional tags for Hamilton telemetry (default: {"environment": "DEV", "team": "ML_HEADCOUNT"})
        
    Returns:
        Dictionary containing pipeline results
    """
    from pathlib import Path
    from .config import load_config
    
    logger.info("Starting Hamilton ML Headcount Pipeline...")
    
    # Load configuration
    config = load_config(Path(config_path))
    config_dict = config.to_hamilton_inputs()
    
    # Initialize pipeline
    pipeline = HamiltonMLHeadcountPipeline(
        config_dict=config_dict,
        use_remote=use_remote, 
        disable_cache=disable_cache,
        selected_subgraphs=selected_subgraphs,
        enable_telemetry=enable_telemetry,
        project_id=project_id,
        username=username,
        dag_name=dag_name,
        telemetry_tags=telemetry_tags
    )
    
    try:
        # Run pipeline with specified outputs or default behavior
        results = pipeline.run(outputs=outputs)
        
        logger.info("Hamilton pipeline completed successfully!")
        logger.info(f"Generated {len(results)} outputs")
        
        return results
    except Exception as e:
        logger.error(f"Hamilton pipeline failed: {str(e)}")
        raise