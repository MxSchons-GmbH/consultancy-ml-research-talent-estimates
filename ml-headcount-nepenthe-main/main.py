#!/usr/bin/env python3
"""
Main script to run the ML Headcount Pipeline
"""

import sys
import os
import logging
import argparse
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def get_available_subgraphs():
    """Get list of available subgraphs from the Hamilton pipeline."""
    try:
        # Import here to avoid circular imports
        from ml_headcount.hamilton_pipeline import HamiltonMLHeadcountPipeline
        from ml_headcount.config import load_config
        
        # Load default config to get subgraph mappings
        config = load_config()
        pipeline = HamiltonMLHeadcountPipeline(config_dict=config.model_dump())
        
        # Return sorted list of available subgraphs
        return sorted(pipeline.subgraph_mappings.keys())
    except Exception as e:
        # Fallback to hardcoded list if import fails
        print(f"Warning: Could not load subgraphs dynamically: {e}")
        return ["affiliation", "clustering", "cv_analysis", "log_debias", "probit_bootstrap"]

def print_subgraph_info():
    """Print information about available subgraphs and their outputs."""
    print("Available ML Headcount Pipeline Subgraphs:")
    print("=" * 50)
    
    subgraph_descriptions = {
        "affiliation": "Filter academic affiliations and clean data",
        "clustering": "Cluster keywords using HDBSCAN/KMeans",
        "cv_analysis": "Extract keywords from CVs and validate scoring methods",
        "log_debias": "Apply log-debiasing approach to generate 80% CI for ML headcount estimates",
        "probit_bootstrap": "Correlated probit bootstrap analysis for uncertainty quantification"
    }
    
    for subgraph_id, description in subgraph_descriptions.items():
        print(f"\n{subgraph_id.upper()}:")
        print(f"  {description}")
    
    print(f"\nUsage examples:")
    print(f"  python main.py --config config/default.yaml --subgraphs affiliation")
    print(f"  python main.py --config config/test.yaml --subgraphs cv_analysis") 
    print(f"  python main.py --subgraphs organization  # Uses default config")
    print(f"  python main.py --subgraphs probit_bootstrap --local-only")
    print(f"")

def main():
    """Main function to run the ML Headcount Pipeline."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="ML Headcount Pipeline")
    
    # Config file argument
    parser.add_argument("--config", type=str, default="config/default.yaml",
                       help="Path to YAML config file (default: config/default.yaml)")
    
    # Execution arguments (override config)
    parser.add_argument("--use-remote", action="store_true", dest="use_remote_flag",
                       help="Override config: Use Modal Labs for expensive operations")
    parser.add_argument("--local-only", action="store_true",
                       help="Override config: Use local processing only (disable Modal Labs)")
    parser.add_argument("--disable-cache", action="store_true",
                       help="Override config: Disable Hamilton caching")
    parser.add_argument("--no-cache", type=str, nargs="?", const="all",
                       help="Override config: Skip cache for specific functions or all")
    parser.add_argument("--enable-telemetry", action="store_true",
                       help="Override config: Enable Hamilton UI telemetry tracking")
    parser.add_argument("--disable-telemetry", action="store_true",
                       help="Override config: Disable Hamilton UI telemetry tracking")
    parser.add_argument("--project-id", type=str,
                       help="Override config: Hamilton project ID")
    parser.add_argument("--username", type=str,
                       help="Override config: Hamilton username/email")
    parser.add_argument("--dag-name", type=str,
                       help="Override config: DAG name for Hamilton telemetry")
    parser.add_argument("--telemetry-tags", type=str, nargs="*",
                       help="Override config: Additional tags for telemetry (format: key=value)")
    
    # Visualization arguments
    parser.add_argument("--visualize", nargs="?", const="pipeline_dag.png", 
                       help="Visualize pipeline DAG (optionally specify output file path)")
    parser.add_argument("--save-dot", action="store_true", 
                       help="Also save the raw Graphviz DOT file alongside the rendered image")
    
    # Subgraph selection arguments
    available_subgraphs = get_available_subgraphs()
    parser.add_argument("--subgraphs", nargs="+", 
                       choices=available_subgraphs,
                       help=f"Select one or more subgraphs to run: {', '.join(available_subgraphs)}")
    parser.add_argument("--list-subgraphs", action="store_true",
                       help="List available subgraphs and their outputs")
    
    # Cache management arguments
    parser.add_argument("--clear-cache", action="store_true",
                       help="Clear all caches before running")
    parser.add_argument("--cache-info", action="store_true",
                       help="Show cache information and exit")
    parser.add_argument("--list-cached-functions", action="store_true",
                       help="List all cached functions and exit")
    
    args = parser.parse_args()
    
    # Handle subgraph listing
    if args.list_subgraphs:
        print_subgraph_info()
        return 0
    
    # Load configuration
    try:
        from ml_headcount.config import load_config
        config = load_config(Path(args.config))
        print(f"Loaded configuration from: {args.config}")
    except FileNotFoundError:
        print(f"Error: Config file not found: {args.config}")
        print("Available configs: config/default.yaml, config/test.yaml")
        return 1
    except Exception as e:
        print(f"Error loading config: {e}")
        return 1
    
    # Apply CLI overrides to execution config
    # Handle use_remote (--use-remote and --local-only are mutually exclusive)
    if args.use_remote_flag and args.local_only:
        print("Error: --use-remote and --local-only are mutually exclusive")
        return 1
    if args.use_remote_flag:
        use_remote = True
    elif args.local_only:
        use_remote = False
    else:
        use_remote = config.execution.use_remote
    
    # Handle telemetry (--enable-telemetry and --disable-telemetry are mutually exclusive)
    if args.enable_telemetry and args.disable_telemetry:
        print("Error: --enable-telemetry and --disable-telemetry are mutually exclusive")
        return 1
    if args.enable_telemetry:
        enable_telemetry = True
    elif args.disable_telemetry:
        enable_telemetry = False
    else:
        enable_telemetry = config.execution.enable_telemetry
    
    # Other execution config overrides
    project_id = args.project_id if args.project_id else config.execution.project_id
    username = args.username if args.username else config.execution.username
    dag_name = args.dag_name if args.dag_name else config.execution.dag_name
    telemetry_tags = config.execution.telemetry_tags.copy()
    
    # Parse and merge CLI telemetry tags
    if args.telemetry_tags:
        try:
            for tag in args.telemetry_tags:
                if '=' not in tag:
                    print(f"Error: Invalid telemetry tag format: {tag}")
                    print("Expected format: key=value")
                    return 1
                key, value = tag.split('=', 1)
                telemetry_tags[key] = value
        except Exception as e:
            print(f"Error parsing telemetry tags: {e}")
            return 1
    
    # Add subgraphs as a telemetry tag if provided
    if args.subgraphs:
        telemetry_tags['subgraphs'] = ','.join(args.subgraphs)
    
    # Validate telemetry configuration
    if enable_telemetry and not project_id:
        print("Error: Project ID is required when telemetry is enabled")
        print("  Set it in config YAML or use: --project-id YOUR_PROJECT_ID")
        return 1
    
    # Convert config to Hamilton inputs
    config_dict = config.to_hamilton_inputs()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('pipeline.log')
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Starting ML Headcount Pipeline...")
    logger.info(f"Execution mode: {'Remote (Modal Labs)' if use_remote else 'Local'}")
    
    # Log telemetry status
    if enable_telemetry:
        logger.info(f"Hamilton telemetry enabled - Project: {project_id}, User: {username}")
        logger.info(f"Telemetry tags: {telemetry_tags}")
    else:
        logger.info("Hamilton telemetry disabled")
    
    try:
        from ml_headcount.hamilton_pipeline import HamiltonMLHeadcountPipeline
        
        # Parse cache arguments
        # --disable-cache takes precedence, then --no-cache, then config
        if args.disable_cache:
            disable_cache = True
            recompute = None
        elif args.no_cache:
            if args.no_cache == "all":
                disable_cache = True
                recompute = None
            else:
                disable_cache = False
                recompute = [name.strip() for name in args.no_cache.split(",")]
        else:
            disable_cache = config.execution.disable_cache
            recompute = None
        
        # Initialize pipeline
        pipeline = HamiltonMLHeadcountPipeline(
            config_dict=config_dict,
            selected_subgraphs=args.subgraphs,
            use_remote=use_remote,
            disable_cache=disable_cache,
            recompute=recompute,
            enable_telemetry=enable_telemetry,
            project_id=project_id,
            username=username,
            dag_name=dag_name,
            telemetry_tags=telemetry_tags
        )
        
        # Handle cache management commands
        if args.cache_info:
            cache_info = pipeline.get_cache_info()
            print("Cache Information:")
            print("=" * 50)
            
            if not cache_info.get('cache_enabled', True):
                print(f"Status: {cache_info.get('message', 'Cache disabled')}")
            else:
                print(f"Cache directory: {cache_info.get('cache_directory', 'Unknown')}")
                print(f"Total result entries: {cache_info.get('total_result_entries', 0)}")
                print(f"Total metadata entries: {cache_info.get('total_metadata_entries', 0)}")
                print(f"Cached functions: {cache_info.get('cached_functions', 0)}")
                
                if cache_info.get('function_counts'):
                    print("\nFunction cache counts:")
                    for func_name, count in cache_info['function_counts'].items():
                        print(f"  {func_name}: {count} entries")
            return 0
        
        if args.list_cached_functions:
            cached_functions = pipeline.list_cached_functions()
            print("Cached Functions:")
            print("=" * 50)
            if cached_functions:
                for func_name in cached_functions:
                    print(f"  {func_name}")
            else:
                print("  No cached functions found")
            return 0
        
        # Clear caches if requested
        if args.clear_cache:
            pipeline.clear_cache()
            logger.info("Cleared all caches before running pipeline")
        
        # Visualize pipeline if requested
        if args.visualize:
            output_path = args.visualize
            
            # Ensure file has extension
            if not Path(output_path).suffix:
                output_path = f"{output_path}.png"
            
            print(f"Generating pipeline visualization...")
            print(f"Output: {output_path}")
            if args.save_dot:
                print("Also saving DOT file")
            
            pipeline.visualize_pipeline(output_path, save_dot=args.save_dot)
            logger.info(f"Pipeline visualization saved to {output_path}")
            print(f"✅ Pipeline visualization saved to {output_path}")
            
            if args.save_dot and not output_path.endswith('.dot'):
                dot_path = Path(output_path).with_suffix('.dot')
                print(f"✅ DOT file saved to {dot_path}")
            
            return 0
        
        # Run pipeline
        results = pipeline.run()
        
        # Handle results
        if results is not None:
            if isinstance(results, dict):
                result_count = len(results)
            else:
                result_count = len(results.columns) if hasattr(results, 'columns') else len(results)
            
            logger.info("Hamilton pipeline completed successfully!")
            print("\n" + "="*60)
            print("HAMILTON PIPELINE COMPLETED SUCCESSFULLY!")
            print("="*60)
            print(f"Generated outputs: {result_count}")
            print(f"\nCheck the '{config.data_paths.output_dir}' directory for generated files.")
        else:
            logger.error("Hamilton pipeline failed to produce results")
            return 1
            
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
