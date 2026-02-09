#!/usr/bin/env python3
"""
Example script demonstrating Hamilton telemetry integration.

This script shows how to enable telemetry tracking for the ML Headcount pipeline.
"""

import sys
import os
from pathlib import Path

# Add the src directory to the path so we can import the modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ml_headcount.hamilton_pipeline import run_hamilton_pipeline_with_data_files

def main():
    """
    Example of running the Hamilton pipeline with telemetry enabled.
    """
    print("Hamilton Telemetry Example")
    print("=" * 50)
    
    # Example telemetry configuration
    # Replace these with your actual values
    PROJECT_ID = "your-project-id"  # Get this from Hamilton UI
    USERNAME = "your-email@example.com"  # Your email/username
    DAG_NAME = "ml_headcount_analysis"
    
    # Custom tags for this run
    telemetry_tags = {
        "environment": "PROD",
        "team": "ML_HEADCOUNT",
        "version": "1.0.0",
        "experiment": "telemetry_test"
    }
    
    try:
        print(f"Running pipeline with telemetry enabled...")
        print(f"Project ID: {PROJECT_ID}")
        print(f"Username: {USERNAME}")
        print(f"DAG Name: {DAG_NAME}")
        print(f"Tags: {telemetry_tags}")
        print()
        
        # Run the pipeline with telemetry
        results = run_hamilton_pipeline_with_data_files(
            data_dir="data",
            output_dir="outputs_telemetry_test",
            use_remote=False,  # Set to True if you want to use Modal
            disable_cache=False,
            # Telemetry configuration
            enable_telemetry=True,
            project_id=PROJECT_ID,
            username=USERNAME,
            dag_name=DAG_NAME,
            telemetry_tags=telemetry_tags
        )
        
        print(f"Pipeline completed successfully!")
        print(f"Generated {len(results)} outputs")
        
        # List some of the outputs
        if results:
            print("\nSample outputs:")
            for i, (key, value) in enumerate(results.items()):
                if i < 5:  # Show first 5 outputs
                    print(f"  - {key}: {type(value).__name__}")
                else:
                    remaining = len(results) - 5
                    print(f"  ... and {remaining} more outputs")
                    break
        
    except ValueError as e:
        print(f"Configuration error: {e}")
        print("\nTo use telemetry, you need to:")
        print("1. Get a project ID from the Hamilton UI")
        print("2. Set your username/email")
        print("3. Update the PROJECT_ID and USERNAME variables in this script")
        
    except Exception as e:
        print(f"Pipeline failed: {e}")
        raise

def run_without_telemetry():
    """
    Example of running the pipeline without telemetry (default behavior).
    """
    print("\nRunning pipeline without telemetry...")
    print("=" * 50)
    
    try:
        results = run_hamilton_pipeline_with_data_files(
            data_dir="data",
            output_dir="outputs_no_telemetry",
            use_remote=False,
            disable_cache=False
            # No telemetry parameters - defaults to disabled
        )
        
        print(f"Pipeline completed successfully!")
        print(f"Generated {len(results)} outputs")
        
    except Exception as e:
        print(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    # Run with telemetry (will fail if PROJECT_ID/USERNAME not set)
    main()
    
    # Run without telemetry
    run_without_telemetry()
