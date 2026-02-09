"""
Test script for single-step execution of schema-validated pipeline.

This script demonstrates how to run a multi-step pipeline in a single execution
while still using Pandera schema validation.
"""

import logging
import pandas as pd
import sys
from hamilton import driver
import pandera as pa

# Import the module with simplified schema validation functions
from src.ml_headcount import schema_validation_simple
from src.ml_headcount.schemas import CompanyDatabaseSchema

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger("test_single_flow")

def create_mock_company_data() -> pd.DataFrame:
    """Create mock company data for testing."""
    return pd.DataFrame({
        'id': ['company1', 'company2', 'company3', 'company4', 'company5'],
        'employees': [100, 500, 1000, 5000, 20000],
        'company_size': ['Small', 'Medium', 'Medium', 'Large', 'Enterprise'],
        'employees_in_linkedin': [80, 450, 900, 4500, 18000],
        'claude_total_accepted': [5.0, 25.0, 50.0, 250.0, 1000.0],
        'gpt5_total_accepted': [6.0, 30.0, 60.0, 300.0, 1200.0],
        'gemini_total_accepted': [4.0, 20.0, 40.0, 200.0, 800.0]
    })

def create_bad_data() -> pd.DataFrame:
    """Create data that should fail schema validation."""
    return pd.DataFrame({
        'id': ['company1', 'company2', 'company3', 'company4', 'company5'],
        'employees': [100, -500, 1000, 5000, 20000],  # Negative value will fail
        'company_size': ['Small', None, 'Medium', 'Large', 'Enterprise'],  # None will fail
        'employees_in_linkedin': [80, 450, 900, 4500, 18000]
        # Missing ML estimate columns
    })

def main() -> None:
    """Test single-step execution with schema validation."""
    logger.info("Starting single-flow schema validation test")
    
    # Create mock data
    mock_data = create_mock_company_data()
    mock_file_path = "data/mock_company_database.tsv"
    mock_data.to_csv(mock_file_path, sep='\t', index=False)
    logger.info(f"Created mock data at {mock_file_path}")
    
    # Set up paths
    output_path = "outputs/test_single_flow_results.csv"
    
    # Create Hamilton driver with our modules
    dr = driver.Builder().with_modules(
        schema_validation_simple
    ).build()
    
    # List available variables
    logger.info("Available nodes in the pipeline:")
    for var in dr.list_available_variables():
        logger.info(f"- {var.name}")
    
    # Test with good data - single flow execution
    logger.info("\n\n=== Testing with good data - SINGLE FLOW EXECUTION ===")
    try:
        # Verify data passes schema directly
        CompanyDatabaseSchema.validate(mock_data)
        logger.info("Mock data passes schema validation")
        
        # Execute the full pipeline in a single step using the adapter function
        result = dr.execute(
            ['process_full_pipeline'], 
            inputs={
                'company_database': mock_data,
                'debias_factor': 1.2
            }
        )
        
        logger.info("Successfully executed full pipeline in one step")
        logger.info(result['process_full_pipeline'].head())
        
        # Save the result
        result['process_full_pipeline'].to_csv(output_path, index=False)
        logger.info(f"Saved results to {output_path}")
        
        # Check validation results
        validation_nodes = [
            var.name for var in dr.list_available_variables() 
            if var.tags.get('hamilton.data_quality.contains_dq_results')
        ]
        
        if validation_nodes:
            logger.info("\nValidation result nodes available:")
            for node in validation_nodes:
                logger.info(f"- {node}")
        
    except pa.errors.SchemaError as e:
        logger.error(f"Schema validation error: {e}")
    except Exception as e:
        logger.error(f"Error running pipeline: {e}")
    
    # Test with bad data
    logger.info("\n\n=== Testing with bad data (expect validation errors) ===")
    try:
        bad_data = create_bad_data()
        
        # Try running the full pipeline with bad data
        logger.info("Attempting to run full pipeline with bad data")
        result = dr.execute(['process_full_pipeline'], inputs={'company_database': bad_data, 'debias_factor': 1.2})
        logger.info("WARNING: Pipeline accepted bad data (shouldn't reach here)")
    except Exception as e:
        logger.info(f"Success: Pipeline correctly rejected bad data: {str(e)[:200]}...")

if __name__ == "__main__":
    main()
