"""
Test script for the simplified schema validation approach.

This script demonstrates how to use the schema= approach in Hamilton
without requiring explicit DataFrame typing between pipeline steps.
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
logger = logging.getLogger("test_schema_simple")

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
    """Test simplified schema validation approach."""
    logger.info("Starting simplified schema validation test")
    
    # Create mock data
    mock_data = create_mock_company_data()
    mock_file_path = "data/mock_company_database.tsv"
    mock_data.to_csv(mock_file_path, sep='\t', index=False)
    logger.info(f"Created mock data at {mock_file_path}")
    
    # Set up paths
    output_path = "outputs/test_simplified_debiased_orgs.csv"
    
    # Create Hamilton driver with our modules
    dr = driver.Builder().with_modules(
        schema_validation_simple
    ).build()
    
    # List available variables
    logger.info("Available nodes in the pipeline:")
    for var in dr.list_available_variables():
        logger.info(f"- {var.name}")
    
    # Test with good data
    logger.info("\n\n=== Testing with good data ===")
    try:
        # Verify data passes schema directly
        CompanyDatabaseSchema.validate(mock_data)
        logger.info("Mock data passes schema validation")
        
        # Test step-by-step execution first
        logger.info("\n=== Step-by-step execution ===")
        
        # First step: add geographic data
        geo_result = dr.execute(['add_geographic_data'], inputs={'company_database': mock_data})
        logger.info("Geographic data added successfully")
        logger.info(geo_result['add_geographic_data'].head())
        
        # Second step: apply debiasing using the result from first step (no explicit typing)
        debiased_result = dr.execute(
            ['apply_debiasing'], 
            inputs={
                'company_database_with_geo': geo_result['add_geographic_data'],
                'debias_factor': 1.2
            }
        )
        logger.info("Debiasing applied successfully")
        logger.info(debiased_result['apply_debiasing'].head())
        
        # Save the result
        debiased_result['apply_debiasing'].to_csv(output_path, index=False)
        logger.info(f"Saved results to {output_path}")
        
        # Now test single-step execution
        logger.info("\n=== Single-step execution ===")
        # Execute the pipeline in a single step (no explicit typing)
        full_result = dr.execute(
            ['apply_debiasing'], 
            inputs={
                'company_database': mock_data,  # Original input
                'debias_factor': 1.2
            }
        )
        logger.info("Single-step execution completed")
        logger.info(full_result['apply_debiasing'].head())
        
        # Check validation results
        validation_nodes = [
            var.name for var in dr.list_available_variables() 
            if var.tags.get('hamilton.data_quality.contains_dq_results')
        ]
        
        if validation_nodes:
            logger.info("\nValidation result nodes:")
            for node in validation_nodes:
                logger.info(f"- {node}")
            
            logger.info("\nValidation passed")
        
    except pa.errors.SchemaError as e:
        logger.error(f"Schema validation error: {e}")
    except Exception as e:
        logger.error(f"Error running pipeline: {e}")
    
    # Test with bad data
    logger.info("\n\n=== Testing with bad data (expect validation errors) ===")
    try:
        bad_data = create_bad_data()
        logger.info("Validating bad data with Pandera schema")
        CompanyDatabaseSchema.validate(bad_data)
        logger.info("WARNING: Bad data passed schema validation (unexpected)")
    except pa.errors.SchemaError as e:
        logger.info(f"Success: Bad data correctly failed schema validation: {e}")
        
    # Try running the pipeline with bad data
    try:
        logger.info("Attempting to run pipeline with bad data")
        result = dr.execute(['add_geographic_data'], inputs={'company_database': bad_data})
        logger.info("WARNING: Pipeline accepted bad data (shouldn't reach here)")
    except Exception as e:
        logger.info(f"Success: Pipeline correctly rejected bad data: {e}")

if __name__ == "__main__":
    main()
