"""
Test script for the schema validation example module.

This script demonstrates how to use the Hamilton functions with schema validation
in a simple pipeline.
"""

import logging
import pandas as pd
import sys
from hamilton import driver
import pandera as pa

# Import the module with schema validation functions
from src.ml_headcount import schema_validation_example
from src.ml_headcount import hamilton_dataloaders
from src.ml_headcount.schemas import CompanyDatabaseSchema

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger("test_schema_validation")

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

def main():
    logger.info("Starting schema validation test")
    
    # Create a mock company database file for testing
    mock_data = create_mock_company_data()
    mock_file_path = "data/mock_company_database.tsv"
    mock_data.to_csv(mock_file_path, sep='\t', index=False)
    logger.info(f"Created mock data at {mock_file_path}")
    
    # Set up paths
    input_path = mock_file_path
    output_path = "outputs/test_debiased_organizations.csv"
    
    # Create Hamilton driver with our modules
    dr = driver.Builder().with_modules(
        schema_validation_example
    ).build()
    
    # List available variables to see what our pipeline can produce
    logger.info("Available nodes in the pipeline:")
    for var in dr.list_available_variables():
        logger.info(f"- {var.name}")
    
    # Test with good data
    logger.info("\n\n=== Testing with good data ===")
    try:
        # Validate that mock data conforms to schema
        validated_input = CompanyDatabaseSchema.validate(mock_data)
        logger.info("Mock data passes schema validation")
        
        # Create inputs dict with the properly typed DataFrame
        from pandera.typing import DataFrame
        typed_df = DataFrame[CompanyDatabaseSchema](mock_data)
        
        # Execute the pipeline steps individually to ensure proper typing
        # First get geographic data
        geo_result = dr.execute(['add_geographic_data'], inputs={'company_database': typed_df})
        
        # Convert to the proper Pandera typed DataFrame 
        from src.ml_headcount.schemas import CompanyDatabaseWithSubregionsSchema
        geo_df = DataFrame[CompanyDatabaseWithSubregionsSchema](geo_result['add_geographic_data'])
        
        # Then run debiasing with the result from the previous step
        result = dr.execute(
            ['apply_debiasing'], 
            inputs={
                'company_database_with_geo': geo_df,
                'debias_factor': 1.2
            }
        )
        
        # Show results
        logger.info("\nGeographic data results:")
        logger.info(geo_result['add_geographic_data'].head())
        
        logger.info("\nDebiased results:")
        logger.info(result['apply_debiasing'].head())
        
        # Save output
        result['apply_debiasing'].to_csv(output_path, index=False)
        logger.info(f"Saved results to {output_path}")
        
        # Check validation results
        validation_nodes = [
            var.name for var in dr.list_available_variables() 
            if var.tags.get('hamilton.data_quality.contains_dq_results')
        ]
        
        if validation_nodes:
            logger.info("\nValidation result nodes:")
            for node in validation_nodes:
                logger.info(f"- {node}")
            
            # Note: We can't directly access the validation results because they require
            # the same inputs as the functions they validate. This is a limitation of
            # Hamilton's current data quality implementation.
            logger.info("\nValidation was successful - our pipelines ran without errors")
            logger.info("To access validation results directly, we would need to run a more complex pipeline")
        else:
            logger.info("No validation result nodes found")
            
    except pa.errors.SchemaError as e:
        logger.error(f"Schema validation error: {e}")
    except Exception as e:
        logger.error(f"Error running pipeline with good data: {e}")
    
    # Test with bad data that should fail validation
    logger.info("\n\n=== Testing with bad data (expect validation errors) ===")
    try:
        # Try to validate the bad data directly with Pandera schema
        bad_data = create_bad_data()
        logger.info("Validating bad data with Pandera schema...")
        CompanyDatabaseSchema.validate(bad_data)
        logger.info("WARNING: Bad data passed schema validation (unexpected)")
    except pa.errors.SchemaError as e:
        logger.info(f"Success: Bad data correctly failed schema validation: {e}")
        
    # Try running the pipeline with bad data anyway
    try:
        from pandera.typing import DataFrame
        typed_bad_df = DataFrame[CompanyDatabaseSchema](bad_data)  # This should fail
        geo_result = dr.execute(['add_geographic_data'], inputs={'company_database': typed_bad_df})
        logger.info("WARNING: Pipeline accepted bad data (shouldn't reach here)")
    except pa.errors.SchemaError as e:
        logger.info(f"Success: Pipeline correctly rejected bad data: {e}")
    except Exception as e:
        logger.error(f"Unexpected error with bad data: {e}")

if __name__ == "__main__":
    main()
