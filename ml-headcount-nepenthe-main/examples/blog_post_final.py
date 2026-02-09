"""
Final version of script that mimics the DAGWorks blog post example.
https://blog.dagworks.io/p/data-quality-with-hamilton-and-pandera
"""

import logging
import sys
import pandas as pd
import numpy as np
from hamilton import driver
from hamilton.function_modifiers import check_output
import pandera.pandas as pa

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)

# Define functions directly as in the blog post

def spend() -> pd.Series:
    """Generate sample spend data."""
    return pd.Series([10, 100, 20, 40, 40, 50])

def signups() -> pd.Series:
    """Generate sample signup data."""
    return pd.Series([1, 0.02, 50, 100, 200, 400])

def avg_3wk_spend(spend: pd.Series) -> pd.Series:
    """Rolling 3 week average spend."""
    return spend.rolling(3, min_periods=1).mean()

@check_output(
    range=(0, 1000),
    data_type=np.float64,
    importance="warn"
)
def acquisition_cost(avg_3wk_spend: pd.Series, signups: pd.Series) -> pd.Series:
    """The cost per signup in relation to a rolling average of spend."""
    return avg_3wk_spend / signups

# Add a DataFrame example with Pandera
@check_output(
    schema=pa.DataFrameSchema(
        {
            'id': pa.Column(int),
            'value': pa.Column(float, pa.Check(lambda s: s > 0)),
        },
    ),
    importance="fail"
)
def simple_dataframe() -> pd.DataFrame:
    """Create a simple dataframe with schema validation."""
    return pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'value': [10.1, 20.2, 30.3, 40.4, 50.5],
    })

if __name__ == "__main__":
    print("Running example similar to the blog post")
    
    # Import the current module for Hamilton
    import blog_post_final as my_functions
    
    # Create a Hamilton driver with our module
    dr = driver.Builder().with_modules(my_functions).build()
    
    # List available nodes
    print("\nAvailable nodes:")
    for var in dr.list_available_variables():
        print(f"- {var.name}")
    
    # Execute the pipeline to get acquisition_cost
    print("\n=== Executing acquisition_cost (should show warning) ===")
    result = dr.execute(['acquisition_cost'])
    df = result['acquisition_cost']
    print(df.to_string())
    
    # Execute the DataFrame example
    print("\n=== Executing simple_dataframe example ===")
    result = dr.execute(['simple_dataframe'])
    df_result = result['simple_dataframe']
    print(df_result.to_string())
    
    # Access validation results
    print("\n=== Accessing validation results ===")
    all_validator_variables = [
        var.name for var in dr.list_available_variables() 
        if var.tags.get('hamilton.data_quality.contains_dq_results')
    ]
    
    if all_validator_variables:
        print(f"Found validation variables: {all_validator_variables}")
        try:
            # Execute to get both the main result and validation results
            data = dr.execute(['acquisition_cost', 'simple_dataframe'] + all_validator_variables)
            
            # Display results for each validator
            for var_name in all_validator_variables:
                if var_name in data:
                    print(f"\n{var_name}:")
                    print(f"  Passes: {data[var_name].passes}")
                    print(f"  Message: {data[var_name].message}")
                    if hasattr(data[var_name], 'diagnostics'):
                        print(f"  Diagnostics: {data[var_name].diagnostics}")
        except Exception as e:
            print(f"Error accessing validation results: {e}")
    else:
        print("No validation variables found")
