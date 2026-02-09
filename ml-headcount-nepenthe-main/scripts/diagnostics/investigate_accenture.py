#!/usr/bin/env python3
"""
Investigate Accenture specifically:
1. Check if it has nonzero annotation aggregates in company database
2. Check if it has zero probit estimates
3. Count individual employees with LLM or keyword estimates
4. Sum raw values and compare with aggregates
"""

import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from ml_headcount.hamilton_pipeline import HamiltonMLHeadcountPipeline
from ml_headcount.hamilton_dataloaders import company_database_complete, INPUT_DATA_CONFIG
import yaml

def main():
    # Load config
    with open("config/default.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize pipeline
    pipeline = HamiltonMLHeadcountPipeline(
        config_dict=config,
        selected_subgraphs=[],
        use_remote=False,
        disable_cache=False
    )
    
    # Load company database
    print("Loading company database...")
    data_dir = Path(config['data_paths']['data_dir'])
    company_db_path = data_dir / INPUT_DATA_CONFIG['company_database_complete']['file_path']
    
    company_db = company_database_complete(
        df=pd.read_excel(
            company_db_path,
            sheet_name=INPUT_DATA_CONFIG['company_database_complete']['sheet_name']
        )
    )
    
    # Find Accenture
    accenture_db = company_db[
        company_db['organization_name'].str.contains('Accenture', case=False, na=False) |
        company_db['linkedin_id'].str.contains('accenture', case=False, na=False)
    ]
    
    print("\n" + "="*80)
    print("ACCENTURE IN COMPANY DATABASE")
    print("="*80)
    if len(accenture_db) > 0:
        print(accenture_db[['linkedin_id', 'organization_name', 
                            'gemini_total_accepted', 'claude_total_accepted', 'gpt5_total_accepted',
                            'filter_broad_yes', 'filter_strict_no', 'filter_broad_yes_strict_no',
                            'total_headcount']].to_string(index=False))
    else:
        print("Accenture not found in company database")
        return
    
    accenture_linkedin_id = accenture_db.iloc[0]['linkedin_id']
    print(f"\nAccenture linkedin_id: {accenture_linkedin_id}")
    
    # Check probit results
    print("\n" + "="*80)
    print("ACCENTURE IN PROBIT RESULTS")
    print("="*80)
    probit_path = Path("outputs/probit_results_main_orgs.csv")
    if probit_path.exists():
        probit = pd.read_csv(probit_path)
        accenture_probit = probit[
            probit['linkedin_id'].str.contains('accenture', case=False, na=False)
        ]
        if len(accenture_probit) > 0:
            print(accenture_probit.to_string(index=False))
        else:
            print("Accenture NOT in probit results (zero or missing)")
    else:
        print(f"Probit results file not found at {probit_path}")
    
    # Check log-debias aggregates (from test data)
    print("\n" + "="*80)
    print("ACCENTURE IN LOG-DEBIAS AGGREGATES (from test data)")
    print("="*80)
    log_debias_path = Path("outputs/log_debias_company_aggregates.csv")
    if log_debias_path.exists():
        log_debias = pd.read_csv(log_debias_path)
        accenture_log = log_debias[
            log_debias['company_id'].str.contains('accenture', case=False, na=False)
        ]
        if len(accenture_log) > 0:
            print(accenture_log[['company_id', 'company_name',
                                 'gemini_total_accepted', 'claude_total_accepted', 'gpt5_total_accepted',
                                 'filter_broad_yes', 'filter_strict_no', 'filter_broad_yes_strict_no']].to_string(index=False))
        else:
            print("Accenture NOT in log-debias aggregates (not in test data)")
    else:
        print(f"Log-debias file not found at {log_debias_path}")
    
    # Now try to load annotation data using Hamilton driver
    print("\n" + "="*80)
    print("LOADING INDIVIDUAL PROFILE DATA")
    print("="*80)
    
    try:
        # Use Hamilton driver to load cached data
        driver = pipeline.dr
        
        # Try to get annotation data - driver will use cached data if available
        try:
            # Get required inputs from config
            inputs = {
                'enable_85k_profiles': config.get('linkedin_datasets', {}).get('enable_85k_profiles', True),
                'enable_big_consulting': config.get('linkedin_datasets', {}).get('enable_big_consulting', True),
                'enable_comparator': config.get('linkedin_datasets', {}).get('enable_comparator', True),
            }
            
            annotation_data = driver.execute(
                ["dawid_skene_annotation_data"],
                inputs=inputs
            )["dawid_skene_annotation_data"]
            
            # Find Accenture profiles
            accenture_profiles = annotation_data[
                (annotation_data['company_id'].str.contains('accenture', case=False, na=False)) |
                (annotation_data['company_name'].str.contains('Accenture', case=False, na=False))
            ]
            
            print(f"\nFound {len(accenture_profiles)} Accenture profiles in annotation data")
            
            if len(accenture_profiles) > 0:
                print(f"Company IDs: {accenture_profiles['company_id'].unique()}")
                print(f"Company names: {accenture_profiles['company_name'].unique()}")
                
                # Get annotator columns
                annotator_cols = [col for col in accenture_profiles.columns 
                                 if col not in ['public_identifier', 'company_id', 'company_name']]
                
                print(f"\nAnnotator columns: {annotator_cols}")
                
                # Sum up annotations
                print("\nAnnotation sums from individual profiles:")
                sums = {}
                for col in annotator_cols:
                    if col in accenture_profiles.columns:
                        total = accenture_profiles[col].fillna(0).sum()
                        count = accenture_profiles[col].notna().sum()
                        count_positive = (accenture_profiles[col].fillna(0) == 1).sum()
                        sums[col] = total
                        print(f"  {col}: {total} positives (from {count} profiles with data, {count_positive} with value=1)")
                
                # Compare with database aggregates
                print("\n" + "="*80)
                print("COMPARISON: Database Aggregates vs Individual Profile Sums")
                print("="*80)
                
                # Map column names
                mapping = {
                    'llm_gemini-2.5-flash': ('gemini_total_accepted', 'Gemini'),
                    'llm_sonnet-4': ('claude_total_accepted', 'Claude'),
                    'llm_gpt-5-mini': ('gpt5_total_accepted', 'GPT-5'),
                    'filter_broad_yes': ('filter_broad_yes', 'Filter Broad Yes'),
                    'filter_strict_no': ('filter_strict_no', 'Filter Strict No'),
                    'filter_broad_yes_strict_no': ('filter_broad_yes_strict_no', 'Filter Broad Yes Strict No'),
                }
                
                for col, (db_col, name) in mapping.items():
                    if col in sums:
                        db_value = accenture_db.iloc[0][db_col] if db_col in accenture_db.columns else None
                        profile_sum = sums[col]
                        print(f"{name:30s} | DB: {db_value:8.0f} | Profiles: {profile_sum:8.0f} | Match: {db_value == profile_sum if db_value is not None else 'N/A'}")
                
                # Show sample profiles
                print(f"\n\nFirst 20 Accenture profiles:")
                display_cols = ['public_identifier', 'company_id', 'company_name'] + annotator_cols
                print(accenture_profiles[display_cols].head(20).to_string(index=False))
                
            else:
                print("No Accenture profiles found in annotation data")
                
        except Exception as e:
            print(f"Error loading annotation data: {e}")
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        print(f"Error initializing Hamilton driver: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

