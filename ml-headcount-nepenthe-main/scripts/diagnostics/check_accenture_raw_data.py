#!/usr/bin/env python3
"""
Check raw data files for Accenture profiles using the correct paths from the pipeline.
"""

import pandas as pd
import json
import gzip
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from ml_headcount.hamilton_dataloaders import INPUT_DATA_CONFIG
import yaml

def main():
    # Load config to get data_dir
    with open("config/default.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    data_dir = Path(config['data_paths']['data_dir'])
    print(f"Data directory: {data_dir}")
    
    # Check LinkedIn profile files
    print("\n" + "="*80)
    print("CHECKING LINKEDIN PROFILE FILES")
    print("="*80)
    
    profile_files = [
        ("85k profiles", INPUT_DATA_CONFIG['dawid_skene_linkedin_profiles_raw']['file_path']),
        ("Big consulting", INPUT_DATA_CONFIG['dawid_skene_linkedin_profiles_big_consulting']['file_path']),
        ("Comparator", INPUT_DATA_CONFIG['dawid_skene_linkedin_profiles_comparator']['file_path']),
    ]
    
    accenture_profiles = []
    
    for name, rel_path in profile_files:
        file_path = data_dir / rel_path
        print(f"\n{name}: {file_path}")
        
        if not file_path.exists():
            print(f"  File not found: {file_path}")
            continue
        
        print(f"  Checking file...")
        count = 0
        accenture_count = 0
        
        try:
            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                for line in f:
                    count += 1
                    if count % 10000 == 0:
                        print(f"    Processed {count} profiles...")
                    
                    try:
                        profile = json.loads(line.strip())
                        current_company = profile.get('current_company', {})
                        company_id = current_company.get('company_id', '')
                        company_name = current_company.get('name', '')
                        linkedin_id = profile.get('linkedin_id') or profile.get('id', '')
                        
                        if 'accenture' in company_id.lower() or 'accenture' in company_name.lower():
                            accenture_count += 1
                            accenture_profiles.append({
                                'source': name,
                                'linkedin_id': linkedin_id,
                                'company_id': company_id,
                                'company_name': company_name
                            })
                    except Exception as e:
                        if count < 10:
                            print(f"    Error parsing line {count}: {e}")
                        continue
            
            print(f"  Total profiles: {count}")
            print(f"  Accenture profiles: {accenture_count}")
            
        except Exception as e:
            print(f"  Error reading file: {e}")
    
    print(f"\n\nTotal Accenture profiles found: {len(accenture_profiles)}")
    if len(accenture_profiles) > 0:
        print(f"\nFirst 10 Accenture profiles:")
        for i, p in enumerate(accenture_profiles[:10], 1):
            print(f"  {i}. {p['company_id']} / {p['company_name']} (from {p['source']})")
    
    # Check keyword filter files
    print("\n" + "="*80)
    print("CHECKING KEYWORD FILTER FILES")
    print("="*80)
    
    keyword_files = [
        ("85k profiles", INPUT_DATA_CONFIG['dawid_skene_keyword_filters_raw']['file_path']),
        ("Big consulting", INPUT_DATA_CONFIG['dawid_skene_keyword_filters_big_consulting']['file_path']),
        ("Comparator", INPUT_DATA_CONFIG['dawid_skene_keyword_filters_comparator']['file_path']),
    ]
    
    accenture_keyword_data = []
    
    for name, rel_path in keyword_files:
        file_path = data_dir / rel_path
        print(f"\n{name}: {file_path}")
        
        if not file_path.exists():
            print(f"  File not found: {file_path}")
            continue
        
        try:
            df = pd.read_csv(file_path)
            print(f"  Total profiles: {len(df)}")
            print(f"  Columns: {list(df.columns)}")
            
            # Check for company_id or need to match via id/public_identifier
            if 'company_id' in df.columns:
                accenture_df = df[df['company_id'].str.contains('accenture', case=False, na=False)]
            elif 'id' in df.columns:
                # Match with LinkedIn profiles we found
                accenture_linkedin_ids = {p['linkedin_id'] for p in accenture_profiles}
                accenture_df = df[df['id'].isin(accenture_linkedin_ids)]
            elif 'public_identifier' in df.columns:
                accenture_linkedin_ids = {p['linkedin_id'] for p in accenture_profiles}
                accenture_df = df[df['public_identifier'].isin(accenture_linkedin_ids)]
            else:
                print(f"  No matching column found for Accenture")
                accenture_df = pd.DataFrame()
            
            if len(accenture_df) > 0:
                print(f"  Accenture profiles: {len(accenture_df)}")
                
                # Check for filter columns
                filter_cols = ['filter_broad_yes', 'filter_strict_no', 'filter_broad_yes_strict_no']
                available_cols = [col for col in filter_cols if col in accenture_df.columns]
                
                if available_cols:
                    print(f"  Filter columns found: {available_cols}")
                    for col in available_cols:
                        total = accenture_df[col].sum()
                        print(f"    {col}: {total}")
                    
                    accenture_keyword_data.append({
                        'source': name,
                        'count': len(accenture_df),
                        'filter_broad_yes': accenture_df['filter_broad_yes'].sum() if 'filter_broad_yes' in accenture_df.columns else 0,
                        'filter_strict_no': accenture_df['filter_strict_no'].sum() if 'filter_strict_no' in accenture_df.columns else 0,
                        'filter_broad_yes_strict_no': accenture_df['filter_broad_yes_strict_no'].sum() if 'filter_broad_yes_strict_no' in accenture_df.columns else 0,
                    })
                else:
                    print(f"  No filter columns found")
            else:
                print(f"  No Accenture profiles found in keyword filters")
                
        except Exception as e:
            print(f"  Error reading file: {e}")
            import traceback
            traceback.print_exc()
    
    # Check LLM result files
    print("\n" + "="*80)
    print("CHECKING LLM RESULT FILES")
    print("="*80)
    
    llm_dirs = [
        ("85k profiles", INPUT_DATA_CONFIG['dawid_skene_llm_results_dir']['file_path']),
        ("Big consulting", INPUT_DATA_CONFIG['dawid_skene_llm_results_dir_big_consulting']['file_path']),
        ("Comparator", INPUT_DATA_CONFIG['dawid_skene_llm_results_dir_comparator']['file_path']),
    ]
    
    accenture_llm_data = {'gemini': 0, 'claude': 0, 'gpt5': 0}
    accenture_llm_profiles = set()
    
    for name, rel_path in llm_dirs:
        dir_path = data_dir / rel_path
        print(f"\n{name}: {dir_path}")
        
        if not dir_path.exists():
            print(f"  Directory not found: {dir_path}")
            continue
        
        # Find JSONL files
        jsonl_files = list(dir_path.glob('*.jsonl'))
        print(f"  Found {len(jsonl_files)} JSONL files")
        
        accenture_linkedin_ids = {p['linkedin_id'] for p in accenture_profiles}
        
        for jsonl_file in jsonl_files:
            model_name = jsonl_file.name.lower()
            model_type = None
            if 'gemini' in model_name:
                model_type = 'gemini'
            elif 'claude' in model_name or 'sonnet' in model_name:
                model_type = 'claude'
            elif 'gpt' in model_name:
                model_type = 'gpt5'
            
            if model_type:
                print(f"    Checking {jsonl_file.name} ({model_type})...")
                count = 0
                accepts = 0
                
                try:
                    with open(jsonl_file, 'r') as f:
                        for line in f:
                            try:
                                result = json.loads(line.strip())
                                linkedin_id = result.get('linkedin_id') or result.get('id', '')
                                
                                if linkedin_id in accenture_linkedin_ids:
                                    count += 1
                                    accenture_llm_profiles.add(linkedin_id)
                                    evaluation = result.get('evaluation', '')
                                    if evaluation == 'ACCEPT':
                                        accepts += 1
                                        accenture_llm_data[model_type] += 1
                            except:
                                pass
                    
                    if count > 0:
                        print(f"      Accenture profiles: {count}, Accepts: {accepts}")
                except Exception as e:
                    print(f"      Error reading file: {e}")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Accenture profiles in raw data: {len(accenture_profiles)}")
    print(f"Accenture profiles with keyword filters: {sum(d['count'] for d in accenture_keyword_data)}")
    print(f"Accenture profiles with LLM results: {len(accenture_llm_profiles)}")
    
    print(f"\nKeyword filter sums:")
    total_broad_yes = sum(d['filter_broad_yes'] for d in accenture_keyword_data)
    total_strict_no = sum(d['filter_strict_no'] for d in accenture_keyword_data)
    total_broad_yes_strict_no = sum(d['filter_broad_yes_strict_no'] for d in accenture_keyword_data)
    print(f"  filter_broad_yes: {total_broad_yes} (DB has 3500)")
    print(f"  filter_strict_no: {total_strict_no} (DB has 3400)")
    print(f"  filter_broad_yes_strict_no: {total_broad_yes_strict_no} (DB has 186)")
    
    print(f"\nLLM accepts:")
    print(f"  Gemini: {accenture_llm_data['gemini']} (DB has 0)")
    print(f"  Claude: {accenture_llm_data['claude']} (DB has 0)")
    print(f"  GPT-5: {accenture_llm_data['gpt5']} (DB has 0)")


if __name__ == "__main__":
    main()





