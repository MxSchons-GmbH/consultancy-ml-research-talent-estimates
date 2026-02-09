#!/usr/bin/env python3
"""
Explore company covariates that predict latent correlation structure.

This script:
1. Loads company database and correlation analysis results
2. Merges them by company name/slug
3. Explores relationships between company features and correlation distance
4. Identifies predictive covariates

NOTE: This analysis uses ONLY keyword filter annotators (3 filters), ignoring LLMs.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns

def normalize_company_name(name):
    """Normalize company name for matching."""
    if pd.isna(name):
        return ""
    # Convert to lowercase, remove special chars, replace spaces with hyphens
    name = str(name).lower()
    name = re.sub(r'[^a-z0-9\s-]', '', name)
    name = re.sub(r'\s+', '-', name.strip())
    return name

def match_companies(corr_data, company_db):
    """Match companies between correlation data and database."""
    # Create normalized names for matching
    corr_data['name_normalized'] = corr_data['company_id'].apply(normalize_company_name)
    company_db['name_normalized'] = company_db['Organization Name'].apply(normalize_company_name)
    
    # Try exact match first
    merged = corr_data.merge(
        company_db,
        on='name_normalized',
        how='left',
        suffixes=('_corr', '_db')
    )
    
    # For unmatched, try fuzzy matching on organization name
    unmatched = merged[merged['Organization Name'].isna()]
    if len(unmatched) > 0:
        print(f"Unmatched companies: {len(unmatched)}")
        # Try matching on partial name
        for idx, row in unmatched.iterrows():
            corr_name = row['company_id']
            # Try to find in organization names
            matches = company_db[
                company_db['Organization Name'].str.lower().str.contains(
                    corr_name.split('-')[0], na=False, case=False
                )
            ]
            if len(matches) > 0:
                # Take first match
                db_row = matches.iloc[0]
                for col in company_db.columns:
                    merged.loc[idx, col] = db_row[col]
    
    return merged

def main():
    print("=" * 80)
    print("CORRELATION STRUCTURE PREDICTOR EXPLORATION")
    print("=" * 80)
    print()
    
    # Load data - using keyword filter analysis results
    print("Loading data (keyword filters only)...")
    
    # Load keyword filter correlation analysis
    keyword_filter_results = pd.read_csv('outputs/diagnostics/keyword_filter_correlation_analysis.csv')
    company_data = keyword_filter_results.copy()
    company_data['distance'] = company_data['distance_keyword_filters']
    
    print(f"Loaded {len(company_data)} companies with keyword filter correlation distances")
    
    # Load company database
    db_path = Path('raw_data/raw_data_search/2025-08-05_systematic_search_all.xlsx')
    company_db = pd.read_excel(db_path, sheet_name='ML Consultancies_merged_with_co')
    
    print(f"Correlation data: {len(company_data)} companies")
    print(f"Company database: {len(company_db)} companies")
    print()
    
    # Match companies
    print("Matching companies...")
    merged = match_companies(company_data, company_db)
    matched = merged[merged['Organization Name'].notna()]
    print(f"Matched: {len(matched)} / {len(company_data)} companies")
    print()
    
    # Save merged data
    output_path = Path('outputs/diagnostics/company_correlation_with_covariates_keyword_filters.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    print(f"Saved merged data to: {output_path}")
    print()
    
    # Explore relationships
    print("=" * 80)
    print("EXPLORATORY ANALYSIS")
    print("=" * 80)
    print()
    
    # Identify numeric and categorical columns
    numeric_cols = []
    categorical_cols = []
    
    for col in company_db.columns:
        if col in ['ID', 'Organization Name', 'name_normalized']:
            continue
        if merged[col].dtype in ['int64', 'float64']:
            if merged[col].notna().sum() > 10:  # At least 10 non-null values
                numeric_cols.append(col)
        elif merged[col].dtype == 'object':
            if merged[col].nunique() < 50 and merged[col].notna().sum() > 10:
                categorical_cols.append(col)
    
    print(f"Numeric covariates: {len(numeric_cols)}")
    print(f"Categorical covariates: {len(categorical_cols)}")
    print()
    
    # Analyze numeric relationships
    print("=" * 80)
    print("NUMERIC COVARIATE ANALYSIS")
    print("=" * 80)
    print()
    
    numeric_results = []
    for col in numeric_cols:
        if col not in merged.columns:
            continue
        data = merged[[col, 'distance']].dropna()
        if len(data) < 10:
            continue
        
        corr, pval = stats.pearsonr(data[col], data['distance'])
        numeric_results.append({
            'covariate': col,
            'correlation': corr,
            'p_value': pval,
            'n': len(data),
            'mean': data[col].mean(),
            'std': data[col].std()
        })
    
    numeric_df = pd.DataFrame(numeric_results)
    numeric_df = numeric_df.sort_values('correlation', key=abs, ascending=False)
    
    print("Top 20 numeric covariates by absolute correlation with distance:")
    print(numeric_df.head(20).to_string(index=False))
    print()
    
    # Analyze categorical relationships
    print("=" * 80)
    print("CATEGORICAL COVARIATE ANALYSIS")
    print("=" * 80)
    print()
    
    categorical_results = []
    for col in categorical_cols:
        if col not in merged.columns:
            continue
        data = merged[[col, 'distance']].dropna()
        if len(data) < 10:
            continue
        
        # One-way ANOVA or Kruskal-Wallis
        groups = [group['distance'].values for name, group in data.groupby(col) if len(group) >= 5]
        if len(groups) < 2:
            continue
        
        try:
            f_stat, pval = stats.f_oneway(*groups)
            test_name = 'ANOVA'
        except:
            try:
                h_stat, pval = stats.kruskal(*groups)
                test_name = 'Kruskal-Wallis'
            except:
                continue
        
        # Compute effect size (eta-squared approximation)
        overall_mean = data['distance'].mean()
        between_var = sum(len(g) * (g.mean() - overall_mean)**2 for g in groups) / (len(groups) - 1) if len(groups) > 1 else 0
        total_var = data['distance'].var()
        eta_sq = between_var / total_var if total_var > 0 else 0
        
        categorical_results.append({
            'covariate': col,
            'test': test_name,
            'p_value': pval,
            'effect_size': eta_sq,
            'n': len(data),
            'n_categories': len(groups),
            'mean_distance': data['distance'].mean(),
            'std_distance': data['distance'].std()
        })
    
    categorical_df = pd.DataFrame(categorical_results)
    categorical_df = categorical_df.sort_values('effect_size', ascending=False)
    
    print("Top 20 categorical covariates by effect size:")
    print(categorical_df.head(20).to_string(index=False))
    print()
    
    # Machine learning feature importance
    print("=" * 80)
    print("MACHINE LEARNING FEATURE IMPORTANCE")
    print("=" * 80)
    print()
    
    # Prepare features
    feature_cols = []
    X_data = []
    y_data = []
    
    for idx, row in merged.iterrows():
        if pd.isna(row['distance']):
            continue
        
        features = []
        feature_names = []
        
        # Add numeric features
        for col in numeric_cols[:30]:  # Limit to top 30 to avoid overfitting
            if col in merged.columns and pd.notna(row[col]):
                features.append(row[col])
                feature_names.append(col)
            else:
                features.append(0)
                feature_names.append(col)
        
        # Add categorical features (one-hot encoded)
        for col in categorical_cols[:20]:  # Limit to top 20
            if col in merged.columns and pd.notna(row[col]):
                # Simple encoding: use category index
                categories = merged[col].dropna().unique()
                cat_idx = np.where(categories == row[col])[0]
                if len(cat_idx) > 0:
                    features.append(cat_idx[0])
                else:
                    features.append(0)
            else:
                features.append(0)
            feature_names.append(col)
        
        X_data.append(features)
        y_data.append(row['distance'])
        if len(feature_cols) == 0:
            feature_cols = feature_names
    
    if len(X_data) > 20:  # Need sufficient data
        X = np.array(X_data)
        y = np.array(y_data)
        
        # Remove features with no variance
        feature_mask = X.std(axis=0) > 1e-6
        X = X[:, feature_mask]
        feature_cols_filtered = [f for f, m in zip(feature_cols, feature_mask) if m]
        
        # Train random forest
        rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        
        # Get feature importance
        importance_df = pd.DataFrame({
            'feature': feature_cols_filtered,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("Top 30 features by Random Forest importance:")
        print(importance_df.head(30).to_string(index=False))
        print()
        
        # Model performance
        r2 = rf.score(X, y)
        print(f"Random Forest R²: {r2:.4f}")
        print()
    
    # Save results
    results_path = Path('outputs/diagnostics/correlation_predictor_analysis_keyword_filters.csv')
    all_results = pd.concat([
        numeric_df.assign(type='numeric'),
        categorical_df.assign(type='categorical')
    ], ignore_index=True)
    all_results.to_csv(results_path, index=False)
    print(f"Saved analysis results to: {results_path}")
    print()
    
    # Create summary report
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print(f"Total companies analyzed: {len(matched)}")
    print(f"Significant numeric predictors (p < 0.05): {len(numeric_df[numeric_df['p_value'] < 0.05])}")
    print(f"Significant categorical predictors (p < 0.05): {len(categorical_df[categorical_df['p_value'] < 0.05])}")
    print()
    
    if len(numeric_df) > 0:
        print("Top 5 numeric predictors:")
        for _, row in numeric_df.head(5).iterrows():
            print(f"  {row['covariate']:40s} | r={row['correlation']:6.3f} | p={row['p_value']:.4f}")
        print()
    
    if len(categorical_df) > 0:
        print("Top 5 categorical predictors:")
        for _, row in categorical_df.head(5).iterrows():
            print(f"  {row['covariate']:40s} | η²={row['effect_size']:.3f} | p={row['p_value']:.4f}")
        print()

if __name__ == '__main__':
    main()
