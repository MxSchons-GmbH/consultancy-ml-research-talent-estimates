import pandas as pd
import re

def is_academic(aff):
    """Function to check if an affiliation looks academic"""
    if pd.isna(aff):
        return True  # Remove if NaN
    keywords = [
        # generic academic/research stems (catch most languages)
        'university', 'univ', 'universit', 'college',
        'institute', 'institut', 'inst',  # short but high-value
        'academy', 'faculty', 'facult', 'school',
        'department', 'depart', 'dept',
        'laboratory', 'laboratoire', 'collaboration',
        'centre', 'center',
        'observatory', 'observatoire',
        'polytechnic', 'polytechnique',
        'hochschule', 'forschung', 'physik', 'astronom',
        # national/government modifiers
        'national', 'state', 'federal',
        'space flight center',
        # headline research agencies & labs
        'cern', 'slac', 'infn', 'desy', 'cnrs', 'cea',
        'kek', 'riken', 'jaxa',
        'esa', 'nasa', 'jpl', 'gsfc', 'msfc',
        'fermilab', 'argonne', 'brookhaven', 'bnl',
        'los alamos', 'lanl', 'lawrence berkeley', 'lbnl', 'lbl',
        'nist', 'noaa', 'nrc',
        # big non-English institutes you'll see a lot
        'max-planck', 'max planck', 'niels bohr',
        'helmholtz', 'forschungszentrum', 'trieste', 'ictp'
    ]
    return any(kw in str(aff).lower() for kw in keywords)

def filter_academic_affiliations(df):
    """Processing of affiliations from ML conferences and arXiv
    
    Args:
        df (pd.DataFrame): DataFrame with 'affiliation' column
        
    Returns:
        pd.DataFrame: DataFrame with academic affiliations filtered out
    """
    # Filter out academic-looking affiliations
    df_non_academic = df[~df['affiliation'].apply(is_academic)]

    # Optional: Display the first few rows of the cleaned DataFrame
    print("Cleaned DataFrame preview:")
    print(df_non_academic.head())
    
    return df_non_academic

def preview_changes(df, affiliation_column=None, max_preview=10):
    """
    Preview what changes will be made without actually processing the file.
    
    Args:
        df (pd.DataFrame): DataFrame to preview
        affiliation_column (str, optional): Column name containing affiliations
        max_preview (int): Maximum number of preview rows to show
    """
    if affiliation_column is None:
        common_names = ['affiliation', 'company', 'organization', 'org', 'institution']
        for col in df.columns:
            if any(name in col.lower() for name in common_names):
                affiliation_column = col
                break
        if affiliation_column is None:
            affiliation_column = df.columns[0]

    print(f"Preview of changes for column '{affiliation_column}':")
    print("-" * 50)

    preview_count = 0
    for idx, row in df.iterrows():
        if preview_count >= max_preview:
            break

        affiliation_value = row[affiliation_column]
        if pd.notna(affiliation_value) and ',' in str(affiliation_value):
            affiliations = [aff.strip() for aff in str(affiliation_value).split(',')]
            if len(affiliations) > 1:
                print(f"Row {idx}:")
                print(f"  Original: {affiliation_value}")
                print(f"  Will split into: {affiliations}")
                print()
                preview_count += 1