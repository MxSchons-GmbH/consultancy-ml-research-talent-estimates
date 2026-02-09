import pandas as pd
import re

def clean_and_split_csv(df, affiliation_column=None):
    """
    Clean CSV data and split comma/forward slash separated values into separate rows.

    Parameters:
    df (pd.DataFrame): Input DataFrame
    affiliation_column (str): Name of the column containing affiliations to split.
                            If None, will attempt to auto-detect or use first column.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame with split affiliations
    """

    print(f"Original data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # Auto-detect affiliation column if not specified
    if affiliation_column is None:
        # Look for common affiliation column names
        common_names = ['affiliation', 'company', 'organization', 'org', 'institution']
        for col in df.columns:
            if any(name in col.lower() for name in common_names):
                affiliation_column = col
                break

        # If still not found, use the first column
        if affiliation_column is None:
            affiliation_column = df.columns[0]

    print(f"Using column '{affiliation_column}' for splitting")

    # Step 1: Clean the data
    print("Cleaning data...")

    def clean_text(text):
        if pd.isna(text):
            return text

        # Convert to string
        text = str(text)

        # Remove extra whitespace (multiple spaces, tabs, newlines)
        text = re.sub(r'\s+', ' ', text)

        # Strip leading/trailing whitespace
        text = text.strip()

        # Remove any weird characters that might cause issues
        text = re.sub(r'[^\w\s,.-]', '', text)

        return text

    # Clean all string columns
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].apply(clean_text)

    # Step 2: Split rows with comma and forward slash separated values
    print("Splitting comma and forward slash separated values...")

    # Create a list to store all the new rows
    new_rows = []

    for idx, row in df.iterrows():
        affiliation_value = row[affiliation_column]

        if pd.isna(affiliation_value):
            # Keep rows with missing values as-is
            new_rows.append(row)
        else:
            # Split by both comma and forward slash
            # First split by comma, then split each result by forward slash
            affiliations = []
            comma_splits = [aff.strip() for aff in str(affiliation_value).split(',')]

            for comma_split in comma_splits:
                slash_splits = [aff.strip() for aff in comma_split.split('/')]
                affiliations.extend(slash_splits)

            # Remove empty strings after splitting
            affiliations = [aff for aff in affiliations if aff]

            if len(affiliations) <= 1:
                # No splitting needed
                new_rows.append(row)
            else:
                # Create a new row for each affiliation
                for affiliation in affiliations:
                    new_row = row.copy()
                    new_row[affiliation_column] = affiliation
                    new_rows.append(new_row)

    # Create new DataFrame from the list of rows
    df_cleaned = pd.DataFrame(new_rows)

    # Reset index
    df_cleaned = df_cleaned.reset_index(drop=True)

    print(f"Cleaned data shape: {df_cleaned.shape}")
    print(f"Rows increased by: {df_cleaned.shape[0] - df.shape[0]}")

    print("Done!")

    # Show sample of the results
    print("\nSample of original data:")
    print(df.head())
    print("\nSample of cleaned data:")
    print(df_cleaned.head())

    return df_cleaned

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
        if pd.notna(affiliation_value) and (',' in str(affiliation_value) or '/' in str(affiliation_value)):
            # Split by both comma and forward slash (same logic as main function)
            affiliations = []
            comma_splits = [aff.strip() for aff in str(affiliation_value).split(',')]

            for comma_split in comma_splits:
                slash_splits = [aff.strip() for aff in comma_split.split('/')]
                affiliations.extend(slash_splits)

            affiliations = [aff for aff in affiliations if aff]

            if len(affiliations) > 1:
                print(f"Row {idx}:")
                print(f"  Original: {affiliation_value}")
                print(f"  Will split into: {affiliations}")
                print()
                preview_count += 1

# Example usage:
if __name__ == "__main__":
    print("To use this script:")
    print("1. Load your DataFrame")
    print("2. Call clean_and_split_csv(df) or preview_changes(df)")
    print()
    print("Examples of what will be split:")
    print("- 'Company A, Company B' → 'Company A' and 'Company B'")
    print("- 'Company A/Company B' → 'Company A' and 'Company B'")
    print("- 'Company A, Company B/Company C' → 'Company A', 'Company B', and 'Company C'")

