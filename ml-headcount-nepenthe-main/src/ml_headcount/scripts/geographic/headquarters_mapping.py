import pandas as pd
import numpy as np

def process_headquarters_with_subregions(df):
    """Process headquarters data and add UN M49 subregion mapping
    
    Args:
        df (pd.DataFrame): DataFrame with 'Headquarters Location' column
        
    Returns:
        pd.DataFrame: DataFrame with added 'Country' and 'Subregion' columns
    """
    result_df = process_headquarters_tsv(df)
    
    if result_df is not None:
        print(f"\nFirst 5 rows of processed data:")
        print(result_df[['Headquarters Location', 'Country', 'Subregion']].head())
    
    return result_df

def extract_country_from_location(location):
    """Extract country from headquarters location string."""
    if pd.isna(location) or location == "":
        return ""

    # Split by comma and take the last part as country
    parts = [part.strip() for part in str(location).split(',')]
    country = parts[-1] if parts else ""

    # Handle common variations and clean up
    country_variations = {
        'USA': 'United States',
        'US': 'United States',
        'UK': 'United Kingdom',
        'United Kingdom of Great Britain and Northern Ireland': 'United Kingdom',
        'The Netherlands': 'Netherlands',
        'NA - South Africa': 'South Africa',
        'NA - Vietnam': 'Vietnam',
        'NA - Uruguay': 'Uruguay',
        'Russian Federation': 'Russia',
    }

    return country_variations.get(country, country)

def map_country_to_subregion(country):
    """Map country to UN M49 subregion."""

    # UN M49 subregion mapping
    country_to_subregion = {
        # Northern America
        'United States': 'Northern America',
        'Canada': 'Northern America',
        'Bermuda': 'Northern America',
        'Greenland': 'Northern America',
        'Saint Pierre and Miquelon': 'Northern America',

        # Central America
        'Belize': 'Central America',
        'Costa Rica': 'Central America',
        'El Salvador': 'Central America',
        'Guatemala': 'Central America',
        'Honduras': 'Central America',
        'Mexico': 'Central America',
        'Nicaragua': 'Central America',
        'Panama': 'Central America',

        # South America
        'Argentina': 'South America',
        'Bolivia': 'South America',
        'Brazil': 'South America',
        'Chile': 'South America',
        'Colombia': 'South America',
        'Ecuador': 'South America',
        'French Guiana': 'South America',
        'Guyana': 'South America',
        'Paraguay': 'South America',
        'Peru': 'South America',
        'Suriname': 'South America',
        'Uruguay': 'South America',
        'Venezuela': 'South America',

        # Western Europe
        'Austria': 'Western Europe',
        'Belgium': 'Western Europe',
        'France': 'Western Europe',
        'Germany': 'Western Europe',
        'Liechtenstein': 'Western Europe',
        'Luxembourg': 'Western Europe',
        'Monaco': 'Western Europe',
        'Netherlands': 'Western Europe',
        'Switzerland': 'Western Europe',

        # Northern Europe
        'Denmark': 'Northern Europe',
        'Estonia': 'Northern Europe',
        'Finland': 'Northern Europe',
        'Iceland': 'Northern Europe',
        'Ireland': 'Northern Europe',
        'Latvia': 'Northern Europe',
        'Lithuania': 'Northern Europe',
        'Norway': 'Northern Europe',
        'Sweden': 'Northern Europe',
        'United Kingdom': 'Northern Europe',

        # Eastern Europe
        'Belarus': 'Eastern Europe',
        'Bulgaria': 'Eastern Europe',
        'Czech Republic': 'Eastern Europe',
        'Czechia': 'Eastern Europe',
        'Hungary': 'Eastern Europe',
        'Poland': 'Eastern Europe',
        'Moldova': 'Eastern Europe',
        'Romania': 'Eastern Europe',
        'Russia': 'Eastern Europe',
        'Russian Federation': 'Eastern Europe',
        'Slovakia': 'Eastern Europe',
        'Ukraine': 'Eastern Europe',

        # Southern Europe
        'Albania': 'Southern Europe',
        'Andorra': 'Southern Europe',
        'Bosnia and Herzegovina': 'Southern Europe',
        'Croatia': 'Southern Europe',
        'Gibraltar': 'Southern Europe',
        'Greece': 'Southern Europe',
        'Italy': 'Southern Europe',
        'Malta': 'Southern Europe',
        'Montenegro': 'Southern Europe',
        'North Macedonia': 'Southern Europe',
        'Portugal': 'Southern Europe',
        'San Marino': 'Southern Europe',
        'Serbia': 'Southern Europe',
        'Slovenia': 'Southern Europe',
        'Spain': 'Southern Europe',
        'Vatican City': 'Southern Europe',

        # Eastern Asia
        'China': 'Eastern Asia',
        'Hong Kong': 'Eastern Asia',
        'Japan': 'Eastern Asia',
        'Macao': 'Eastern Asia',
        'Mongolia': 'Eastern Asia',
        'North Korea': 'Eastern Asia',
        'South Korea': 'Eastern Asia',
        'Taiwan': 'Eastern Asia',

        # South-Eastern Asia
        'Brunei': 'South-Eastern Asia',
        'Cambodia': 'South-Eastern Asia',
        'Indonesia': 'South-Eastern Asia',
        'Laos': 'South-Eastern Asia',
        'Malaysia': 'South-Eastern Asia',
        'Myanmar': 'South-Eastern Asia',
        'Philippines': 'South-Eastern Asia',
        'Singapore': 'South-Eastern Asia',
        'Thailand': 'South-Eastern Asia',
        'Timor-Leste': 'South-Eastern Asia',
        'Vietnam': 'South-Eastern Asia',

        # Southern Asia
        'Afghanistan': 'Southern Asia',
        'Bangladesh': 'Southern Asia',
        'Bhutan': 'Southern Asia',
        'India': 'Southern Asia',
        'Iran': 'Southern Asia',
        'Maldives': 'Southern Asia',
        'Nepal': 'Southern Asia',
        'Pakistan': 'Southern Asia',
        'Sri Lanka': 'Southern Asia',

        # Western Asia
        'Armenia': 'Western Asia',
        'Azerbaijan': 'Western Asia',
        'Bahrain': 'Western Asia',
        'Cyprus': 'Western Asia',
        'Georgia': 'Western Asia',
        'Iraq': 'Western Asia',
        'Israel': 'Western Asia',
        'Jordan': 'Western Asia',
        'Kuwait': 'Western Asia',
        'Lebanon': 'Western Asia',
        'Oman': 'Western Asia',
        'Qatar': 'Western Asia',
        'Saudi Arabia': 'Western Asia',
        'Syria': 'Western Asia',
        'Turkey': 'Western Asia',
        'United Arab Emirates': 'Western Asia',
        'Yemen': 'Western Asia',

        # Central Asia
        'Kazakhstan': 'Central Asia',
        'Kyrgyzstan': 'Central Asia',
        'Tajikistan': 'Central Asia',
        'Turkmenistan': 'Central Asia',
        'Uzbekistan': 'Central Asia',

        # Northern Africa
        'Algeria': 'Northern Africa',
        'Egypt': 'Northern Africa',
        'Libya': 'Northern Africa',
        'Morocco': 'Northern Africa',
        'Sudan': 'Northern Africa',
        'Tunisia': 'Northern Africa',
        'Western Sahara': 'Northern Africa',

        # Eastern Africa
        'Burundi': 'Eastern Africa',
        'Comoros': 'Eastern Africa',
        'Djibouti': 'Eastern Africa',
        'Eritrea': 'Eastern Africa',
        'Ethiopia': 'Eastern Africa',
        'Kenya': 'Eastern Africa',
        'Madagascar': 'Eastern Africa',
        'Malawi': 'Eastern Africa',
        'Mauritius': 'Eastern Africa',
        'Mozambique': 'Eastern Africa',
        'Rwanda': 'Eastern Africa',
        'Seychelles': 'Eastern Africa',
        'Somalia': 'Eastern Africa',
        'South Sudan': 'Eastern Africa',
        'Tanzania': 'Eastern Africa',
        'Uganda': 'Eastern Africa',
        'Zambia': 'Eastern Africa',
        'Zimbabwe': 'Eastern Africa',

        # Southern Africa
        'Botswana': 'Southern Africa',
        'Eswatini': 'Southern Africa',
        'Lesotho': 'Southern Africa',
        'Namibia': 'Southern Africa',
        'South Africa': 'Southern Africa',

        # Western Africa
        'Benin': 'Western Africa',
        'Burkina Faso': 'Western Africa',
        'Cabo Verde': 'Western Africa',
        'Côte d\'Ivoire': 'Western Africa',
        'Gambia': 'Western Africa',
        'Ghana': 'Western Africa',
        'Guinea': 'Western Africa',
        'Guinea-Bissau': 'Western Africa',
        'Liberia': 'Western Africa',
        'Mali': 'Western Africa',
        'Mauritania': 'Western Africa',
        'Niger': 'Western Africa',
        'Nigeria': 'Western Africa',
        'Senegal': 'Western Africa',
        'Sierra Leone': 'Western Africa',
        'Togo': 'Western Africa',

        # Middle Africa
        'Angola': 'Middle Africa',
        'Cameroon': 'Middle Africa',
        'Central African Republic': 'Middle Africa',
        'Chad': 'Middle Africa',
        'Congo': 'Middle Africa',
        'Democratic Republic of the Congo': 'Middle Africa',
        'Equatorial Guinea': 'Middle Africa',
        'Gabon': 'Middle Africa',
        'São Tomé and Príncipe': 'Middle Africa',

        # Australia and New Zealand
        'Australia': 'Australia and New Zealand',
        'New Zealand': 'Australia and New Zealand',

        # Caribbean
        'Antigua and Barbuda': 'Caribbean',
        'Bahamas': 'Caribbean',
        'Barbados': 'Caribbean',
        'Cuba': 'Caribbean',
        'Dominica': 'Caribbean',
        'Dominican Republic': 'Caribbean',
        'Grenada': 'Caribbean',
        'Haiti': 'Caribbean',
        'Jamaica': 'Caribbean',
        'Saint Kitts and Nevis': 'Caribbean',
        'Saint Lucia': 'Caribbean',
        'Saint Vincent and the Grenadines': 'Caribbean',
        'Trinidad and Tobago': 'Caribbean',

        # Melanesia
        'Fiji': 'Melanesia',
        'New Caledonia': 'Melanesia',
        'Papua New Guinea': 'Melanesia',
        'Solomon Islands': 'Melanesia',
        'Vanuatu': 'Melanesia',

        # Micronesia
        'Guam': 'Micronesia',
        'Kiribati': 'Micronesia',
        'Marshall Islands': 'Micronesia',
        'Micronesia': 'Micronesia',
        'Nauru': 'Micronesia',
        'Northern Mariana Islands': 'Micronesia',
        'Palau': 'Micronesia',

        # Polynesia
        'American Samoa': 'Polynesia',
        'Cook Islands': 'Polynesia',
        'French Polynesia': 'Polynesia',
        'Niue': 'Polynesia',
        'Samoa': 'Polynesia',
        'Tonga': 'Polynesia',
        'Tuvalu': 'Polynesia',
    }

    return country_to_subregion.get(country, 'Unknown')

def process_headquarters_tsv(df):
    """
    Process DataFrame with headquarters data and add UN M49 subregion column.

    Args:
        df (pd.DataFrame): DataFrame with headquarters data

    Returns:
        pandas.DataFrame: Processed dataframe with subregion column added
    """

    print(f"Processing {len(df)} rows")

    # Check if 'Headquarters Location' column exists
    if 'Headquarters Location' not in df.columns:
        print("Error: 'Headquarters Location' column not found in the DataFrame.")
        print(f"Available columns: {list(df.columns)}")
        return None

    # Extract countries and map to subregions
    print("Processing headquarters locations...")
    df['Country'] = df['Headquarters Location'].apply(extract_country_from_location)
    df['Subregion'] = df['Country'].apply(map_country_to_subregion)

    # Generate summary statistics
    total_rows = len(df)
    unknown_count = len(df[df['Subregion'] == 'Unknown'])
    success_rate = (total_rows - unknown_count) / total_rows * 100

    print(f"\nProcessing Summary:")
    print(f"Total locations processed: {total_rows}")
    print(f"Successfully mapped: {total_rows - unknown_count}")
    print(f"Unknown mappings: {unknown_count}")
    print(f"Success rate: {success_rate:.1f}%")

    # Show subregion distribution
    print(f"\nSubregion distribution:")
    subregion_counts = df['Subregion'].value_counts()
    for subregion, count in subregion_counts.items():
        print(f"  {subregion}: {count}")

    # Show unknown countries if any
    if unknown_count > 0:
        print(f"\nUnknown countries found:")
        unknown_countries = df[df['Subregion'] == 'Unknown']['Country'].unique()
        for country in unknown_countries:
            print(f"  {country}")

    return df

# Example usage
if __name__ == "__main__":
    print("To use this script:")
    print("1. Load your DataFrame with 'Headquarters Location' column")
    print("2. Call process_headquarters_with_subregions(df)")
    print("3. The function will return a DataFrame with added 'Country' and 'Subregion' columns")

