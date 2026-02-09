import pandas as pd
import json

def extract_linkedin_titles_to_csv(data):
    """Extract LinkedIn job titles from JSON data
    
    Args:
        data (dict or list): LinkedIn JSON data (can be dict with 'profiles' key or list of profiles)
        
    Returns:
        tuple: (all_titles_df, last_titles_df) - DataFrames with all titles and last titles per profile
    """
    # 2. Load the JSON
    if isinstance(data, str):
        data = json.loads(data)

    # 3. Pull out profiles
    profiles = data.get('profiles', []) if isinstance(data, dict) else data

    # 4. Build rows for both CSVs
    all_rows = []
    last_rows = []
    base_url = "https://www.linkedin.com/in/"

    for prof in profiles:
        pid = prof.get('public_identifier', '')
        pid_url = f"{base_url}{pid}"
        exps = prof.get('profile_data', {}).get('experiences', [])

        # all titles: one row per experience
        for exp in exps:
            title = exp.get('title')
            if title:
                all_rows.append({
                    'public_identifier': pid_url,
                    'title': title
                })

        # last title: one row per profile
        last_title = exps[-1].get('title') if exps and exps[-1].get('title') else None
        last_rows.append({
            'public_identifier': pid_url,
            'title': last_title
        })

    # 5. Create DataFrames
    df_all = pd.DataFrame(all_rows)
    df_last = pd.DataFrame(last_rows)

    print(f"Extracted {len(df_all)} total titles and {len(df_last)} last titles")
    
    return df_all, df_last

