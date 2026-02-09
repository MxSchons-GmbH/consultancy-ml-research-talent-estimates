import pandas as pd

def split_keywords_into_strict_broad_lists(df):
    """Split keywords DataFrame into strict/broad & yes/no lists.

    Args:
        df (pd.DataFrame): DataFrame with keyword data
        
    Expected columns in df:
    --------------------------------
    - category                  : 'yes' | 'no'
    - keyword                   : term itself
    - total_raw_frequency       : integer count
    - discriminative_score      : float 0-1
    - raw_category_specificity  : float 0-1

    Rules
    -----
    All groups : total_raw_frequency ≥ 5
    Broad      : discriminative_score ≥ 0.8  AND raw_category_specificity ≥ 0.6
    Strict     : discriminative_score ≥ 0.9  AND raw_category_specificity ≥ 0.8
    
    Returns:
        dict: Dictionary with keyword lists
    """

    # 1 – baseline filter on frequency
    df = df[df["total_raw_frequency"] >= 5]

    # 2 – strict vs. broad masks
    broad_mask  = (df["discriminative_score"] >= 0.8) & (df["raw_category_specificity"] >= 0.7)
    strict_mask = (df["discriminative_score"] >= 0.9) & (df["raw_category_specificity"] >= 0.8)

    broad  = df[broad_mask]
    strict = df[strict_mask]

    def to_list(series):
        """Return Python-list literal with quoted keywords."""
        return [*series.astype(str)]

    strict_yes = to_list(strict[strict["category"].str.lower() == "yes"]["keyword"])
    strict_no  = to_list(strict[strict["category"].str.lower() == "no"]["keyword"])
    broad_yes  = to_list(broad[broad["category"].str.lower() == "yes"]["keyword"])
    broad_no   = to_list(broad[broad["category"].str.lower() == "no"]["keyword"])

    print("strict_yes =", strict_yes)
    print("strict_no  =", strict_no)
    print("broad_yes  =", broad_yes)
    print("broad_no   =", broad_no)

    print("(" + " OR ".join(f'"{kw}"' for kw in strict_yes) + ")")
    print("(" + " OR ".join(f'"{kw}"' for kw in strict_no) + ")")
    print("(" + " OR ".join(f'"{kw}"' for kw in broad_yes) + ")")
    print("(" + " OR ".join(f'"{kw}"' for kw in broad_no) + ")")
    
    return {
        "strict_yes": strict_yes,
        "strict_no": strict_no,
        "broad_yes": broad_yes,
        "broad_no": broad_no
    }

