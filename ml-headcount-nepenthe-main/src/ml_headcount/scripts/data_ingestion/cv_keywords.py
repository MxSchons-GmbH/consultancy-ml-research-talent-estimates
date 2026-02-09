import pandas as pd
import numpy as np
import re
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer

def extract_cv_keyword_frequencies(df):
    """Extract keyword frequencies from CV text data
    
    Args:
        df (pd.DataFrame): DataFrame with 'cv_text' column
        
    Returns:
        pd.DataFrame: DataFrame with keyword frequencies
    """
    if "cv_text" not in df.columns:
        raise KeyError("'cv_text' column not found in the DataFrame")

    # ----------------- 3. Build n-grams & count -----------------
    corpus = (
        df["cv_text"]
        .fillna("")               # handle missing
        .astype(str)
        .str.lower()              # lower-case everything up-front
        .tolist()
    )

    # Token pattern: words of ≥2 letters; remove digits/punct.
    token_pattern = r"\b[a-z]{2,}\b"

    vec = CountVectorizer(
        stop_words="english",        # built-in stop-word list
        ngram_range=(1, 2),          # 1-, 2-, 3-grams
        token_pattern=token_pattern
    )

    X = vec.fit_transform(corpus)     # sparse term-document matrix
    counts = np.asarray(X.sum(axis=0)).ravel()
    terms  = vec.get_feature_names_out()

    freq_df = (
        pd.DataFrame({"phrase": terms, "count": counts})
        .query("count > 4")                         # keep > 3 occurrences
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )

    # ----------------- 4. Display results -----------------
    print(f"Found {len(freq_df)} phrases occurring > 3 times\n")
    print(freq_df.head(200))          # show top 200

    return freq_df

