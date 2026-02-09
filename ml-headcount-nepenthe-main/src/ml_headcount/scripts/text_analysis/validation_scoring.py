import pandas as pd
import numpy as np
import re

def score_validation_cvs_with_filters(df, keyword_lists):
    """
    Score each CV in validation DataFrame with keyword filters.

    Args:
        df (pd.DataFrame): DataFrame with 'cv_text' column
        keyword_lists (dict): Dictionary with keyword lists containing:
            - strict_yes: List of strict yes keywords
            - strict_no: List of strict no keywords  
            - broad_yes: List of broad yes keywords
            - broad_no: List of broad no keywords

    Returns:
        pd.DataFrame: DataFrame with eight scoring columns added

    Eight numeric columns:
        strict_yes_strict_no
        strict_yes_broad_no
        broad_yes_broad_no
        broad_yes_strict_no
        strict_yes_only
        broad_yes_only
        strict_no_only
        broad_no_only

    * "*_yes_only"  – 1 ↦ text contains the YES-terms, regardless of any NO-terms.
    * "*_no_only"   – 1 ↦ text avoids the NO-terms (no YES requirement).

    Scores from LLMs already included in file
    """
    # Call the main processing function
    return main(df, keyword_lists)




# ── helpers ────────────────────────────────────────────────────────────────────

def _compile(words):
    """Whole-word OR-regex, case-insensitive."""
    return re.compile(r"\b(?:%s)\b" % "|".join(map(re.escape, words)), re.I)

def create_patterns(keyword_lists):
    """Create regex patterns from keyword lists."""
    patterns = {}
    for key, words in keyword_lists.items():
        patterns[key] = _compile(words)
    
    # sentinels
    patterns["_all"] = re.compile(r"")        # always matches
    patterns["_none"] = re.compile(r"(?!x)")   # never matches
    
    return patterns

# common base: must see "Machine Learning" or ML
COMMON_BASE = _compile(["machine learning", "machine‐learning", "ML", "deep learning",
                        "deep-learning", "reinforcement learning", "reinforcement-learning", "RL"])

def score(text: str, yes_key: str, no_key: str, patterns: dict) -> int:
    """YES-hit ∧ NO-miss ∧ COMMON_BASE‐hit  → 1, else 0."""
    text = text or ""
    return int(
        COMMON_BASE.search(text) is not None
        and patterns[yes_key].search(text) is not None
        and patterns[no_key].search(text) is None
    )

# ── main ───────────────────────────────────────────────────────────────────────

def main(df, keyword_lists):
    patterns = create_patterns(keyword_lists)

    # original four
    df["strict_yes_strict_no"] = df["cv_text"].apply(
        lambda t: score(t, "strict_yes", "strict_no", patterns))
    df["strict_yes_broad_no"] = df["cv_text"].apply(
        lambda t: score(t, "strict_yes", "broad_no", patterns))
    df["broad_yes_broad_no"] = df["cv_text"].apply(
        lambda t: score(t, "broad_yes", "broad_no", patterns))
    df["broad_yes_strict_no"] = df["cv_text"].apply(
        lambda t: score(t, "broad_yes", "strict_no", patterns))

    # new four
    df["strict_yes_only"] = df["cv_text"].apply(
        lambda t: score(t, "strict_yes", "_none", patterns))
    df["broad_yes_only"] = df["cv_text"].apply(
        lambda t: score(t, "broad_yes", "_none", patterns))
    df["strict_no_only"] = df["cv_text"].apply(
        lambda t: score(t, "_all", "strict_no", patterns))
    df["broad_no_only"] = df["cv_text"].apply(
        lambda t: score(t, "_all", "broad_no", patterns))

    print("Added eight scoring columns to DataFrame")
    return df

if __name__ == "__main__":
    print("To use this script:")
    print("1. Load your DataFrame with 'cv_text' column")
    print("2. Prepare keyword lists dictionary")
    print("3. Call score_validation_cvs_with_filters(df, keyword_lists)")

