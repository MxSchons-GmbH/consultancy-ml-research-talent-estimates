import pandas as pd
import numpy as np

def clean_llm_estimates_outliers(df):
    """
    Remove all LLM estimates for all the ones where the LLM total (claude_total_employees) 
    is ≤ or ≥ 3x the value of "Total Headcount"
    
    Args:
        df (pd.DataFrame): DataFrame with LLM estimates and Total Headcount columns
        
    Returns:
        pd.DataFrame: DataFrame with outlier LLM estimates removed
    """
    llm_cols = [
        "claude_total_employees","claude_total_rejected","claude_total_accepted",
        "gpt5_total_employees","gpt5_total_rejected","gpt5_total_accepted",
        "gemini_total_employees","gemini_total_rejected","gemini_total_accepted",
    ]

    c = pd.to_numeric(df["claude_total_employees"], errors="coerce")
    h = pd.to_numeric(df["Total Headcount"], errors="coerce")

    mask = (c >= 3*h) | (c <= h/3)
    df.loc[mask, llm_cols] = pd.NA

    return df

