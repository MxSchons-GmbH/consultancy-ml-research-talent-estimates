import pandas as pd
import numpy as np

# 1) Filter for max_headcount == True AND max_population == True
def create_boolean_mask(df, column_name):
    """Helper function to create boolean mask from column, robust to string/boolean types"""
    if column_name not in df.columns:
        return None

    col = df[column_name]
    if col.dtype == bool or col.dtype.name == "bool":
        return col
    else:
        return col.astype(str).str.strip().str.lower().isin(["true","t","1","yes"])

def debias_ml_headcount_estimates(df):
    """Debias ML headcount estimates and compute robust consensus with uncertainty
    
    Args:
        df (pd.DataFrame): DataFrame with ML headcount estimates
        
    Returns:
        dict: Dictionary containing debiased DataFrame and filtered subsets
    """
    # Make a copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    METHOD_COLS = [
        "filter_broad_yes",
        "filter_strict_no",
        "filter_broad_yes_strict_no",
        "claude_total_accepted",
        "gpt5_total_accepted",
        "gemini_total_accepted",
    ]

    TOTAL_COL = "Total Headcount"   # authoritative total headcount
    EPSILON = 0.5                   # used to replace zeros/nonpositive before log

    orig_cols = df.columns.tolist()

    missing = [c for c in METHOD_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected method columns: {missing}")

    # ----------------------------
    # Numeric & adjusted copies (keep originals untouched)
    # ----------------------------
    NUM_METHOD_COLS = [f"{c}__num" for c in METHOD_COLS]
    ADJ_METHOD_COLS = [f"{c}__adj" for c in METHOD_COLS]

    for c in METHOD_COLS:
        df[f"{c}__num"] = pd.to_numeric(df[c], errors="coerce")
        df[f"{c}__zero_replaced"] = (df[f"{c}__num"] <= 0).fillna(False)
        df[f"{c}__adj"] = df[f"{c}__num"].mask(df[f"{c}__num"] <= 0, EPSILON)

    # Count available methods per row before any replacement impacts logs
    df["k_methods"] = df[NUM_METHOD_COLS].notna().sum(axis=1)

    # ----------------------------
    # Log transform on the adjusted copies
    # ----------------------------
    X = np.log(df[ADJ_METHOD_COLS].to_numpy(dtype=float))  # shape (N, M), NaNs preserved

    # Within-company robust center: median across methods (row-wise)
    m_i = np.nanmedian(X, axis=1)  # shape (N,)
    # Residuals r_ij = x_ij - m_i
    R = X - m_i[:, None]           # broadcast

    # Per-method bias α_j = median_i r_ij (column-wise median)
    alpha = np.nanmedian(R, axis=0)   # shape (M,)

    # Debiased logs: y_ij = x_ij - α_j
    Y = X - alpha[None, :]

    # ----------------------------
    # Consensus and uncertainty
    # ----------------------------
    # Point estimate μ_i = row-wise median of Y
    mu = np.nanmedian(Y, axis=1)

    # 10th and 90th percentiles in log space (ignore NaNs)
    q10 = np.nanquantile(Y, 0.10, axis=1, method="linear", keepdims=False)
    q90 = np.nanquantile(Y, 0.90, axis=1, method="linear", keepdims=False)

    # Back to counts
    ml_consensus     = np.exp(mu)
    ml_lower80       = np.exp(q10)
    ml_upper80       = np.exp(q90)

    # Rounded integer versions (nullable Int64 to preserve NaNs)
    df["ml_consensus_round"] = pd.Series(np.rint(ml_consensus)).astype("Int64")
    df["ml_lower80_round"]   = pd.Series(np.rint(ml_lower80)).astype("Int64")
    df["ml_upper80_round"]   = pd.Series(np.rint(ml_upper80)).astype("Int64")

    # Uncertainty factor ×/÷ = upper / lower
    df["uncertainty_factor_x"] = ml_upper80 / ml_lower80

    # Outlier flag if any debiased estimate differs from consensus by > 3×
    log3 = np.log(3.0)
    abs_dev = np.abs(Y - mu[:, None])
    df["outlier_flag_gt3x"] = np.nanmax(abs_dev, axis=1) > log3

    # ----------------------------
    # Per-method debiased counts (transparency)
    # ----------------------------
    debiased_counts = np.exp(Y)  # same shape as X
    for j, c in enumerate(METHOD_COLS):
        df[f"debiased__{c}"] = debiased_counts[:, j]

    # ----------------------------
    # Shares using TOTAL_COL (rounded counts; FRACTIONS, not %)
    # ----------------------------
    if TOTAL_COL in df.columns:
        total_numeric = pd.to_numeric(df[TOTAL_COL], errors="coerce")
        valid_total = (total_numeric > 0)

        # Use rounded integer estimates (convert to float so NaNs propagate cleanly)
        num = df["ml_consensus_round"].astype("Float64")
        lo  = df["ml_lower80_round"].astype("Float64")
        hi  = df["ml_upper80_round"].astype("Float64")

        ml_share         = np.where(valid_total, num.to_numpy() / total_numeric.to_numpy(), np.nan)
        ml_share_lower80 = np.where(valid_total, lo.to_numpy()  / total_numeric.to_numpy(), np.nan)
        ml_share_upper80 = np.where(valid_total, hi.to_numpy()  / total_numeric.to_numpy(), np.nan)

        df["ml_share"]         = ml_share          # fraction, not %
        df["ml_share_lower80"] = ml_share_lower80  # fraction, not %
        df["ml_share_upper80"] = ml_share_upper80  # fraction, not %

    # ----------------------------
    # Prepare export columns
    # ----------------------------
    appended_cols = [
        "ml_consensus_round",
        "ml_lower80_round",
        "ml_upper80_round",
        "uncertainty_factor_x",
        "k_methods",
        "outlier_flag_gt3x",
    ]

    for c in ["ml_share", "ml_share_lower80", "ml_share_upper80"]:
        if c in df.columns:
            appended_cols.append(c)

    # Per-method debiased columns
    appended_cols += [f"debiased__{c}" for c in METHOD_COLS]
    # Zero replacement flags
    appended_cols += [f"{c}__zero_replaced" for c in METHOD_COLS]
    # (Optional but handy) include numeric + adjusted copies for auditing
    appended_cols += [f"{c}__num" for c in METHOD_COLS]
    appended_cols += [f"{c}__adj" for c in METHOD_COLS]

    export_cols = orig_cols + [c for c in appended_cols if c in df.columns]
    df_export = df[export_cols].copy()

    print(f"Processed {len(df)} rows with {len(export_cols)} columns")

    # ----------------------------
    # Create filtered subsets
    # ----------------------------
    results = {"all_orgs": df_export}

    # Check for both required columns
    mask_headcount = create_boolean_mask(df, "max_headcount")
    mask_population = create_boolean_mask(df, "max_population")

    if mask_headcount is not None and mask_population is not None:
        # Combine both conditions with AND
        mask_max10 = mask_headcount & mask_population
        results["orgs_ML"] = df_export[mask_max10]
    else:
        print("Columns 'max_headcount' and/or 'max_population' not found — skipping ML filter.")

    # 2) Filter for Stage Reached == "5 - Work Trial"
    if "Stage Reached" in df.columns:
        results["orgs_stage5_work_trial"] = df_export[df_export["Stage Reached"].astype(str).str.strip() == "5 - Work Trial"]
    else:
        print("Column 'Stage Reached' not found — skipping stage 5 filter.")

    # 4) Extra scale-band exports based on ml_consensus_round (count) and ml_share (percent/fraction)
    def _to_numeric(series):
        """Coerce a series to numeric; invalids -> NaN."""
        return pd.to_numeric(series, errors="coerce")

    def _to_share_fraction(series):
        """Convert ml_share into a fraction in [0,1]."""
        def conv(x):
            if pd.isna(x):
                return np.nan
            if isinstance(x, str):
                s = x.strip().replace(",", ".")
                if s.endswith("%"):
                    try:
                        return float(s[:-1]) / 100.0
                    except:
                        return np.nan
                try:
                    val = float(s)
                except:
                    return np.nan
            else:
                try:
                    val = float(x)
                except:
                    return np.nan
            # If it's > 1, assume it's a percent value like "5" meaning 5%
            return val / 100.0 if val > 1.0 else val
        return series.apply(conv)

    # Check columns exist
    if ("ml_consensus_round" in df.columns) and ("ml_share" in df.columns):
        ml_count = _to_numeric(df["ml_consensus_round"])
        ml_share_frac = _to_share_fraction(df["ml_share"])

        # Thresholds:
        # Enterprise: count >= 500 AND share >= 0.5%
        # Midscale:   count >= 50  AND share >= 1%
        # Boutique:   count >= 10  AND share >= 5%
        m_enterprise = (ml_count >= 500) & (ml_share_frac >= 0.005)
        m_midscale   = (ml_count >= 50)  & (ml_share_frac >= 0.01)
        m_boutique   = (ml_count >= 10)  & (ml_share_frac >= 0.05) & (ml_count < 50)

        results["orgs_enterprise_500ml_0p5pct"] = df_export[m_enterprise]
        results["orgs_midscale_50ml_1pct"] = df_export[m_midscale]
        results["orgs_boutique_10ml_5pct"] = df_export[m_boutique]

        print("Created scale-banded filters:")
        print(f"- Enterprise: {m_enterprise.sum()} organizations")
        print(f"- Mid-scale: {m_midscale.sum()} organizations")
        print(f"- Boutique: {m_boutique.sum()} organizations")
    else:
        missing = [c for c in ["ml_consensus_round","ml_share"] if c not in df.columns]
        print(f"Columns {missing} not found — skipping scale-banded filters.")

    # Extra: Stage 5 AND category ∈ {"Recommendation", "Conditional Recommendation"}
    if ("Stage Reached" in df.columns) and ("category" in df.columns):
        sr = df["Stage Reached"].astype(str).str.strip()
        cat = df["category"].astype(str).str.strip()
        mask_stage5_rec = (sr == "5 - Work Trial") & (cat.isin(["Recommendation", "Conditional Recommendation"]))
        results["orgs_stage5_work_trial_recommended"] = df_export[mask_stage5_rec]
    else:
        print("Columns 'Stage Reached' and/or 'category' not found — skipping stage 5 recommended filter.")

    print(f"Created {len(results)} filtered datasets")
    return results

