import pandas as pd
import numpy as np
from pathlib import Path

# Constants
PCT_DECIMALS = 1

# Size sections
SIZE_SECTIONS = [
    ("Small (< 100 employees)",   lambda hc: hc < 100),
    ("Medium (100-999 employees)", lambda hc: (hc >= 100) & (hc <= 999)),
    ("Large (1000-9999 employees)",lambda hc: (hc >= 1000) & (hc <= 9999)),
    ("Giant (>10.000 employees)",  lambda hc: hc >= 10000),
]

# Formatting controls
PCT_DECIMALS = 1   # for appended percentages and ML % in sections
INDENT = "\u00A0" * 6  # non-breaking spaces so Sheets/Excel keep indentation

# Regions in the requested order; include both "Unknown" and "Unkown"
REGION_ORDER = [
    "Northern America",
    "Western Europe",
    "Southern Asia",
    "Eastern Asia",
    "Northern Europe",
    "Eastern Europe",
    "Western Asia",
    "South America",
    "Southern Europe",
    "Unknown",
    "Unkown",  # keep as its own bucket
    "Australia and New Zealand",
    "South-Eastern Asia",
    "Central America",
    "Southern Africa",
    "Northern Africa",
    "Western Africa",
    "Caribbean",
]

def create_combined_summary_statistics(dataframes, labels=None):
    """
    Create combined summary statistics for ML datasets.
    
    Args:
        dataframes (list): List of DataFrames with organizational data
        labels (list, optional): List of labels for each DataFrame. If None, uses default labels.
        
    Returns:
        pd.DataFrame: Combined summary statistics
    """
    # Call the main processing function
    return main(dataframes, labels)

def to_numeric_clean(series):
    """Coerce to numeric after stripping commas and spaces."""
    if series.dtype.kind in ("i", "u", "f"):
        return pd.to_numeric(series, errors="coerce")
    s = series.astype(str).str.replace(r"[,\s]", "", regex=True)
    return pd.to_numeric(s, errors="coerce")

def extract_year(series):
    """Extract year from 'Founded Date' if present; used for 'Total' only."""
    dt = pd.to_datetime(series, errors="coerce")
    years_from_dt = dt.dt.year
    as_num = pd.to_numeric(series, errors="coerce")
    years = years_from_dt.fillna(as_num)
    years = years.where((years >= 1700) & (years <= 2100))
    return years.astype("Int64")

def format_int_iso(n):
    """Format integers with spaces between groups of three digits."""
    if n is None or (isinstance(n, float) and not np.isfinite(n)) or pd.isna(n):
        return ""
    n = int(n)
    sign = "-" if n < 0 else ""
    s = str(abs(n))
    groups = []
    while s:
        groups.append(s[-3:])
        s = s[:-3]
    return sign + " ".join(reversed(groups))

def pct_string(numer, denom, decimals=PCT_DECIMALS):
    if denom is None or denom == 0 or not np.isfinite(denom) or pd.isna(denom):
        return "n/a"
    pct = 100.0 * float(numer) / float(denom)
    return f"{pct:.{decimals}f} %"

def _ensure_fraction_share(series):
    """
    Ensure shares are fractions in [0,1]. If most values look like percents (>1),
    convert by dividing by 100.
    """
    s = pd.to_numeric(series, errors="coerce")
    vals = s.dropna()
    if len(vals) == 0:
        return s
    if (vals > 1.0).mean() > 0.5:
        return s / 100.0
    return s

def _weighted_mean(values, weights):
    """Headcount-weighted mean, ignoring rows with NaNs or zero total weight."""
    if values is None or weights is None:
        return None
    mask = values.notna() & weights.notna()
    if not mask.any():
        return None
    v = values[mask].astype(float)
    w = weights[mask].astype(float)
    wsum = float(w.sum())
    if wsum <= 0:
        return None
    return float((v * w).sum()) / wsum

def pct_from_fraction(frac, decimals=PCT_DECIMALS):
    """Format a fraction (e.g., 0.123) as a percent string with given decimals."""
    if frac is None or not np.isfinite(frac):
        return ""
    return f"{100.0 * float(frac):.{decimals}f} %"


def read_one_tsv(df, region_order=None):
    required_cols = [
        "Total Headcount",
        "Founded Date",
        "ml_consensus_round",
        "ml_lower80_round",
        "ml_upper80_round",
        "Subregion",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Coerce numerics
    df["Total Headcount"]     = to_numeric_clean(df["Total Headcount"])
    df["ml_consensus_round"]  = to_numeric_clean(df["ml_consensus_round"])
    df["ml_lower80_round"]    = to_numeric_clean(df["ml_lower80_round"])
    df["ml_upper80_round"]    = to_numeric_clean(df["ml_upper80_round"])
    df["Founded_Year"]        = extract_year(df["Founded Date"])

    # Normalize Subregion; keep both 'Unknown' and 'Unkown'; non-listed -> 'Unknown'
    sub = df["Subregion"].astype(str).replace(
        {None: "Unknown", "nan": "Unknown", "NaN": "Unknown", "": "Unknown"}
    )
    if region_order:
        sub = sub.where(sub.isin(region_order), other="Unknown")
    df["Subregion"] = sub
    return df

def section_metrics(df, mask=None, include_total_employees=False, include_median_year=False, total_n_for_pct=None):
    """
    Compute formatted metrics for a section.
    Returns a dict of row_label -> formatted string.
    """
    if mask is None:
        df_sub = df
    else:
        df_sub = df[mask]

    total_n_all = int(len(df)) if total_n_for_pct is None else int(total_n_for_pct)
    org_n = int(len(df_sub))
    org_n_str = f"{format_int_iso(org_n)} / {format_int_iso(total_n_all)} ({pct_string(org_n, total_n_all)})"

    out = {}
    out["Organization N"] = org_n_str

    # Total employees (only for overall 'Total' section)
    if include_total_employees:
        hc_all = df_sub["Total Headcount"]
        total_employees_num = int(hc_all.dropna().sum()) if hc_all.notna().any() else 0
        out["Total employees"] = format_int_iso(total_employees_num)
    else:
        total_employees_num = int(df_sub["Total Headcount"].dropna().sum()) if df_sub["Total Headcount"].notna().any() else 0

    # Median founding year (only for 'Total')
    if include_median_year:
        fy = df_sub["Founded_Year"].dropna()
        med_year = int(np.round(fy.median())) if len(fy) > 0 else None
        out["Median founding year"] = "" if med_year is None else str(med_year)

    # Median total employees (exclude missing headcount)
    hc_valid = df_sub["Total Headcount"].dropna()
    med_emp = int(np.round(hc_valid.median())) if len(hc_valid) > 0 else None
    out["Median total employees"] = "" if med_emp is None else format_int_iso(med_emp)

    # Debiased ML engineers: sums of existing rounded consensus and bounds (interval without "80%")
    ml_c = df_sub["ml_consensus_round"].dropna().astype(float)
    ml_l = df_sub["ml_lower80_round"].dropna().astype(float)
    ml_u = df_sub["ml_upper80_round"].dropna().astype(float)

    total_ml_c = int(ml_c.sum()) if len(ml_c) else 0
    total_ml_l = int(ml_l.sum()) if len(ml_l) else 0
    total_ml_u = int(ml_u.sum()) if len(ml_u) else 0

    out["Debiased log-median ML engineers"] = f"{format_int_iso(total_ml_c)} ({format_int_iso(total_ml_l)} - {format_int_iso(total_ml_u)})"

    # Percentage of ML talent within the section (existing central estimate, from counts)
    denom = total_employees_num if total_employees_num > 0 else None
    base_pct = pct_string(total_ml_c, denom) if denom else "n/a"

    # Append interval from ml_share_* columns using headcount-weighted means
    interval_str = None
    hc_w = df_sub["Total Headcount"].astype(float) if "Total Headcount" in df_sub.columns else None

    has_lower = "ml_share_lower80" in df_sub.columns
    has_upper = "ml_share_upper80" in df_sub.columns

    if hc_w is not None and has_lower and has_upper:
        share_l = _ensure_fraction_share(pd.to_numeric(df_sub["ml_share_lower80"], errors="coerce"))
        share_u = _ensure_fraction_share(pd.to_numeric(df_sub["ml_share_upper80"], errors="coerce"))
        low = _weighted_mean(share_l, hc_w)
        high = _weighted_mean(share_u, hc_w)

        # If the central percentage is n/a but we do have ml_share, fill it from shares
        if base_pct == "n/a" and "ml_share" in df_sub.columns:
            share_c = _ensure_fraction_share(pd.to_numeric(df_sub["ml_share"], errors="coerce"))
            center = _weighted_mean(share_c, hc_w)
            if center is not None and np.isfinite(center):
                base_pct = pct_from_fraction(center)

        if (low is not None) and (high is not None) and np.isfinite(low) and np.isfinite(high):
            interval_str = f"{pct_from_fraction(low)} - {pct_from_fraction(high)}"

    out["Percentage ML talent of total"] = base_pct if interval_str is None else f"{base_pct} ({interval_str})"

    return out


def regional_breakdown(df, total_n_for_pct):
    """
    Returns two dicts:
    orgs_map: "Region (orgs)" -> "count (p %)"
    emps_map: "Region (employees)" -> "sum_employees"
    Plus prints consistency warnings using raw counts/sums (not parsing strings).
    """
    orgs_map, emps_map = {}, {}

    # Raw totals for checks
    total_orgs = int(len(df))
    total_employees = int(df["Total Headcount"].dropna().sum()) if df["Total Headcount"].notna().any() else 0

    for region in REGION_ORDER:
        mask_r = (df["Subregion"] == region)
        count = int(mask_r.sum())
        orgs_map[f"{region} (orgs)"] = f"{format_int_iso(count)} ({pct_string(count, total_n_for_pct)})"

        emp_sum = int(df.loc[mask_r & df["Total Headcount"].notna(), "Total Headcount"].sum())
        emps_map[f"{region} (employees)"] = format_int_iso(emp_sum)

    # Consistency checks (raw)
    if sum(df["Subregion"].isin([r for r in REGION_ORDER]).astype(int)) != total_orgs:
        # In practice this shouldn't happen because we mapped non-listed to 'Unknown'
        print("Warning: some organizations not assigned to a listed region.")
    if sum(int((df["Subregion"] == r).sum()) for r in REGION_ORDER) != total_orgs:
        print("Warning: regional org-count sum does not equal total organizations.")
    if sum(int(df.loc[df["Subregion"] == r, "Total Headcount"].dropna().sum()) for r in REGION_ORDER) != total_employees:
        print("Warning: regional employee totals do not equal overall total employees.")

    return orgs_map, emps_map

def build_index_entries():
    """
    Build ordered display index entries with indentation for sub-rows.
    Each entry is a dict: {display, type, section, row_label}
    """
    entries = []

    # Total section
    entries.append({"display": "Total", "type": "header", "section": "Total", "row_label": None})
    for lbl in ["Organization N",
                "Total employees",
                "Median founding year",
                "Median total employees",
                "Debiased log-median ML engineers",
                "Percentage ML talent of total"]:
        entries.append({"display": f"{INDENT}{lbl}", "type": "row", "section": "Total", "row_label": lbl})

    # Size sections
    for size_label, _ in SIZE_SECTIONS:
        entries.append({"display": size_label, "type": "header", "section": size_label, "row_label": None})
        for lbl in ["Organization N",
                    "Median total employees",
                    "Debiased log-median ML engineers",
                    "Percentage ML talent of total"]:
            entries.append({"display": f"{INDENT}{lbl}", "type": "row", "section": size_label, "row_label": lbl})

    # Regions header + rows
    entries.append({"display": "Regions", "type": "header", "section": "Regions", "row_label": None})
    for r in REGION_ORDER:
        entries.append({"display": f"{INDENT}{r} (orgs)", "type": "row", "section": "Regions", "row_label": f"{r} (orgs)"})
    for r in REGION_ORDER:
        entries.append({"display": f"{INDENT}{r} (employees)", "type": "row", "section": "Regions", "row_label": f"{r} (employees)"})

    return entries

def summarize_one_df(df):
    """
    Returns a nested dict:
    summary["Total"][row_label] = value
    summary[size_label][row_label] = value
    summary["Regions"][row_label] = value
    """
    summary = {}

    # Total
    total_n = int(len(df))
    summary["Total"] = section_metrics(
        df,
        mask=None,
        include_total_employees=True,
        include_median_year=True,
        total_n_for_pct=total_n
    )

    # Sizes
    summary.update({})  # placeholder for clarity
    hc = df["Total Headcount"]
    for size_label, cond in SIZE_SECTIONS:
        mask = hc.notna() & cond(hc)
        summary[size_label] = section_metrics(
            df,
            mask=mask,
            include_total_employees=False,
            include_median_year=False,
            total_n_for_pct=total_n
        )

    # Regions
    orgs_map, emps_map = regional_breakdown(df, total_n_for_pct=total_n)
    reg = {}
    reg.update(orgs_map)
    reg.update(emps_map)
    summary["Regions"] = reg

    return summary

def build_summary_table(dataframes, labels=None, region_order=None):
    if len(dataframes) != 6:
        raise ValueError("Please provide exactly 6 DataFrames.")
    
    if labels is None:
        labels = [f"Dataset_{i+1}" for i in range(len(dataframes))]
    
    # Process each DataFrame
    processed_dfs = [read_one_tsv(df, region_order) for df in dataframes]
    summaries = [summarize_one_df(df) for df in processed_dfs]

    entries = build_index_entries()
    index_display = [e["display"] for e in entries]

    data_cols = {}
    for label, summ in zip(labels, summaries):
        col_vals = []
        for e in entries:
            if e["type"] == "header":
                col_vals.append("")
            else:
                section = e["section"]
                row_lbl = e["row_label"]
                col_vals.append(summ.get(section, {}).get(row_lbl, ""))
        data_cols[label] = col_vals

    summary_df = pd.DataFrame(data_cols, index=index_display)
    summary_df.index.name = "Characteristic"
    return summary_df

def main(dataframes, labels=None):
    # ----------------------------
    # Execute
    # ----------------------------

    summary = build_summary_table(dataframes, labels, REGION_ORDER)
    print("Combined summary statistics created")
    print("Preview of first 30 rows:")
    print(summary.iloc[:30])
    
    return summary

