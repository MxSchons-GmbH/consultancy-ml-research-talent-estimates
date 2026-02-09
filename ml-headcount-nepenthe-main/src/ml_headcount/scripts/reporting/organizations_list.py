import pandas as pd
import numpy as np
import re
from pathlib import Path

def build_organizations_list_csv(df):
    """
    Build organizations list CSV with the requested columns and formatting.
    
    Args:
        df (pd.DataFrame): DataFrame with organizational data
        
    Returns:
        pd.DataFrame: Formatted organizations list
    """
    # Order for the Individual estimates array (six items)
    INDIVIDUAL_METHODS = [
        "filter_broad_yes_strict_no",
        "filter_strict_no",
        "filter_broad_yes",
        "claude_total_accepted",
        "gpt5_total_accepted",
        "gemini_total_accepted",
    ]

    # Required columns overall
    REQUIRED = [
        "Organization Name",
        "Country",
        "Stage Reached",
        "category",
        "Total Headcount",
        "Founded Date",
        "ml_share",
        "ml_share_lower80",
        "ml_share_upper80"
    ] + INDIVIDUAL_METHODS

    # Call the main processing function
    return main(df)

def to_numeric_series_clean(series):
    """Vectorized numeric coercion for numeric columns by stripping commas/spaces."""
    if series.dtype.kind in ("i", "u", "f"):
        return pd.to_numeric(series, errors="coerce")
    s = series.astype(str).str.replace(r"[,\s]", "", regex=True)
    return pd.to_numeric(s, errors="coerce")

def extract_year_only(series):
    """
    Return an Int64 series with just the year:
    - First try pandas datetime year
    - Fallback to the first 4-digit year in [1700, 2100]
    """
    dt = pd.to_datetime(series, errors="coerce")
    years = dt.dt.year

    need_fallback = years.isna()
    if need_fallback.any():
        s = series.astype(str)
        # find first 4-digit token
        def find_year(x):
            m = re.search(r"(\d{4})", x)
            if not m:
                return np.nan
            y = int(m.group(1))
            return y if 1700 <= y <= 2100 else np.nan
        years_fb = s.map(find_year)
        years = years.fillna(years_fb)

    return years.astype("Int64")

def format_int_iso(n):
    """Format integers with spaces between groups of three digits."""
    if n is None or (isinstance(n, float) and not np.isfinite(n)) or pd.isna(n):
        return None
    n = int(n)
    sign = "-" if n < 0 else ""
    s = str(abs(n))
    groups = []
    while s:
        groups.append(s[-3:])
        s = s[:-3]
    return sign + " ".join(reversed(groups))

def interval_str(c, l, u):
    """Return 'C (L - U)' using ISO-spaced integers, '-' if all missing."""
    if pd.isna(c) and pd.isna(l) and pd.isna(u):
        return "-"
    def s(x):
        return format_int_iso(x) if not pd.isna(x) else None
    c_s, l_s, u_s = s(c), s(l), s(u)
    if c_s is None and l_s is None and u_s is None:
        return "-"
    if l_s and u_s:
        return f"{c_s or '-'} ({l_s} - {u_s})"
    if l_s:
        return f"{c_s or '-'} ({l_s})"
    if u_s:
        return f"{c_s or '-'} ({u_s})"
    return f"{c_s}"

def pct_str(numer, denom, decimals=1):
    """Format percentage with one decimal; return '-' if denom is missing/zero."""
    if denom is None or not np.isfinite(denom) or denom <= 0:
        return "-"
    if numer is None or not np.isfinite(numer):
        return "-"
    pct = 100.0 * float(numer) / float(denom)
    return f"{pct:.{decimals}f} %"

def _share_to_fraction(value):
    """
    Convert a share that may be given either as a fraction (0..1) or a percent (>1)
    into a fraction in [0,1]. Returns None if missing or non-numeric.
    """
    x = to_numeric_scalar(value)
    if x is None or not np.isfinite(x):
        return None
    x = float(x)
    return x / 100.0 if x > 1.0 else x

def rate_org_row(r):
    """
    Return one of:
    'Enterprise ML Consultancies'
    'Mid-Scale ML Consultancies'
    'Boutique ML Consultancies'
    '-'
    using ml_consensus_round for staff count and ml_share for percent.
    Precedence: Enterprise > Mid-Scale > Boutique.
    Thresholds:
    Enterprise:  ML staff ≥ 500  AND  ml_share > 0.5%
    Mid-Scale:   ML staff ≥ 50   AND  ml_share > 1%
    Boutique:    ML staff ≥ 10   AND  ml_share > 5%
    """
    ml_staff = to_numeric_scalar(r.get("ml_consensus_round"))
    share = _share_to_fraction(r.get("ml_share"))  # strictly from ml_share as requested
    if ml_staff is None or share is None:
        return "-"
    # Highest tier first
    if ml_staff >= 500 and share >= 0.005:
        return "Enterprise ML Consultancies"
    if ml_staff >= 50 and share >= 0.01:
        return "Mid-Scale ML Consultancies"
    if ml_staff >= 10 and share >= 0.05 and ml_staff < 50 :
        return "Boutique ML Consultancies"
    return "-"


def pct_from_fraction(frac, decimals=1):
    """Format a fraction (e.g., 0.123) as a percent string."""
    if frac is None or not np.isfinite(frac):
        return "-"
    return f"{100.0 * float(frac):.{decimals}f} %"

def percent_with_interval_row(r, decimals=1):
    """
    Build 'x.y % (l.y % - u.y %)' for a single organization.
    Central value defaults to ml_consensus_round / Total Headcount.
    If headcount is missing/zero, fall back to ml_share when present.
    The interval comes from ml_share_lower80 and ml_share_upper80.
    Values may be given as fractions in [0,1] or percents (>1); both are handled.
    """
    # Central percent from counts
    def _to_num(x):
        return to_numeric_scalar(x)

    numer = _to_num(r.get("ml_consensus_round"))
    denom = _to_num(r.get("Total Headcount"))
    base = pct_str(numer, denom, decimals=decimals)  # '-' if denom missing/zero

    # Shares for interval (and optional central fallback)
    def _norm_frac(x):
        if x is None or not np.isfinite(x):
            return None
        x = float(x)
        return x / 100.0 if x > 1.0 else x  # interpret >1 as percent

    l = _norm_frac(_to_num(r.get("ml_share_lower80"))) if "ml_share_lower80" in r else None
    u = _norm_frac(_to_num(r.get("ml_share_upper80"))) if "ml_share_upper80" in r else None
    c = _norm_frac(_to_num(r.get("ml_share")))          if "ml_share" in r else None

    # If base is unavailable but we have a central share, use it
    if base == "-" and c is not None:
        base = pct_from_fraction(c, decimals=decimals)

    # Assemble interval string
    l_s = pct_from_fraction(l, decimals=decimals) if l is not None else None
    u_s = pct_from_fraction(u, decimals=decimals) if u is not None else None

    if l_s and u_s:
        return f"{base} ({l_s} - {u_s})" if base != "-" else f"{pct_from_fraction(c, decimals)} ({l_s} - {u_s})" if c is not None else f"- ({l_s} - {u_s})"
    if l_s:
        return f"{base} ({l_s})" if base != "-" else f"{pct_from_fraction(c, decimals)} ({l_s})" if c is not None else f"- ({l_s})"
    if u_s:
        return f"{base} ({u_s})" if base != "-" else f"{pct_from_fraction(c, decimals)} ({u_s})" if c is not None else f"- ({u_s})"
    return base


def to_numeric_scalar(value):
    """Best-effort scalar coercion to number; booleans -> 1/0; None/empty -> None."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    s = str(value).strip()
    if s == "":
        return None
    low = s.lower()
    if low in {"true","t","yes","y"}:
        return 1
    if low in {"false","f","no","n"}:
        return 0
    s2 = s.replace(",", "").replace(" ", "")
    try:
        x = float(s2)
        if np.isfinite(x):
            if abs(x - int(round(x))) < 1e-9:
                return int(round(x))
            return x
    except Exception:
        pass
    # Not numeric: return None to output '-'
    return None

def array_numbers_only(values):
    """
    Build the 'Individual estimates ML Talent' array string:
    - numbers only (ints or floats), or '-' for missing
    - no quotes around elements
    - no thousands grouping inside the array
    Example: [12, -, 3, 15, 2, 0]
    """
    elems = []
    for v in values:
        num = to_numeric_scalar(v)
        if num is None:
            elems.append("-")
        elif isinstance(num, int):
            elems.append(str(num))
        else:
            # float: trim trailing .0 if integer-like
            if abs(num - int(round(num))) < 1e-9:
                elems.append(str(int(round(num))))
            else:
                elems.append(str(num))
    return "[" + ", ".join(elems) + "]"


def main(df):
    # ----------------------------
    # Load and validate
    # ----------------------------
    # Order for the Individual estimates array (six items)
    INDIVIDUAL_METHODS = [
        "filter_broad_yes_strict_no",
        "filter_strict_no",
        "filter_broad_yes",
        "claude_total_accepted",
        "gpt5_total_accepted",
        "gemini_total_accepted",
    ]

    # Required columns overall
    REQUIRED = [
        "Organization Name",
        "Country",
        "Stage Reached",
        "category",
        "Total Headcount",
        "Founded Date",
        "ml_share",
        "ml_share_lower80",
        "ml_share_upper80"
    ] + INDIVIDUAL_METHODS
    
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Numeric coercion for computations
    df["ml_consensus_round"] = to_numeric_series_clean(df["ml_consensus_round"])
    df["ml_lower80_round"]   = to_numeric_series_clean(df["ml_lower80_round"])
    df["ml_upper80_round"]   = to_numeric_series_clean(df["ml_upper80_round"])
    df["Total Headcount"]    = to_numeric_series_clean(df["Total Headcount"])

    # Sort by median ML estimate descending (NaNs last)
    df = df.sort_values("ml_consensus_round", ascending=False, na_position="last").reset_index(drop=True)

    # Build pieces
    company_name = df["Organization Name"].fillna("").str.strip()
    founded_year = extract_year_only(df["Founded Date"])
    country      = df["Country"].fillna("").str.strip()

    total_staff_iso = df["Total Headcount"].apply(lambda x: format_int_iso(x) or "-")

    indiv_arrays = df.apply(lambda r: array_numbers_only([r[col] for col in INDIVIDUAL_METHODS]), axis=1)

    debiased = df.apply(lambda r: interval_str(r["ml_consensus_round"], r["ml_lower80_round"], r["ml_upper80_round"]), axis=1)

    pct_ml = df.apply(lambda r: percent_with_interval_row(r, decimals=1), axis=1)

    rating = df.apply(rate_org_row, axis=1)


    stage = df["Stage Reached"].fillna("").astype(str).str.strip()
    cat   = df["category"].fillna("").astype(str).str.strip()
    worktrial = []
    for a, b in zip(stage, cat):
        if a and b:
            worktrial.append(f"{a} - {b}")
        elif a:
            worktrial.append(a)
        elif b:
            worktrial.append(b)
        else:
            worktrial.append("-")

    # Assemble final DataFrame in the specified order and names
    out = pd.DataFrame({
        "Company Name": company_name.replace(r"^\s*$", "-", regex=True),
        "Founded": founded_year.map(lambda y: str(int(y)) if pd.notna(y) else "-"),
        "Country": country.replace(r"^\s*$", "-", regex=True),
        "Total Staff Count": total_staff_iso,
        "Individual estimates ML Talent (filter_broad_yes_strict_no, filter_strict_no, filter_broad_yes, claude_total_accepted, gpt5_total_accepted, gemini_total_accepted)": indiv_arrays,
        "debiased log-median ML Talent": debiased,
        "% ML of total staff count": pd.Series(pct_ml),
        "Category": pd.Series(rating),
        "Worktrial Outcome": pd.Series(worktrial),
    })

    # Hyphen cleanup
    out = out.replace({np.nan: "-"})
    out = out.replace(r"^\s*$", "-", regex=True)

    # Preview top rows
    print("Preview of top 10 rows:")
    print(out.head(10))
    
    return out

