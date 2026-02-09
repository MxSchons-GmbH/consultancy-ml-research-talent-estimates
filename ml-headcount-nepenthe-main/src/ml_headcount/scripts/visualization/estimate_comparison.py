import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Colorblind-safe Tol "bright" palette
TOL_BRIGHT = [
    "#4477AA", "#EE6677", "#228833",
    "#CCBB44", "#66CCEE", "#AA3377", "#BBBBBB"
]

# Custom legend labels for LLMs
LLM_LABELS = {
    'gemini_total_accepted': 'gemini-2.5-flash',
    'claude_total_accepted': 'sonnet-4-20250514',
    'gpt5_total_accepted': 'gpt-5-mini-thinking-2025-08-07 Total accepted',
}

def plot_debiased_ml_estimates_comparison(df, y_scale="log", headcount_threshold=1000000):
    """
    Plot debiased ML estimates comparison with full-width display and top legend
    
    Args:
        df (pd.DataFrame): DataFrame with ML estimates and organization data
        y_scale (str): Scale for y-axis ("log" or "linear")
        headcount_threshold (int): Maximum headcount for filtering
        
    Returns:
        tuple: (fig, ax, df_processed) - Figure, axis, and processed DataFrame
    """
    # Call the main processing function
    return main(df, y_scale, headcount_threshold)

def _read_table(file_path: str) -> pd.DataFrame:
    lower = file_path.lower()
    if lower.endswith(".tsv") or lower.endswith(".tab"):
        return pd.read_csv(file_path, sep="\t")
    if lower.endswith(".csv"):
        return pd.read_csv(file_path)
    return pd.read_csv(file_path, sep=None, engine="python")

def load_and_filter_data(df: pd.DataFrame, headcount_threshold: int = 1000) -> pd.DataFrame:
    # Replace problematic values with NaN
    for val in ['#N/A', '#ERROR!', '']:
        df = df.replace(val, np.nan)
    df = df.infer_objects(copy=False)

    MAYBE_NUMERIC = [
        'filter_broad_yes', 'filter_strict_no',
        'filter_broad_yes_strict_no', 'claude_total_accepted',
        'gpt5_total_accepted', 'gemini_total_accepted',
        'ml_consensus_round', 'ml_lower80_round', 'ml_upper80_round'
    ]

    # Coerce specified columns to numeric
    for col in MAYBE_NUMERIC:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Optional headcount filter if available
    if 'Total Headcount' in df.columns and headcount_threshold is not None:
        df['Total Headcount'] = pd.to_numeric(df['Total Headcount'], errors='coerce')
        before = len(df)
        df = df[df['Total Headcount'].notna() & (df['Total Headcount'] <= headcount_threshold)].copy()
        print(f"Loaded {before} rows. After filtering (Total Headcount <= {headcount_threshold}): {len(df)} rows.")
    else:
        print(f"Loaded {len(df)} rows. No Total Headcount column, skipping headcount filter.")

    # Ensure consensus columns exist
    required = ['ml_consensus_round', 'ml_lower80_round', 'ml_upper80_round']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing consensus columns: {missing}")

    return df

def create_visualization(
    df: pd.DataFrame,
    y_scale: str = "log",
    org_col_candidates = ("Organization Name", "organization", "org", "Company", "company")
):
    # Series to plot
    filter_cols = [c for c in ['filter_broad_yes','filter_strict_no','filter_broad_yes_strict_no'] if c in df.columns]
    # Order LLMs as requested in the legend: gemini, sonnet(claude), gpt-5-mini...
    llm_cols_pref = ['gemini_total_accepted','claude_total_accepted','gpt5_total_accepted']
    llm_cols = [c for c in llm_cols_pref if c in df.columns]

    # Sort by overall central estimate
    df_valid = df.dropna(subset=['ml_consensus_round']).copy()
    df_valid = df_valid.sort_values('ml_consensus_round').reset_index(drop=True)

    # Organization label column
    org_col = next((c for c in org_col_candidates if c in df_valid.columns), None)

    # A wide base size, will expand to full width via CSS above
    fig, ax = plt.subplots(figsize=(18, 9))

    x = np.arange(len(df_valid))

    # Assign colors and markers
    filter_colors  = TOL_BRIGHT[:len(filter_cols)]
    filter_markers = ['o','v','X']  # circle, down-triangle, filled X

    llm_colors  = TOL_BRIGHT[3:3+len(llm_cols)]
    llm_markers = ['s','^','D']     # square, up-triangle, diamond

    offset_step = 0.08
    handles = []
    labels  = []

    # Helper for log masks
    def _mask_vals(y):
        if y_scale == "log":
            return (~np.isnan(y)) & (y > 0)
        return ~np.isnan(y)

    # Plot keyword filters
    for i, col in enumerate(filter_cols):
        y = df_valid[col].values
        mask = _mask_vals(y)
        x_pos = x + (i - len(filter_cols)/2) * offset_step
        sc = ax.scatter(
            x_pos[mask], y[mask],
            s=30, marker=filter_markers[i % len(filter_markers)],
            c=filter_colors[i], alpha=0.6,
            label=f"Keyword Filter: {col.replace('_',' ').title()}"
        )
        handles.append(sc)
        labels.append(f"Keyword Filter: {col.replace('_',' ').title()}")

    # Plot LLMs with distinct forms and custom labels
    for j, col in enumerate(llm_cols):
        y = df_valid[col].values
        mask = _mask_vals(y)
        x_pos = x + (j - len(llm_cols)/2) * offset_step
        sc = ax.scatter(
            x_pos[mask], y[mask],
            s=50, marker=llm_markers[j % len(llm_markers)],
            facecolors=llm_colors[j], edgecolors="black",
            linewidths=0.8, alpha=0.9,
            label=LLM_LABELS.get(col, col)
        )
        handles.append(sc)
        labels.append(LLM_LABELS.get(col, col))

    # Overall central estimate (ML + Keyword Filter) with 80% CI
    c = df_valid['ml_consensus_round'].values
    l = df_valid['ml_lower80_round'].values
    u = df_valid['ml_upper80_round'].values

    # Compute epsilon for log-scale clipping
    all_pos_candidates = []
    for col in filter_cols + llm_cols + ['ml_consensus_round','ml_lower80_round','ml_upper80_round']:
        if col in df_valid.columns:
            all_pos_candidates.extend(df_valid[col][df_valid[col] > 0].tolist())
    eps = (np.nanmin(all_pos_candidates) * 0.5) if (y_scale == "log" and len(all_pos_candidates) > 0) else 1e-9

    lower_bound = np.where(np.isfinite(l), np.maximum(l, eps), np.nan)
    upper_bound = np.where(np.isfinite(u), u, np.nan)
    upper_bound = np.maximum(upper_bound, c)  # guard against anomalies

    yerr_lower_all = np.clip(c - lower_bound, a_min=0, a_max=None)
    yerr_upper_all = np.clip(upper_bound - c, a_min=0, a_max=None)

    # Determine which orgs have any LLM value present (NaN means absent; zero counts as present)
    if llm_cols:
        llm_present = np.any(df_valid[llm_cols].notna().values, axis=1)
    else:
        llm_present = np.zeros(len(df_valid), dtype=bool)

    # Mask consensus for plotting, respecting y_scale
    mask_c = np.isfinite(c)
    if y_scale == "log":
        mask_c &= (c > 0)

    # Split into two groups: with LLMs vs filter-only
    mask_with_llm = mask_c & llm_present
    mask_filter_only = mask_c & (~llm_present)

    # Plot consensus where LLMs are present (white marker, black errorbars)
    if np.any(mask_with_llm):
        err_llm = ax.errorbar(
            x[mask_with_llm], c[mask_with_llm],
            yerr=np.vstack([yerr_lower_all[mask_with_llm], yerr_upper_all[mask_with_llm]]),
            fmt='o', mfc='white', mec='black', mew=1.0, ms=5,
            ecolor='black', elinewidth=1.2, capsize=2.5, capthick=1.2,
            label='Overall central estimate (LLMs + Keyword Filter, 80% CI)',
            zorder=3
        )
        handles.append(err_llm.lines[0])
        labels.append('Overall central estimate (LLMs + Keyword Filter, 80% CI)')

    # Plot consensus where ONLY filters are present (gray marker, gray errorbars)
    if np.any(mask_filter_only):
        err_filter = ax.errorbar(
            x[mask_filter_only], c[mask_filter_only],
            yerr=np.vstack([yerr_lower_all[mask_filter_only], yerr_upper_all[mask_filter_only]]),
            fmt='o', mfc='#DDDDDD', mec='#555555', mew=1.0, ms=5,
            ecolor='#555555', elinewidth=1.2, capsize=2.5, capthick=1.2,
            label='Overall central estimate (Keyword Filter only, 80% CI)',
            zorder=3
        )
        handles.append(err_filter.lines[0])
        labels.append('Overall central estimate (Keyword Filter only, 80% CI)')

    # Labels and title
    ax.set_xlabel('Organizations (sorted by overall central estimate)', fontsize=12)
    ax.set_ylabel('Estimate technical ML talent at organization', fontsize=12)
    ax.set_title('Per-organization estimates (80% CI)', fontsize=14, pad=60)

    # Y scale and limits
    if y_scale not in ("linear", "log"):
        raise ValueError("y_scale must be 'linear' or 'log'")

    all_vals = []
    for col in filter_cols + llm_cols + ['ml_consensus_round','ml_lower80_round','ml_upper80_round']:
        if col in df_valid.columns:
            all_vals.extend(df_valid[col].dropna().tolist())

    if all_vals:
        if y_scale == "log":
            pos_vals = [v for v in all_vals if v > 0]
            if len(pos_vals) == 0:
                raise ValueError("No positive values available for a log-scale y-axis.")
            ymin = min(pos_vals) * 0.9
            ymax = max(pos_vals) * 1.1
            # Ensure minimum y-axis extends to at least 5e3 to include CI upper bounds
            ymax = max(ymax, 5000)
            ax.set_yscale('log')
            ax.set_ylim(ymin, ymax)
        else:
            ymax = max(all_vals) * 1.05
            # Ensure minimum y-axis extends to at least 5e3
            ymax = max(ymax, 5000)
            ax.set_ylim(0, ymax)

    # Company names along the bottom, rotated for readability
    if org_col is not None:
        labels_bottom = df_valid[org_col].astype(str).tolist()
        ax.set_xticks(x)
        ax.set_xticklabels(labels_bottom, rotation=45, ha='right', fontsize=8)
        bottom_margin = 0.25
    else:
        ax.set_xticks(x)
        ax.set_xticklabels([f"Org {i}" for i in x], rotation=45, ha='right', fontsize=8)
        bottom_margin = 0.25

    # Single combined legend at top
    ncol = min(len(labels), 4) if len(labels) > 0 else 1
    ax.legend(
        handles, labels,
        loc='lower center',
        bbox_to_anchor=(0.5, 1.02),
        ncol=ncol,
        fontsize=9,
        title='Series',
        title_fontsize=10,
        frameon=False,
        borderaxespad=0.0
    )

    # Layout tuned for top legend and long x labels
    plt.tight_layout()
    plt.subplots_adjust(top=0.80, bottom=bottom_margin, right=0.98)

    # Grid and spines
    ax.grid(True, which='both', alpha=0.3, linestyle='-', linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    return fig, ax, df_valid

def print_summary_statistics(df: pd.DataFrame):
    s = df['ml_consensus_round'].dropna()
    print("\n================ SUMMARY ================\n")
    if s.empty:
        print("No overall central estimate values found.")
        return
    print(f"Overall central estimate mean:   {s.mean():.3f}")
    print(f"Overall central estimate median: {s.median():.3f}")
    print(f"Overall central estimate std:    {s.std():.3f}")
    print(f"Overall central estimate min:    {s.min():.3f}")
    print(f"Overall central estimate max:    {s.max():.3f}")

    ai_cols = [c for c in ['gemini_total_accepted','claude_total_accepted','gpt5_total_accepted'] if c in df.columns]
    has_ai = sum(any(pd.notna(row.get(c, np.nan)) for c in ai_cols) for _, row in df.iterrows())
    print(f"\nOrganizations with any LLM estimate present: {has_ai} of {len(df)}")

def main(df: pd.DataFrame, y_scale: str = "log", headcount_threshold: int = 1000000):
    df = load_and_filter_data(df, headcount_threshold=headcount_threshold)
    fig, ax, df_proc = create_visualization(df, y_scale=y_scale)
    print_summary_statistics(df_proc)

    return fig, ax, df_proc

# Example run
if __name__ == "__main__":
    print("To use this script:")
    print("1. Load your DataFrame with ML estimates")
    print("2. Call plot_debiased_ml_estimates_comparison(df, y_scale='log')")
    print("3. The function will return (fig, ax, df_processed)")


    print("1. Load your DataFrame with ML estimates")
    print("2. Call plot_debiased_ml_estimates_comparison(df, y_scale='log')")
    print("3. The function will return (fig, ax, df_processed)")

