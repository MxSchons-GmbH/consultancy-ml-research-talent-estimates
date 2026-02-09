import pandas as pd
import numpy as np
import re
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D
from adjustText import adjust_text

# Global flag for adjustText usage
USE_ADJUSTTEXT = True

def create_ml_talent_landscape_plot(df):
    """Create ML talent landscape plot
    
    Args:
        df (pd.DataFrame): DataFrame with organizational data
        
    Returns:
        tuple: (fig, ax, agg) - Figure, axis, and aggregated data
    """
    # === ML Talent Landscape v4: no lost points, exclusive backgrounds, x to 45% ===

    # ---- Helpers ----
    def find_col(df, name_like):
        tgt = re.sub(r"[^a-z0-9]", "", str(name_like).lower())
        for c in df.columns:
            c2 = re.sub(r"[^a-z0-9]", "", str(c).lower())
            if c2 == tgt: return c
        for c in df.columns:
            c2 = re.sub(r"[^a-z0-9]", "", str(c).lower())
            if tgt in c2: return c
        return None

    def parse_employee_range(val):
        if pd.isna(val): return np.nan
        s = str(val).strip().replace(",","").replace("–","-").replace("—","-").replace("−","-")
        m = re.match(r"^\s*(\d+)\s*-\s*(\d+)\s*$", s)
        if m:
            a, b = int(m.group(1)), int(m.group(2)); return (a+b)/2.0
        m = re.match(r"^\s*(\d+)\s*(?:to|-)\s*(\d+)\s*$", s)
        if m:
            a, b = int(m.group(1)), int(m.group(2)); return (a+b)/2.0
        m = re.match(r"^\s*(\d+)\s*\+\s*$", s)
        if m: return float(m.group(1))
        m = re.match(r"^\s*(\d+)\s*$", s)
        if m: return float(m.group(1))
        return np.nan

    # ---- Columns ----
    ORG_COL = find_col(df, "organization_name") or df.columns[0]
    ML_COUNT_COL = find_col(df, "ml_consensus_round") or find_col(df, "ml_count") or find_col(df, "ml")
    STAGE_COL = find_col(df, "stage_reached") or find_col(df, "Stage Reached")

    # Robust employees: combine multiple sources row-wise (first non-null)
    emp_sources = []
    for cand in ["number_of_employees_numeric", "total_headcount", "employees_numeric", "employee_count"]:
        col = find_col(df, cand)
        if col is not None:
            emp_sources.append(pd.to_numeric(df[col], errors="coerce"))
    raw_col = find_col(df, "number_of_employees")
    if raw_col is not None:
        emp_sources.append(df[raw_col].apply(parse_employee_range))

    emp = emp_sources[0].copy() if emp_sources else pd.Series(np.nan, index=df.index, dtype=float)
    for s in emp_sources[1:]:
        emp = emp.fillna(s)

    ml_n = pd.to_numeric(df[ML_COUNT_COL], errors="coerce")
    stage = df[STAGE_COL].astype(str) if STAGE_COL else ""

    plot = pd.DataFrame({"org": df[ORG_COL].astype(str), "ml_n": ml_n, "emp": emp, "stage": stage})
    plot["ml_pct"] = (plot["ml_n"] / plot["emp"]) * 100.0
    plot["ml_pct"] = plot["ml_pct"].clip(lower=0, upper=100)  # keep everything visible; cap at 100%

    plot = plot.replace([np.inf, -np.inf], np.nan).dropna(subset=["ml_n","emp","ml_pct"])
    plot = plot[(plot["ml_n"] > 0) & (plot["emp"] > 0)]

    def any_work_trial(series):
        s = series.dropna().astype(str).str.strip()
        s = s.str.replace("–","-").str.replace("—","-").str.replace("\u2212","-").str.lower()
        return s.str.contains(r"^5\s*-\s*work\s*trial$").any()

    agg = (plot.groupby("org", as_index=False)
                .agg(ml_n=("ml_n","median"),
                    emp=("emp","median"),
                    ml_pct=("ml_pct","median"),
                    work_trial=("stage", any_work_trial)))

    def assign_cluster(n, pct):
        if (n >= 500) and (pct >= 0.5): return "Giant"
        if (n >= 50)  and (pct >= 1.0): return "Powerhouse"
        if (n >= 10)  and (pct >= 5.0): return "Emerging"
        return "Other"
    agg["cluster"] = [assign_cluster(n, p) for n, p in agg[["ml_n","ml_pct"]].to_numpy()]

    palette = {"Giant":"#9467BD", "Powerhouse":"#1F77B4", "Emerging":"#2CA02C", "Other":"#BDBDBD"}
    markers = {"Giant":"D", "Powerhouse":"s", "Emerging":"^", "Other":"o"}

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(12, 7.9))
    ax.set_xlim(0, 45)
    ax.set_yscale("log")

    # points (work trial -> black border)
    for lab, sub in agg.groupby("cluster"):
        edge = np.where(sub["work_trial"], "black", "none")
        lw = np.where(sub["work_trial"], 1.2, 0.0)
        ax.scatter(sub["ml_pct"], sub["ml_n"], s=64, c=palette.get(lab, "#BDBDBD"),
                marker=markers.get(lab, "o"), alpha=0.95, edgecolors=edge, linewidths=lw, zorder=2)

    ax.relim(); ax.autoscale(axis='y')
    ymin, ymax = ax.get_ylim()

    # exclusive faint backgrounds
    def add_band(x0, y0, x1, y1, color, z=0):
        if y1 > y0 and x1 > x0:
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                                facecolor=color, alpha=0.08, edgecolor="none", zorder=z))

    add_band(5.0, 10, 45.0, min(50, ymax), palette["Emerging"], z=0)     # Emerging only
    add_band(1.0, 50, 45.0, min(500, ymax), palette["Powerhouse"], z=0.1)# Powerhouse only
    add_band(0.5, 500, 45.0, ymax,          palette["Giant"], z=0.2)     # Giant only

    # threshold lines
    ax.axhline(10,  linestyle="-.", color="0.45", linewidth=1.2, zorder=1)
    ax.axhline(50,  linestyle="--", color="0.45", linewidth=1.2, zorder=1)
    ax.axhline(500, linestyle=":",  color="0.45", linewidth=1.2, zorder=1)
    ax.axvline(5.0, linestyle="-.", color="0.45", linewidth=1.2, zorder=1)
    ax.axvline(1.0, linestyle="--", color="0.45", linewidth=1.2, zorder=1)
    ax.axvline(0.5, linestyle=":",  color="0.45", linewidth=1.2, zorder=1)

    # labels
    ax.set_xlabel("ML share (% of organization)")
    ax.set_ylabel("ML staff Estimate (probit bootstrap)")
    ax.set_title("ML Talent Landscape — Enterprise / Mid-Scale / Boutique")
    ax.grid(axis="y", which="both", alpha=0.15)

    # annotate all Giant/Powerhouse/Emerging orgs
    texts = [ax.text(r["ml_pct"], r["ml_n"], r["org"], fontsize=8, ha="left", va="bottom")
            for _, r in agg[agg["cluster"].isin(["Giant","Powerhouse","Emerging"])].iterrows()]
    if USE_ADJUSTTEXT and len(texts) > 0:
        adjust_text(texts, ax=ax, expand_text=(1.02,1.08), expand_points=(1.02,1.08),
                    arrowprops=dict(arrowstyle="-", lw=0.5, color="0.5"))

    # legend (threshold descriptions + work trial border)
    handles = [
        Patch(facecolor=palette["Giant"], alpha=0.20, edgecolor="none", label="Enterprise (≥500 ML & 0.5%"),
        Patch(facecolor=palette["Powerhouse"], alpha=0.20, edgecolor="none", label="Mid-Scale (≥50 ML & 1%"),
        Patch(facecolor=palette["Emerging"], alpha=0.20, edgecolor="none", label="Boutique: (≥10 ML & 5%"),
        Line2D([0],[0], marker='o', color='w', label='Work Trial : Black border',
            markerfacecolor="#BDBDBD", markeredgecolor="black", markersize=8, linewidth=0)
    ]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)
    plt.subplots_adjust(right=0.78)

    print("Counts by cluster:\n", agg["cluster"].value_counts())
    print("Work Trial orgs:", int(agg["work_trial"].sum()))
    
    return fig, ax, agg

