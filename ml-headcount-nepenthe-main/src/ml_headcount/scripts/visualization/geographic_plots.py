import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

def generate_geographic_analysis_plots(dataframes_dict, output_dir="analysis_outputs"):
    """Generate geographic analysis plots for multiple DataFrames
    
    Args:
        dataframes_dict (dict): Dictionary mapping dataset names to DataFrames with geographic data.
                               Expected keys: 'ml', 'enterprise', 'midscale', 'boutique', 
                               'stage5', 'stage5_recommended'
        output_dir (str): Directory to save HTML outputs
        
    Returns:
        dict: Dictionary containing figure objects and summary statistics
    """
    if not isinstance(dataframes_dict, dict):
        raise ValueError("dataframes_dict must be a dictionary mapping dataset names to DataFrames")
    
    # Expected dataset names (simplified from original variable names)
    expected_keys = [
        'ml', 
        'enterprise', 
        'midscale', 
        'boutique', 
        'stage5', 
        'stage5_recommended'
    ]
    
    # Use only the datasets that are provided
    available_datasets = {k: v for k, v in dataframes_dict.items() if k in expected_keys}
    
    if not available_datasets:
        raise ValueError(f"No valid datasets found. Expected keys: {expected_keys}")
    
    print(f"Processing {len(available_datasets)} datasets: {list(available_datasets.keys())}")



    # Required columns for the analysis
    required_cols = ["Country", "Subregion", "Source"]

    # Create an output folder to save HTMLs
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    
    for dataset_name, df in available_datasets.items():
        print("="*80)
        print(f"Processing dataset: {dataset_name}")
        
        # Keep only relevant columns and drop rows missing any of them
        df = df[required_cols].copy()
        df = df.dropna(subset=required_cols)
        df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)  # trim whitespace

        if df.empty:
            print("  No valid rows after dropping missing Country/Subregion/Source. Skipping.")
            continue

        # Count country frequencies
        country_counts = df["Country"].value_counts()

        # Map Source to categories (same logic as original)
        df["Source_Category"] = df["Source"].apply(
            lambda x: "Manual Search + Network" if x == "Manual Search + Network" else "Other Sources"
        )

        # Cross-tab for stacked bar chart (Subregion x Source_Category)
        stacked_data = pd.crosstab(df["Subregion"], df["Source_Category"])
        # Reorder subregions by total count (ascending), same as original
        stacked_data = stacked_data.reindex(stacked_data.sum(axis=1).sort_values(ascending=True).index)

        # --- World heat map (country frequencies) ---
        title_map = f"Country Frequency Heat Map — {dataset_name}  (N={len(df)})"
        try:
            fig_map = px.choropleth(
                locations=country_counts.index.tolist(),
                color=country_counts.values,
                locationmode="country names",
                color_continuous_scale="Reds",
                title=title_map,
                labels={"color": "Count"}
            )
            fig_map.update_layout(title_x=0.5, geo=dict(showframe=False, showcoastlines=True))
            # Save
            out_map = os.path.join(output_dir, f"{dataset_name}_map.html")
            fig_map.write_html(out_map)
            print(f"  Saved map HTML to: {out_map}")
            results[f"{dataset_name}_map"] = fig_map
        except Exception as e:
            print(f"  WARNING: Could not create map for {dataset_name}: {e}")

        # --- Stacked horizontal bar chart for subregions ---
        title_bar = f"Subregion Counts by Source — {dataset_name}  (N={len(df)})"
        fig_bar = go.Figure()

        # Add bars for each source category (Other Sources first, Manual Search last)
        if "Other Sources" in stacked_data.columns:
            fig_bar.add_trace(go.Bar(
                name="LinkedIn Analysis Only",
                y=stacked_data.index.tolist(),
                x=stacked_data["Other Sources"].values,
                orientation="h",
                marker_color="#4682B4"
            ))

        if "Manual Search + Network" in stacked_data.columns:
            fig_bar.add_trace(go.Bar(
                name="Additional Worktrial Outreach",
                y=stacked_data.index.tolist(),
                x=stacked_data["Manual Search + Network"].values,
                orientation="h",
                marker_color="#2E8B57"
            ))

        fig_bar.update_layout(
            title=title_bar,
            title_x=0.5,
            xaxis_title="Count",
            yaxis_title="Subregion",
            barmode="stack",
            height=max(400, len(stacked_data) * 25)
        )

        try:
            out_bar = os.path.join(output_dir, f"{dataset_name}_subregion_stack.html")
            fig_bar.write_html(out_bar)
            print(f"  Saved stacked-bar HTML to: {out_bar}")
            results[f"{dataset_name}_bar"] = fig_bar
        except Exception as e:
            print(f"  WARNING: Could not create stacked bar for {dataset_name}: {e}")

        # --- Summary statistics ---
        summary = {
            "total_countries": len(country_counts),
            "total_subregions": len(stacked_data),
            "total_records": len(df),
            "source_distribution": df["Source_Category"].value_counts().to_dict(),
            "top_5_countries": country_counts.head(5).to_dict(),
            "top_5_subregions": stacked_data.sum(axis=1).sort_values(ascending=False).head(5).to_dict()
        }
        results[f"{dataset_name}_summary"] = summary
        
        print(f"  Total countries: {len(country_counts)}")
        print(f"  Total subregions: {len(stacked_data)}")
        print(f"  Total records: {len(df)}")
        print("\n  Source distribution:")
        print(df["Source_Category"].value_counts().to_string())
        print("\n  Top 5 countries:")
        print(country_counts.head(5).to_string())
        print("\n  Top 5 subregions (by total count):")
        subregion_totals = stacked_data.sum(axis=1).sort_values(ascending=False)
        print(subregion_totals.head(5).to_string())

    print("="*80)
    print(f"Done. HTML outputs are in the '{output_dir}' folder.")
    print("Results dictionary contains figure objects and summary statistics.")
    
    return results

