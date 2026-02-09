# AI Agent Orientation Guide

This document provides comprehensive orientation for AI agents (Claude Code, Cursor, Copilot, etc.) working in this repository. Read this before making any changes.

---

## Project Overview

**Project name:** Nepenthe (internal codename for the study)

**What this is:** A research project that systematically estimates the number of technically capable ML researchers employed at IT consulting firms worldwide. The motivation is AI safety: if this talent pool exists, it could be directed toward alignment and evaluation work.

**Key finding:** Across 403 identified ML consultancies (~3.27 million employees), the study estimates roughly 1,100 individuals (80% CI: 252--3,165) with strong ML research profiles. A 3-day work trial validated that top consultancies can execute challenging technical ML tasks at a high level. AI coding agents (GPT-5, Claude Opus 4.1) could not pass the same work trial.

**Funder:** Coefficient Giving

**Authors:** Maximilian Schons (lead), Red Bermejo, Florian Aldehoff-Zeidler, Niccolo Zanichelli, Oliver Evans, Gavin Leech, Samuel Haergestam

---

## Repository Structure and Purpose of Each Component

### `Nepenthe Draft/` -- Paper manuscript

- `NepenthePaper.tex`: The full LaTeX manuscript. This is the authoritative description of methodology, results, and discussion.
- `NepenthePaper.pdf`: Compiled paper.
- `images/`: Figures referenced in the paper.
- If you need to understand *why* something was done a certain way, the paper is the primary reference.

### `ml-headcount-nepenthe-main/` -- Main analysis pipeline (THE CORE CODEBASE)

This is the most complex and important subdirectory. It is a standalone Python project with its own `pyproject.toml`, `uv.lock`, and `README.md`.

**Framework:** Built on [Hamilton](https://hamilton.dagworks.io/), a Python micro-framework where functions define a DAG (directed acyclic graph) of data transformations. Function names become node names; parameter names define edges. This is NOT a typical script-based pipeline.

**Entry point:** `main.py` -- parses CLI args and executes Hamilton subgraphs.

**Key source modules in `src/ml_headcount/`:**

| Module | What it does |
|--------|-------------|
| `hamilton_pipeline.py` | Defines the Hamilton DAG: all pipeline nodes as functions |
| `hamilton_dataloaders.py` | Data loading nodes (CSV/TSV/Excel ingestion) |
| `hamilton_processors.py` | Data processing/transformation nodes |
| `hamilton_plots.py` | Visualization nodes |
| `correlated_probit_bootstrap.py` | **Core statistical model.** Implements the bootstrap multivariate probit inference that combines 6 annotators (3 LLMs + 3 keyword filters) to estimate per-employee ML expert probability, accounting for correlated annotator errors. This is the heart of the methodology. |
| `probit_dawid_skene_inference.py` | Dawid-Skene inspired inference logic; computes posterior probabilities using multivariate normal orthant probabilities |
| `synthetic_data_generation.py` | Generates synthetic employee-level annotations using a Gaussian copula for companies where only aggregate keyword counts are available |
| `keyword_filters.py` | Implements the 3 keyword-based annotators (broad_yes, strict_no, broad_yes_strict_no) |
| `modal_functions.py` | Modal Labs integration for GPU-accelerated KeyBERT |
| `config.py` | Configuration dataclass loaded from YAML |
| `execution_config.py` | Runtime execution settings |
| `schemas.py` | Data schemas and type definitions |
| `annotator_validation_plots.py` | Plots for annotator performance diagnostics |
| `fake_data_validation_plots.py` | Synthetic vs. real data comparison plots |
| `latent_covariance_diagnostic.py` | Diagnostics for the latent correlation structure |

**Script modules in `src/ml_headcount/scripts/`:**

| Directory | Purpose |
|-----------|---------|
| `data_ingestion/` | LinkedIn data conversion, CV keyword extraction, title processing |
| `data_cleaning/` | Affiliation cleaning, academic filtering, outlier removal |
| `text_analysis/` | KeyBERT extraction, keyword filtering, confusion matrix analysis, validation scoring |
| `statistical/` | Log-debiasing estimation |
| `geographic/` | Headquarters mapping and geocoding |
| `visualization/` | Estimate comparison plots, talent landscape, geographic maps |
| `reporting/` | Organisation lists, summary statistics |

**Configuration:** `config/default.yaml` controls all parameters. Key sections:
- `keybert_extraction`: Embedding model and extraction parameters
- `correlated_probit_bootstrap`: Number of bootstrap iterations, prior parameters
- `dawid_skene_filtering`: Data filtering thresholds
- `execution`: Remote (Modal) vs. local processing, caching, telemetry
- `data_paths`: Input/output directory paths
- `linkedin_datasets`: Which datasets to include in the run

**Data files in `ml-headcount-nepenthe-main/data/`:**
- `cv_data.tsv`: CV text data for keyword extraction
- `validation_cvs.xlsx`: 585 manually-rated CVs (ground truth)
- `company_database.tsv`: Company metadata from Crunchbase
- `affiliation_counts.csv`: Academic affiliation counts

**Tests:** `tests/` contains pytest tests. Run with `uv run pytest`.

**Documentation:** `docs/model_description/` has the mathematical framework; `docs/code_description/` has integration and usage docs.

### `Nepenthe Public Data/` -- Shared datasets

Public-facing data files shared alongside the paper. The main file of interest is:
- `final_results_main_orgs.csv`: Final per-company ML headcount estimates (q10, q50, q90)
- `2025-08-05_systematic_search_all.xlsx`: The full screening database of 2,121 organisations
- `2025-08-06_validation_cvs_rated.xlsx`: Ground truth labels for 585 CVs
- `2025_08_20_all_ML_consultancy_evaluation_code.ipynb`: Evaluation notebook

### `main pipeline outputs/` -- Pipeline results

All intermediate and final outputs. Key files:
- `dawid_skene_test_data_main.csv` / `dawid_skene_validation_data.csv`: Annotator label matrices
- `correlated_probit_bootstrap_results_main.csv`: Per-company bootstrap estimates
- `correlated_probit_bootstrap_distributions.json`: Full bootstrap distributions (large file)
- `log_debias_company_aggregates_main.csv`: Alternative debiased estimates
- `final_results_main_orgs.csv` / `final_results_all.csv`: Merged final results
- `real_employee_level_data_all.csv`: Individual-level annotation data (~13 MB)
- `annotator_metrics.csv`: Per-annotator sensitivity, specificity, accuracy
- Various `.png` files: Diagnostic and results plots

### `post-processing notebooks/` -- Geographic and comparator analysis

Two Jupyter notebooks that run AFTER the main pipeline:
1. `01_add_subregion.ipynb`: Adds UN sub-region classifications, generates geographic visualisations (maps, bar charts)
2. `02_comparator_analysis.ipynb`: Applies the estimation pipeline to known ML companies (OpenAI, Anthropic, etc.) and known non-ML companies (Patagonia, Crocs, etc.) as validation

These notebooks depend on `requirements.txt` at the repository root.

### `post-processing output/` -- Post-processing results

Outputs from the above notebooks: company tables by confidence tier (probable, possible, non-zero, not detected), geographic HTML maps, landscape plots, summary CSVs.

### `nepenthe_work_trial_template/` -- Work trial for consultancies

A self-contained codebase sent to consultancies for a 3-day paid work trial. It contains:
- An ML unlearning evaluation framework using RMU (Representation Misdirection for Unlearning)
- The RTT (Retraining on T) attack for robustness evaluation
- Hydra configuration, Ray-based parallel execution
- Task: implement "Sequential Unlearning" -- a multi-stage wrapper that partitions forget data into k folds and progressively unlearns
- Evaluation datasets: MMLU subsets, WMDP (deduped), date-year benchmarks

This is a TEMPLATE -- it does not contain consultancy submissions.

---

## What is NOT in this repository

Two important data sources are **excluded** and should not be expected:

1. **LinkedIn CV data from Brightdata** -- Individual employee profiles (~250,000) sourced via Brightdata's LinkedIn dataset are not included due to data licensing and privacy constraints. The pipeline can still be examined structurally, but the raw input data for the LLM classification step is absent. Only aggregated/anonymised derivatives appear in the outputs.

2. **Work trial submission data** -- Individual consultancy work trial code, reports, Git repositories, evaluations, company identities, and Slack conversations are confidential. Only the template (`nepenthe_work_trial_template/`) and aggregate statistics (in the paper) are shared.

---

## Key Concepts for Understanding the Codebase

### The 6 annotators

Each employee profile is scored by 6 binary classifiers ("annotators"):
1. **Gemini 2.5 Flash** (LLM, Prompt 1)
2. **Claude Sonnet 4** (LLM, Prompt 1)
3. **GPT-5 Mini** (LLM, Prompt 1)
4. **broad_yes** keyword filter (presence of ML-positive keywords)
5. **strict_no** keyword filter (absence of business/management keywords)
6. **broad_yes_strict_no** combined keyword filter

### The correlated probit model

Unlike naive Dawid-Skene (which assumes independent annotator errors), this model uses a multivariate normal latent structure in probit space to capture correlated mistakes (e.g., all 3 LLMs failing on the same ambiguous profile). Separate correlation matrices are maintained for positive and negative instances.

### Bootstrap uncertainty quantification

1,000 bootstrap iterations capture 5 sources of uncertainty:
1. Confusion matrix estimation (resample validation CVs)
2. Prior uncertainty (sample from Beta hyperprior)
3. Within-company sampling variation (resample employees)
4. Correlation structure uncertainty (resample test data)
5. Realization uncertainty (Bernoulli draws of true labels)

### Synthetic vs. real data

For companies where only aggregate LinkedIn keyword counts are available (no individual CV data from Brightdata), the pipeline generates synthetic employee-level annotations using a Gaussian copula. Synthetic estimates are adjusted by a factor of 0.5 based on calibration.

### Company confidence categories

Companies are classified by their estimate distributions:
- **Probable:** q10 > 0 (80% CI excludes zero)
- **Possible:** q50 > 0 but q10 = 0
- **Non-zero:** q90 > 0 but q50 = 0
- **Not detected:** all estimates = 0

### Comparator cohorts

The pipeline was validated on two comparator groups:
- **Comparator ML:** 18 known ML-heavy organisations (e.g., OpenAI, Anthropic, Mistral, HuggingFace)
- **Comparator non-ML:** 18 known non-ML organisations (e.g., Patagonia, Crocs, The British Museum)

---

## Common Tasks an Agent Might Be Asked to Do

### "Re-run the main pipeline"
```bash
cd ml-headcount-nepenthe-main
uv sync
uv run main.py --subgraphs probit_bootstrap log_debias
```
Note: this requires the input data files in `data/` and will not work without the Brightdata LinkedIn data.

### "Run the post-processing notebooks"
```bash
pip install -r requirements.txt
jupyter notebook "post-processing notebooks/01_add_subregion.ipynb"
```

### "Understand how estimates are computed"
Read in this order:
1. Paper methods section: `Nepenthe Draft/NepenthePaper.tex` (search for `\section{Methods}`)
2. Statistical model: `ml-headcount-nepenthe-main/src/ml_headcount/correlated_probit_bootstrap.py`
3. Inference: `ml-headcount-nepenthe-main/src/ml_headcount/probit_dawid_skene_inference.py`
4. Synthetic data: `ml-headcount-nepenthe-main/src/ml_headcount/synthetic_data_generation.py`
5. Math docs: `ml-headcount-nepenthe-main/docs/model_description/`

### "Understand the pipeline architecture"
Read:
1. `ml-headcount-nepenthe-main/main.py` -- entry point and CLI
2. `ml-headcount-nepenthe-main/src/ml_headcount/hamilton_pipeline.py` -- DAG definition
3. `ml-headcount-nepenthe-main/config/default.yaml` -- all parameters
4. `ml-headcount-nepenthe-main/docs/code_description/DATA_FLOW.mmd` -- Mermaid diagram

### "Look at the results"
- Per-company estimates: `main pipeline outputs/final_results_main_orgs.csv` or `post-processing output/company_table_probable.csv`
- Comparator validation: `main pipeline outputs/final_results_comparator_ml.csv` and `final_results_comparator_non_ml.csv`
- Geographic breakdown: `post-processing output/country_map_all.html`
- Annotator performance: `main pipeline outputs/annotator_metrics.csv`

### "Modify the paper"
Edit `Nepenthe Draft/NepenthePaper.tex`. Compile with XeLaTeX (requires Palatino, Helvetica, Menlo fonts). The paper uses Pandoc-style macros.

### "Add a new annotator or keyword filter"
1. Define the filter in `ml-headcount-nepenthe-main/src/ml_headcount/keyword_filters.py`
2. Add it to the Hamilton DAG in `hamilton_pipeline.py`
3. Update `config/default.yaml` if new parameters are needed
4. Re-run validation against `data/validation_cvs.xlsx`

---

## Technical Environment

- **Python version:** 3.12+ (see `ml-headcount-nepenthe-main/.python-version`)
- **Package management:** `uv` for the main pipeline; `pip` for the post-processing notebooks
- **Key dependencies (main pipeline):** Hamilton, pandas, numpy, scipy, scikit-learn, matplotlib, seaborn, plotly, Modal (optional, for GPU), KeyBERT, sentence-transformers
- **Key dependencies (notebooks):** pandas, matplotlib, seaborn, plotly, scipy
- **Tests:** pytest (`uv run pytest` in `ml-headcount-nepenthe-main/`)
- **Task runner:** `just` (see `ml-headcount-nepenthe-main/justfile`)

---

## Important Caveats for Agents

1. **Missing raw data:** The Brightdata LinkedIn profiles are not in this repo. Many pipeline steps will fail without them. The outputs in `main pipeline outputs/` are pre-computed results.

2. **Hamilton is declarative:** Functions in `hamilton_pipeline.py` are NOT called imperatively. Their names and parameter names define the DAG. If you rename a function, you change the DAG topology. Be very careful with refactoring.

3. **The paper is the ground truth:** If there is a discrepancy between code comments and the paper, the paper is authoritative.

4. **Coordinate systems across files:** `final_results_main_orgs.csv` appears in multiple locations (top-level, `Nepenthe Public Data/`, `main pipeline outputs/`). These may differ slightly if generated at different pipeline stages. The canonical version for the paper is in `Nepenthe Public Data/`.

5. **Sensitive data:** Even though Brightdata data is excluded, some filenames and company names appear in intermediate outputs. Treat all data with appropriate care.

6. **Work trial is a template only:** The `nepenthe_work_trial_template/` directory is what was SENT to consultancies. Their responses are not here.
