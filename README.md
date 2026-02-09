# How Much Technical Talent Is There? A Systematic Estimate of the ML Research Pool Among 3 Million IT Consultancy Employees

This repository accompanies the paper *"How much technical talent is there? A systematic estimate of the ML research pool among 3 million IT consultancy employees"* by Schons, Bermejo, Aldehoff-Zeidler, Zanichelli, Evans, Leech, and Haergestam.

The study asks whether IT consulting firms harbour a viable, underutilised pool of ML research talent that could be directed toward AI safety and alignment work. We screened 2,121 organisations globally, identified 403 ML consultancies employing ~3.27 million people, and estimated roughly 1,100 (80% CI: 252--3,165) individuals with strong technical ML research profiles. A subset of companies also completed a 3-day paid work trial benchmarked against state-of-the-art AI coding agents.

Funded by [Coefficient Giving](https://coefficientgiving.com/).

---

## What is included in this repository

### `Nepenthe Draft/`

The paper manuscript in LaTeX (`NepenthePaper.tex`) together with figures and the compiled PDF.

### `ml-headcount-nepenthe-main/`

The **main analysis pipeline**. A Python codebase built on the [Hamilton](https://hamilton.dagworks.io/) framework that implements:

- **Keyword extraction** from manually-rated CVs using KeyBERT and TF-IDF
- **LLM-based classification** of LinkedIn profiles (Gemini 2.5 Flash, Claude Sonnet 4, GPT-5 Mini)
- **Bootstrap probit inference** with correlated annotator error modelling (multivariate normal latent structure) to estimate true ML expert headcounts per company
- **Log-debiasing** as a complementary estimation method
- **Visualisation and reporting** scripts for figures, tables, and geographic maps

See `ml-headcount-nepenthe-main/README.md` for installation, usage, and configuration details.

### `Nepenthe Public Data/`

Public-facing datasets and the evaluation notebook shared alongside the paper:

| File | Description |
|------|-------------|
| `2025-08-05_systematic_search_all.xlsx` | Full systematic search results (2,121 organisations screened) |
| `2025-08-06_validation_cvs_rated.xlsx` | 585 manually-rated validation CVs used for annotator calibration |
| `2025-08-15_compartor_analysis.xlsx` | Comparator analysis data (known ML and non-ML companies) |
| `2025_08_20_all_ML_consultancy_evaluation_code.ipynb` | Master evaluation notebook |
| `cv_keywords_FIXED_frequency_analysis.xlsx` | Keyword frequency analysis for CV scoring |
| `final_results_main_orgs.csv / .xlsx` | Final ML headcount estimates per organisation |

### `main pipeline outputs/`

All intermediate and final outputs produced by the main pipeline, including:

- **Dawid-Skene test/validation data** -- annotator label matrices used for probit inference
- **Correlated probit bootstrap results** -- per-company headcount estimates with credible intervals, bootstrap distributions, and annotator performance plots
- **Log-debiasing outputs** -- alternative company-level estimates and aggregation summaries
- **Annotator diagnostics** -- bias analysis, sensitivity vs. specificity plots, accuracy metrics
- **Final merged results** -- `final_results_all.csv`, `final_results_main_orgs.csv`, plus comparator cohorts

### `post-processing notebooks/`

Jupyter notebooks for steps applied after the main pipeline:

| Notebook | Purpose |
|----------|---------|
| `01_add_subregion.ipynb` | Adds UN sub-region classifications and generates geographic visualisations |
| `02_comparator_analysis.ipynb` | Runs the estimation pipeline on comparator companies (known ML labs and non-ML firms) |

### `post-processing output/`

Outputs from the post-processing notebooks: company tables by confidence category, geographic maps and charts, ML landscape plots, and summary statistics.

### `nepenthe_work_trial_template/`

The **work trial codebase** sent to consultancies. Contains:

- An unlearning evaluation framework (RMU + RTT attack from Deeb & Roger, 2024)
- Hydra-based configuration, Ray-based parallel execution
- The task specification: implement a "Sequential Unlearning" meta-algorithm
- Evaluation data subsets (MMLU, WMDP, etc.)

See `nepenthe_work_trial_template/README.md` for the full task description and evaluation criteria.

### Top-level files

| File | Description |
|------|-------------|
| `2026-01-28_final_results_main_orgs.csv` | Latest snapshot of final results for main organisations |
| `final_results_comparator_ml.csv` | Estimates for comparator ML companies (e.g., OpenAI, Anthropic) |
| `final_results_comparator_non_ml.csv` | Estimates for comparator non-ML companies (e.g., Patagonia, Crocs) |
| `requirements.txt` | Python dependencies for the post-processing notebooks |

---

## What is NOT included

- **LinkedIn CV data from Brightdata** -- Individual employee profiles sourced via Brightdata are not included due to data licensing and privacy constraints. The pipeline outputs and validation datasets contain aggregated or anonymised derivatives only.
- **Work trial submissions and results** -- Individual consultancy work trial code, reports, evaluations, and company identities are withheld for confidentiality. Only the template repository and aggregate results (in the paper) are shared.

---

## Quickstart

### Post-processing notebooks

```bash
pip install -r requirements.txt
jupyter notebook
# Open notebooks in post-processing notebooks/
```

### Main pipeline

```bash
cd ml-headcount-nepenthe-main
pip install uv
uv sync
uv run main.py --subgraphs probit_bootstrap log_debias
```

See `ml-headcount-nepenthe-main/README.md` for full options including Modal Labs GPU acceleration, caching, and telemetry.

---

## Citation

If you use this work, please cite:

```
Schons M, Bermejo R, Aldehoff-Zeidler F, Zanichelli N, Evans O, Leech G, Haergestam S.
How much technical talent is there? A systematic estimate of the ML research pool
among 3 million IT consultancy employees. 2026.
```

---

## License

Please refer to the paper and individual subdirectory READMEs for licensing information.
