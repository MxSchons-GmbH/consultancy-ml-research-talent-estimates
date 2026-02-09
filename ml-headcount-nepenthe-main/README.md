# ML Headcount Estimation

A pipeline for estimating machine learning (ML) expert headcount in companies using programmatic annotations and statistical inference.

## Overview

This project implements a statistical framework to estimate the number of ML experts in organizations from programmatic annotations. The pipeline combines:

- **Keyword extraction** learned from manually-rated CVs to build discriminative filters
- **LLM-based classification** using multiple AI models (Gemini, Claude Sonnet, GPT) to assess ML expertise
- **Probabilistic inference** using a bootstrap probit model to estimate true expertise from noisy annotations
- **Uncertainty quantification** that captures parameter estimation, sampling, and realization uncertainty

The system uses **6 programmatic annotators** to classify employee profiles:
1. **3 LLM annotators**: Gemini 2.5 Flash, Claude Sonnet 4, GPT-5 Mini (all using the same technical expertise prompt)
2. **3 keyword-based filters**: ML keyword matching, broad research indicators, and business/managerial exclusion filters

### Training and Validation Data

The pipeline uses manually-rated CVs to:
- **Extract discriminative keywords**: ~585 manually-rated CVs (category 0: non-ML, category 1: ML expert) are analyzed using KeyBERT and TF-IDF to identify keywords that distinguish ML experts
- **Validate annotator performance**: The same CVs scored by all 6 annotators provide ground-truth labels for estimating annotator confusion matrices and error correlations

### Statistical Model: Bootstrap Probit Inference

These 6 annotators are combined using a **bootstrap probit model** that addresses conditional correlation between annotators. Unlike simple majority voting or independent error assumptions, this model:

- **Models annotator correlation structure**: Uses multivariate normal distributions in probit space to capture when annotators make correlated mistakes (e.g., multiple LLMs misclassifying the same edge cases)
- **Separate correlation structures**: Maintains distinct correlation matrices for positive vs. negative instances, allowing annotators to have different agreement patterns when labeling experts vs. non-experts
- **Bayesian inference**: Computes posterior probabilities for each employee being an ML expert given the full annotation pattern across all 6 annotators

The bootstrap procedure quantifies uncertainty from four sources:
1. **Parameter estimation uncertainty**: Resamples validation data to capture variability in annotator error rate estimates
2. **Prior uncertainty**: Samples from a Beta hyperprior on the base rate of ML experts
3. **Sampling uncertainty**: Resamples observed employees within each company to simulate alternative samples
4. **Realization uncertainty**: Uses Bernoulli sampling to account for stochastic variation in true expertise labels

See `docs/model_description/` for detailed mathematical documentation.

## Quick Start

### Installation

**Install uv (dependency manager):**


```sh
pip install uv
```

**Install project dependencies:**
```sh
uv sync
```

### Modal Labs Setup (Optional but Recommended)

This pipeline uses **Modal Labs** for GPU-accelerated cloud processing of expensive operations (KeyBERT keyword extraction, clustering). Modal Labs is a serverless AI infrastructure platform that runs Python functions in the cloud without managing infrastructure.

**Key benefits:**
- **GPU acceleration**: KeyBERT runs ~4-5x faster on Modal's GPUs than locally
- **Ephemeral deployment**: Functions run on-demand, no persistent deployment required
- **Automatic scaling**: Handles large datasets efficiently
- **Cost-effective**: Pay only for compute time used

**Setup Modal (one-time):**

```sh
# Install Modal Python package (if not already installed)
uv run pip install modal

# Set up your Modal account
uv run modal token new
```

This will prompt you to authenticate with Modal and create an API token. The pipeline automatically uses Modal when enabled (default behavior).

**Note**: For small datasets or development, you can run locally with `--local-only` to avoid Modal costs.

### Usage

The pipeline supports both local and remote (Modal Labs) processing:

```sh
# Run the entire pipeline with Modal Labs (recommended for large datasets)
uv run main.py

# Run the entire pipeline with local processing only
uv run main.py --local-only

# Run specific subgraphs
uv run main.py --subgraphs cv_analysis affiliation log_debias

# Generate pipeline DAG visualization
uv run main.py --visualize pipeline_dag.png
```

**Note**: Modal Labs deploys ephemerally when running the script—no manual deployment required. The first run will auto-deploy.

## Primary Subgraphs

The pipeline is organized into **subgraphs**—modular components that represent specific stages of the data processing pipeline. Subgraphs allow you to run only the parts of the pipeline you need, making development and debugging more efficient.

### The Two Primary Subgraphs

The two most important subgraphs for generating ML headcount estimates are:

#### 1. **Probit Bootstrap** (`probit_bootstrap`)

The **probit bootstrap** subgraph implements the core statistical inference method for estimating ML expert headcounts. It uses a bootstrap probit model that:

- **Models annotator correlations**: Captures when annotators make correlated mistakes (e.g., multiple LLMs misclassifying the same edge cases)
- **Quantifies uncertainty**: Provides comprehensive uncertainty quantification from four sources:
  - Parameter estimation uncertainty (from validation data)
  - Prior uncertainty (from base rate estimates)
  - Sampling uncertainty (from employee sampling)
  - Realization uncertainty (from stochastic label generation)
- **Produces company-level estimates**: Generates headcount estimates with credible intervals for each company

**Command to run:**
```sh
uv run main.py --subgraphs probit_bootstrap
```

#### 2. **Log Debiasing** (`log_debias`)

The **log debiasing** subgraph applies a log-debiasing approach to generate company-level headcount estimates with 80% confidence intervals. This method:

- **Corrects for systematic bias**: Applies log-transformation debiasing to account for systematic errors in annotator estimates
- **Generates confidence intervals**: Produces 80% confidence intervals for ML headcount estimates
- **Provides alternative estimates**: Offers a complementary approach to the probit bootstrap method

**Command to run:**
```sh
uv run main.py --subgraphs log_debias
```

**Run both primary subgraphs together:**
```sh
uv run main.py --subgraphs probit_bootstrap log_debias
```

### Modal vs. Local Processing

The pipeline seamlessly switches between cloud (Modal) and local processing:

- **Modal (default)**: GPU-accelerated, 4-5x faster, handles large datasets efficiently
- **Local (`--local-only`)**: Runs entirely on your machine, suitable for small datasets and development

Both modes produce identical results. Modal is recommended for production runs.

### Available Subgraphs

The pipeline includes several subgraphs for different stages of processing:

**Primary Subgraphs** (for generating headcount estimates):
- `probit_bootstrap`: Bootstrap probit inference with conditional correlation analysis (see [Primary Subgraphs](#primary-subgraphs) above)
- `log_debias`: Generate company-level headcount estimates with log-debiasing approach (see [Primary Subgraphs](#primary-subgraphs) above)

**Supporting Subgraphs**:
- `cv_analysis`: Extract ML keywords from CVs and validate scoring
- `clustering`: Cluster and analyze employee profiles
- `affiliation`: Filter and categorize academic affiliations

### Command Line Options

```sh
# View all options
uv run main.py --help

# List all available subgraphs
uv run main.py --list-subgraphs

# Cache management
uv run main.py --cache-info           # Show cache statistics
uv run main.py --list-cached-functions  # List cached functions
uv run main.py --no-cache=function_name  # Skip cache for specific function(s) (preferred for development)
uv run main.py --no-cache=function1,function2  # Skip cache for multiple functions (comma-separated)
uv run main.py --no-cache=all         # Disable all caching (same as --disable-cache)
uv run main.py --disable-cache       # Disable caching entirely
uv run main.py --clear-cache          # Clear all caches before running (use sparingly)
```

### Caching

The pipeline uses **Hamilton's automatic caching** to speed up repeated runs:

**How it works:**
- **Function-level caching**: Each function decorated with `@cache()` stores its output
- **Automatic invalidation**: Cache is invalidated when function code or inputs change
- **Location**: Cache stored in `outputs/.hamilton_cache/` directory
- **Incremental updates**: Only modified functions are recomputed

**Use cases:**
- **Development**: Test changes to specific functions without re-running everything; use `--no-cache=<function_name>` to skip cache for specific functions
- **Debugging**: Force recomputation of a specific function with `--no-cache=<function_name>` (prefer this over clearing the entire cache)
- **Production**: Keep caching enabled for faster re-runs
- **Reset**: Use `--disable-cache` or `--no-cache=all` to start fresh; use `--clear-cache` only when necessary

## Hamilton Framework

This pipeline is built using **Hamilton**, a Python micro-framework for creating dataflows. Hamilton represents data processing as function graphs where:

- **Functions** encode business logic and data transformations
- **Dependencies** are inferred from function parameters
- **Execution** automatically resolves the execution order
- **Caching** is handled automatically for expensive operations
- **Subgraphs** allow running specific parts of the pipeline

### Running the Full Pipeline vs. Subgraphs

The pipeline is modular—you can run all steps or just specific parts:

**Full pipeline** (processes all data end-to-end):
```sh
uv run main.py
```

**Specific subgraphs** (run only selected processing steps):
```sh
# Run only CV analysis and visualizations
uv run main.py --subgraphs cv_analysis probit_bootstrap

# Run only keyword extraction and clustering
uv run main.py --subgraphs cv_analysis clustering
```

This is useful for:
- Testing individual components
- Debugging specific steps
- Re-running just visualization after data changes
- Caching intermediate results for faster iteration

### Visualization

Generate a DAG visualization of the pipeline structure:

```sh
# Generate default visualization (pipeline_dag.png)
uv run main.py --visualize

# Specify custom output path
uv run main.py --visualize my_analysis_dag.png

# Also save the Graphviz DOT source file
uv run main.py --visualize pipeline_dag.png --save-dot
```

The visualization shows:
- Function nodes (processing steps)
- Data edges (dependencies)
- Color-coding by data type
- Automatic layout optimization

### Hamilton UI Telemetry

Monitor pipeline execution in real-time using Hamilton's UI:

```sh
# Enable telemetry
uv run main.py --enable-telemetry --project-id "my_project"

# With custom configuration
uv run main.py --enable-telemetry --project-id "my_project" \
  --username "user@example.com" --dag-name "production_run"
```

Or configure in `config/default.yaml`:
```yaml
execution:
  enable_telemetry: true
  project_id: "my_project"
  username: "user@example.com"
  dag_name: "production_run"
  telemetry_tags:
    environment: "production"
    team: "ML_HEADCOUNT"
```

The UI provides:
- Real-time execution monitoring
- Performance metrics per function
- Dependency graph visualization
- Execution history and trends

See [TELEMETRY_USAGE.md](docs/code_description/TELEMETRY_USAGE.md) for detailed setup instructions.

## Data Flow

1. **Input**: Raw CV data and LinkedIn profiles from companies
2. **Keyword Extraction**: Extract discriminative ML keywords from manually-rated CVs using KeyBERT
3. **Annotation**: Apply 6 programmatic annotators (3 LLMs + 3 keyword filters) to employee profiles
4. **Validation**: Use ground-truth labels from manually-rated CVs to estimate annotator accuracy
5. **Inference**: Bootstrap probit model with conditional correlation analysis to estimate true ML expertise from correlated annotations
6. **Aggregation**: Company-level headcount estimates with uncertainty intervals
7. **Visualization**: Plots of talent landscape, uncertainty distributions, and annotator performance

See [DATA_FLOW.mmd](docs/code_description/DATA_FLOW.mmd) for a diagram of the pipeline.

## Configuration

Configuration is managed via YAML files in the `config/` directory. The main configuration file is `config/default.yaml`.

### Configuration Structure

```yaml
# KeyBERT keyword extraction
keybert_extraction:
  model_name: "Salesforce/SFR-Embedding-Mistral"
  batch_size: 1024
  top_n: 30

# Hamilton execution settings
execution:
  use_remote: true              # Use Modal Labs for expensive operations
  disable_cache: false           # Enable Hamilton caching
  enable_telemetry: false        # Enable Hamilton UI telemetry
  project_id: "1"                # Required if enable_telemetry=true
```

### Overriding Configuration

Override settings from the command line:

```sh
# Use local processing instead of Modal Labs
uv run main.py --local-only

# Disable caching (force recomputation)
uv run main.py --disable-cache

# Enable telemetry from command line
uv run main.py --enable-telemetry --project-id "my_project"

# Custom config file
uv run main.py --config config/test.yaml
```

### Key Configuration Sections

- **`keybert_extraction`**: Keyword extraction parameters (model, batch size, etc.)
- **`keybert_clustering`**: Clustering algorithm settings
- **`dawid_skene_filtering`**: Dataset filtering parameters (max items, companies, min profiles per company)
- **`correlated_probit_bootstrap`**: Bootstrap sampling parameters
- **`execution`**: Runtime behavior (remote/local, caching, telemetry)
- **`data_paths`**: Input/output directory paths
- **`linkedin_datasets`**: Which datasets to include

See `config/default.yaml` for all available options.

## Documentation

- `docs/model_description/`: Mathematical framework and uncertainty analysis
- `docs/code_description/MODAL_INTEGRATION.md`: Details on Modal Labs cloud processing
- `docs/code_description/DATA_SOURCES_DOCUMENTATION.md`: Input data specifications
- `docs/code_description/TELEMETRY_USAGE.md`: Hamilton UI setup and usage
- `docs/code_description/MODAL_QUICK_REFERENCE.md`: Quick Modal commands reference