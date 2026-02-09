# Hamilton Telemetry Integration

This document explains how to use the Hamilton telemetry integration in the ML Headcount pipeline.

## Overview

The Hamilton telemetry integration allows you to track pipeline execution, performance metrics, and usage patterns through the Hamilton UI. This is useful for:

- Monitoring pipeline performance
- Debugging execution issues
- Tracking usage patterns
- Sharing pipeline insights with your team

## Setup

1. **Get Hamilton UI Access**: Sign up for Hamilton UI and get your project ID
2. **Install Dependencies**: The required `hamilton-sdk` and `python-dotenv` are already included in `pyproject.toml`
3. **Configure Telemetry**: Set the required parameters via environment variables, .env file, or command line

## Configuration Methods

### Method 1: Environment Variables (Recommended)

Set environment variables in your shell or .env file:

```bash
# Set environment variables
export HAMILTON_PROJECT_ID=your-project-id
export HAMILTON_USERNAME=your-email@example.com  # Optional, defaults to OS username
export HAMILTON_DAG_NAME=ml_headcount_analysis   # Optional, defaults to ml_headcount_pipeline

# Run with telemetry (no additional arguments needed)
python main.py --enable-telemetry
```

### Method 2: .env File

Create a `.env` file in your project root:

```bash
# .env file
HAMILTON_PROJECT_ID=your-project-id
HAMILTON_USERNAME=your-email@example.com
HAMILTON_DAG_NAME=ml_headcount_analysis
```

Then run:
```bash
python main.py --enable-telemetry
```

### Method 3: Command Line Arguments

```bash
python main.py --enable-telemetry --project-id your-project-id --username your-email@example.com
```

### Method 4: Programmatic Usage

```python
from ml_headcount.hamilton_pipeline import run_hamilton_pipeline_with_data_files

# Run with telemetry enabled
results = run_hamilton_pipeline_with_data_files(
    data_dir="data",
    output_dir="outputs",
    use_remote=True,
    disable_cache=False,
    # Telemetry configuration
    enable_telemetry=True,
    project_id="your-project-id",  # Get from Hamilton UI
    username="your-email@example.com",
    dag_name="ml_headcount_analysis",
    telemetry_tags={
        "environment": "PROD",
        "team": "ML_HEADCOUNT",
        "version": "1.0.0"
    }
)
```

### Using the Pipeline Class Directly

```python
from ml_headcount.hamilton_pipeline import HamiltonMLHeadcountPipeline

# Initialize pipeline with telemetry
pipeline = HamiltonMLHeadcountPipeline(
    data_dir="data",
    output_dir="outputs",
    use_remote=True,
    disable_cache=False,
    # Required parameters
    ds_draws=1000,
    ds_tune=1500,
    ds_chains=8,
    # Telemetry configuration
    enable_telemetry=True,
    project_id="your-project-id",
    username="your-email@example.com",
    dag_name="ml_headcount_analysis",
    telemetry_tags={
        "environment": "PROD",
        "team": "ML_HEADCOUNT",
        "version": "1.0.0"
    }
)

# Run the pipeline
results = pipeline.run_with_available_files()
```

## Configuration Parameters

### Required Parameters (when `enable_telemetry=True`)

- **`project_id`**: Your Hamilton project ID (get from Hamilton UI)

### Optional Parameters

- **`enable_telemetry`**: Enable/disable telemetry (default: `False`)
- **`username`**: Your email/username for Hamilton (default: OS username)
- **`dag_name`**: Name for this DAG in Hamilton UI (default: `"ml_headcount_pipeline"`)
- **`telemetry_tags`**: Additional tags for categorization (default: `{"environment": "DEV", "team": "ML_HEADCOUNT"}`)

## Environment Variable Names

The system uses the following environment variable:
- `HAMILTON_PROJECT_ID`: Hamilton project ID (required for telemetry)

## Configuration Precedence

When multiple configuration methods are used, the precedence is:
1. **Command line arguments** (highest priority)
2. **Environment variables** (medium priority)
3. **Default values** (lowest priority)

This means command line arguments will always override environment variables and defaults.

## Example Tags

You can use tags to categorize and filter your pipeline runs:

```python
telemetry_tags = {
    "environment": "PROD",           # DEV, STAGING, PROD
    "team": "ML_HEADCOUNT",         # Team name
    "version": "1.0.0",             # Pipeline version
    "experiment": "keyword_analysis", # Experiment name
    "dataset": "linkedin_profiles",  # Dataset being processed
    "model": "dawid_skene"          # Model being used
}
```

## Running Without Telemetry

If you don't want to use telemetry, simply omit the telemetry parameters or set `enable_telemetry=False`:

```python
# This will run without telemetry
results = run_hamilton_pipeline_with_data_files(
    data_dir="data",
    output_dir="outputs",
    use_remote=True,
    disable_cache=False
    # No telemetry parameters needed
)
```

## Viewing Telemetry Data

1. Log into your Hamilton UI account
2. Navigate to your project
3. View pipeline executions, performance metrics, and execution graphs
4. Use tags to filter and search for specific runs

## Troubleshooting

### Common Issues

1. **"project_id is required when enable_telemetry=True"**
   - Make sure you've set a valid project ID from Hamilton UI

2. **"username is required when enable_telemetry=True"**
   - Make sure you've set your username/email

3. **Connection issues**
   - Check your internet connection
   - Verify your Hamilton UI credentials

### Getting Help

- Check the Hamilton documentation: https://hamilton.dagworks.io/
- Review the example script: `examples/telemetry_example.py`
- Check the pipeline logs for detailed error messages

## Example Script

See `examples/telemetry_example.py` for a complete working example that demonstrates both telemetry-enabled and telemetry-disabled pipeline runs.
